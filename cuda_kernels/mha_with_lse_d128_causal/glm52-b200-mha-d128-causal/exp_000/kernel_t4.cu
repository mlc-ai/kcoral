#include <cuda_bf16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace flash_attn_impl {

constexpr int WMM = 16, WMN = 16, WMK = 16;
constexpr int D = 128;
constexpr int BR = 128;
constexpr int BC = 128;
constexpr int D_PAD = 136;       // Q bf16 stride (16-byte aligned, bank-conflict aware)
constexpr int KV_STR = 136;      // K/V bf16 stride
constexpr int S_STR = 128;       // S float stride (16-byte aligned)
constexpr int P_STR = 136;       // P bf16 stride (reuses S buffer)
constexpr int O_STR = 128;       // O float stride (16-byte aligned)
constexpr float SCALE = 0.08838834764831840f;  // 1/sqrt(128)

__global__ __launch_bounds__(128, 1)
void flash_attn_kernel(
    const __nv_bfloat16* __restrict__ Qg,
    const __nv_bfloat16* __restrict__ Kg,
    const __nv_bfloat16* __restrict__ Vg,
    __nv_bfloat16* __restrict__ Og,
    float* __restrict__ LSE,
    int H, int S) {

    const int bh = blockIdx.x, batch = bh / H, head = bh % H;
    const int q_start = blockIdx.y * BR;
    const int tid = threadIdx.x, wid = tid / 32;

    extern __shared__ char sm[];
    char* p = sm;
    __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(p); p += BR*D_PAD*2;
    __nv_bfloat16* KVs = reinterpret_cast<__nv_bfloat16*>(p); p += BC*KV_STR*2;
    float* Ss = reinterpret_cast<float*>(p); p += BR*S_STR*4;
    float* Os = reinterpret_cast<float*>(p); p += BR*O_STR*4;
    float* ms = reinterpret_cast<float*>(p); p += BR*4;
    float* ls = reinterpret_cast<float*>(p);
    __nv_bfloat16* Ps = reinterpret_cast<__nv_bfloat16*>(Ss);

    const int64_t off = ((int64_t)batch * H + head) * S * D;

    // Load Q with padding (4-byte vectorized)
    for (int i = tid; i < BR*64; i += 128) {
        int row = i/64, col = (i%64)*2, gr = q_start+row;
        if (gr < S)
            *reinterpret_cast<uint32_t*>(&Qs[row*D_PAD+col]) =
                *reinterpret_cast<const uint32_t*>(&Qg[off+(int64_t)gr*D+col]);
        else
            *reinterpret_cast<uint32_t*>(&Qs[row*D_PAD+col]) = 0u;
    }
    for (int i = tid; i < BR*(D_PAD-D); i += 128)
        Qs[(i/(D_PAD-D))*D_PAD + D + (i%(D_PAD-D))] = __float2bfloat16(0.f);

    // Init O, m, l
    for (int i = tid; i < BR*O_STR; i += 128) Os[i] = 0.f;
    if (tid < BR) { ms[tid] = -INFINITY; ls[tid] = 0.f; }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, WMM, WMN, WMK, __nv_bfloat16, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, WMM, WMN, WMK, __nv_bfloat16, wmma::col_major> bfkt;
    wmma::fragment<wmma::matrix_b, WMM, WMN, WMK, __nv_bfloat16, wmma::row_major> bfv;
    wmma::fragment<wmma::accumulator, WMM, WMN, WMK, float> cf, of;

    const int kv_end = min(q_start + BR, S);
    const int nkv = (kv_end + BC - 1) / BC;

    for (int kb = 0; kb < nkv; kb++) {
        const int ks = kb * BC;

        // Load K (4-byte vectorized)
        for (int i = tid; i < BC*64; i += 128) {
            int row = i/64, col = (i%64)*2, gr = ks+row;
            if (gr < kv_end)
                *reinterpret_cast<uint32_t*>(&KVs[row*KV_STR+col]) =
                    *reinterpret_cast<const uint32_t*>(&Kg[off+(int64_t)gr*D+col]);
            else
                *reinterpret_cast<uint32_t*>(&KVs[row*KV_STR+col]) = 0u;
        }
        for (int i = tid; i < BC*(KV_STR-D); i += 128)
            KVs[(i/(KV_STR-D))*KV_STR + D + (i%(KV_STR-D))] = __float2bfloat16(0.f);
        __syncthreads();

        // S = Q @ K^T (8 m-tiles x 8 n-tiles x 8 k-iter, each warp: 2 m-tiles)
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            int mt = wid*2 + mi;
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                wmma::fill_fragment(cf, 0.f);
                #pragma unroll
                for (int ki = 0; ki < 8; ki++) {
                    wmma::load_matrix_sync(af, &Qs[mt*16*D_PAD + ki*16], D_PAD);
                    wmma::load_matrix_sync(bfkt, &KVs[ni*16*KV_STR + ki*16], KV_STR);
                    wmma::mma_sync(cf, af, bfkt, cf);
                }
                wmma::store_matrix_sync(&Ss[mt*16*S_STR + ni*16], cf, S_STR, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // Softmax: each thread = one row
        if (tid < BR) {
            int qi = q_start + tid;
            if (qi < S) {
                float sv[BC];
                float rmax = -INFINITY;
                #pragma unroll
                for (int j = 0; j < BC; j++) {
                    float v = Ss[tid*S_STR + j] * SCALE;
                    int ki = ks + j;
                    if (ki > qi || ki >= kv_end) v = -INFINITY;
                    sv[j] = v;
                    rmax = fmaxf(rmax, v);
                }
                float om = ms[tid], nm = fmaxf(om, rmax);
                float rs = (om > -INFINITY) ? __expf(om - nm) : 0.f;
                float rsum = 0.f;
                #pragma unroll
                for (int j = 0; j < BC; j++) {
                    float pv = (sv[j] > -INFINITY) ? __expf(sv[j] - nm) : 0.f;
                    sv[j] = pv;
                    rsum += pv;
                }
                #pragma unroll
                for (int d = 0; d < D; d++) Os[tid*O_STR + d] *= rs;
                float ol = ls[tid];
                ms[tid] = nm;
                ls[tid] = ol * rs + rsum;
                #pragma unroll
                for (int j = 0; j < BC; j++)
                    Ps[tid*P_STR + j] = __float2bfloat16(sv[j]);
            } else {
                #pragma unroll
                for (int j = 0; j < BC; j++) Ps[tid*P_STR + j] = __float2bfloat16(0.f);
            }
        }
        __syncthreads();

        // Load V (overwrites K buffer)
        for (int i = tid; i < BC*64; i += 128) {
            int row = i/64, col = (i%64)*2, gr = ks+row;
            if (gr < kv_end)
                *reinterpret_cast<uint32_t*>(&KVs[row*KV_STR+col]) =
                    *reinterpret_cast<const uint32_t*>(&Vg[off+(int64_t)gr*D+col]);
            else
                *reinterpret_cast<uint32_t*>(&KVs[row*KV_STR+col]) = 0u;
        }
        for (int i = tid; i < BC*(KV_STR-D); i += 128)
            KVs[(i/(KV_STR-D))*KV_STR + D + (i%(KV_STR-D))] = __float2bfloat16(0.f);
        __syncthreads();

        // O += P @ V (8 m-tiles x 8 n-tiles x 8 k-iter)
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            int mt = wid*2 + mi;
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                wmma::load_matrix_sync(of, &Os[mt*16*O_STR + ni*16], O_STR, wmma::mem_row_major);
                #pragma unroll
                for (int ki = 0; ki < 8; ki++) {
                    wmma::load_matrix_sync(af, &Ps[mt*16*P_STR + ki*16], P_STR);
                    wmma::load_matrix_sync(bfv, &KVs[ki*16*KV_STR + ni*16], KV_STR);
                    wmma::mma_sync(of, af, bfv, of);
                }
                wmma::store_matrix_sync(&Os[mt*16*O_STR + ni*16], of, O_STR, wmma::mem_row_major);
            }
        }
        __syncthreads();
    }

    // Write LSE and O: each thread handles its own row entirely
    if (tid < BR) {
        int gr = q_start + tid;
        if (gr < S) {
            float l = ls[tid], m = ms[tid];
            float inv_l = (l > 0.f) ? (1.f / l) : 0.f;
            LSE[((int64_t)batch*H+head)*S + gr] = (l > 0.f) ? (m + __logf(l)) : -INFINITY;
            for (int d = 0; d < D; d++)
                Og[off + (int64_t)gr*D + d] = __float2bfloat16(Os[tid*O_STR + d] * inv_l);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    const int B = 4, H = 48, S = static_cast<int>(Q.size(2));
    const __nv_bfloat16 *Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16 *Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16 *Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16 *Op = static_cast<__nv_bfloat16*>(O.data_ptr());
    float *Lp = static_cast<float*>(LSE.data_ptr());
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    const int smem = BR*D_PAD*2 + BC*KV_STR*2 + BR*S_STR*4 + BR*O_STR*4 + BR*4 + BR*4;
    CUDA_CHECK(cudaFuncSetAttribute(flash_attn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    dim3 grid(B*H, (S+BR-1)/BR);
    flash_attn_kernel<<<grid, 128, smem, stream>>>(Qp, Kp, Vp, Op, Lp, H, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flash_attn_impl::run);

}  // namespace flash_attn_impl