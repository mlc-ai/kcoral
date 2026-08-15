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
constexpr int BC = 64;
constexpr int D_PAD = 130;       // Q bf16 stride (conflict-free for wmma)
constexpr int KV_STR = 130;      // K/V bf16 stride (conflict-free for wmma)
constexpr int S_STRIDE = 65;     // S float stride (conflict-free for wmma)
constexpr int P_STRIDE = 130;    // P bf16 stride (same buffer as S, 65*4=130*2)
constexpr int O_STRIDE = 129;    // O float stride (transposed [D][O_STRIDE], conflict-free for wmma)
constexpr float SCALE = 0.08838834764831840f;

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
    __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(p); p += BR * D_PAD * 2;
    __nv_bfloat16* Ks = reinterpret_cast<__nv_bfloat16*>(p); p += BC * KV_STR * 2;
    __nv_bfloat16* Vs = reinterpret_cast<__nv_bfloat16*>(p); p += BC * KV_STR * 2;
    float* Ss = reinterpret_cast<float*>(p); p += BR * S_STRIDE * 4;
    float* Os = reinterpret_cast<float*>(p); p += D * O_STRIDE * 4;
    float* ms = reinterpret_cast<float*>(p); p += BR * 4;
    float* ls = reinterpret_cast<float*>(p);
    __nv_bfloat16* Ps = reinterpret_cast<__nv_bfloat16*>(Ss);

    const int64_t off = ((int64_t)batch * H + head) * S * D;

    // Load Q with padding (4-byte vectorized)
    for (int i = tid; i < BR * 64; i += 128) {
        int row = i / 64, col = (i % 64) * 2, gr = q_start + row;
        if (gr < S)
            *reinterpret_cast<uint32_t*>(&Qs[row * D_PAD + col]) =
                *reinterpret_cast<const uint32_t*>(&Qg[off + (int64_t)gr * D + col]);
        else
            *reinterpret_cast<uint32_t*>(&Qs[row * D_PAD + col]) = 0u;
    }
    for (int i = tid; i < BR; i += 128) {
        Qs[i * D_PAD + D] = __float2bfloat16(0.f);
        Qs[i * D_PAD + D + 1] = __float2bfloat16(0.f);
    }

    // Init O (transposed [D][O_STRIDE]), m, l
    for (int i = tid; i < D * O_STRIDE; i += 128) Os[i] = 0.f;
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

        // Load K and V together
        for (int i = tid; i < BC * 64; i += 128) {
            int row = i / 64, col = (i % 64) * 2, gr = ks + row;
            if (gr < kv_end) {
                *reinterpret_cast<uint32_t*>(&Ks[row * KV_STR + col]) =
                    *reinterpret_cast<const uint32_t*>(&Kg[off + (int64_t)gr * D + col]);
                *reinterpret_cast<uint32_t*>(&Vs[row * KV_STR + col]) =
                    *reinterpret_cast<const uint32_t*>(&Vg[off + (int64_t)gr * D + col]);
            } else {
                *reinterpret_cast<uint32_t*>(&Ks[row * KV_STR + col]) = 0u;
                *reinterpret_cast<uint32_t*>(&Vs[row * KV_STR + col]) = 0u;
            }
        }
        for (int i = tid; i < BC; i += 128) {
            Ks[i * KV_STR + D] = __float2bfloat16(0.f);
            Ks[i * KV_STR + D + 1] = __float2bfloat16(0.f);
            Vs[i * KV_STR + D] = __float2bfloat16(0.f);
            Vs[i * KV_STR + D + 1] = __float2bfloat16(0.f);
        }
        __syncthreads();

        // S = Q @ K^T (128x64: 8 m-tiles x 4 n-tiles x 8 k-iter)
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            int mt = wid * 2 + mi;
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                wmma::fill_fragment(cf, 0.f);
                #pragma unroll
                for (int ki = 0; ki < 8; ki++) {
                    wmma::load_matrix_sync(af, &Qs[mt * 16 * D_PAD + ki * 16], D_PAD);
                    wmma::load_matrix_sync(bfkt, &Ks[ni * 16 * KV_STR + ki * 16], KV_STR);
                    wmma::mma_sync(cf, af, bfkt, cf);
                }
                wmma::store_matrix_sync(&Ss[mt * 16 * S_STRIDE + ni * 16], cf, S_STRIDE, wmma::mem_row_major);
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
                    float v = Ss[tid * S_STRIDE + j] * SCALE;
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
                // Rescale O (transposed layout, 4-way bank conflict)
                #pragma unroll
                for (int d = 0; d < D; d++)
                    Os[d * O_STRIDE + tid] *= rs;
                float ol = ls[tid];
                ms[tid] = nm;
                ls[tid] = ol * rs + rsum;
                // Store P (bf16, reuses S buffer)
                #pragma unroll
                for (int j = 0; j < BC; j++)
                    Ps[tid * P_STRIDE + j] = __float2bfloat16(sv[j]);
            } else {
                #pragma unroll
                for (int j = 0; j < BC; j++)
                    Ps[tid * P_STRIDE + j] = __float2bfloat16(0.f);
            }
        }
        __syncthreads();

        // O += P @ V (128x128: 8 m-tiles x 8 n-tiles x 4 k-iter)
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            int mt = wid * 2 + mi;
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                wmma::load_matrix_sync(of, &Os[mt * 16 + ni * 16 * O_STRIDE], O_STRIDE, wmma::mem_col_major);
                #pragma unroll
                for (int ki = 0; ki < 4; ki++) {
                    wmma::load_matrix_sync(af, &Ps[mt * 16 * P_STRIDE + ki * 16], P_STRIDE);
                    wmma::load_matrix_sync(bfv, &Vs[ki * 16 * KV_STR + ni * 16], KV_STR);
                    wmma::mma_sync(of, af, bfv, of);
                }
                wmma::store_matrix_sync(&Os[mt * 16 + ni * 16 * O_STRIDE], of, O_STRIDE, wmma::mem_col_major);
            }
        }
        __syncthreads();
    }

    // Compute inv_l and write LSE
    if (tid < BR) {
        int gr = q_start + tid;
        if (gr < S) {
            float l = ls[tid], m = ms[tid];
            ls[tid] = (l > 0.f) ? (1.f / l) : 0.f;
            LSE[((int64_t)batch * H + head) * S + gr] = (l > 0.f) ? (m + __logf(l)) : -INFINITY;
        }
    }
    __syncthreads();

    // Write O cooperatively: 4 rows at a time, 32 threads per row, 4 bf16 per thread (uint2)
    for (int rg = 0; rg < BR; rg += 4) {
        int row = rg + tid / 32;
        int col = (tid % 32) * 4;
        int gr = q_start + row;
        if (gr < S) {
            float il = ls[row];
            float v0 = Os[(col + 0) * O_STRIDE + row] * il;
            float v1 = Os[(col + 1) * O_STRIDE + row] * il;
            float v2 = Os[(col + 2) * O_STRIDE + row] * il;
            float v3 = Os[(col + 3) * O_STRIDE + row] * il;
            __nv_bfloat16 b0 = __float2bfloat16(v0);
            __nv_bfloat16 b1 = __float2bfloat16(v1);
            __nv_bfloat16 b2 = __float2bfloat16(v2);
            __nv_bfloat16 b3 = __float2bfloat16(v3);
            uint32_t p0 = *reinterpret_cast<uint16_t*>(&b0) | ((uint32_t)*reinterpret_cast<uint16_t*>(&b1) << 16);
            uint32_t p1 = *reinterpret_cast<uint16_t*>(&b2) | ((uint32_t)*reinterpret_cast<uint16_t*>(&b3) << 16);
            *reinterpret_cast<uint2*>(&Og[off + (int64_t)gr * D + col]) = make_uint2(p0, p1);
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

    const int smem = BR * D_PAD * 2 + BC * KV_STR * 2 + BC * KV_STR * 2
                   + BR * S_STRIDE * 4 + D * O_STRIDE * 4 + BR * 4 + BR * 4;

    CUDA_CHECK(cudaFuncSetAttribute(flash_attn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    dim3 grid(B * H, (S + BR - 1) / BR);
    flash_attn_kernel<<<grid, 128, smem, stream>>>(Qp, Kp, Vp, Op, Lp, H, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flash_attn_impl::run);

}  // namespace flash_attn_impl