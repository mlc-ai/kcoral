#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd_d128_causal {

constexpr int BM = 64;
constexpr int BN = 32;
constexpr int D  = 128;
constexpr int WARPS = 4;
constexpr int THREADS = 128;
constexpr float SCALE = 0.08838834764831845f;

struct Smem {
    alignas(16) __nv_bfloat16 Kj[BN * D];
    alignas(16) __nv_bfloat16 Vj[BN * D];
    alignas(16) __nv_bfloat16 Qi[BM * D];
    alignas(16) __nv_bfloat16 dOi[BM * D];
    alignas(16) float scratch_f[BM * BN];
    alignas(16) float P_f[BM * BN];
    alignas(16) float dS_f[BM * BN];
    alignas(16) __nv_bfloat16 P_b[BM * BN];
    alignas(16) __nv_bfloat16 dS_b[BM * BN];
    alignas(16) float dV_acc[BN * D];
    alignas(16) float dK_acc[BN * D];
    alignas(16) float D_i[BM];
    alignas(16) float L_i[BM];
    alignas(16) float dQ_tmp[WARPS * 16 * 16];
};

__global__ void __launch_bounds__(THREADS)
mha_bwd_kernel(const __nv_bfloat16* __restrict__ Q,
               const __nv_bfloat16* __restrict__ K,
               const __nv_bfloat16* __restrict__ V,
               const __nv_bfloat16* __restrict__ O,
               const __nv_bfloat16* __restrict__ dO,
               const float*        __restrict__ L,
               float*              dQ_fp32,
               __nv_bfloat16*      dK_g,
               __nv_bfloat16*      dV_g,
               int B, int H, int Sg, int Dd)
{
    extern __shared__ char smem_buf[];
    Smem& smem = *reinterpret_cast<Smem*>(smem_buf);

    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int j_block = blockIdx.y;
    int tid = threadIdx.x;
    int warp = tid / 32;
    int lane = tid % 32;

    if (j_block * BN >= Sg) return;

    const int64_t base_off = (int64_t)(b * H + h) * Sg;
    const __nv_bfloat16* Kg = K + base_off * D;
    const __nv_bfloat16* Vg = V + base_off * D;
    const __nv_bfloat16* Qg = Q + base_off * D;
    const __nv_bfloat16* Og = O + base_off * D;
    const __nv_bfloat16* dOg = dO + base_off * D;
    const float* Lg = L + base_off;
    __nv_bfloat16* dKb = dK_g + base_off * D;
    __nv_bfloat16* dVb = dV_g + base_off * D;
    float* dQb = dQ_fp32 + base_off * D;

    for (int idx = tid; idx < BN * D; idx += THREADS) {
        smem.dV_acc[idx] = 0.0f;
        smem.dK_acc[idx] = 0.0f;
    }

    for (int rb = 0; rb < BN; rb += 8) {
        int rib = tid / 16;
        int cu  = tid % 16;
        int n = rb + rib;
        bool valid = (j_block * BN + n) < Sg;
        const uint4* gK = reinterpret_cast<const uint4*>(Kg + (int64_t)(j_block * BN) * D);
        const uint4* gV = reinterpret_cast<const uint4*>(Vg + (int64_t)(j_block * BN) * D);
        uint4 vk = valid ? gK[n * 16 + cu] : make_uint4(0, 0, 0, 0);
        uint4 vv = valid ? gV[n * 16 + cu] : make_uint4(0, 0, 0, 0);
        reinterpret_cast<uint4*>(smem.Kj)[n * 16 + cu] = vk;
        reinterpret_cast<uint4*>(smem.Vj)[n * 16 + cu] = vv;
    }
    __syncthreads();

    int nq = (Sg + BM - 1) / BM;

    for (int i_block = 0; i_block < nq; i_block++) {
        if (i_block * BM + BM - 1 < j_block * BN) continue;

        const uint4* gQi = reinterpret_cast<const uint4*>(Qg + (int64_t)(i_block * BM) * D);
        for (int rb = 0; rb < BM; rb += 8) {
            int rib = tid / 16; int cu = tid % 16;
            int m = rb + rib;
            bool valid = (i_block * BM + m) < Sg;
            uint4 v = valid ? gQi[m * 16 + cu] : make_uint4(0, 0, 0, 0);
            reinterpret_cast<uint4*>(smem.Qi)[m * 16 + cu] = v;
        }
        const uint4* gdO = reinterpret_cast<const uint4*>(dOg + (int64_t)(i_block * BM) * D);
        for (int rb = 0; rb < BM; rb += 8) {
            int rib = tid / 16; int cu = tid % 16;
            int m = rb + rib;
            bool valid = (i_block * BM + m) < Sg;
            uint4 v = valid ? gdO[m * 16 + cu] : make_uint4(0, 0, 0, 0);
            reinterpret_cast<uint4*>(smem.dOi)[m * 16 + cu] = v;
        }
        for (int m = tid; m < BM; m += THREADS) {
            smem.L_i[m] = (i_block * BM + m < Sg) ? Lg[i_block * BM + m] : 0.0f;
            smem.D_i[m] = 0.0f;
        }
        __syncthreads();

        // D_i = rowsum(dO * O) — FP32 accumulation
        {
            int row_local = lane / 2;
            int half = lane % 2;
            int m = warp * 16 + row_local;
            bool valid = (i_block * BM + m) < Sg;
            const __nv_bfloat16* gOrow = Og + (int64_t)(i_block * BM + m) * D;
            const __nv_bfloat16* dOirow = smem.dOi + m * D;
            float partial = 0.0f;
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                uint4 ov = valid ? reinterpret_cast<const uint4*>(gOrow)[half * 8 + e]
                                 : make_uint4(0, 0, 0, 0);
                uint4 dv = reinterpret_cast<const uint4*>(dOirow)[half * 8 + e];
                __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);
                __nv_bfloat16* db = reinterpret_cast<__nv_bfloat16*>(&dv);
                #pragma unroll
                for (int i = 0; i < 8; i++)
                    partial += __bfloat162float(ob[i]) * __bfloat162float(db[i]);
            }
            partial += __shfl_xor_sync(0xffffffff, partial, 1);
            if (half == 0 && valid) smem.D_i[m] = partial;
        }
        __syncthreads();

        // S = Q @ K^T
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
            #pragma unroll
            for (int t = 0; t < 2; t++) {
                int tile = warp * 2 + t;
                int mi = tile / 2, ni = tile % 2;
                wmma::fill_fragment(c, 0.0f);
                #pragma unroll
                for (int ki = 0; ki < D / 16; ki++) {
                    wmma::load_matrix_sync(a, &smem.Qi[(mi * 16) * D + ki * 16], D);
                    wmma::load_matrix_sync(b, &smem.Kj[(ni * 16) * D + ki * 16], D);
                    wmma::mma_sync(c, a, b, c);
                }
                wmma::store_matrix_sync(&smem.scratch_f[(mi * 16) * BN + ni * 16], c, BN, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // P = exp(S*scale - L) in FP32, also convert to BF16 for dV matmul
        for (int idx = tid; idx < BM * BN; idx += THREADS) {
            int m = idx / BN, n = idx % BN;
            int qi = i_block * BM + m;
            int nj = j_block * BN + n;
            float s = smem.scratch_f[m * BN + n];
            float p;
            if (qi >= Sg || nj > qi || nj >= Sg) p = 0.0f;
            else p = expf(s * SCALE - smem.L_i[m]);
            smem.P_f[m * BN + n] = p;
            smem.P_b[m * BN + n] = __float2bfloat16(p);
        }
        __syncthreads();

        // dP = dO @ V^T
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
            #pragma unroll
            for (int t = 0; t < 2; t++) {
                int tile = warp * 2 + t;
                int mi = tile / 2, ni = tile % 2;
                wmma::fill_fragment(c, 0.0f);
                #pragma unroll
                for (int ki = 0; ki < D / 16; ki++) {
                    wmma::load_matrix_sync(a, &smem.dOi[(mi * 16) * D + ki * 16], D);
                    wmma::load_matrix_sync(b, &smem.Vj[(ni * 16) * D + ki * 16], D);
                    wmma::mma_sync(c, a, b, c);
                }
                wmma::store_matrix_sync(&smem.scratch_f[(mi * 16) * BN + ni * 16], c, BN, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // dS = P_f * (dP - D) * scale — FP32, then convert to BF16 for matmuls
        for (int idx = tid; idx < BM * BN; idx += THREADS) {
            int m = idx / BN, n = idx % BN;
            int qi = i_block * BM + m;
            int nj = j_block * BN + n;
            float p = smem.P_f[m * BN + n];
            float dp = smem.scratch_f[m * BN + n];
            float ds;
            if (qi >= Sg || nj > qi || nj >= Sg) ds = 0.0f;
            else ds = p * (dp - smem.D_i[m]) * SCALE;
            smem.dS_f[m * BN + n] = ds;
            smem.dS_b[m * BN + n] = __float2bfloat16(ds);
        }
        __syncthreads();

        // dV_acc += P^T @ dO
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
            #pragma unroll
            for (int t = 0; t < 4; t++) {
                int tile = warp * 4 + t;
                int mi = tile / 8, ni = tile % 8;
                wmma::load_matrix_sync(c, &smem.dV_acc[(mi * 16) * D + ni * 16], D, wmma::mem_row_major);
                #pragma unroll
                for (int ki = 0; ki < BM / 16; ki++) {
                    wmma::load_matrix_sync(a, &smem.P_b[(ki * 16) * BN + mi * 16], BN);
                    wmma::load_matrix_sync(b, &smem.dOi[(ki * 16) * D + ni * 16], D);
                    wmma::mma_sync(c, a, b, c);
                }
                wmma::store_matrix_sync(&smem.dV_acc[(mi * 16) * D + ni * 16], c, D, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // dK_acc += dS^T @ Q
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
            #pragma unroll
            for (int t = 0; t < 4; t++) {
                int tile = warp * 4 + t;
                int mi = tile / 8, ni = tile % 8;
                wmma::load_matrix_sync(c, &smem.dK_acc[(mi * 16) * D + ni * 16], D, wmma::mem_row_major);
                #pragma unroll
                for (int ki = 0; ki < BM / 16; ki++) {
                    wmma::load_matrix_sync(a, &smem.dS_b[(ki * 16) * BN + mi * 16], BN);
                    wmma::load_matrix_sync(b, &smem.Qi[(ki * 16) * D + ni * 16], D);
                    wmma::mma_sync(c, a, b, c);
                }
                wmma::store_matrix_sync(&smem.dK_acc[(mi * 16) * D + ni * 16], c, D, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // dQ += dS @ K — atomic add to global FP32
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
            #pragma unroll
            for (int t = 0; t < 8; t++) {
                int tile = warp * 8 + t;
                int mi = tile / 8, ni = tile % 8;
                wmma::fill_fragment(c, 0.0f);
                #pragma unroll
                for (int ki = 0; ki < BN / 16; ki++) {
                    wmma::load_matrix_sync(a, &smem.dS_b[(mi * 16) * BN + ki * 16], BN);
                    wmma::load_matrix_sync(b, &smem.Kj[(ki * 16) * D + ni * 16], D);
                    wmma::mma_sync(c, a, b, c);
                }
                wmma::store_matrix_sync(&smem.dQ_tmp[warp * 256], c, 16, wmma::mem_row_major);
                __syncwarp();
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    int gidx = lane * 8 + e;
                    int r = gidx / 16, cc = gidx % 16;
                    int qi = i_block * BM + mi * 16 + r;
                    int col = ni * 16 + cc;
                    if (qi < Sg) {
                        float val = smem.dQ_tmp[warp * 256 + r * 16 + cc];
                        atomicAdd(&dQb[(int64_t)qi * D + col], val);
                    }
                }
            }
        }
        __syncthreads();
    }

    // Store dK, dV
    for (int rb = 0; rb < BN; rb += 8) {
        int rib = tid / 16; int cu = tid % 16;
        int n = rb + rib;
        bool valid = (j_block * BN + n) < Sg;
        if (valid) {
            __nv_bfloat16 bk[8], bv[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                bk[i] = __float2bfloat16_rn(smem.dK_acc[n * D + cu * 8 + i]);
                bv[i] = __float2bfloat16_rn(smem.dV_acc[n * D + cu * 8 + i]);
            }
            uint4* ok = reinterpret_cast<uint4*>(dKb + (int64_t)(j_block * BN) * D);
            uint4* ov = reinterpret_cast<uint4*>(dVb + (int64_t)(j_block * BN) * D);
            ok[n * 16 + cu] = *reinterpret_cast<uint4*>(bk);
            ov[n * 16 + cu] = *reinterpret_cast<uint4*>(bv);
        }
    }
}

__global__ void convert_f32_bf16(const float* __restrict__ src,
                                 __nv_bfloat16* __restrict__ dst, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2bfloat16_rn(src[idx]);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV)
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0), H = Q.size(1), S = Q.size(2), Dd = Q.size(3);
    const __nv_bfloat16* Qp = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* Kp = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* Vp = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* Op = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dOp = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* Lp = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQp = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dKp = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dVp = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    size_t ws_count = (size_t)B * H * S * Dd;
    float* dQ_fp32;
    CUDA_CHECK(cudaMalloc(&dQ_fp32, ws_count * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(dQ_fp32, 0, ws_count * sizeof(float), stream));

    int num_kv = (int)((S + BN - 1) / BN);
    dim3 grid((int)(B * H), num_kv);
    dim3 block(THREADS);
    int smem_bytes = (int)sizeof(Smem);
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    mha_bwd_kernel<<<grid, block, smem_bytes, stream>>>(
        Qp, Kp, Vp, Op, dOp, Lp, dQ_fp32, dKp, dVp,
        (int)B, (int)H, (int)S, (int)Dd);
    CUDA_CHECK(cudaGetLastError());

    int n = (int)ws_count;
    int tpb = 256;
    int bpg = (n + tpb - 1) / tpb;
    convert_f32_bf16<<<bpg, tpb, 0, stream>>>(dQ_fp32, dQp, n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(dQ_fp32));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128_causal::run);

}  // namespace mha_bwd_d128_causal