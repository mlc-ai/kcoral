#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
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

namespace mha_bwd_d128_causal {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int D  = 128;
constexpr int THREADS = 128;
constexpr float SCALE = 0.08838834764831845f;

// Shared memory for dkdv kernel (~104KB → 2 CTAs/SM)
struct SmemDKDV {
    alignas(16) __nv_bfloat16 Kj[BN * D];         // 16KB
    alignas(16) __nv_bfloat16 Vj[BN * D];         // 16KB
    alignas(16) __nv_bfloat16 Qi[BM * D];          // 16KB
    alignas(16) __nv_bfloat16 dOi[BM * D];         // 16KB
    alignas(16) float scratch_f[BM * BN];          // 16KB (S, then dP)
    alignas(16) float P_f[BM * BN];                // 16KB
    alignas(16) __nv_bfloat16 PdS_b[BM * BN];      // 8KB
    alignas(16) float L_i[BM];                     // 0.25KB
    alignas(16) float D_i[BM];                     // 0.25KB
};

// Shared memory for dq kernel (~88KB → 2 CTAs/SM)
struct SmemDQ {
    alignas(16) __nv_bfloat16 Qi[BM * D];          // 16KB
    alignas(16) __nv_bfloat16 dOi[BM * D];         // 16KB
    alignas(16) __nv_bfloat16 Kj[BN * D];          // 16KB
    alignas(16) __nv_bfloat16 Vj[BN * D];          // 16KB
    alignas(16) float scratch_f[BM * BN];          // 16KB
    alignas(16) float P_f[BM * BN];                // 16KB (can overlap with Kj/Vj after use... but kept separate for simplicity)
    alignas(16) __nv_bfloat16 PdS_b[BM * BN];      // 8KB
    alignas(16) float L_i[BM];                     // 0.25KB
    alignas(16) float D_i[BM];                     // 0.25KB
};

__device__ __forceinline__ void load_64x128_bf16(__nv_bfloat16* dst, const __nv_bfloat16* src, int tid, int row_offset, int Sg) {
    for (int rb = 0; rb < BM; rb += 8) {
        int rib = tid / 16; int cu = tid % 16;
        int m = rb + rib;
        bool valid = (row_offset + m) < Sg;
        uint4 v = valid ? reinterpret_cast<const uint4*>(src)[m * 16 + cu] : make_uint4(0, 0, 0, 0);
        reinterpret_cast<uint4*>(dst)[m * 16 + cu] = v;
    }
}

__device__ __forceinline__ void load_bn_x128_bf16(__nv_bfloat16* dst, const __nv_bfloat16* src, int tid, int row_offset, int Sg) {
    for (int rb = 0; rb < BN; rb += 8) {
        int rib = tid / 16; int cu = tid % 16;
        int n = rb + rib;
        bool valid = (row_offset + n) < Sg;
        uint4 v = valid ? reinterpret_cast<const uint4*>(src)[n * 16 + cu] : make_uint4(0, 0, 0, 0);
        reinterpret_cast<uint4*>(dst)[n * 16 + cu] = v;
    }
}

__device__ __forceinline__ void compute_D_i(float* D_i, const __nv_bfloat16* dOi, const __nv_bfloat16* O_global,
                                             int warp, int lane, int i_block, int Sg) {
    int row_local = lane / 2;
    int half = lane % 2;
    int m = warp * 16 + row_local;
    bool valid = (i_block * BM + m) < Sg;
    const __nv_bfloat16* gOrow = O_global + (int64_t)(i_block * BM + m) * D;
    const __nv_bfloat16* dOirow = dOi + m * D;
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
    if (half == 0 && valid) D_i[m] = partial;
}

__device__ __forceinline__ void compute_S(float* scratch, const __nv_bfloat16* Qi, const __nv_bfloat16* Kj,
                                           int warp) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    int mi = warp;
    #pragma unroll
    for (int ni = 0; ni < 4; ni++) {
        wmma::fill_fragment(c, 0.0f);
        #pragma unroll
        for (int ki = 0; ki < D / 16; ki++) {
            wmma::load_matrix_sync(a, &Qi[mi * 16 * D + ki * 16], D);
            wmma::load_matrix_sync(b, &Kj[ni * 16 * D + ki * 16], D);
            wmma::mma_sync(c, a, b, c);
        }
        wmma::store_matrix_sync(&scratch[mi * 16 * BN + ni * 16], c, BN, wmma::mem_row_major);
    }
}

__device__ __forceinline__ void compute_dP(float* scratch, const __nv_bfloat16* dOi, const __nv_bfloat16* Vj,
                                            int warp) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    int mi = warp;
    #pragma unroll
    for (int ni = 0; ni < 4; ni++) {
        wmma::fill_fragment(c, 0.0f);
        #pragma unroll
        for (int ki = 0; ki < D / 16; ki++) {
            wmma::load_matrix_sync(a, &dOi[mi * 16 * D + ki * 16], D);
            wmma::load_matrix_sync(b, &Vj[ni * 16 * D + ki * 16], D);
            wmma::mma_sync(c, a, b, c);
        }
        wmma::store_matrix_sync(&scratch[mi * 16 * BN + ni * 16], c, BN, wmma::mem_row_major);
    }
}

__device__ __forceinline__ void compute_P_and_dS(
    float* P_f, __nv_bfloat16* PdS_b, const float* scratch, const float* L_i, const float* D_i,
    int tid, int i_block, int j_block, int Sg, bool compute_dS) {
    
    int BM_BN = BM * BN;
    for (int idx = tid; idx < BM_BN; idx += THREADS) {
        int m = idx / BN, n = idx % BN;
        int qi = i_block * BM + m;
        int kj = j_block * BN + n;
        float s = scratch[m * BN + n];
        if (qi >= Sg || kj > qi || kj >= Sg) {
            if (!compute_dS) {
                P_f[m * BN + n] = 0.0f;
                PdS_b[m * BN + n] = __float2bfloat16(0.0f);
            } else {
                PdS_b[m * BN + n] = __float2bfloat16(0.0f);
            }
        } else {
            if (!compute_dS) {
                float p = __expf(s * SCALE - L_i[m]);
                P_f[m * BN + n] = p;
                PdS_b[m * BN + n] = __float2bfloat16(p);
            } else {
                float p = P_f[m * BN + n];
                float dp = scratch[m * BN + n];
                float ds = p * (dp - D_i[m]) * SCALE;
                PdS_b[m * BN + n] = __float2bfloat16(ds);
            }
        }
    }
}

// ============================================================
// DKDV Kernel: dK and dV with register accumulators
// ============================================================
__global__ void __launch_bounds__(THREADS, 2)
mha_bwd_dkdv_kernel(const __nv_bfloat16* __restrict__ Q,
                    const __nv_bfloat16* __restrict__ K,
                    const __nv_bfloat16* __restrict__ V,
                    const __nv_bfloat16* __restrict__ O,
                    const __nv_bfloat16* __restrict__ dO,
                    const float*        __restrict__ L,
                    __nv_bfloat16*      dK_g,
                    __nv_bfloat16*      dV_g,
                    int B, int H, int Sg)
{
    extern __shared__ char smem_buf[];
    SmemDKDV& smem = *reinterpret_cast<SmemDKDV*>(smem_buf);

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

    // Register accumulators for dV and dK (8 fragments each = 64 registers)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dV_acc[8];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dK_acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        wmma::fill_fragment(dV_acc[i], 0.0f);
        wmma::fill_fragment(dK_acc[i], 0.0f);
    }

    // Load Kj, Vj once
    load_bn_x128_bf16(smem.Kj, Kg + (int64_t)(j_block * BN) * D, tid, j_block * BN, Sg);
    load_bn_x128_bf16(smem.Vj, Vg + (int64_t)(j_block * BN) * D, tid, j_block * BN, Sg);
    __syncthreads();

    int nq = (Sg + BM - 1) / BM;

    for (int i_block = 0; i_block < nq; i_block++) {
        if (i_block * BM + BM - 1 < j_block * BN) continue;

        // Load Qi, dOi
        load_64x128_bf16(smem.Qi, Qg + (int64_t)(i_block * BM) * D, tid, i_block * BM, Sg);
        load_64x128_bf16(smem.dOi, dOg + (int64_t)(i_block * BM) * D, tid, i_block * BM, Sg);
        for (int m = tid; m < BM; m += THREADS) {
            smem.L_i[m] = (i_block * BM + m < Sg) ? Lg[i_block * BM + m] : 0.0f;
        }
        __syncthreads();

        // D_i = rowsum(dO * O)
        compute_D_i(smem.D_i, smem.dOi, Og, warp, lane, i_block, Sg);
        __syncthreads();

        // S = Q @ K^T
        compute_S(smem.scratch_f, smem.Qi, smem.Kj, warp);
        __syncthreads();

        // P = exp(S*scale - L)
        compute_P_and_dS(smem.P_f, smem.PdS_b, smem.scratch_f, smem.L_i, smem.D_i,
                         tid, i_block, j_block, Sg, false);
        __syncthreads();

        // dV += P^T @ dO (register accumulators)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                #pragma unroll
                for (int ki = 0; ki < BM / 16; ki++) {
                    wmma::load_matrix_sync(a, &smem.PdS_b[ki * 16 * BN + warp * 16], BN);
                    wmma::load_matrix_sync(b, &smem.dOi[ki * 16 * D + ni * 16], D);
                    wmma::mma_sync(dV_acc[ni], a, b, dV_acc[ni]);
                }
            }
        }
        __syncthreads();

        // dP = dO @ V^T
        compute_dP(smem.scratch_f, smem.dOi, smem.Vj, warp);
        __syncthreads();

        // dS = P * (dP - D) * scale
        compute_P_and_dS(smem.P_f, smem.PdS_b, smem.scratch_f, smem.L_i, smem.D_i,
                         tid, i_block, j_block, Sg, true);
        __syncthreads();

        // dK += dS^T @ Q (register accumulators)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                #pragma unroll
                for (int ki = 0; ki < BM / 16; ki++) {
                    wmma::load_matrix_sync(a, &smem.PdS_b[ki * 16 * BN + warp * 16], BN);
                    wmma::load_matrix_sync(b, &smem.Qi[ki * 16 * D + ni * 16], D);
                    wmma::mma_sync(dK_acc[ni], a, b, dK_acc[ni]);
                }
            }
        }
        __syncthreads();
    }

    // Store dV, dK from registers through shared memory to global
    // Reuse Kj+Vj space (32KB) as float storage
    float* store_buf = reinterpret_cast<float*>(smem.Kj);

    // Store dV
    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        wmma::store_matrix_sync(&store_buf[warp * 16 * D + ni * 16], dV_acc[ni], D, wmma::mem_row_major);
    }
    __syncthreads();
    for (int rb = 0; rb < BN; rb += 8) {
        int rib = tid / 16; int cu = tid % 16;
        int n = rb + rib;
        if (j_block * BN + n < Sg) {
            __nv_bfloat16 bv[8];
            #pragma unroll
            for (int i = 0; i < 8; i++)
                bv[i] = __float2bfloat16_rn(store_buf[n * D + cu * 8 + i]);
            uint4* ov = reinterpret_cast<uint4*>(dVb + (int64_t)(j_block * BN) * D);
            ov[n * 16 + cu] = *reinterpret_cast<uint4*>(bv);
        }
    }
    __syncthreads();

    // Store dK
    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        wmma::store_matrix_sync(&store_buf[warp * 16 * D + ni * 16], dK_acc[ni], D, wmma::mem_row_major);
    }
    __syncthreads();
    for (int rb = 0; rb < BN; rb += 8) {
        int rib = tid / 16; int cu = tid % 16;
        int n = rb + rib;
        if (j_block * BN + n < Sg) {
            __nv_bfloat16 bk[8];
            #pragma unroll
            for (int i = 0; i < 8; i++)
                bk[i] = __float2bfloat16_rn(store_buf[n * D + cu * 8 + i]);
            uint4* ok = reinterpret_cast<uint4*>(dKb + (int64_t)(j_block * BN) * D);
            ok[n * 16 + cu] = *reinterpret_cast<uint4*>(bk);
        }
    }
}

// ============================================================
// DQ Kernel: dQ with register accumulators, no atomics
// ============================================================
__global__ void __launch_bounds__(THREADS, 2)
mha_bwd_dq_kernel(const __nv_bfloat16* __restrict__ Q,
                  const __nv_bfloat16* __restrict__ K,
                  const __nv_bfloat16* __restrict__ V,
                  const __nv_bfloat16* __restrict__ O,
                  const __nv_bfloat16* __restrict__ dO,
                  const float*        __restrict__ L,
                  __nv_bfloat16*      dQ_g,
                  int B, int H, int Sg)
{
    extern __shared__ char smem_buf[];
    SmemDQ& smem = *reinterpret_cast<SmemDQ*>(smem_buf);

    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int i_block = blockIdx.y;
    int tid = threadIdx.x;
    int warp = tid / 32;
    int lane = tid % 32;

    if (i_block * BM >= Sg) return;

    const int64_t base_off = (int64_t)(b * H + h) * Sg;
    const __nv_bfloat16* Kg = K + base_off * D;
    const __nv_bfloat16* Vg = V + base_off * D;
    const __nv_bfloat16* Qg = Q + base_off * D;
    const __nv_bfloat16* Og = O + base_off * D;
    const __nv_bfloat16* dOg = dO + base_off * D;
    const float* Lg = L + base_off;
    __nv_bfloat16* dQb = dQ_g + base_off * D;

    // Load Qi, dOi once
    load_64x128_bf16(smem.Qi, Qg + (int64_t)(i_block * BM) * D, tid, i_block * BM, Sg);
    load_64x128_bf16(smem.dOi, dOg + (int64_t)(i_block * BM) * D, tid, i_block * BM, Sg);
    for (int m = tid; m < BM; m += THREADS) {
        smem.L_i[m] = (i_block * BM + m < Sg) ? Lg[i_block * BM + m] : 0.0f;
    }
    __syncthreads();

    // Compute D_i once
    compute_D_i(smem.D_i, smem.dOi, Og, warp, lane, i_block, Sg);
    __syncthreads();

    // Register accumulators for dQ (8 fragments = 64 registers)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dQ_acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        wmma::fill_fragment(dQ_acc[i], 0.0f);
    }

    int num_kv = (Sg + BN - 1) / BN;

    for (int j_block = 0; j_block < num_kv; j_block++) {
        if (j_block * BN > i_block * BM + BM - 1) break;

        // Load Kj, Vj
        load_bn_x128_bf16(smem.Kj, Kg + (int64_t)(j_block * BN) * D, tid, j_block * BN, Sg);
        load_bn_x128_bf16(smem.Vj, Vg + (int64_t)(j_block * BN) * D, tid, j_block * BN, Sg);
        __syncthreads();

        // S = Q @ K^T
        compute_S(smem.scratch_f, smem.Qi, smem.Kj, warp);
        __syncthreads();

        // P = exp(S*scale - L)
        compute_P_and_dS(smem.P_f, smem.PdS_b, smem.scratch_f, smem.L_i, smem.D_i,
                         tid, i_block, j_block, Sg, false);
        __syncthreads();

        // dP = dO @ V^T
        compute_dP(smem.scratch_f, smem.dOi, smem.Vj, warp);
        __syncthreads();

        // dS = P * (dP - D) * scale
        compute_P_and_dS(smem.P_f, smem.PdS_b, smem.scratch_f, smem.L_i, smem.D_i,
                         tid, i_block, j_block, Sg, true);
        __syncthreads();

        // dQ += dS @ K (register accumulators)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                #pragma unroll
                for (int ki = 0; ki < BN / 16; ki++) {
                    wmma::load_matrix_sync(a, &smem.PdS_b[warp * 16 * BN + ki * 16], BN);
                    wmma::load_matrix_sync(b, &smem.Kj[ki * 16 * D + ni * 16], D);
                    wmma::mma_sync(dQ_acc[ni], a, b, dQ_acc[ni]);
                }
            }
        }
        __syncthreads();
    }

    // Store dQ from registers through shared memory to global
    // Reuse Kj+Vj space (32KB) as float storage
    float* store_buf = reinterpret_cast<float*>(smem.Kj);

    #pragma unroll
    for (int ni = 0; ni < 8; ni++) {
        wmma::store_matrix_sync(&store_buf[warp * 16 * D + ni * 16], dQ_acc[ni], D, wmma::mem_row_major);
    }
    __syncthreads();
    for (int rb = 0; rb < BM; rb += 8) {
        int rib = tid / 16; int cu = tid % 16;
        int m = rb + rib;
        if (i_block * BM + m < Sg) {
            __nv_bfloat16 bq[8];
            #pragma unroll
            for (int i = 0; i < 8; i++)
                bq[i] = __float2bfloat16_rn(store_buf[m * D + cu * 8 + i]);
            uint4* oq = reinterpret_cast<uint4*>(dQb + (int64_t)(i_block * BM) * D);
            oq[m * 16 + cu] = *reinterpret_cast<uint4*>(bq);
        }
    }
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

    int num_kv = (int)((S + BN - 1) / BN);
    dim3 grid1((int)(B * H), num_kv);
    dim3 block(THREADS);
    int smem1 = (int)sizeof(SmemDKDV);
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_dkdv_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem1));

    mha_bwd_dkdv_kernel<<<grid1, block, smem1, stream>>>(
        Qp, Kp, Vp, Op, dOp, Lp, dKp, dVp,
        (int)B, (int)H, (int)S);
    CUDA_CHECK(cudaGetLastError());

    int num_q = (int)((S + BM - 1) / BM);
    dim3 grid2((int)(B * H), num_q);
    int smem2 = (int)sizeof(SmemDQ);
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_dq_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem2));

    mha_bwd_dq_kernel<<<grid2, block, smem2, stream>>>(
        Qp, Kp, Vp, Op, dOp, Lp, dQp,
        (int)B, (int)H, (int)S);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128_causal::run);

}  // namespace mha_bwd_d128_causal