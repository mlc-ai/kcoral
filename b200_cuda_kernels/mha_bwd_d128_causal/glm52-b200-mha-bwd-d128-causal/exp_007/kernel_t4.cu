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

struct SmemDKDV {
    alignas(16) __nv_bfloat16 Kj[BN * D];
    alignas(16) __nv_bfloat16 Vj[BN * D];
    alignas(16) __nv_bfloat16 Qi[BM * D];
    alignas(16) __nv_bfloat16 dOi[BM * D];
    alignas(16) float dV_acc[BN * D];
    alignas(16) float dK_acc[BN * D];
    alignas(16) float scratch_f[BM * BN];
    alignas(16) float P_f[BM * BN];
    alignas(16) __nv_bfloat16 PdS_b[BM * BN];
    alignas(16) float L_i[BM];
    alignas(16) float D_i[BM];
};

struct SmemDQ {
    alignas(16) __nv_bfloat16 Qi[BM * D];
    alignas(16) __nv_bfloat16 dOi[BM * D];
    alignas(16) __nv_bfloat16 Kj[BN * D];
    alignas(16) __nv_bfloat16 Vj[BN * D];
    alignas(16) float dQ_acc[BM * D];
    alignas(16) float scratch_f[BM * BN];
    alignas(16) float P_f[BM * BN];
    alignas(16) __nv_bfloat16 PdS_b[BM * BN];
    alignas(16) float L_i[BM];
    alignas(16) float D_i[BM];
};

__global__ void __launch_bounds__(THREADS)
mha_bwd_dkdv_kernel(const __nv_bfloat16* __restrict__ Q,
                    const __nv_bfloat16* __restrict__ K,
                    const __nv_bfloat16* __restrict__ V,
                    const __nv_bfloat16* __restrict__ O,
                    const __nv_bfloat16* __restrict__ dO,
                    const float*        __restrict__ L,
                    __nv_bfloat16*      dK_g,
                    __nv_bfloat16*      dV_g,
                    int B, int H, int Sg, int Dd)
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
        const uint4* gdO = reinterpret_cast<const uint4*>(dOg + (int64_t)(i_block * BM) * D);
        for (int rb = 0; rb < BM; rb += 8) {
            int rib = tid / 16; int cu = tid % 16;
            int m = rb + rib;
            bool valid = (i_block * BM + m) < Sg;
            uint4 vq = valid ? gQi[m * 16 + cu] : make_uint4(0, 0, 0, 0);
            uint4 vd = valid ? gdO[m * 16 + cu] : make_uint4(0, 0, 0, 0);
            reinterpret_cast<uint4*>(smem.Qi)[m * 16 + cu] = vq;
            reinterpret_cast<uint4*>(smem.dOi)[m * 16 + cu] = vd;
        }
        for (int m = tid; m < BM; m += THREADS) {
            smem.L_i[m] = (i_block * BM + m < Sg) ? Lg[i_block * BM + m] : 0.0f;
            smem.D_i[m] = 0.0f;
        }
        __syncthreads();

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

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
            int mi = warp;
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                wmma::fill_fragment(c, 0.0f);
                #pragma unroll
                for (int ki = 0; ki < D / 16; ki++) {
                    wmma::load_matrix_sync(a, &smem.Qi[mi * 16 * D + ki * 16], D);
                    wmma::load_matrix_sync(b, &smem.Kj[ni * 16 * D + ki * 16], D);
                    wmma::mma_sync(c, a, b, c);
                }
                wmma::store_matrix_sync(&smem.scratch_f[mi * 16 * BN + ni * 16], c, BN, wmma::mem_row_major);
            }
        }
        __syncthreads();

        for (int idx = tid; idx < BM * BN; idx += THREADS) {
            int m = idx / BN, n = idx % BN;
            int qi = i_block * BM + m;
            int kj = j_block * BN + n;
            float s = smem.scratch_f[m * BN + n];
            float p;
            if (qi >= Sg || kj > qi || kj >= Sg) p = 0.0f;
            else p = __expf(s * SCALE - smem.L_i[m]);
            smem.P_f[m * BN + n] = p;
            smem.PdS_b[m * BN + n] = __float2bfloat16(p);
        }
        __syncthreads();

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
            int mi = warp;
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                wmma::load_matrix_sync(c, &smem.dV_acc[mi * 16 * D + ni * 16], D, wmma::mem_row_major);
                #pragma unroll
                for (int ki = 0; ki < BM / 16; ki++) {
                    wmma::load_matrix_sync(a, &smem.PdS_b[ki * 16 * BN + mi * 16], BN);
                    wmma::load_matrix_sync(b, &smem.dOi[ki * 16 * D + ni * 16], D);
                    wmma::mma_sync(c, a, b, c);
                }
                wmma::store_matrix_sync(&smem.dV_acc[mi * 16 * D + ni * 16], c, D, wmma::mem_row_major);
            }
        }
        __syncthreads();

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
            int mi = warp;
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                wmma::fill_fragment(c, 0.0f);
                #pragma unroll
                for (int ki = 0; ki < D / 16; ki++) {
                    wmma::load_matrix_sync(a, &smem.dOi[mi * 16 * D + ki * 16], D);
                    wmma::load_matrix_sync(b, &smem.Vj[ni * 16 * D + ki * 16], D);
                    wmma::mma_sync(c, a, b, c);
                }
                wmma::store_matrix_sync(&smem.scratch_f[mi * 16 * BN + ni * 16], c, BN, wmma::mem_row_major);
            }
        }
        __syncthreads();

        for (int idx = tid; idx < BM * BN; idx += THREADS) {
            int m = idx / BN, n = idx % BN;
            int qi = i_block * BM + m;
            int kj = j_block * BN + n;
            float p = smem.P_f[m * BN + n];
            float dp = smem.scratch_f[m * BN + n];
            float ds;
            if (qi >= Sg || kj > qi || kj >= Sg) ds = 0.0f;
            else ds = p * (dp - smem.D_i[m]) * SCALE;
            smem.PdS_b[m * BN + n] = __float2bfloat16(ds);
        }
        __syncthreads();

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
            int mi = warp;
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                wmma::load_matrix_sync(c, &smem.dK_acc[mi * 16 * D + ni * 16], D, wmma::mem_row_major);
                #pragma unroll
                for (int ki = 0; ki < BM / 16; ki++) {
                    wmma::load_matrix_sync(a, &smem.PdS_b[ki * 16 * BN + mi * 16], BN);
                    wmma::load_matrix_sync(b, &smem.Qi[ki * 16 * D + ni * 16], D);
                    wmma::mma_sync(c, a, b, c);
                }
                wmma::store_matrix_sync(&smem.dK_acc[mi * 16 * D + ni * 16], c, D, wmma::mem_row_major);
            }
        }
        __syncthreads();
    }

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

__global__ void __launch_bounds__(THREADS)
mha_bwd_dq_kernel(const __nv_bfloat16* __restrict__ Q,
                  const __nv_bfloat16* __restrict__ K,
                  const __nv_bfloat16* __restrict__ V,
                  const __nv_bfloat16* __restrict__ O,
                  const __nv_bfloat16* __restrict__ dO,
                  const float*        __restrict__ L,
                  __nv_bfloat16*      dQ_g,
                  int B, int H, int Sg, int Dd)
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

    const uint4* gQi = reinterpret_cast<const uint4*>(Qg + (int64_t)(i_block * BM) * D);
    const uint4* gdO = reinterpret_cast<const uint4*>(dOg + (int64_t)(i_block * BM) * D);
    for (int rb = 0; rb < BM; rb += 8) {
        int rib = tid / 16; int cu = tid % 16;
        int m = rb + rib;
        bool valid = (i_block * BM + m) < Sg;
        uint4 vq = valid ? gQi[m * 16 + cu] : make_uint4(0, 0, 0, 0);
        uint4 vd = valid ? gdO[m * 16 + cu] : make_uint4(0, 0, 0, 0);
        reinterpret_cast<uint4*>(smem.Qi)[m * 16 + cu] = vq;
        reinterpret_cast<uint4*>(smem.dOi)[m * 16 + cu] = vd;
    }
    for (int m = tid; m < BM; m += THREADS) {
        smem.L_i[m] = (i_block * BM + m < Sg) ? Lg[i_block * BM + m] : 0.0f;
        smem.D_i[m] = 0.0f;
    }
    __syncthreads();

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

    for (int idx = tid; idx < BM * D; idx += THREADS) {
        smem.dQ_acc[idx] = 0.0f;
    }
    __syncthreads();

    int num_kv = (Sg + BN - 1) / BN;

    for (int j_block = 0; j_block < num_kv; j_block++) {
        if (j_block * BN > i_block * BM + BM - 1) break;

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

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
            int mi = warp;
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                wmma::fill_fragment(c, 0.0f);
                #pragma unroll
                for (int ki = 0; ki < D / 16; ki++) {
                    wmma::load_matrix_sync(a, &smem.Qi[mi * 16 * D + ki * 16], D);
                    wmma::load_matrix_sync(b, &smem.Kj[ni * 16 * D + ki * 16], D);
                    wmma::mma_sync(c, a, b, c);
                }
                wmma::store_matrix_sync(&smem.scratch_f[mi * 16 * BN + ni * 16], c, BN, wmma::mem_row_major);
            }
        }
        __syncthreads();

        for (int idx = tid; idx < BM * BN; idx += THREADS) {
            int m = idx / BN, n = idx % BN;
            int qi = i_block * BM + m;
            int kj = j_block * BN + n;
            float s = smem.scratch_f[m * BN + n];
            float p;
            if (qi >= Sg || kj > qi || kj >= Sg) p = 0.0f;
            else p = __expf(s * SCALE - smem.L_i[m]);
            smem.P_f[m * BN + n] = p;
            smem.PdS_b[m * BN + n] = __float2bfloat16(p);
        }
        __syncthreads();

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
            int mi = warp;
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                wmma::fill_fragment(c, 0.0f);
                #pragma unroll
                for (int ki = 0; ki < D / 16; ki++) {
                    wmma::load_matrix_sync(a, &smem.dOi[mi * 16 * D + ki * 16], D);
                    wmma::load_matrix_sync(b, &smem.Vj[ni * 16 * D + ki * 16], D);
                    wmma::mma_sync(c, a, b, c);
                }
                wmma::store_matrix_sync(&smem.scratch_f[mi * 16 * BN + ni * 16], c, BN, wmma::mem_row_major);
            }
        }
        __syncthreads();

        for (int idx = tid; idx < BM * BN; idx += THREADS) {
            int m = idx / BN, n = idx % BN;
            int qi = i_block * BM + m;
            int kj = j_block * BN + n;
            float p = smem.P_f[m * BN + n];
            float dp = smem.scratch_f[m * BN + n];
            float ds;
            if (qi >= Sg || kj > qi || kj >= Sg) ds = 0.0f;
            else ds = p * (dp - smem.D_i[m]) * SCALE;
            smem.PdS_b[m * BN + n] = __float2bfloat16(ds);
        }
        __syncthreads();

        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
            int mi = warp;
            #pragma unroll
            for (int ni = 0; ni < 8; ni++) {
                wmma::load_matrix_sync(c, &smem.dQ_acc[mi * 16 * D + ni * 16], D, wmma::mem_row_major);
                #pragma unroll
                for (int ki = 0; ki < BN / 16; ki++) {
                    wmma::load_matrix_sync(a, &smem.PdS_b[mi * 16 * BN + ki * 16], BN);
                    wmma::load_matrix_sync(b, &smem.Kj[ki * 16 * D + ni * 16], D);
                    wmma::mma_sync(c, a, b, c);
                }
                wmma::store_matrix_sync(&smem.dQ_acc[mi * 16 * D + ni * 16], c, D, wmma::mem_row_major);
            }
        }
        __syncthreads();
    }

    for (int rb = 0; rb < BM; rb += 8) {
        int rib = tid / 16; int cu = tid % 16;
        int m = rb + rib;
        bool valid = (i_block * BM + m) < Sg;
        if (valid) {
            __nv_bfloat16 bq[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                bq[i] = __float2bfloat16_rn(smem.dQ_acc[m * D + cu * 8 + i]);
            }
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
        (int)B, (int)H, (int)S, (int)Dd);
    CUDA_CHECK(cudaGetLastError());

    int num_q = (int)((S + BM - 1) / BM);
    dim3 grid2((int)(B * H), num_q);
    int smem2 = (int)sizeof(SmemDQ);
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_dq_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem2));

    mha_bwd_dq_kernel<<<grid2, block, smem2, stream>>>(
        Qp, Kp, Vp, Op, dOp, Lp, dQp,
        (int)B, (int)H, (int)S, (int)Dd);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_d128_causal::run);

}  // namespace mha_bwd_d128_causal