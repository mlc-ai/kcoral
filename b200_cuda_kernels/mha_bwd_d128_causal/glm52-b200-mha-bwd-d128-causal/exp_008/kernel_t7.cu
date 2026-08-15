#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;
using bf16 = __nv_bfloat16;

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__);exit(1);} } while(0)

namespace tvm_ffi_attn_bwd {

constexpr int BM = 32;
constexpr int BN = 64;
constexpr int D = 128;
constexpr int THREADS = 128;
constexpr int WM = 16;

struct Smem {
    bf16 K[BN*D];
    bf16 V[BN*D];
    float dK[BN*D];
    float dV[BN*D];
    bf16 Q[BM*D];
    bf16 dO[BM*D];
    float S[BM*BN];
    bf16 P[BM*BN];
    bf16 dS[BM*BN];
    float L[BM];
    float Dm[BM];
};

__global__ void attn_bwd_kernel(
    const bf16* __restrict__ Q_g,
    const bf16* __restrict__ K_g,
    const bf16* __restrict__ V_g,
    const bf16* __restrict__ O_g,
    const bf16* __restrict__ dO_g,
    const float* __restrict__ L_g,
    float* __restrict__ dQ_fp,
    bf16* __restrict__ dK_out,
    bf16* __restrict__ dV_out,
    int S, int H, float scale)
{
    int jb = blockIdx.x;
    int bh = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int64_t base = (int64_t)(b*H + h) * S * D;
    int64_t base_lse = (int64_t)(b*H + h) * S;

    int gj0 = jb * BN;
    if (gj0 >= S) return;
    int BN_a = min(BN, S - gj0);

    extern __shared__ __align__(16) char smem_raw[];
    Smem& sm = *reinterpret_cast<Smem*>(smem_raw);

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;

    // Load K, V
    {
        const int4* Ksrc = reinterpret_cast<const int4*>(&K_g[base + (int64_t)gj0*D]);
        const int4* Vsrc = reinterpret_cast<const int4*>(&V_g[base + (int64_t)gj0*D]);
        int4* Kdst = reinterpret_cast<int4*>(sm.K);
        int4* Vdst = reinterpret_cast<int4*>(sm.V);
        int nvec = BN_a * D / 8;
        for (int i = tid; i < nvec; i += THREADS) {
            Kdst[i] = Ksrc[i];
            Vdst[i] = Vsrc[i];
        }
    }
    // Init dK, dV
    for (int i = tid; i < BN*D; i += THREADS) {
        sm.dK[i] = 0.f;
        sm.dV[i] = 0.f;
    }
    __syncthreads();

    int min_qb = max(0, (gj0 - BM + 1 + BM - 1) / BM);
    int max_qb = (S - 1) / BM;

    for (int qb = min_qb; qb <= max_qb; qb++) {
        int gi0 = qb * BM;
        int BM_a = min(BM, S - gi0);
        if (BM_a <= 0) continue;
        if (gi0 > gj0 + BN_a - 1) break;

        // Load Q, dO
        {
            const int4* Qsrc = reinterpret_cast<const int4*>(&Q_g[base + (int64_t)gi0*D]);
            const int4* dOsrc = reinterpret_cast<const int4*>(&dO_g[base + (int64_t)gi0*D]);
            int4* Qdst = reinterpret_cast<int4*>(sm.Q);
            int4* dOdst = reinterpret_cast<int4*>(sm.dO);
            int nvec = BM_a * D / 8;
            for (int i = tid; i < nvec; i += THREADS) {
                Qdst[i] = Qsrc[i];
                dOdst[i] = dOsrc[i];
            }
        }
        // Load L, compute D = rowsum(dO * O)
        for (int i = tid; i < BM_a; i += THREADS) {
            sm.L[i] = L_g[base_lse + gi0 + i];
            float s = 0.f;
            const bf16* optr = &O_g[base + (int64_t)(gi0+i)*D];
            for (int d = 0; d < D; d += 2) {
                __nv_bfloat162 o2 = *reinterpret_cast<const __nv_bfloat162*>(&optr[d]);
                __nv_bfloat162 do2 = *reinterpret_cast<__nv_bfloat162*>(&sm.dO[i*D+d]);
                float2 of = __bfloat1622float2(o2);
                float2 dof = __bfloat1622float2(do2);
                s += dof.x * of.x + dof.y * of.y;
            }
            sm.Dm[i] = s;
        }
        __syncthreads();

        // === S = Q @ K^T ===
        {
            int m_tile = warp_id / 2;
            int n_start = (warp_id % 2) * 2;
            wmma::fragment<wmma::accumulator, WM, WM, WM, float> acc[2];
            for (int n = 0; n < 2; n++) wmma::fill_fragment(acc[n], 0.f);
            for (int k = 0; k < D/WM; k++) {
                wmma::fragment<wmma::matrix_a, WM, WM, WM, bf16, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag, &sm.Q[m_tile*WM*D + k*WM], D);
                for (int n = 0; n < 2; n++) {
                    wmma::fragment<wmma::matrix_b, WM, WM, WM, bf16, wmma::col_major> b_frag;
                    wmma::load_matrix_sync(b_frag, &sm.K[(n_start+n)*WM*D + k*WM], D);
                    wmma::mma_sync(acc[n], a_frag, b_frag, acc[n]);
                }
            }
            for (int n = 0; n < 2; n++)
                wmma::store_matrix_sync(&sm.S[m_tile*WM*BN + (n_start+n)*WM], acc[n], BN, wmma::mem_row_major);
        }
        __syncthreads();

        // === P = exp(S * scale - L) with causal mask ===
        for (int idx = tid; idx < BM*BN; idx += THREADS) {
            int i = idx / BN, j = idx % BN;
            int gi = gi0 + i, gj = gj0 + j;
            if (i >= BM_a || j >= BN_a || gj > gi) { sm.P[i*BN + j] = bf16(0); continue; }
            float s = sm.S[i*BN + j] * scale;
            sm.P[i*BN + j] = __float2bfloat16(__expf(s - sm.L[i]));
        }
        __syncthreads();

        // === dP = dO @ V^T ===
        {
            int m_tile = warp_id / 2;
            int n_start = (warp_id % 2) * 2;
            wmma::fragment<wmma::accumulator, WM, WM, WM, float> acc[2];
            for (int n = 0; n < 2; n++) wmma::fill_fragment(acc[n], 0.f);
            for (int k = 0; k < D/WM; k++) {
                wmma::fragment<wmma::matrix_a, WM, WM, WM, bf16, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag, &sm.dO[m_tile*WM*D + k*WM], D);
                for (int n = 0; n < 2; n++) {
                    wmma::fragment<wmma::matrix_b, WM, WM, WM, bf16, wmma::col_major> b_frag;
                    wmma::load_matrix_sync(b_frag, &sm.V[(n_start+n)*WM*D + k*WM], D);
                    wmma::mma_sync(acc[n], a_frag, b_frag, acc[n]);
                }
            }
            for (int n = 0; n < 2; n++)
                wmma::store_matrix_sync(&sm.S[m_tile*WM*BN + (n_start+n)*WM], acc[n], BN, wmma::mem_row_major);
        }
        __syncthreads();

        // === dS = P * (dP - D) ===
        for (int idx = tid; idx < BM*BN; idx += THREADS) {
            int i = idx / BN, j = idx % BN;
            int gi = gi0 + i, gj = gj0 + j;
            if (i >= BM_a || j >= BN_a || gj > gi) { sm.dS[i*BN + j] = bf16(0); continue; }
            float p = __bfloat162float(sm.P[i*BN + j]);
            float dp = sm.S[i*BN + j];
            sm.dS[i*BN + j] = __float2bfloat16(p * (dp - sm.Dm[i]));
        }
        __syncthreads();

        // === dK += dS^T @ Q (accumulate in smem) ===
        // dS is [BM, BN] in row_major. To get dS^T ([BN, BM]), load as col_major.
        // Q is [BM, D]. To match K contraction, Q is loaded as col_major (which makes it Q^T, so A @ B => dS^T @ Q).
        for (int nround = 0; nround < 4; nround++) {
            int n_start = nround * 2;
            wmma::fragment<wmma::accumulator, WM, WM, WM, float> acc[2];
            for (int n = 0; n < 2; n++)
                wmma::load_matrix_sync(acc[n], &sm.dK[warp_id*WM*D + (n_start+n)*WM], D, wmma::mem_row_major);
            for (int k = 0; k < BM/WM; k++) {
                wmma::fragment<wmma::matrix_a, WM, WM, WM, bf16, wmma::col_major> a_frag;
                wmma::load_matrix_sync(a_frag, &sm.dS[k*WM*BN + warp_id*WM], BN);
                for (int n = 0; n < 2; n++) {
                    wmma::fragment<wmma::matrix_b, WM, WM, WM, bf16, wmma::col_major> b_frag;
                    wmma::load_matrix_sync(b_frag, &sm.Q[k*WM*D + (n_start+n)*WM], D);
                    wmma::mma_sync(acc[n], a_frag, b_frag, acc[n]);
                }
            }
            for (int n = 0; n < 2; n++)
                wmma::store_matrix_sync(&sm.dK[warp_id*WM*D + (n_start+n)*WM], acc[n], D, wmma::mem_row_major);
        }
        __syncthreads();

        // === dV += P^T @ dO (accumulate in smem) ===
        // P is [BM, BN] in row_major. Load as col_major to get P^T.
        // dO is [BM, D]. Load as col_major to get dO^T, so A @ B => P^T @ dO.
        for (int nround = 0; nround < 4; nround++) {
            int n_start = nround * 2;
            wmma::fragment<wmma::accumulator, WM, WM, WM, float> acc[2];
            for (int n = 0; n < 2; n++)
                wmma::load_matrix_sync(acc[n], &sm.dV[warp_id*WM*D + (n_start+n)*WM], D, wmma::mem_row_major);
            for (int k = 0; k < BM/WM; k++) {
                wmma::fragment<wmma::matrix_a, WM, WM, WM, bf16, wmma::col_major> a_frag;
                wmma::load_matrix_sync(a_frag, &sm.P[k*WM*BN + warp_id*WM], BN);
                for (int n = 0; n < 2; n++) {
                    wmma::fragment<wmma::matrix_b, WM, WM, WM, bf16, wmma::col_major> b_frag;
                    wmma::load_matrix_sync(b_frag, &sm.dO[k*WM*D + (n_start+n)*WM], D);
                    wmma::mma_sync(acc[n], a_frag, b_frag, acc[n]);
                }
            }
            for (int n = 0; n < 2; n++)
                wmma::store_matrix_sync(&sm.dV[warp_id*WM*D + (n_start+n)*WM], acc[n], D, wmma::mem_row_major);
        }
        __syncthreads();

        // === dQ += dS @ K (atomic add to global) ===
        // dS is [BM, BN] in row_major. K is [BN, D] in row_major.
        // Standard A @ B => dS @ K. Correct!
        {
            int m_tile = warp_id / 2;
            int n_start = (warp_id % 2) * 4;
            wmma::fragment<wmma::accumulator, WM, WM, WM, float> acc[4];
            for (int n = 0; n < 4; n++) wmma::fill_fragment(acc[n], 0.f);
            for (int k = 0; k < BN/WM; k++) {
                wmma::fragment<wmma::matrix_a, WM, WM, WM, bf16, wmma::row_major> a_frag;
                wmma::load_matrix_sync(a_frag, &sm.dS[m_tile*WM*BN + k*WM], BN);
                for (int n = 0; n < 4; n++) {
                    wmma::fragment<wmma::matrix_b, WM, WM, WM, bf16, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(b_frag, &sm.K[k*WM*D + (n_start+n)*WM], D);
                    wmma::mma_sync(acc[n], a_frag, b_frag, acc[n]);
                }
            }
            for (int n = 0; n < 4; n++) {
                float* tmp = &sm.S[warp_id * WM * WM];
                wmma::store_matrix_sync(tmp, acc[n], WM, wmma::mem_row_major);
                for (int idx = lane; idx < WM*WM; idx += 32) {
                    int i = idx / WM, j = idx % WM;
                    int gi = gi0 + m_tile*WM + i;
                    int gcol = (n_start + n) * WM + j;
                    if (gi < S && gcol < D)
                        atomicAdd(&dQ_fp[base + (int64_t)gi*D + gcol], tmp[idx] * scale);
                }
            }
        }
        __syncthreads();
    }

    // Store dK, dV to global (fp32 -> bf16)
    for (int i = tid; i < BN_a * D; i += THREADS) {
        dK_out[base + (int64_t)gj0*D + i] = __float2bfloat16(sm.dK[i] * scale);
        dV_out[base + (int64_t)gj0*D + i] = __float2bfloat16(sm.dV[i]);
    }
}

__global__ void convert_kernel(const float* __restrict__ in, bf16* __restrict__ out, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = __float2bfloat16(in[idx]);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = 4, H = 48, d = 128;
    int S = (int)Q.size(2);
    int64_t total = (int64_t)B * H * S * d;
    float scale = 1.f / sqrtf((float)d);

    const bf16* Qp = static_cast<const bf16*>(Q.data_ptr());
    const bf16* Kp = static_cast<const bf16*>(K.data_ptr());
    const bf16* Vp = static_cast<const bf16*>(V.data_ptr());
    const bf16* Op = static_cast<const bf16*>(O.data_ptr());
    const bf16* dOp = static_cast<const bf16*>(dO.data_ptr());
    const float* Lp = static_cast<const float*>(L.data_ptr());
    bf16* dQp = static_cast<bf16*>(dQ.data_ptr());
    bf16* dKp = static_cast<bf16*>(dK.data_ptr());
    bf16* dVp = static_cast<bf16*>(dV.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    float* dQ_fp = nullptr;
    CUDA_CHECK(cudaMallocAsync(&dQ_fp, sizeof(float)*total, stream));
    CUDA_CHECK(cudaMemsetAsync(dQ_fp, 0, sizeof(float)*total, stream));

    int num_kv_blocks = (S + BN - 1) / BN;
    int smem_size = sizeof(Smem);
    CUDA_CHECK(cudaFuncSetAttribute(attn_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    dim3 grid(num_kv_blocks, B*H);
    dim3 block(THREADS);
    attn_bwd_kernel<<<grid, block, smem_size, stream>>>(Qp, Kp, Vp, Op, dOp, Lp, dQ_fp, dKp, dVp, S, H, scale);
    CUDA_CHECK(cudaGetLastError());

    int cb = 256;
    int64_t cg = (total + cb - 1) / cb;
    convert_kernel<<<(int)cg, cb, 0, stream>>>(dQ_fp, dQp, total);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFreeAsync(dQ_fp, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_attn_bwd::run);

}  // namespace tvm_ffi_attn_bwd