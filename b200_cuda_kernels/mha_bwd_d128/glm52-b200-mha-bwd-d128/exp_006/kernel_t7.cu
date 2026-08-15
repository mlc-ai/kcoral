#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace tvm_ffi_attn_bwd {

constexpr int D = 128;
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int NTHR = 128;
constexpr int NWARPS = NTHR / 32;
constexpr float SCALE = 0.08838834764831845f;
constexpr int ACC_PER_THREAD = BN * D / NTHR; // 64

template <int M, int N, int K, bool A_COL, bool B_COL, bool C_COL>
__device__ __forceinline__ void wmma_gemm(const __nv_bfloat16* A, int lda,
                          const __nv_bfloat16* B, int ldb,
                          float* C, int ldc, bool accum) {
    constexpr int TM = 16, TN = 16, TK = 16;
    constexpr int ntm = M / TM, ntn = N / TN, ntk = K / TK;
    const int warp = threadIdx.x / 32;
    const int total = ntm * ntn;
    for (int t = warp; t < total; t += NWARPS) {
        int ti = t / ntn, tj = t % ntn;
        wmma::fragment<wmma::accumulator, TM, TN, TK, float> cf;
        if (accum) {
            const float* cptr = C + (C_COL ? (tj * TN * ldc + ti * TM) : (ti * TM * ldc + tj * TN));
            wmma::load_matrix_sync(cf, cptr, ldc, C_COL ? wmma::mem_col_major : wmma::mem_row_major);
        } else {
            wmma::fill_fragment(cf, 0.0f);
        }
        for (int kk = 0; kk < ntk; kk++) {
            if constexpr (!A_COL && !B_COL) {
                wmma::fragment<wmma::matrix_a, TM, TN, TK, __nv_bfloat16, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, TM, TN, TK, __nv_bfloat16, wmma::row_major> bf;
                wmma::load_matrix_sync(af, A + ti * TM * lda + kk * TK, lda);
                wmma::load_matrix_sync(bf, B + kk * TK * ldb + tj * TN, ldb);
                wmma::mma_sync(cf, af, bf, cf);
            } else if constexpr (!A_COL && B_COL) {
                wmma::fragment<wmma::matrix_a, TM, TN, TK, __nv_bfloat16, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, TM, TN, TK, __nv_bfloat16, wmma::col_major> bf;
                wmma::load_matrix_sync(af, A + ti * TM * lda + kk * TK, lda);
                wmma::load_matrix_sync(bf, B + tj * TN * ldb + kk * TK, ldb);
                wmma::mma_sync(cf, af, bf, cf);
            } else if constexpr (A_COL && !B_COL) {
                wmma::fragment<wmma::matrix_a, TM, TN, TK, __nv_bfloat16, wmma::col_major> af;
                wmma::fragment<wmma::matrix_b, TM, TN, TK, __nv_bfloat16, wmma::row_major> bf;
                wmma::load_matrix_sync(af, A + kk * TK * lda + ti * TM, lda);
                wmma::load_matrix_sync(bf, B + kk * TK * ldb + tj * TN, ldb);
                wmma::mma_sync(cf, af, bf, cf);
            } else {
                wmma::fragment<wmma::matrix_a, TM, TN, TK, __nv_bfloat16, wmma::col_major> af;
                wmma::fragment<wmma::matrix_b, TM, TN, TK, __nv_bfloat16, wmma::col_major> bf;
                wmma::load_matrix_sync(af, A + kk * TK * lda + ti * TM, lda);
                wmma::load_matrix_sync(bf, B + tj * TN * ldb + kk * TK, ldb);
                wmma::mma_sync(cf, af, bf, cf);
            }
        }
        float* cptr = C + (C_COL ? (tj * TN * ldc + ti * TM) : (ti * TM * ldc + tj * TN));
        wmma::store_matrix_sync(cptr, cf, ldc, C_COL ? wmma::mem_col_major : wmma::mem_row_major);
    }
}

__device__ __forceinline__ void cp_async_16B(void* smem, const void* gmem) {
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

__device__ void load_tile_async(
        const __nv_bfloat16* gptr, int64_t gstride,
        __nv_bfloat16* sptr, int sstride,
        int rows, int cols, int row_base, int S_limit) {
    int tid = threadIdx.x;
    int nvec = cols / 8;
    int total = rows * nvec;
    for (int idx = tid; idx < total; idx += NTHR) {
        int r = idx / nvec;
        int c = (idx % nvec) * 8;
        if (row_base + r < S_limit) {
            cp_async_16B(sptr + r * sstride + c, gptr + (size_t)r * gstride + c);
        } else {
            *reinterpret_cast<float4*>(sptr + r * sstride + c) = {0.0f, 0.0f, 0.0f, 0.0f};
        }
    }
}

__global__ void attn_bwd_kernel(
        const __nv_bfloat16* __restrict__ Q,
        const __nv_bfloat16* __restrict__ K,
        const __nv_bfloat16* __restrict__ V,
        const __nv_bfloat16* __restrict__ O,
        const __nv_bfloat16* __restrict__ dO,
        const float* __restrict__ L,
        float* __restrict__ dQ_ws,
        __nv_bfloat16* __restrict__ dK_g,
        __nv_bfloat16* __restrict__ dV_g,
        int S, int H,
        int64_t stride_b, int64_t stride_h, int64_t stride_s,
        int64_t l_stride_b, int64_t l_stride_h, int64_t l_stride_s)
{
    int kv_block = blockIdx.x;
    int bh = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int col_base = kv_block * BN;
    if (col_base >= S) return;

    size_t base4 = (size_t)b * stride_b + h * stride_h;
    size_t base_l = (size_t)b * l_stride_b + h * l_stride_h;

    const __nv_bfloat16* Q_bh = Q + base4;
    const __nv_bfloat16* K_bh = K + base4;
    const __nv_bfloat16* V_bh = V + base4;
    const __nv_bfloat16* O_bh = O + base4;
    const __nv_bfloat16* dO_bh = dO + base4;
    const float* L_bh = L + base_l;
    float* dQ_ws_bh = dQ_ws + base4;
    __nv_bfloat16* dK_bh = dK_g + base4;
    __nv_bfloat16* dV_bh = dV_g + base4;

    // Shared memory layout (~112KB for 2 CTAs/SM):
    // sK:      [BN, D]  bf16  16KB  (persistent)
    // sV:      [BN, D]  bf16  16KB  (persistent)
    // sQ:      [BM, D]  bf16  16KB  (reloaded)
    // sdO:     [BM, D]  bf16  16KB  (reloaded)
    // sS/sO:   [BM, BN] f32  16KB   (O→S→dP→temp1)
    // sP:      [BM, BN] f32  16KB   (P→dS→temp2)
    // sP_bf16: [BM, BN] bf16  8KB
    // sdS_bf16:[BM, BN] bf16  8KB
    // sL:      [BM]     f32  256B
    // sD:      [BM]     f32  256B
    // Total: 112.5KB → 2 CTAs/SM (228KB max)
    extern __shared__ char smem_raw[];
    __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sV = sK + BN * D;
    __nv_bfloat16* sQ = sV + BN * D;
    __nv_bfloat16* sdO = sQ + BM * D;
    float* sS = reinterpret_cast<float*>(sdO + BM * D);
    float* sP = sS + BM * BN;
    __nv_bfloat16* sP_bf16 = reinterpret_cast<__nv_bfloat16*>(sP + BM * BN);
    __nv_bfloat16* sdS_bf16 = sP_bf16 + BM * BN;
    float* sL = reinterpret_cast<float*>(sdS_bf16 + BM * BN);
    float* sD = sL + BM;
    // Aliases
    __nv_bfloat16* sO = reinterpret_cast<__nv_bfloat16*>(sS); // O reuses sS buffer
    float* sTemp = sS; // sS+sP = 32KB temp for dV/dK/dQ

    int tid = threadIdx.x;

    // Register-based dK/dV accumulators (eliminates 64KB shared memory)
    float dK_reg[ACC_PER_THREAD];
    float dV_reg[ACC_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ACC_PER_THREAD; i++) {
        dK_reg[i] = 0.0f;
        dV_reg[i] = 0.0f;
    }

    // Load persistent K, V tiles
    load_tile_async(K_bh + (size_t)col_base * stride_s, stride_s, sK, D, BN, D, col_base, S);
    load_tile_async(V_bh + (size_t)col_base * stride_s, stride_s, sV, D, BN, D, col_base, S);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    for (int qb = 0; qb < S; qb += BM) {
        // Async load Q, dO, O for this iteration
        load_tile_async(Q_bh + (size_t)qb * stride_s, stride_s, sQ, D, BM, D, qb, S);
        load_tile_async(dO_bh + (size_t)qb * stride_s, stride_s, sdO, D, BM, D, qb, S);
        load_tile_async(O_bh + (size_t)qb * stride_s, stride_s, sO, D, BM, D, qb, S);

        // Load LSE (small, sync load)
        for (int i = tid; i < BM; i += NTHR)
            sL[i] = (qb + i < S) ? L_bh[(size_t)(qb + i) * l_stride_s] : 0.0f;

        cp_async_wait_all();
        __syncthreads();

        // D[r] = sum_c O[r,c] * dO[r,c] -- 2 threads per row, vectorized
        {
            int r = tid / 2;
            int sub = tid % 2;
            if (r < BM) {
                float acc = 0.0f;
                if (qb + r < S) {
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        int c = sub * 64 + i * 8;
                        float4 ov = *reinterpret_cast<float4*>(&sO[r * D + c]);
                        float4 dv = *reinterpret_cast<float4*>(&sdO[r * D + c]);
                        __nv_bfloat16* oh = reinterpret_cast<__nv_bfloat16*>(&ov);
                        __nv_bfloat16* dh = reinterpret_cast<__nv_bfloat16*>(&dv);
                        #pragma unroll
                        for (int k = 0; k < 8; k++)
                            acc += __bfloat162float(oh[k]) * __bfloat162float(dh[k]);
                    }
                }
                acc += __shfl_xor_sync(0xffffffff, acc, 1);
                if (sub == 0) sD[r] = acc;
            }
        }
        __syncthreads(); // D done, sO(=sS) can be overwritten

        // === Reordered: S → P → dP → dS → dV → dK → dQ ===

        // S = Q @ K^T -> sS (overwrites sO)
        wmma_gemm<BM, BN, D, false, true, false>(sQ, D, sK, D, sS, BN, false);
        __syncthreads();

        // P = exp(S*scale - L) -> sP, sP_bf16
        for (int idx = tid; idx < BM * BN; idx += NTHR) {
            int r = idx / BN, c = idx % BN;
            float p = __expf(sS[r * BN + c] * SCALE - sL[r]);
            sP[r * BN + c] = p;
            sP_bf16[r * BN + c] = __float2bfloat16(p);
        }
        __syncthreads();

        // dP = dO @ V^T -> sS (overwrites S)
        wmma_gemm<BM, BN, D, false, true, false>(sdO, D, sV, D, sS, BN, false);
        __syncthreads();

        // dS = P * (dP - D) -> sP (overwrites P), sdS_bf16
        for (int idx = tid; idx < BM * BN; idx += NTHR) {
            int r = idx / BN, c = idx % BN;
            float p = sP[r * BN + c];
            float ds = p * (sS[r * BN + c] - sD[r]);
            sP[r * BN + c] = ds;
            sdS_bf16[r * BN + c] = __float2bfloat16(ds);
        }
        __syncthreads();

        // Now sS and sP are free → sTemp = sS+sP (32KB) for dV/dK/dQ

        // dV = P^T @ dO -> sTemp [BN, D]
        wmma_gemm<BN, D, BM, true, false, false>(sP_bf16, BN, sdO, D, sTemp, D, false);
        __syncthreads();

        // Accumulate dV from sTemp into registers
        // Thread tid owns sTemp[tid*64 .. tid*64+63] = row tid/2, cols (tid%2)*64..(tid%2)*64+63
        {
            float4* st4 = reinterpret_cast<float4*>(sTemp);
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                float4 v = st4[tid * 16 + i];
                dV_reg[i*4 + 0] += v.x;
                dV_reg[i*4 + 1] += v.y;
                dV_reg[i*4 + 2] += v.z;
                dV_reg[i*4 + 3] += v.w;
            }
        }
        __syncthreads();

        // dK = dS^T @ Q -> sTemp [BN, D]
        wmma_gemm<BN, D, BM, true, false, false>(sdS_bf16, BN, sQ, D, sTemp, D, false);
        __syncthreads();

        // Accumulate dK from sTemp into registers
        {
            float4* st4 = reinterpret_cast<float4*>(sTemp);
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                float4 v = st4[tid * 16 + i];
                dK_reg[i*4 + 0] += v.x;
                dK_reg[i*4 + 1] += v.y;
                dK_reg[i*4 + 2] += v.z;
                dK_reg[i*4 + 3] += v.w;
            }
        }
        __syncthreads();

        // dQ = dS @ K -> sTemp [BM, D]
        wmma_gemm<BM, D, BN, false, false, false>(sdS_bf16, BN, sK, D, sTemp, D, false);
        __syncthreads();

        // AtomicAdd dQ to global workspace (with scale)
        // Optimized: r = tid % BM, c_start = (tid / BM) * (D / (NTHR/BM))
        // Each warp accesses 32 different rows = 32 different cache lines → no conflicts
        {
            int r = tid % BM;
            int c_start = (tid / BM) * (D / (NTHR / BM));
            int global_row = qb + r;
            if (global_row < S) {
                float* gptr = &dQ_ws_bh[(size_t)global_row * stride_s + c_start];
                float* sptr = &sTemp[r * D + c_start];
                #pragma unroll
                for (int c = 0; c < D / (NTHR / BM); c++) {
                    atomicAdd(&gptr[c], sptr[c] * SCALE);
                }
            }
        }
        __syncthreads();
    }

    // Store dK, dV from registers to global (apply scale to dK)
    // Thread tid owns row tid/2, cols (tid%2)*64..(tid%2)*64+63
    {
        int r = tid / 2;
        int c_start = (tid % 2) * 64;
        if (col_base + r < S) {
            __nv_bfloat16* dK_row = dK_bh + (size_t)(col_base + r) * stride_s + c_start;
            __nv_bfloat16* dV_row = dV_bh + (size_t)(col_base + r) * stride_s + c_start;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                float4 dkv, dvv;
                __nv_bfloat16* dkp = reinterpret_cast<__nv_bfloat16*>(&dkv);
                __nv_bfloat16* dvp = reinterpret_cast<__nv_bfloat16*>(&dvv);
                #pragma unroll
                for (int k = 0; k < 8; k++) {
                    dkp[k] = __float2bfloat16(dK_reg[i*8 + k] * SCALE);
                    dvp[k] = __float2bfloat16(dV_reg[i*8 + k]);
                }
                *reinterpret_cast<float4*>(dK_row + i*8) = dkv;
                *reinterpret_cast<float4*>(dV_row + i*8) = dvv;
            }
        }
    }
}

__global__ void convert_kernel(const float* src, __nv_bfloat16* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2bfloat16(src[idx]);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = (int)Q.size(0), H = (int)Q.size(1), S = (int)Q.size(2);
    int BH = B * H;

    int64_t stride_b = (int64_t)H * S * D;
    int64_t stride_h = (int64_t)S * D;
    int64_t stride_s = (int64_t)D;
    int64_t l_stride_b = (int64_t)H * S;
    int64_t l_stride_h = (int64_t)S;
    int64_t l_stride_s = 1;

    const __nv_bfloat16* Q_p = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_p = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_p = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_p = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_p = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_p = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_p = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_p = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_p = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    size_t ws_count = (size_t)B * H * S * D;
    float* dQ_ws = nullptr;
    CUDA_CHECK(cudaMallocAsync(&dQ_ws, ws_count * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dQ_ws, 0, ws_count * sizeof(float), stream));

    int num_kv = (S + BN - 1) / BN;
    dim3 grid(num_kv, BH);
    dim3 block(NTHR);

    // Shared memory: 16+16+16+16+16+16+8+8+0.25+0.25 = 112.5KB → 2 CTAs/SM
    int smem = BN*D*2 + BN*D*2 + BM*D*2 + BM*D*2  // sK, sV, sQ, sdO (64KB bf16)
             + BM*BN*4 + BM*BN*4                    // sS, sP (32KB fp32)
             + BM*BN*2 + BM*BN*2                    // sP_bf16, sdS_bf16 (16KB bf16)
             + BM*4 + BM*4;                         // sL, sD (512B fp32)
    smem = (smem + 15) & ~15;

    CUDA_CHECK(cudaFuncSetAttribute(attn_bwd_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    CUDA_CHECK(cudaFuncSetAttribute(attn_bwd_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));

    attn_bwd_kernel<<<grid, block, smem, stream>>>(
        Q_p, K_p, V_p, O_p, dO_p, L_p, dQ_ws, dK_p, dV_p,
        S, H, stride_b, stride_h, stride_s,
        l_stride_b, l_stride_h, l_stride_s);
    CUDA_CHECK(cudaGetLastError());

    int total = (int)ws_count;
    convert_kernel<<<(total+255)/256, 256, 0, stream>>>(dQ_ws, dQ_p, total);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFreeAsync(dQ_ws, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_attn_bwd::run);

}  // namespace tvm_ffi_attn_bwd