#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda.h>
#include <cstdint>
#include <cmath>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

using namespace nvcuda;

namespace attn_bwd {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int D = 128;
constexpr int WARPS = 8;
constexpr int THREADS = 256;
constexpr float SCALE = 0.08838834764831843f;

// K-major SMEM layout (dK/dV in SMEM, dQ via atomicAdd)
constexpr int K_OFF    = 0;
constexpr int V_OFF    = K_OFF  + BN * D * 2;           // 16384
constexpr int dK_OFF   = V_OFF  + BN * D * 2;           // 32768
constexpr int dV_OFF   = dK_OFF + BN * D * 4;           // 65536
constexpr int Q_OFF    = dV_OFF + BN * D * 4;           // 98304
constexpr int dO_OFF   = Q_OFF  + BM * D * 2;           // 114688
constexpr int S_OFF    = dO_OFF + BM * D * 2;           // 131072
constexpr int dP_OFF   = S_OFF  + BM * BN * 4;           // 147456
constexpr int P_OFF    = dP_OFF + BM * BN * 4;           // 163840
constexpr int D_OFF    = P_OFF  + BM * BN * 2;           // 172032
constexpr int L_OFF    = D_OFF  + BM * 4;                 // 172288
constexpr int SMEM_SIZE = L_OFF + BM * 4;                 // 172544

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n" ::: "memory"); }

__device__ __forceinline__ void load_tile_async(
    __nv_bfloat16* smem, const __nv_bfloat16* gmem, int rows, int row_start, int S, uint64_t bh_off)
{
    constexpr int VEC = 8;
    int total = rows * (D / VEC);
    for (int idx = threadIdx.x; idx < total; idx += THREADS) {
        int i = idx / (D / VEC), dd = idx % (D / VEC);
        int gr = row_start + i;
        if (gr < S)
            cp_async_16(&smem[i * D + dd * VEC], &gmem[bh_off + (uint64_t)gr * D + dd * VEC]);
        else {
            int4 z = make_int4(0, 0, 0, 0);
            *reinterpret_cast<int4*>(&smem[i * D + dd * VEC]) = z;
        }
    }
}

// C[M,N] = A[M,K] @ B[K,N]^T * scale (B stored row_major as [N,K])
template<int M, int N, int K>
__device__ __forceinline__ void wmma_gemm_rc(
    const __nv_bfloat16* A_smem, const __nv_bfloat16* B_smem,
    float* C_smem, float scale, int warp_id)
{
    constexpr int M_T = M/16, N_T = N/16, K_T = K/16;
    constexpr int TOT = M_T * N_T;
    for (int t = 0; t < (TOT + WARPS - 1) / WARPS; t++) {
        int ti = warp_id + t * WARPS;
        if (ti >= TOT) break;
        int mt = ti / N_T, nt = ti % N_T;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
        wmma::fill_fragment(c, 0.0f);
        for (int kt = 0; kt < K_T; kt++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::load_matrix_sync(a, A_smem + mt*16*K + kt*16, K);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
            wmma::load_matrix_sync(b, B_smem + nt*16*K + kt*16, K);
            wmma::mma_sync(c, a, b, c);
        }
        if (scale != 1.0f) for (int i = 0; i < c.num_elements; i++) c.x[i] *= scale;
        wmma::store_matrix_sync(C_smem + mt*16*N + nt*16, c, N, wmma::mem_row_major);
    }
}

// dV_acc[BN,D] += P^T[BN,BM] @ dO[BM,D] (accumulate in SMEM)
__device__ __forceinline__ void compute_dV_acc(
    const __nv_bfloat16* P_smem, const __nv_bfloat16* dO_smem,
    float* dV_acc, int warp_id)
{
    constexpr int M = BN, N = D, K = BM;
    constexpr int M_T = M/16, N_T = N/16, K_T = K/16;
    constexpr int TOT = M_T * N_T;
    for (int t = 0; t < (TOT + WARPS - 1) / WARPS; t++) {
        int ti = warp_id + t * WARPS;
        if (ti >= TOT) break;
        int mt = ti / N_T, nt = ti % N_T;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
        wmma::load_matrix_sync(c, dV_acc + mt*16*D + nt*16, D, wmma::mem_row_major);
        for (int kt = 0; kt < K_T; kt++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a;
            wmma::load_matrix_sync(a, P_smem + kt*16*BN + mt*16, BN);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
            wmma::load_matrix_sync(b, dO_smem + kt*16*D + nt*16, D);
            wmma::mma_sync(c, a, b, c);
        }
        wmma::store_matrix_sync(dV_acc + mt*16*D + nt*16, c, D, wmma::mem_row_major);
    }
}

// dK_acc[BN,D] += dS^T[BN,BM] @ Q[BM,D] (accumulate in SMEM)
__device__ __forceinline__ void compute_dK_acc(
    const __nv_bfloat16* dS_smem, const __nv_bfloat16* Q_smem,
    float* dK_acc, int warp_id)
{
    constexpr int M = BN, N = D, K = BM;
    constexpr int M_T = M/16, N_T = N/16, K_T = K/16;
    constexpr int TOT = M_T * N_T;
    for (int t = 0; t < (TOT + WARPS - 1) / WARPS; t++) {
        int ti = warp_id + t * WARPS;
        if (ti >= TOT) break;
        int mt = ti / N_T, nt = ti % N_T;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
        wmma::load_matrix_sync(c, dK_acc + mt*16*D + nt*16, D, wmma::mem_row_major);
        for (int kt = 0; kt < K_T; kt++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> a;
            wmma::load_matrix_sync(a, dS_smem + kt*16*BN + mt*16, BN);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
            wmma::load_matrix_sync(b, Q_smem + kt*16*D + nt*16, D);
            wmma::mma_sync(c, a, b, c);
        }
        wmma::store_matrix_sync(dK_acc + mt*16*D + nt*16, c, D, wmma::mem_row_major);
    }
}

// dQ[BM,D] += dS[BM,BN] @ K[BN,D] (atomicAdd to global fp32)
__device__ __forceinline__ void compute_dQ_atomic(
    const __nv_bfloat16* dS_smem, const __nv_bfloat16* K_smem,
    float* dQ_gmem, int q_start, uint64_t bh_off, int S,
    float* tmp_smem, int warp_id)
{
    constexpr int M = BM, N = D, K = BN;
    constexpr int M_T = M/16, N_T = N/16, K_T = K/16;
    constexpr int TOT = M_T * N_T;
    int lane_id = threadIdx.x % 32;
    for (int t = 0; t < (TOT + WARPS - 1) / WARPS; t++) {
        int ti = warp_id + t * WARPS;
        if (ti >= TOT) break;
        int mt = ti / N_T, nt = ti % N_T;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
        wmma::fill_fragment(c, 0.0f);
        for (int kt = 0; kt < K_T; kt++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::load_matrix_sync(a, dS_smem + mt*16*BN + kt*16, BN);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
            wmma::load_matrix_sync(b, K_smem + kt*16*D + nt*16, D);
            wmma::mma_sync(c, a, b, c);
        }
        float* tile = tmp_smem + warp_id * 256;
        wmma::store_matrix_sync(tile, c, 16, wmma::mem_row_major);
        __syncwarp();
        int row_base = q_start + mt * 16;
        int col_base = nt * 16;
        for (int idx = lane_id; idx < 256; idx += 32) {
            int lr = idx / 16, lc = idx % 16;
            int gr = row_base + lr;
            if (gr < S)
                atomicAdd(&dQ_gmem[bh_off + (uint64_t)gr * D + col_base + lc], tile[lr * 16 + lc]);
        }
        __syncwarp();
    }
}

__global__ void compute_D_kernel(
    const __nv_bfloat16* __restrict__ dO,
    const __nv_bfloat16* __restrict__ O,
    float* __restrict__ D_buf,
    int S)
{
    int bh = blockIdx.x;
    int s = blockIdx.y;
    int tid = threadIdx.x;
    
    uint64_t base = (uint64_t)bh * S * D + (uint64_t)s * D;
    float sum = 0.0f;
    if (tid < D)
        sum = __bfloat162float(dO[base + tid]) * __bfloat162float(O[base + tid]);
    
    __shared__ float smem[128];
    smem[tid] = sum;
    __syncthreads();
    
    for (int stride = 64; stride > 0; stride /= 2) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    
    if (tid == 0) D_buf[bh * S + s] = smem[0];
}

__global__ __launch_bounds__(256, 1)
void attn_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    const float* __restrict__ D_buf,
    float* __restrict__ dQ_fp32,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    int S)
{
    int bh = blockIdx.x;
    int k_blk = blockIdx.y;
    int k_start = k_blk * BN;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (k_start >= S) return;

    extern __shared__ char smem[];
    __nv_bfloat16* K_s    = reinterpret_cast<__nv_bfloat16*>(smem + K_OFF);
    __nv_bfloat16* V_s    = reinterpret_cast<__nv_bfloat16*>(smem + V_OFF);
    float* dK_acc         = reinterpret_cast<float*>(smem + dK_OFF);
    float* dV_acc         = reinterpret_cast<float*>(smem + dV_OFF);
    __nv_bfloat16* Q_s    = reinterpret_cast<__nv_bfloat16*>(smem + Q_OFF);
    __nv_bfloat16* dO_s   = reinterpret_cast<__nv_bfloat16*>(smem + dO_OFF);
    float* S_s            = reinterpret_cast<float*>(smem + S_OFF);
    float* dP_s           = reinterpret_cast<float*>(smem + dP_OFF);
    __nv_bfloat16* P_s    = reinterpret_cast<__nv_bfloat16*>(smem + P_OFF);
    float* D_s            = reinterpret_cast<float*>(smem + D_OFF);
    float* L_s            = reinterpret_cast<float*>(smem + L_OFF);

    uint64_t bh_off = (uint64_t)bh * S * D;

    // Load K, V persistently
    load_tile_async(K_s, K, BN, k_start, S, bh_off);
    load_tile_async(V_s, V, BN, k_start, S, bh_off);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    // Zero dK_acc, dV_acc
    for (int idx = tid; idx < BN * D; idx += THREADS) {
        dK_acc[idx] = 0.0f;
        dV_acc[idx] = 0.0f;
    }
    __syncthreads();

    int num_q = (S + BM - 1) / BM;
    for (int qj = 0; qj < num_q; qj++) {
        int q_start = qj * BM;

        // Load Q, dO via cp.async
        load_tile_async(Q_s, Q, BM, q_start, S, bh_off);
        load_tile_async(dO_s, dO, BM, q_start, S, bh_off);
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        // Load D, L (scalar)
        for (int i = tid; i < BM; i += THREADS) {
            int qr = q_start + i;
            D_s[i] = (qr < S) ? D_buf[bh * S + qr] : 0.0f;
            L_s[i] = (qr < S) ? L[bh * S + qr] : 0.0f;
        }
        __syncthreads();

        // S = Q @ K^T * scale
        wmma_gemm_rc<BM, BN, D>(Q_s, K_s, S_s, SCALE, warp_id);
        __syncthreads();

        // dP = dO @ V^T
        wmma_gemm_rc<BM, BN, D>(dO_s, V_s, dP_s, 1.0f, warp_id);
        __syncthreads();

        // P = exp(S - L), store bf16 in P_s, keep fp32 in S_s
        for (int idx = tid; idx < BM * BN; idx += THREADS) {
            int i = idx / BN, j = idx % BN;
            int kr = k_start + j;
            if (kr >= S) {
                S_s[idx] = 0.0f;
                P_s[idx] = __float2bfloat16(0.0f);
            } else {
                float pv = expf(S_s[idx] - L_s[i]);
                S_s[idx] = pv;
                P_s[idx] = __float2bfloat16(pv);
            }
        }
        __syncthreads();

        // dV_acc += P^T @ dO (in SMEM, no atomics!)
        compute_dV_acc(P_s, dO_s, dV_acc, warp_id);
        __syncthreads();

        // dS = P * (dP - D) * scale, store bf16 in P_s (overwrite P)
        for (int idx = tid; idx < BM * BN; idx += THREADS) {
            int i = idx / BN;
            float pv = S_s[idx];
            float dpv = dP_s[idx];
            P_s[idx] = __float2bfloat16(pv * (dpv - D_s[i]) * SCALE);
        }
        __syncthreads();

        // dK_acc += dS^T @ Q (in SMEM, no atomics!)
        compute_dK_acc(P_s, Q_s, dK_acc, warp_id);
        __syncthreads();

        // dQ += dS @ K (atomicAdd to global fp32, reuse V_s as tmp)
        compute_dQ_atomic(P_s, K_s, dQ_fp32, q_start, bh_off, S,
                          reinterpret_cast<float*>(V_s), warp_id);
        __syncthreads();
    }

    // Store dK, dV from SMEM to global bf16
    for (int idx = tid; idx < BN * D; idx += THREADS) {
        int i = idx / D, dd = idx % D;
        int kr = k_start + i;
        if (kr < S) {
            dK_out[bh_off + (uint64_t)kr * D + dd] = __float2bfloat16(dK_acc[idx]);
            dV_out[bh_off + (uint64_t)kr * D + dd] = __float2bfloat16(dV_acc[idx]);
        }
    }
}

__global__ void convert_fp32_to_bf16_kernel(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = __float2bfloat16(src[idx]);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV)
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int B = static_cast<int>(Q.size(0));
    int H = static_cast<int>(Q.size(1));
    int S = static_cast<int>(Q.size(2));
    int BH = B * H;

    const __nv_bfloat16* Q_p  = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_p  = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_p  = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_p  = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_p = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_p          = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_p       = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_p       = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_p       = static_cast<__nv_bfloat16*>(dV.data_ptr());

    cudaStream_t stream =
        static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    size_t total_elems = (size_t)BH * S * D;
    size_t fp32_bytes = total_elems * sizeof(float);
    size_t d_buf_bytes = (size_t)BH * S * sizeof(float);

    float* dQ_fp32 = nullptr;
    float* D_buf = nullptr;
    CUDA_CHECK(cudaMallocAsync(&dQ_fp32, fp32_bytes, stream));
    CUDA_CHECK(cudaMallocAsync(&D_buf, d_buf_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dQ_fp32, 0, fp32_bytes, stream));

    // Precompute D = rowsum(dO * O)
    dim3 d_grid(BH, S, 1);
    dim3 d_block(128, 1, 1);
    compute_D_kernel<<<d_grid, d_block, 0, stream>>>(dO_p, O_p, D_buf, S);

    // Main backward kernel
    int num_k_blocks = (S + BN - 1) / BN;
    dim3 grid(BH, num_k_blocks, 1);
    dim3 block(THREADS, 1, 1);

    CUDA_CHECK(cudaFuncSetAttribute(
        attn_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    attn_bwd_kernel<<<grid, block, SMEM_SIZE, stream>>>(
        Q_p, K_p, V_p, dO_p, L_p, D_buf, dQ_fp32, dK_p, dV_p, S);

    // Convert dQ from fp32 to bf16
    int conv_threads = 256;
    int conv_blocks = (total_elems + conv_threads - 1) / conv_threads;
    convert_fp32_to_bf16_kernel<<<conv_blocks, conv_threads, 0, stream>>>(dQ_fp32, dQ_p, total_elems);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(dQ_fp32, stream));
    CUDA_CHECK(cudaFreeAsync(D_buf, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attn_bwd::run);

} // namespace attn_bwd