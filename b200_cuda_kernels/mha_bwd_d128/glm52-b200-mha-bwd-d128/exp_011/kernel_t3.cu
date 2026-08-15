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

constexpr int SMEM_SIZE =
    BN * D * 2 + BN * D * 2 + BN * D * 4 + BN * D * 4 +
    BM * D * 2 + BM * D * 2 + BM * D * 4 +
    BM * BN * 4 + BM * BN * 4 + BM * BN * 2 + BM * 4 + BM * 4;

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
    uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

__device__ __forceinline__ void load_tile_async(
    __nv_bfloat16* smem, const __nv_bfloat16* gmem,
    int rows, int row_start, int S, uint64_t bh_off)
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
        if (scale != 1.0f)
            for (int i = 0; i < c.num_elements; i++) c.x[i] *= scale;
        wmma::store_matrix_sync(C_smem + mt*16*N + nt*16, c, N, wmma::mem_row_major);
    }
}

__device__ __forceinline__ void compute_dV(
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

__device__ __forceinline__ void compute_dK(
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

__device__ __forceinline__ void compute_dQ_acc(
    const __nv_bfloat16* dS_smem, const __nv_bfloat16* K_smem,
    float* dQ_acc, int warp_id)
{
    constexpr int M = BM, N = D, K = BN;
    constexpr int M_T = M/16, N_T = N/16, K_T = K/16;
    constexpr int TOT = M_T * N_T;
    for (int t = 0; t < (TOT + WARPS - 1) / WARPS; t++) {
        int ti = warp_id + t * WARPS;
        if (ti >= TOT) break;
        int mt = ti / N_T, nt = ti % N_T;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
        wmma::load_matrix_sync(c, dQ_acc + mt*16*D + nt*16, D, wmma::mem_row_major);
        for (int kt = 0; kt < K_T; kt++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
            wmma::load_matrix_sync(a, dS_smem + mt*16*BN + kt*16, BN);
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
            wmma::load_matrix_sync(b, K_smem + kt*16*D + nt*16, D);
            wmma::mma_sync(c, a, b, c);
        }
        wmma::store_matrix_sync(dQ_acc + mt*16*D + nt*16, c, D, wmma::mem_row_major);
    }
}

__global__ void attn_bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ_out,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    float* __restrict__ dQ_fp32,
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
    __nv_bfloat16* K_s    = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* V_s    = K_s + BN * D;
    float* dK_acc         = reinterpret_cast<float*>(V_s + BN * D);
    float* dV_acc         = dK_acc + BN * D;
    __nv_bfloat16* Q_s    = reinterpret_cast<__nv_bfloat16*>(dV_acc + BN * D);
    __nv_bfloat16* dO_s   = Q_s + BM * D;
    float* dQ_acc         = reinterpret_cast<float*>(dO_s + BM * D);
    float* S_s            = dQ_acc + BM * D;
    float* dP_s           = S_s + BM * BN;
    __nv_bfloat16* P_dS_s = reinterpret_cast<__nv_bfloat16*>(dP_s + BM * BN);
    float* D_s            = reinterpret_cast<float*>(P_dS_s + BM * BN);
    float* L_s            = D_s + BM;

    uint64_t bh_off = (uint64_t)bh * S * D;

    load_tile_async(K_s, K, BN, k_start, S, bh_off);
    load_tile_async(V_s, V, BN, k_start, S, bh_off);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    for (int idx = tid; idx < BN * D; idx += THREADS) {
        dK_acc[idx] = 0.0f;
        dV_acc[idx] = 0.0f;
    }
    __syncthreads();

    int num_q = (S + BM - 1) / BM;
    for (int qj = 0; qj < num_q; qj++) {
        int q_start = qj * BM;

        load_tile_async(Q_s, Q, BM, q_start, S, bh_off);
        load_tile_async(dO_s, dO, BM, q_start, S, bh_off);
        __nv_bfloat16* O_tmp = reinterpret_cast<__nv_bfloat16*>(S_s);
        load_tile_async(O_tmp, O, BM, q_start, S, bh_off);
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        for (int i = tid; i < BM; i += THREADS)
            L_s[i] = (q_start + i < S) ? L[bh * S + q_start + i] : 0.0f;

        for (int i = warp_id; i < BM; i += WARPS) {
            float sum = 0.0f;
            for (int dd = lane_id; dd < D; dd += 32)
                sum += __bfloat162float(dO_s[i*D+dd]) * __bfloat162float(O_tmp[i*D+dd]);
            for (int offset = 16; offset > 0; offset /= 2)
                sum += __shfl_xor_sync(0xffffffff, sum, offset);
            if (lane_id == 0) D_s[i] = sum;
        }

        for (int idx = tid; idx < BM * D; idx += THREADS)
            dQ_acc[idx] = 0.0f;
        __syncthreads();

        wmma_gemm_rc<BM, BN, D>(Q_s, K_s, S_s, SCALE, warp_id);
        __syncthreads();

        wmma_gemm_rc<BM, BN, D>(dO_s, V_s, dP_s, 1.0f, warp_id);
        __syncthreads();

        for (int idx = tid; idx < BM * BN; idx += THREADS) {
            int i = idx / BN, j = idx % BN;
            int kr = k_start + j;
            if (kr >= S) {
                S_s[idx] = 0.0f;
                P_dS_s[idx] = __float2bfloat16(0.0f);
            } else {
                float pv = __expf(S_s[idx] - L_s[i]);
                S_s[idx] = pv;
                P_dS_s[idx] = __float2bfloat16(pv);
            }
        }
        __syncthreads();

        compute_dV(P_dS_s, dO_s, dV_acc, warp_id);
        __syncthreads();

        for (int idx = tid; idx < BM * BN; idx += THREADS) {
            int i = idx / BN;
            float pv = S_s[idx];
            float dpv = dP_s[idx];
            P_dS_s[idx] = __float2bfloat16(pv * (dpv - D_s[i]) * SCALE);
        }
        __syncthreads();

        compute_dK(P_dS_s, Q_s, dK_acc, warp_id);
        __syncthreads();

        compute_dQ_acc(P_dS_s, K_s, dQ_acc, warp_id);
        __syncthreads();

        for (int idx = tid; idx < BM * D; idx += THREADS) {
            int i = idx / D, dd = idx % D;
            int qr = q_start + i;
            if (qr < S)
                atomicAdd(&dQ_fp32[(uint64_t)bh * S * D + (uint64_t)qr * D + dd], dQ_acc[idx]);
        }
    }

    for (int idx = tid; idx < BN * D; idx += THREADS) {
        int i = idx / D, dd = idx % D;
        int kr = k_start + i;
        if (kr < S) {
            dK_out[(uint64_t)bh * S * D + (uint64_t)kr * D + dd] = __float2bfloat16(dK_acc[idx]);
            dV_out[(uint64_t)bh * S * D + (uint64_t)kr * D + dd] = __float2bfloat16(dV_acc[idx]);
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

    float* dQ_fp32 = nullptr;
    CUDA_CHECK(cudaMallocAsync(&dQ_fp32, fp32_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(dQ_fp32, 0, fp32_bytes, stream));

    int num_k_blocks = (S + BN - 1) / BN;
    dim3 grid(BH, num_k_blocks, 1);
    dim3 block(THREADS, 1, 1);

    CUDA_CHECK(cudaFuncSetAttribute(
        attn_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));

    attn_bwd_kernel<<<grid, block, SMEM_SIZE, stream>>>(
        Q_p, K_p, V_p, O_p, dO_p, L_p, dQ_p, dK_p, dV_p, dQ_fp32, S);

    int conv_threads = 256;
    int conv_blocks = (total_elems + conv_threads - 1) / conv_threads;
    convert_fp32_to_bf16_kernel<<<conv_blocks, conv_threads, 0, stream>>>(dQ_fp32, dQ_p, total_elems);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaFreeAsync(dQ_fp32, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, attn_bwd::run);

} // namespace attn_bwd