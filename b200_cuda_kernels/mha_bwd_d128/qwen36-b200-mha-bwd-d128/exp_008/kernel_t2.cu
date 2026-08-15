#include <cuda_runtime.h>
#include <cuda_bfloat16.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd_impl {

__forceinline__ __device__ static float bf16_to_f32(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__forceinline__ __device__ static __nv_bfloat16 f32_to_bf16(float x) {
    return __float2bfloat16(x);
}

// ============================================================
// Compute P = softmax(Q @ K^T / sqrt(d)) for given (b,h,qpos)
// Each block computes one row of attention scores
// Uses grid-stride loops so any S fits in blockDim.x threads
// ============================================================
__global__ void compute_P_and_D_kernel(
    const __nv_bfloat16* Q,     // [B, H, S, d]
    const __nv_bfloat16* K,     // [B, H, S, d]
    const __nv_bfloat16* O,     // [B, H, S, d]
    const __nv_bfloat16* dO,    // [B, H, S, d]
    float* P_out,               // [B, H, S, S] -- softmax probs
    float* D_out,               // [B, H, S]    -- dO·O rowsum
    int B, int H, int S, int d)
{
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int qpos = blockIdx.y;
    
    if (b >= B || h >= H || qpos >= S) return;
    
    int bh_off = (int)((uint64_t)(b * H + h) * S * d);
    float inv_sqrt_d = 1.0f / sqrtf((float)d);
    
    extern __shared__ char shared_mem[];
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)shared_mem;
    float* smem_reduce = (float*)(smem_Q + d);
    
    // Load Q[qpos, :] into shared mem
    for (int di = threadIdx.x; di < d; di += blockDim.x) {
        smem_Q[di] = Q[bh_off + qpos * d + di];
    }
    __syncthreads();
    
    // Compute D[qpos] = sum_d(dO[qpos, d] * O[qpos, d])
    float d_local = 0.0f;
    for (int di = threadIdx.x; di < d; di += blockDim.x) {
        float do_v = bf16_to_f32(dO[bh_off + qpos * d + di]);
        float o_v = bf16_to_f32(O[bh_off + qpos * d + di]);
        d_local += do_v * o_v;
    }
    
    smem_reduce[threadIdx.x] = d_local;
    __syncthreads();
    
    // Reduce D
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem_reduce[threadIdx.x] += smem_reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        D_out[b * H * S + h * S + qpos] = smem_reduce[0];
    }
    __syncthreads();
    
    // Strategy: two-pass softmax using registers for grid-stride
    // Pass 1: find max
    float local_max = -1e20f;
    for (int kpos = threadIdx.x; kpos < S; kpos += blockDim.x) {
        float score = 0.0f;
        for (int di = 0; di < d; ++di) {
            score += bf16_to_f32(smem_Q[di]) * bf16_to_f32(K[bh_off + kpos * d + di]);
        }
        score *= inv_sqrt_d;
        if (score > local_max) local_max = score;
    }
    smem_reduce[threadIdx.x] = local_max;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem_reduce[threadIdx.x] = fmaxf(smem_reduce[threadIdx.x], smem_reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float global_max = smem_reduce[0];
    __syncthreads();
    
    // Pass 2: compute exp and sum
    float local_sum = 0.0f;
    for (int kpos = threadIdx.x; kpos < S; kpos += blockDim.x) {
        float score = 0.0f;
        for (int di = 0; di < d; ++di) {
            score += bf16_to_f32(smem_Q[di]) * bf16_to_f32(K[bh_off + kpos * d + di]);
        }
        score *= inv_sqrt_d;
        float exp_val = expf(score - global_max);
        P_out[(uint64_t)b * H * S * S + h * S * S + qpos * S + kpos] = exp_val;
        local_sum += exp_val;
    }
    smem_reduce[threadIdx.x] = local_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem_reduce[threadIdx.x] += smem_reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float global_sum = smem_reduce[0];
    float inv_sum = 1.0f / global_sum;
    
    // Pass 3: normalize
    for (int kpos = threadIdx.x; kpos < S; kpos += blockDim.x) {
        P_out[(uint64_t)b * H * S * S + h * S * S + qpos * S + kpos] *= inv_sum;
    }
}

// ============================================================
// Compute dP = dO @ V^T for given (b,h,qpos) -> output dP[qpos, :]
// ============================================================
__global__ void compute_dP_kernel(
    const __nv_bfloat16* dO,   // [B, H, S, d]
    const __nv_bfloat16* V,    // [B, H, S, d]
    float* dP_out,             // [B, H, S, S]
    int B, int H, int S, int d)
{
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int qpos = blockIdx.y;
    
    if (b >= B || h >= H || qpos >= S) return;
    
    int bh_off = (int)((uint64_t)(b * H + h) * S * d);
    
    extern __shared__ char shared_mem[];
    __nv_bfloat16* smem_dO = (__nv_bfloat16*)shared_mem;
    
    // Load dO[qpos, :]
    for (int di = threadIdx.x; di < d; di += blockDim.x) {
        smem_dO[di] = dO[bh_off + qpos * d + di];
    }
    __syncthreads();
    
    // Grid-stride: each thread handles one or more kpos values
    for (int kpos = threadIdx.x; kpos < S; kpos += blockDim.x) {
        float acc = 0.0f;
        for (int di = 0; di < d; ++di) {
            acc += bf16_to_f32(smem_dO[di]) * bf16_to_f32(V[bh_off + kpos * d + di]);
        }
        dP_out[(uint64_t)b * H * S * S + h * S * S + qpos * S + kpos] = acc;
    }
}

// ============================================================
// Compute dS = P * (dP - D), element-wise
// ============================================================
__global__ void compute_dS_kernel(
    const float* P,      // [B, H, S, S]
    const float* dP,     // [B, H, S, S]
    const float* D,      // [B, H, S]
    float* dS_out,       // [B, H, S, S]
    int B, int H, int S)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)B * H * S * S;
    
    if (tid >= total) return;
    
    int rest = tid / S;
    int kpos = tid % S;
    int qpos = rest % S;
    int bh = rest / S;
    int b = bh / H;
    int h = bh % H;
    
    float p_val = P[tid];
    float dp_val = dP[tid];
    float d_val = D[b * H * S + h * S + qpos];
    
    dS_out[tid] = p_val * (dp_val - d_val);
}

// ============================================================
// Compute dQ = dS @ K, grid-stride over (b,h,qpos,dpos)
// ============================================================
__global__ void compute_dQ_kernel(
    const float* dS,              // [B, H, S, S]
    const __nv_bfloat16* K,       // [B, H, S, d]
    __nv_bfloat16* dQ_out,        // [B, H, S, d]
    int B, int H, int S, int d)
{
    uint64_t tidx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)B * H * S * d;
    
    if (tidx >= total) return;
    
    int dpos = tidx % d;
    tidx /= d;
    int qpos = tidx % S;
    tidx /= S;
    int bh = tidx;
    int b = bh / H;
    int h = bh % H;
    
    uint64_t bh_S_off = (uint64_t)(b * H + h) * S * S;
    uint64_t bh_off = (uint64_t)(b * H + h) * S * d;
    
    float acc = 0.0f;
    for (int kpos = 0; kpos < S; ++kpos) {
        float ds_val = dS[bh_S_off + qpos * S + kpos];
        float k_val = bf16_to_f32(K[bh_off + kpos * d + dpos]);
        acc += ds_val * k_val;
    }
    
    dQ_out[bh_off + qpos * d + dpos] = f32_to_bf16(acc);
}

// ============================================================
// Compute dK = dS^T @ Q
// dK[b,h,kpos,dpos] = sum_qpos dS[b,h,qpos,kpos] * Q[b,h,qpos,dpos]
// ============================================================
__global__ void compute_dK_kernel(
    const float* dS,              // [B, H, S, S]
    const __nv_bfloat16* Q,       // [B, H, S, d]
    __nv_bfloat16* dK_out,        // [B, H, S, d]
    int B, int H, int S, int d)
{
    uint64_t tidx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)B * H * S * d;
    
    if (tidx >= total) return;
    
    int dpos = tidx % d;
    tidx /= d;
    int kpos = tidx % S;
    tidx /= S;
    int bh = tidx;
    int b = bh / H;
    int h = bh % H;
    
    uint64_t bh_S_off = (uint64_t)(b * H + h) * S * S;
    uint64_t bh_off = (uint64_t)(b * H + h) * S * d;
    
    float acc = 0.0f;
    for (int qpos = 0; qpos < S; ++qpos) {
        float ds_val = dS[bh_S_off + qpos * S + kpos];
        float q_val = bf16_to_f32(Q[bh_off + qpos * d + dpos]);
        acc += ds_val * q_val;
    }
    
    dK_out[bh_off + kpos * d + dpos] = f32_to_bf16(acc);
}

// ============================================================
// Compute dV = P^T @ dO
// dV[b,h,kpos,dpos] = sum_qpos P[b,h,qpos,kpos] * dO[b,h,qpos,dpos]
// ============================================================
__global__ void compute_dV_kernel(
    const float* P,               // [B, H, S, S]
    const __nv_bfloat16* dO,      // [B, H, S, d]
    __nv_bfloat16* dV_out,        // [B, H, S, d]
    int B, int H, int S, int d)
{
    uint64_t tidx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)B * H * S * d;
    
    if (tidx >= total) return;
    
    int dpos = tidx % d;
    tidx /= d;
    int kpos = tidx % S;
    tidx /= S;
    int bh = tidx;
    int b = bh / H;
    int h = bh % H;
    
    uint64_t bh_S_off = (uint64_t)(b * H + h) * S * S;
    uint64_t bh_off = (uint64_t)(b * H + h) * S * d;
    
    float acc = 0.0f;
    for (int qpos = 0; qpos < S; ++qpos) {
        float p_val = P[bh_S_off + qpos * S + kpos];
        float do_val = bf16_to_f32(dO[bh_off + qpos * d + dpos]);
        acc += p_val * do_val;
    }
    
    dV_out[bh_off + kpos * d + dpos] = f32_to_bf16(acc);
}

void run(
    tvm::ffi::TensorView Q,
    tvm::ffi::TensorView K,
    tvm::ffi::TensorView V,
    tvm::ffi::TensorView O,
    tvm::ffi::TensorView dO,
    tvm::ffi::TensorView L,
    tvm::ffi::TensorView dQ,
    tvm::ffi::TensorView dK,
    tvm::ffi::TensorView dV)
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int BH = (int)(B * H);
    int S_int = (int)S;
    int d_int = (int)d;
    int dS_threads = 256;
    
    // Allocate intermediate buffers
    size_t BS2_size = (size_t)B * H * S * S * sizeof(float);
    size_t BS_size = (size_t)B * H * S * sizeof(float);
    
    float* d_P = nullptr;
    float* d_dP = nullptr;
    float* d_dS = nullptr;
    float* d_D = nullptr;
    
    CUDA_CHECK(cudaMallocAsync(&d_P, BS2_size, stream));
    CUDA_CHECK(cudaMallocAsync(&d_dP, BS2_size, stream));
    CUDA_CHECK(cudaMallocAsync(&d_dS, BS2_size, stream));
    CUDA_CHECK(cudaMallocAsync(&d_D, BS_size, stream));
    
    // Shared memory for P+D kernel: d BF16 for Q + 256 floats for reduction
    size_t smem_P_D = sizeof(__nv_bfloat16) * d_int + sizeof(float) * dS_threads;
    
    // Kernel 1: Compute P and D
    dim3 grid_P(BH, S_int);
    dim3 block_P(dS_threads);
    compute_P_and_D_kernel<<<grid_P, block_P, smem_P_D, stream>>>(
        Q_ptr, K_ptr, O_ptr, dO_ptr, d_P, d_D, B, H, S_int, d_int);
    CUDA_CHECK(cudaGetLastError());
    
    // Kernel 2: Compute dP = dO @ V^T
    size_t smem_dP = sizeof(__nv_bfloat16) * d_int;
    dim3 grid_dP(BH, S_int);
    dim3 block_dP(dS_threads);
    compute_dP_kernel<<<grid_dP, block_dP, smem_dP, stream>>>(
        dO_ptr, V_ptr, d_dP, B, H, S_int, d_int);
    CUDA_CHECK(cudaGetLastError());
    
    // Kernel 3: Compute dS = P * (dP - D)
    uint64_t total_elements = (uint64_t)B * H * S * S;
    int blocks_dS = (int)((total_elements + dS_threads - 1) / dS_threads);
    compute_dS_kernel<<<blocks_dS, dS_threads, 0, stream>>>(
        d_P, d_dP, d_D, d_dS, B, H, S_int);
    CUDA_CHECK(cudaGetLastError());
    
    // Kernel 4-6: dQ, dK, dV
    uint64_t total_BHSD = (uint64_t)B * H * S * d;
    int blocks_out = (int)((total_BHSD + dS_threads - 1) / dS_threads);
    
    compute_dQ_kernel<<<blocks_out, dS_threads, 0, stream>>>(
        d_dS, K_ptr, dQ_ptr, B, H, S_int, d_int);
    CUDA_CHECK(cudaGetLastError());
    
    compute_dK_kernel<<<blocks_out, dS_threads, 0, stream>>>(
        d_dS, Q_ptr, dK_ptr, B, H, S_int, d_int);
    CUDA_CHECK(cudaGetLastError());
    
    compute_dV_kernel<<<blocks_out, dS_threads, 0, stream>>>(
        d_P, dO_ptr, dV_ptr, B, H, S_int, d_int);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFreeAsync(d_P, stream));
    CUDA_CHECK(cudaFreeAsync(d_dP, stream));
    CUDA_CHECK(cudaFreeAsync(d_dS, stream));
    CUDA_CHECK(cudaFreeAsync(d_D, stream));
}

} // namespace mha_bwd_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);