#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_bwd_impl {

// Helper: fast bfloat16 conversions
__forceinline__ __device__ static float bf16_to_f32(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__forceinline__ __device__ static __nv_bfloat16 f32_to_bf16(float x) {
    return __float2bfloat16(x);
}

__forceinline__ __device__ static float approx_sqrtf(float x) {
    float y;
    asm volatile("sqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// ============================================================
// Step 1: Compute P = softmax(Q @ K^T / sqrt(d)) and D = rowsum(dO * O)
// P stored as float[B*H*S*S], D stored as float[B*H*S]
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
    // Each block computes one (b, h, qpos)
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int qpos = blockIdx.y;
    
    if (b >= B || h >= H || qpos >= S) return;
    
    // Base offsets
    int bh_off = (b * H + h) * S * d;
    float inv_sqrt_d = 1.0f / approx_sqrtf((float)d);
    
    // Shared memory for Q[qpos, :] and D accumulation
    extern __shared__ char shared_mem[];
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)shared_mem;
    __nv_bfloat16* smem_K = smem_Q + d;
    float* smem_D = (float*)(smem_K + d);
    
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
    smem_D[threadIdx.x] = d_local;
    __syncthreads();
    
    // Reduce D
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem_D[threadIdx.x] += smem_D[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        D_out[b * H * S + h * S + qpos] = smem_D[0];
    }
    
    // Compute attention scores and softmax for P[qpos, :]
    float max_val = -1e20f;
    float local_score[S]; // Too large for stack? Let's use different approach.
    
    // Use private array for small S, otherwise shared memory
    // For correctness, compute scores using shared memory
    float* smem_scores = (float*)(smem_D + blockDim.x);
    
    // Load all K[:, :] tiles and compute scores
    // We need to iterate over kpos and compute Q·K^T
    // Score[qpos, kpos] = sum_d(Q[qpos, d] * K[kpos, d]) / sqrt(d)
    
    // Load K row by row and accumulate scores
    int bh_K_off = bh_off; // Same bh offset for K
    for (int kpos = threadIdx.x; kpos < S; kpos += blockDim.x) {
        float score = 0.0f;
        for (int di = 0; di < d; ++di) {
            score += bf16_to_f32(smem_Q[di]) * bf16_to_f32(K[bh_K_off + kpos * d + di]);
        }
        score *= inv_sqrt_d;
        smem_scores[kpos] = score;
    }
    __syncthreads();
    
    // Find max for numerically stable softmax
    float local_max = -1e20f;
    for (int kpos = threadIdx.x; kpos < S; kpos += blockDim.x) {
        local_max = fmaxf(local_max, smem_scores[kpos]);
    }
    smem_D[threadIdx.x] = local_max;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem_D[threadIdx.x] = fmaxf(smem_D[threadIdx.x], smem_D[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float global_max = smem_D[0];
    
    // Compute exp(score - max) and sum
    float local_sum = 0.0f;
    for (int kpos = threadIdx.x; kpos < S; kpos += blockDim.x) {
        float shifted = smem_scores[kpos] - global_max;
        float exp_val = expf(shifted);
        smem_scores[kpos] = exp_val;
        local_sum += exp_val;
    }
    smem_D[threadIdx.x] = local_sum;
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            smem_D[threadIdx.x] += smem_D[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float global_sum = smem_D[0];
    float inv_sum = 1.0f / global_sum;
    
    // Normalize and write P
    for (int kpos = threadIdx.x; kpos < S; kpos += blockDim.x) {
        P_out[b * H * S * S + h * S * S + qpos * S + kpos] = smem_scores[kpos] * inv_sum;
    }
}

// ============================================================
// Step 2: Compute dP = dO @ V^T (result as float[B,H,S,S])
// ============================================================
__global__ void compute_dP_kernel(
    const __nv_bfloat16* dO,   // [B, H, S, d]
    const __nv_bfloat16* V,    // [B, H, S, d]
    float* dP_out,             // [B, H, S, S] (dO @ V^T)
    int B, int H, int S, int d)
{
    int bh = blockIdx.x;
    int b = bh / H;
    int h = bh % H;
    int qpos = blockIdx.y;
    
    if (b >= B || h >= H || qpos >= S) return;
    
    int bh_off = (b * H + h) * S * d;
    
    extern __shared__ char shared_mem[];
    __nv_bfloat16* smem_dO = (__nv_bfloat16*)shared_mem;
    float* smem_acc = (float*)(smem_dO + d);
    
    // Load dO[qpos, :]
    for (int di = threadIdx.x; di < d; di += blockDim.x) {
        smem_dO[di] = dO[bh_off + qpos * d + di];
    }
    __syncthreads();
    
    // Compute dot product with each K row of V
    float local_acc = 0.0f;
    int kpos = threadIdx.x;
    if (kpos < S) {
        for (int di = 0; di < d; ++di) {
            local_acc += bf16_to_f32(smem_dO[di]) * bf16_to_f32(V[bh_off + kpos * d + di]);
        }
    }
    smem_acc[threadIdx.x] = local_acc;
    __syncthreads();
    
    // Each kpos is handled by its own thread, just store
    if (kpos < S) {
        dP_out[b * H * S * S + h * S * S + qpos * S + kpos] = local_acc;
    }
}

// ============================================================
// Step 3: Compute dS = P * (dP - D)
// ============================================================
__global__ void compute_dS_kernel(
    const float* P,      // [B, H, S, S]
    const float* dP,     // [B, H, S, S]
    const float* D,      // [B, H, S]
    float* dS_out,       // [B, H, S, S]
    int B, int H, int S)
{
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)B * H * S * S;
    
    if (idx >= total) return;
    
    int rest = idx / S;
    int kpos = idx % S;
    int qpos = rest % S;
    int bh = rest / S;
    int b = bh / H;
    int h = bh % H;
    
    float p_val = P[idx];
    float dp_val = dP[idx];
    float d_val = D[b * H * S + h * S + qpos];
    
    dS_out[idx] = p_val * (dp_val - d_val);
}

// ============================================================
// Step 4: Compute dQ = dS @ K
// Each thread computes dQ[b, h, qpos, dpos]
// ============================================================
__global__ void compute_dQ_kernel(
    const float* dS,              // [B, H, S, S]
    const __nv_bfloat16* K,       // [B, H, S, d]
    __nv_bfloat16* dQ_out,        // [B, H, S, d]
    int B, int H, int S, int d)
{
    // Thread maps to (b, h, qpos, dpos)
    unsigned long long tidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)B * H * S * d;
    
    if (tidx >= total) return;
    
    int dpos = tidx % d;
    tidx /= d;
    int qpos = tidx % S;
    tidx /= S;
    int bh = tidx;
    int b = bh / H;
    int h = bh % H;
    
    int bh_off = (b * H + h) * S * d;
    int bh_S_off = (b * H + h) * S * S;
    
    float acc = 0.0f;
    for (int kpos = 0; kpos < S; ++kpos) {
        float ds_val = dS[bh_S_off + qpos * S + kpos];
        float k_val = bf16_to_f32(K[bh_off + kpos * d + dpos]);
        acc += ds_val * k_val;
    }
    
    dQ_out[bh_off + qpos * d + dpos] = f32_to_bf16(acc);
}

// ============================================================
// Step 5: Compute dK = dS^T @ Q
// dK[b, h, kpos, dpos] = sum_qpos dS[b, h, qpos, kpos] * Q[b, h, qpos, dpos]
// ============================================================
__global__ void compute_dK_kernel(
    const float* dS,              // [B, H, S, S]
    const __nv_bfloat16* Q,       // [B, H, S, d]
    __nv_bfloat16* dK_out,        // [B, H, S, d]
    int B, int H, int S, int d)
{
    unsigned long long tidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)B * H * S * d;
    
    if (tidx >= total) return;
    
    int dpos = tidx % d;
    tidx /= d;
    int kpos = tidx % S;
    tidx /= S;
    int bh = tidx;
    int b = bh / H;
    int h = bh % H;
    
    int bh_off = (b * H + h) * S * d;
    int bh_S_off = (b * H + h) * S * S;
    
    float acc = 0.0f;
    for (int qpos = 0; qpos < S; ++qpos) {
        float ds_val = dS[bh_S_off + qpos * S + kpos];
        float q_val = bf16_to_f32(Q[bh_off + qpos * d + dpos]);
        acc += ds_val * q_val;
    }
    
    dK_out[bh_off + kpos * d + dpos] = f32_to_bf16(acc);
}

// ============================================================
// Step 6: Compute dV = P^T @ dO
// dV[b, h, kpos, dpos] = sum_qpos P[b, h, qpos, kpos] * dO[b, h, qpos, dpos]
// ============================================================
__global__ void compute_dV_kernel(
    const float* P,               // [B, H, S, S]
    const __nv_bfloat16* dO,      // [B, H, S, d]
    __nv_bfloat16* dV_out,        // [B, H, S, d]
    int B, int H, int S, int d)
{
    unsigned long long tidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long total = (unsigned long long)B * H * S * d;
    
    if (tidx >= total) return;
    
    int dpos = tidx % d;
    tidx /= d;
    int kpos = tidx % S;
    tidx /= S;
    int bh = tidx;
    int b = bh / H;
    int h = bh % H;
    
    int bh_off = (b * H + h) * S * d;
    int bh_S_off = (b * H + h) * S * S;
    
    float acc = 0.0f;
    for (int qpos = 0; qpos < S; ++qpos) {
        float p_val = P[bh_S_off + qpos * S + kpos];
        float do_val = bf16_to_f32(dO[bh_off + qpos * d + dpos]);
        acc += p_val * do_val;
    }
    
    dV_out[bh_off + kpos * d + dpos] = f32_to_bf16(acc);
}

// Host-side run function
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
    
    // Extract dimensions
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
    
    // Allocate intermediate buffers
    // P: [B*H*S*S] floats, dP: [B*H*S*S] floats, dS: [B*H*S*S] floats, D: [B*H*S] floats
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
    
    // Kernel configurations
    int BH = B * H;
    int dS_threads = 256;
    
    // Shared memory size for compute_P_and_D: d BF16 + d BF16 + d float + S float
    // But we need d BF16 for Q, d BF16 for K (not used actually, we read K from gmem), 
    // S float for scores, blockDim.x float for D reduction
    size_t smem_P_D = sizeof(__nv_bfloat16) * d + sizeof(float) * (S + dS_threads);
    
    // Kernel 1: Compute P and D
    dim3 grid_P(BH, S);
    dim3 block_P(dS_threads);
    compute_P_and_D_kernel<<<grid_P, block_P, smem_P_D, stream>>>(
        Q_ptr, K_ptr, O_ptr, dO_ptr, d_P, d_D, B, H, S, d);
    CUDA_CHECK(cudaGetLastError());
    
    // Kernel 2: Compute dP = dO @ V^T
    size_t smem_dP = sizeof(__nv_bfloat16) * d + sizeof(float) * S;
    dim3 grid_dP(BH, S);
    dim3 block_dP(S);  // One thread per kpos
    if (block_dP.x > 1024) block_dP.x = 1024;
    compute_dP_kernel<<<grid_dP, block_dP, smem_dP, stream>>>(
        dO_ptr, V_ptr, d_dP, B, H, S, d);
    CUDA_CHECK(cudaGetLastError());
    
    // Kernel 3: Compute dS = P * (dP - D)
    unsigned long long total_elements = (unsigned long long)B * H * S * S;
    int blocks_dS = (total_elements + dS_threads - 1) / dS_threads;
    compute_dS_kernel<<<blocks_dS, dS_threads, 0, stream>>>(
        d_P, d_dP, d_D, d_dS, B, H, S);
    CUDA_CHECK(cudaGetLastError());
    
    // Kernel 4: Compute dQ = dS @ K
    unsigned long long total_BHSD = (unsigned long long)B * H * S * d;
    int blocks_dQ = (total_BHSD + dS_threads - 1) / dS_threads;
    compute_dQ_kernel<<<blocks_dQ, dS_threads, 0, stream>>>(
        d_dS, K_ptr, dQ_ptr, B, H, S, d);
    CUDA_CHECK(cudaGetLastError());
    
    // Kernel 5: Compute dK = dS^T @ Q
    int blocks_dK = (total_BHSD + dS_threads - 1) / dS_threads;
    compute_dK_kernel<<<blocks_dK, dS_threads, 0, stream>>>(
        d_dS, Q_ptr, dK_ptr, B, H, S, d);
    CUDA_CHECK(cudaGetLastError());
    
    // Kernel 6: Compute dV = P^T @ dO
    int blocks_dV = (total_BHSD + dS_threads - 1) / dS_threads;
    compute_dV_kernel<<<blocks_dV, dS_threads, 0, stream>>>(
        d_P, dO_ptr, dV_ptr, B, H, S, d);
    CUDA_CHECK(cudaGetLastError());
    
    // Synchronize and cleanup
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFreeAsync(d_P, stream));
    CUDA_CHECK(cudaFreeAsync(d_dP, stream));
    CUDA_CHECK(cudaFreeAsync(d_dS, stream));
    CUDA_CHECK(cudaFreeAsync(d_D, stream));
}

} // namespace mha_bwd_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);