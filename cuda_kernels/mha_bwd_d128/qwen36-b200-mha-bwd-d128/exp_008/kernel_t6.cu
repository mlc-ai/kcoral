#include <cuda_runtime.h>
#include <device_launch_parameters.h>
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

__forceinline__ __device__ static float bf16_to_f32(uint16_t x) {
    uint32_t y = (uint32_t)x << 16;
    return *(float*)&y;
}

__forceinline__ __device__ static uint16_t f32_to_bf16(float x) {
    uint32_t i = ((*(uint32_t*)&x) + 0x7FFFu) >> 16;
    return (uint16_t)i;
}

// Kernel to compute P = softmax(Q @ K^T / sqrt(d)) and D[qpos] = sum_d(dO * O)
// Each block handles one (b, h) pair. Threads split work over qpos via grid-stride.
__global__ void compute_P_and_D_kernel(
    const uint16_t* Q, const uint16_t* K,
    const uint16_t* O, const uint16_t* dO,
    float* P_out, float* D_out,
    int B, int H, int S, int d)
{
    int bh = blockIdx.x;
    if (bh >= B * H) return;
    int b = bh / H, h = bh % H;
    uint64_t bho = (uint64_t)bh * S * d;
    uint64_t bo = (uint64_t)bh * S * S;
    float inv_sqrt_d = 1.0f / sqrtf((float)d);

    extern __shared__ char smem_raw[];
    float* Qf = (float*)smem_raw;
    float* tmp = Qf + d;

    // Each thread loads its own row of Q into Qf[threadIdx.x][di]
    // But we only need one Q row at a time per inner loop iteration
    // So load Q[qpos_i] on demand inside the loop

    int tid = threadIdx.x;
    
    // Compute D values first
    int qstart = blockIdx.y * blockDim.x + tid;
    int qstride = blockDim.x * gridDim.y;
    for (int qpos = qstart; qpos < S; qpos += qstride) {
        float dl = 0.f;
        uint64_t qq = bho + (uint64_t)qpos * d;
        for (int dd = tid; dd < d; dd += blockDim.x)
            dl += bf16_to_f32(dO[qq + dd]) * bf16_to_f32(O[qq + dd]);
        tmp[tid] = dl;
        __syncthreads();
        for (int st = blockDim.x/2; st > 0; st >>= 1) {
            if (tid < st) tmp[tid] += tmp[tid + st];
            __syncthreads();
        }
        if (!tid) D_out[b * H * S + h * S + qpos] = tmp[0];
        __syncthreads();
    }

    // Now compute P = softmax(QK^T/sqrt(d))
    qstart = blockIdx.y * blockDim.x + tid;
    for (int qpos = qstart; qpos < S; qpos += qstride) {
        uint64_t qq = bho + (uint64_t)qpos * d;
        
        // Load Q[qpos,:] into shared memory
        for (int dd = tid; dd < d; dd += blockDim.x)
            Qf[dd] = bf16_to_f32(Q[qq + dd]);
        __syncthreads();
        
        // Pass 1: compute scores and find max
        float lmax = -1e20f;
        for (int kpos = tid; kpos < S; kpos += blockDim.x) {
            float sc = 0.f;
            uint64_t kk = bho + (uint64_t)kpos * d;
            for (int dd = 0; dd < d; ++dd)
                sc += Qf[dd] * bf16_to_f32(K[kk + dd]);
            sc *= inv_sqrt_d;
            tmp[kpos] = sc;
            if (sc > lmax) lmax = sc;
        }
        // Reduce max
        for (int st = blockDim.x/2; st > 0; st >>= 1) {
            if (tid < st && tmp[tid + st] > tmp[tid]) tmp[tid] = tmp[tid + st];
            __syncthreads();
        }
        float gmax = tmp[0];
        
        // Pass 2: compute exp(score - max)
        for (int kpos = tid; kpos < S; kpos += blockDim.x) {
            float v = expf(tmp[kpos] - gmax);
            P_out[bo + qpos * S + kpos] = v;
            tmp[kpos] = v;
        }
        __syncthreads();
        
        // Sum
        float lsum = 0.f;
        for (int i = tid; i < S; i += blockDim.x)
            lsum += tmp[i];
        tmp[tid] = lsum;
        __syncthreads();
        for (int st = blockDim.x/2; st > 0; st >>= 1) {
            if (tid < st) tmp[tid] += tmp[tid + st];
            __syncthreads();
        }
        float inv_sum = 1.f / tmp[0];
        
        // Normalize
        for (int kpos = tid; kpos < S; kpos += blockDim.x)
            P_out[bo + qpos * S + kpos] *= inv_sum;
    }
}

// Kernel to compute dP = dO @ V^T
__global__ void compute_dP_kernel(
    const uint16_t* dO, const uint16_t* V,
    float* dP_out, int B, int H, int S, int d)
{
    int bh = blockIdx.x;
    if (bh >= B * H) return;
    uint64_t bho = (uint64_t)bh * S * d;
    uint64_t bo = (uint64_t)bh * S * S;

    extern __shared__ char smem_raw[];
    float* dOf = (float*)smem_raw;

    int tid = threadIdx.x;
    int qstart = blockIdx.y * blockDim.x + tid;
    int qstride = blockDim.x * gridDim.y;

    for (int qpos = qstart; qpos < S; qpos += qstride) {
        uint64_t dqo = bho + (uint64_t)qpos * d;
        
        // Load dO[qpos,:]
        for (int dd = tid; dd < d; dd += blockDim.x)
            dOf[dd] = bf16_to_f32(dO[dqo + dd]);
        __syncthreads();
        
        // Compute dot products with each V[kpos,:]
        for (int kpos = tid; kpos < S; kpos += blockDim.x) {
            float acc = 0.f;
            uint64_t voff = bho + (uint64_t)kpos * d;
            for (int dd = 0; dd < d; ++dd)
                acc += dOf[dd] * bf16_to_f32(V[voff + dd]);
            dP_out[bo + qpos * S + kpos] = acc;
        }
    }
}

// Element-wise: dS = P * (dP - D)
__global__ void compute_dS_kernel(
    const float* P, const float* dP, const float* D,
    float* dS_out, int B, int H, int S)
{
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)B * H * S * S;
    if (tid >= total) return;

    uint64_t rest = tid / S;
    int qpos = rest % S;
    int bh = rest / S;
    int b = bh / H, h = bh % H;

    dS_out[tid] = P[tid] * (dP[tid] - D[b * H * S + h * S + qpos]);
}

// dQ = dS @ K
__global__ void compute_dQ_kernel(
    const float* dS, const uint16_t* K,
    uint16_t* dQ_out, int B, int H, int S, int d)
{
    uint64_t tidx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)B * H * S * d;
    if (tidx >= total) return;

    int dpos = tidx % d;
    tidx /= d;
    int qpos = tidx % S;
    tidx /= S;
    int bh = tidx;
    uint64_t bhSO = (uint64_t)bh * S * S;
    uint64_t bhKD = (uint64_t)bh * S * d;

    float acc = 0.f;
    for (int kpos = 0; kpos < S; ++kpos)
        acc += dS[bhSO + qpos * S + kpos] * bf16_to_f32(K[bhKD + kpos * d + dpos]);

    dQ_out[bhKD + qpos * d + dpos] = f32_to_bf16(acc);
}

// dK = dS^T @ Q
__global__ void compute_dK_kernel(
    const float* dS, const uint16_t* Q,
    uint16_t* dK_out, int B, int H, int S, int d)
{
    uint64_t tidx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)B * H * S * d;
    if (tidx >= total) return;

    int dpos = tidx % d;
    tidx /= d;
    int kpos = tidx % S;
    tidx /= S;
    int bh = tidx;
    uint64_t bhSO = (uint64_t)bh * S * S;
    uint64_t bhKD = (uint64_t)bh * S * d;

    float acc = 0.f;
    for (int qpos = 0; qpos < S; ++qpos)
        acc += dS[bhSO + qpos * S + kpos] * bf16_to_f32(Q[bhKD + qpos * d + dpos]);

    dK_out[bhKD + kpos * d + dpos] = f32_to_bf16(acc);
}

// dV = P^T @ dO
__global__ void compute_dV_kernel(
    const float* P, const uint16_t* dO,
    uint16_t* dV_out, int B, int H, int S, int d)
{
    uint64_t tidx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)B * H * S * d;
    if (tidx >= total) return;

    int dpos = tidx % d;
    tidx /= d;
    int kpos = tidx % S;
    tidx /= S;
    int bh = tidx;
    uint64_t bhSO = (uint64_t)bh * S * S;
    uint64_t bhKD = (uint64_t)bh * S * d;

    float acc = 0.f;
    for (int qpos = 0; qpos < S; ++qpos)
        acc += P[bhSO + qpos * S + kpos] * bf16_to_f32(dO[bhKD + qpos * d + dpos]);

    dV_out[bhKD + kpos * d + dpos] = f32_to_bf16(acc);
}

void run(
    tvm::ffi::TensorView Q, tvm::ffi::TensorView K,
    tvm::ffi::TensorView V, tvm::ffi::TensorView O,
    tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
    tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK,
    tvm::ffi::TensorView dV)
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0), H = Q.size(1), S = Q.size(2), d = Q.size(3);
    
    const uint16_t* Qp = static_cast<const uint16_t*>(Q.data_ptr());
    const uint16_t* Kp = static_cast<const uint16_t*>(K.data_ptr());
    const uint16_t* Vp = static_cast<const uint16_t*>(V.data_ptr());
    const uint16_t* Op = static_cast<const uint16_t*>(O.data_ptr());
    const uint16_t* dOp = static_cast<const uint16_t*>(dO.data_ptr());
    
    uint16_t* dQp = static_cast<uint16_t*>(dQ.data_ptr());
    uint16_t* dKp = static_cast<uint16_t*>(dK.data_ptr());
    uint16_t* dVp = static_cast<uint16_t*>(dV.data_ptr());
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int BH = (int)(B * H), Si = (int)S, di = (int)d;
    size_t bs2 = (size_t)B * H * S * S * sizeof(float);
    size_t bs = (size_t)B * H * S * sizeof(float);
    
    float *dP, *ddP, *dds, *dD;
    CUDA_CHECK(cudaMallocAsync(&dP, bs2, stream));
    CUDA_CHECK(cudaMallocAsync(&ddP, bs2, stream));
    CUDA_CHECK(cudaMallocAsync(&dds, bs2, stream));
    CUDA_CHECK(cudaMallocAsync(&dD, bs, stream));
    
    // For P/D and dP kernels: grid = (BH, num_blocks_for_S)
    int ty = 256;
    int gy = (Si + ty - 1) / ty;
    size_t smem_sz = (size_t)di * sizeof(float) + Si * sizeof(float);
    
    dim3 gp(BH, gy), bp(ty);
    compute_P_and_D_kernel<<<gp, bp, smem_sz, stream>>>(
        Qp, Kp, Op, dOp, dP, dD, B, H, Si, di);
    CUDA_CHECK(cudaGetLastError());
    
    compute_dP_kernel<<<gp, bp, smem_sz, stream>>>(
        dOp, Vp, ddP, B, H, Si, di);
    CUDA_CHECK(cudaGetLastError());
    
    // dS kernel
    uint64_t tot_el = (uint64_t)B * H * S * S;
    int th = 256;
    int nblk_ds = (int)((tot_el + th - 1) / th);
    compute_dS_kernel<<<nblk_ds, th, 0, stream>>>(
        dP, ddP, dD, dds, B, H, Si);
    CUDA_CHECK(cudaGetLastError());
    
    // dQ, dK, dV kernels
    uint64_t tot_out = (uint64_t)B * H * S * d;
    int nblk_out = (int)((tot_out + th - 1) / th);
    
    compute_dQ_kernel<<<nblk_out, th, 0, stream>>>(
        dds, Kp, dQp, B, H, Si, di);
    CUDA_CHECK(cudaGetLastError());
    
    compute_dK_kernel<<<nblk_out, th, 0, stream>>>(
        dds, Qp, dKp, B, H, Si, di);
    CUDA_CHECK(cudaGetLastError());
    
    compute_dV_kernel<<<nblk_out, th, 0, stream>>>(
        dP, dOp, dVp, B, H, Si, di);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFreeAsync(dP, stream));
    CUDA_CHECK(cudaFreeAsync(ddP, stream));
    CUDA_CHECK(cudaFreeAsync(dds, stream));
    CUDA_CHECK(cudaFreeAsync(dD, stream));
}

} // namespace mha_bwd_impl

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_impl::run);