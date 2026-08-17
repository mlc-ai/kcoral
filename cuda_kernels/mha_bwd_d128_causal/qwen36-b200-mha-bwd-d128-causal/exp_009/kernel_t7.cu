#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cmath>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

static constexpr int KERNEL_BLOCK_M = 64;
static constexpr int KERNEL_BLOCK_N = 64;
static constexpr int KERNEL_NUM_THREADS = 128;

// ============================================================
// Device Helper Functions
// ============================================================

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t parity) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(parity));
}

__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* desc, uint64_t* mbar, 
                                                void* smem_dst, int32_t coord0, int32_t coord1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem_dst)),
           "l"((uint64_t)desc),
           "r"((uint32_t)__cvta_generic_to_shared(mbar)),
           "r"(coord0), "r"(coord1) : "memory");
}

// ============================================================
// MHA Backward Kernel - No Large Local Arrays
// ============================================================

namespace mha_bwd_blackwell {

template<int BM, int BN, int HD>
__global__ void mha_bwd_kernel_sm100(
    const __grid_constant__ CUtensorMap* const tma_Q,
    const __grid_constant__ CUtensorMap* const tma_K,
    const __grid_constant__ CUtensorMap* const tma_V,
    const __grid_constant__ CUtensorMap* const tma_dO,
    __nv_bfloat16* __restrict__ dQ_out,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    const float* __restrict__ LSE_in,
    const float* __restrict__ D_precomp,
    int B, int H, int S, int D,
    float attn_scale)
{
    extern __shared__ char smem[];

    uint32_t tid = threadIdx.x;

    // Shared memory layout
    uint32_t off = 0;
    uint64_t* bar_tma = reinterpret_cast<uint64_t*>(smem + off); off += 8;
    
    off = (off + 127) & ~127U;
    __nv_bfloat16* smem_Q = reinterpret_cast<__nv_bfloat16*>(smem + off);
    off += BM * HD * sizeof(__nv_bfloat16);
    
    off = (off + 127) & ~127U;
    __nv_bfloat16* smem_K = reinterpret_cast<__nv_bfloat16*>(smem + off);
    off += BN * HD * sizeof(__nv_bfloat16);
    
    off = (off + 127) & ~127U;
    __nv_bfloat16* smem_V = reinterpret_cast<__nv_bfloat16*>(smem + off);
    off += BN * HD * sizeof(__nv_bfloat16);
    
    off = (off + 127) & ~127U;
    __nv_bfloat16* smem_dO = reinterpret_cast<__nv_bfloat16*>(smem + off);
    off += BN * HD * sizeof(__nv_bfloat16);

    off = (off + 127) & ~127U;
    __nv_bfloat16* smem_dQ_epilogue = reinterpret_cast<__nv_bfloat16*>(smem + off);
    off += BM * HD * sizeof(__nv_bfloat16);
    
    off = (off + 63) & ~63U;
    float* shared_LSE = reinterpret_cast<float*>(smem + off);
    off += BM * sizeof(float);

    if (tid == 0) {
        init_smem_barrier_fn(bar_tma, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    uint32_t bh = blockIdx.x;
    uint32_t h = bh % H;
    uint32_t b = bh / H;

    prefetch_tma_descriptor_fn(tma_Q + bh);
    prefetch_tma_descriptor_fn(tma_K + bh);
    prefetch_tma_descriptor_fn(tma_V + bh);
    prefetch_tma_descriptor_fn(tma_dO + bh);

    int64_t seq_feat = S * HD;
    float* f_dQ_base = reinterpret_cast<float*>(dQ_out + static_cast<int64_t>(b) * H * seq_feat + h * seq_feat);
    float* f_dK_base = reinterpret_cast<float*>(dK_out + static_cast<int64_t>(b) * H * seq_feat + h * seq_feat);
    float* f_dV_base = reinterpret_cast<float*>(dV_out + static_cast<int64_t>(b) * H * seq_feat + h * seq_feat);

    uint32_t num_q_blks = (S + BM - 1) / BM;
    uint32_t num_kv_blks = (S + BN - 1) / BN;

    for (uint32_t q_blk = 0; q_blk < num_q_blks; ++q_blk) {
        int q_start = static_cast<int>(q_blk) * BM;
        int q_end = min(q_start + BM, S);
        int cur_bm = q_end - q_start;

        // Zero dQ epilogue buffer
        for (int64_t idx = tid; idx < static_cast<int64_t>(cur_bm) * HD; idx += KERNEL_NUM_THREADS) {
            smem_dQ_epilogue[idx] = __float2bfloat16(0.0f);
        }
        
        tma_load_2d_fn(tma_Q + bh, bar_tma, smem_Q, q_start, 0);
        mbarrier_wait_fn(bar_tma, 0);
        __syncthreads();
        
        const float* lse_bh = LSE_in + b * H * S + h * S;
        const float* D_bh = D_precomp + b * H * S + h * S;
        for (int i = tid; i < cur_bm; i += KERNEL_NUM_THREADS) {
            shared_LSE[i] = lse_bh[q_start + i];
        }
        __syncthreads();

        for (uint32_t kv_blk = 0; kv_blk < num_kv_blks; ++kv_blk) {
            int kv_start = static_cast<int>(kv_blk) * BN;
            int kv_end = min(kv_start + BN, S);
            int cur_bn = kv_end - kv_start;

            if (q_start + cur_bm <= kv_start) continue;

            tma_load_2d_fn(tma_K + bh, bar_tma, smem_K, kv_start, 0);
            tma_load_2d_fn(tma_V + bh, bar_tma, smem_V, kv_start, 0);
            tma_load_2d_fn(tma_dO + bh, bar_tma, smem_dO, kv_start, 0);
            mbarrier_wait_fn(bar_tma, 0);
            __syncthreads();
            
            // STRATEGY: Split work between two passes to avoid large local arrays
            // Pass 1 (threads 0..min(cur_bn-1, NUM_THREADS)): compute dV and dK contributions
            // Pass 2 (threads 0..min(cur_bm-1, NUM_THREADS)): compute dQ contributions
            
            // Pass 1: Each thread handles one KV position, accumulates dV/f and dK/f across all Q positions
            if (tid < cur_bn) {
                int kv_local = tid;
                int kv_global = kv_start + kv_local;
                
                // Accumulate per-feature to reduce register pressure
                // Process features in groups to allow atomicAdd batching
                for (int f_group = 0; f_group < HD; f_group += 2) {
                    float dv_f[2] = {};
                    float dk_f[2] = {};
                    
                    for (int fg = 0; fg < 2 && f_group + fg < HD; ++fg) {
                        int f = f_group + fg;
                        float kf = __bfloat162float(smem_K[kv_local * HD + f]);
                        
                        for (int q_local = 0; q_local < cur_bm; ++q_local) {
                            int q_global = q_start + q_local;
                            if (q_global < kv_global) continue;
                            
                            // Compute S[q,kv] - full dot product needed for attention score
                            float s_dot = 0.0f;
                            for (int ff = 0; ff < HD; ++ff) {
                                s_dot += __bfloat162float(smem_Q[q_local * HD + ff]) *
                                         __bfloat162float(smem_K[kv_local * HD + ff]);
                            }
                            float p_val = expf(s_dot * attn_scale - lse_bh[q_global]);
                            
                            // dV[f] += P * dO[q,f]
                            dv_f[fg] += p_val * __bfloat162float(smem_dO[q_local * HD + f]);
                            
                            // Compute E[q,kv] = sum_f dO[q,f]*V[kv,f] for ds
                            float e_dot = 0.0f;
                            for (int ff = 0; ff < HD; ++ff) {
                                e_dot += __bfloat162float(smem_dO[q_local * HD + ff]) *
                                         __bfloat162float(smem_V[kv_local * HD + ff]);
                            }
                            float ds_val = p_val * (e_dot - D_bh[q_global]);
                            dk_f[fg] += ds_val * __bfloat162float(smem_Q[q_local * HD + f]);
                        }
                    }
                    
                    atomicAdd(&f_dV_base[kv_global * HD + f_group], dv_f[0]);
                    if (f_group + 1 < HD)
                        atomicAdd(&f_dV_base[kv_global * HD + f_group + 1], dv_f[1]);
                    atomicAdd(&f_dK_base[kv_global * HD + f_group], dk_f[0]);
                    if (f_group + 1 < HD)
                        atomicAdd(&f_dK_base[kv_global * HD + f_group + 1], dk_f[1]);
                }
            }
            __syncthreads();
            
            // Pass 2: Each thread handles one Q position, computes dQ contribution across all KV positions
            if (tid < cur_bm) {
                int q_local = tid;
                int q_global = q_start + q_local;
                
                for (int f_group = 0; f_group < HD; f_group += 2) {
                    float dq_f[2] = {};
                    
                    for (int fg = 0; fg < 2 && f_group + fg < HD; ++fg) {
                        int f = f_group + fg;
                        
                        for (int kv_local = 0; kv_local < cur_bn; ++kv_local) {
                            int kv_global = kv_start + kv_local;
                            if (q_global < kv_global) continue;
                            
                            float s_dot = 0.0f;
                            for (int ff = 0; ff < HD; ++ff) {
                                s_dot += __bfloat162float(smem_Q[q_local * HD + ff]) *
                                         __bfloat162float(smem_K[kv_local * HD + ff]);
                            }
                            float p_val = expf(s_dot * attn_scale - lse_bh[q_global]);
                            
                            float e_dot = 0.0f;
                            for (int ff = 0; ff < HD; ++ff) {
                                e_dot += __bfloat162float(smem_dO[q_local * HD + ff]) *
                                         __bfloat162float(smem_V[kv_local * HD + ff]);
                            }
                            float ds_val = p_val * (e_dot - D_bh[q_global]);
                            dq_f[fg] += ds_val * __bfloat162float(smem_K[kv_local * HD + f]);
                        }
                    }
                    
                    atomicAdd(&reinterpret_cast<float*>(smem_dQ_epilogue)[q_local * HD + f_group], dq_f[0]);
                    if (f_group + 1 < HD)
                        atomicAdd(&reinterpret_cast<float*>(smem_dQ_epilogue)[q_local * HD + f_group + 1], dq_f[1]);
                }
            }
            __syncthreads();
        }
        
        // Store accumulated dQ to global after all kv blocks for this q_blk
        for (int64_t idx = tid; idx < static_cast<int64_t>(cur_bm) * HD; idx += KERNEL_NUM_THREADS) {
            float dq_val = __bfloat162float(smem_dQ_epilogue[idx]);
            atomicAdd(&f_dQ_base[q_start * HD + idx], dq_val);
        }
        __syncthreads();
    }
}

}  // namespace mha_bwd_blackwell

static CUresult encode_tma_2d_bf16(CUtensorMap* tensorMap, void* globalAddr,
                                    uint64_t gmem_inner, uint64_t gmem_outer,
                                    uint32_t box_inner, uint32_t box_outer) 
{
    cuuint64_t globalDim[2] = {gmem_inner, gmem_outer};
    cuuint64_t globalStrides[2] = {sizeof(__nv_bfloat16), gmem_inner * sizeof(__nv_bfloat16)};
    cuuint32_t boxDim[2] = {box_inner, box_outer};
    cuuint32_t elemStrides[2] = {1, 1};
    
    return cuTensorMapEncodeTiled(tensorMap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        globalAddr, globalDim, globalStrides, boxDim, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

namespace mha_bwd_run {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) 
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0), H = Q.size(1), S = Q.size(2), D = Q.size(3);
    float attn_scale = 1.0f / std::sqrt(static_cast<float>(D));
    int64_t num_bh = B * H;
    
    auto* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    auto* dO_ptr = static_cast<__nv_bfloat16*>(dO.data_ptr());
    float* d_D_host = new float[B * H * S]();
    int64_t bh_stride = S * D;
    
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t hh = 0; hh < H; ++hh) {
            int64_t offset = b * H * bh_stride + hh * bh_stride;
            for (int64_t q = 0; q < S; ++q) {
                float sum = 0.0f;
                for (int64_t f = 0; f < D; ++f)
                    sum += __bfloat162float(dO_ptr[offset + q * D + f]) *
                           __bfloat162float(O_ptr[offset + q * D + f]);
                d_D_host[b * H * S + hh * S + q] = sum;
            }
        }
    }
    
    float* d_D_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&d_D_dev, B * H * S * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_D_dev, d_D_host, B * H * S * sizeof(float), cudaMemcpyHostToDevice));
    delete[] d_D_host;
    
    size_t desc_size = sizeof(CUtensorMap);
    CUtensorMap* d_tma[4] = {};
    for (int i = 0; i < 4; ++i) CUDA_CHECK(cudaMalloc(&d_tma[i], desc_size * num_bh));
    
    CUtensorMap* h_desc_Q = new CUtensorMap[num_bh]();
    CUtensorMap* h_desc_K = new CUtensorMap[num_bh]();
    CUtensorMap* h_desc_V = new CUtensorMap[num_bh]();
    CUtensorMap* h_desc_dO = new CUtensorMap[num_bh]();
    
    __nv_bfloat16* bases[4] = {
        static_cast<__nv_bfloat16*>(Q.data_ptr()),
        static_cast<__nv_bfloat16*>(K.data_ptr()),
        static_cast<__nv_bfloat16*>(V.data_ptr()),
        static_cast<__nv_bfloat16*>(dO.data_ptr())
    };
    int boxes[4] = {KERNEL_BLOCK_M, KERNEL_BLOCK_N, KERNEL_BLOCK_N, KERNEL_BLOCK_N};
    
    for (int64_t bh = 0; bh < num_bh; ++bh) {
        int64_t b = bh / H, hh = bh % H;
        int64_t offset = b * H * bh_stride + hh * bh_stride;
        encode_tma_2d_bf16(h_desc_Q + bh, bases[0] + offset, D, S, D, boxes[0]);
        encode_tma_2d_bf16(h_desc_K + bh, bases[1] + offset, D, S, D, boxes[1]);
        encode_tma_2d_bf16(h_desc_V + bh, bases[2] + offset, D, S, D, boxes[2]);
        encode_tma_2d_bf16(h_desc_dO + bh, bases[3] + offset, D, S, D, boxes[3]);
    }
    
    CUDA_CHECK(cudaMemcpy(d_tma[0], h_desc_Q, desc_size * num_bh, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tma[1], h_desc_K, desc_size * num_bh, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tma[2], h_desc_V, desc_size * num_bh, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tma[3], h_desc_dO, desc_size * num_bh, cudaMemcpyHostToDevice));
    
    delete[] h_desc_Q;
    delete[] h_desc_K;
    delete[] h_desc_V;
    delete[] h_desc_dO;
    
    size_t smem_size = 8 + 64;
    smem_size = (smem_size + 127) & ~127ULL;
    smem_size += KERNEL_BLOCK_M * D * sizeof(__nv_bfloat16);
    smem_size = (smem_size + 127) & ~127ULL;
    smem_size += KERNEL_BLOCK_N * D * sizeof(__nv_bfloat16);
    smem_size = (smem_size + 127) & ~127ULL;
    smem_size += KERNEL_BLOCK_N * D * sizeof(__nv_bfloat16);
    smem_size = (smem_size + 127) & ~127ULL;
    smem_size += KERNEL_BLOCK_N * D * sizeof(__nv_bfloat16);
    smem_size = (smem_size + 127) & ~127ULL;
    smem_size += KERNEL_BLOCK_M * D * sizeof(__nv_bfloat16);
    smem_size = (smem_size + 63) & ~63ULL;
    smem_size += KERNEL_BLOCK_M * sizeof(float);
    
    dim3 grid(num_bh), block(KERNEL_NUM_THREADS);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid; cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem_size;
    cfg.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = 2;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    
    auto fn = mha_bwd_blackwell::mha_bwd_kernel_sm100<KERNEL_BLOCK_M, KERNEL_BLOCK_N, 128>;
    cudaLaunchKernelEx(&cfg, fn,
        d_tma[0], d_tma[1], d_tma[2], d_tma[3],
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        d_D_dev, (int)B, (int)H, (int)S, (int)D, attn_scale);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(cfg.stream));
    for (int i = 0; i < 4; ++i) CUDA_CHECK(cudaFree(d_tma[i]));
    CUDA_CHECK(cudaFree(d_D_dev));
}

}  // namespace mha_bwd_run

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_bwd_run::run);