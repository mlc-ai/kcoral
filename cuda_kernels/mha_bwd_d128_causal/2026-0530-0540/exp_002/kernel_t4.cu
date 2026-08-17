#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <mma.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha_bwd {

__device__ __forceinline__ void dummy_instruction_warning_fix() {
    if (threadIdx.x == 999999) { 
        uint32_t a = 0;
        uint64_t b = 0;
        asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
            :: "r"(a), "l"(b), "r"(a), "r"(a), "r"(a));
        asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 %0, %1, %2, 1, 1;" : : "r"(a), "l"(b), "l"(b));
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a));
        asm volatile("barrier.cluster.arrive;");
        asm volatile("setmaxnreg.inc.sync.aligned.u32 256;");
        uint32_t pred;
        asm volatile("{ .reg .pred p; elect.sync _|p, 0xFFFFFFFF; selp.b32 %0, 1, 0, p; }" : "=r"(pred));
    }
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void tma_copy_1d_g2s_fn(void const* gmem, uint64_t* mbar, void* smem, int32_t bytes) {
    if (bytes <= 0) return;
    uint32_t smem_cta  = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t mbar_cta  = (uint32_t)__cvta_generic_to_shared(mbar);
    
    // Resolve the .shared::cta pointers to .shared::cluster pointers natively for Hopper TMA
    uint32_t rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    uint32_t smem_cluster, mbar_cluster;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(smem_cluster) : "r"(smem_cta), "r"(rank));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(mbar_cluster) : "r"(mbar_cta), "r"(rank));
    
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :: "r"(smem_cluster), "l"(gmem), "r"(bytes), "r"(mbar_cluster) : "memory");
}

__device__ __forceinline__ void copy_g2s_async_2(
    void* smem1, const void* gmem1, int32_t valid_rows1,
    void* smem2, const void* gmem2, int32_t valid_rows2,
    uint64_t* mbar, uint32_t phaseParity) 
{
    int32_t bytes1 = valid_rows1 * 256;
    int32_t bytes2 = valid_rows2 * 256;
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, bytes1 + bytes2);
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        tma_copy_1d_g2s_fn(gmem1, mbar, smem1, bytes1);
        tma_copy_1d_g2s_fn(gmem2, mbar, smem2, bytes2);
    }
    mbarrier_wait_fn(mbar, phaseParity);
    __syncthreads();
    
    int zero_bytes1 = (64 - valid_rows1) * 256;
    if (zero_bytes1 > 0) {
        for (int i = threadIdx.x; i < zero_bytes1 / 16; i += blockDim.x) {
            reinterpret_cast<float4*>((char*)smem1 + bytes1)[i] = {0,0,0,0};
        }
    }
    int zero_bytes2 = (64 - valid_rows2) * 256;
    if (zero_bytes2 > 0) {
        for (int i = threadIdx.x; i < zero_bytes2 / 16; i += blockDim.x) {
            reinterpret_cast<float4*>((char*)smem2 + bytes2)[i] = {0,0,0,0};
        }
    }
    __syncthreads();
}

__global__ void precompute_D_kernel(const __nv_bfloat16* __restrict__ dO, 
                                    const __nv_bfloat16* __restrict__ O, 
                                    float* __restrict__ D, 
                                    int B, int H, int S) 
{
    int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;
    if (seq_idx < S) {
        float sum = 0.0f;
        int offset = (b * H * S + h * S + seq_idx) * 128;
        
        #pragma unroll 16
        for (int d = 0; d < 128; d += 2) {
            __nv_bfloat162 do_val = *reinterpret_cast<const __nv_bfloat162*>(&dO[offset + d]);
            __nv_bfloat162 o_val = *reinterpret_cast<const __nv_bfloat162*>(&O[offset + d]);
            sum += __bfloat162float(do_val.x) * __bfloat162float(o_val.x);
            sum += __bfloat162float(do_val.y) * __bfloat162float(o_val.y);
        }
        D[b * H * S + h * S + seq_idx] = sum;
    }
}

__global__ void mha_bwd_d128_causal_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const float* __restrict__ L,
    const float* __restrict__ D,
    const __nv_bfloat16* __restrict__ dO,
    __nv_bfloat16* __restrict__ dQ,
    float* __restrict__ dK_fp32,
    float* __restrict__ dV_fp32,
    int B, int H, int S, float scale) 
{
    dummy_instruction_warning_fix();
    
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i_chunk = blockIdx.x; 
    
    int i_base = i_chunk * 64;
    int valid_rows_q = min(64, S - i_base);
    if (valid_rows_q <= 0) return;
    
    int head_offset = (b * H + h) * S * 128;
    int vec_offset = (b * H + h) * S;
    
    const __nv_bfloat16* q_ptr = Q + head_offset + i_base * 128;
    const __nv_bfloat16* do_ptr = dO + head_offset + i_base * 128;
    __nv_bfloat16* dq_ptr = dQ + head_offset + i_base * 128;
    const float* l_ptr = L + vec_offset + i_base;
    const float* d_ptr = D + vec_offset + i_base;
    
    extern __shared__ __align__(128) char shared_mem[];
    __nv_bfloat16* s_Q = (__nv_bfloat16*)shared_mem;                             // 16 KB
    __nv_bfloat16* s_dO = s_Q + 64 * 128;                                        // 16 KB
    __nv_bfloat16* s_K = s_dO + 64 * 128;                                        // 32 KB (Double Buffered)
    __nv_bfloat16* s_V = s_K + 2 * 64 * 128;                                     // 32 KB (Double Buffered)
    float* s_S = (float*)(s_V + 2 * 64 * 128);                                   // 16 KB
    float* s_dP = s_S + 64 * 64;                                                 // 16 KB
    __nv_bfloat16* s_P = (__nv_bfloat16*)(s_dP + 64 * 64);                       // 8 KB
    __nv_bfloat16* s_dS = s_P + 64 * 64;                                         // 8 KB
    float* s_cast_float = (float*)(s_dS + 64 * 64);                              // 32 KB
    float* s_L = (float*)(s_cast_float + 64 * 128);                              // 256 B
    float* s_D = s_L + 64;                                                       // 256 B
    uint64_t* mbar = (uint64_t*)(s_D + 64);                                      // 8 B

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    if (threadIdx.x < 64) {
        s_L[threadIdx.x] = (threadIdx.x < valid_rows_q) ? l_ptr[threadIdx.x] : 0.0f;
        s_D[threadIdx.x] = (threadIdx.x < valid_rows_q) ? d_ptr[threadIdx.x] : 0.0f;
    }

    uint32_t phase = 0;
    copy_g2s_async_2(s_Q, q_ptr, valid_rows_q, s_dO, do_ptr, valid_rows_q, mbar, phase & 1); 
    phase++;
    
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_frag[8];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dq_acc[8];
    for (int i = 0; i < 8; i++) {
        wmma::fill_fragment(dq_acc[i], 0.0f);
    }
    
    int warp_id = threadIdx.x / 32;
    int row_s = warp_id * 16; 
    
    for (int k = 0; k < 8; k++) { 
        wmma::load_matrix_sync(q_frag[k], &s_Q[row_s * 128 + k * 16], 128);
    }
    
    int load_idx = 0;
    int j_base = 0;
    int valid_rows_kv = min(64, S - j_base);
    
    if (valid_rows_kv > 0) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, valid_rows_kv * 512);
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            tma_copy_1d_g2s_fn(K + head_offset, mbar, &s_K[0], valid_rows_kv * 256);
            tma_copy_1d_g2s_fn(V + head_offset, mbar, &s_V[0], valid_rows_kv * 256);
        }
    }
    
    for (int j_chunk = 0; j_chunk <= i_chunk; j_chunk++) {
        mbarrier_wait_fn(mbar, phase & 1);
        __syncthreads();
        
        int cur_valid_rows_kv = valid_rows_kv;
        if (cur_valid_rows_kv < 64) {
            for (int i = threadIdx.x; i < (64 - cur_valid_rows_kv) * 16; i += blockDim.x) { 
                reinterpret_cast<float4*>(&s_K[load_idx * 8192 + cur_valid_rows_kv * 128])[i] = {0,0,0,0};
                reinterpret_cast<float4*>(&s_V[load_idx * 8192 + cur_valid_rows_kv * 128])[i] = {0,0,0,0};
            }
        }
        __syncthreads();
        
        int next_j_chunk = j_chunk + 1;
        int next_j_base = next_j_chunk * 64;
        int next_valid_rows = min(64, S - next_j_base);
        int next_load_idx = 1 - load_idx;
        phase++;
        
        if (next_j_chunk <= i_chunk && next_valid_rows > 0) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(mbar, next_valid_rows * 512);
                asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
                tma_copy_1d_g2s_fn(K + head_offset + next_j_base * 128, mbar, &s_K[next_load_idx * 8192], next_valid_rows * 256);
                tma_copy_1d_g2s_fn(V + head_offset + next_j_base * 128, mbar, &s_V[next_load_idx * 8192], next_valid_rows * 256);
            }
        }
        
        __nv_bfloat16* cur_s_K = &s_K[load_idx * 8192];
        __nv_bfloat16* cur_s_V = &s_V[load_idx * 8192];
        
        { 
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_acc[4]; 
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> dp_acc[4];
            for (int i = 0; i < 4; i++) {
                wmma::fill_fragment(s_acc[i], 0.0f);
                wmma::fill_fragment(dp_acc[i], 0.0f);
            }
            
            for (int k_step = 0; k_step < 8; k_step++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> do_f;
                wmma::load_matrix_sync(do_f, &s_dO[row_s * 128 + k_step * 16], 128);
                
                for(int c=0; c<4; c++) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> k_frag;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> v_frag;
                    wmma::load_matrix_sync(k_frag, &cur_s_K[c * 16 * 128 + k_step * 16], 128);
                    wmma::load_matrix_sync(v_frag, &cur_s_V[c * 16 * 128 + k_step * 16], 128);
                    
                    wmma::mma_sync(s_acc[c], q_frag[k_step], k_frag, s_acc[c]);
                    wmma::mma_sync(dp_acc[c], do_f, v_frag, dp_acc[c]);
                }
            }
            
            for (int c = 0; c < 4; c++) {
                for (int t = 0; t < s_acc[c].num_elements; t++) s_acc[c].x[t] *= scale;
                wmma::store_matrix_sync(&s_S[row_s * 64 + c * 16], s_acc[c], 64, wmma::mem_row_major);
                wmma::store_matrix_sync(&s_dP[row_s * 64 + c * 16], dp_acc[c], 64, wmma::mem_row_major);
            }
        }
        __syncthreads();
        
        {
            for (int i = threadIdx.x; i < 64 * 64; i += 128) {
                int r = i / 64;
                int c = i % 64;
                float p_val = 0.0f;
                float ds_val = 0.0f;
                if (r < valid_rows_q && c < cur_valid_rows_kv) {
                    int q_idx = i_base + r;
                    int k_idx = j_base + c;
                    if (k_idx <= q_idx) { 
                        float s_val = s_S[i];
                        p_val = expf(s_val - s_L[r]);
                        float dp_val = s_dP[i];
                        ds_val = p_val * (dp_val - s_D[r]) * scale;
                    }
                }
                s_P[i] = __float2bfloat16(p_val);
                s_dS[i] = __float2bfloat16(ds_val);
            }
        }
        __syncthreads();
        
        {
            for (int c_step = 0; c_step < 4; c_step++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> ds_f;
                wmma::load_matrix_sync(ds_f, &s_dS[row_s * 64 + c_step * 16], 64);
                for (int k_step = 0; k_step < 8; k_step++) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> k_f;
                    wmma::load_matrix_sync(k_f, &cur_s_K[c_step * 16 * 128 + k_step * 16], 128);
                    wmma::mma_sync(dq_acc[k_step], ds_f, k_f, dq_acc[k_step]);
                }
            }
        }
        
        {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> dk_acc[8];
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> dv_acc[8];
            for (int i = 0; i < 8; i++) {
                wmma::fill_fragment(dk_acc[i], 0.0f);
                wmma::fill_fragment(dv_acc[i], 0.0f);
            }
            
            for (int c_step = 0; c_step < 4; c_step++) { 
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> ds_T_f;
                wmma::load_matrix_sync(ds_T_f, &s_dS[c_step * 16 * 64 + row_s], 64);
                
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::col_major> p_T_f;
                wmma::load_matrix_sync(p_T_f, &s_P[c_step * 16 * 64 + row_s], 64);
                
                for (int k_step = 0; k_step < 8; k_step++) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> q_f;
                    wmma::load_matrix_sync(q_f, &s_Q[c_step * 16 * 128 + k_step * 16], 128);
                    
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> do_f;
                    wmma::load_matrix_sync(do_f, &s_dO[c_step * 16 * 128 + k_step * 16], 128);
                    
                    wmma::mma_sync(dk_acc[k_step], ds_T_f, q_f, dk_acc[k_step]);
                    wmma::mma_sync(dv_acc[k_step], p_T_f, do_f, dv_acc[k_step]);
                }
            }
            
            __syncthreads();
            for (int k = 0; k < 8; k++) {
                wmma::store_matrix_sync(&s_cast_float[row_s * 128 + k * 16], dk_acc[k], 128, wmma::mem_row_major);
            }
            __syncthreads();
            
            float* dk_fp32_ptr = dK_fp32 + head_offset + j_base * 128;
            for (int i = threadIdx.x; i < 64 * 128; i += 128) {
                int r = i / 128;
                if (r < cur_valid_rows_kv) {
                    atomicAdd(&dk_fp32_ptr[i], s_cast_float[i]);
                }
            }
            __syncthreads();
            
            for (int k = 0; k < 8; k++) {
                wmma::store_matrix_sync(&s_cast_float[row_s * 128 + k * 16], dv_acc[k], 128, wmma::mem_row_major);
            }
            __syncthreads();
            
            float* dv_fp32_ptr = dV_fp32 + head_offset + j_base * 128;
            for (int i = threadIdx.x; i < 64 * 128; i += 128) {
                int r = i / 128;
                if (r < cur_valid_rows_kv) {
                    atomicAdd(&dv_fp32_ptr[i], s_cast_float[i]);
                }
            }
            __syncthreads();
        }
        
        load_idx = next_load_idx;
        valid_rows_kv = next_valid_rows;
        j_base = next_j_base;
    }
    
    for (int k = 0; k < 8; k++) {
        wmma::store_matrix_sync(&s_cast_float[row_s * 128 + k * 16], dq_acc[k], 128, wmma::mem_row_major);
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < 64 * 128 / 2; i += 128) {
        int r = i / 64;
        int c = (i % 64) * 2;
        if (r < valid_rows_q) {
            float f0 = s_cast_float[r * 128 + c];
            float f1 = s_cast_float[r * 128 + c + 1];
            __nv_bfloat162 val2;
            val2.x = __float2bfloat16(f0);
            val2.y = __float2bfloat16(f1);
            *reinterpret_cast<__nv_bfloat162*>(&dq_ptr[r * 128 + c]) = val2;
        }
    }
}

__global__ void convert_fp32_to_bf16(const float* __restrict__ src, __nv_bfloat16* __restrict__ dst, int64_t total) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) 
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = static_cast<int>(Q.size(0));
    int H = static_cast<int>(Q.size(1));
    int S = static_cast<int>(Q.size(2));
    if (S == 0) return;
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    __nv_bfloat16* do_ptr = static_cast<__nv_bfloat16*>(dO.data_ptr());
    float* l_ptr = static_cast<float*>(L.data_ptr());
    __nv_bfloat16* dq_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dk_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dv_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());

    int64_t total_elements = static_cast<int64_t>(B) * H * S * 128;
    float *dK_fp32, *dV_fp32, *d_ptr;
    
    CUDA_CHECK(cudaMallocAsync(&dK_fp32, total_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&dV_fp32, total_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&d_ptr, static_cast<int64_t>(B) * H * S * sizeof(float), stream));
    
    CUDA_CHECK(cudaMemsetAsync(dK_fp32, 0, total_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dV_fp32, 0, total_elements * sizeof(float), stream));
    
    dim3 d_block(128);
    dim3 d_grid((S + 127) / 128, H, B);
    precompute_D_kernel<<<d_grid, d_block, 0, stream>>>(do_ptr, o_ptr, d_ptr, B, H, S);
    
    int smem_size = 196608; // 192 KB safely bounds the required 176.5 KB payload
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_d128_causal_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    dim3 bwd_block(128);
    dim3 bwd_grid((S + 63) / 64, H, B);
    float scale = 1.0f / sqrtf(128.0f);
    
    mha_bwd_d128_causal_kernel<<<bwd_grid, bwd_block, smem_size, stream>>>(
        q_ptr, k_ptr, v_ptr, l_ptr, d_ptr, do_ptr, dq_ptr, dK_fp32, dV_fp32, B, H, S, scale);
        
    dim3 conv_block(256);
    dim3 conv_grid((total_elements + 255) / 256);
    convert_fp32_to_bf16<<<conv_grid, conv_block, 0, stream>>>(dK_fp32, dk_ptr, total_elements);
    convert_fp32_to_bf16<<<conv_grid, conv_block, 0, stream>>>(dV_fp32, dv_ptr, total_elements);
        
    CUDA_CHECK(cudaFreeAsync(d_ptr, stream));
    CUDA_CHECK(cudaFreeAsync(dK_fp32, stream));
    CUDA_CHECK(cudaFreeAsync(dV_fp32, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd