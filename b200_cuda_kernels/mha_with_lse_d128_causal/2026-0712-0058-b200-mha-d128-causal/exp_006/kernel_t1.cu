#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <stdio.h>
#include <cmath>
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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

__device__ __forceinline__ void named_barrier_arrive_fn(int bar_id, int count) {
    asm volatile("barrier.arrive.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

CUresult create_tma_3d_descriptor(CUtensorMap* d, void* globalAddress,
                                  uint64_t dim0, uint64_t dim1, uint64_t dim2,
                                  uint32_t box0, uint32_t box1,
                                  CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, 1};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3,
        globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_k_major(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= (0u << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_b_mn_major(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= (1u << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void umma_f16_cg1_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ int swizzle_128B_offset(int r, int c) {
    int x = c / 8;
    int x_swizzled = x ^ (r % 8);
    return r * 64 + x_swizzled * 8 + (c % 8);
}

CUresult create_tma_4d_descriptor(CUtensorMap* d, void* globalAddress,
                                  uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3,
                                  uint32_t box0, uint32_t box1,
                                  CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0 * 2, dim0 * dim1 * 2, dim0 * dim1 * dim2 * 2};
    cuuint32_t boxDim[4] = {box0, box1, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
        globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3); 
    
    if (S == 0) return;
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    float* global_running_max = new float[B * H * S];
    float* global_running_sum = new float[B * H * S];
    
    CUDA_CHECK(cudaMemcpyAsync(
        global_running_max, global_running_max, B * H * S * sizeof(float),
        cudaMemcpyDefault, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        global_running_sum, global_running_sum, B * H * S * sizeof(float),
        cudaMemcpyDefault, stream));
        
    CUtensorMap tma_Q, tma_K, tma_V;
    create_tma_4d_descriptor(&tma_Q, Q.data_ptr(), D, S, H, B, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_4d_descriptor(&tma_K, K.data_ptr(), D, S, H, B, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_4d_descriptor(&tma_V, V.data_ptr(), D, S, H, B, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B);
    
    uint32_t smem_size = 57344;
    dim3 grid((S + 63) / 64, H, B);
    dim3 block(128);
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, run_kernel, tma_Q, tma_K, tma_V, O.data_ptr(), LSE.data_ptr(), S, H, B, global_running_max, global_running_sum));
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaFreeAsync(global_running_max, stream));
    CUDA_CHECK(cudaFreeAsync(global_running_sum, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

__global__ void run_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O_ptr, float* LSE_ptr,
    int S, int num_heads, int batch_size, float* global_running_max, float* global_running_sum) 
{
    extern __shared__ __align__(128) uint8_t smem_pool[];
    uintptr_t pool_addr = (uintptr_t)smem_pool;
    uint8_t* cur = smem_pool + (1024 - (pool_addr % 1024)) % 1024;
    
    __nv_bfloat16* s_Q0 = (__nv_bfloat16*)cur;
    __nv_bfloat16* s_Q1 = s_Q0 + 4096;
    __nv_bfloat16* s_K0 = s_Q1 + 4096;
    __nv_bfloat16* s_K1 = s_K0 + 4096;
    __nv_bfloat16* s_V0 = s_K1 + 4096;
    __nv_bfloat16* s_V1 = s_V0 + 4096;
    __nv_bfloat16* s_P  = s_V1 + 4096;
    uint64_t* mbar = (uint64_t*)(s_P + 4096);
    
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    int q_base = blockIdx.x * 64;
    int global_q_base = batch_idx * num_heads * S + head_idx * S + q_base;
    
    int flattened_head = head_idx + batch_idx * num_heads;
    int q_idx = flattened_head * S + q_base;

    uint32_t tmem_S[2], tmem_P[2], tmem_O[2];
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_S[0], 64);
        tmem_alloc_fn(&tmem_S[1], 64);
        tmem_alloc_fn(&tmem_P[0], 64);
        tmem_alloc_fn(&tmem_P[1], 64);
        tmem_alloc_fn(&tmem_O[0], 64);
        tmem_alloc_fn(&tmem_O[1], 64);
    }

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar[0], 1);
        init_smem_barrier_fn(&mbar[1], 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar[0], 16384);
        tma_load_3d_fn(&tma_Q, &mbar[0], s_Q0, 0, q_base, flattened_head);
        tma_load_3d_fn(&tma_Q, &mbar[0], s_Q1, 64, q_base, flattened_head);
        
        mbarrier_arrive_and_expect_tx_fn(&mbar[1], 32768);
        tma_load_3d_fn(&tma_K, &mbar[1], s_K0, 0, 0, flattened_head);
        tma_load_3d_fn(&tma_K, &mbar[1], s_K1, 64, 0, flattened_head);
        tma_load_3d_fn(&tma_V, &mbar[1], s_V0, 0, 0, flattened_head);
        tma_load_3d_fn(&tma_V, &mbar[1], s_V1, 64, 0, flattened_head);
    }
    mbarrier_wait_fn(&mbar[0], 0);
    __syncthreads();
    
    int num_steps = q_base / 64;
    if (num_steps > (S - 1) / 64) {
        num_steps = (S - 1) / 64;
    }
    
    float O_left[64] = {0};
    float O_right[64] = {0};
    
    int next_barrier_phase = 0;
    float scale_factor = 1.0f / sqrtf(128.0f);
    
    for (int step = 0; step <= num_steps; ++step) {
        int k_base = step * 64;
        int next_k_base = (step + 1) * 64;
        
        if (step + 1 <= num_steps) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar[1], 32768);
                tma_load_3d_fn(&tma_K, &mbar[1], s_K0, 0, next_k_base, flattened_head);
                tma_load_3d_fn(&tma_K, &mbar[1], s_K1, 64, next_k_base, flattened_head);
                tma_load_3d_fn(&tma_V, &mbar[1], s_V0, 0, next_k_base, flattened_head);
                tma_load_3d_fn(&tma_V, &mbar[1], s_V1, 64, next_k_base, flattened_head);
            }
        }
        
        int wait_parity = (step) & 1;
        mbarrier_wait_fn(&mbar[1], wait_parity);
        
        uint32_t row_base = (threadIdx.x / 32) * 16;
        uint32_t addr_S0 = tmem_S[0] + row_base * 64;
        uint32_t addr_S1 = tmem_S[1] + row_base * 64;
        float S_val[32];
        asm volatile(
            "tcgen05.ld.sync.aligned.16x128b.x8.pack::16b.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n"
            "tcgen05.wait::ld.sync.aligned;\n"
            : "=r"(S_val[0]), "=r"(S_val[1]), "=r"(S_val[2]), "=r"(S_val[3]),
              "=r"(S_val[4]), "=r"(S_val[5]), "=r"(S_val[6]), "=r"(S_val[7]),
              "=r"(S_val[8]), "=r"(S_val[9]), "=r"(S_val[10]), "=r"(S_val[11]),
              "=r"(S_val[12]), "=r"(S_val[13]), "=r"(S_val[14]), "=r"(S_val[15])
            : "r"(addr_S0));
        
        asm volatile(
            "tcgen05.ld.sync.aligned.16x128b.x8.pack::16b.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n"
            "tcgen05.wait::ld.sync.aligned;\n"
            : "=r"(S_val[16]), "=r"(S_val[17]), "=r"(S_val[18]), "=r"(S_val[19]),
              "=r"(S_val[20]), "=r"(S_val[21]), "=r"(S_val[22]), "=r"(S_val[23]),
              "=r"(S_val[24]), "=r"(S_val[25]), "=r"(S_val[26]), "=r"(S_val[27]),
              "=r"(S_val[28]), "=r"(S_val[29]), "=r"(S_val[30]), "=r"(S_val[31])
            : "r"(addr_S1));

        for (int i = 0; i < 32; i++) {
            int idx = i;
            int r = (threadIdx.x / 32) * 16 + (idx / 2);
            int c = (threadIdx.x % 32) * 2 + (idx % 2);
            int global_q_idx = q_base + r;
            int global_k_idx = k_base + c;
            if (global_q_idx < global_k_idx || global_k_idx >= S) {
                S_val[i] = -INFINITY;
            } else {
                S_val[i] = __uint_as_float(S_val[i]); // Convert from packed P to standard float mapping scaling normalization bounds mapping  
                S_val[i] *= scale_factor;
            }
        }
        
        float row_max[16] = {-INFINITY};
        for (int i = 0; i < 32; i++) {
            int r = (i / 2) % 16;
            row_max[r] = fmaxf(row_max[r], S_val[i]);
        }
        
        float current_row_max[16];
        for (int r = 0; r < 16; r++) {
            float max_val = row_max[r];
            for (int offset = 1; offset < 32; offset *= 2) {
                max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
            }
            current_row_max[r] = max_val;
        }
        
        for (int r = 0; r < 16; r++) {
            int abs_r = (threadIdx.x / 32) * 16 + r;
            int global_q_idx = q_base + abs_r;
            if (global_q_idx >= S) continue; 
            
            float prev_max = global_running_max[q_idx + abs_r];
            if (prev_max > -INFINITY) {
                float scale = fast_exp2f_fn((prev_max - current_row_max[r]) * 1.4426950f);
                for (int c = 0; c < 64; c++) {
                    if (c < 32) O_left[c] *= scale;
                    else O_right[c - 32] *= scale;
                }
                
                float global_sum = global_running_sum[q_idx + abs_r];
                global_sum *= scale;
                if (threadIdx.x % 32 == 0) {
                    global_running_sum[q_idx + abs_r] = global_sum;
                }
            }
            
            if (threadIdx.x % 32 == 0) {
                global_running_max[q_idx + abs_r] = current_row_max[r];
            }
        }
        
        float row_sum[16] = {0};
        for (int i = 0; i < 32; i++) {
            int idx = i;
            int r = (idx / 2) % 16;
            int c = (threadIdx.x % 32) * 2 + (idx % 2);
            
            float p = 0;
            if (S_val[i] != -INFINITY) {
                p = fast_exp2f_fn((S_val[i] - current_row_max[r]) * 1.4426950f);
            }
            row_sum[r] += p;
            
            int abs_c = (idx % 2 == 0) ? c : c + 1; 
            int pack_idx = abs_c / 2;
            float p0 = (abs_c % 2 == 0) ? S_val[i] : ((idx % 2 == 0) ? S_val[i+1] : S_val[i-1]);
            float p1 = (abs_c % 2 == 1) ? S_val[i] : ((idx % 2 == 0) ? S_val[i+1] : S_val[i-1]);
            
            if (p0 == -INFINITY) p0 = 0;
            if (p1 == -INFINITY) p1 = 0;
            
            uint32_t packed = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
            int abs_r = (threadIdx.x / 32) * 16 + r;
            int sc = swizzle_128B_offset(abs_r, abs_c);
            *(uint32_t*)&s_P[sc] = packed;
        }
        
        for (int r = 0; r < 16; r++) {
            float sum_val = row_sum[r];
            for (int offset = 1; offset < 32; offset *= 2) {
                sum_val += __shfl_xor_sync(0xFFFFFFFF, sum_val, offset);
            }
            int abs_r = (threadIdx.x / 32) * 16 + r;
            if (threadIdx.x % 32 == 0) {
                int global_q_idx = q_base + abs_r;
                if (global_q_idx < S) {
                    global_running_sum[q_idx + abs_r] += sum_val;
                }
            }
        }
        
        __syncthreads();
        fence_async_shared_fn();
        
        named_barrier_sync_fn(1, 128); 
        
        for (int k = 0; k < 4; ++k) {
            uint32_t col = k * 16;
            uint32_t addr_P = tmem_P[0] + col;
            uint32_t addr_V0 = tmem_V[0] + col * 64;
            uint32_t idesc_PV = make_instr_desc_b_mn_major(64, 64);
            umma_f16_cg1_fn(tmem_O[0], addr_P, addr_V0, idesc_PV, k == 0 ? 0 : 1);
        }
        for (int k = 0; k < 4; ++k) {
            uint32_t col = k * 16;
            uint32_t addr_P = tmem_P[1] + col;
            uint32_t addr_V1 = tmem_V[1] + col * 64;
            uint32_t idesc_PV = make_instr_desc_b_mn_major(64, 64);
            umma_f16_cg1_fn(tmem_O[1], addr_P, addr_V1, idesc_PV, k == 0 ? 0 : 1);
        }
        umma_commit_fn(&mbar[0]);
        mbarrier_wait_fn(&mbar[0], next_barrier_phase & 1);
        named_barrier_sync_fn(2, 128);
        next_barrier_phase++;
        
        __syncthreads();
    }
    
    __syncwarp();
    if (threadIdx.x < 64) {
        int tid = threadIdx.x;
        float sum = global_running_sum[q_idx + tid];
        for (int c_step = 0; c_step < 64; c_step+=2) {
            float out0 = O_left[c_step] / sum;
            float out1 = O_left[c_step+1] / sum;
            uint32_t packed_L = pack_bf16_fn(__float_as_uint(out0), __float_as_uint(out1));
            int sc = swizzle_128B_offset(tid, c_step);
            *(uint32_t*)(&s_Q0[sc]) = packed_L;
        }
        
        for (int c_step = 0; c_step < 64; c_step+=2) {
            float out0 = O_right[c_step] / sum;
            float out1 = O_right[c_step+1] / sum;
            uint32_t packed_R = pack_bf16_fn(__float_as_uint(out0), __float_as_uint(out1));
            int sc = swizzle_128B_offset(tid, c_step);
            *(uint32_t*)(&s_Q1[sc]) = packed_R;
        }
    }
    
    __syncthreads(); 
    
    extern __shared__ __align__(128) uint8_t smem_out[];
    uint8_t* smem_out_L = smem_out;
    uint8_t* smem_out_R = smem_out + 4096;
    __nv_bfloat16* bq0 = (__nv_bfloat16*)smem_out_L;
    __nv_bfloat16* bq1 = (__nv_bfloat16*)smem_out_R;
    
    for (int i = 0; i < 4096; i++) {
        smem_out_L[i] = s_Q0[i];
        smem_out_R[i] = s_Q1[i];
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < 4096 / 4; i += blockDim.x) {
        int tid = (i / 16) % 64; 
        int c_step = (i % 16) * 2;
        int g_row = q_base + tid;
        int g_col_L = c_step;
        int g_col_R = c_step + 64;
        
        if (g_row < S && g_col_L < 128) {
            uint32_t val = *(uint32_t*)&smem_out_L[i * 4];
            uint64_t off_L = (uint64_t)(batch_idx * num_heads * S + head_idx * S) * 128 + g_row * 128 + g_col_L;
            *(uint32_t*)&O_ptr[off_L] = val;
        }
        if (g_row < S && g_col_R < 128) {
            uint32_t val = *(uint32_t*)&smem_out_R[i * 4];
            uint64_t off_R = (uint64_t)(batch_idx * num_heads * S + head_idx * S) * 128 + g_row * 128 + g_col_R;
            *(uint32_t*)&O_ptr[off_R] = val;
        }
    }
    
    if (threadIdx.x < 64) {
        int tid = threadIdx.x;
        int global_q_idx = q_base + tid;
        if (global_q_idx < S) {
            float max_val = global_running_max[q_idx + tid];
            float sum_val = global_running_sum[q_idx + tid];
            LSE_ptr[global_q_base + global_q_idx] = max_val + logf(sum_val);
        }
    }
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_S[0], 64);
        tmem_dealloc_fn(tmem_S[1], 64);
        tmem_dealloc_fn(tmem_P[0], 64);
        tmem_dealloc_fn(tmem_P[1], 64);
        tmem_dealloc_fn(tmem_O[0], 64);
        tmem_dealloc_fn(tmem_O[1], 64);
    }
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);