#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
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

#define DRV_CHECK(call) do {                                       \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "Driver error %d at %s:%d\n",             \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha {

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
                                     uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3,
                                     uint32_t box0, uint32_t box1, uint32_t box2, uint32_t box3, 
                                     CUtensorMapDataType dataType,
                                     CUtensorMapSwizzle swizzle, 
                                     CUtensorMapL2promotion l2Promotion, 
                                     CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0 * 2, dim0 * dim1 * 2, dim0 * dim1 * dim2 * 2}; 
    cuuint32_t boxDim[4] = {box0, box1, box2, box3};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, dataType, 4, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        l2Promotion, oobFill
    );
}

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
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

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::: "memory");
}

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void umma_f16_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;    // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // FP32
    d |= (1u << 7);    // BF16
    d |= (1u << 10);   // BF16
    d |= (0u << 15);   // K-major A
    d |= (0u << 16);   // K-major B
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__global__ __launch_bounds__(128, 1) void mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O,
    float* LSE,
    uint32_t S_len)
{
    setmaxnreg_inc_sync_fn<248>();

    uint32_t q_block = blockIdx.x * 128;
    uint32_t bh = blockIdx.y;
    uint32_t b = bh / 48;
    uint32_t h = bh % 48;
    uint32_t cr = cluster_rank_fn();
    uint32_t q_start = q_block + cr * 64;
    
    extern __shared__ char smem_raw[];
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_raw);
    uint32_t smem_aligned = (smem_addr + 1023) & ~1023;
    char* smem = smem_raw + (smem_aligned - smem_addr);
    
    __nv_bfloat16* smem_Q_0 = (__nv_bfloat16*)smem;                 // 8KB
    __nv_bfloat16* smem_Q_1 = (__nv_bfloat16*)(smem + 8192);         // 8KB
    __nv_bfloat16* smem_K_0 = (__nv_bfloat16*)(smem + 16384);        // 8KB
    __nv_bfloat16* smem_K_1 = (__nv_bfloat16*)(smem + 24576);        // 8KB
    __nv_bfloat16* smem_V_0 = (__nv_bfloat16*)(smem + 32768);        // 8KB
    __nv_bfloat16* smem_V_1 = (__nv_bfloat16*)(smem + 40960);        // 8KB
    __nv_bfloat16* smem_P   = (__nv_bfloat16*)(smem + 49152);        // 8KB
    __nv_bfloat16* smem_O   = (__nv_bfloat16*)(smem + 57344);        // 16KB
    
    uint64_t* bar_q = (uint64_t*)(smem + 73728);
    uint64_t* bar_k = (uint64_t*)(smem + 73736);
    uint64_t* bar_v = (uint64_t*)(smem + 73744);
    uint64_t* bar_umma = (uint64_t*)(smem + 73752);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_q, 1);
        init_smem_barrier_fn(bar_k, 1);
        init_smem_barrier_fn(bar_v, 1);
        init_smem_barrier_fn(bar_umma, 1);
    }
    fence_smem_barrier_init_fn();
    cluster_sync_fn();

    uint32_t tmem_S_addr, tmem_O_addr;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_S_addr, 64);
        tmem_alloc_fn(&tmem_O_addr, 128);
    }
    cluster_sync_fn();

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_q, 32768);
        tma_load_4d_fn(&tma_Q, bar_q, smem_Q_0, 0, q_start, h, b);
        tma_load_4d_fn(&tma_Q, bar_q, smem_Q_1, 64, q_start, h, b);
    }
    mbarrier_wait_fn(bar_q, 0);

    float curr_max_val = -INFINITY;
    float curr_sum = 0.0f;
    float scale = 1.0f / sqrtf(128.0f);

    uint32_t num_steps = (S_len + 63) / 64;
    for (int step = 0; step < num_steps; ++step) {
        uint32_t phase = step % 2;
        uint32_t kv_block = step * 64;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_k, 16384);
            tma_load_4d_fn(&tma_K, bar_k, smem_K_0, 0, kv_block, h, b);
            tma_load_4d_fn(&tma_K, bar_k, smem_K_1, 64, kv_block, h, b);
            
            mbarrier_arrive_and_expect_tx_fn(bar_v, 16384);
            tma_load_4d_fn(&tma_V, bar_v, smem_V_0, 0, kv_block, h, b);
            tma_load_4d_fn(&tma_V, bar_v, smem_V_1, 64, kv_block, h, b);
        }
        mbarrier_wait_fn(bar_k, phase);
        mbarrier_wait_fn(bar_v, phase);

        if (threadIdx.x == 0) {
            for(int half = 0; half < 2; ++half) {
                __nv_bfloat16* q_ptr = (half == 0) ? smem_Q_0 : smem_Q_1;
                __nv_bfloat16* k_ptr = (half == 0) ? smem_K_0 : smem_K_1;
                
                for (int iter = 0; iter < 4; ++iter) {
                    uint64_t desc_a = make_smem_desc_sm100_fn((char*)q_ptr + iter * 32, 0, 1024);
                    uint64_t desc_b = make_smem_desc_sm100_fn((char*)k_ptr + iter * 32, 0, 1024);
                    uint32_t idesc = make_instr_desc_fn(128, 64);
                    uint32_t acc = (step == 0 && iter == 0 && half == 0) ? 0 : 1;
                    umma_f16_cg2_fn(tmem_S_addr, desc_a, desc_b, idesc, acc);
                }
            }
            umma_commit_2sm_fn(bar_umma);
        }
        mbarrier_wait_fn(bar_umma, phase);

        uint32_t lane_offset = (threadIdx.x / 32) * 32;
        uint32_t tmem_S_addr_lane = tmem_S_addr + (lane_offset << 16);
        
        uint32_t S_regs[64];
        for (int i = 0; i < 16; ++i) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S_addr_lane + i * 4, &r0, &r1, &r2, &r3);
            S_regs[i * 4 + 0] = r0;
            S_regs[i * 4 + 1] = r1;
            S_regs[i * 4 + 2] = r2;
            S_regs[i * 4 + 3] = r3;
        }
        tmem_load_fence_fn();

        uint32_t global_row = q_start + lane_offset + (threadIdx.x % 32);
        bool valid_row = global_row < S_len;
        float row_max = -INFINITY;
        
        for (int i = 0; i < 64; ++i) {
            uint32_t global_kv_idx = kv_block + i;
            if (!valid_row || global_kv_idx >= S_len || isNaN(__uint_as_float(S_regs[i]))) {
                S_regs[i] = __float_as_uint(-INFINITY);
            } else {
                float val = __uint_as_float(S_regs[i]) * scale;
                S_regs[i] = __float_as_uint(val);
            }
            row_max = fmaxf(row_max, __uint_as_float(S_regs[i]));
        }

        float old_max = curr_max_val;
        float new_max = fmaxf(old_max, row_max);
        curr_sum *= expf(old_max - new_max);
        curr_max_val = new_max;

        for (int i = 0; i < 64; ++i) {
            if (isinf(-__uint_as_float(S_regs[i]))) {
                S_regs[i] = __float_as_uint(0.0f);
            } else {
                float val = expf(__uint_as_float(S_regs[i]) - new_max);
                S_regs[i] = __float_as_uint(val);
                curr_sum += val;
            }
        }

        int row = lane_offset + (threadIdx.x % 32);
        for (int i = 0; i < 64; ++i) {
            float val = (curr_sum > 0.0f) ? (__uint_as_float(S_regs[i]) / curr_sum) : 0.0f;
            int chunk = i / 8;
            int in_chunk = i % 8;
            int swizzled_chunk = chunk ^ (row % 8);
            int swizzled_col = swizzled_chunk * 8 + in_chunk;
            smem_P[row * 64 + swizzled_col] = __float2bfloat16(val);
        }

        __syncthreads();

        if (threadIdx.x == 0) {
            for(int half = 0; half < 2; ++half) {
                __nv_bfloat16* v_ptr = (half == 0) ? smem_V_0 : smem_V_1;
                for (int iter = 0; iter < 4; ++iter) {
                    uint64_t desc_a = make_smem_desc_sm100_fn((char*)smem_P + iter * 32, 0, 1024);
                    uint64_t desc_b = make_smem_desc_sm100_fn((char*)v_ptr + iter * 2048, 8192, 1024);
                    uint32_t idesc = make_instr_desc_fn(128, 128);
                    idesc |= (1u << 16); // Transpose B (MN-major)
                    uint32_t acc = (step == 0 && iter == 0 && half == 0) ? 0 : 1;
                    umma_f16_cg2_fn(tmem_O_addr, desc_a, desc_b, idesc, acc);
                }
            }
            umma_commit_2sm_fn(bar_umma);
        }
        mbarrier_wait_fn(bar_umma, phase);
        
        __syncthreads();
    }

    uint32_t tmem_O_addr_lane = tmem_O_addr + (lane_offset << 16);
    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_O_addr_lane + col, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        
        smem_O[threadIdx.x * 128 + col + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_O[threadIdx.x * 128 + col + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_O[threadIdx.x * 128 + col + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_O[threadIdx.x * 128 + col + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps_epilogue = (64 + 3) / 4;
    
    for (uint32_t step_e = 0; step_e < num_steps_epilogue; ++step_e) {
        int row = step_e * 4 + warp_id;
        if (row >= 64) continue;
        
        uint32_t global_row = q_start + row;
        if (global_row >= S_len) continue;
        
        uint32_t col_start = lane_id * 4;
        uint32_t global_col = col_start; 
        
        __nv_bfloat162 out_val = *reinterpret_cast<__nv_bfloat162*>(&smem_O[row * 128 + col_start]);
        *reinterpret_cast<__nv_bfloat162*>(&O[(bh * S_len + global_row) * 128 + global_col]) = out_val;
    }

    if (lane_id == 0) {
        for (int i = 0; i < 4; ++i) {
            int row = step_e * 4 + i;
            if (row >= 64) continue;
            uint32_t global_row = q_start + row;
            if (global_row >= S_len) continue;
            
            if (curr_sum > 0.0f) {
                LSE[bh * S_len + global_row] = curr_max_val + logf(curr_sum);
            } else {
                LSE[bh * S_len + global_row] = -INFINITY;
            }
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S_len = Q.size(2);
    uint32_t D = Q.size(3); 
    
    CUtensorMap tma_Q, tma_K, tma_V;
    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    
    DRV_CHECK(create_tma_4d_descriptor_2B(&tma_Q, q_ptr, D, S_len, H, B, 64, 128, 1, 1, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    DRV_CHECK(create_tma_4d_descriptor_2B(&tma_K, k_ptr, D, S_len, H, B, 64, 64, 1, 1, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    DRV_CHECK(create_tma_4d_descriptor_2B(&tma_V, v_ptr, D, S_len, H, B, 64, 64, 1, 1, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());
    
    uint32_t num_blocks_x = (S_len + 127) / 128;
    uint32_t num_blocks_y = B * H;
    
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(num_blocks_x, num_blocks_y);
    config.blockDim = dim3(128);
    config.dynamicSmemBytes = 74784; 
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 74784));
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_kernel, tma_Q, tma_K, tma_V, o_ptr, lse_ptr, S_len));
    CUDA_CHECK(cudaGetLastError()); 
    CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha