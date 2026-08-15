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

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t k_offset_elements) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t sbo = 1024;
    uint32_t lbo = 16; 
    
    uint32_t addr_low = addr & 0x3ffff;
    uint32_t offset_bytes = k_offset_elements * 2;
    
    d += (uint64_t)((addr_low + offset_bytes) & 0x3ffff) >> 4;
    d += (uint64_t)((lbo & 0x3ffff) >> 4) << 16; 
    d += (uint64_t)((sbo & 0x3ffff) >> 4) << 32; 
    d += (uint64_t)1 << 46;   
    d += (uint64_t)2 << 61;    // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (0u << 15);   // a_major = 0 (A is K-Major)
    d |= (0u << 16);   // b_major = 0 (B is K-Major)
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

    uint32_t q_block = blockIdx.x * 64;
    uint32_t bh = blockIdx.y;
    uint32_t b = bh / 48;
    uint32_t h = bh % 48;
    
    extern __shared__ char smem_raw[];
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_raw);
    uint32_t smem_aligned = (smem_addr + 1023) & ~1023;
    char* smem = smem_raw + (smem_aligned - smem_addr);
    
    __nv_bfloat16* smem_Q_0 = (__nv_bfloat16*)smem;                   // 8KB
    __nv_bfloat16* smem_Q_1 = (__nv_bfloat16*)(smem + 8192);          // 8KB
    __nv_bfloat16* smem_K_0 = (__nv_bfloat16*)(smem + 16384);         // 8KB
    __nv_bfloat16* smem_K_1 = (__nv_bfloat16*)(smem + 24576);         // 8KB
    __nv_bfloat16* smem_V_0 = (__nv_bfloat16*)(smem + 32768);         // 8KB
    __nv_bfloat16* smem_V_1 = (__nv_bfloat16*)(smem + 40960);         // 8KB
    __nv_bfloat16* smem_P   = (__nv_bfloat16*)(smem + 49152);         // 8KB
    float* smem_S           = (float*)(smem + 57344);                 // 16KB
    
    uint64_t* bar_q   = (uint64_t*)(smem + 73728);
    uint64_t* bar_k0  = (uint64_t*)(smem + 73736);
    uint64_t* bar_k1  = (uint64_t*)(smem + 73744);
    uint64_t* bar_umma = (uint64_t*)(smem + 73752);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_q, 1);
        init_smem_barrier_fn(bar_k0, 1);
        init_smem_barrier_fn(bar_k1, 1);
        init_smem_barrier_fn(bar_umma, 1);
    }
    fence_smem_barrier_init_fn();

    uint32_t tmem_S_addr;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_S_addr, 64);
    }
    __syncthreads();

    uint32_t lane_offset = (threadIdx.x / 32) * 32;
    uint32_t lane_id = threadIdx.x % 32;

    float running_max[64];
    float running_sum_exp[64];
    if (threadIdx.x < 64) {
        running_max[threadIdx.x] = -INFINITY;
        running_sum_exp[threadIdx.x] = 0.0f;
    }

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_q, 16384); 
        tma_load_4d_fn(&tma_Q, bar_q, smem_Q_0, 0, q_block, h, b);
        tma_load_4d_fn(&tma_Q, bar_q, smem_Q_1, 64, q_block, h, b);
    }
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_k0, 16384);
        tma_load_4d_fn(&tma_K, bar_k0, smem_K_0, 0, 0, h, b);
        tma_load_4d_fn(&tma_K, bar_k0, smem_K_1, 64, 0, h, b);
        tma_load_4d_fn(&tma_V, bar_k0, smem_V_0, 0, 0, h, b);
        tma_load_4d_fn(&tma_V, bar_k0, smem_V_1, 64, 0, h, b);
    }
    
    mbarrier_wait_fn(bar_q, 0);

    uint32_t num_steps = (S_len + 63) / 64;
    uint32_t phase_k0 = 0;
    uint32_t phase_k1 = 0;
    
    float scale = 1.0f / sqrtf(128.0f);

    for (int step = 0; step < num_steps; ++step) {
        uint32_t next_step = step + 1;
        uint32_t kv_block = step * 64;
        uint32_t next_kv_block = next_step * 64;
        
        int buf = step % 2;
        int next_buf = next_step % 2;
        
        uint64_t* current_bar_k = (buf == 0) ? bar_k0 : bar_k1;
        uint64_t* next_bar_k = (next_buf == 0) ? bar_k0 : bar_k1;
        
        __nv_bfloat16* current_K_0 = (buf == 0) ? smem_K_0 : smem_K_1;
        __nv_bfloat16* current_V_0 = (buf == 0) ? smem_V_0 : smem_V_1;
        __nv_bfloat16* current_K_1 = (buf == 0) ? smem_K_1 : smem_K_0;
        __nv_bfloat16* current_V_1 = (buf == 0) ? smem_V_1 : smem_V_0;
        
        __nv_bfloat16* next_K_0 = (next_buf == 0) ? smem_K_0 : smem_K_1;
        __nv_bfloat16* next_V_0 = (next_buf == 0) ? smem_V_0 : smem_V_1;
        __nv_bfloat16* next_K_1 = (next_buf == 0) ? smem_K_1 : smem_K_0;
        __nv_bfloat16* next_V_1 = (next_buf == 0) ? smem_V_1 : smem_V_0;

        if (next_step < num_steps) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(next_bar_k, 16384);
                tma_load_4d_fn(&tma_K, next_bar_k, next_K_0, 0, next_kv_block, h, b);
                tma_load_4d_fn(&tma_K, next_bar_k, next_K_1, 64, next_kv_block, h, b);
                tma_load_4d_fn(&tma_V, next_bar_k, next_V_0, 0, next_kv_block, h, b);
                tma_load_4d_fn(&tma_V, next_bar_k, next_V_1, 64, next_kv_block, h, b);
            }
        }
        
        mbarrier_wait_fn(current_bar_k, (buf == 0) ? phase_k0 : phase_k1);
        if (buf == 0) phase_k0 ^= 1;
        else phase_k1 ^= 1;

        uint32_t acc_flag = (step == 0) ? 0 : 1;
        if (threadIdx.x == 0) {
            for(int k = 0; k < 4; ++k) {
                uint64_t desc_q = make_smem_desc_sm100_fn(smem_Q_0, k * 16);
                uint64_t desc_k = make_smem_desc_sm100_fn(current_K_0, k * 16);
                uint32_t idesc = make_instr_desc_fn(64, 64);
                umma_f16_cg1_fn(tmem_S_addr + k * 2, desc_q, desc_k, idesc, acc_flag);
            }
            for(int k = 0; k < 4; ++k) {
                uint64_t desc_q = make_smem_desc_sm100_fn(smem_Q_1, k * 16);
                uint64_t desc_k = make_smem_desc_sm100_fn(current_K_1, k * 16);
                uint32_t idesc = make_instr_desc_fn(64, 64);
                umma_f16_cg1_fn(tmem_S_addr + 32 + k * 2, desc_q, desc_k, idesc, 1);
            }
            umma_commit_1sm_fn(bar_umma);
        }
        mbarrier_wait_fn(bar_umma, step % 2);

        uint32_t tmem_S_addr_row = tmem_S_addr + (lane_offset << 16);
        
        for (int i = lane_id; i < 64; i += 32) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S_addr_row + i, &r0, &r1, &r2, &r3);
            smem_S[lane_offset * 64 + i + 0] = __uint_as_float(r0);
            smem_S[lane_offset * 64 + i + 1] = __uint_as_float(r1);
            smem_S[lane_offset * 64 + i + 2] = __uint_as_float(r2);
            smem_S[lane_offset * 64 + i + 3] = __uint_as_float(r3);
        }
        tmem_load_fence_fn();
        __syncthreads();

        if (threadIdx.x < 64) {
            int row = threadIdx.x;
            float row_max = -INFINITY;
            uint32_t global_row = q_block + row;
            
            for (int col = 0; col < 64; ++col) {
                uint32_t global_kv_idx = kv_block + col;
                if (global_row >= S_len || global_kv_idx >= S_len) {
                    smem_S[row * 64 + col] = -INFINITY;
                } else {
                    float val = smem_S[row * 64 + col] * scale;
                    smem_S[row * 64 + col] = val;
                }
                row_max = fmaxf(row_max, smem_S[row * 64 + col]);
            }

            float old_max = running_max[row];
            float new_max = fmaxf(old_max, row_max);
            running_sum_exp[row] *= expf(old_max - new_max);
            running_max[row] = new_max;

            float local_sum = 0;
            for (int col = 0; col < 64; ++col) {
                if (isinf(-smem_S[row * 64 + col])) {
                    smem_S[row * 64 + col] = 0.0f;
                } else {
                    float val = expf(smem_S[row * 64 + col] - new_max);
                    smem_S[row * 64 + col] = val;
                    local_sum += val;
                }
            }
            running_sum_exp[row] += local_sum;

            for (int col = 0; col < 64; ++col) {
                float val = (running_sum_exp[row] > 0.0f) ? (smem_S[row * 64 + col] / running_sum_exp[row]) : 0.0f;
                int swizzled_col = (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
                smem_P[row * 64 + swizzled_col] = __float2bfloat16(val);
            }
        }
        __syncthreads();

        if (threadIdx.x < 64) {
            int row = threadIdx.x;
            float2 acc_o0[32] = {0};
            float2 acc_o1[32] = {0};
            
            for (int j = 0; j < 64; j += 2) {
                float p = __bfloat162float(smem_P[(row << 6) | (((j >> 3) ^ (row & 7)) << 3) | (j & 7)]);
                float p1 = __bfloat162float(smem_P[(row << 6) | ((((j+1) >> 3) ^ (row & 7)) << 3) | ((j+1) & 7)]);
                
                __nv_bfloat162* v0_j = (__nv_bfloat162*)&smem_V_0[j * 64];
                __nv_bfloat162* v0_j1 = (__nv_bfloat162*)&smem_V_0[(j + 1) * 64];
                __nv_bfloat162* v1_j = (__nv_bfloat162*)&smem_V_1[j * 64];
                __nv_bfloat162* v1_j1 = (__nv_bfloat162*)&smem_V_1[(j + 1) * 64];
                
                for (int d = 0; d < 32; ++d) {
                    int swizzled_d = ((d >> 2) ^ (j & 7)) << 2 | (d & 3);
                    acc_o0[d] += make_float2(p, p) * __bfloat1622float2(v0_j[swizzled_d]);
                    acc_o0[d] += make_float2(p1, p1) * __bfloat1622float2(v0_j1[swizzled_d]);
                    
                    acc_o1[d] += make_float2(p, p) * __bfloat1622float2(v1_j[swizzled_d]);
                    acc_o1[d] += make_float2(p1, p1) * __bfloat1622float2(v1_j1[swizzled_d]);
                }
            }
            
            for (int d = 0; d < 32; ++d) {
                *reinterpret_cast<__nv_bfloat162*>(&smem_Q_0[row * 64 + d * 2]) = __float22bfloat162(acc_o0[d]);
                *reinterpret_cast<__nv_bfloat162*>(&smem_Q_1[row * 64 + d * 2]) = __float22bfloat162(acc_o1[d]);
            }
        }
        __syncthreads();
    }

    auto write_O = [&](uint32_t global_row, uint32_t row) {
        if (global_row < S_len) {
            for (int half = 0; half < 2; ++half) {
                __nv_bfloat16* o_ptr = (half == 0) ? smem_Q_0 : smem_Q_1;
                for (uint32_t col = 0; col < 64; col += 4) {
                    uint32_t global_col = half * 64 + col; 
                    __nv_bfloat162 out_val = *reinterpret_cast<__nv_bfloat162*>(&o_ptr[row * 64 + col]);
                    *reinterpret_cast<__nv_bfloat162*>(&O[(bh * S_len + global_row) * 128 + global_col]) = out_val;
                }
            }
        }
    };

    for (uint32_t col = 0; col < 64; col += 4) {
        uint32_t global_row = q_block + lane_offset + lane_id;
        int row = lane_offset + lane_id;
        write_O(global_row, row);
    }
    
    if (threadIdx.x < 64) {
        int row = threadIdx.x;
        uint32_t global_row = q_block + row;
        if (global_row < S_len) {
            if (running_sum_exp[row] > 0.0f) {
                LSE[bh * S_len + global_row] = running_max[row] + logf(running_sum_exp[row]);
            } else {
                LSE[bh * S_len + global_row] = -INFINITY;
            }
        }
    }
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_S_addr, 64);
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
    
    DRV_CHECK(create_tma_4d_descriptor_2B(&tma_Q, q_ptr, D, S_len, H, B, 64, 64, 1, 1, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    DRV_CHECK(create_tma_4d_descriptor_2B(&tma_K, k_ptr, D, S_len, H, B, 64, 64, 1, 1, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    DRV_CHECK(create_tma_4d_descriptor_2B(&tma_V, v_ptr, D, S_len, H, B, 64, 64, 1, 1, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());
    
    uint32_t num_blocks_x = (S_len + 63) / 64;
    uint32_t num_blocks_y = B * H;
    
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(num_blocks_x, num_blocks_y);
    config.blockDim = dim3(128);
    config.dynamicSmemBytes = 74800; 
    config.stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 74800));
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, mha_kernel, tma_Q, tma_K, tma_V, o_ptr, lse_ptr, S_len));
    CUDA_CHECK(cudaGetLastError()); 
    CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id))));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha