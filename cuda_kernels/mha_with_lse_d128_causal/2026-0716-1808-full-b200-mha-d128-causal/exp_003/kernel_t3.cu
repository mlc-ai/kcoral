#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <math.h>
#include <stdio.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_fmha {

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

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void commit_umma_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t layout_type) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)layout_type << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
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

__device__ __forceinline__ uint32_t make_instr_desc_fn_pv(uint32_t M, uint32_t N) {
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

__global__ __launch_bounds__(128, 1)
void fmha_4_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE,
    int64_t S, int64_t D)
{
    int row_block = blockIdx.x;
    int bh = blockIdx.y;
    int tid = threadIdx.x;

    __nv_bfloat16* ptr_O_bh = O + bh * S * D;
    float* ptr_LSE_bh = LSE + bh * S;

    int s_start = row_block * 128;

    extern __shared__ char smem_pool[];
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_pool);
    uint32_t align_offset = (16 - (smem_addr % 16)) % 16;
    char* aligned_smem = smem_pool + align_offset;

    __nv_bfloat16* Q_smem = (__nv_bfloat16*)aligned_smem;
    __nv_bfloat16* K_smem = Q_smem + 128 * 128;
    __nv_bfloat16* V_smem = K_smem + 128 * 128;
    float* O_smem_f32 = (float*)(V_smem + 128 * 128);
    __nv_bfloat16* O_smem_bf16 = (__nv_bfloat16*)(O_smem_f32 + 128 * 128);
    __nv_bfloat16* P_smem = O_smem_bf16;
    
    uint64_t* mbar = (uint64_t*)(O_smem_bf16 + 128 * 128);

    extern __shared__ __align__(16) uint32_t tmem_pool[];
    uint32_t* tmem_S_ptr = tmem_pool;
    uint32_t* tmem_O_ptr = tmem_pool + 1;
    
    if (tid == 0) {
        init_smem_barrier_fn(mbar, 1);
        fence_smem_barrier_init_fn();
        tmem_alloc_fn(tmem_S_ptr, 128);
        tmem_alloc_fn(tmem_O_ptr, 128);
    }
    __syncthreads();
    
    uint32_t tmem_S = tmem_S_ptr[0];
    uint32_t tmem_O = tmem_O_ptr[0];

    float m_prev_row[128];
    float l_prev_row[128];
    for (int i = tid; i < 128; i += 128) {
        m_prev_row[i] = -1e20f;
        l_prev_row[i] = 0.0f;
    }
    for (int i = tid; i < 128 * 128; i += 128) {
        O_smem_f32[i] = 0.0f;
    }
    __syncthreads();
    
    uint32_t phase = 0;

    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, 128 * 128 * sizeof(__nv_bfloat16));
        tma_load_3d_fn(&tma_Q, mbar, Q_smem, 0, s_start, bh);
    }
    mbarrier_wait_fn(mbar, phase);
    phase ^= 1;

    float scale = 1.0f / sqrtf((float)D);

    for (int j = 0; j <= row_block; ++j) {
        int kv_start = j * 128;

        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 2 * 128 * 128 * sizeof(__nv_bfloat16));
            tma_load_3d_fn(&tma_K, mbar, K_smem, 0, kv_start, bh);
            
            tma_load_3d_fn(&tma_V, mbar, V_smem, 0, kv_start, bh);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        uint32_t col_offset = 0;
        for (int k = 0; k < 128; k += 16) {
            uint64_t desc_Q_k = make_smem_desc_sm100_fn(Q_smem + k, 2048, 128, 0);
            uint64_t desc_K_k = make_smem_desc_sm100_fn(K_smem + k, 2048, 128, 0);
            uint32_t idesc_QK = make_instr_desc_fn(128, 128);
            umma_f16_cg1_fn(tmem_S + col_offset, desc_Q_k, desc_K_k, idesc_QK, k == 0 ? 0 : 1);
            col_offset += 16;
        }
        
        commit_umma_1sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        float thread_max[4];
        thread_max[0] = -1e20f;
        thread_max[1] = -1e20f;
        thread_max[2] = -1e20f;
        thread_max[3] = -1e20f;
        
        float S_reg[4][8]; 
        
        uint32_t col_base = 0;
        do {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),
                 "=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_S + col_base));
            
            int row_idx = tid / 32;
            int lane_idx = tid % 32;
            int col_idx = col_base / 8;
            
            int r_idx0 = (lane_idx + 0 * 8) / 16;
            int c_idx0 = (lane_idx + 0 * 8) % 16;
            int r_idx1 = (lane_idx + 1 * 8) / 16;
            int c_idx1 = (lane_idx + 1 * 8) % 16;
            int r_idx2 = (lane_idx + 2 * 8) / 16;
            int c_idx2 = (lane_idx + 2 * 8) % 16;
            int r_idx3 = (lane_idx + 3 * 8) / 16;
            int c_idx3 = (lane_idx + 3 * 8) % 16;
            
            S_reg[r_idx0][0] = __uint_as_float(r0) * scale;
            S_reg[r_idx0][1] = __uint_as_float(r1) * scale;
            S_reg[r_idx0][2] = __uint_as_float(r2) * scale;
            S_reg[r_idx0][3] = __uint_as_float(r3) * scale;
            S_reg[r_idx0][4] = __uint_as_float(r4) * scale;
            S_reg[r_idx0][5] = __uint_as_float(r5) * scale;
            S_reg[r_idx0][6] = __uint_as_float(r6) * scale;
            S_reg[r_idx0][7] = __uint_as_float(r7) * scale;
            
            S_reg[r_idx1][0] = __uint_as_float(r0) * scale;
            S_reg[r_idx1][1] = __uint_as_float(r1) * scale;
            S_reg[r_idx1][2] = __uint_as_float(r2) * scale;
            S_reg[r_idx1][3] = __uint_as_float(r3) * scale;
            S_reg[r_idx1][4] = __uint_as_float(r4) * scale;
            S_reg[r_idx1][5] = __uint_as_float(r5) * scale;
            S_reg[r_idx1][6] = __uint_as_float(r6) * scale;
            S_reg[r_idx1][7] = __uint_as_float(r7) * scale;
            
            S_reg[r_idx2][0] = __uint_as_float(r0) * scale;
            S_reg[r_idx2][1] = __uint_as_float(r1) * scale;
            S_reg[r_idx2][2] = __uint_as_float(r2) * scale;
            S_reg[r_idx2][3] = __uint_as_float(r3) * scale;
            S_reg[r_idx2][4] = __uint_as_float(r4) * scale;
            S_reg[r_idx2][5] = __uint_as_float(r5) * scale;
            S_reg[r_idx2][6] = __uint_as_float(r6) * scale;
            S_reg[r_idx2][7] = __uint_as_float(r7) * scale;
            
            S_reg[r_idx3][0] = __uint_as_float(r0) * scale;
            S_reg[r_idx3][1] = __uint_as_float(r1) * scale;
            S_reg[r_idx3][2] = __uint_as_float(r2) * scale;
            S_reg[r_idx3][3] = __uint_as_float(r3) * scale;
            S_reg[r_idx3][4] = __uint_as_float(r4) * scale;
            S_reg[r_idx3][5] = __uint_as_float(r5) * scale;
            S_reg[r_idx3][6] = __uint_as_float(r6) * scale;
            S_reg[r_idx3][7] = __uint_as_float(r7) * scale;
            
            col_base += 8;
        } while (col_base < 128);
        
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

        for(int r = 0; r < 4; ++r) {
            int global_row = s_start + tid / 32 + r * 16;
            for (int c = 0; c < 8; ++c) {
                int col = col_base - 128 + c; // Adjusted since col_base is 128 here
                int global_col = kv_start + col;
                if (global_col > global_row || global_col >= S) {
                    S_reg[r][c] = -1e20f;
                }
                thread_max[r] = fmaxf(thread_max[r], S_reg[r][c]);
            }
        }

        for(int r = 0; r < 4; ++r) {
            for (int i = 1; i < 16; i *= 2) {
                thread_max[r] = fmaxf(thread_max[r], __shfl_xor_sync(0xffffffff, thread_max[r], i));
            }
        }
        
        float m_new_row[4];
        float temp_l_row[4];
        for(int r = 0; r < 4; ++r) {
            m_new_row[r] = fmaxf(m_prev_row[tid/32 + r*16], thread_max[r]);
            temp_l_row[r] = 0.0f;
        }
        
        for(int r = 0; r < 4; ++r) {
            for (int c = 0; c < 8; ++c) {
                float p = (S_reg[r][c] <= -1e20f) ? 0.0f : __expf(S_reg[r][c] - m_new_row[r]);
                temp_l_row[r] += p;
                S_reg[r][c] = p;
            }
        }
        
        for(int r = 0; r < 4; ++r) {
            for (int i = 1; i < 16; i *= 2) {
                temp_l_row[r] += __shfl_xor_sync(0xffffffff, temp_l_row[r], i);
            }
        }
        
        float factor_row[4];
        for(int r = 0; r < 4; ++r) {
            if (m_prev_row[tid/32 + r*16] <= -1e19f) {
                m_prev_row[tid/32 + r*16] = m_new_row[r];
                l_prev_row[tid/32 + r*16] = temp_l_row[r];
                factor_row[r] = 1.0f;
            } else {
                factor_row[r] = __expf(m_prev_row[tid/32 + r*16] - m_new_row[r]);
                m_prev_row[tid/32 + r*16] = m_new_row[r];
                l_prev_row[tid/32 + r*16] = l_prev_row[tid/32 + r*16] * factor_row[r] + temp_l_row[r];
            }
        }
        
        for(int r = 0; r < 4; ++r) {
            for (int c = 0; c < 8; ++c) {
                S_reg[r][c] *= factor_row[r];
            }
        }
        
        for (int i = tid; i < 128 * 128; i += 128) {
            int row = i / 128;
            O_smem_f32[i] *= factor_row[row];
        }
        
        for(int r = 0; r < 4; ++r) {
            int base_r = tid / 32 + r * 16;
            for (int c = 0; c < 8; ++c) {
                int col = col_base - 128 + c;
                P_smem[base_r * 128 + col] = __float2bfloat16(S_reg[r][c]);
            }
        }
        
        __syncthreads(); 
        
        uint32_t row_offset = 0;
        for (int k = 0; k < 128; k += 16) {
            uint64_t desc_P_k = make_smem_desc_sm100_fn(P_smem + k, 2048, 128, 0);
            uint64_t desc_V_k = make_smem_desc_sm100_fn(V_smem + k * 128, 128, 2048, 0);
            uint32_t idesc_PV = make_instr_desc_fn_pv(128, 128);
            umma_f16_cg1_fn(tmem_O + row_offset, desc_P_k, desc_V_k, idesc_PV, k == 0 ? 0 : 1);
            row_offset += 16;
        }
        
        commit_umma_1sm_fn(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        for (int i = tid; i < 128 * 128; i += 128) {
            int col = i % 128;
            int r_idx = col / 16;
            int c_idx = col % 16;
            uint32_t tmp_O_reg = 0;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
                : "=r"(tmp_O_reg) : "r"(tmem_O + i / 2)); 
            O_smem_f32[i] += __uint_as_float(tmp_O_reg);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        __syncthreads();
    }

    __syncthreads();
    
    for (int i = tid; i < 128 * 128; i += 128) {
        int row = i / 128;
        int col = i % 128;
        float val = O_smem_f32[i];
        if (l_prev_row[row] > 0.0f) {
            val /= l_prev_row[row];
        }
        O_smem_bf16[row * 128 + col] = __float2bfloat16(val);
    }
    __syncthreads();
    
    uint32_t warp_id = tid / 32;
    uint32_t lane_id = tid % 32;
    uint32_t num_steps = (128 + 3) / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= 128) continue;
        uint32_t global_row = s_start + row;
        uint32_t col_start = lane_id * 4;
        uint32_t global_col = col_start;
        if (global_row < S && global_col + 3 < 128) {
            uint2 data = *reinterpret_cast<uint2*>(&O_smem_bf16[row * 128 + col_start]);
            *reinterpret_cast<uint2*>(ptr_O_bh + (uint64_t)global_row * D + global_col) = data;
        }
    }
    
    if (tid < 128 && s_start + tid < S) {
        ptr_LSE_bh[s_start + tid] = m_prev_row[tid] + logf(l_prev_row[tid]);
    }

    if (tid == 0) {
        tmem_dealloc_fn(tmem_S, 128);
        tmem_dealloc_fn(tmem_O, 128);
    }
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_dim0, uint64_t gmem_dim1, uint64_t gmem_dim2, uint32_t smem_dim0, uint32_t smem_dim1, uint32_t smem_dim2, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[3] = {gmem_dim0, gmem_dim1, gmem_dim2};
    cuuint64_t globalStrides[2] = {gmem_dim0 * 2, gmem_dim0 * gmem_dim1 * 2};
    cuuint32_t boxDim[3] = {smem_dim0, smem_dim1, smem_dim2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3,
        globalAddress,
        globalDim,
        globalStrides,
        boxDim, 
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle,
        l2Promotion,
        oobFill
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V;
    uint64_t gmem_outer_dim = B * H;
    
    CUresult res_q = create_tma_3d_descriptor_2B(&tma_Q, (void*)Q_ptr, D, S, gmem_outer_dim, 128, 128, 1, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUresult res_k = create_tma_3d_descriptor_2B(&tma_K, (void*)K_ptr, D, S, gmem_outer_dim, 128, 128, 1, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUresult res_v = create_tma_3d_descriptor_2B(&tma_V, (void*)V_ptr, D, S, gmem_outer_dim, 128, 128, 1, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    if (res_q != CUDA_SUCCESS || res_k != CUDA_SUCCESS || res_v != CUDA_SUCCESS) {
        fprintf(stderr, "TMA descriptor creation failed!\n");
        exit(1);
    }
    
    int64_t num_row_blocks = (S + 127) / 128;
    dim3 grid(num_row_blocks, B * H);
    dim3 block(128);
    
    int smem_size = 16 + 4 * 128 * 128 * sizeof(__nv_bfloat16) + 128 * 128 * sizeof(float) + 8 + 8 + 128;
                    
    CUDA_CHECK(cudaFuncSetAttribute(
        fmha_4_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    ));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    fmha_4_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, S, D
    );
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_fmha