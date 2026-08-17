#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, 
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

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_mbarrier_init_fn() {
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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
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

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_store_4x_fn(uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
   :: "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_no_swizzle(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)0 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool a_trans, bool b_trans) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= ((a_trans ? 1u : 0u) << 15);   
    d |= ((b_trans ? 1u : 0u) << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void umma_f16_cg1_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, bool accum) {
    uint32_t acc_val = accum ? 1 : 0;
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(acc_val));
}

__device__ __forceinline__ void umma_f16_a_tmem_fn(uint32_t tmem_d, uint32_t tmem_a, uint64_t desc_b, uint32_t idesc, bool accum) {
    uint32_t acc_val = accum ? 1 : 0;
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
        :: "r"(tmem_d), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(acc_val));
}

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a));
}

extern __shared__ __align__(128) uint8_t dynamic_smem[];

__global__ __launch_bounds__(128)
void CausalAttentionBlackwellKernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    uint16_t* __restrict__ O,
    float* __restrict__ LSE,
    int S, int H)
{
    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int bz = blockIdx.z; 

    int q_start = bx * 128;
    if (q_start >= S) return;
    int q_len = min(128, S - q_start);

    uint16_t* smem_Q = (uint16_t*)dynamic_smem;
    uint16_t* smem_K = smem_Q + 16384;
    uint16_t* smem_V = smem_K + 16384;
    
    uint64_t* mbar_Q    = (uint64_t*)(smem_V + 16384);
    uint64_t* mbar_KV   = mbar_Q + 1;
    uint64_t* mbar_umma = mbar_KV + 1;
    uint32_t* tmem_addr_ptr = (uint32_t*)(mbar_umma + 1);

    if (threadIdx.x < 32) {
        if (threadIdx.x == 0) {
            init_smem_barrier_fn(mbar_Q, 1);
            init_smem_barrier_fn(mbar_KV, 1);
            init_smem_barrier_fn(mbar_umma, 1);
            fence_mbarrier_init_fn();
        }
        tmem_alloc_fn(tmem_addr_ptr, 512); // Alloc 512 TMEM columns
    }
    __syncthreads();

    uint32_t tmem_addr = *tmem_addr_ptr;
    uint32_t tmem_S = tmem_addr;
    uint32_t tmem_P = tmem_addr + 128;
    uint32_t tmem_O = tmem_addr + 256;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768);
        int coord_y = bz * H * S + by * S + q_start;
        tma_load_2d_fn(&tma_Q, mbar_Q, smem_Q, 0, coord_y);
    }
    
    mbarrier_wait_fn(mbar_Q, 0);
    fence_async_shared_fn();

    if (q_len < 128) {
        for (int i = threadIdx.x; i < (128 - q_len) * 128; i += blockDim.x) {
            smem_Q[q_len * 128 + i] = 0;
        }
    }
    __syncthreads();

    uint32_t w = threadIdx.x / 32;
    uint32_t lane = threadIdx.x % 32;
    uint32_t my_row = w * 32 + lane;
    uint32_t global_row = q_start + my_row;

    float m_i = -INFINITY;
    float l_i = 0.0f;

    uint32_t idesc_QK = make_instr_desc_fn(128, 128, true, true);
    uint32_t idesc_PV = make_instr_desc_fn(128, 128, true, false);
    float scale = 1.0f / sqrtf(128.0f);
    
    int kv_phase = 0;
    int umma_phase = 0;
    
    for (int k_start = 0; k_start <= q_start; k_start += 128) {
        int k_len = min(128, S - k_start);
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_KV, 32768 * 2);
            int coord_y = bz * H * S + by * S + k_start;
            tma_load_2d_fn(&tma_K, mbar_KV, smem_K, 0, coord_y);
            tma_load_2d_fn(&tma_V, mbar_KV, smem_V, 0, coord_y);
        }
        
        mbarrier_wait_fn(mbar_KV, kv_phase & 1);
        fence_async_shared_fn();
        
        if (k_len < 128) {
            for (int i = threadIdx.x; i < (128 - k_len) * 128; i += blockDim.x) {
                smem_K[k_len * 128 + i] = 0;
                smem_V[k_len * 128 + i] = 0;
            }
        }
        __syncthreads();
        
        if (threadIdx.x == 0) {
            for (int k_step = 0; k_step < 8; ++k_step) {
                uint32_t q_addr = (uint32_t)__cvta_generic_to_shared(smem_Q) + k_step * 32;
                uint32_t k_addr = (uint32_t)__cvta_generic_to_shared(smem_K) + k_step * 32;
                uint64_t desc_Q = make_smem_desc_sm100_no_swizzle((void*)q_addr, 128, 256);
                uint64_t desc_K = make_smem_desc_sm100_no_swizzle((void*)k_addr, 128, 256);
                bool accum = (k_step > 0);
                umma_f16_cg1_fn(tmem_S, desc_Q, desc_K, idesc_QK, accum);
            }
            umma_commit_1sm_fn(mbar_umma);
        }
        mbarrier_wait_fn(mbar_umma, umma_phase & 1);
        tcgen05_fence_after_fn();
        umma_phase++;
        
        float row_max = -INFINITY;
        for (int c = 0; c < 128; c += 4) {
            uint32_t r[4];
            tmem_load_4x_fn(tmem_S + c, &r[0], &r[1], &r[2], &r[3]);
            tmem_load_fence_fn();
            for (int i = 0; i < 4; ++i) {
                float val = __uint_as_float(r[i]) * scale;
                int global_col = k_start + c + i;
                if (global_col > global_row || global_row >= S || global_col >= S) val = -INFINITY;
                row_max = fmaxf(row_max, val);
            }
        }
        
        float m_new = fmaxf(m_i, row_max);
        float m_new_safe = (m_new == -INFINITY) ? 0.0f : m_new;
        float exp_diff = (m_i == -INFINITY) ? 0.0f : expf(m_i - m_new_safe);
        
        int need_scale = (k_start > 0) && (exp_diff != 1.0f);
        if (__any_sync(0xffffffff, need_scale)) {
            for (int c = 0; c < 128; c += 4) {
                uint32_t o_regs[4];
                tmem_load_4x_fn(tmem_O + c, &o_regs[0], &o_regs[1], &o_regs[2], &o_regs[3]);
                tmem_load_fence_fn();
                for (int i = 0; i < 4; ++i) {
                    float o_val = __uint_as_float(o_regs[i]);
                    o_val *= exp_diff;
                    o_regs[i] = __float_as_uint(o_val);
                }
                tmem_store_4x_fn(tmem_O + c, o_regs[0], o_regs[1], o_regs[2], o_regs[3]);
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }
        
        float row_sum = 0.0f;
        for (int c = 0; c < 128; c += 8) {
            uint32_t r[8];
            tmem_load_4x_fn(tmem_S + c, &r[0], &r[1], &r[2], &r[3]);
            tmem_load_4x_fn(tmem_S + c + 4, &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            
            uint32_t p_packed[4];
            for (int i = 0; i < 4; ++i) {
                float val0 = __uint_as_float(r[2*i]) * scale;
                float val1 = __uint_as_float(r[2*i+1]) * scale;
                
                int global_col0 = k_start + c + 2*i;
                int global_col1 = k_start + c + 2*i + 1;
                
                if (global_col0 > global_row || global_row >= S || global_col0 >= S) val0 = -INFINITY;
                if (global_col1 > global_row || global_row >= S || global_col1 >= S) val1 = -INFINITY;
                
                float p0 = (val0 == -INFINITY) ? 0.0f : expf(val0 - m_new_safe);
                float p1 = (val1 == -INFINITY) ? 0.0f : expf(val1 - m_new_safe);
                
                row_sum += p0 + p1;
                
                p_packed[i] = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
            }
            
            tmem_store_4x_fn(tmem_P + (c / 2), p_packed[0], p_packed[1], p_packed[2], p_packed[3]);
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        
        l_i = l_i * exp_diff + row_sum;
        m_i = m_new;
        
        __syncthreads();
        
        if (threadIdx.x == 0) {
            for (int k_step = 0; k_step < 8; ++k_step) {
                uint32_t a_tmem = tmem_P + k_step * 8;
                uint32_t v_addr = (uint32_t)__cvta_generic_to_shared(smem_V) + k_step * 4096;
                uint64_t b_desc = make_smem_desc_sm100_no_swizzle((void*)v_addr, 2048, 128);
                bool accum_O = (k_start > 0 || k_step > 0);
                umma_f16_a_tmem_fn(tmem_O, a_tmem, b_desc, idesc_PV, accum_O);
            }
            umma_commit_1sm_fn(mbar_umma);
        }
        mbarrier_wait_fn(mbar_umma, umma_phase & 1);
        tcgen05_fence_after_fn();
        umma_phase++;
        
        kv_phase++;
        __syncthreads();
    }

    for (int c = 0; c < 128; c += 4) {
        uint32_t o_regs[4];
        tmem_load_4x_fn(tmem_O + c, &o_regs[0], &o_regs[1], &o_regs[2], &o_regs[3]);
        tmem_load_fence_fn();
        
        uint32_t out_packed[2];
        for (int i = 0; i < 4; i+=2) {
            float val0 = __uint_as_float(o_regs[i]) / l_i;
            float val1 = __uint_as_float(o_regs[i+1]) / l_i;
            out_packed[i/2] = pack_bf16_fn(__float_as_uint(val0), __float_as_uint(val1));
        }
        
        int global_col = c;
        if (global_row < S && global_col < 128) {
            uint32_t* O_ptr = (uint32_t*)(O + (bz * H * S + by * S + global_row) * 128 + global_col);
            O_ptr[0] = out_packed[0];
            O_ptr[1] = out_packed[1];
        }
    }

    if (global_row < S) {
        LSE[bz * H * S + by * S + global_row] = m_i + logf(l_i);
    }
    
    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_addr, 512);
    }
}

namespace tvm_ffi_mha {
void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);

    const uint16_t* Q_data = static_cast<const uint16_t*>(Q.data_ptr());
    const uint16_t* K_data = static_cast<const uint16_t*>(K.data_ptr());
    const uint16_t* V_data = static_cast<const uint16_t*>(V.data_ptr());
    uint16_t* O_data = static_cast<uint16_t*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V;
    create_tma_2d_descriptor_2B(&tma_Q, (void*)Q_data, 128, B * H * S, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, (void*)K_data, 128, B * H * S, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, (void*)V_data, 128, B * H * S, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    dim3 blocks((S + 127) / 128, H, B);
    dim3 threads(128);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int smem_size = 98560; 
    CUDA_CHECK(cudaFuncSetAttribute(CausalAttentionBlackwellKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    CausalAttentionBlackwellKernel<<<blocks, threads, smem_size, stream>>>(tma_Q, tma_K, tma_V, O_data, LSE_data, S, H);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

} // namespace tvm_ffi_mha