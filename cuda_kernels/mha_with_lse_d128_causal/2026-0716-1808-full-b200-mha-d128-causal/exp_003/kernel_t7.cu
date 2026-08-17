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

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
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

__device__ __forceinline__ void fence_proxy_async_shared_fn() {
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

__device__ __forceinline__ void commit_umma_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ void umma_f16_cg1_init(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, 0, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc));
}

__device__ __forceinline__ void umma_f16_cg1_acc(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, 1, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc));
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

    setmaxnreg_inc_sync_fn<248>();

    __nv_bfloat16* ptr_O_bh = O + bh * S * D;
    float* ptr_LSE_bh = LSE + bh * S;

    int s_start = row_block * 128;

    extern __shared__ char smem_pool[];
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_pool);
    uint32_t align_offset = (1024 - (smem_addr % 1024)) % 1024;
    char* aligned_smem = smem_pool + align_offset;

    __nv_bfloat16* Q_smem = (__nv_bfloat16*)aligned_smem;
    __nv_bfloat16* K_smem = Q_smem + 128 * 128;
    __nv_bfloat16* V_smem = K_smem + 128 * 128;
    __nv_bfloat16* P_smem = V_smem + 128 * 128;
    float* smem_S = (float*)(P_smem + 128 * 128);
    
    uint64_t* mbar = (uint64_t*)(smem_S + 128 * 128);

    extern __shared__ __align__(16) uint32_t tmem_pool[];
    uint32_t* tmem_S_ptr = tmem_pool;
    uint32_t* tmem_O_ptr = tmem_pool + 1;
    uint32_t* tmem_P_ptr = tmem_pool + 2;
    
    if (tid == 0) {
        init_smem_barrier_fn(mbar, 1);
        fence_smem_barrier_init_fn();
        tmem_alloc_fn(tmem_S_ptr, 128);
        tmem_alloc_fn(tmem_O_ptr, 128);
        tmem_alloc_fn(tmem_P_ptr, 128);
    }
    __syncthreads();
    
    uint32_t tmem_S = tmem_S_ptr[0];
    uint32_t tmem_O = tmem_O_ptr[0];
    uint32_t tmem_P = tmem_P_ptr[0];

    float m_prev_row[128];
    float l_prev_row[128];
    for (int i = tid; i < 128; i += 128) {
        m_prev_row[i] = -1e20f;
        l_prev_row[i] = 0.0f;
    }
    __syncthreads();
    
    uint32_t phase = 0;

    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, 2 * 64 * 128 * sizeof(__nv_bfloat16));
        tma_load_3d_fn(&tma_Q, mbar, Q_smem, 0, s_start, bh);
        tma_load_3d_fn(&tma_Q, mbar, Q_smem + 8192, 64, s_start, bh);
    }
    mbarrier_wait_fn(mbar, phase);
    phase ^= 1;

    if (tid == 0) {
        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0;
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};\n"
                :: "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(tmem_O + col));
        }
        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0 = 0, r1 = 0, r2 = 0, r3 = 0;
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};\n"
                :: "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(tmem_O + 64 + col));
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }
    __syncthreads();

    float scale = 1.0f / sqrtf((float)D);

    for (int j = 0; j <= row_block; ++j) {
        int kv_start = j * 128;

        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 4 * 64 * 128 * sizeof(__nv_bfloat16));
            tma_load_3d_fn(&tma_K, mbar, K_smem, 0, kv_start, bh);
            tma_load_3d_fn(&tma_K, mbar, K_smem + 8192, 64, kv_start, bh);
            
            tma_load_3d_fn(&tma_V, mbar, V_smem, 0, kv_start, bh);
            tma_load_3d_fn(&tma_V, mbar, V_smem + 8192, 64, kv_start, bh);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        for(int k = 0; k < 128; k += 16) { 
            __nv_bfloat16* q_ptr = Q_smem + k;
            __nv_bfloat16* k_ptr = K_smem + k;
            
            uint64_t desc_Q_k = make_smem_desc_sm100_fn(q_ptr, 1, 1024, 2);
            uint64_t desc_K_k = make_smem_desc_sm100_fn(k_ptr, 1, 1024, 2);
            uint32_t idesc_QK = make_instr_desc_fn(128, 128);
            
            if (k == 0) {
                umma_f16_cg1_init(tmem_S, desc_Q_k, desc_K_k, idesc_QK);
            } else {
                umma_f16_cg1_acc(tmem_S, desc_Q_k, desc_K_k, idesc_QK);
            }
        }
        
        if (tid == 0) {
            commit_umma_1sm_fn(mbar);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_S + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            smem_S[tid * 128 + col] = __uint_as_float(r0);
            smem_S[tid * 128 + col + 1] = __uint_as_float(r1);
            smem_S[tid * 128 + col + 2] = __uint_as_float(r2);
            smem_S[tid * 128 + col + 3] = __uint_as_float(r3);
        }
        __syncthreads();

        float thread_max = -1e20f;
        
        for (int i = 0; i < 128; i += 4) {
            float val0 = smem_S[tid * 128 + i] * scale;
            float val1 = smem_S[tid * 128 + i + 1] * scale;
            float val2 = smem_S[tid * 128 + i + 2] * scale;
            float val3 = smem_S[tid * 128 + i + 3] * scale;
            
            int global_row = s_start + tid;
            int global_col0 = kv_start + i;
            int global_col1 = kv_start + i + 1;
            int global_col2 = kv_start + i + 2;
            int global_col3 = kv_start + i + 3;
            
            if (global_col0 > global_row || global_col0 >= S) val0 = -1e20f;
            if (global_col1 > global_row || global_col1 >= S) val1 = -1e20f;
            if (global_col2 > global_row || global_col2 >= S) val2 = -1e20f;
            if (global_col3 > global_row || global_col3 >= S) val3 = -1e20f;
            
            thread_max = fmaxf(thread_max, fmaxf(val0, fmaxf(val1, fmaxf(val2, val3))));
            
            smem_S[tid * 128 + i] = val0;
            smem_S[tid * 128 + i + 1] = val1;
            smem_S[tid * 128 + i + 2] = val2;
            smem_S[tid * 128 + i + 3] = val3;
        }

        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            thread_max = fmaxf(thread_max, __shfl_xor_sync(0xffffffff, thread_max, offset));
        }

        float m_new = fmaxf(m_prev_row[tid], thread_max);
        float temp_l = 0.0f;
        
        for (int i = 0; i < 128; i += 4) {
            float val0 = smem_S[tid * 128 + i];
            float val1 = smem_S[tid * 128 + i + 1];
            float val2 = smem_S[tid * 128 + i + 2];
            float val3 = smem_S[tid * 128 + i + 3];
            
            float p0 = (val0 <= -1e20f) ? 0.0f : __expf(val0 - m_new);
            float p1 = (val1 <= -1e20f) ? 0.0f : __expf(val1 - m_new);
            float p2 = (val2 <= -1e20f) ? 0.0f : __expf(val2 - m_new);
            float p3 = (val3 <= -1e20f) ? 0.0f : __expf(val3 - m_new);
            
            temp_l += p0 + p1 + p2 + p3;
            
            smem_S[tid * 128 + i] = p0;
            smem_S[tid * 128 + i + 1] = p1;
            smem_S[tid * 128 + i + 2] = p2;
            smem_S[tid * 128 + i + 3] = p3;
        }

        #pragma unroll
        for (int offset = 4; offset > 0; offset /= 2) {
            temp_l += __shfl_xor_sync(0xffffffff, temp_l, offset);
        }

        float factor = 1.0f;
        if (m_prev_row[tid] <= -1e19f) {
            m_prev_row[tid] = m_new;
            l_prev_row[tid] = temp_l;
        } else {
            factor = __expf(m_prev_row[tid] - m_new);
            m_prev_row[tid] = m_new;
            l_prev_row[tid] = l_prev_row[tid] * factor + temp_l;
        }

        if (factor != 1.0f) {
            for(uint32_t col = 0; col < 128; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_O + col));
                float f0 = __uint_as_float(r0) * factor;
                float f1 = __uint_as_float(r1) * factor;
                float f2 = __uint_as_float(r2) * factor;
                float f3 = __uint_as_float(r3) * factor;
                uint32_t o0 = __float_as_uint(f0);
                uint32_t o1 = __float_as_uint(f1);
                uint32_t o2 = __float_as_uint(f2);
                uint32_t o3 = __float_as_uint(f3);
                asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};\n"
                    :: "r"(o0), "r"(o1), "r"(o2), "r"(o3), "r"(tmem_O + col));
            }
            for(uint32_t col = 0; col < 128; col += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_O + 64 + col));
                float f0 = __uint_as_float(r0) * factor;
                float f1 = __uint_as_float(r1) * factor;
                float f2 = __uint_as_float(r2) * factor;
                float f3 = __uint_as_float(r3) * factor;
                uint32_t o0 = __float_as_uint(f0);
                uint32_t o1 = __float_as_uint(f1);
                uint32_t o2 = __float_as_uint(f2);
                uint32_t o3 = __float_as_uint(f3);
                asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};\n"
                    :: "r"(o0), "r"(o1), "r"(o2), "r"(o3), "r"(tmem_O + 64 + col));
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }
        
        for (int i = 0; i < 128; i += 4) {
            float p0 = smem_S[tid * 128 + i] * factor;
            float p1 = smem_S[tid * 128 + i + 1] * factor;
            float p2 = smem_S[tid * 128 + i + 2] * factor;
            float p3 = smem_S[tid * 128 + i + 3] * factor;
            
            int col = i;
            int chunk = col / 64;
            int subchunk = (col % 64) / 8;
            int element = col % 8;
            int swizzled_subchunk = (tid % 8) ^ subchunk;
            int swizzled_col = chunk * 64 + swizzled_subchunk * 8 + element;
            P_smem[tid * 128 + swizzled_col] = __float2bfloat16(p0);
            
            col = i + 1;
            chunk = col / 64;
            subchunk = (col % 64) / 8;
            element = col % 8;
            swizzled_subchunk = (tid % 8) ^ subchunk;
            swizzled_col = chunk * 64 + swizzled_subchunk * 8 + element;
            P_smem[tid * 128 + swizzled_col] = __float2bfloat16(p1);
            
            col = i + 2;
            chunk = col / 64;
            subchunk = (col % 64) / 8;
            element = col % 8;
            swizzled_subchunk = (tid % 8) ^ subchunk;
            swizzled_col = chunk * 64 + swizzled_subchunk * 8 + element;
            P_smem[tid * 128 + swizzled_col] = __float2bfloat16(p2);
            
            col = i + 3;
            chunk = col / 64;
            subchunk = (col % 64) / 8;
            element = col % 8;
            swizzled_subchunk = (tid % 8) ^ subchunk;
            swizzled_col = chunk * 64 + swizzled_subchunk * 8 + element;
            P_smem[tid * 128 + swizzled_col] = __float2bfloat16(p3);
        }
        
        fence_proxy_async_shared_fn();
        
        for(int k = 0; k < 128; k += 16) { 
            for(int n = 0; n < 2; n++) { 
                __nv_bfloat16* p_ptr = P_smem + k;
                __nv_bfloat16* v_ptr = V_smem + k * 128 + n * 64;
                
                uint64_t desc_P_k = make_smem_desc_sm100_fn(p_ptr, 1, 1024, 2);
                uint64_t desc_V_k = make_smem_desc_sm100_fn(v_ptr, 16384, 1024, 2);
                uint32_t idesc_PV = make_instr_desc_fn_pv(128, 128);
                
                if (j == 0 && k == 0 && n == 0) {
                    umma_f16_cg1_init(tmem_O + n * 64, desc_P_k, desc_V_k, idesc_PV);
                } else {
                    umma_f16_cg1_acc(tmem_O + n * 64, desc_P_k, desc_V_k, idesc_PV);
                }
            }
        }
        
        if (tid == 0) {
            commit_umma_1sm_fn(mbar);
        }
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        __syncthreads();
    }

    __syncthreads();
    
    __nv_bfloat16* smem_out = (__nv_bfloat16*)aligned_smem;
    
    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
           : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_O + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        
        if (l_prev_row[tid] > 0.0f) {
            f0 /= l_prev_row[tid];
            f1 /= l_prev_row[tid];
            f2 /= l_prev_row[tid];
            f3 /= l_prev_row[tid];
        }
        
        uint32_t base = tid * 128 + col;
        smem_out[base] = __float2bfloat16(f0);
        smem_out[base + 1] = __float2bfloat16(f1);
        smem_out[base + 2] = __float2bfloat16(f2);
        smem_out[base + 3] = __float2bfloat16(f3);
    }
    
    for (uint32_t col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
           : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_O + 64 + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        
        if (l_prev_row[tid] > 0.0f) {
            f0 /= l_prev_row[tid];
            f1 /= l_prev_row[tid];
            f2 /= l_prev_row[tid];
            f3 /= l_prev_row[tid];
        }
        
        uint32_t base = tid * 128 + col + 64;
        smem_out[base] = __float2bfloat16(f0);
        smem_out[base + 1] = __float2bfloat16(f1);
        smem_out[base + 2] = __float2bfloat16(f2);
        smem_out[base + 3] = __float2bfloat16(f3);
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
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * 128 + col_start]);
            *reinterpret_cast<uint2*>(ptr_O_bh + (uint64_t)global_row * D + global_col) = data;
        }
    }
    
    if (tid < 128 && s_start + tid < S) {
        ptr_LSE_bh[s_start + tid] = m_prev_row[tid] + logf(l_prev_row[tid]);
    }

    if (tid == 0) {
        tmem_dealloc_fn(tmem_S, 128);
        tmem_dealloc_fn(tmem_O, 128);
        tmem_dealloc_fn(tmem_P, 128);
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
    
    CUresult res_q = create_tma_3d_descriptor_2B(&tma_Q, (void*)Q_ptr, D, S, gmem_outer_dim, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUresult res_k = create_tma_3d_descriptor_2B(&tma_K, (void*)K_ptr, D, S, gmem_outer_dim, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    CUresult res_v = create_tma_3d_descriptor_2B(&tma_V, (void*)V_ptr, D, S, gmem_outer_dim, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    if (res_q != CUDA_SUCCESS || res_k != CUDA_SUCCESS || res_v != CUDA_SUCCESS) {
        fprintf(stderr, "TMA descriptor creation failed!\n");
        exit(1);
    }
    
    int64_t num_row_blocks = (S + 127) / 128;
    dim3 grid(num_row_blocks, B * H);
    dim3 block(128);
    
    int smem_size = 196608 + 1024;
                    
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