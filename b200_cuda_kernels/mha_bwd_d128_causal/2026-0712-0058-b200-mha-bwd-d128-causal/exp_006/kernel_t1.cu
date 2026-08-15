#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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

__device__ __forceinline__ int swizzle_128B(int row, int col) {
    return ((row % 8) ^ (col / 8)) * 8 + (col % 8);
}

__device__ __forceinline__ void load_tile_swizzled(const __nv_bfloat16* gmem, __nv_bfloat16* smem, int S, int global_i) {
    float4 gmem_vals[16];
    for (int i = 0; i < 16; ++i) {
        int col = i * 8;
        int row = threadIdx.x;
        int g_row = global_i + row;
        if (g_row < S && col < 128) {
            gmem_vals[i] = *(const float4*)&gmem[g_row * 128 + col];
        } else {
            gmem_vals[i] = {0, 0, 0, 0};
        }
    }
    for (int i = 0; i < 16; ++i) {
        int col = i * 8;
        int row = threadIdx.x;
        int g_row = global_i + row;
        int sc = swizzle_128B(row, col);
        if (g_row < S && col < 128) {
            *(float4*)&smem[row * 128 + sc] = gmem_vals[i];
        } else {
            *(float4*)&smem[row * 128 + sc] = {0, 0, 0, 0};
        }
    }
}

__device__ __forceinline__ void store_tile_swizzled(__nv_bfloat16* gmem, const __nv_bfloat16* smem, int S, int global_i) {
    for (int col = threadIdx.x; col < 128; col += 128) {
        int row = 0; // Stored linearly row by row from the chunk base
        int sc = swizzle_128B(row, col);
        int g_row = global_i + row;
        if (g_row < S && col < 128) {
            gmem[g_row * 128 + col] = smem[row * 128 + sc];
        }
    }
    for (int col = threadIdx.x % 128; col < 128; col += 128) {
        int row = threadIdx.x / 128;
        int sc = swizzle_128B(row, col);
        int g_row = global_i + row;
        if (g_row < S && col < 128) {
            gmem[g_row * 128 + col] = smem[row * 128 + sc];
        }
    }
}

__device__ __forceinline__ void atomicAdd_bf16(__nv_bfloat16* address, __nv_bfloat16 val) {
    float val_f = __bfloat162float(val);
    int32_t* addr_i = (int32_t*)((uintptr_t)address & ~1);
    bool is_odd = ((uintptr_t)address) & 1;
    
    while (true) {
        int32_t old = *addr_i;
        uint16_t old_bf = is_odd ? (old >> 16) : (old & 0xFFFF);
        float old_f = __bfloat162float(*(reinterpret_cast<__nv_bfloat16*>(&old_bf)));
        float new_f = old_f + val_f;
        __nv_bfloat16 new_bf16 = __float2bfloat16(new_f);
        uint16_t new_bf = *(reinterpret_cast<uint16_t*>(&new_bf16));
        
        int32_t new_val = is_odd ? (old & 0xFFFF) | (new_bf << 16) : (old & 0xFFFF0000) | new_bf;
        int32_t replaced = atomicCAS(addr_i, old, new_val);
        if (replaced == old) break;
    }
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
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

__device__ __forceinline__ void commit_mbarrier(uint64_t* bar) {
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
}

__device__ __forceinline__ void tmem_alloc_cta(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cta(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void umma_cg1(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint64_t d = 0;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major(void* smem_ptr) {
    uint32_t lbo = 1;
    uint32_t sbo = 1024;
    return make_smem_desc(smem_ptr, lbo, sbo);
}

__device__ __forceinline__ uint64_t make_smem_desc_n_major(void* smem_ptr) {
    uint32_t lbo = 128;
    uint32_t sbo = 1024;
    return make_smem_desc(smem_ptr, lbo, sbo);
}

__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N, int a_major, int b_major) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= ((uint32_t)a_major << 15);
    d |= ((uint32_t)b_major << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void gemm_kk(uint32_t tmem_base, uint64_t desc_a_base, uint64_t desc_b_base, bool accumulate) {
    for (int k = 0; k < 8; k++) {
        uint32_t idesc = make_instr_desc(128, 128, 0, 0);
        uint32_t acc = (k == 0 && !accumulate) ? 0 : 1;
        umma_cg1(tmem_base + (k * 2), desc_a_base + (k * 2), desc_b_base + (k * 2), idesc, acc);
    }
}

__device__ __forceinline__ void gemm_kn(uint32_t tmem_base, uint64_t desc_a_base, uint64_t desc_b_base, bool accumulate) {
    for (int k = 0; k < 8; k++) {
        uint32_t idesc = make_instr_desc(128, 128, 0, 1);
        uint32_t acc = (k == 0 && !accumulate) ? 0 : 1;
        umma_cg1(tmem_base + (k * 2), desc_a_base + (k * 2), desc_b_base + (k * 16), idesc, acc);
    }
}

__global__ __launch_bounds__(128)
void bwd_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ O,
    const __nv_bfloat16* __restrict__ dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int S,
    float attn_scale,
    uint64_t batch_head_stride)
{
    int num_j = (S + 127) / 128;
    int j_idx = blockIdx.x / num_j;
    int global_j = (blockIdx.x % num_j) * 128;

    if (global_j >= S) return;

    const __nv_bfloat16* q_ptr = Q + j_idx * S * 128;
    const __nv_bfloat16* k_ptr = K + j_idx * S * 128;
    const __nv_bfloat16* v_ptr = V + j_idx * S * 128;
    const __nv_bfloat16* o_ptr = O + j_idx * S * 128;
    const __nv_bfloat16* do_ptr = dO + j_idx * S * 128;
    
    __nv_bfloat16* dq_ptr = dQ + j_idx * S * 128;
    __nv_bfloat16* dk_ptr = dK + j_idx * S * 128;
    __nv_bfloat16* dv_ptr = dV + j_idx * S * 128;
    
    const float* l_ptr = L + j_idx * batch_head_stride;

    extern __shared__ __align__(1024) char smem_pool[];
    __nv_bfloat16* s_k = (__nv_bfloat16*)(smem_pool);                
    __nv_bfloat16* s_v = (__nv_bfloat16*)(smem_pool + 32768);        
    __nv_bfloat16* s_q = (__nv_bfloat16*)(smem_pool + 65536);        
    __nv_bfloat16* s_o = (__nv_bfloat16*)(smem_pool + 98304);        
    __nv_bfloat16* s_do = (__nv_bfloat16*)(smem_pool + 131072);      
    __nv_bfloat16* s_s = (__nv_bfloat16*)(smem_pool + 163840);       
    __nv_bfloat16* s_p = (__nv_bfloat16*)(smem_pool + 196608);       
    __nv_bfloat16* s_ds = (__nv_bfloat16*)(smem_pool + 229376);      
    float* s_d = (float*)(smem_pool + 262144);                       
    float* s_l = (float*)(smem_pool + 262656);                       
    uint64_t* mbar = (uint64_t*)(smem_pool + 263168);               

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t tmem_s, tmem_dp, tmem_dv, tmem_dk;
    if (threadIdx.x == 0) {
        tmem_alloc_cta(&tmem_s, 128);
        tmem_alloc_cta(&tmem_dp, 128);
        tmem_alloc_cta(&tmem_dv, 128);
        tmem_alloc_cta(&tmem_dk, 128);
    }
    __syncthreads();

    load_tile_swizzled(k_ptr, s_k, S, global_j);
    load_tile_swizzled(v_ptr, s_v, S, global_j);
    
    uint32_t phase = 0;
    bool is_first_dk = true;
    bool is_first_dv = true;

    for (int global_i = global_j; global_i < S; global_i += 128) {
        load_tile_swizzled(q_ptr, s_q, S, global_i);
        load_tile_swizzled(o_ptr, s_o, S, global_i);
        load_tile_swizzled(do_ptr, s_do, S, global_i);
        
        if (global_i + threadIdx.x < S) {
            s_l[threadIdx.x] = l_ptr[global_i + threadIdx.x];
        } else {
            s_l[threadIdx.x] = 0.0f;
        }
        
        float d_val = 0;
        for (int c = 0; c < 128; ++c) {
            int sc = swizzle_128B(threadIdx.x, c);
            d_val += __bfloat162float(s_o[threadIdx.x * 128 + sc]) * 
                     __bfloat162float(s_do[threadIdx.x * 128 + sc]);
        }
        if (threadIdx.x < 128) {
            s_d[threadIdx.x] = d_val;
        }
        __syncthreads(); 
        
        uint32_t desc_k = make_smem_desc_k_major(s_k);
        uint32_t desc_q = make_smem_desc_k_major(s_q);
        
        asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %0, 0;\n}" :: "r"(1));
        gemm_kk(tmem_s, desc_k, desc_q, true);
        commit_mbarrier(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        for (int col = threadIdx.x; col < 128; col += 128) {
            int sc = swizzle_128B(threadIdx.x, col);
            float s_val = __bfloat162float(s_s[threadIdx.x * 128 + sc]);
            int g_i = global_i + threadIdx.x;
            int g_j = global_j + col;
            float p_val = 0;
            if (g_j <= g_i && g_j < S) {
                p_val = expf(s_val * attn_scale - s_l[threadIdx.x]);
            }
            s_p[threadIdx.x * 128 + sc] = __float2bfloat16(p_val);
        }
        __syncthreads();
        
        uint32_t desc_v = make_smem_desc_k_major(s_v);
        uint32_t desc_do = make_smem_desc_k_major(s_do);
        
        asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %0, 0;\n}" :: "r"(1));
        gemm_kk(tmem_dp, desc_v, desc_do, true);
        commit_mbarrier(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        for (int col = threadIdx.x; col < 128; col += 128) {
            int sc = swizzle_128B(threadIdx.x, col);
            float dp_val = __bfloat162float(s_ds[threadIdx.x * 128 + sc]);
            float p_val = __bfloat162float(s_p[threadIdx.x * 128 + sc]);
            float ds_val = p_val * (dp_val - s_d[threadIdx.x]);
            s_ds[threadIdx.x * 128 + sc] = __float2bfloat16(ds_val);
        }
        __syncthreads();
        
        uint32_t desc_p = make_smem_desc_k_major(s_p);
        uint32_t desc_do_nm = make_smem_desc_n_major(s_do);
        
        asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %0, 0;\n}" :: "r"(is_first_dv));
        gemm_kn(tmem_dv, desc_p, desc_do_nm, is_first_dv);
        is_first_dv = false;
        commit_mbarrier(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        uint32_t desc_ds = make_smem_desc_k_major(s_ds);
        uint32_t desc_q_nm = make_smem_desc_n_major(s_q);
        
        asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %0, 0;\n}" :: "r"(is_first_dk));
        gemm_kn(tmem_dk, desc_ds, desc_q_nm, is_first_dk);
        is_first_dk = false;
        commit_mbarrier(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        uint32_t desc_ds2 = make_smem_desc_k_major(s_ds);
        uint32_t desc_k_nm = make_smem_desc_n_major(s_k);
        
        asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %0, 0;\n}" :: "r"(0));
        gemm_kn(tmem_s, desc_ds2, desc_k_nm, false);
        commit_mbarrier(mbar);
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_s + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int g_i = global_i + threadIdx.x;
            if (g_i < S && col < 128) {
                atomicAdd_bf16(&dq_ptr[g_i * 128 + col], __float2bfloat16(__uint_as_float(r0)));
                atomicAdd_bf16(&dq_ptr[g_i * 128 + col + 1], __float2bfloat16(__uint_as_float(r1)));
                atomicAdd_bf16(&dq_ptr[g_i * 128 + col + 2], __float2bfloat16(__uint_as_float(r2)));
                atomicAdd_bf16(&dq_ptr[g_i * 128 + col + 3], __float2bfloat16(__uint_as_float(r3)));
            }
        }
        __syncthreads();
    } 
    
    for (int col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dk + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        int g_j = global_j + threadIdx.x;
        if (g_j < S && col < 128) {
            dk_ptr[g_j * 128 + col] = __float2bfloat16(__uint_as_float(r0));
            dk_ptr[g_j * 128 + col + 1] = __float2bfloat16(__uint_as_float(r1));
            dk_ptr[g_j * 128 + col + 2] = __float2bfloat16(__uint_as_float(r2));
            dk_ptr[g_j * 128 + col + 3] = __float2bfloat16(__uint_as_float(r3));
        }
    }
    
    for (int col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_dv + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        int g_j = global_j + threadIdx.x;
        if (g_j < S && col < 128) {
            dv_ptr[g_j * 128 + col] = __float2bfloat16(__uint_as_float(r0));
            dv_ptr[g_j * 128 + col + 1] = __float2bfloat16(__uint_as_float(r1));
            dv_ptr[g_j * 128 + col + 2] = __float2bfloat16(__uint_as_float(r2));
            dv_ptr[g_j * 128 + col + 3] = __float2bfloat16(__uint_as_float(r3));
        }
    }
    
    if (threadIdx.x == 0) {
        tmem_dealloc_cta(tmem_s, 128);
        tmem_dealloc_cta(tmem_dp, 128);
        tmem_dealloc_cta(tmem_dv, 128);
        tmem_dealloc_cta(tmem_dk, 128);
    }
}

namespace tvm_ffi_mha_bwd {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaMemsetAsync(dQ.data_ptr(), 0, B * H * S * d * sizeof(__nv_bfloat16), stream));
    
    int num_j = (S + 127) / 128;
    dim3 grid(B * H * num_j);
    dim3 block(128);
    
    float attn_scale = 1.0f / sqrtf((float)d);
    
    int smem_size = 263168 + 256;
    CUDA_CHECK(cudaFuncSetAttribute(bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    bwd_kernel<<<grid, block, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(Q.data_ptr()),
        static_cast<const __nv_bfloat16*>(K.data_ptr()),
        static_cast<const __nv_bfloat16*>(V.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        S, attn_scale, L.stride(1)
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd