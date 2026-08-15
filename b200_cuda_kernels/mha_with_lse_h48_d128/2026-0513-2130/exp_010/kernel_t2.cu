#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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

namespace tvm_ffi_mha_sm100 {

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols) : "memory");
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols) : "memory");
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(count) : "memory");
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(tx_bytes) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(bar)), "r"(phase) : "memory");
}

__device__ __forceinline__ void cp_async_bulk_1d(void* smem, const void* gmem, uint32_t size, uint64_t* mbar) {
    uint32_t smem_ptr = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t mbar_ptr = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        :: "r"(smem_ptr), "l"(gmem), "r"(size), "r"(mbar_ptr) : "memory");
}

__device__ __forceinline__ void fence_proxy_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t layout_type) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16; 
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)layout_type << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool transA, bool transB) {
    uint32_t d = 0;
    d |= (1u << 4);           
    d |= (1u << 7);           
    d |= (1u << 10);          
    if (transA) d |= (1u << 15);
    if (transB) d |= (1u << 16);
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void umma_f16_cg1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum) : "memory");
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
                   "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col) : "memory");
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

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__global__ void mha_fwd_sm100_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S, int D) {
    
    setmaxnreg_inc_sync_fn<240>();

    int q_tile_idx = blockIdx.x;
    int head = blockIdx.y;
    int batch = blockIdx.z;
    int q_start = q_tile_idx * 128;

    if (q_start >= S) return;

    uint32_t q_size = min(128, S - q_start) * 256;

    uint64_t B_stride = (uint64_t)H * S * D;
    uint64_t H_stride = (uint64_t)S * D;
    const __nv_bfloat16* Q_bh = Q + batch * B_stride + head * H_stride;
    const __nv_bfloat16* K_bh = K + batch * B_stride + head * H_stride;
    const __nv_bfloat16* V_bh = V + batch * B_stride + head * H_stride;
    __nv_bfloat16* O_bh = O + batch * B_stride + head * H_stride;

    uint64_t B_stride_lse = (uint64_t)H * S;
    uint64_t H_stride_lse = (uint64_t)S;
    float* LSE_bh = LSE + batch * B_stride_lse + head * H_stride_lse;

    extern __shared__ uint8_t shared_mem[];
    uint8_t* smem_Q = shared_mem;                    
    uint8_t* smem_K = smem_Q + 32768;                
    uint8_t* smem_V = smem_K + 32768;                
    uint8_t* smem_P = smem_V + 32768;                
    float* smem_O_accum = (float*)(smem_P + 32768);  
    
    uint64_t* mbar_Q = (uint64_t*)((uint8_t*)smem_O_accum + 65536);
    uint64_t* mbar_K = mbar_Q + 1;
    uint64_t* mbar_V = mbar_K + 1;
    uint64_t* mbar_MMA = mbar_V + 1;

    __shared__ uint32_t tmem_base;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K, 1);
        init_smem_barrier_fn(mbar_V, 1);
        init_smem_barrier_fn(mbar_MMA, 1);
        fence_smem_barrier_init_fn();
    }
    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(&tmem_base, 256);
    }
    __syncthreads();

    uint32_t tmem_S = tmem_base;
    uint32_t tmem_O = tmem_base + 128;

    uint32_t phase_Q = 0, phase_K = 0, phase_V = 0, phase_MMA = 0;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, q_size);
        if (q_size > 0) cp_async_bulk_1d(smem_Q, Q_bh + q_start * 128, q_size, mbar_Q);
    }

    if (q_size < 32768) {
        uint32_t* smem_Q_u32 = (uint32_t*)((char*)smem_Q + q_size);
        uint32_t rem = 32768 - q_size;
        for (int i = threadIdx.x; i < rem / 4; i += 128) {
            smem_Q_u32[i] = 0;
        }
    }

    for (int i = threadIdx.x; i < 16384; i += 128) {
        smem_O_accum[i] = 0.0f;
    }

    float m_accum = -1e20f;
    float l_accum = 0.0f;
    
    mbarrier_wait_fn(mbar_Q, phase_Q);
    phase_Q ^= 1;
    __syncthreads();

    fence_proxy_async_shared_fn();

    uint32_t idesc_QK = make_instr_desc_fn(128, 128, false, false);
    uint32_t idesc_PV = make_instr_desc_fn(128, 128, false, true);

    int num_kv_tiles = (S + 127) / 128;
    float scale = 0.08838834764f; 
    float log2e = 1.4426950408889f;

    for (int kv_idx = 0; kv_idx < num_kv_tiles; kv_idx++) {
        int kv_start = kv_idx * 128;
        uint32_t kv_size = min(128, S - kv_start) * 256;
        
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_K, kv_size);
            if (kv_size > 0) cp_async_bulk_1d(smem_K, K_bh + kv_start * 128, kv_size, mbar_K);
            
            mbarrier_arrive_and_expect_tx_fn(mbar_V, kv_size);
            if (kv_size > 0) cp_async_bulk_1d(smem_V, V_bh + kv_start * 128, kv_size, mbar_V);
        }
        
        if (kv_size < 32768) {
            uint32_t* smem_K_u32 = (uint32_t*)((char*)smem_K + kv_size);
            uint32_t* smem_V_u32 = (uint32_t*)((char*)smem_V + kv_size);
            uint32_t rem = 32768 - kv_size;
            for (int i = threadIdx.x; i < rem / 4; i += 128) {
                smem_K_u32[i] = 0;
                smem_V_u32[i] = 0;
            }
        }
        
        mbarrier_wait_fn(mbar_K, phase_K);
        phase_K ^= 1;
        __syncthreads();

        fence_proxy_async_shared_fn();

        if (threadIdx.x == 0) {
            for (int k_step = 0; k_step < 8; ++k_step) {
                uint64_t desc_Q = make_smem_desc_sm100_fn(smem_Q + k_step * 32, 16, 2048, 0); 
                uint64_t desc_K = make_smem_desc_sm100_fn(smem_K + k_step * 32, 16, 2048, 0);
                uint32_t accum = (k_step == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_S, desc_Q, desc_K, idesc_QK, accum);
            }
            uint32_t a = (uint32_t)__cvta_generic_to_shared(mbar_MMA);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(a) : "memory");
        }
        mbarrier_wait_fn(mbar_MMA, phase_MMA);
        phase_MMA ^= 1;
        __syncthreads();

        uint32_t S_u32[128];
        for (int col = 0; col < 128; col += 8) {
            tmem_load_8x_fn(tmem_S + col,
                &S_u32[col+0], &S_u32[col+1], &S_u32[col+2], &S_u32[col+3],
                &S_u32[col+4], &S_u32[col+5], &S_u32[col+6], &S_u32[col+7]);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

        float max_S = -1e20f;
        float S_vals[128];
        for(int col = 0; col < 128; ++col) {
            float f = __uint_as_float(S_u32[col]) * scale;
            if (kv_start + col >= S) f = -1e20f;
            S_vals[col] = f;
            max_S = fmaxf(max_S, f);
        }

        float m_new = fmaxf(m_accum, max_S);
        float exp_diff = fast_exp2f_fn((m_accum - m_new) * log2e);

        float sum_P = 0.0f;
        for (int col = 0; col < 128; col += 2) {
            float p0 = fast_exp2f_fn((S_vals[col] - m_new) * log2e);
            float p1 = fast_exp2f_fn((S_vals[col+1] - m_new) * log2e);
            sum_P += p0 + p1;
            uint32_t packed = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
            ((uint32_t*)smem_P)[threadIdx.x * 64 + col / 2] = packed;
        }

        mbarrier_wait_fn(mbar_V, phase_V);
        phase_V ^= 1;
        __syncthreads();

        fence_proxy_async_shared_fn();

        if (threadIdx.x == 0) {
            for (int k_step = 0; k_step < 8; ++k_step) {
                uint64_t desc_P = make_smem_desc_sm100_fn(smem_P + k_step * 32, 16, 2048, 0);
                uint64_t desc_V = make_smem_desc_sm100_fn(smem_V + k_step * 4096, 128, 256, 0);
                uint32_t accum = (k_step == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_O, desc_P, desc_V, idesc_PV, accum);
            }
            uint32_t a = (uint32_t)__cvta_generic_to_shared(mbar_MMA);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(a) : "memory");
        }
        mbarrier_wait_fn(mbar_MMA, phase_MMA);
        phase_MMA ^= 1;
        __syncthreads();

        uint32_t O_u32[128];
        for (int col = 0; col < 128; col += 8) {
            tmem_load_8x_fn(tmem_O + col,
                &O_u32[col+0], &O_u32[col+1], &O_u32[col+2], &O_u32[col+3],
                &O_u32[col+4], &O_u32[col+5], &O_u32[col+6], &O_u32[col+7]);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

        l_accum = l_accum * exp_diff + sum_P;
        m_accum = m_new;

        for (int col = 0; col < 128; ++col) {
            float o_old = smem_O_accum[threadIdx.x * 128 + col];
            float o_new = __uint_as_float(O_u32[col]);
            smem_O_accum[threadIdx.x * 128 + col] = o_old * exp_diff + o_new;
        }
    }

    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_base, 256);
    }
    
    if (q_start + threadIdx.x < S) {
        float lse = m_accum + logf(l_accum);
        LSE_bh[q_start + threadIdx.x] = lse;
    }

    float inv_l = 1.0f / l_accum;
    if (q_start + threadIdx.x < S) {
        uint4* out_ptr = (uint4*)(&O_bh[(q_start + threadIdx.x) * 128]);
        for (int col = 0; col < 16; ++col) {
            uint32_t p0 = pack_bf16_fn(__float_as_uint(smem_O_accum[threadIdx.x * 128 + col * 8 + 0] * inv_l),
                                       __float_as_uint(smem_O_accum[threadIdx.x * 128 + col * 8 + 1] * inv_l));
            uint32_t p1 = pack_bf16_fn(__float_as_uint(smem_O_accum[threadIdx.x * 128 + col * 8 + 2] * inv_l),
                                       __float_as_uint(smem_O_accum[threadIdx.x * 128 + col * 8 + 3] * inv_l));
            uint32_t p2 = pack_bf16_fn(__float_as_uint(smem_O_accum[threadIdx.x * 128 + col * 8 + 4] * inv_l),
                                       __float_as_uint(smem_O_accum[threadIdx.x * 128 + col * 8 + 5] * inv_l));
            uint32_t p3 = pack_bf16_fn(__float_as_uint(smem_O_accum[threadIdx.x * 128 + col * 8 + 6] * inv_l),
                                       __float_as_uint(smem_O_accum[threadIdx.x * 128 + col * 8 + 7] * inv_l));
            out_ptr[col] = make_uint4(p0, p1, p2, p3);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3);

    const __nv_bfloat16* q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* k_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* v_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());

    int smem_bytes = 196608 + 1024;

    CUDA_CHECK(cudaFuncSetAttribute(mha_fwd_sm100_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));

    int q_tiles = (S + 127) / 128;
    dim3 grid(q_tiles, H, B);
    dim3 block(128);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    mha_fwd_sm100_kernel<<<grid, block, smem_bytes, stream>>>(
        q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr, B, H, S, D
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_mha_sm100