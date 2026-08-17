#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <math.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
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

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void umma_f16_smemA_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_f16_tmemA_fn(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
        :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void tma_store_4d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4, %5}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, int a_major, int b_major) {
    uint32_t d = 0;
    d |= (1u << 4);    // c_format = FP32
    d |= (1u << 7);    // a_format = BF16
    d |= (1u << 10);   // b_format = BF16
    d |= (a_major << 15);
    d |= (b_major << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn_k_major_none(void* smem_ptr, uint32_t M_or_N) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t sbo = 128;
    uint32_t lbo = (M_or_N / 8) * 128; 
    
    uint64_t d = 0;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   // version = 1 (SM100)
    d |= (uint64_t)0 << 61;   // layout_type = SWIZZLE_NONE
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn_mn_major_none(void* smem_ptr, uint32_t M_or_N) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t lbo = 128;
    uint32_t sbo = (M_or_N / 8) * 128; 
    
    uint64_t d = 0;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   // version = 1 (SM100)
    d |= (uint64_t)0 << 61;   // layout_type = SWIZZLE_NONE
    return d;
}

__device__ __forceinline__ uint16_t float_to_bf16(float x) {
    __nv_bfloat16 bf = __float2bfloat16(x);
    return *(uint16_t*)&bf;
}

CUresult create_tma_4d_descriptor_2B_none(CUtensorMap* d, void* globalAddress, uint64_t D, uint64_t S, uint64_t H, uint64_t B, uint32_t smem_D, uint32_t smem_S) {
    cuuint64_t globalDim[4] = {D, S, H, B};
    cuuint64_t globalStrides[3] = {D * 2, D * S * 2, D * S * H * 2};
    cuuint32_t boxDim[4] = {smem_D, smem_S, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
        globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

__global__ __launch_bounds__(128) void mha_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* __restrict__ LSE,
    int B, int H, int seq_len
) {
    int b = blockIdx.x / H;
    int h = blockIdx.x % H;
    int bq = blockIdx.y;
    
    if (bq * 64 >= seq_len) return;
    
    int row = threadIdx.x;
    
    extern __shared__ __align__(128) uint8_t shared_mem[];
    uint16_t* smem_Q = (uint16_t*)shared_mem;               
    uint16_t* smem_K = (uint16_t*)(shared_mem + 16384);     
    uint16_t* smem_V = (uint16_t*)(shared_mem + 32768);     
    
    uint64_t* tma_bar = (uint64_t*)(shared_mem + 49152);
    uint64_t* umma_bar = (uint64_t*)(shared_mem + 49160);
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(tma_bar, 1);
        init_smem_barrier_fn(umma_bar, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t tma_phase = 0;
    uint32_t umma_phase = 0;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(tma_bar, 16384);
        tma_load_4d_fn(&tma_Q, tma_bar, smem_Q, 0, bq * 64, h, b);
    }
    mbarrier_wait_fn(tma_bar, tma_phase);
    tma_phase ^= 1;
    
    __shared__ uint32_t shared_tmem_ptr;
    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(&shared_tmem_ptr, 256);
    }
    __syncthreads();
    uint32_t tmem_ptr = shared_tmem_ptr;
    
    uint32_t tmem_S = tmem_ptr; 
    uint32_t tmem_P = tmem_ptr + 64; 
    uint32_t tmem_O_temp = tmem_ptr + 96; // 32 cols for P, so 64 + 32 = 96. And O_temp takes 128 cols, 96 + 128 = 224 <= 256. Perfect.
    
    uint64_t desc_Q = make_smem_desc_sm100_fn_k_major_none(smem_Q, 64); 
    uint64_t desc_K = make_smem_desc_sm100_fn_mn_major_none(smem_K, 64); 
    uint64_t desc_V = make_smem_desc_sm100_fn_k_major_none(smem_V, 128); 
    
    uint32_t idesc_QK = make_instr_desc_fn(64, 64, 0, 1);
    uint32_t idesc_PV = make_instr_desc_fn(64, 128, 0, 0);
    
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float O_accum[128];
    for (int i = 0; i < 128; ++i) O_accum[i] = 0.0f;
    
    for (int bk = 0; bk <= bq; ++bk) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(tma_bar, 32768);
            tma_load_4d_fn(&tma_K, tma_bar, smem_K, 0, bk * 64, h, b);
            tma_load_4d_fn(&tma_V, tma_bar, smem_V, 0, bk * 64, h, b);
        }
        mbarrier_wait_fn(tma_bar, tma_phase);
        tma_phase ^= 1;
        
        for (int step = 0; step < 8; ++step) {
            uint32_t accum = (step == 0) ? 0 : 1;
            if (threadIdx.x == 0) umma_f16_smemA_fn(tmem_S, desc_Q + step * 2, desc_K + step * 2, idesc_QK, accum);
        }
        
        if (threadIdx.x == 0) {
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(&umma_bar[0])));
        }
        mbarrier_wait_fn(umma_bar, umma_phase);
        umma_phase ^= 1;
        
        float S_local[64];
        for (int i = 0; i < 16; ++i) {
            uint32_t col = i * 4;
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_S + col));
            S_local[i*4+0] = __uint_as_float(r0);
            S_local[i*4+1] = __uint_as_float(r1);
            S_local[i*4+2] = __uint_as_float(r2);
            S_local[i*4+3] = __uint_as_float(r3);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        __syncthreads();
        
        float max_val = -INFINITY;
        float scale = 1.0f / sqrtf(128.0f);
        if (row < 64) {
            for (int c = 0; c < 64; ++c) {
                int global_q_r = bq * 64 + row;
                int global_k_c = bk * 64 + c;
                if (global_q_r >= seq_len || global_k_c > global_q_r || global_k_c >= seq_len) {
                    S_local[c] = -INFINITY;
                } else {
                    S_local[c] *= scale;
                }
                max_val = fmaxf(max_val, S_local[c]);
            }
        }
        
        float m_new = fmaxf(m_i, max_val);
        float exp_scale = 1.0f;
        if (m_new != -INFINITY) {
            exp_scale = expf(m_i - m_new);
        }
        l_i *= exp_scale;
        
        float sum = 0.0f;
        uint32_t P_packed[32];
        
        if (row < 64) {
            for (int i = 0; i < 128; ++i) {
                O_accum[i] *= exp_scale;
            }
            for (int i = 0; i < 32; ++i) {
                float p0 = 0.0f;
                float p1 = 0.0f;
                if (m_new != -INFINITY) {
                    if (S_local[2*i] != -INFINITY) p0 = expf(S_local[2*i] - m_new);
                    if (S_local[2*i+1] != -INFINITY) p1 = expf(S_local[2*i+1] - m_new);
                }
                sum += p0 + p1;
                uint16_t lo = float_to_bf16(p0);
                uint16_t hi = float_to_bf16(p1);
                P_packed[i] = ((uint32_t)hi << 16) | lo;
            }
        } else {
            for (int i = 0; i < 32; ++i) P_packed[i] = 0;
        }
        
        for (int i = 0; i < 8; ++i) {
            uint32_t col = i * 4;
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
                :: "r"(tmem_P + col), "r"(P_packed[i*4+0]), "r"(P_packed[i*4+1]), "r"(P_packed[i*4+2]), "r"(P_packed[i*4+3]));
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        __syncthreads();
        
        l_i += sum;
        m_i = m_new;
        
        for (int step = 0; step < 4; ++step) {
            uint32_t accum = (step == 0) ? 0 : 1;
            uint32_t a_tmem = tmem_P + step * 8;
            uint64_t b_desc = desc_V + step * 256;
            if (threadIdx.x == 0) {
                umma_f16_tmemA_fn(tmem_O_temp, a_tmem, b_desc, idesc_PV, accum);
            }
        }
        
        if (threadIdx.x == 0) {
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(&umma_bar[0])));
        }
        mbarrier_wait_fn(umma_bar, umma_phase);
        umma_phase ^= 1;
        
        for (int i = 0; i < 32; ++i) {
            uint32_t col = i * 4;
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_O_temp + col));
            if (row < 64) {
                O_accum[i*4+0] += __uint_as_float(r0);
                O_accum[i*4+1] += __uint_as_float(r1);
                O_accum[i*4+2] += __uint_as_float(r2);
                O_accum[i*4+3] += __uint_as_float(r3);
            }
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        __syncthreads();
    }
    
    if (row < 64) {
        for (int i = 0; i < 128; ++i) {
            float val = (l_i > 0.0f) ? (O_accum[i] / l_i) : 0.0f;
            int s_idx = row * 128 + i;
            smem_Q[s_idx] = float_to_bf16(val);
        }
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        tma_store_4d_fn(&tma_O, smem_Q, 0, bq * 64, h, b);
        asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
        tma_store_wait_fn<0>();
    }
    
    if (row < 64) {
        int global_r = bq * 64 + row;
        if (global_r < seq_len) {
            LSE[b * H * seq_len + h * seq_len + global_r] = logf(l_i) + m_i;
        }
    }
    
    __syncthreads();
    if (threadIdx.x < 32) tmem_dealloc_cg1_fn(tmem_ptr, 256);
}

namespace tvm_ffi_example_cuda {
    void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
             tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
        CUDA_CHECK(cudaSetDevice(Q.device().device_id));
        
        int64_t B = Q.size(0);
        int64_t H = Q.size(1);
        int64_t S = Q.size(2);
        
        CUtensorMap tma_Q, tma_K, tma_V, tma_O;
        create_tma_4d_descriptor_2B_none(&tma_Q, Q.data_ptr(), 128, S, H, B, 128, 64);
        create_tma_4d_descriptor_2B_none(&tma_K, K.data_ptr(), 128, S, H, B, 128, 64);
        create_tma_4d_descriptor_2B_none(&tma_V, V.data_ptr(), 128, S, H, B, 128, 64);
        create_tma_4d_descriptor_2B_none(&tma_O, O.data_ptr(), 128, S, H, B, 128, 64);
        
        int grid_x = B * H;
        int grid_y = (S + 63) / 64;
        dim3 grid(grid_x, grid_y);
        dim3 block(128);
        
        cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
        
        int smem_size = 49168;
        CUDA_CHECK(cudaFuncSetAttribute(mha_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        
        mha_fwd_kernel<<<grid, block, smem_size, stream>>>(
            tma_Q, tma_K, tma_V, tma_O,
            static_cast<float*>(LSE.data_ptr()),
            B, H, S
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    
    TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);
}