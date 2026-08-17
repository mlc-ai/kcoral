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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        const char* err_str;                                       \
        cuGetErrorString(_e, &err_str);                            \
        fprintf(stderr, "CU error %s at %s:%d\n",                 \
                err_str, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

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
        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_16x_fn(uint32_t col, 
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7,
    uint32_t* r8, uint32_t* r9, uint32_t* r10, uint32_t* r11,
    uint32_t* r12, uint32_t* r13, uint32_t* r14, uint32_t* r15) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
        : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
          "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7),
          "=r"(*r8),"=r"(*r9),"=r"(*r10),"=r"(*r11),
          "=r"(*r12),"=r"(*r13),"=r"(*r14),"=r"(*r15)
        : "r"(col));
}

__device__ __forceinline__ void tmem_store_8x_fn(uint32_t col, 
    uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3,
    uint32_t r4, uint32_t r5, uint32_t r6, uint32_t r7) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
        :: "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3),
           "r"(r4), "r"(r5), "r"(r6), "r"(r7) : "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, int layout_type) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    if (layout_type == 2) {
        uint32_t base_offset = (addr >> 7) & 0x7;
        d |= (uint64_t)base_offset << 49;
    }
    d |= (uint64_t)layout_type << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, uint32_t b_major) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (0u << 15);   
    d |= (b_major << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
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

__global__ void __launch_bounds__(128) attention_kernel(
    const __grid_constant__ CUtensorMap desc_Q,
    const __grid_constant__ CUtensorMap desc_K,
    const __grid_constant__ CUtensorMap desc_V,
    __nv_bfloat16* O_ptr, float* LSE_ptr, int S)
{
    int b = blockIdx.z;
    int h = blockIdx.y;
    int m_start = blockIdx.x * 128;
    int H = gridDim.y;

    extern __shared__ __align__(1024) uint8_t smem_dynamic[];
    uint8_t* smem_Q = smem_dynamic;                                    
    uint8_t* smem_K[2] = {smem_dynamic + 32768, smem_dynamic + 65536}; 
    uint8_t* smem_V[2] = {smem_dynamic + 98304, smem_dynamic + 131072}; 
    
    uint64_t* mbar_Q = (uint64_t*)(smem_dynamic + 163840);
    uint64_t* mbar_K[2] = {(uint64_t*)(smem_dynamic + 163848), (uint64_t*)(smem_dynamic + 163856)};
    uint64_t* mbar_V[2] = {(uint64_t*)(smem_dynamic + 163864), (uint64_t*)(smem_dynamic + 163872)};
    uint64_t* mbar_cp = (uint64_t*)(smem_dynamic + 163880);
    uint64_t* mbar_umma = (uint64_t*)(smem_dynamic + 163888);
    __shared__ uint32_t tmem_base_smem;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K[0], 1);
        init_smem_barrier_fn(mbar_K[1], 1);
        init_smem_barrier_fn(mbar_V[0], 1);
        init_smem_barrier_fn(mbar_V[1], 1);
        init_smem_barrier_fn(mbar_cp, 1);
        init_smem_barrier_fn(mbar_umma, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&tmem_base_smem, 512);
    }
    __syncthreads();
    
    float O_accum_left[64];
    float O_accum_right[64];
    #pragma unroll
    for(int i=0; i<64; i++) {
        O_accum_left[i] = 0.0f;
        O_accum_right[i] = 0.0f;
    }
    float m_prev = -INFINITY;
    float l_prev = 0.0f;
    
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768);
        tma_load_3d_fn(&desc_Q, mbar_Q, smem_Q, 0, m_start, b * H + h);
        tma_load_3d_fn(&desc_Q, mbar_Q, smem_Q + 16384, 64, m_start, b * H + h);
        
        mbarrier_arrive_and_expect_tx_fn(mbar_K[0], 32768);
        tma_load_3d_fn(&desc_K, mbar_K[0], smem_K[0], 0, 0, b * H + h);
        tma_load_3d_fn(&desc_K, mbar_K[0], smem_K[0] + 16384, 64, 0, b * H + h);
        
        mbarrier_arrive_and_expect_tx_fn(mbar_V[0], 32768);
        tma_load_3d_fn(&desc_V, mbar_V[0], smem_V[0], 0, 0, b * H + h);
        tma_load_3d_fn(&desc_V, mbar_V[0], smem_V[0] + 16384, 64, 0, b * H + h);
    }
    
    uint32_t phase_Q = 0;
    uint32_t phase_K[2] = {0, 0};
    uint32_t phase_V[2] = {0, 0};
    uint32_t phase_cp = 0;
    uint32_t phase_umma = 0;
    
    mbarrier_wait_fn(mbar_Q, phase_Q);
    phase_Q ^= 1;
    __syncthreads();
    
    if (threadIdx.x == 0) {
        for (int i = 0; i < 8; ++i) {
            uint32_t addr = (i < 4 ? 0 : 16384) + (i % 4) * 32;
            uint64_t sdesc = make_smem_desc_sm100_fn(smem_Q + addr, 2048, 128, 0); 
            uint32_t taddr = 0 | (i * 8);
            asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;" :: "r"(taddr), "l"(sdesc));
        }
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(mbar_cp)));
    }
    mbarrier_wait_fn(mbar_cp, phase_cp);
    phase_cp ^= 1;
    __syncthreads();
    
    float p_scale = 0.1275225023f; // log2(e) / sqrt(128)
    int m_global = m_start + threadIdx.x;
    
    for (int n_idx = 0; n_idx <= m_start; n_idx += 128) {
        int n_step = n_idx / 128;
        int buf = n_step % 2;
        int next_buf = (n_step + 1) % 2;
        int n_next = n_idx + 128;
        
        if (n_next <= m_start && threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_K[next_buf], 32768);
            tma_load_3d_fn(&desc_K, mbar_K[next_buf], smem_K[next_buf], 0, n_next, b * H + h);
            tma_load_3d_fn(&desc_K, mbar_K[next_buf], smem_K[next_buf] + 16384, 64, n_next, b * H + h);
            
            mbarrier_arrive_and_expect_tx_fn(mbar_V[next_buf], 32768);
            tma_load_3d_fn(&desc_V, mbar_V[next_buf], smem_V[next_buf], 0, n_next, b * H + h);
            tma_load_3d_fn(&desc_V, mbar_V[next_buf], smem_V[next_buf] + 16384, 64, n_next, b * H + h);
        }
        
        mbarrier_wait_fn(mbar_K[buf], phase_K[buf]);
        phase_K[buf] ^= 1;
        __syncthreads();
        
        uint32_t taddr_S = 64;
        uint32_t taddr_P = 64;
        
        if (threadIdx.x == 0) {
            for (int k = 0; k < 128; k += 16) {
                uint32_t addr = (k < 64 ? 0 : 16384) + (k % 64) * 2;
                uint64_t sdesc = make_smem_desc_sm100_fn(smem_K[buf] + addr, 1, 1024, 2); 
                uint32_t taddr_Q = 0 | ((k / 16) * 8);
                uint32_t idesc = make_instr_desc_fn(128, 128, 0); 
                uint32_t accum = (k == 0) ? 0 : 1;
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
                    :: "r"(taddr_S), "r"(taddr_Q), "l"(sdesc), "r"(idesc), "r"(accum));
            }
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(mbar_umma)));
        }
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;
        __syncthreads();
        
        float max_val = -INFINITY;
        for (int c = 0; c < 128; c += 16) {
            uint32_t r[16];
            tmem_load_16x_fn(taddr_S + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7], &r[8], &r[9], &r[10], &r[11], &r[12], &r[13], &r[14], &r[15]);
            tmem_load_fence_fn();
            #pragma unroll
            for(int i=0; i<16; i++) {
                float f = __uint_as_float(r[i]) * p_scale;
                if (m_global < n_idx + c + i) f = -INFINITY;
                max_val = fmaxf(max_val, f);
            }
        }
        
        float m_new = fmaxf(m_prev, max_val);
        float scale = fast_exp2f_fn(m_prev - m_new);
        
        float l_sum = 0.0f;
        for (int c = 0; c < 128; c += 16) {
            uint32_t r[16];
            tmem_load_16x_fn(taddr_S + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7], &r[8], &r[9], &r[10], &r[11], &r[12], &r[13], &r[14], &r[15]);
            tmem_load_fence_fn();
            
            uint32_t p[8];
            #pragma unroll
            for(int i=0; i<16; i+=2) {
                float f0 = __uint_as_float(r[i]) * p_scale;
                float f1 = __uint_as_float(r[i+1]) * p_scale;
                
                if (m_global < n_idx + c + i + 0) f0 = -INFINITY;
                if (m_global < n_idx + c + i + 1) f1 = -INFINITY;
                
                f0 = fast_exp2f_fn(f0 - m_new);
                f1 = fast_exp2f_fn(f1 - m_new);
                
                l_sum += f0 + f1;
                p[i/2] = pack_bf16_fn(__float_as_uint(f0), __float_as_uint(f1));
            }
            tmem_store_8x_fn(taddr_P + c/2, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
        }
        
        l_prev = l_prev * scale + l_sum;
        
        #pragma unroll
        for(int i=0; i<64; i++) {
            O_accum_left[i] *= scale;
            O_accum_right[i] *= scale;
        }
        
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        
        mbarrier_wait_fn(mbar_V[buf], phase_V[buf]);
        phase_V[buf] ^= 1;
        __syncthreads();
        
        if (threadIdx.x == 0) {
            for (int n = 0; n < 128; n += 16) {
                uint32_t addr_left = n * 128;
                uint32_t addr_right = 16384 + n * 128;
                uint64_t desc_left = make_smem_desc_sm100_fn(smem_V[buf] + addr_left, 2048, 1024, 2);
                uint64_t desc_right = make_smem_desc_sm100_fn(smem_V[buf] + addr_right, 2048, 1024, 2);
                uint32_t taddr_P_mma = taddr_P | ((n / 16) * 8);
                uint32_t taddr_O_L = 320;
                uint32_t taddr_O_R = 384;
                uint32_t idesc = make_instr_desc_fn(128, 64, 1); 
                uint32_t accum = (n == 0) ? 0 : 1;
                
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
                    :: "r"(taddr_O_L), "r"(taddr_P_mma), "l"(desc_left), "r"(idesc), "r"(accum));
                    
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, %4, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
                    :: "r"(taddr_O_R), "r"(taddr_P_mma), "l"(desc_right), "r"(idesc), "r"(accum));
            }
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(mbar_umma)));
        }
        mbarrier_wait_fn(mbar_umma, phase_umma);
        phase_umma ^= 1;
        __syncthreads();
        
        for (int c = 0; c < 64; c += 16) {
            uint32_t rL[16];
            uint32_t rR[16];
            tmem_load_16x_fn(320 + c, &rL[0], &rL[1], &rL[2], &rL[3], &rL[4], &rL[5], &rL[6], &rL[7], &rL[8], &rL[9], &rL[10], &rL[11], &rL[12], &rL[13], &rL[14], &rL[15]);
            tmem_load_16x_fn(384 + c, &rR[0], &rR[1], &rR[2], &rR[3], &rR[4], &rR[5], &rR[6], &rR[7], &rR[8], &rR[9], &rR[10], &rR[11], &rR[12], &rR[13], &rR[14], &rR[15]);
            tmem_load_fence_fn();
            #pragma unroll
            for(int i=0; i<16; i++) {
                O_accum_left[c+i] += __uint_as_float(rL[i]);
                O_accum_right[c+i] += __uint_as_float(rR[i]);
            }
        }
        
        m_prev = m_new;
    }
    
    float inv_l = 1.0f / l_prev;
    __nv_bfloat16* smem_bf16 = (__nv_bfloat16*)smem_dynamic;
    for(int i=0; i<64; i++) {
        smem_bf16[threadIdx.x * 128 + i] = __float2bfloat16(O_accum_left[i] * inv_l);
        smem_bf16[threadIdx.x * 128 + 64 + i] = __float2bfloat16(O_accum_right[i] * inv_l);
    }
    __syncthreads();
    
    if (m_global < S) {
        float lse = (m_prev + log2f(l_prev)) * 0.69314718056f;
        LSE_ptr[(uint64_t)b * H * S + h * S + m_global] = lse;
    }
    
    if (m_start < S) {
        uint32_t valid_rows = min(128, S - m_start);
        uint64_t base_offset = ((uint64_t)b * H * S + h * S + m_start) * 128;
        for (uint32_t i = threadIdx.x; i < valid_rows * 16; i += 128) {
            uint32_t r = i / 16;
            uint32_t lane = i % 16;
            uint4* O_vec = (uint4*)(O_ptr + base_offset + r * 128);
            uint4* smem_vec = (uint4*)(smem_bf16 + r * 128);
            O_vec[lane] = smem_vec[lane]; 
        }
    }
    
    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_base_smem, 512); 
    }
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint32_t box0, uint32_t box1, uint32_t box2, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
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
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);

    CUtensorMap desc_Q, desc_K, desc_V;
    
    CU_CHECK(create_tma_3d_descriptor_2B(&desc_Q, Q.data_ptr(), 128, S, B*H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_NONE));
    CU_CHECK(create_tma_3d_descriptor_2B(&desc_K, K.data_ptr(), 128, S, B*H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_3d_descriptor_2B(&desc_V, V.data_ptr(), 128, S, B*H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B));

    dim3 grid((S + 127) / 128, H, B);
    dim3 block(128); 
    size_t smem_bytes = 165000;
    
    CUDA_CHECK(cudaFuncSetAttribute(tvm_ffi_example_cuda::attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    attention_kernel<<<grid, block, smem_bytes, stream>>>(desc_Q, desc_K, desc_V, (__nv_bfloat16*)O.data_ptr(), (float*)LSE.data_ptr(), S);
    
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda