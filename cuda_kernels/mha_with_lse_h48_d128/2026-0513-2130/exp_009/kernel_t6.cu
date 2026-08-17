#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

__device__ __forceinline__ float exp2f_emulate(float x) {
    x = fmaxf(x, -126.0f);
    float x_floor = floorf(x);
    float x_frac = x - x_floor;
    
    float p = 0.0555041f;
    p = fmaf(p, x_frac, 0.2402265f);
    p = fmaf(p, x_frac, 0.69314718f);
    p = fmaf(p, x_frac, 1.0f);
    
    uint32_t exp_bits = ((int)x_floor + 127) << 23;
    float two_to_floor = __uint_as_float(exp_bits);
    
    return p * two_to_floor;
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

__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
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

__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint32_t box0, uint32_t box1, uint32_t box2) {
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
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
    );
}

__device__ __forceinline__ void tma_load_3d_cg1_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
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

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
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

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr >> 4) & 0x3FFF;
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)0 << 49;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_idesc_QK() {
    uint32_t d = 0;
    d |= (1u << 4);           
    d |= (1u << 7);           
    d |= (1u << 10);          
    d |= (0u << 15);          
    d |= (0u << 16);          
    d |= ((128 / 8) << 17);   
    d |= ((128 / 16) << 24);  
    return d;
}

__device__ __forceinline__ uint32_t make_idesc_PV_64() {
    uint32_t d = 0;
    d |= (1u << 4);           
    d |= (1u << 7);           
    d |= (1u << 10);          
    d |= (0u << 15);          
    d |= (1u << 16);          
    d |= ((64 / 8) << 17);    
    d |= ((128 / 16) << 24);  
    return d;
}

__global__ void mha_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O_global, float* LSE_global,
    int B, int H, int S) 
{
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x;
    if (i * 128 >= S) return;

    extern __shared__ char smem_buf_raw[];
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_buf_raw);
    uint32_t align_offset = (1024 - (smem_addr % 1024)) % 1024;
    char* smem_buf = smem_buf_raw + align_offset;

    char* Q_smem_b = smem_buf;                             
    char* K_smem_b = smem_buf + 32768;                     
    char* V_smem_b = smem_buf + 98304;                     
    char* P_smem_b = smem_buf + 163840;                    

    uint64_t* mbar_Q   = (uint64_t*)(smem_buf + 229376);   
    uint64_t* mbar_K   = (uint64_t*)(smem_buf + 229384);   
    uint64_t* mbar_V   = (uint64_t*)(smem_buf + 229400);   
    uint64_t* mbar_mma = (uint64_t*)(smem_buf + 229416);   
    uint32_t* tmem_base_smem = (uint32_t*)(smem_buf + 229432); 

    if (threadIdx.x == 0) {
        prefetch_tma_descriptor_fn(&tma_Q);
        prefetch_tma_descriptor_fn(&tma_K);
        prefetch_tma_descriptor_fn(&tma_V);
        
        init_smem_barrier_fn(&mbar_Q[0], 1);
        init_smem_barrier_fn(&mbar_K[0], 1);
        init_smem_barrier_fn(&mbar_K[1], 1);
        init_smem_barrier_fn(&mbar_V[0], 1);
        init_smem_barrier_fn(&mbar_V[1], 1);
        init_smem_barrier_fn(&mbar_mma[0], 1);
        init_smem_barrier_fn(&mbar_mma[1], 1);
    }
    __syncthreads();

    uint32_t phase_K[2] = {1, 0}; 
    uint32_t phase_V[2] = {0, 0};
    uint32_t phase_mma[2] = {0, 0};

    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(tmem_base_smem, 512);
    }
    __syncthreads();
    uint32_t tmem_base = *tmem_base_smem;

    // Prologue
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_Q[0], 32768);
        tma_load_3d_cg1_fn(&tma_Q, &mbar_Q[0], Q_smem_b, 0, i * 128, b * H + h);
        tma_load_3d_cg1_fn(&tma_Q, &mbar_Q[0], Q_smem_b + 16384, 64, i * 128, b * H + h);

        mbarrier_arrive_and_expect_tx_fn(&mbar_K[0], 32768);
        tma_load_3d_cg1_fn(&tma_K, &mbar_K[0], K_smem_b, 0, 0, b * H + h);
        tma_load_3d_cg1_fn(&tma_K, &mbar_K[0], K_smem_b + 16384, 64, 0, b * H + h);
        
        mbarrier_arrive_and_expect_tx_fn(&mbar_V[0], 32768);
        tma_load_3d_cg1_fn(&tma_V, &mbar_V[0], V_smem_b, 0, 0, b * H + h);
        tma_load_3d_cg1_fn(&tma_V, &mbar_V[0], V_smem_b + 16384, 64, 0, b * H + h);
        
        if (S > 128) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_K[1], 32768);
            tma_load_3d_cg1_fn(&tma_K, &mbar_K[1], K_smem_b + 32768, 0, 128, b * H + h);
            tma_load_3d_cg1_fn(&tma_K, &mbar_K[1], K_smem_b + 32768 + 16384, 64, 128, b * H + h);
            
            mbarrier_arrive_and_expect_tx_fn(&mbar_V[1], 32768);
            tma_load_3d_cg1_fn(&tma_V, &mbar_V[1], V_smem_b + 32768, 0, 128, b * H + h);
            tma_load_3d_cg1_fn(&tma_V, &mbar_V[1], V_smem_b + 32768 + 16384, 64, 128, b * H + h);
        }
    }
    
    mbarrier_wait_fn(&mbar_Q[0], 0);

    float O_i[128];
    #pragma unroll
    for(int c=0; c<128; c++) O_i[c] = 0.0f;
    
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float scale = 0.0883883476f;
    uint32_t idesc_QK = make_idesc_QK();
    uint32_t idesc_PV_64 = make_idesc_PV_64();
    
    mbarrier_wait_fn(&mbar_K[0], 0);
    phase_K[0] ^= 1;
    if (threadIdx.x == 0) {
        for(int k=0; k<8; k++) {
            uint32_t offset = (k < 4) ? (k * 32) : (16384 + (k - 4) * 32);
            uint64_t desc_A = make_smem_desc(Q_smem_b + offset, 0, 1024);
            uint64_t desc_B = make_smem_desc(K_smem_b + offset, 0, 1024);
            uint32_t accum = (k == 0) ? 0 : 1;
            umma_f16_cg1_fn(tmem_base, desc_A, desc_B, idesc_QK, accum);
        }
        umma_commit_cg1_fn(&mbar_mma[0]);
    }

    for (int j = 0; j < S; j += 128) {
        int c = (j / 128) % 2;
        int n = (c + 1) % 2;
        
        mbarrier_wait_fn(&mbar_mma[c], phase_mma[c]);
        phase_mma[c] ^= 1;
        
        if (j > 0) {
            uint32_t O_tmem_prev = tmem_base + 256 + n * 128;
            #pragma unroll
            for(int k = 0; k < 128; k += 16) {
                uint32_t r0[8], r1[8];
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(r0[0]),"=r"(r0[1]),"=r"(r0[2]),"=r"(r0[3]),"=r"(r0[4]),"=r"(r0[5]),"=r"(r0[6]),"=r"(r0[7]) : "r"(O_tmem_prev + k));
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(r1[0]),"=r"(r1[1]),"=r"(r1[2]),"=r"(r1[3]),"=r"(r1[4]),"=r"(r1[5]),"=r"(r1[6]),"=r"(r1[7]) : "r"(O_tmem_prev + k + 8));
                tmem_load_fence_fn();
                #pragma unroll
                for(int idx=0; idx<8; idx++) {
                    O_i[k+idx] += __uint_as_float(r0[idx]);
                    O_i[k+8+idx] += __uint_as_float(r1[idx]);
                }
            }
        }
        
        if (j > 0 && j + 128 < S) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar_V[n], 32768);
                tma_load_3d_cg1_fn(&tma_V, &mbar_V[n], V_smem_b + n * 32768, 0, j + 128, b * H + h);
                tma_load_3d_cg1_fn(&tma_V, &mbar_V[n], V_smem_b + n * 32768 + 16384, 64, j + 128, b * H + h);
            }
        }
        
        if (j + 256 < S) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar_K[c], 32768);
                tma_load_3d_cg1_fn(&tma_K, &mbar_K[c], K_smem_b + c * 32768, 0, j + 256, b * H + h);
                tma_load_3d_cg1_fn(&tma_K, &mbar_K[c], K_smem_b + c * 32768 + 16384, 64, j + 256, b * H + h);
            }
        }
        
        if (j + 128 < S) {
            mbarrier_wait_fn(&mbar_K[n], phase_K[n]);
            phase_K[n] ^= 1;
            
            if (threadIdx.x == 0) {
                for(int k=0; k<8; k++) {
                    uint32_t offset = (k < 4) ? (k * 32) : (16384 + (k - 4) * 32);
                    uint64_t desc_A = make_smem_desc(Q_smem_b + offset, 0, 1024);
                    uint64_t desc_B = make_smem_desc(K_smem_b + n * 32768 + offset, 0, 1024);
                    uint32_t accum = (k == 0) ? 0 : 1;
                    umma_f16_cg1_fn(tmem_base + n * 128, desc_A, desc_B, idesc_QK, accum);
                }
            }
        }

        uint32_t S_tmem_c = tmem_base + c * 128;
        float m_j = -INFINITY;
        #pragma unroll
        for(int k = 0; k < 128; k += 16) {
            uint32_t r0[8], r1[8];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(r0[0]),"=r"(r0[1]),"=r"(r0[2]),"=r"(r0[3]),"=r"(r0[4]),"=r"(r0[5]),"=r"(r0[6]),"=r"(r0[7]) : "r"(S_tmem_c + k));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(r1[0]),"=r"(r1[1]),"=r"(r1[2]),"=r"(r1[3]),"=r"(r1[4]),"=r"(r1[5]),"=r"(r1[6]),"=r"(r1[7]) : "r"(S_tmem_c + k + 8));
            tmem_load_fence_fn();
            #pragma unroll
            for(int idx=0; idx<8; idx++) {
                float v0 = __uint_as_float(r0[idx]) * scale;
                float v1 = __uint_as_float(r1[idx]) * scale;
                v0 = (j + k + idx >= S) ? -INFINITY : v0;
                v1 = (j + k + 8 + idx >= S) ? -INFINITY : v1;
                m_j = fmaxf(m_j, fmaxf(v0, v1));
            }
        }
        
        float m_prev = m_i;
        m_i = max(m_prev, m_j);
        float exp_diff = (m_prev == -INFINITY && m_i == -INFINITY) ? 0.0f : exp2f_emulate((m_prev - m_i) * 1.44269504f);
        
        #pragma unroll
        for(int k=0; k<128; k++) O_i[k] *= exp_diff;
        
        float sum_j = 0;
        uint32_t base_addr = (uint32_t)__cvta_generic_to_shared(P_smem_b + c * 32768);
        int row_group = threadIdx.x / 8;
        int row_in_group = threadIdx.x % 8;
        uint32_t thread_base_addr = row_group * 1024 + row_in_group * 128;
        
        #pragma unroll
        for(int k = 0; k < 128; k += 16) {
            uint32_t r0[8], r1[8];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(r0[0]),"=r"(r0[1]),"=r"(r0[2]),"=r"(r0[3]),"=r"(r0[4]),"=r"(r0[5]),"=r"(r0[6]),"=r"(r0[7]) : "r"(S_tmem_c + k));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(r1[0]),"=r"(r1[1]),"=r"(r1[2]),"=r"(r1[3]),"=r"(r1[4]),"=r"(r1[5]),"=r"(r1[6]),"=r"(r1[7]) : "r"(S_tmem_c + k + 8));
            tmem_load_fence_fn();
            
            uint32_t packed0[4], packed1[4];
            #pragma unroll
            for(int idx=0; idx<4; idx++) {
                float v0_0 = __uint_as_float(r0[2*idx]) * scale;
                float v0_1 = __uint_as_float(r0[2*idx+1]) * scale;
                float in0_0 = (j + k + 2*idx >= S) ? -50.0f : (v0_0 - m_i) * 1.44269504f;
                float in0_1 = (j + k + 2*idx+1 >= S) ? -50.0f : (v0_1 - m_i) * 1.44269504f;
                float p0_0 = exp2f_emulate(in0_0);
                float p0_1 = exp2f_emulate(in0_1);
                sum_j += p0_0 + p0_1;
                packed0[idx] = pack_bf16_fn(__float_as_uint(p0_0), __float_as_uint(p0_1));
                
                float v1_0 = __uint_as_float(r1[2*idx]) * scale;
                float v1_1 = __uint_as_float(r1[2*idx+1]) * scale;
                float in1_0 = (j + k + 8 + 2*idx >= S) ? -50.0f : (v1_0 - m_i) * 1.44269504f;
                float in1_1 = (j + k + 8 + 2*idx+1 >= S) ? -50.0f : (v1_1 - m_i) * 1.44269504f;
                float p1_0 = exp2f_emulate(in1_0);
                float p1_1 = exp2f_emulate(in1_1);
                sum_j += p1_0 + p1_1;
                packed1[idx] = pack_bf16_fn(__float_as_uint(p1_0), __float_as_uint(p1_1));
            }
            
            int chunk0 = k / 64;
            int x_16b0 = (k % 64) / 8;
            uint32_t addr0 = base_addr + chunk0 * 16384 + thread_base_addr + (row_in_group ^ x_16b0) * 16;
            st_shared_128_fn(addr0, packed0[0], packed0[1], packed0[2], packed0[3]);
            
            int k1 = k + 8;
            int chunk1 = k1 / 64;
            int x_16b1 = (k1 % 64) / 8;
            uint32_t addr1 = base_addr + chunk1 * 16384 + thread_base_addr + (row_in_group ^ x_16b1) * 16;
            st_shared_128_fn(addr1, packed1[0], packed1[1], packed1[2], packed1[3]);
        }
        l_i = l_i * exp_diff + sum_j;
        
        mbarrier_wait_fn(&mbar_V[c], phase_V[c]);
        phase_V[c] ^= 1;
        
        __syncthreads();
        
        if (threadIdx.x == 0) {
            fence_proxy_async_fn();
            uint32_t O_tmem_c = tmem_base + 256 + c * 128; 
            
            for(int k=0; k<8; k++) {
                uint32_t offset_A = (k < 4) ? (k * 32) : (16384 + (k - 4) * 32);
                uint64_t desc_A = make_smem_desc(P_smem_b + c * 32768 + offset_A, 0, 1024);
                uint32_t offset_B = k * 2048;
                uint64_t desc_B = make_smem_desc(V_smem_b + c * 32768 + offset_B, 16384, 1024);
                uint32_t accum = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(O_tmem_c, desc_A, desc_B, idesc_PV_64, accum);
            }
            for(int k=0; k<8; k++) {
                uint32_t offset_A = (k < 4) ? (k * 32) : (16384 + (k - 4) * 32);
                uint64_t desc_A = make_smem_desc(P_smem_b + c * 32768 + offset_A, 0, 1024);
                uint32_t offset_B = 16384 + k * 2048;
                uint64_t desc_B = make_smem_desc(V_smem_b + c * 32768 + offset_B, 16384, 1024);
                uint32_t accum = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(O_tmem_c + 64, desc_A, desc_B, idesc_PV_64, accum);
            }
            umma_commit_cg1_fn(&mbar_mma[n]);
        }
    }

    int last_c = ((S - 1) / 128) % 2;
    int last_n = (last_c + 1) % 2;
    mbarrier_wait_fn(&mbar_mma[last_n], phase_mma[last_n]);
    
    uint32_t O_tmem_prev = tmem_base + 256 + last_c * 128;
    #pragma unroll
    for(int k = 0; k < 128; k += 16) {
        uint32_t r0[8], r1[8];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(r0[0]),"=r"(r0[1]),"=r"(r0[2]),"=r"(r0[3]),"=r"(r0[4]),"=r"(r0[5]),"=r"(r0[6]),"=r"(r0[7]) : "r"(O_tmem_prev + k));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(r1[0]),"=r"(r1[1]),"=r"(r1[2]),"=r"(r1[3]),"=r"(r1[4]),"=r"(r1[5]),"=r"(r1[6]),"=r"(r1[7]) : "r"(O_tmem_prev + k + 8));
        tmem_load_fence_fn();
        #pragma unroll
        for(int idx=0; idx<8; idx++) {
            O_i[k+idx] += __uint_as_float(r0[idx]);
            O_i[k+8+idx] += __uint_as_float(r1[idx]);
        }
    }

    __nv_bfloat16* O_smem_linear = (__nv_bfloat16*)smem_buf;
    int stride = 136; 
    #pragma unroll 4
    for(int c = 0; c < 128; c++) {
        O_smem_linear[threadIdx.x * stride + c] = __float2bfloat16(O_i[c] / l_i);
    }
    __syncthreads();

    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    for (uint32_t step = 0; step < 32; ++step) {
        uint32_t row = step * 4 + warp_id;
        uint32_t col_start = lane_id * 4;
        
        if (i * 128 + row < S) {
            uint64_t global_row = b * H * S + h * S + i * 128 + row;
            uint64_t global_col = col_start;
            uint2 data = *reinterpret_cast<uint2*>(&O_smem_linear[row * stride + col_start]);
            *reinterpret_cast<uint2*>(O_global + global_row * 128 + global_col) = data;
        }
    }

    if (i * 128 + threadIdx.x < S) {
        int global_idx = b * H * S + h * S + i * 128 + threadIdx.x;
        LSE_global[global_idx] = m_i + logf(l_i);
    }

    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_base, 512);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    CUtensorMap tma_Q, tma_K, tma_V;
    
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, S, B * H, 64, 128, 1));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_K, K.data_ptr(), 128, S, B * H, 64, 128, 1));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_V, V.data_ptr(), 128, S, B * H, 64, 128, 1));
        
    int num_blocks_S = (S + 127) / 128;
    dim3 grid(num_blocks_S, H, B);
    dim3 block(128);
    
    size_t smem_bytes = 230400; 
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    
    mha_fwd_kernel<<<grid, block, smem_bytes, stream>>>(tma_Q, tma_K, tma_V, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        B, H, S);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda