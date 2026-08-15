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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        const char* err_str;                                       \
        cuGetErrorString(_e, &err_str);                            \
        fprintf(stderr, "CU error %s at %s:%d\n",                  \
                err_str, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_with_lse_d128 {

// Helper Functions
template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
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

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("barrier.sync.aligned %0, %1;" :: "r"(bar_id), "r"(count));
}

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
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

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
     "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_after_fn() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t swizzle_128B_addr(uint32_t base_addr, uint32_t row, uint32_t col_16B) {
    uint32_t swizzled_col = (row % 8) ^ col_16B;
    return base_addr + row * 128 + swizzled_col * 16;
}

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
    uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3,
    uint32_t box0, uint32_t box1, 
    CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0*2, dim0*dim1*2, dim0*dim1*dim2*2};
    cuuint32_t boxDim[4] = {box0, box1, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, globalAddress, globalDim, globalStrides, boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

extern __shared__ uint8_t smem[];

__global__ void __launch_bounds__(128, 1) mha_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O_global,
    float* __restrict__ LSE_global,
    int seqlen, int H
) {
    setmaxnreg_inc_sync_fn<256>();

    int q_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int batch_idx = blockIdx.z;

    uint32_t smem_offset = 0;
    
    uint64_t* mbar_K = (uint64_t*)(smem + smem_offset); smem_offset += 2 * 8;
    uint64_t* mbar_V = (uint64_t*)(smem + smem_offset); smem_offset += 2 * 8;
    uint64_t* mbar_Q = (uint64_t*)(smem + smem_offset); smem_offset += 8;
    uint64_t* mbar_UMMA = (uint64_t*)(smem + smem_offset); smem_offset += 8;
    
    smem_offset = (smem_offset + 127) & ~127;
    
    void* smem_Q_left = smem + smem_offset; smem_offset += 16384;
    void* smem_Q_right = smem + smem_offset; smem_offset += 16384;
    
    void* smem_K_left[2];
    void* smem_K_right[2];
    void* smem_V00[2];
    void* smem_V01[2];
    void* smem_V10[2];
    void* smem_V11[2];
    
    for(int i=0; i<2; i++) {
        smem_K_left[i] = smem + smem_offset; smem_offset += 16384;
        smem_K_right[i] = smem + smem_offset; smem_offset += 16384;
        smem_V00[i] = smem + smem_offset; smem_offset += 8192;
        smem_V01[i] = smem + smem_offset; smem_offset += 8192;
        smem_V10[i] = smem + smem_offset; smem_offset += 8192;
        smem_V11[i] = smem + smem_offset; smem_offset += 8192;
    }
    
    void* smem_P_left = smem + smem_offset; smem_offset += 16384;
    void* smem_P_right = smem + smem_offset; smem_offset += 16384;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar_K[0], 1);
        init_smem_barrier_fn(&mbar_K[1], 1);
        init_smem_barrier_fn(&mbar_V[0], 1);
        init_smem_barrier_fn(&mbar_V[1], 1);
        init_smem_barrier_fn(&mbar_Q[0], 1);
        init_smem_barrier_fn(&mbar_UMMA[0], 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();

    __shared__ uint32_t smem_tmem_base;
    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(&smem_tmem_base, 256);
    }
    __syncthreads();
    uint32_t tmem_base = smem_tmem_base;

    int k_iters = (seqlen + 127) / 128;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_Q[0], 32768); // 16384 * 2
        tma_load_4d_fn(&tma_Q, &mbar_Q[0], smem_Q_left, 0, q_idx * 128, head_idx, batch_idx);
        tma_load_4d_fn(&tma_Q, &mbar_Q[0], smem_Q_right, 64, q_idx * 128, head_idx, batch_idx);
        
        if (k_iters > 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_K[0], 32768); // 16384 * 2
            tma_load_4d_fn(&tma_K, &mbar_K[0], smem_K_left[0], 0, 0, head_idx, batch_idx);
            tma_load_4d_fn(&tma_K, &mbar_K[0], smem_K_right[0], 64, 0, head_idx, batch_idx);
            
            mbarrier_arrive_and_expect_tx_fn(&mbar_V[0], 32768); // 8192 * 4
            tma_load_4d_fn(&tma_V, &mbar_V[0], smem_V00[0], 0, 0, head_idx, batch_idx);
            tma_load_4d_fn(&tma_V, &mbar_V[0], smem_V01[0], 64, 0, head_idx, batch_idx);
            tma_load_4d_fn(&tma_V, &mbar_V[0], smem_V10[0], 0, 64, head_idx, batch_idx);
            tma_load_4d_fn(&tma_V, &mbar_V[0], smem_V11[0], 64, 64, head_idx, batch_idx);
        }
    }
    
    mbarrier_wait_fn(&mbar_Q[0], 0);

    uint64_t desc_Q_left = make_smem_desc_sm100_fn(smem_Q_left, 1, 1024);
    uint64_t desc_Q_right = make_smem_desc_sm100_fn(smem_Q_right, 1, 1024);
    uint64_t desc_P_left = make_smem_desc_sm100_fn(smem_P_left, 1, 1024);
    uint64_t desc_P_right = make_smem_desc_sm100_fn(smem_P_right, 1, 1024);

    uint32_t idesc_S = (1u<<4) | (1u<<7) | (1u<<10) | (0u<<15) | (0u<<16) | ((128/8)<<17) | ((128/16)<<24);
    uint32_t idesc_O_left = (1u<<4) | (1u<<7) | (1u<<10) | (0u<<15) | (1u<<16) | ((64/8)<<17) | ((128/16)<<24);
    uint32_t idesc_O_right = idesc_O_left;

    float O_acc[128];
    for(int i=0; i<128; i++) O_acc[i] = 0.0f;
    float m_val = -INFINITY;
    float d_val = 0.0f;
    float scale = 0.0883883476f;

    int umma_phase = 0;

    for(int k=0; k<k_iters; k++) {
        int buf = k % 2;
        int next_buf = (k + 1) % 2;
        
        if (k + 1 < k_iters) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(&mbar_K[next_buf], 32768);
                tma_load_4d_fn(&tma_K, &mbar_K[next_buf], smem_K_left[next_buf], 0, (k+1)*128, head_idx, batch_idx);
                tma_load_4d_fn(&tma_K, &mbar_K[next_buf], smem_K_right[next_buf], 64, (k+1)*128, head_idx, batch_idx);
                
                mbarrier_arrive_and_expect_tx_fn(&mbar_V[next_buf], 32768);
                tma_load_4d_fn(&tma_V, &mbar_V[next_buf], smem_V00[next_buf], 0, (k+1)*128, head_idx, batch_idx);
                tma_load_4d_fn(&tma_V, &mbar_V[next_buf], smem_V01[next_buf], 64, (k+1)*128, head_idx, batch_idx);
                tma_load_4d_fn(&tma_V, &mbar_V[next_buf], smem_V10[next_buf], 0, (k+1)*128 + 64, head_idx, batch_idx);
                tma_load_4d_fn(&tma_V, &mbar_V[next_buf], smem_V11[next_buf], 64, (k+1)*128 + 64, head_idx, batch_idx);
            }
        }
        
        mbarrier_wait_fn(&mbar_K[buf], (k / 2) % 2);
        
        uint64_t desc_K_left = make_smem_desc_sm100_fn(smem_K_left[buf], 1, 1024);
        uint64_t desc_K_right = make_smem_desc_sm100_fn(smem_K_right[buf], 1, 1024);
        
        if (threadIdx.x == 0) {
            tcgen05_fence_after_fn();
            
            asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 0;\n" 
                         :: "r"(tmem_base), "l"(desc_Q_left), "l"(desc_K_left), "r"(idesc_S));
            asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1;\n" 
                         :: "r"(tmem_base), "l"(desc_Q_right), "l"(desc_K_right), "r"(idesc_S));
                         
            umma_commit_cg1_fn(&mbar_UMMA[0]);
        }
        
        mbarrier_wait_fn(&mbar_UMMA[0], umma_phase % 2);
        umma_phase++;
        
        float S_row[128];
        for(int i=0; i<16; i++) {
            uint32_t r[8];
            tmem_load_8x_fn(tmem_base + i*8, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            for(int j=0; j<8; j++) S_row[i*8+j] = __uint_as_float(r[j]);
        }
        tmem_load_fence_fn();
        
        float m_local = -INFINITY;
        for(int j=0; j<128; j++) {
            if (k * 128 + j >= seqlen) S_row[j] = -INFINITY;
            else {
                S_row[j] *= scale;
                m_local = fmaxf(m_local, S_row[j]);
            }
        }
        
        float m_new = fmaxf(m_val, m_local);
        float exp_diff = fast_exp2f_fn((m_val - m_new) * 1.44269504f);
        d_val *= exp_diff;
        for(int i=0; i<128; i++) O_acc[i] *= exp_diff;
        m_val = m_new;
        
        float d_local = 0.0f;
        for(int j=0; j<128; j++) {
            float exp_val = fast_exp2f_fn((S_row[j] - m_val) * 1.44269504f);
            S_row[j] = exp_val;
            d_local += exp_val;
        }
        d_val += d_local;
        
        uint32_t smem_P_left_addr = (uint32_t)__cvta_generic_to_shared(smem_P_left);
        uint32_t smem_P_right_addr = (uint32_t)__cvta_generic_to_shared(smem_P_right);
        
        for(int c_blk = 0; c_blk < 2; c_blk++) {
            uint32_t smem_base_addr = (c_blk == 0) ? smem_P_left_addr : smem_P_right_addr;
            for(int j=0; j<64; j+=8) {
                uint32_t packed[4];
                for(int p=0; p<4; p++) {
                    packed[p] = pack_bf16_fn(__float_as_uint(S_row[c_blk*64 + j + p*2]), 
                                             __float_as_uint(S_row[c_blk*64 + j + p*2 + 1]));
                }
                uint32_t col_16B = j / 8;
                uint32_t addr = swizzle_128B_addr(smem_base_addr, threadIdx.x, col_16B);
                st_shared_128_fn(addr, packed[0], packed[1], packed[2], packed[3]);
            }
        }
        fence_async_shared_fn();
        named_barrier_sync_fn(0, 128); 
        
        mbarrier_wait_fn(&mbar_V[buf], (k / 2) % 2);
        
        uint64_t desc_V00 = make_smem_desc_sm100_fn(smem_V00[buf], 8192, 1024);
        uint64_t desc_V01 = make_smem_desc_sm100_fn(smem_V01[buf], 8192, 1024);
        uint64_t desc_V10 = make_smem_desc_sm100_fn(smem_V10[buf], 8192, 1024);
        uint64_t desc_V11 = make_smem_desc_sm100_fn(smem_V11[buf], 8192, 1024);
        
        if (threadIdx.x == 0) {
            tcgen05_fence_after_fn();
            
            asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 0;\n" 
                         :: "r"(tmem_base + 128), "l"(desc_P_left), "l"(desc_V00), "r"(idesc_O_left));
            asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1;\n" 
                         :: "r"(tmem_base + 128), "l"(desc_P_right), "l"(desc_V10), "r"(idesc_O_left));
    
            asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 0;\n" 
                         :: "r"(tmem_base + 192), "l"(desc_P_left), "l"(desc_V01), "r"(idesc_O_right));
            asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1;\n" 
                         :: "r"(tmem_base + 192), "l"(desc_P_right), "l"(desc_V11), "r"(idesc_O_right));
                         
            umma_commit_cg1_fn(&mbar_UMMA[0]);
        }
        
        mbarrier_wait_fn(&mbar_UMMA[0], umma_phase % 2);
        umma_phase++;
        
        for(int i=0; i<8; i++) {
            uint32_t r[8];
            tmem_load_8x_fn(tmem_base + 128 + i*8, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            for(int j=0; j<8; j++) O_acc[i*8+j] += __uint_as_float(r[j]);
        }
        for(int i=0; i<8; i++) {
            uint32_t r[8];
            tmem_load_8x_fn(tmem_base + 192 + i*8, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            for(int j=0; j<8; j++) O_acc[64 + i*8+j] += __uint_as_float(r[j]);
        }
        tmem_load_fence_fn();
    }

    int row = q_idx * 128 + threadIdx.x;
    if (row < seqlen) {
        float out_scale = 1.0f / d_val;
        for(int i=0; i<128; i++) O_acc[i] *= out_scale;
        
        LSE_global[((uint64_t)batch_idx * H + head_idx) * seqlen + row] = m_val + logf(d_val);
    }
    
    __syncthreads();
    
    __nv_bfloat16* smem_O = (__nv_bfloat16*)smem_Q_left; // Safely reuse the 32768 byte contiguous buffer (Q_left + Q_right)
    for(int i=0; i<128; i+=2) {
        uint32_t packed = pack_bf16_fn(__float_as_uint(O_acc[i]), __float_as_uint(O_acc[i+1]));
        int smem_idx = threadIdx.x * 128 + i;
        *(uint32_t*)((uint8_t*)smem_O + smem_idx * 2) = packed;
    }
    __syncthreads();
    
    uint32_t* smem_O_u32 = (uint32_t*)smem_O;
    for(int i=0; i<64; i++) {
        int smem_idx = i * 128 + threadIdx.x;
        int row_o = smem_idx / 64; 
        int col_o = smem_idx % 64; 
        
        int global_row = q_idx * 128 + row_o;
        if (global_row < seqlen) {
            uint64_t global_idx = ((uint64_t)batch_idx * H * seqlen + (uint64_t)head_idx * seqlen + global_row) * 64 + col_o;
            ((uint32_t*)O_global)[global_idx] = smem_O_u32[smem_idx];
        }
    }
    
    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_base, 256);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_Q, Q.data_ptr(), 128, S, H, B, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_K, K.data_ptr(), 128, S, H, B, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_V, V.data_ptr(), 128, S, H, B, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B));
    
    int q_chunks = (S + 127) / 128;
    dim3 grid(q_chunks, H, B);
    dim3 block(128);
    
    int smem_size = 200 * 1024;
    CUDA_CHECK(cudaFuncSetAttribute(mha_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    mha_fwd_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, 
        static_cast<__nv_bfloat16*>(O.data_ptr()), 
        static_cast<float*>(LSE.data_ptr()), 
        S, H
    );
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}