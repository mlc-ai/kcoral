#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <math.h>
#include <stdio.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_none_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)0 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= (a_major << 15);
    d |= (b_major << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void umma_f16_cg1_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__global__ void __launch_bounds__(128) AttentionKernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S) 
{
    setmaxnreg_inc_sync_fn<248>();
    
    int bh = blockIdx.x;
    int s_block = blockIdx.y;
    int s_start = s_block * 128;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    
    extern __shared__ char smem[];
    __nv_bfloat16* Q_smem = (__nv_bfloat16*)smem;
    __nv_bfloat16* K_smem = Q_smem + 128 * 128;
    __nv_bfloat16* V_smem = K_smem + 64 * 128;
    __nv_bfloat16* P_smem = V_smem + 64 * 128;
    
    uint64_t* mbar_Q = (uint64_t*)(smem + 80 * 1024);
    uint64_t* mbar_K = mbar_Q + 1;
    uint64_t* mbar_V = mbar_K + 1;
    uint64_t* mbar_umma = mbar_V + 1;
    
    if (tid == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K, 1);
        init_smem_barrier_fn(mbar_V, 1);
        init_smem_barrier_fn(mbar_umma, 1);
    }
    __syncthreads();
    
    uint32_t tmem_s_ptr, tmem_pv_ptr;
    if (warp_id == 0) {
        tmem_alloc_cg1_fn(&tmem_s_ptr, 64);
        tmem_alloc_cg1_fn(&tmem_pv_ptr, 128);
    }
    __syncthreads();
    uint32_t tmem_s = tmem_s_ptr;
    uint32_t tmem_pv = tmem_pv_ptr;
    
    float O_reg[128];
    for (int d = 0; d < 128; d++) O_reg[d] = 0.0f;
    float m_i = -1e20f;
    float l_i = 0.0f;
    float scale = 1.0f / sqrtf(128.0f);
    
    int k_phase = 0;
    int umma_phase = 0;
    
    for (int k_start = 0; k_start < S; k_start += 64) {
        if (tid == 0) {
            if (k_start == 0) {
                mbarrier_arrive_and_expect_tx_fn(mbar_Q, 128 * 128 * 2);
                tma_load_2d_fn(&tma_Q, mbar_Q, Q_smem, 0, bh * S + s_start);
            }
            mbarrier_arrive_and_expect_tx_fn(mbar_K, 64 * 128 * 2);
            tma_load_2d_fn(&tma_K, mbar_K, K_smem, 0, bh * S + k_start);
            
            mbarrier_arrive_and_expect_tx_fn(mbar_V, 64 * 128 * 2);
            tma_load_2d_fn(&tma_V, mbar_V, V_smem, 0, bh * S + k_start);
        }
        
        if (k_start == 0) mbarrier_wait_fn(mbar_Q, 0);
        mbarrier_wait_fn(mbar_K, k_phase);
        mbarrier_wait_fn(mbar_V, k_phase);
        fence_async_shared_fn();
        
        // S = Q * K^T
        for (int k = 0; k < 8; k++) {
            uint64_t desc_q = make_smem_desc_sm100_none_fn(Q_smem + k * 16, 16, 2048);
            uint64_t desc_k = make_smem_desc_sm100_none_fn(K_smem + k * 16, 16, 2048);
            uint32_t idesc = make_instr_desc_fn(128, 64, 0, 0);
            umma_f16_cg1_fn(tmem_s, desc_q, desc_k, idesc, k > 0 ? 1 : 0);
        }
        if (tid == 0) umma_commit_cg1_fn(mbar_umma);
        mbarrier_wait_fn(mbar_umma, umma_phase);
        umma_phase ^= 1;
        
        // Load S
        float S_reg[64];
        for (int i = 0; i < 64; i += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_s + i, &r0, &r1, &r2, &r3);
            S_reg[i+0] = __uint_as_float(r0);
            S_reg[i+1] = __uint_as_float(r1);
            S_reg[i+2] = __uint_as_float(r2);
            S_reg[i+3] = __uint_as_float(r3);
        }
        tmem_load_fence_fn();
        
        // Softmax
        float m_curr = -1e20f;
        int valid_k = S - k_start;
        if (valid_k > 64) valid_k = 64;
        
        for (int j = 0; j < 64; j++) {
            if (j < valid_k) {
                S_reg[j] *= scale;
                if (S_reg[j] > m_curr) m_curr = S_reg[j];
            } else {
                S_reg[j] = -1e20f;
            }
        }
        
        float m_new = fmaxf(m_i, m_curr);
        float scale_old = expf(m_i - m_new);
        float l_curr = 0.0f;
        
        for (int j = 0; j < 64; j++) {
            float p = expf(S_reg[j] - m_new);
            S_reg[j] = p; // This is P_reg
            l_curr += p;
        }
        
        float l_new = l_i * scale_old + l_curr;
        for (int d = 0; d < 128; d++) {
            O_reg[d] *= scale_old;
        }
        m_i = m_new;
        l_i = l_new;
        
        // Store P to SMEM
        for (int i = 0; i < 64; i++) {
            P_smem[tid * 64 + i] = __float2bfloat16(S_reg[i]);
        }
        __syncthreads();
        fence_async_shared_fn();
        
        // PV = P * V
        for (int k = 0; k < 4; k++) {
            uint64_t desc_p = make_smem_desc_sm100_none_fn(P_smem + k * 16, 16, 1024);
            uint64_t desc_v = make_smem_desc_sm100_none_fn(V_smem + k * 16 * 128, 256, 16);
            uint32_t idesc_pv = make_instr_desc_fn(128, 128, 0, 1);
            umma_f16_cg1_fn(tmem_pv, desc_p, desc_v, idesc_pv, k > 0 ? 1 : 0);
        }
        if (tid == 0) umma_commit_cg1_fn(mbar_umma);
        mbarrier_wait_fn(mbar_umma, umma_phase);
        umma_phase ^= 1;
        
        // Accumulate PV into O_reg
        for (int c = 0; c < 128; c += 32) {
            for (int i = 0; i < 32; i += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_load_4x_fn(tmem_pv + c + i, &r0, &r1, &r2, &r3);
                O_reg[c + i + 0] += __uint_as_float(r0);
                O_reg[c + i + 1] += __uint_as_float(r1);
                O_reg[c + i + 2] += __uint_as_float(r2);
                O_reg[c + i + 3] += __uint_as_float(r3);
            }
            tmem_load_fence_fn();
        }
        
        k_phase ^= 1;
        __syncthreads();
    }
    
    if (warp_id == 0) {
        tmem_dealloc_cg1_fn(tmem_s, 64);
        tmem_dealloc_cg1_fn(tmem_pv, 128);
    }
    
    float inv_l = 1.0f / l_i;
    __nv_bfloat16* O_smem = Q_smem; // Reuse Q_smem to coalesce global writes
    for (int d = 0; d < 128; d++) {
        O_smem[tid * 128 + d] = __float2bfloat16(O_reg[d] * inv_l);
    }
    __syncthreads();
    
    uint4* O_gmem_u4 = (uint4*)(O + bh * S * 128 + s_start * 128);
    uint4* O_smem_u4 = (uint4*)O_smem;
    for (int i = 0; i < 16; i++) { 
        int offset = i * 128 + tid;
        int row = offset / 16;
        if (s_start + row < S) {
            O_gmem_u4[offset] = O_smem_u4[offset];
        }
    }
    
    if (s_start + tid < S) {
        LSE[bh * S + s_start + tid] = m_i + logf(l_i);
    }
}

namespace tvm_ffi_mha {

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim, globalStrides, boxDim,
        elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
             
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    __nv_bfloat16* q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V;
    create_tma_2d_descriptor_2B(&tma_Q, q_ptr, 128, B * H * S, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, k_ptr, 128, B * H * S, 128, 64, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, v_ptr, 128, B * H * S, 128, 64, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    dim3 block(128, 1, 1);
    dim3 grid(B * H, (S + 127) / 128, 1);
    int shared_mem_size = 81 * 1024;
    
    CUDA_CHECK(cudaFuncSetAttribute(AttentionKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
    
    AttentionKernel<<<grid, block, shared_mem_size, stream>>>(tma_Q, tma_K, tma_V, o_ptr, lse_ptr, S);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

}  // namespace tvm_ffi_mha