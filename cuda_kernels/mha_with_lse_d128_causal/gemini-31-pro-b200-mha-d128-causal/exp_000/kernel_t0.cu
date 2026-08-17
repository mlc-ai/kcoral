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
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tma_store_fence_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
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

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols) : "memory");
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols) : "memory");
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
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
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum) : "memory");
}

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)0 << 61;   // layout_type = NONE
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

__global__ void flash_fwd_sm100_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* LSE_ptr,
    int S)
{
    setmaxnreg_inc_sync_fn<256>();

    int seq_idx = blockIdx.x;
    int h = blockIdx.y;
    int b = blockIdx.z;

    extern __shared__ __align__(128) uint8_t smem_buf[];
    __nv_bfloat16* Q_smem = (__nv_bfloat16*)smem_buf;                   // 32KB
    __nv_bfloat16* K_smem = Q_smem + 128 * 128;                         // 32KB
    __nv_bfloat16* V_smem = K_smem + 128 * 128;                         // 32KB
    __nv_bfloat16* P_smem = V_smem + 128 * 128;                         // 32KB
    __nv_bfloat16* O_smem = P_smem + 128 * 128;                         // 32KB
    
    uint64_t* mbar_q = (uint64_t*)(O_smem + 128 * 128);
    uint64_t* mbar_k = mbar_q + 1;
    uint64_t* mbar_v = mbar_k + 1;
    uint64_t* mbar_umma = mbar_v + 1;

    int tid = threadIdx.x; 

    if (tid == 0) {
        init_smem_barrier_fn(mbar_q, 1);
        init_smem_barrier_fn(mbar_k, 1);
        init_smem_barrier_fn(mbar_v, 1);
        init_smem_barrier_fn(mbar_umma, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_q, 128 * 128 * 2);
        tma_load_4d_fn(&tma_Q, mbar_q, Q_smem, 0, seq_idx * 128, h, b);
    }
    mbarrier_wait_fn(mbar_q, 0); 

    __shared__ uint32_t tmem_addr_S;
    __shared__ uint32_t tmem_addr_O;
    if (tid < 32) {
        tmem_alloc_cg1_fn(&tmem_addr_S, 128);
        tmem_alloc_cg1_fn(&tmem_addr_O, 128);
    }
    __syncthreads();
    
    uint32_t tmem_S = tmem_addr_S;
    uint32_t tmem_O = tmem_addr_O;

    uint64_t desc_Q = make_smem_desc_sm100_fn(Q_smem, 16, 2048);
    uint64_t desc_K = make_smem_desc_sm100_fn(K_smem, 16, 2048);
    uint64_t desc_V = make_smem_desc_sm100_fn(V_smem, 256, 16);
    uint64_t desc_P = make_smem_desc_sm100_fn(P_smem, 16, 2048);
    
    uint32_t idesc_QK = make_instr_desc_fn(128, 128, 0, 0); 
    uint32_t idesc_PV = make_instr_desc_fn(128, 128, 0, 1); 

    float O_reg[128];
    for (int i = 0; i < 128; i++) O_reg[i] = 0.0f;
    float m_val = -1e20f;
    float l_val = 0.0f;
    float scale_s = 0.08838834764f; 

    int K_MAX = seq_idx + 1;
    int k_phase = 0;
    int umma_phase = 0;

    for (int k_idx = 0; k_idx < K_MAX; k_idx++) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_k, 128 * 128 * 2);
            tma_load_4d_fn(&tma_K, mbar_k, K_smem, 0, k_idx * 128, h, b);
            
            mbarrier_arrive_and_expect_tx_fn(mbar_v, 128 * 128 * 2);
            tma_load_4d_fn(&tma_V, mbar_v, V_smem, 0, k_idx * 128, h, b);
        }
        
        mbarrier_wait_fn(mbar_k, k_phase);
        
        if (tid == 0) {
            for (int k = 0; k < 8; k++) {
                uint64_t a_desc = desc_Q + k * 32;
                uint64_t b_desc = desc_K + k * 32;
                uint32_t accum = (k > 0) ? 1 : 0;
                umma_f16_cg1_fn(tmem_S, a_desc, b_desc, idesc_QK, accum);
            }
            umma_commit_cg1_fn(mbar_umma);
        }
        mbarrier_wait_fn(mbar_umma, umma_phase);
        umma_phase++;

        float row_max = -1e20f;
        for (int c = 0; c < 128; c += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S + c, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            float f0 = __uint_as_float(r0) * scale_s;
            float f1 = __uint_as_float(r1) * scale_s;
            float f2 = __uint_as_float(r2) * scale_s;
            float f3 = __uint_as_float(r3) * scale_s;
            
            int global_q = seq_idx * 128 + tid;
            int global_k = k_idx * 128 + c;
            
            if (global_q < global_k + 0) f0 = -1e20f;
            if (global_q < global_k + 1) f1 = -1e20f;
            if (global_q < global_k + 2) f2 = -1e20f;
            if (global_q < global_k + 3) f3 = -1e20f;
            
            row_max = fmaxf(row_max, f0);
            row_max = fmaxf(row_max, f1);
            row_max = fmaxf(row_max, f2);
            row_max = fmaxf(row_max, f3);
        }
        
        float m_new = fmaxf(m_val, row_max);
        float exp_scale = exp2f((m_val - m_new) * 1.44269504f); 
        float l_new = l_val * exp_scale;
        
        for (int c = 0; c < 128; c += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S + c, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            float f0 = __uint_as_float(r0) * scale_s;
            float f1 = __uint_as_float(r1) * scale_s;
            float f2 = __uint_as_float(r2) * scale_s;
            float f3 = __uint_as_float(r3) * scale_s;
            
            int global_q = seq_idx * 128 + tid;
            int global_k = k_idx * 128 + c;
            
            if (global_q < global_k + 0) f0 = -1e20f;
            if (global_q < global_k + 1) f1 = -1e20f;
            if (global_q < global_k + 2) f2 = -1e20f;
            if (global_q < global_k + 3) f3 = -1e20f;
            
            float p0 = exp2f((f0 - m_new) * 1.44269504f);
            float p1 = exp2f((f1 - m_new) * 1.44269504f);
            float p2 = exp2f((f2 - m_new) * 1.44269504f);
            float p3 = exp2f((f3 - m_new) * 1.44269504f);
            
            l_new += p0 + p1 + p2 + p3;
            
            P_smem[tid * 128 + c + 0] = __float2bfloat16(p0);
            P_smem[tid * 128 + c + 1] = __float2bfloat16(p1);
            P_smem[tid * 128 + c + 2] = __float2bfloat16(p2);
            P_smem[tid * 128 + c + 3] = __float2bfloat16(p3);
        }
        
        fence_proxy_async_fn();
        __syncthreads();
        
        mbarrier_wait_fn(mbar_v, k_phase);
        
        if (tid == 0) {
            for (int k = 0; k < 8; k++) {
                uint64_t a_desc = desc_P + k * 32;
                uint64_t b_desc = desc_V + k * 4096; 
                uint32_t accum = (k > 0) ? 1 : 0;
                umma_f16_cg1_fn(tmem_O, a_desc, b_desc, idesc_PV, accum);
            }
            umma_commit_cg1_fn(mbar_umma);
        }
        mbarrier_wait_fn(mbar_umma, umma_phase);
        umma_phase++;
        
        for (int c = 0; c < 128; c += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_O + c, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            O_reg[c+0] = O_reg[c+0] * exp_scale + __uint_as_float(r0);
            O_reg[c+1] = O_reg[c+1] * exp_scale + __uint_as_float(r1);
            O_reg[c+2] = O_reg[c+2] * exp_scale + __uint_as_float(r2);
            O_reg[c+3] = O_reg[c+3] * exp_scale + __uint_as_float(r3);
        }
        
        m_val = m_new;
        l_val = l_new;
        
        k_phase++;
        __syncthreads();
    }
    
    float inv_l = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
    for (int c = 0; c < 128; c += 4) {
        float o0 = O_reg[c+0] * inv_l;
        float o1 = O_reg[c+1] * inv_l;
        float o2 = O_reg[c+2] * inv_l;
        float o3 = O_reg[c+3] * inv_l;
        
        O_smem[tid * 128 + c + 0] = __float2bfloat16(o0);
        O_smem[tid * 128 + c + 1] = __float2bfloat16(o1);
        O_smem[tid * 128 + c + 2] = __float2bfloat16(o2);
        O_smem[tid * 128 + c + 3] = __float2bfloat16(o3);
    }
    
    float lse_val = m_val + logf(l_val);
    
    fence_proxy_async_fn();
    __syncthreads();
    tma_store_fence_fn();
    
    if (tid == 0) {
        tma_store_4d_fn(&tma_O, O_smem, 0, seq_idx * 128, h, b);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    
    if (tid < 32) {
        tmem_dealloc_cg1_fn(tmem_addr_S, 128);
        tmem_dealloc_cg1_fn(tmem_addr_O, 128);
    }
    
    int lse_idx = b * gridDim.y * S + h * S + seq_idx * 128 + tid;
    if (seq_idx * 128 + tid < S) {
        LSE_ptr[lse_idx] = lse_val;
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    if (S <= 0) return;
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    cuuint64_t globalDim[4] = {(cuuint64_t)D, (cuuint64_t)S, (cuuint64_t)H, (cuuint64_t)B};
    cuuint64_t globalStrides[3] = {(cuuint64_t)(D * 2), (cuuint64_t)(D * S * 2), (cuuint64_t)(D * S * H * 2)};
    cuuint32_t boxDim[4] = {128, 128, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    
    CU_CHECK(cuTensorMapEncodeTiled(&tma_Q, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, 
                                    Q.data_ptr(), globalDim, globalStrides, boxDim, elementStrides, 
                                    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, 
                                    CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    CU_CHECK(cuTensorMapEncodeTiled(&tma_K, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, 
                                    K.data_ptr(), globalDim, globalStrides, boxDim, elementStrides, 
                                    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, 
                                    CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
                                    
    CU_CHECK(cuTensorMapEncodeTiled(&tma_V, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, 
                                    V.data_ptr(), globalDim, globalStrides, boxDim, elementStrides, 
                                    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, 
                                    CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
                                    
    CU_CHECK(cuTensorMapEncodeTiled(&tma_O, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, 
                                    O.data_ptr(), globalDim, globalStrides, boxDim, elementStrides, 
                                    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, 
                                    CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
                                    
    int num_seq_blocks = (S + 127) / 128;
    dim3 grid(num_seq_blocks, H, B);
    dim3 block(128); 
    
    int smem_size = 5 * 128 * 128 * 2 + 1024; 
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(flash_fwd_sm100_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, flash_fwd_sm100_kernel, tma_Q, tma_K, tma_V, tma_O, (float*)LSE.data_ptr(), S));
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda