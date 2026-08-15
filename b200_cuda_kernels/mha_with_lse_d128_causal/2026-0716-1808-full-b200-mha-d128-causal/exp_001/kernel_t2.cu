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

#define CU_CHECK_DRIVER(call) do {                                 \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        fprintf(stderr, "CUDA Driver error %d at %s:%d\n",         \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill
    );
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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)), "l"((uint64_t)d),
           "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
       :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
       : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void umma_f16_cg1_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_2sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::2"
        ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
        " [%0], %1;"
        :: "r"(a), "h"((uint16_t)0x3));
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

__device__ __forceinline__ void copy_smem_to_tmem(uint32_t tmem_base, const __nv_bfloat16* smem_0, const __nv_bfloat16* smem_1, int S_len, int abs_m_start, int bh_idx) {
    int tid = threadIdx.x;
    int rows_per_thread = (128 + 3) / 4;
    int logic_row = tid / 4;
    int row = logic_row * 4 + (tid % 4);
    
    if (abs_m_start + row >= S_len) return;
    
    uint32_t* tmem_ptr = (uint32_t*)&tmem_base;
    int log_row = row;
    
    for(int c = 0; c < 64; c++) {
        int x_chunk = c / 8;
        int swizzled_x_chunk = (row % 8) ^ x_chunk;
        int swizzled_col = swizzled_x_chunk * 8 + (c % 8);
        
        __nv_bfloat16 val_0 = smem_0[row * 64 + swizzled_col];
        __nv_bfloat16 val_1 = smem_1[row * 64 + swizzled_col];
        
        uint32_t packed;
        asm volatile("mov.b32 %0, {%1, %2};" : "=r"(packed) : "h"(*(uint16_t*)&val_0), "h"(*(uint16_t*)&val_1));
        tmem_ptr[log_row * 128 + c] = packed;
    }
}

__device__ __forceinline__ void softmax_step(const float* s_shared, __nv_bfloat16* smem_p, float* o_acc, float* m_val, float* l_val, int abs_m_start, int abs_n_start, int S_len, float scale, int tid) {
    int row = tid;
    int abs_row = abs_m_start + row;
    
    float row_max = -INFINITY;
    for (int col = 0; col < 128; col++) {
        int abs_col = abs_n_start + col;
        if (abs_row >= S_len || abs_col > abs_row || abs_col >= S_len) {
            row_max = fmaxf(row_max, -INFINITY);
        } else {
            row_max = fmaxf(row_max, s_shared[row * 128 + col] * scale);
        }
    }
    
    float new_m = fmaxf(*m_val, row_max);
    float o_s = __expf(*m_val - new_m);
    
    float row_sum = 0;
    for (int col = 0; col < 128; col++) {
        int abs_col = abs_n_start + col;
        float p = 0.0f;
        if (!(abs_row >= S_len || abs_col > abs_row || abs_col >= S_len)) {
            p = __expf(s_shared[row * 128 + col] * scale - new_m);
            row_sum += p;
        }
        
        int span = col / 64;
        int chunk_in_span = (col % 64) / 8;
        int swizzled_chunk = chunk_in_span ^ (row % 8);
        int swizzled_col = span * 64 + swizzled_chunk * 8 + (col % 8);
        smem_p[row * 128 + swizzled_col] = __float2bfloat16(p);
    }
    
    *l_val = (*l_val) * o_s + row_sum * __expf(row_max - new_m);
    *m_val = new_m;
    
    for (int d = 0; d < 128; d++) {
        o_acc[d] *= o_s;
    }
}

__global__ void __launch_bounds__(128, 1) attention_forward_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE, int S_len, float scale)
{
    int m_block = blockIdx.x;
    int bh_idx = blockIdx.y;
    int abs_m_start = m_block * 128;
    
    if (abs_m_start >= S_len) return;
    
    extern __shared__ __align__(1024) char smem_pool[];
    char* smem_ptr = smem_pool;
    
    __nv_bfloat16* smem_q_0 = (__nv_bfloat16*)smem_ptr; smem_ptr += 16384;
    __nv_bfloat16* smem_q_1 = (__nv_bfloat16*)smem_ptr; smem_ptr += 16384;
    __nv_bfloat16* smem_k_0 = (__nv_bfloat16*)smem_ptr; smem_ptr += 16384;
    __nv_bfloat16* smem_k_1 = (__nv_bfloat16*)smem_ptr; smem_ptr += 16384;
    __nv_bfloat16* smem_v_0 = (__nv_bfloat16*)smem_ptr; smem_ptr += 16384;
    __nv_bfloat16* smem_v_1 = (__nv_bfloat16*)smem_ptr; smem_ptr += 16384;
    __nv_bfloat16* smem_p   = (__nv_bfloat16*)smem_ptr; smem_ptr += 32768;
    uint64_t* smem_mbarrier = (uint64_t*)smem_ptr;
    
    uint32_t* tmem_Q_ptr = (uint32_t*)smem_ptr; smem_ptr += sizeof(uint32_t);
    uint32_t* tmem_S_ptr = (uint32_t*)smem_ptr; smem_ptr += sizeof(uint32_t);
    uint32_t* tmem_P_ptr = (uint32_t*)smem_ptr; smem_ptr += sizeof(uint32_t);
    uint32_t* tmem_O_ptr = (uint32_t*)smem_ptr; smem_ptr += sizeof(uint32_t);
    
    if (threadIdx.x == 0) {
        tmem_alloc_fn(tmem_Q_ptr, 256);
        tmem_alloc_fn(tmem_S_ptr, 256);
        tmem_alloc_fn(tmem_P_ptr, 256);
        tmem_alloc_fn(tmem_O_ptr, 256);
        
        init_smem_barrier_fn(smem_mbarrier, 1);
        
        uint32_t q_bytes = 2 * 128 * 64 * sizeof(__nv_bfloat16); 
        mbarrier_arrive_and_expect_tx_fn(smem_mbarrier, q_bytes);
        
        tma_load_2d_fn(&tma_Q, smem_mbarrier, smem_q_0, 0, bh_idx * S_len + abs_m_start);
        tma_load_2d_fn(&tma_Q, smem_mbarrier, smem_q_1, 64, bh_idx * S_len + abs_m_start);
    }
    __syncthreads();
    
    int tid = threadIdx.x;
    float o_acc[128] = {0};
    float m_val = -INFINITY;
    float l_val = 0;
    
    uint32_t idesc_qkt = 0;
    idesc_qkt |= (1u << 4);
    idesc_qkt |= (1u << 7);
    idesc_qkt |= (1u << 10);
    idesc_qkt |= ((128 / 8) << 17);     
    idesc_qkt |= ((128 / 16) << 24);    
    
    uint32_t idesc_pv_0 = 0;
    idesc_pv_0 |= (1u << 4);
    idesc_pv_0 |= (1u << 7);
    idesc_pv_0 |= (1u << 10);
    idesc_pv_0 |= ((64 / 8) << 17);     
    idesc_pv_0 |= ((128 / 16) << 24);    
    
    uint32_t accum = 1;
    uint32_t phase = 0;
    
    mbarrier_wait_fn(smem_mbarrier, phase);
    phase ^= 1;
    fence_proxy_async_fn();
    
    copy_smem_to_tmem(tmem_Q_ptr[0], smem_q_0, smem_q_1, S_len, abs_m_start, bh_idx);
    __syncthreads();
    
    for (int n_block = 0; n_block <= m_block; n_block++) {
        if (threadIdx.x == 0) {
            uint32_t kv_bytes = 4 * 128 * 64 * sizeof(__nv_bfloat16);
            mbarrier_arrive_and_expect_tx_fn(smem_mbarrier, kv_bytes);
            
            tma_load_2d_fn(&tma_K, smem_mbarrier, smem_k_0, 0, bh_idx * S_len + n_block * 128);
            tma_load_2d_fn(&tma_K, smem_mbarrier, smem_k_1, 64, bh_idx * S_len + n_block * 128);
            tma_load_2d_fn(&tma_V, smem_mbarrier, smem_v_0, 0, bh_idx * S_len + n_block * 128);
            tma_load_2d_fn(&tma_V, smem_mbarrier, smem_v_1, 64, bh_idx * S_len + n_block * 128);
        }
        
        mbarrier_wait_fn(smem_mbarrier, phase);
        phase ^= 1;
        fence_proxy_async_fn();
        
        copy_smem_to_tmem(tmem_Q_ptr[0], smem_q_0, smem_q_1, S_len, abs_m_start, bh_idx);
        copy_smem_to_tmem(tmem_P_ptr[0], smem_k_0, smem_k_1, S_len, n_block * 128, bh_idx); 
        
        __syncthreads(); 
        
        for (uint32_t k_block = 0; k_block < 64; k_block += 16) {
            uint64_t desc_q_0 = make_smem_desc_sm100_fn(smem_q_0, 1, 1024); 
            desc_q_0 += k_block * 2;
            uint64_t desc_k_0 = make_smem_desc_sm100_fn(smem_k_0, 1, 1024);
            desc_k_0 += k_block * 2;
            
            umma_f16_cg1_fn(tmem_S_ptr[0], desc_q_0, desc_k_0, idesc_qkt, accum);
            
            uint64_t desc_q_1 = make_smem_desc_sm100_fn(smem_q_1, 1, 1024);
            desc_q_1 += k_block * 2;
            uint64_t desc_k_1 = make_smem_desc_sm100_fn(smem_k_1, 1, 1024);
            desc_k_1 += k_block * 2;
            
            umma_f16_cg1_fn(tmem_S_ptr[0], desc_q_1, desc_k_1, idesc_qkt, accum);
        }
        
        umma_commit_2sm_fn(smem_mbarrier);
        mbarrier_wait_fn(smem_mbarrier, phase);
        phase ^= 1;
        fence_proxy_async_fn();
        
        float s_shared[128];
        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S_ptr[0] + col, &r0, &r1, &r2, &r3);
            s_shared[tid * 128 + col + 0] = __uint_as_float(r0);
            s_shared[tid * 128 + col + 1] = __uint_as_float(r1);
            s_shared[tid * 128 + col + 2] = __uint_as_float(r2);
            s_shared[tid * 128 + col + 3] = __uint_as_float(r3);
        }
        tmem_load_fence_fn();
        
        softmax_step(s_shared, smem_p, o_acc, &m_val, &l_val, abs_m_start, n_block * 128, S_len, scale, tid);
        __syncthreads(); 
        
        copy_smem_to_tmem(tmem_P_ptr[0], smem_p, smem_p, S_len, abs_m_start, bh_idx); 
        __syncthreads();
        
        for (uint32_t k_block = 0; k_block < 128; k_block += 16) {
            uint64_t desc_p = make_smem_desc_sm100_fn(smem_p, 1, 1024);
            desc_p += k_block * 2;
            
            uint64_t desc_v_0 = make_smem_desc_sm100_fn(smem_v_0, 16384, 1024);
            desc_v_0 += k_block * 128; 
            
            uint64_t desc_v_1 = make_smem_desc_sm100_fn(smem_v_1, 16384, 1024);
            desc_v_1 += k_block * 128;
            
            umma_f16_cg1_fn(tmem_O_ptr[0], desc_p, desc_v_0, idesc_pv_0, accum);
            umma_f16_cg1_fn(tmem_O_ptr[0], desc_p, desc_v_1, idesc_pv_0, accum);
        }
        
        umma_commit_2sm_fn(smem_mbarrier);
        mbarrier_wait_fn(smem_mbarrier, phase);
        phase ^= 1;
        fence_proxy_async_fn();
    }
    
    __syncthreads();
    
    if (abs_m_start + tid < S_len) {
        for (uint32_t col = 0; col < 32; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_O_ptr[0] + col, &r0, &r1, &r2, &r3);
            
            __nv_bfloat16 out[4];
            out[0] = __float2bfloat16(__uint_as_float(r0) / l_val);
            out[1] = __float2bfloat16(__uint_as_float(r1) / l_val);
            out[2] = __float2bfloat16(__uint_as_float(r2) / l_val);
            out[3] = __float2bfloat16(__uint_as_float(r3) / l_val);
            
            int abs_row = abs_m_start + tid;
            *reinterpret_cast<uint2*>(&O[bh_idx * S_len * 128 + abs_row * 128 + col]) = *reinterpret_cast<uint2*>(out);
        }
        
        LSE[bh_idx * S_len + abs_m_start + tid] = m_val + __logf(l_val);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, 
         tvm::ffi::TensorView V, tvm::ffi::TensorView O, 
         tvm::ffi::TensorView LSE) {
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
    
    CU_CHECK_DRIVER(create_tma_2d_descriptor_2B(
        &tma_Q, (void*)Q_ptr, D, B * H * S, 64, 128,
        CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    CU_CHECK_DRIVER(create_tma_2d_descriptor_2B(
        &tma_K, (void*)K_ptr, D, B * H * S, 64, 128,
        CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    CU_CHECK_DRIVER(create_tma_2d_descriptor_2B(
        &tma_V, (void*)V_ptr, D, B * H * S, 64, 128,
        CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    float scale = 1.0f / std::sqrt((float)D);
    
    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128);
    
    int smem_size = 140000; 
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(
        attention_forward_kernel, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        smem_size));
    
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
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, attention_forward_kernel, tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, S, scale));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda