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

namespace causal_attention {

constexpr int H = 48;

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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_alloc_fn_cg2(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
       :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn_cg2(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
       :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ uint32_t get_tmem_addr(uint32_t tmem_base, int lane, int col) {
    return tmem_base + ((lane) << 16) + col;
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t addr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(addr));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void umma_f16_cg2_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
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

__device__ __forceinline__ uint64_t make_smem_desc_cg2(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn_cg2(uint32_t M, uint32_t N) {
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

__device__ __forceinline__ uint32_t make_instr_desc_fn_cg2_mnmaj(uint32_t M, uint32_t N) {
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

__device__ __forceinline__ uint32_t pack_bf16_fn(float a, float b) {
    __nv_bfloat16 ba = __float2bfloat16(a);
    __nv_bfloat16 bb = __float2bfloat16(b);
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&ba)),
          "h"(*reinterpret_cast<uint16_t*>(&bb)));
    return result;
}

__device__ __forceinline__ void tma_load_2d_cg2_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]) & 0xFEFFFFFF;
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(ba) : "memory");
}

__device__ __forceinline__ void tma_store_2d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void gemm_QK_T_128x128x128(uint32_t tmem_S_base, __nv_bfloat16* smem_q, __nv_bfloat16* smem_k) {
    for (int k = 0; k < 8; ++k) {
        uint64_t desc_q = make_smem_desc_cg2(smem_q + k * 16, 1, 1024);
        uint64_t desc_k = make_smem_desc_cg2(smem_k + k * 16, 1, 1024);
        uint32_t accum = (k == 0) ? 0 : 1;
        uint32_t idesc_qkt = make_instr_desc_fn_cg2(128, 128);
        umma_f16_cg2_fn(tmem_S_base, desc_q, desc_k, idesc_qkt, accum);
    }
}

__device__ __forceinline__ void gemm_PV_128x128x128(uint32_t tmem_O_base, __nv_bfloat16* smem_p, __nv_bfloat16* smem_v) {
    for (int k = 0; k < 8; ++k) {
        uint64_t desc_p = make_smem_desc_cg2(smem_p + k * 16, 1, 1024);
        uint64_t desc_v = make_smem_desc_cg2(smem_v + k * 16 * 128, 16384, 1024);
        uint32_t accum = (k == 0) ? 0 : 1;
        uint32_t idesc_pv = make_instr_desc_fn_cg2_mnmaj(128, 128);
        umma_f16_cg2_fn(tmem_O_base, desc_p, desc_v, idesc_pv, accum);
    }
}

__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

__global__ void causal_attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_LSE,
    int S)
{
    setmaxnreg_inc_sync_fn<256>();

    extern __shared__ __align__(1024) char smem[];
    __nv_bfloat16* smem_q = (__nv_bfloat16*)smem;                   
    __nv_bfloat16* smem_k = (__nv_bfloat16*)(smem + 32768);         
    __nv_bfloat16* smem_v = (__nv_bfloat16*)(smem + 65536);         
    __nv_bfloat16* smem_p = (__nv_bfloat16*)(smem + 98304);         
    __nv_bfloat16* smem_o = (__nv_bfloat16*)(smem + 131072);         
    float* smem_lse = (float*)(smem + 163840);                       
    uint64_t* bar0 = (uint64_t*)(smem + 163848);                     

    uint32_t tmem_S_addr, tmem_O_addr;
    if (threadIdx.x == 0) {
        tmem_alloc_fn_cg2(&tmem_S_addr, 128);
        tmem_alloc_fn_cg2(&tmem_O_addr, 128);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar0, 1);
    }
    __syncthreads();
    
    int total_bh = gridDim.x / ((S + 127) / 128);
    int seq_tile_idx = blockIdx.x / total_bh;
    int bh_idx = blockIdx.x % total_bh;
    int b = bh_idx / H;
    int h = bh_idx % H;
    int bh = b * H + h;

    int cta_offset_q = cluster_rank_fn() * 128;
    int s_offset_q = seq_tile_idx * 256 + cta_offset_q;
    
    float global_max = -INFINITY;
    float global_sum = 0.0f;
    float scale = 1.0f / sqrtf(128.0f);
    int phase_bar = 0;
    
    int tid = threadIdx.x;
    int q_idx = s_offset_q + tid;
    
    if (threadIdx.x == 0) {
        uint32_t tx_bytes_q = 32768; 
        mbarrier_arrive_and_expect_tx_fn(bar0, tx_bytes_q);
        fence_proxy_async_fn();
        
        tma_load_2d_cg2_fn(&tma_Q, bar0, smem_q, 0, bh * S + s_offset_q);
        tma_load_2d_cg2_fn(&tma_Q, bar0, smem_q + 8192, 64, bh * S + s_offset_q);
    }
    mbarrier_wait_fn(bar0, phase_bar);
    phase_bar ^= 1;

    uint32_t my_tmem_S = tmem_S_addr + (threadIdx.x << 16);
    uint32_t my_tmem_O = tmem_O_addr + (threadIdx.x << 16);

    int max_k = s_offset_q + 128;
    if (max_k > S) max_k = S;

    for (int s_offset_k = 0; s_offset_k < max_k; s_offset_k += 128) {
        uint32_t tx_bytes_kv = 65536;
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar0, tx_bytes_kv);
            fence_proxy_async_fn();
            tma_load_2d_cg2_fn(&tma_K, bar0, smem_k, 0, bh * S + s_offset_k);
            tma_load_2d_cg2_fn(&tma_K, bar0, smem_k + 8192, 64, bh * S + s_offset_k);
            
            tma_load_2d_cg2_fn(&tma_V, bar0, smem_v, 0, bh * S + s_offset_k);
            tma_load_2d_cg2_fn(&tma_V, bar0, smem_v + 8192, 64, bh * S + s_offset_k);
        }
        mbarrier_wait_fn(bar0, phase_bar);
        phase_bar ^= 1;
        __syncthreads();
        
        fence_proxy_async_fn();
        if (threadIdx.x == 0) {
            gemm_QK_T_128x128x128(tmem_S_addr, smem_q, smem_k);
            umma_commit_2sm_fn(bar0);
        }
        mbarrier_wait_fn(bar0, phase_bar);
        phase_bar ^= 1;
        
        float my_max = -INFINITY;
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(my_tmem_S + col, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            float f0 = __uint_as_float(r0) * scale;
            float f1 = __uint_as_float(r1) * scale;
            float f2 = __uint_as_float(r2) * scale;
            float f3 = __uint_as_float(r3) * scale;
            
            int k_idx0 = s_offset_k + col;
            if (k_idx0 > q_idx || k_idx0 >= S) f0 = -INFINITY;
            if (k_idx0 + 1 > q_idx || k_idx0 + 1 >= S) f1 = -INFINITY;
            if (k_idx0 + 2 > q_idx || k_idx0 + 2 >= S) f2 = -INFINITY;
            if (k_idx0 + 3 > q_idx || k_idx0 + 3 >= S) f3 = -INFINITY;
            
            my_max = fmaxf(my_max, fmaxf(fmaxf(f0, f1), fmaxf(f2, f3)));
        }
        
        float row_max = my_max;
        row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, 1));
        row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, 2));
        row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, 4));
        row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, 8));
        row_max = fmaxf(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, 16));
        
        float curr_max = fmaxf(global_max, row_max);
        float curr_sum = global_sum * (curr_max > -INFINITY && global_max > -INFINITY ? expf(global_max - curr_max) : 0.0f);
        
        float my_sum = 0;
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(my_tmem_S + col, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            float f0 = __uint_as_float(r0) * scale;
            float f1 = __uint_as_float(r1) * scale;
            float f2 = __uint_as_float(r2) * scale;
            float f3 = __uint_as_float(r3) * scale;
            
            int k_idx0 = s_offset_k + col;
            if (k_idx0 > q_idx || k_idx0 >= S) f0 = -INFINITY;
            if (k_idx0 + 1 > q_idx || k_idx0 + 1 >= S) f1 = -INFINITY;
            if (k_idx0 + 2 > q_idx || k_idx0 + 2 >= S) f2 = -INFINITY;
            if (k_idx0 + 3 > q_idx || k_idx0 + 3 >= S) f3 = -INFINITY;
            
            float p0 = (f0 > -INFINITY) ? expf(f0 - curr_max) : 0.0f;
            float p1 = (f1 > -INFINITY) ? expf(f1 - curr_max) : 0.0f;
            float p2 = (f2 > -INFINITY) ? expf(f2 - curr_max) : 0.0f;
            float p3 = (f3 > -INFINITY) ? expf(f3 - curr_max) : 0.0f;
            
            my_sum += p0 + p1 + p2 + p3;
            
            uint32_t p01 = pack_bf16_fn(p0, p1);
            uint32_t p23 = pack_bf16_fn(p2, p3);
            
            int sc0 = ((col / 8) ^ (tid % 8)) * 8 + (col % 8);
            int sc2 = (((col + 2) / 8) ^ (tid % 8)) * 8 + ((col + 2) % 8);
            *(uint32_t*)&smem_p[tid * 128 + sc0] = p01;
            *(uint32_t*)&smem_p[tid * 128 + sc2] = p23;
        }
        
        float row_sum = my_sum;
        row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, 1);
        row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, 2);
        row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, 4);
        row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, 8);
        row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, 16);
        
        curr_sum += row_sum;
        global_sum = curr_sum;
        global_max = curr_max;
        
        __syncthreads();
        fence_proxy_async_fn();
        
        if (threadIdx.x == 0) {
            gemm_PV_128x128x128(tmem_O_addr, smem_p, smem_v);
            umma_commit_2sm_fn(bar0);
        }
        mbarrier_wait_fn(bar0, phase_bar);
        phase_bar ^= 1;
        __syncthreads();
    }
    
    __syncthreads();
    
    int base_off = tid * 128;
    for (int c = 0; c < 128; c++) {
        float f_val = -INFINITY;
        if (c < 128) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(my_tmem_O + c, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            f_val = __uint_as_float(r0);
            if (global_sum > 0.0f) {
                f_val /= global_sum;
            }
        }
        int sc = ((c / 8) ^ (tid % 8)) * 8 + (c % 8);
        smem_o[base_off + sc] = __float2bfloat16(f_val);
    }
    
    if (tid < 128) {
        smem_lse[tid] = (global_sum > 0.0f) ? (global_max + logf(global_sum)) : INFINITY;
    }
    
    __syncthreads();
    
    if (s_offset_q < S) {
        if (threadIdx.x == 0) {
            tma_store_2d_fn(&tma_O, smem_o, 0, bh * S + s_offset_q);
            tma_store_2d_fn(&tma_O, smem_o + 8192, 64, bh * S + s_offset_q);
            tma_store_2d_fn(&tma_LSE, smem_lse, 0, bh * S + s_offset_q);
            tma_store_commit_fn();
            tma_store_wait_fn<0>();
        }
    }
    
    if (threadIdx.x == 0) {
        tmem_dealloc_fn_cg2(tmem_S_addr, 128);
        tmem_dealloc_fn_cg2(tmem_O_addr, 128);
    }
    __syncthreads();
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H_ = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);

    if (H_ != H) {
        fprintf(stderr, "Expected H=%d, got H=%ld\n", H, H_);
        exit(1);
    }

    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    CUtensorMap tma_LSE;

    CUresult res;
    res = create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), D, B * H_ * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA Q failed\n"); exit(1); }
    res = create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), D, B * H_ * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    res = create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), D, B * H_ * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    res = create_tma_2d_descriptor_2B(&tma_O, O.data_ptr(), D, B * H_ * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    cuuint64_t globalDim_LSE[2] = {1, (cuuint64_t)(B * H_ * S)};
    cuuint64_t globalStrides_LSE[1] = {sizeof(float)};
    cuuint32_t boxDim_LSE[2] = {1, 128};
    cuuint32_t elementStrides_LSE[2] = {1, 1};
    res = cuTensorMapEncodeTiled(&tma_LSE, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, LSE.data_ptr(), globalDim_LSE, globalStrides_LSE, boxDim_LSE, elementStrides_LSE, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA LSE failed\n"); exit(1); }

    int64_t blocks = ((S + 127) / 128) * B * H_;
    int64_t threads = 128;
    
    int smem_size = 164000;
    CUDA_CHECK(cudaFuncSetAttribute(
        causal_attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));
        
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(blocks, 1, 1);
    config.blockDim = dim3(threads, 1, 1);
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, causal_attention_kernel,
        tma_Q, tma_K, tma_V, tma_O, tma_LSE, S));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
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

} // namespace causal_attention