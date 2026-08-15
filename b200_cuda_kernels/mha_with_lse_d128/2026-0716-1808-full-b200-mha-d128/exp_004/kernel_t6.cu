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

namespace tvm_ffi_example_cuda {

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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
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
        ".mbarrier::arrive::one.shared::cluster.b64"
        " [%0];"
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
    uint64_t base_offset = (addr >> 7) & 0x7;
    d |= (base_offset << 49);
    return d;
}

template <uint32_t M, uint32_t N, uint32_t A_MAJOR, uint32_t B_MAJOR>
__device__ __forceinline__ uint32_t make_instr_desc_fn() {
    uint32_t d = 0;
    d |= (1u << 4);     
    d |= (1u << 7);     
    d |= (1u << 10);    
    d |= (A_MAJOR << 15);   
    d |= (B_MAJOR << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ uint64_t desc_k_major_128b(void* smem_ptr) {
    return make_smem_desc_sm100_fn(smem_ptr, 1, 1024);
}

__device__ __forceinline__ uint64_t desc_mn_major_128b(void* smem_ptr) {
    return make_smem_desc_sm100_fn(smem_ptr, 16384, 1024);
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t swizzle_128B_bf16(uint32_t row, uint32_t col) {
    uint32_t chunk_idx = col / 8;
    uint32_t swizzled_chunk_idx = (row % 8) ^ chunk_idx;
    return swizzled_chunk_idx * 8 + (col % 8);
}

struct KVBuffer {
    __nv_bfloat16* smem_K_0;
    __nv_bfloat16* smem_K_1;
    __nv_bfloat16* smem_V_0;
    __nv_bfloat16* smem_V_1;
};

__global__ __launch_bounds__(128, 1) void attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O_gmem,
    float* LSE_gmem,
    uint32_t S,
    uint32_t stride_O,
    uint32_t stride_BH)
{
    extern __shared__ __align__(1024) char smem_pool[];
    
    // Memory layout definition
    __nv_bfloat16* smem_Q_0 = (__nv_bfloat16*)smem_pool;
    __nv_bfloat16* smem_Q_1 = smem_Q_0 + 128 * 64;
    
    // Double buffering applied to K and V tensors
    __nv_bfloat16* smem_K_0 = smem_Q_1 + 128 * 64;
    __nv_bfloat16* smem_K_1 = smem_K_0 + 2 * 128 * 64;
    __nv_bfloat16* smem_V_0 = smem_K_1 + 2 * 128 * 64;
    __nv_bfloat16* smem_V_1 = smem_V_0 + 2 * 128 * 64;
    
    // Reuse unused memory regions safely 
    __nv_bfloat16* smem_P_0 = smem_V_1 + 2 * 128 * 64;
    __nv_bfloat16* smem_P_1 = smem_P_0 + 128 * 64;
    
    uint64_t* mbar_kv0 = (uint64_t*)(smem_P_1 + 128 * 64);
    uint64_t* mbar_kv1 = mbar_kv0 + 1;
    
    __shared__ __align__(4) uint32_t tmem_c_pool[2];

    uint32_t tid = threadIdx.x;
    uint32_t cta_id = cluster_rank() % 2;
    uint32_t s_block = blockIdx.x;
    uint32_t bh_idx = blockIdx.y;
    uint32_t q_offset = s_block * 256 + cta_id * 128;

    if (tid == 0) {
        tmem_alloc_fn(&tmem_c_pool[0], 128);
        tmem_alloc_fn(&tmem_c_pool[1], 128);
    }
    __syncthreads();
    
    uint32_t tmem_c_0 = tmem_c_pool[0];
    uint32_t tmem_c_1 = tmem_c_pool[1];

    KVBuffer buf[2];
    buf[0].smem_K_0 = smem_K_0;
    buf[0].smem_K_1 = smem_K_1;
    buf[0].smem_V_0 = smem_V_0;
    buf[0].smem_V_1 = smem_V_1;
    
    buf[1].smem_K_0 = smem_K_0 + 128 * 64;
    buf[1].smem_K_1 = smem_K_1 + 128 * 64;
    buf[1].smem_V_0 = smem_V_0 + 128 * 64;
    buf[1].smem_V_1 = smem_V_1 + 128 * 64;

    if (tid == 0) {
        init_smem_barrier_fn(mbar_kv0, 1);
        init_smem_barrier_fn(mbar_kv1, 1);
        
        mbarrier_arrive_and_expect_tx_fn(mbar_kv0, 4 * 128 * 64 * sizeof(__nv_bfloat16));
        tma_load_3d_fn(&tma_K, mbar_kv0, buf[0].smem_K_0, 0, 0, bh_idx);
        tma_load_3d_fn(&tma_K, mbar_kv0, buf[0].smem_K_1, 64, 0, bh_idx);
        tma_load_3d_fn(&tma_V, mbar_kv0, buf[0].smem_V_0, 0, 0, bh_idx);
        tma_load_3d_fn(&tma_V, mbar_kv0, buf[0].smem_V_1, 64, 0, bh_idx);
    }

    uint32_t phase_kv[2] = {0, 0};

    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_kv0, 2 * 128 * 64 * sizeof(__nv_bfloat16));
        tma_load_3d_fn(&tma_Q, mbar_kv0, smem_Q_0, 0, q_offset, bh_idx);
        tma_load_3d_fn(&tma_Q, mbar_kv0, smem_Q_1, 64, q_offset, bh_idx);
    }
    mbarrier_wait_fn(mbar_kv0, phase_kv[0]);
    phase_kv[0] ^= 1;

    float global_max = -INFINITY;
    float global_sum = 0.0f;

    uint32_t idesc_QK = make_instr_desc_fn<128, 128, 0, 0>();
    uint32_t idesc_PV = make_instr_desc_fn<128, 64, 0, 1>();
    
    float scale = 0.08838834764f; // 1 / sqrt(128)

    for (uint32_t ks_offset = 0; ks_offset < S; ks_offset += 128) {
        uint32_t buf_idx = ks_offset / 128 % 2;
        uint32_t next_buf_idx = (ks_offset / 128 + 1) % 2;
        KVBuffer* cur_kv = &buf[buf_idx];
        KVBuffer* next_kv = &buf[next_buf_idx];
        
        mbarrier_wait_fn(mbar_kv0, phase_kv[0]);
        phase_kv[0] ^= 1;
        
        fence_proxy_async_fn();

        // First QK^T tile
        if (tid == 0) {
            for (int k = 0; k < 64; k += 16) {
                uint64_t d_Q0 = desc_k_major_128b((char*)smem_Q_0 + k * 2);
                uint64_t d_K0 = desc_k_major_128b((char*)cur_kv->smem_K_0 + k * 2);
                
                umma_f16_cg2_fn(tmem_c_0, d_Q0, d_K0, idesc_QK, (k == 0) ? 0 : 1);
            }
            for (int k = 0; k < 64; k += 16) {
                uint64_t d_Q1 = desc_k_major_128b((char*)smem_Q_1 + k * 2);
                uint64_t d_K1 = desc_k_major_128b((char*)cur_kv->smem_K_1 + k * 2);
                
                umma_f16_cg2_fn(tmem_c_0, d_Q1, d_K1, idesc_QK, 1);
            }
        }
        umma_commit_2sm_fn(mbar_kv0);
        mbarrier_wait_fn(mbar_kv0, phase_kv[0]);
        phase_kv[0] ^= 1;

        float S_vals_0[64], S_vals_1[64];
        for (int col = 0; col < 64; col += 4) {
            uint32_t r[4];
            uint32_t col_base = (tmem_c_0 & 0xFFFF) | ((tid << 16) & 0xFFFF0000);
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(col_base + col));
            S_vals_0[col] = __uint_as_float(r[0]);
            S_vals_0[col+1] = __uint_as_float(r[1]);
            S_vals_0[col+2] = __uint_as_float(r[2]);
            S_vals_0[col+3] = __uint_as_float(r[3]);
            
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(col_base + col + 64));
            S_vals_1[col] = __uint_as_float(r[0]);
            S_vals_1[col+1] = __uint_as_float(r[1]);
            S_vals_1[col+2] = __uint_as_float(r[2]);
            S_vals_1[col+3] = __uint_as_float(r[3]);
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float row_max = -INFINITY;
        for (int c = 0; c < 64; ++c) {
            if (ks_offset + c >= S) S_vals_0[c] = -INFINITY;
            else row_max = fmaxf(row_max, S_vals_0[c] * scale);
        }
        for (int c = 0; c < 64; ++c) {
            if (ks_offset + c + 64 >= S) S_vals_1[c] = -INFINITY;
            else row_max = fmaxf(row_max, S_vals_1[c] * scale);
        }
        
        float new_max = fmaxf(global_max, row_max);
        float rescale = fast_exp2f_fn((global_max - new_max) * 1.44269504089f);
        
        // Optimization: skip rescale for jumps < 1e-3
        if (new_max - global_max < 0.001f && global_max > -INFINITY) {
            rescale = 1.0f;
            new_max = global_max;
        }
        
        global_sum *= rescale;
        for (int i = 0; i < 64; ++i) {
            S_vals_0[i] *= rescale;
            S_vals_1[i] *= rescale;
        }
        
        float row_sum = 0.0f;
        for (int c = 0; c < 64; ++c) {
            if (ks_offset + c >= S) {
                S_vals_0[c] = 0.0f;
            } else {
                S_vals_0[c] = fast_exp2f_fn((S_vals_0[c] * scale - new_max) * 1.44269504089f);
            }
            row_sum += S_vals_0[c];
        }
        for (int c = 0; c < 64; ++c) {
            if (ks_offset + c + 64 >= S) {
                S_vals_1[c] = 0.0f;
            } else {
                S_vals_1[c] = fast_exp2f_fn((S_vals_1[c] * scale - new_max) * 1.44269504089f);
            }
            row_sum += S_vals_1[c];
        }
        global_sum += row_sum;
        global_max = new_max;
        
        __syncthreads(); 
        
        for (int c = 0; c < 64; ++c) {
            __nv_bfloat16 p0 = __float2bfloat16(S_vals_0[c]);
            __nv_bfloat16 p1 = __float2bfloat16(S_vals_1[c]);
            
            uint32_t swizzled_c = swizzle_128B_bf16(tid, c);
            smem_P_0[tid * 64 + swizzled_c] = p0;
            smem_P_1[tid * 64 + swizzled_c] = p1;
        }
        
        __syncthreads();
        fence_proxy_async_fn();

        // First PV tile
        if (tid == 0) {
            for (int k = 0; k < 64; k += 16) {
                uint64_t d_P0 = desc_k_major_128b((char*)smem_P_0 + k * 2);
                uint64_t d_P1 = desc_k_major_128b((char*)smem_P_1 + k * 2);
                
                uint64_t d_V0 = desc_mn_major_128b((char*)cur_kv->smem_V_0 + k * 128);
                uint64_t d_V1 = desc_mn_major_128b((char*)cur_kv->smem_V_1 + k * 128);
                
                if (cta_id == 0) {
                    umma_f16_cg2_fn(tmem_c_0, d_P0, d_V0, idesc_PV, (k == 0) ? 0 : 1);
                    umma_f16_cg2_fn(tmem_c_1, d_P1, d_V1, idesc_PV, (k == 0) ? 0 : 1);
                } else {
                    umma_f16_cg2_fn(tmem_c_0, d_P0, d_V0, idesc_PV, (k == 0) ? 0 : 1);
                    umma_f16_cg2_fn(tmem_c_1, d_P1, d_V1, idesc_PV, (k == 0) ? 0 : 1);
                }
            }
        }
        umma_commit_2sm_fn(mbar_kv0);
        mbarrier_wait_fn(mbar_kv0, phase_kv[0]);
        phase_kv[0] ^= 1;
        
        __syncthreads();
        
        // Prefetch asynchronous load for the next iteration chunk to overlap compute with memory latency fetches
        if (ks_offset + 128 < S && tid == 0) {
            uint64_t* bar = (buf_idx == 0) ? mbar_kv0 : mbar_kv1;
            mbarrier_arrive_and_expect_tx_fn(bar, 4 * 128 * 64 * sizeof(__nv_bfloat16));
            tma_load_3d_fn(&tma_K, bar, next_kv->smem_K_0, 0, ks_offset + 128, bh_idx);
            tma_load_3d_fn(&tma_K, bar, next_kv->smem_K_1, 64, ks_offset + 128, bh_idx);
            tma_load_3d_fn(&tma_V, bar, next_kv->smem_V_0, 0, ks_offset + 128, bh_idx);
            tma_load_3d_fn(&tma_V, bar, next_kv->smem_V_1, 64, ks_offset + 128, bh_idx);
        }
    }

    float final_O_0[64], final_O_1[64];
    for (int col = 0; col < 64; col += 4) {
        uint32_t r[4];
        uint32_t col_base_0 = (tmem_c_0 & 0xFFFF) | ((tid << 16) & 0xFFFF0000);
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(col_base_0 + col));
        final_O_0[col] = __uint_as_float(r[0]);
        final_O_0[col+1] = __uint_as_float(r[1]);
        final_O_0[col+2] = __uint_as_float(r[2]);
        final_O_0[col+3] = __uint_as_float(r[3]);
        
        uint32_t col_base_1 = (tmem_c_1 & 0xFFFF) | ((tid << 16) & 0xFFFF0000);
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(col_base_1 + col));
        final_O_1[col] = __uint_as_float(r[0]);
        final_O_1[col+1] = __uint_as_float(r[1]);
        final_O_1[col+2] = __uint_as_float(r[2]);
        final_O_1[col+3] = __uint_as_float(r[3]);
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

    __nv_bfloat16* O_gmem_bh = O_gmem + bh_idx * stride_BH;
    for (int col = 0; col < 64; col += 2) {
        float o0_0 = final_O_0[col] / global_sum;
        float o0_1 = final_O_0[col+1] / global_sum;
        __nv_bfloat16 o_0[2];
        o_0[0] = __float2bfloat16(o0_0);
        o_0[1] = __float2bfloat16(o0_1);
        
        float o1_0 = final_O_1[col] / global_sum;
        float o1_1 = final_O_1[col+1] / global_sum;
        __nv_bfloat16 o_1[2];
        o_1[0] = __float2bfloat16(o1_0);
        o_1[1] = __float2bfloat16(o1_1);
        
        uint32_t row_idx = q_offset + tid;
        if (row_idx < S) {
            *(uint32_t*)&O_gmem_bh[row_idx * stride_O + col] = *(uint32_t*)&o_0;
            *(uint32_t*)&O_gmem_bh[row_idx * stride_O + col + 64] = *(uint32_t*)&o_1;
        }
    }

    uint32_t row_idx = q_offset + tid;
    if (row_idx < S) {
        float* LSE_ptr = LSE_gmem + bh_idx * S + row_idx;
        *LSE_ptr = logf(global_sum) + global_max;
    }

    if (tid == 0) {
        tmem_dealloc_fn(tmem_c_pool[0], 128);
        tmem_dealloc_fn(tmem_c_pool[1], 128);
    }
    __syncthreads();
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
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    uint32_t D = Q.size(3); 

    CUtensorMap tma_Q, tma_K, tma_V;
    create_tma_3d_descriptor_2B(&tma_Q, Q.data_ptr(), D, S, B * H, 64, 128, 1);
    create_tma_3d_descriptor_2B(&tma_K, K.data_ptr(), D, S, B * H, 64, 128, 1);
    create_tma_3d_descriptor_2B(&tma_V, V.data_ptr(), D, S, B * H, 64, 128, 1);

    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    dim3 grid(((S + 255) / 256), B * H, 1);
    dim3 block(128, 1, 1);

    uint32_t smem_bytes = 228000;
    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeClusterDimension, make_cuda_cluster_dim(2, 1, 1)));

    attention_kernel<<<grid, block, smem_bytes, stream>>>(tma_Q, tma_K, tma_V, O_ptr, LSE_ptr, S, D, S * D);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda