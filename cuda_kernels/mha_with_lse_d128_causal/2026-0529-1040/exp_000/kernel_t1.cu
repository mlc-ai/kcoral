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

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, const void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint32_t box0, uint32_t box1, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, 1};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, (void*)globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
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
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3, uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3),
     "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ void tmem_store_8x_fn(uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3, uint32_t r4, uint32_t r5, uint32_t r6, uint32_t r7) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x8.b32 [%8], {%0,%1,%2,%3,%4,%5,%6,%7};"
   :: "r"(r0),"r"(r1),"r"(r2),"r"(r3),
      "r"(r4),"r"(r5),"r"(r6),"r"(r7), "r"(col));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tcgen05_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a) : "memory");
}

__device__ __forceinline__ void umma_f16_cg1_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
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

__device__ __forceinline__ uint64_t make_smem_desc_none_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)0 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
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

__device__ __forceinline__ uint32_t make_instr_desc_pv_fn(uint32_t M, uint32_t N) {
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

__device__ __forceinline__ uint64_t advance_desc_bytes(uint64_t desc, uint32_t bytes) {
    uint32_t addr = (desc & 0x3FFF) << 4;
    addr += bytes;
    desc &= ~0x3FFF;
    desc |= ((addr & 0x3FFFF) >> 4);
    uint32_t base_offset = (addr >> 7) & 0x7;
    desc &= ~(0x7ull << 49);
    desc |= ((uint64_t)base_offset << 49);
    return desc;
}

__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tma_store_3d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
}

__device__ __forceinline__ void tma_store_fence_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_store_commit_fn() {
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

union SharedMem {
    struct {
        __nv_bfloat16 K0[2][128 * 64];
        __nv_bfloat16 K1[2][128 * 64];
        __nv_bfloat16 V0[2][128 * 64];
        __nv_bfloat16 V1[2][128 * 64];
    } kv;
    __nv_bfloat16 O[128 * 128];
};

__global__ void mha_fwd_kernel_sm100(
    const CUtensorMap __grid_constant__ tma_Q,
    const CUtensorMap __grid_constant__ tma_K,
    const CUtensorMap __grid_constant__ tma_V,
    const CUtensorMap __grid_constant__ tma_O,
    float* __restrict__ LSE,
    int S) 
{
    int batch_head = blockIdx.y;
    int q_start = blockIdx.x * 128;
    int tid = threadIdx.x;

    __shared__ alignas(1024) __nv_bfloat16 smem_Q0[128 * 64];
    __shared__ alignas(1024) __nv_bfloat16 smem_Q1[128 * 64];
    __shared__ alignas(1024) __nv_bfloat16 smem_P[128 * 136];
    __shared__ alignas(1024) SharedMem smem;
    
    __shared__ alignas(8) uint64_t mbar_Q;
    __shared__ alignas(8) uint64_t mbar_K[2];
    __shared__ alignas(8) uint64_t mbar_umma;
    __shared__ uint32_t tmem_P, tmem_O0, tmem_O1;

    if (tid == 0) {
        init_smem_barrier_fn(&mbar_Q, 1);
        init_smem_barrier_fn(&mbar_K[0], 1);
        init_smem_barrier_fn(&mbar_K[1], 1);
        init_smem_barrier_fn(&mbar_umma, 1);
        tmem_alloc_fn(&tmem_P, 128);
        tmem_alloc_fn(&tmem_O0, 64);
        tmem_alloc_fn(&tmem_O1, 64);
        
        mbarrier_arrive_and_expect_tx_fn(&mbar_Q, 32768);
        tma_load_3d_fn(&tma_Q, &mbar_Q, smem_Q0, 0, q_start, batch_head);
        tma_load_3d_fn(&tma_Q, &mbar_Q, smem_Q1, 64, q_start, batch_head);
    }
    __syncthreads();

    uint64_t desc_Q0 = make_smem_desc_sm100_fn(smem_Q0, 1, 1024);
    uint64_t desc_Q1 = make_smem_desc_sm100_fn(smem_Q1, 1, 1024);
    uint64_t desc_P  = make_smem_desc_none_fn(smem_P, 16, 2176);

    float l_vec = 0.0f;
    float m_vec = -1e20f;
    
    int k_phase[2] = {0, 0};
    int umma_phase = 0;

    mbarrier_wait_fn(&mbar_Q, 0);

    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_K[0], 65536);
        tma_load_3d_fn(&tma_K, &mbar_K[0], smem.kv.K0[0], 0, 0, batch_head);
        tma_load_3d_fn(&tma_K, &mbar_K[0], smem.kv.K1[0], 64, 0, batch_head);
        tma_load_3d_fn(&tma_V, &mbar_K[0], smem.kv.V0[0], 0, 0, batch_head);
        tma_load_3d_fn(&tma_V, &mbar_K[0], smem.kv.V1[0], 64, 0, batch_head);
    }

    for (int k_start = 0; k_start <= q_start; k_start += 128) {
        int db = (k_start / 128) % 2;
        int next_k = k_start + 128;
        int next_db = (next_k / 128) % 2;
        
        mbarrier_wait_fn(&mbar_K[db], k_phase[db]);
        
        if (next_k <= q_start && tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_K[next_db], 65536);
            tma_load_3d_fn(&tma_K, &mbar_K[next_db], smem.kv.K0[next_db], 0, next_k, batch_head);
            tma_load_3d_fn(&tma_K, &mbar_K[next_db], smem.kv.K1[next_db], 64, next_k, batch_head);
            tma_load_3d_fn(&tma_V, &mbar_K[next_db], smem.kv.V0[next_db], 0, next_k, batch_head);
            tma_load_3d_fn(&tma_V, &mbar_K[next_db], smem.kv.V1[next_db], 64, next_k, batch_head);
        }

        uint64_t desc_K0 = make_smem_desc_sm100_fn(smem.kv.K0[db], 1, 1024);
        uint64_t desc_K1 = make_smem_desc_sm100_fn(smem.kv.K1[db], 1, 1024);
        uint64_t desc_V0 = make_smem_desc_sm100_fn(smem.kv.V0[db], 8192, 1024);
        uint64_t desc_V1 = make_smem_desc_sm100_fn(smem.kv.V1[db], 8192, 1024);

        uint32_t idesc_P = make_instr_desc_fn(128, 128);
        for (int step = 0; step < 4; ++step) {
            uint64_t d_Q0 = advance_desc_bytes(desc_Q0, step * 32);
            uint64_t d_K0 = advance_desc_bytes(desc_K0, step * 32);
            uint32_t accum = (step == 0) ? 0 : 1;
            umma_f16_cg1_fn(tmem_P, d_Q0, d_K0, idesc_P, accum);
        }
        for (int step = 0; step < 4; ++step) {
            uint64_t d_Q1 = advance_desc_bytes(desc_Q1, step * 32);
            uint64_t d_K1 = advance_desc_bytes(desc_K1, step * 32);
            umma_f16_cg1_fn(tmem_P, d_Q1, d_K1, idesc_P, 1);
        }

        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_umma, 1);
            tcgen05_commit_cg1_fn(&mbar_umma);
        }
        mbarrier_wait_fn(&mbar_umma, umma_phase);
        umma_phase ^= 1;

        float p_row[128];
        for (int c = 0; c < 128; c += 8) {
            uint32_t r[8];
            tmem_load_8x_fn(tmem_P + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
            tmem_load_fence_fn();
            #pragma unroll
            for(int i=0; i<8; ++i) p_row[c+i] = __uint_as_float(r[i]) * 0.08838834764831843f;
        }

        int row_in_global = q_start + tid;
        for (int c = 0; c < 128; ++c) {
            int col_in_global = k_start + c;
            if (col_in_global > row_in_global) p_row[c] = -1e20f;
        }

        float m_new = m_vec;
        for (int c = 0; c < 128; ++c) m_new = fmaxf(m_new, p_row[c]);
        float exp_diff = expf(m_vec - m_new);
        l_vec *= exp_diff;
        for (int c = 0; c < 128; ++c) {
            p_row[c] = expf(p_row[c] - m_new);
            l_vec += p_row[c];
        }
        m_vec = m_new;

        for (int c = 0; c < 128; c += 8) {
            uint32_t b01 = pack_bf16_fn(*(uint32_t*)&p_row[c+0], *(uint32_t*)&p_row[c+1]);
            uint32_t b23 = pack_bf16_fn(*(uint32_t*)&p_row[c+2], *(uint32_t*)&p_row[c+3]);
            uint32_t b45 = pack_bf16_fn(*(uint32_t*)&p_row[c+4], *(uint32_t*)&p_row[c+5]);
            uint32_t b67 = pack_bf16_fn(*(uint32_t*)&p_row[c+6], *(uint32_t*)&p_row[c+7]);
            st_shared_128_fn((uint32_t)__cvta_generic_to_shared(&smem_P[tid * 136 + c]), b01, b23, b45, b67);
        }

        if (k_start > 0) {
            for (int c = 0; c < 64; c += 8) {
                uint32_t r[8];
                tmem_load_8x_fn(tmem_O0 + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
                tmem_load_fence_fn();
                #pragma unroll
                for(int i=0; i<8; ++i) r[i] = __float_as_uint(__uint_as_float(r[i]) * exp_diff);
                tmem_store_8x_fn(tmem_O0 + c, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
            }
            for (int c = 0; c < 64; c += 8) {
                uint32_t r[8];
                tmem_load_8x_fn(tmem_O1 + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
                tmem_load_fence_fn();
                #pragma unroll
                for(int i=0; i<8; ++i) r[i] = __float_as_uint(__uint_as_float(r[i]) * exp_diff);
                tmem_store_8x_fn(tmem_O1 + c, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        }

        fence_proxy_async_fn();

        uint32_t idesc_O = make_instr_desc_pv_fn(128, 64);
        for (int step = 0; step < 8; ++step) {
            uint64_t d_P = advance_desc_bytes(desc_P, step * 32);
            uint64_t d_V0 = advance_desc_bytes(desc_V0, step * 2048);
            uint32_t accum = (k_start == 0 && step == 0) ? 0 : 1;
            umma_f16_cg1_fn(tmem_O0, d_P, d_V0, idesc_O, accum);
        }
        for (int step = 0; step < 8; ++step) {
            uint64_t d_P = advance_desc_bytes(desc_P, step * 32);
            uint64_t d_V1 = advance_desc_bytes(desc_V1, step * 2048);
            uint32_t accum = (k_start == 0 && step == 0) ? 0 : 1;
            umma_f16_cg1_fn(tmem_O1, d_P, d_V1, idesc_O, accum);
        }

        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_umma, 1);
            tcgen05_commit_cg1_fn(&mbar_umma);
        }
        mbarrier_wait_fn(&mbar_umma, umma_phase);
        umma_phase ^= 1;
        k_phase[db] ^= 1;
    }

    float l_inv = 1.0f / l_vec;
    
    for (int c = 0; c < 64; c += 8) {
        uint32_t r[8];
        tmem_load_8x_fn(tmem_O0 + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
        tmem_load_fence_fn();
        #pragma unroll
        for(int i=0; i<8; ++i) r[i] = __float_as_uint(__uint_as_float(r[i]) * l_inv);
        uint32_t b01 = pack_bf16_fn(r[0], r[1]);
        uint32_t b23 = pack_bf16_fn(r[2], r[3]);
        uint32_t b45 = pack_bf16_fn(r[4], r[5]);
        uint32_t b67 = pack_bf16_fn(r[6], r[7]);
        st_shared_128_fn((uint32_t)__cvta_generic_to_shared(&smem.O[tid * 128 + c]), b01, b23, b45, b67);
    }
    for (int c = 0; c < 64; c += 8) {
        uint32_t r[8];
        tmem_load_8x_fn(tmem_O1 + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
        tmem_load_fence_fn();
        #pragma unroll
        for(int i=0; i<8; ++i) r[i] = __float_as_uint(__uint_as_float(r[i]) * l_inv);
        uint32_t b01 = pack_bf16_fn(r[0], r[1]);
        uint32_t b23 = pack_bf16_fn(r[2], r[3]);
        uint32_t b45 = pack_bf16_fn(r[4], r[5]);
        uint32_t b67 = pack_bf16_fn(r[6], r[7]);
        st_shared_128_fn((uint32_t)__cvta_generic_to_shared(&smem.O[tid * 128 + 64 + c]), b01, b23, b45, b67);
    }

    int row_in_global = q_start + tid;
    if (row_in_global < S) {
        LSE[(int64_t)batch_head * S + row_in_global] = m_vec + logf(l_vec);
    }

    tma_store_fence_fn();
    if (tid == 0) {
        tma_store_3d_fn(&tma_O, smem.O, 0, q_start, batch_head);
        tma_store_commit_fn();
    }
    tma_store_wait_fn<0>();

    if (tid == 0) {
        tmem_dealloc_fn(tmem_P, 128);
        tmem_dealloc_fn(tmem_O0, 64);
        tmem_dealloc_fn(tmem_O1, 64);
    }
}

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) 
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = 4;
    int64_t H = 48;
    int64_t S = Q.size(2);
    int64_t D = 128;
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    create_tma_3d_descriptor_2B(&tma_Q, Q.data_ptr(), D, S, B*H, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_3d_descriptor_2B(&tma_K, K.data_ptr(), D, S, B*H, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_3d_descriptor_2B(&tma_V, V.data_ptr(), D, S, B*H, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B);
    create_tma_3d_descriptor_2B(&tma_O, O.data_ptr(), D, S, B*H, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE);

    int threads = 128;
    int grid_x = (S + 127) / 128;
    int grid_y = B * H;
    dim3 blocks(grid_x, grid_y, 1);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
        
    mha_fwd_kernel_sm100<<<blocks, threads, 0, stream>>>(tma_Q, tma_K, tma_V, tma_O, static_cast<float*>(LSE.data_ptr()), S);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda