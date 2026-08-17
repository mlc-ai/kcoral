#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <math.h>

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

namespace tvm_ffi_mha {

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
                                     uint64_t dim0, uint64_t dim1, uint64_t dim2, 
                                     uint32_t box0, uint32_t box1, uint32_t box2) {
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

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float2 ex2_emulation_packed_asm_fn(float x, float y) {
    float ox, oy;
    asm("{\n\t"
        ".reg .f32 f1, f2, f3, f4, f5, f6, f7;\n\t"
        ".reg .b64 l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;\n\t"
        ".reg .s32 r1, r2, r3, r4, r5, r6, r7, r8;\n\t"
        "max.ftz.f32 f1, %2, 0fC2FE0000;\n\t"
        "max.ftz.f32 f2, %3, 0fC2FE0000;\n\t"
        "mov.b64 l1, {f1, f2};\n\t"
        "mov.f32 f3, 0f4B400000;\n\t"
        "mov.b64 l2, {f3, f3};\n\t"
        "add.rm.ftz.f32x2 l7, l1, l2;\n\t"
        "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"
        "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"
        "mov.f32 f7, 0f3D9DF09D;\n\t"
        "mov.b64 l6, {f7, f7};\n\t"
        "mov.f32 f6, 0f3E6906A4;\n\t"
        "mov.b64 l5, {f6, f6};\n\t"
        "mov.f32 f5, 0f3F31F519;\n\t"
        "mov.b64 l4, {f5, f5};\n\t"
        "mov.f32 f4, 0f3F800000;\n\t"
        "mov.b64 l3, {f4, f4};\n\t"
        "fma.rn.ftz.f32x2 l10, l9, l6, l5;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l4;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l3;\n\t"
        "mov.b64 {r1, r2}, l7;\n\t"
        "mov.b64 {r3, r4}, l10;\n\t"
        "shl.b32 r5, r1, 23;\n\t"
        "add.s32 r7, r5, r3;\n\t"
        "shl.b32 r6, r2, 23;\n\t"
        "add.s32 r8, r6, r4;\n\t"
        "mov.b32 %0, r7;\n\t"
        "mov.b32 %1, r8;\n\t"
        "}\n"
        : "=f"(ox), "=f"(oy)
        : "f"(x), "f"(y));
    return make_float2(ox, oy);
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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
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
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t sbo, uint32_t lbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr) >> 4;
    d |= (uint64_t)(addr & 0x3FFF);
    d |= (uint64_t)((lbo >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFF) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool trans_a=false, bool trans_b=false) {
    uint32_t d = 0;
    d |= (1u << 4);           
    d |= (1u << 7);           
    d |= (1u << 10);          
    if (trans_a) d |= (1u << 15);
    if (trans_b) d |= (1u << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__global__ void __launch_bounds__(128, 2) cta_gemm_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* lse_ptr,
    int S)
{
    setmaxnreg_inc_sync_fn<248>();

    int b = blockIdx.z;
    int h = blockIdx.y;
    int m_base = blockIdx.x * 128;
    int bh = b * gridDim.y + h;

    extern __shared__ uint8_t smem[];
    uint8_t* smem_Q0       = smem; 
    uint8_t* smem_Q1       = smem_Q0 + 16384; 
    uint8_t* smem_K0_0     = smem_Q1 + 16384; 
    uint8_t* smem_K0_1     = smem_K0_0 + 16384; 
    uint8_t* smem_K1_0     = smem_K0_1 + 16384; 
    uint8_t* smem_K1_1     = smem_K1_0 + 16384; 
    uint8_t* smem_V0_left  = smem_K1_1 + 16384; 
    uint8_t* smem_V0_right = smem_V0_left + 16384; 
    uint8_t* smem_V1_left  = smem_V0_right + 16384; 
    uint8_t* smem_V1_right = smem_V1_left + 16384; 
    uint8_t* smem_P0_0     = smem_V1_right + 16384; 
    uint8_t* smem_P0_1     = smem_P0_0 + 16384; 
    uint8_t* smem_P1_0     = smem_P0_1 + 16384; 
    uint8_t* smem_P1_1     = smem_P1_0 + 16384; 
    uint8_t* smem_O0       = smem_P0_0; 
    uint8_t* smem_O1       = smem_P0_1; 

    uint64_t* mbar_Q       = (uint64_t*)(smem_P1_1 + 16384);
    uint64_t* mbar_K       = mbar_Q + 1;
    uint64_t* mbar_V       = mbar_K + 2;
    uint64_t* mbar_mma_QK  = mbar_V + 2;
    uint64_t* mbar_mma_PV  = mbar_mma_QK + 2;
    uint32_t* smem_tmem_base = (uint32_t*)(mbar_mma_PV + 2);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(&mbar_K[0], 1);
        init_smem_barrier_fn(&mbar_K[1], 1);
        init_smem_barrier_fn(&mbar_V[0], 1);
        init_smem_barrier_fn(&mbar_V[1], 1);
        init_smem_barrier_fn(&mbar_mma_QK[0], 1);
        init_smem_barrier_fn(&mbar_mma_QK[1], 1);
        init_smem_barrier_fn(&mbar_mma_PV[0], 1);
        init_smem_barrier_fn(&mbar_mma_PV[1], 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads(); 

    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(smem_tmem_base, 512);
    }
    __syncthreads();
    
    uint32_t tmem_base = *smem_tmem_base;
    uint32_t tmem_QK0 = tmem_base;
    uint32_t tmem_QK1 = tmem_base + 128;
    uint32_t tmem_O_left = tmem_base + 256;
    uint32_t tmem_O_right = tmem_base + 320;

    uint64_t smem_desc_Q0       = make_smem_desc_sm100_fn(smem_Q0,       1024, 16); 
    uint64_t smem_desc_Q1       = make_smem_desc_sm100_fn(smem_Q1,       1024, 16); 
    uint64_t smem_desc_K0_0     = make_smem_desc_sm100_fn(smem_K0_0,     1024, 16); 
    uint64_t smem_desc_K0_1     = make_smem_desc_sm100_fn(smem_K0_1,     1024, 16); 
    uint64_t smem_desc_K1_0     = make_smem_desc_sm100_fn(smem_K1_0,     1024, 16); 
    uint64_t smem_desc_K1_1     = make_smem_desc_sm100_fn(smem_K1_1,     1024, 16); 
    uint64_t smem_desc_V0_left  = make_smem_desc_sm100_fn(smem_V0_left,  16, 1024); 
    uint64_t smem_desc_V0_right = make_smem_desc_sm100_fn(smem_V0_right, 16, 1024); 
    uint64_t smem_desc_V1_left  = make_smem_desc_sm100_fn(smem_V1_left,  16, 1024); 
    uint64_t smem_desc_V1_right = make_smem_desc_sm100_fn(smem_V1_right, 16, 1024); 
    uint64_t smem_desc_P0_0     = make_smem_desc_sm100_fn(smem_P0_0,     1024, 16); 
    uint64_t smem_desc_P0_1     = make_smem_desc_sm100_fn(smem_P0_1,     1024, 16); 
    uint64_t smem_desc_P1_0     = make_smem_desc_sm100_fn(smem_P1_0,     1024, 16); 
    uint64_t smem_desc_P1_1     = make_smem_desc_sm100_fn(smem_P1_1,     1024, 16); 

    uint32_t idesc_QK = make_instr_desc_fn(128, 128, false, false);
    uint32_t idesc_O  = make_instr_desc_fn(128, 64, false, true);

    uint32_t phase_Q = 0;
    uint32_t phase_K[2] = {0, 0};
    uint32_t phase_V[2] = {0, 0};
    uint32_t phase_mma_QK[2] = {0, 0};
    uint32_t phase_mma_PV[2] = {0, 0};

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 32768);
        tma_load_3d_fn(&tma_Q, mbar_Q, smem_Q0, 0, m_base, bh);
        tma_load_3d_fn(&tma_Q, mbar_Q, smem_Q1, 64, m_base, bh);
        
        mbarrier_arrive_and_expect_tx_fn(&mbar_K[0], 32768);
        tma_load_3d_fn(&tma_K, &mbar_K[0], smem_K0_0, 0, 0, bh);
        tma_load_3d_fn(&tma_K, &mbar_K[0], smem_K0_1, 64, 0, bh);
        
        mbarrier_arrive_and_expect_tx_fn(&mbar_V[0], 32768);
        tma_load_3d_fn(&tma_V, &mbar_V[0], smem_V0_left, 0, 0, bh);
        tma_load_3d_fn(&tma_V, &mbar_V[0], smem_V0_right, 64, 0, bh);
    }

    mbarrier_wait_fn(mbar_Q, phase_Q);
    phase_Q ^= 1;
    
    mbarrier_wait_fn(&mbar_K[0], phase_K[0]);
    phase_K[0] ^= 1;
    fence_async_shared_fn();

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int step = 0; step < 8; step++) {
            uint64_t desc_Q = (step < 4) ? smem_desc_Q0 : smem_desc_Q1;
            uint32_t offset_Q = (step % 4) * 32;
            desc_Q += (offset_Q >> 4);

            uint64_t desc_K = (step < 4) ? smem_desc_K0_0 : smem_desc_K0_1;
            uint32_t offset_K = (step % 4) * 32;
            desc_K += (offset_K >> 4);

            uint32_t accum = (step == 0) ? 0 : 1;
            umma_f16_cg1_fn(tmem_QK0, desc_Q, desc_K, idesc_QK, accum);
        }
        tcgen05_fence_after_fn();
        umma_commit_cg1_fn(&mbar_mma_QK[0]);
    }

    float O_reg[128] = {0};
    float m_prev = -INFINITY;
    float d_prev = 0;

    int num_n_blocks = (S + 127) / 128;

    for (int n_idx = 0; n_idx < num_n_blocks; n_idx++) {
        int next_n = n_idx + 1;
        bool has_next = next_n < num_n_blocks;
        int cur_buf = n_idx % 2;
        int nxt_buf = next_n % 2;
        
        if (has_next) {
            int n_base_next = next_n * 128;
            if (threadIdx.x == 0) {
                uint8_t* smem_K_nxt_0 = (nxt_buf == 0) ? smem_K0_0 : smem_K1_0;
                uint8_t* smem_K_nxt_1 = (nxt_buf == 0) ? smem_K0_1 : smem_K1_1;
                mbarrier_arrive_and_expect_tx_fn(&mbar_K[nxt_buf], 32768);
                tma_load_3d_fn(&tma_K, &mbar_K[nxt_buf], smem_K_nxt_0, 0, n_base_next, bh);
                tma_load_3d_fn(&tma_K, &mbar_K[nxt_buf], smem_K_nxt_1, 64, n_base_next, bh);

                uint8_t* smem_V_nxt_l = (nxt_buf == 0) ? smem_V0_left : smem_V1_left;
                uint8_t* smem_V_nxt_r = (nxt_buf == 0) ? smem_V0_right : smem_V1_right;
                mbarrier_arrive_and_expect_tx_fn(&mbar_V[nxt_buf], 32768);
                tma_load_3d_fn(&tma_V, &mbar_V[nxt_buf], smem_V_nxt_l, 0, n_base_next, bh);
                tma_load_3d_fn(&tma_V, &mbar_V[nxt_buf], smem_V_nxt_r, 64, n_base_next, bh);
            }
        }
        
        mbarrier_wait_fn(&mbar_mma_QK[cur_buf], phase_mma_QK[cur_buf]);
        phase_mma_QK[cur_buf] ^= 1;

        uint32_t tmem_QK_cur = (cur_buf == 0) ? tmem_QK0 : tmem_QK1;

        float m_curr = -INFINITY;
        int y = threadIdx.x;
        
        #pragma unroll
        for (int c = 0; c < 128; c += 32) {
            float S_chunk[32];
            #pragma unroll
            for(int i = 0; i < 32; i += 8) {
                tmem_load_8x_fn(tmem_QK_cur + c + i, 
                    (uint32_t*)&S_chunk[i+0], (uint32_t*)&S_chunk[i+1], (uint32_t*)&S_chunk[i+2], (uint32_t*)&S_chunk[i+3], 
                    (uint32_t*)&S_chunk[i+4], (uint32_t*)&S_chunk[i+5], (uint32_t*)&S_chunk[i+6], (uint32_t*)&S_chunk[i+7]);
            }
            tmem_load_fence_fn();
            #pragma unroll
            for(int i = 0; i < 32; i++) {
                if (m_base + y < S && n_idx * 128 + c + i < S) {
                    m_curr = fmaxf(m_curr, S_chunk[i] * 0.088388347648f);
                }
            }
        }
        
        float m_new = fmaxf(m_prev, m_curr);
        float d_curr = 0;

        #pragma unroll
        for (int c = 0; c < 128; c += 32) {
            float S_chunk[32];
            #pragma unroll
            for(int i = 0; i < 32; i += 8) {
                tmem_load_8x_fn(tmem_QK_cur + c + i, 
                    (uint32_t*)&S_chunk[i+0], (uint32_t*)&S_chunk[i+1], (uint32_t*)&S_chunk[i+2], (uint32_t*)&S_chunk[i+3], 
                    (uint32_t*)&S_chunk[i+4], (uint32_t*)&S_chunk[i+5], (uint32_t*)&S_chunk[i+6], (uint32_t*)&S_chunk[i+7]);
            }
            tmem_load_fence_fn();

            #pragma unroll
            for(int i = 0; i < 32; i += 2) {
                bool valid0 = (m_base + y < S) && (n_idx * 128 + c + i < S) && (m_new != -INFINITY);
                bool valid1 = (m_base + y < S) && (n_idx * 128 + c + i + 1 < S) && (m_new != -INFINITY);
                
                float val0 = valid0 ? (S_chunk[i] * 0.088388347648f - m_new) * 1.4426950408889634f : 0.0f;
                float val1 = valid1 ? (S_chunk[i+1] * 0.088388347648f - m_new) * 1.4426950408889634f : 0.0f;

                if (valid0 || valid1) {
                    if ((i % 8) == 0) {
                        float2 exp_val = ex2_emulation_packed_asm_fn(val0, val1);
                        S_chunk[i]   = valid0 ? exp_val.x : 0.0f;
                        S_chunk[i+1] = valid1 ? exp_val.y : 0.0f;
                    } else {
                        S_chunk[i]   = valid0 ? fast_exp2f_fn(val0) : 0.0f;
                        S_chunk[i+1] = valid1 ? fast_exp2f_fn(val1) : 0.0f;
                    }
                } else {
                    S_chunk[i] = 0.0f;
                    S_chunk[i+1] = 0.0f;
                }
                d_curr += S_chunk[i] + S_chunk[i+1];
            }
            
            #pragma unroll
            for(int i = 0; i < 32; i += 8) {
                uint32_t b01 = pack_bf16_fn(__float_as_uint(S_chunk[i+0]), __float_as_uint(S_chunk[i+1]));
                uint32_t b23 = pack_bf16_fn(__float_as_uint(S_chunk[i+2]), __float_as_uint(S_chunk[i+3]));
                uint32_t b45 = pack_bf16_fn(__float_as_uint(S_chunk[i+4]), __float_as_uint(S_chunk[i+5]));
                uint32_t b67 = pack_bf16_fn(__float_as_uint(S_chunk[i+6]), __float_as_uint(S_chunk[i+7]));
                
                int chunk = (c + i) / 8;
                int swizzled_chunk = (y % 8) ^ (chunk % 8);
                int offset = y * 128 + swizzled_chunk * 16;
                uint8_t* target_P0 = (cur_buf == 0) ? smem_P0_0 : smem_P1_0;
                uint8_t* target_P1 = (cur_buf == 0) ? smem_P0_1 : smem_P1_1;
                uint8_t* target = (chunk < 8) ? target_P0 : target_P1;
                st_shared_128_fn((uint32_t)__cvta_generic_to_shared(target + offset), b01, b23, b45, b67);
            }
        }

        float rescale = (m_prev == -INFINITY) ? 0.0f : fast_exp2f_fn((m_prev - m_new) * 1.4426950408889634f);
        d_prev = d_prev * rescale + d_curr;
        m_prev = m_new;

        fence_async_shared_fn();
        __syncthreads(); 

        mbarrier_wait_fn(&mbar_V[cur_buf], phase_V[cur_buf]);
        phase_V[cur_buf] ^= 1;
        fence_async_shared_fn();

        if (threadIdx.x == 0) {
            uint64_t desc_P_0 = (cur_buf == 0) ? smem_desc_P0_0 : smem_desc_P1_0;
            uint64_t desc_P_1 = (cur_buf == 0) ? smem_desc_P0_1 : smem_desc_P1_1;
            uint64_t desc_V_l = (cur_buf == 0) ? smem_desc_V0_left : smem_desc_V1_left;
            uint64_t desc_V_r = (cur_buf == 0) ? smem_desc_V0_right : smem_desc_V1_right;

            #pragma unroll
            for (int step = 0; step < 8; step++) {
                uint64_t dp = (step < 4) ? desc_P_0 : desc_P_1;
                uint32_t offset_P = (step % 4) * 32;
                dp += (offset_P >> 4);

                uint64_t dv = desc_V_l + ((step * 2048) >> 4);
                uint32_t accum = (step == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_O_left, dp, dv, idesc_O, accum);
            }

            #pragma unroll
            for (int step = 0; step < 8; step++) {
                uint64_t dp = (step < 4) ? desc_P_0 : desc_P_1;
                uint32_t offset_P = (step % 4) * 32;
                dp += (offset_P >> 4);

                uint64_t dv = desc_V_r + ((step * 2048) >> 4);
                uint32_t accum = (step == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_O_right, dp, dv, idesc_O, accum);
            }

            tcgen05_fence_after_fn();
            umma_commit_cg1_fn(&mbar_mma_PV[cur_buf]);
        }

        if (has_next) {
            mbarrier_wait_fn(&mbar_K[nxt_buf], phase_K[nxt_buf]);
            phase_K[nxt_buf] ^= 1;
            fence_async_shared_fn();

            if (threadIdx.x == 0) {
                uint64_t desc_K0 = (nxt_buf == 0) ? smem_desc_K0_0 : smem_desc_K1_0;
                uint64_t desc_K1 = (nxt_buf == 0) ? smem_desc_K0_1 : smem_desc_K1_1;
                uint32_t tmem_QK_nxt = (nxt_buf == 0) ? tmem_QK0 : tmem_QK1;

                #pragma unroll
                for (int step = 0; step < 8; step++) {
                    uint64_t dq = (step < 4) ? smem_desc_Q0 : smem_desc_Q1;
                    uint32_t offset_Q = (step % 4) * 32;
                    dq += (offset_Q >> 4);

                    uint64_t dk = (step < 4) ? desc_K0 : desc_K1;
                    uint32_t offset_K = (step % 4) * 32;
                    dk += (offset_K >> 4);

                    uint32_t accum = (step == 0) ? 0 : 1;
                    umma_f16_cg1_fn(tmem_QK_nxt, dq, dk, idesc_QK, accum);
                }
                tcgen05_fence_after_fn();
                umma_commit_cg1_fn(&mbar_mma_QK[nxt_buf]);
            }
        }

        mbarrier_wait_fn(&mbar_mma_PV[cur_buf], phase_mma_PV[cur_buf]);
        phase_mma_PV[cur_buf] ^= 1;

        #pragma unroll
        for (int c = 0; c < 64; c += 32) {
            float PV_chunk[32];
            #pragma unroll
            for(int i = 0; i < 32; i += 8) {
                tmem_load_8x_fn(tmem_O_left + c + i, 
                    (uint32_t*)&PV_chunk[i+0], (uint32_t*)&PV_chunk[i+1], (uint32_t*)&PV_chunk[i+2], (uint32_t*)&PV_chunk[i+3], 
                    (uint32_t*)&PV_chunk[i+4], (uint32_t*)&PV_chunk[i+5], (uint32_t*)&PV_chunk[i+6], (uint32_t*)&PV_chunk[i+7]);
            }
            tmem_load_fence_fn();
            #pragma unroll
            for(int i = 0; i < 32; i++) {
                O_reg[c + i] = O_reg[c + i] * rescale + PV_chunk[i];
            }
        }
        
        #pragma unroll
        for (int c = 0; c < 64; c += 32) {
            float PV_chunk[32];
            #pragma unroll
            for(int i = 0; i < 32; i += 8) {
                tmem_load_8x_fn(tmem_O_right + c + i, 
                    (uint32_t*)&PV_chunk[i+0], (uint32_t*)&PV_chunk[i+1], (uint32_t*)&PV_chunk[i+2], (uint32_t*)&PV_chunk[i+3], 
                    (uint32_t*)&PV_chunk[i+4], (uint32_t*)&PV_chunk[i+5], (uint32_t*)&PV_chunk[i+6], (uint32_t*)&PV_chunk[i+7]);
            }
            tmem_load_fence_fn();
            #pragma unroll
            for(int i = 0; i < 32; i++) {
                O_reg[64 + c + i] = O_reg[64 + c + i] * rescale + PV_chunk[i];
            }
        }

        __syncthreads();
    }

    float inv_d = (d_prev > 0.0f) ? (1.0f / d_prev) : 0.0f;
    int y = threadIdx.x;
    
    #pragma unroll
    for (int c = 0; c < 64; c += 8) {
        uint32_t b01 = pack_bf16_fn(__float_as_uint(O_reg[c+0] * inv_d), __float_as_uint(O_reg[c+1] * inv_d));
        uint32_t b23 = pack_bf16_fn(__float_as_uint(O_reg[c+2] * inv_d), __float_as_uint(O_reg[c+3] * inv_d));
        uint32_t b45 = pack_bf16_fn(__float_as_uint(O_reg[c+4] * inv_d), __float_as_uint(O_reg[c+5] * inv_d));
        uint32_t b67 = pack_bf16_fn(__float_as_uint(O_reg[c+6] * inv_d), __float_as_uint(O_reg[c+7] * inv_d));
        
        int chunk = c / 8;
        int swizzled_chunk = (y % 8) ^ chunk;
        int offset = y * 128 + swizzled_chunk * 16;
        st_shared_128_fn((uint32_t)__cvta_generic_to_shared(smem_O0 + offset), b01, b23, b45, b67);
    }
    
    #pragma unroll
    for (int c = 64; c < 128; c += 8) {
        uint32_t b01 = pack_bf16_fn(__float_as_uint(O_reg[c+0] * inv_d), __float_as_uint(O_reg[c+1] * inv_d));
        uint32_t b23 = pack_bf16_fn(__float_as_uint(O_reg[c+2] * inv_d), __float_as_uint(O_reg[c+3] * inv_d));
        uint32_t b45 = pack_bf16_fn(__float_as_uint(O_reg[c+4] * inv_d), __float_as_uint(O_reg[c+5] * inv_d));
        uint32_t b67 = pack_bf16_fn(__float_as_uint(O_reg[c+6] * inv_d), __float_as_uint(O_reg[c+7] * inv_d));
        
        int chunk = (c - 64) / 8;
        int swizzled_chunk = (y % 8) ^ chunk;
        int offset = y * 128 + swizzled_chunk * 16;
        st_shared_128_fn((uint32_t)__cvta_generic_to_shared(smem_O1 + offset), b01, b23, b45, b67);
    }
    
    fence_async_shared_fn();
    tma_store_fence_fn();

    __syncthreads();
    
    if (threadIdx.x == 0) {
        tma_store_3d_fn(&tma_O, smem_O0, 0, m_base, bh);
        tma_store_3d_fn(&tma_O, smem_O1, 64, m_base, bh);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    __syncthreads();

    if (m_base + y < S) {
        int lse_idx = bh * S + m_base + y;
        lse_ptr[lse_idx] = m_prev + logf(d_prev);
    }
    
    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_base, 512);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3); 

    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_Q, Q.data_ptr(), D, S, B * H, 64, 128, 1));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_K, K.data_ptr(), D, S, B * H, 64, 128, 1));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_V, V.data_ptr(), D, S, B * H, 64, 128, 1));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_O, O.data_ptr(), D, S, B * H, 64, 128, 1));

    int m_blocks = (S + 127) / 128;
    dim3 grid(m_blocks, H, B);
    dim3 block(128); 

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(cta_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 229376 + 1024));

    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 229376 + 1024; 
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    CUDA_CHECK(cudaLaunchKernelEx(&config, cta_gemm_kernel, tma_Q, tma_K, tma_V, tma_O, static_cast<float*>(LSE.data_ptr()), static_cast<int>(S)));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

}  // namespace tvm_ffi_mha