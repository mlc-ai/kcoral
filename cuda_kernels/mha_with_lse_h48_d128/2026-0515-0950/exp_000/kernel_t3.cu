#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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
                (int)_e, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

CUresult create_tma_4d_descriptor_2B(CUtensorMap* d, void* globalAddress, 
    uint64_t D, uint64_t S, uint64_t H, uint64_t B,
    uint32_t box_D, uint32_t box_S) {
    cuuint64_t globalDim[4] = {D, S, H, B};
    cuuint64_t globalStrides[3] = {D * 2, S * D * 2, H * S * D * 2};
    cuuint32_t boxDim[4] = {box_D, box_S, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        4, 
        globalAddress,
        globalDim,
        globalStrides,
        boxDim, 
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
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

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t swizzle) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    if (swizzle == 2) d |= (uint64_t)((addr >> 7) & 0x7) << 49; 
    d |= (uint64_t)swizzle << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
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

__device__ __forceinline__ void issue_umma_Q_K(
    __nv_bfloat16* Q0, __nv_bfloat16* Q1,
    __nv_bfloat16* K0, __nv_bfloat16* K1,
    uint32_t tmem_S_addr) {
    uint32_t idesc_S = make_instr_desc(128, 128, 0, 0); 
    #pragma unroll
    for (int k = 0; k < 4; ++k) { 
        uint64_t a_desc = make_smem_desc_sm100_fn(Q0 + k * 16, 1, 1024, 2);
        uint64_t b_desc = make_smem_desc_sm100_fn(K0 + k * 16, 1, 1024, 2);
        uint32_t accum = (k == 0) ? 0 : 1;
        asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                     :: "r"(tmem_S_addr), "l"(a_desc), "l"(b_desc), "r"(idesc_S), "r"(accum));
    }
    #pragma unroll
    for (int k = 0; k < 4; ++k) { 
        uint64_t a_desc = make_smem_desc_sm100_fn(Q1 + k * 16, 1, 1024, 2);
        uint64_t b_desc = make_smem_desc_sm100_fn(K1 + k * 16, 1, 1024, 2);
        uint32_t accum = 1;
        asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                     :: "r"(tmem_S_addr), "l"(a_desc), "l"(b_desc), "r"(idesc_S), "r"(accum));
    }
}

__device__ __forceinline__ void issue_umma_P_V(
    __nv_bfloat16* P0, __nv_bfloat16* P1,
    __nv_bfloat16* V0, __nv_bfloat16* V1,
    uint32_t tmem_O0_addr, uint32_t tmem_O1_addr) {
    uint32_t idesc_P_V0 = make_instr_desc(128, 64, 0, 1); 
    #pragma unroll
    for (int k = 0; k < 4; ++k) { 
        uint64_t a_desc = make_smem_desc_sm100_fn(P0 + k * 16, 1, 1024, 2);
        uint64_t b_desc = make_smem_desc_sm100_fn(V0 + k * 1024, 16384, 1024, 2);
        uint32_t accum = (k == 0) ? 0 : 1;
        asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                     :: "r"(tmem_O0_addr), "l"(a_desc), "l"(b_desc), "r"(idesc_P_V0), "r"(accum));
    }
    #pragma unroll
    for (int k = 0; k < 4; ++k) { 
        uint64_t a_desc = make_smem_desc_sm100_fn(P1 + k * 16, 1, 1024, 2);
        uint64_t b_desc = make_smem_desc_sm100_fn(V0 + (k + 4) * 1024, 16384, 1024, 2);
        asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                     :: "r"(tmem_O0_addr), "l"(a_desc), "l"(b_desc), "r"(idesc_P_V0), "r"(1));
    }
    #pragma unroll
    for (int k = 0; k < 4; ++k) { 
        uint64_t a_desc = make_smem_desc_sm100_fn(P0 + k * 16, 1, 1024, 2);
        uint64_t b_desc = make_smem_desc_sm100_fn(V1 + k * 1024, 16384, 1024, 2);
        uint32_t accum = (k == 0) ? 0 : 1;
        asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                     :: "r"(tmem_O1_addr), "l"(a_desc), "l"(b_desc), "r"(idesc_P_V0), "r"(accum));
    }
    #pragma unroll
    for (int k = 0; k < 4; ++k) { 
        uint64_t a_desc = make_smem_desc_sm100_fn(P1 + k * 16, 1, 1024, 2);
        uint64_t b_desc = make_smem_desc_sm100_fn(V1 + (k + 4) * 1024, 16384, 1024, 2);
        asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                     :: "r"(tmem_O1_addr), "l"(a_desc), "l"(b_desc), "r"(idesc_P_V0), "r"(1)); 
    }
}

__device__ __forceinline__ void accumulate_O_reg(
    uint32_t tmem_O0_addr, uint32_t tmem_O1_addr,
    float* O0_reg, float* O1_reg) {
    for (int col = 0; col < 64; col += 32) {
        uint32_t r[32];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) : "r"(tmem_O0_addr + col + 0));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]) : "r"(tmem_O0_addr + col + 8));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]) : "r"(tmem_O0_addr + col + 16));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31]) : "r"(tmem_O0_addr + col + 24));
        tmem_load_fence_fn();
        #pragma unroll
        for (int j = 0; j < 32; ++j) O0_reg[col + j] += __uint_as_float(r[j]);
    }
    for (int col = 0; col < 64; col += 32) {
        uint32_t r[32];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) : "r"(tmem_O1_addr + col + 0));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]) : "r"(tmem_O1_addr + col + 8));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]) : "r"(tmem_O1_addr + col + 16));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31]) : "r"(tmem_O1_addr + col + 24));
        tmem_load_fence_fn();
        #pragma unroll
        for (int j = 0; j < 32; ++j) O1_reg[col + j] += __uint_as_float(r[j]);
    }
}

__device__ __forceinline__ void compute_softmax_and_write_P(
    uint32_t tmem_S_addr,
    __nv_bfloat16* P0_smem, __nv_bfloat16* P1_smem,
    float& m_max, float& sum_exp,
    float* O0_reg, float* O1_reg,
    int kv_start, int S_len, int row) {
    
    for (int col = 0; col < 128; col += 32) {
        uint32_t r[32];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]) : "r"(tmem_S_addr + col + 0));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]) : "r"(tmem_S_addr + col + 8));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]) : "r"(tmem_S_addr + col + 16));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];" : "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31]) : "r"(tmem_S_addr + col + 24));
        tmem_load_fence_fn();

        float f[32];
        float local_max = -INFINITY;
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            float val = __uint_as_float(r[j]) * 0.0883883476f;
            if (kv_start + col + j >= S_len) val = -INFINITY;
            f[j] = val;
            local_max = max(local_max, val);
        }

        float m_new = max(m_max, local_max);
        float exp_diff = exp2f((m_max - m_new) * 1.44269504f);
        if (m_max == -INFINITY) exp_diff = 0.0f;
        if (m_new == -INFINITY) exp_diff = 1.0f;

        #pragma unroll
        for (int i = 0; i < 64; ++i) { 
            O0_reg[i] *= exp_diff; 
            O1_reg[i] *= exp_diff; 
        }
        sum_exp *= exp_diff;

        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            float p = (f[j] == -INFINITY) ? 0.0f : exp2f((f[j] - m_new) * 1.44269504f);
            sum_exp += p;
            
            int c = col + j;
            int chunk_idx = (c % 64) / 8;
            int swizzled_chunk = (row % 8) ^ chunk_idx;
            int swizzled_col = swizzled_chunk * 8 + (c % 8);
            if (c < 64) P0_smem[row * 64 + swizzled_col] = __float2bfloat16(p);
            else        P1_smem[row * 64 + swizzled_col] = __float2bfloat16(p);
        }
        m_max = m_new;
    }
}

__global__ void mha_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* LSE,
    int S_len, int H, int D) 
{
    if (threadIdx.x < 128) {
        setmaxnreg_inc_sync_fn<248>(); 
    }

    __shared__ __align__(1024) __nv_bfloat16 Q0_smem[128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 Q1_smem[128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 K0_smem[2][128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 K1_smem[2][128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 V0_smem[2][128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 V1_smem[2][128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 P0_smem[128 * 64];
    __shared__ __align__(1024) __nv_bfloat16 P1_smem[128 * 64];

    __shared__ __align__(8) uint64_t mbar_q;
    __shared__ __align__(8) uint64_t mbar_kv[2];
    __shared__ __align__(8) uint64_t mbar_umma;

    int q_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int b_idx = blockIdx.z;
    int q_start = q_idx * 128;
    
    if (q_start >= S_len) return;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&mbar_q, 1);
        init_smem_barrier_fn(&mbar_kv[0], 1);
        init_smem_barrier_fn(&mbar_kv[1], 1);
        init_smem_barrier_fn(&mbar_umma, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_q, 128 * 128 * 2);
        tma_load_4d_fn(&tma_Q, &mbar_q, Q0_smem, 0,  q_start, h_idx, b_idx);
        tma_load_4d_fn(&tma_Q, &mbar_q, Q1_smem, 64, q_start, h_idx, b_idx);
    }

    __shared__ uint32_t tmem_S_addr;
    __shared__ uint32_t tmem_O0_addr;
    __shared__ uint32_t tmem_O1_addr;
    if (threadIdx.x < 32) {
        tmem_alloc_fn(&tmem_S_addr, 128);
        tmem_alloc_fn(&tmem_O0_addr, 64);
        tmem_alloc_fn(&tmem_O1_addr, 64);
    }
    __syncthreads(); 

    float m_max = -INFINITY;
    float sum_exp = 0.0f;
    float O0_reg[64] = {0};
    float O1_reg[64] = {0};

    int num_kv_blocks = (S_len + 127) / 128;
    int phase_kv[2] = {0, 0};
    int phase_umma = 0;

    mbarrier_wait_fn(&mbar_q, 0);
    __syncthreads();

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(&mbar_kv[0], 128 * 128 * 2 * 2);
        tma_load_4d_fn(&tma_K, &mbar_kv[0], K0_smem[0], 0,  0, h_idx, b_idx);
        tma_load_4d_fn(&tma_K, &mbar_kv[0], K1_smem[0], 64, 0, h_idx, b_idx);
        tma_load_4d_fn(&tma_V, &mbar_kv[0], V0_smem[0], 0,  0, h_idx, b_idx);
        tma_load_4d_fn(&tma_V, &mbar_kv[0], V1_smem[0], 64, 0, h_idx, b_idx);

        if (num_kv_blocks > 1) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_kv[1], 128 * 128 * 2 * 2);
            tma_load_4d_fn(&tma_K, &mbar_kv[1], K0_smem[1], 0,  128, h_idx, b_idx);
            tma_load_4d_fn(&tma_K, &mbar_kv[1], K1_smem[1], 64, 128, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, &mbar_kv[1], V0_smem[1], 0,  128, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, &mbar_kv[1], V1_smem[1], 64, 128, h_idx, b_idx);
        }
    }

    mbarrier_wait_fn(&mbar_kv[0], phase_kv[0]);
    __syncthreads();
    phase_kv[0] ^= 1;

    if (threadIdx.x == 0) {
        issue_umma_Q_K(Q0_smem, Q1_smem, K0_smem[0], K1_smem[0], tmem_S_addr);
        uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(&mbar_umma);
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(mbar_addr));
    }
    mbarrier_wait_fn(&mbar_umma, phase_umma);
    __syncthreads();
    phase_umma ^= 1;

    for (int i = 0; i < num_kv_blocks; ++i) {
        int stage = i % 2;
        int next_i = i + 1;
        int next_stage = next_i % 2;
        int next_next_i = i + 2;

        compute_softmax_and_write_P(tmem_S_addr, P0_smem, P1_smem, m_max, sum_exp, O0_reg, O1_reg, i * 128, S_len, threadIdx.x);

        __syncthreads();
        fence_async_shared_fn();
        __syncthreads();

        if (next_i < num_kv_blocks) {
            mbarrier_wait_fn(&mbar_kv[next_stage], phase_kv[next_stage]);
            __syncthreads(); 
            phase_kv[next_stage] ^= 1;
        }

        if (threadIdx.x == 0) {
            issue_umma_P_V(P0_smem, P1_smem, V0_smem[stage], V1_smem[stage], tmem_O0_addr, tmem_O1_addr);
            
            if (next_i < num_kv_blocks) {
                issue_umma_Q_K(Q0_smem, Q1_smem, K0_smem[next_stage], K1_smem[next_stage], tmem_S_addr);
            }

            uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(&mbar_umma);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(mbar_addr));
        }

        mbarrier_wait_fn(&mbar_umma, phase_umma);
        __syncthreads();
        phase_umma ^= 1;

        if (threadIdx.x == 0 && next_next_i < num_kv_blocks) {
            mbarrier_arrive_and_expect_tx_fn(&mbar_kv[stage], 128 * 128 * 2 * 2);
            tma_load_4d_fn(&tma_K, &mbar_kv[stage], K0_smem[stage], 0,  next_next_i * 128, h_idx, b_idx);
            tma_load_4d_fn(&tma_K, &mbar_kv[stage], K1_smem[stage], 64, next_next_i * 128, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, &mbar_kv[stage], V0_smem[stage], 0,  next_next_i * 128, h_idx, b_idx);
            tma_load_4d_fn(&tma_V, &mbar_kv[stage], V1_smem[stage], 64, next_next_i * 128, h_idx, b_idx);
        }

        accumulate_O_reg(tmem_O0_addr, tmem_O1_addr, O0_reg, O1_reg);
    }

    __syncthreads();
    if (threadIdx.x < 32) {
        tmem_dealloc_fn(tmem_S_addr, 128);
        tmem_dealloc_fn(tmem_O0_addr, 64);
        tmem_dealloc_fn(tmem_O1_addr, 64);
    }
    __syncthreads();

    float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < 64; ++i) {
        O0_reg[i] *= inv_sum;
        O1_reg[i] *= inv_sum;
    }

    __syncthreads();
    int row = threadIdx.x;
    for (int col = 0; col < 64; col++) {
        int chunk_idx = col / 8;
        int swizzled_chunk = (row % 8) ^ chunk_idx;
        int swizzled_col = swizzled_chunk * 8 + (col % 8);
        P0_smem[row * 64 + swizzled_col] = __float2bfloat16(O0_reg[col]);
        P1_smem[row * 64 + swizzled_col] = __float2bfloat16(O1_reg[col]);
    }

    __syncthreads();
    fence_async_shared_fn();
    __syncthreads();

    if (threadIdx.x == 0) {
        tma_store_4d_fn(&tma_O, P0_smem, 0,  q_start, h_idx, b_idx);
        tma_store_4d_fn(&tma_O, P1_smem, 64, q_start, h_idx, b_idx);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }

    if (q_start + row < S_len) {
        size_t lse_idx = (size_t)b_idx * H * S_len + (size_t)h_idx * S_len + q_start + row;
        LSE[lse_idx] = m_max + logf(sum_exp);
    }
}

namespace tvm_ffi_example_cuda {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int64_t B = 4;
    int64_t H = 48;
    int64_t S = Q.size(2);
    int64_t D = 128;
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_Q, Q.data_ptr(), D, S, H, B, 64, 128));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_K, K.data_ptr(), D, S, H, B, 64, 128));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_V, V.data_ptr(), D, S, H, B, 64, 128));
    CU_CHECK(create_tma_4d_descriptor_2B(&tma_O, O.data_ptr(), D, S, H, B, 64, 128));

    int blocks_S = (S + 127) / 128;
    dim3 grid(blocks_S, H, B);
    dim3 block(128, 1, 1);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    mha_kernel<<<grid, block, 0, stream>>>(
        tma_Q, tma_K, tma_V, tma_O,
        static_cast<float*>(LSE.data_ptr()),
        S, H, D
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda