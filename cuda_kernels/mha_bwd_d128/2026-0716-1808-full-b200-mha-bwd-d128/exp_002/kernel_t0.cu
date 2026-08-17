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

// ----------------------------------------------------------------------
// PTX Wrappers
// ----------------------------------------------------------------------

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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
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

__device__ __forceinline__ void write_to_tmem_packed(const uint32_t* regs, uint32_t tmem_addr) {
    asm volatile("tcgen05.st.sync.aligned.16x32b.b32 [%0], {%1, %2, %3, %4};" 
        :: "r"(tmem_addr), "r"(regs[0]), "r"(regs[1]), "r"(regs[2]), "r"(regs[3]));
}

__device__ __forceinline__ void read_from_tmem_packed(uint32_t* regs, uint32_t tmem_addr) {
    asm volatile("tcgen05.ld.sync.aligned.16x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3]) : "r"(tmem_addr));
}

__device__ __forceinline__ void tmem_commit_cp_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ void tmem_wait_cp_fn(uint64_t* bar, uint32_t phase) {
    mbarrier_wait_fn(bar, phase);
}

__device__ __forceinline__ void tmem_commit_mma_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ void tmem_wait_mma_fn(uint64_t* bar, uint32_t phase) {
    mbarrier_wait_fn(bar, phase);
}

__device__ __forceinline__ void umma_f16_cta1_fn(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_f16_cta1_with_tmem_a(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
        :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ uint64_t make_smem_desc_k_major(void* smem_ptr, void* base_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t base = (uint32_t)__cvta_generic_to_shared(base_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    uint64_t lbo = 16; // Encodes to 1, effectively leaving it unused in swizzled mode
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    uint64_t sbo = 1024;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_mn_major(void* smem_ptr, void* base_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    uint64_t lbo = 8192; // (64 / 8) * 1024
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    uint64_t sbo = 1024;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// ----------------------------------------------------------------------
// Shared Storage & Kernels
// ----------------------------------------------------------------------

struct __align__(128) BwdSharedStorage {
    __nv_bfloat16 s_Q0[128 * 64];
    __nv_bfloat16 s_Q1[128 * 64];
    __nv_bfloat16 s_K0[128 * 64];
    __nv_bfloat16 s_K1[128 * 64];
    __nv_bfloat16 s_V0[128 * 64];
    __nv_bfloat16 s_V1[128 * 64];
    __nv_bfloat16 s_dO0[128 * 64];
    __nv_bfloat16 s_dO1[128 * 64];
    __nv_bfloat16 s_P_T[128 * 128];
    float s_L[128];
    uint64_t s_bar;
};

struct TensorInfo {
    const __nv_bfloat16* base_ptr;
    int stride0;
    int stride1;
    int offset_x;
    int offset_y;
};

__global__ void bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* L, float* dQ, float* dK, float* dV,
    uint32_t S, float scale)
{
    constexpr uint32_t BM = 128;
    constexpr uint32_t BN = 128;
    constexpr uint32_t BN0 = 64;
    constexpr uint32_t BN1 = 64;

    uint32_t m_block = blockIdx.x * BM;
    uint32_t n_block = blockIdx.y * BN;
    uint32_t bh = blockIdx.z;
    uint32_t offset_bh = bh * S;

    if (m_block >= S || n_block >= S) return;

    extern __shared__ __align__(128) char smem_buf[];
    BwdSharedStorage* shared = (BwdSharedStorage*)smem_buf;
    BwdSharedStorage& s = *shared;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(s.s_bar, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    uint32_t phase = 0;

    if (threadIdx.x == 0) {
        uint32_t tx_bytes = 8 * (BM * BN0 * sizeof(__nv_bfloat16));
        mbarrier_arrive_and_expect_tx_fn(s.s_bar, tx_bytes);

        tma_load_2d_fn(&tma_Q, s.s_bar, s.Q0, 0, offset_bh + m_block);
        tma_load_2d_fn(&tma_Q, s.s_bar, s.Q1, 64, offset_bh + m_block);
        tma_load_2d_fn(&tma_K, s.s_bar, s.K0, 0, offset_bh + n_block);
        tma_load_2d_fn(&tma_K, s.s_bar, s.K1, 64, offset_bh + n_block);
        tma_load_2d_fn(&tma_V, s.s_bar, s.V0, 0, offset_bh + n_block);
        tma_load_2d_fn(&tma_V, s.s_bar, s.V1, 64, offset_bh + n_block);
        tma_load_2d_fn(&tma_dO, s.s_bar, s.dO0, 0, offset_bh + m_block);
        tma_load_2d_fn(&tma_dO, s.s_bar, s.dO1, 64, offset_bh + m_block);
    }

    uint32_t tid = threadIdx.x;
    if (tid < BM) {
        s.L[tid] = (m_block + tid < S) ? L[offset_bh + m_block + tid] : 0.0f;
    }
    mbarrier_wait_fn(s.bar, phase);
    phase ^= 1;
    __syncthreads();
    fence_proxy_async_fn();

    uint32_t tmem_S, tmem_dP0, tmem_dP1, tmem_dQ0, tmem_dQ1;
    uint32_t tmem_dPt0, tmem_dPt1, tmem_dK0, tmem_dK1, tmem_PT, tmem_dV0, tmem_dV1;

    if (threadIdx.x == 0) {
        tmem_alloc_fn(&tmem_S, 128);
        tmem_alloc_fn(&tmem_dP0, 128);
        tmem_alloc_fn(&tmem_dP1, 128);
        tmem_alloc_fn(&tmem_dQ0, 64);
        tmem_alloc_fn(&tmem_dQ1, 64);
        tmem_alloc_fn(&tmem_dPt0, 128);
        tmem_alloc_fn(&tmem_dPt1, 128);
        tmem_alloc_fn(&tmem_dK0, 64);
        tmem_alloc_fn(&tmem_dK1, 64);
        tmem_alloc_fn(&tmem_PT, 128);
        tmem_alloc_fn(&tmem_dV0, 64);
        tmem_alloc_fn(&tmem_dV1, 64);
        
        mbarrier_arrive_and_expect_tx_fn(s.bar, 0);
    }
    __syncthreads();

    float S_local[128] = {0};

    // S = Q * K^T
    for (uint32_t step = 0; step < 64; step += 16) {
        __nv_bfloat16* cur_s_Q0 = s.Q0 + step * 1024;
        __nv_bfloat16* cur_s_K0 = s.K0 + step * 1024;
        uint64_t desc_A_Q0 = make_smem_desc_k_major(cur_s_Q0, cur_Q->base_ptr);
        uint64_t desc_B_K0 = make_smem_desc_mn_major(cur_s_K0, cur_K->base_ptr);
        uint32_t idesc = make_instr_desc_fn(128, 128);
        idesc |= (0u << 15); 
        idesc |= (1u << 16); 
        if (threadIdx.x == 0) {
            umma_f16_cta1_fn(tmem_S, desc_A_Q0, desc_B_K0, idesc, 1);
        }
    }
    
    tmem_commit_mma_fn(s.bar);
    tmem_wait_mma_fn(s.bar, phase);
    phase ^= 1;

    for (uint32_t i = 0; i < 4; i++) {
        uint32_t r_S[4];
        read_from_tmem_packed(r_S, tmem_S + (tid * 4 + i * 16));
        uint32_t r_idx[4];
        r_idx[0] = (tid * 4 + i * 16) / 2 % 128;
        r_idx[1] = (tid * 4 + i * 16 + 1) / 2 % 128;
        r_idx[2] = (tid * 4 + i * 16 + 2) / 2 % 128;
        r_idx[3] = (tid * 4 + i * 16 + 3) / 2 % 128;
        
        float f_S0 = __uint_as_float(r_S[0]);
        float f_S1 = __uint_as_float(r_S[1]);
        float f_S2 = __uint_as_float(r_S[2]);
        float f_S3 = __uint_as_float(r_S[3]);
        
        S_local[r_idx[0]] = f_S0;
        S_local[r_idx[1]] = f_S1;
        S_local[r_idx[2]] = f_S2;
        S_local[r_idx[3]] = f_S3;
    }

    __nv_bfloat16 P_T_local[128];
    for (uint32_t i = 0; i < 128; i++) {
        float s_val = S_local[i] * scale;
        float l_val = s.L[i];
        float p = fast_exp2f_fn((s_val - l_val) * 1.44269504f);
        P_T_local[i] = __float2bfloat16(p);
    }

    for (uint32_t j = 0; j < 128; j += 8) {
        uint32_t swizzled_j = ((tid % 8) ^ (j / 8)) * 8 + (j % 8);
        uint32_t addr = tid * 128 + swizzled_j;
        __nv_bfloat16* ptr = &s.P_T[addr];
        *(uint4*)ptr = *(uint4*)&P_T_local[j];
    }
    __syncthreads();

    // dV = P_T * dO
    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(s.bar, 0);
    }
    for (uint32_t step = 0; step < 128; step += 16) {
        uint64_t desc_A_PT = make_smem_desc_k_major(&s.P_T[step], s.P_T);
        uint64_t desc_B_dO0 = make_smem_desc_mn_major(s.dO0 + step * 1024, cur_dO->base_ptr);
        uint32_t idesc = make_instr_desc_fn(128, 64);
        idesc |= (0u << 15);
        idesc |= (1u << 16);
        if (threadIdx.x == 0) {
            umma_f16_cta1_fn(tmem_dV0, desc_A_PT, desc_B_dO0, idesc, 0);
            umma_f16_cta1_fn(tmem_dV1, desc_A_PT, desc_B_dO1, idesc, 0);
        }
    }
    tmem_commit_mma_fn(s.bar);
    tmem_wait_mma_fn(s.bar, phase);
    phase ^= 1;

    float dV0_out[8] = {0}, dV1_out[8] = {0};
    for (uint32_t i = 0; i < 4; i++) {
        uint32_t r_dV0[4], r_dV1[4];
        read_from_tmem_packed(r_dV0, tmem_dV0 + (tid * 4 + i * 16));
        read_from_tmem_packed(r_dV1, tmem_dV1 + (tid * 4 + i * 16));
        uint32_t r_idx0 = (tid * 4 + i * 16) / 2 % 64;
        uint32_t r_idx1 = (tid * 4 + i * 16 + 1) / 2 % 64;
        dV0_out[r_idx0] += __uint_as_float(r_dV0[0]);
        dV0_out[r_idx1] += __uint_as_float(r_dV0[1]);
        dV1_out[r_idx0] += __uint_as_float(r_dV1[0]);
        dV1_out[r_idx1] += __uint_as_float(r_dV1[1]);
    }

    if (n_block + tid * 2 < S) {
        uint32_t offset0 = (uint64_t)(n_block + tid * 2) * 128;
        atomicAdd(&dQ[offset0 + 0], dQ0_out[0]);
        atomicAdd(&dQ[offset0 + 1], dQ1_out[0]);
    }

    __syncthreads();
    
    // Softmax -> P_T
    for (uint32_t step = 0; step < 128; step += 8) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(s.bar, 0);
        }
        __syncthreads();
        // ... missing logic bridging S_local to P_T ... 
    }
    
    // ... Truncated internally ...
    
    tmem_dealloc_fn(tmem_S, 128);
    tmem_dealloc_fn(tmem_dP0, 128);
    tmem_dealloc_fn(tmem_dP1, 128);
    tmem_dealloc_fn(tmem_dQ0, 64);
    tmem_dealloc_fn(tmem_dQ1, 64);
    tmem_dealloc_fn(tmem_dPt0, 128);
    tmem_dealloc_fn(tmem_dPt1, 128);
    tmem_dealloc_fn(tmem_dK0, 64);
    tmem_dealloc_fn(tmem_dK1, 64);
    tmem_dealloc_fn(tmem_PT, 128);
    tmem_dealloc_fn(tmem_dV0, 64);
    tmem_dealloc_fn(tmem_dV1, 64);
}

namespace tvm_ffi_mha_bwd_d128 {

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

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
         
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3); 

    const __nv_bfloat16* q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* k_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* v_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* do_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* l_ptr = static_cast<const float*>(L.data_ptr());

    __nv_bfloat16* dq_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dk_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dv_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V,