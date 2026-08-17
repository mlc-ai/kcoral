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
    uint32_t phase_parity = phase & 1;
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase_parity));
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tma_store_fence_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void tma_load_4d_cg1_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    uint32_t ba = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(sa), "l"((uint64_t)d), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(ba) : "memory");
}

__device__ __forceinline__ void tma_store_4d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    uint32_t sa = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group [%0, {%2, %3, %4, %5}], [%1];"
        :: "l"((uint64_t)d), "r"(sa), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
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
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t swizzle_mode = 0) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)swizzle_mode << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, int a_major, int b_major) {
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

CUresult create_tma_4d_descriptor_2B(
    CUtensorMap* d, void* globalAddress, 
    uint64_t D, uint64_t S, uint64_t H, uint64_t B,
    uint32_t smem_D, uint32_t smem_S,
    CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[4] = {D, S, H, B};
    cuuint64_t globalStrides[3] = {D * 2, D * S * 2, D * S * H * 2};
    cuuint32_t boxDim[4] = {smem_D, smem_S, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4,
        globalAddress, globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

__device__ __forceinline__ uint32_t get_tma_tx_bytes(int c1, int S, int D) {
    int valid_rows = S - c1;
    if (valid_rows > 128) valid_rows = 128;
    if (valid_rows < 0) valid_rows = 0;
    return valid_rows * D * 2;
}

extern __shared__ __align__(1024) char smem[];

__global__ void __launch_bounds__(128) AttentionKernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* lse_ptr,
    int S_seq, int H_heads
) {
    int m_block_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int b_idx = blockIdx.z;
    int n_blocks = (S_seq + 127) / 128;

    char* smem_Q = smem; // 32768
    char* smem_K0 = smem + 32768; // 32768
    char* smem_K1 = smem + 65536; // 32768
    char* smem_V0 = smem + 98304; // 32768
    char* smem_V1 = smem + 131072; // 32768
    char* smem_PO = smem + 163840; // 32768

    uint64_t* bar_Q = (uint64_t*)(smem + 196608);
    uint64_t* bar_K0 = (uint64_t*)(smem + 196616);
    uint64_t* bar_K1 = (uint64_t*)(smem + 196624);
    uint64_t* bar_V0 = (uint64_t*)(smem + 196632);
    uint64_t* bar_V1 = (uint64_t*)(smem + 196640);
    uint64_t* bar_S = (uint64_t*)(smem + 196648);
    uint64_t* bar_PV = (uint64_t*)(smem + 196656);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(bar_Q, 1);
        init_smem_barrier_fn(bar_K0, 1);
        init_smem_barrier_fn(bar_K1, 1);
        init_smem_barrier_fn(bar_V0, 1);
        init_smem_barrier_fn(bar_V1, 1);
        init_smem_barrier_fn(bar_S, 1);
        init_smem_barrier_fn(bar_PV, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t tx_bytes = get_tma_tx_bytes(m_block_idx * 128, S_seq, 128);
        mbarrier_arrive_and_expect_tx_fn(bar_Q, tx_bytes);
        tma_load_4d_cg1_fn(&tma_Q, bar_Q, smem_Q, 0, m_block_idx * 128, h_idx, b_idx);
    }

    if (threadIdx.x == 0 && 0 < n_blocks) {
        uint32_t tx_bytes = get_tma_tx_bytes(0, S_seq, 128);
        mbarrier_arrive_and_expect_tx_fn(bar_K0, tx_bytes);
        tma_load_4d_cg1_fn(&tma_K, bar_K0, smem_K0, 0, 0, h_idx, b_idx);
        
        mbarrier_arrive_and_expect_tx_fn(bar_V0, tx_bytes);
        tma_load_4d_cg1_fn(&tma_V, bar_V0, smem_V0, 0, 0, h_idx, b_idx);
    }

    mbarrier_wait_fn(bar_Q, 0);

    __shared__ uint32_t tmem_S_smem;
    __shared__ uint32_t tmem_PV_smem;
    
    if (threadIdx.x < 32) {
        tmem_alloc_cg1_fn(&tmem_S_smem, 128);
        tmem_alloc_cg1_fn(&tmem_PV_smem, 128);
    }
    __syncthreads();
    uint32_t tmem_S = tmem_S_smem;
    uint32_t tmem_PV = tmem_PV_smem;

    float O_regs[128];
    for (int i = 0; i < 128; i++) O_regs[i] = 0.0f;
    float m_val = -INFINITY;
    float l_val = 0.0f;

    for (int n = 0; n < n_blocks; ++n) {
        uint64_t* cur_bar_K = (n % 2 == 0) ? bar_K0 : bar_K1;
        uint64_t* cur_bar_V = (n % 2 == 0) ? bar_V0 : bar_V1;
        
        uint64_t* nxt_bar_K = (n % 2 == 0) ? bar_K1 : bar_K0;
        uint64_t* nxt_bar_V = (n % 2 == 0) ? bar_V1 : bar_V0;
        char* cur_smem_K = (n % 2 == 0) ? smem_K0 : smem_K1;
        char* cur_smem_V = (n % 2 == 0) ? smem_V0 : smem_V1;
        char* nxt_smem_K = (n % 2 == 0) ? smem_K1 : smem_K0;
        char* nxt_smem_V = (n % 2 == 0) ? smem_V1 : smem_V0;

        mbarrier_wait_fn(cur_bar_K, n / 2);
        mbarrier_wait_fn(cur_bar_V, n / 2);

        if (threadIdx.x == 0) {
            uint32_t idesc_QK = make_instr_desc_fn(128, 128, 1, 0); // Q: M-major, K: K-major
            for (int k = 0; k < 128; k += 16) {
                uint64_t desc_Q = make_smem_desc_sm100_fn(smem_Q + k * 2, 128, 2048, 0);
                uint64_t desc_K = make_smem_desc_sm100_fn(cur_smem_K + k * 2, 2048, 128, 0);
                umma_f16_cg1_fn(tmem_S, desc_Q, desc_K, idesc_QK, (k > 0 ? 1 : 0));
            }
            umma_commit_cg1_fn(bar_S);
        }

        if (threadIdx.x == 0 && n + 1 < n_blocks) {
            uint32_t tx_bytes = get_tma_tx_bytes((n + 1) * 128, S_seq, 128);
            mbarrier_arrive_and_expect_tx_fn(nxt_bar_K, tx_bytes);
            tma_load_4d_cg1_fn(&tma_K, nxt_bar_K, nxt_smem_K, 0, (n + 1) * 128, h_idx, b_idx);
            
            mbarrier_arrive_and_expect_tx_fn(nxt_bar_V, tx_bytes);
            tma_load_4d_cg1_fn(&tma_V, nxt_bar_V, nxt_smem_V, 0, (n + 1) * 128, h_idx, b_idx);
        }

        mbarrier_wait_fn(bar_S, n % 2);

        int valid_cols = S_seq - n * 128;
        float row_max = -INFINITY;
        
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            uint32_t taddr = tmem_S + col;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float s0 = __uint_as_float(r0) * 0.0883883476f;
            float s1 = __uint_as_float(r1) * 0.0883883476f;
            float s2 = __uint_as_float(r2) * 0.0883883476f;
            float s3 = __uint_as_float(r3) * 0.0883883476f;

            if (n == n_blocks - 1) {
                if (col + 0 >= valid_cols) s0 = -INFINITY;
                if (col + 1 >= valid_cols) s1 = -INFINITY;
                if (col + 2 >= valid_cols) s2 = -INFINITY;
                if (col + 3 >= valid_cols) s3 = -INFINITY;
            }

            row_max = fmaxf(row_max, s0);
            row_max = fmaxf(row_max, s1);
            row_max = fmaxf(row_max, s2);
            row_max = fmaxf(row_max, s3);
        }

        float m_new = fmaxf(m_val, row_max);
        float exp_diff = (m_val == -INFINITY) ? 0.0f : fast_exp2f_fn((m_val - m_new) * 1.44269504089f);
        float row_sum = 0.0f;

        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            uint32_t taddr = tmem_S + col;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float s0 = __uint_as_float(r0) * 0.0883883476f;
            float s1 = __uint_as_float(r1) * 0.0883883476f;
            float s2 = __uint_as_float(r2) * 0.0883883476f;
            float s3 = __uint_as_float(r3) * 0.0883883476f;

            if (n == n_blocks - 1) {
                if (col + 0 >= valid_cols) s0 = -INFINITY;
                if (col + 1 >= valid_cols) s1 = -INFINITY;
                if (col + 2 >= valid_cols) s2 = -INFINITY;
                if (col + 3 >= valid_cols) s3 = -INFINITY;
            }

            float p0 = fast_exp2f_fn((s0 - m_new) * 1.44269504089f);
            float p1 = fast_exp2f_fn((s1 - m_new) * 1.44269504089f);
            float p2 = fast_exp2f_fn((s2 - m_new) * 1.44269504089f);
            float p3 = fast_exp2f_fn((s3 - m_new) * 1.44269504089f);

            row_sum += p0 + p1 + p2 + p3;

            uint32_t p01 = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
            uint32_t p23 = pack_bf16_fn(__float_as_uint(p2), __float_as_uint(p3));
            
            uint32_t row = threadIdx.x;
            uint32_t offset = (row * 128 + col) * 2;
            uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_PO) + offset;
            asm volatile("st.shared.v2.b32 [%0], {%1, %2};" :: "r"(smem_addr), "r"(p01), "r"(p23) : "memory");
        }

        float l_new = l_val * exp_diff + row_sum;
        for (int i = 0; i < 128; i++) {
            O_regs[i] *= exp_diff;
        }
        m_val = m_new;
        l_val = l_new;

        fence_proxy_async_fn();
        __syncthreads();

        if (n == n_blocks - 1) {
            if (threadIdx.x >= valid_cols && threadIdx.x < 128) {
                uint32_t* v_row = (uint32_t*)(cur_smem_V + threadIdx.x * 128 * 2);
                #pragma unroll
                for(int c = 0; c < 128/2; c += 4) {
                    v_row[c+0] = 0;
                    v_row[c+1] = 0;
                    v_row[c+2] = 0;
                    v_row[c+3] = 0;
                }
            }
            fence_proxy_async_fn();
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            uint32_t idesc_PV = make_instr_desc_fn(128, 128, 1, 1); // P: M-major, V: N-major
            for (int k = 0; k < 128; k += 16) {
                uint64_t desc_P = make_smem_desc_sm100_fn(smem_PO + k * 2, 128, 2048, 0);
                uint64_t desc_V = make_smem_desc_sm100_fn(cur_smem_V + k * 256, 128, 2048, 0);
                umma_f16_cg1_fn(tmem_PV, desc_P, desc_V, idesc_PV, (k > 0 ? 1 : 0));
            }
            umma_commit_cg1_fn(bar_PV);
        }

        mbarrier_wait_fn(bar_PV, n % 2);

        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            uint32_t taddr = tmem_PV + col;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            O_regs[col+0] += __uint_as_float(r0);
            O_regs[col+1] += __uint_as_float(r1);
            O_regs[col+2] += __uint_as_float(r2);
            O_regs[col+3] += __uint_as_float(r3);
        }
    }

    if (n_blocks > 0) {
        for (int col = 0; col < 128; col++) {
            O_regs[col] /= l_val;
        }

        float lse = m_val + logf(l_val);
        if (m_block_idx * 128 + threadIdx.x < S_seq) {
            int idx = b_idx * (H_heads * S_seq) + h_idx * S_seq + (m_block_idx * 128 + threadIdx.x);
            lse_ptr[idx] = lse;
        }

        for (int col = 0; col < 128; col += 4) {
            uint32_t p01 = pack_bf16_fn(__float_as_uint(O_regs[col+0]), __float_as_uint(O_regs[col+1]));
            uint32_t p23 = pack_bf16_fn(__float_as_uint(O_regs[col+2]), __float_as_uint(O_regs[col+3]));
            
            uint32_t row = threadIdx.x;
            uint32_t offset = (row * 128 + col) * 2;
            uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_PO) + offset;
            asm volatile("st.shared.v2.b32 [%0], {%1, %2};" :: "r"(smem_addr), "r"(p01), "r"(p23) : "memory");
        }

        tma_store_fence_fn();
        __syncthreads();
        
        if (threadIdx.x == 0) {
            tma_store_4d_fn(&tma_O, smem_PO, 0, m_block_idx * 128, h_idx, b_idx);
            tma_store_commit_fn();
            tma_store_wait_fn<0>();
        }
    }

    __syncthreads();
    
    if (threadIdx.x < 32) {
        tmem_dealloc_cg1_fn(tmem_S, 128);
        tmem_dealloc_cg1_fn(tmem_PV, 128);
    }
    
    __syncthreads();
}

namespace tvm_ffi_example_cuda {
void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    void* q_ptr = Q.data_ptr();
    void* k_ptr = K.data_ptr();
    void* v_ptr = V.data_ptr();
    void* o_ptr = O.data_ptr();
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    CUresult res;
    res = create_tma_4d_descriptor_2B(&tma_Q, q_ptr, D, S, H, B, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA Q failed\n"); exit(1); }
    res = create_tma_4d_descriptor_2B(&tma_K, k_ptr, D, S, H, B, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA K failed\n"); exit(1); }
    res = create_tma_4d_descriptor_2B(&tma_V, v_ptr, D, S, H, B, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA V failed\n"); exit(1); }
    res = create_tma_4d_descriptor_2B(&tma_O, o_ptr, D, S, H, B, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "TMA O failed\n"); exit(1); }
    
    int grids_s = (S + 127) / 128;
    dim3 grid(grids_s, H, B);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    CUDA_CHECK(cudaFuncSetAttribute((void*)AttentionKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 200000));
    
    AttentionKernel<<<grid, block, 200000, stream>>>(tma_Q, tma_K, tma_V, tma_O, lse_ptr, S, H);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);
}  // namespace tvm_ffi_example_cuda