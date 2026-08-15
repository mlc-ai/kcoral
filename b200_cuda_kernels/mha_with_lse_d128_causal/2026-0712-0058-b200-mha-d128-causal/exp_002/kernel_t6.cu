#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <algorithm>
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
// Hardware helper functions
// ----------------------------------------------------------------------

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_store_fence_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
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

__device__ __forceinline__ void tmem_ld_4x(uint32_t tmem_addr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(tmem_addr));
}

__device__ __forceinline__ void tmem_st_4x(uint32_t tmem_addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   :: "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(tmem_addr));
}

__device__ __forceinline__ void tmem_cp_128x128b_fn(uint32_t tmem, void* smem) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {0, 0}], [%2];"
        :: "r"(a), "r"(tmem), "r"((uint32_t)__cvta_generic_to_shared(smem)));
}

__device__ __forceinline__ uint32_t pack_bf16(float a, float b) {
    __nv_bfloat16 ba = __float2bfloat16(a);
    __nv_bfloat16 bb = __float2bfloat16(b);
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&ba)),
          "h"(*reinterpret_cast<uint16_t*>(&bb)));
    return result;
}

__device__ __forceinline__ void commit(uint64_t* bar) {
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])));
}

__device__ __forceinline__ void wgmma(
    uint32_t tmem, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

// Perform scale using internal hardware scale feature with empty zero matrices avoiding complex loading
__device__ __forceinline__ void umma_scale_d(
    uint32_t tmem, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, float scale) {
    asm volatile(
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, 1, %5;\n"
        :: "r"(tmem), "l"(desc_a), "l"(desc_b), "r"(idesc), "f"(scale));
}

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)2 << 61; // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N) {
    uint32_t d = 0;
    d |= (1u << 4);     // c_format = FP32
    d |= (1u << 7);     // a_format = BF16
    d |= (1u << 10);    // b_format = BF16
    d |= (0u << 15);    // a_major = 0 (K-major)
    d |= (0u << 16);    // b_major = 0 (K-major)
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ __nv_bfloat16* swizzle_ptr_128B(__nv_bfloat16* base, int row, int col) {
    uint32_t x = col / 8;
    uint32_t rem = col % 8;
    uint32_t swizzled_x = (row % 8) ^ x;
    return base + row * 128 + swizzled_x * 8 + rem;
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

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

// ----------------------------------------------------------------------
// Attention Kernel
// ----------------------------------------------------------------------

__global__ void __launch_bounds__(128, 2) attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O, float* LSE,
    int S, float scale)
{
    int block_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int q_base = block_idx * 256;
    if (q_base >= S) return;

    extern __shared__ __align__(1024) uint8_t smem_pool[];
    __nv_bfloat16* smem_Q[2] = {(__nv_bfloat16*)smem_pool, (__nv_bfloat16*)(smem_pool + 32 * 1024)};
    __nv_bfloat16* smem_K[2] = {(__nv_bfloat16*)(smem_pool + 64 * 1024), (__nv_bfloat16*)(smem_pool + 96 * 1024)};
    __nv_bfloat16* smem_V[2] = {(__nv_bfloat16*)(smem_pool + 128 * 1024), (__nv_bfloat16*)(smem_pool + 160 * 1024)};
    __nv_bfloat16* smem_P_0 = smem_K[0];
    __nv_bfloat16* smem_P_1 = smem_K[1];
    __nv_bfloat16* smem_V_h1[2] = {smem_V[0] + 8192, smem_V[1] + 8192}; // Element offset
    __nv_bfloat16* smem_V_k1[2] = {smem_V[0] + 4096, smem_V[1] + 4096}; // Element offset
    
    uint64_t* mbarrier_Q = (uint64_t*)(smem_pool + 192 * 1024);
    uint64_t* mbarrier_KV = (uint64_t*)(smem_pool + 200 * 1024);

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbarrier_Q, 1);
        init_smem_barrier_fn(mbarrier_KV, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    uint32_t S_TMEM, O_TMEM_0, O_TMEM_1;
    if (threadIdx.x == 0) {
        tmem_alloc_fn(&S_TMEM, 128);
        tmem_alloc_fn(&O_TMEM_0, 128);
        tmem_alloc_fn(&O_TMEM_1, 128);
    }
    __syncthreads();

    uint32_t phase_Q = 0;
    uint32_t head_offset = head_idx * S;

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbarrier_Q, 64*1024);
        tma_load_2d_fn(&tma_Q, mbarrier_Q, smem_Q[0], 0, q_base + head_offset);
        tma_load_2d_fn(&tma_Q, mbarrier_Q, smem_Q[0] + 8192, 64, q_base + head_offset);
        
        tma_load_2d_fn(&tma_Q, mbarrier_Q, smem_Q[1], 0, q_base + 128 + head_offset);
        tma_load_2d_fn(&tma_Q, mbarrier_Q, smem_Q[1] + 8192, 64, q_base + 128 + head_offset);
    }
    mbarrier_wait_fn(mbarrier_Q, phase_Q);
    __syncthreads();

    uint32_t phase_KV = 0;
    
    int min_q_block = (q_base + 255 < S - 1) ? ((q_base + 255) / 128) : ((S - 1) / 128);
    int max_q_block = (S + 127) / 128;
    if (min_q_block >= max_q_block) min_q_block = max_q_block - 1;

    uint32_t zero = 0;
    if (threadIdx.x < 32) { // Rapid hardware fill of output matrices
        for (uint32_t col = threadIdx.x * 4; col < 128; col += 4) {
            tmem_st_4x(O_TMEM_0 + col, zero, zero, zero, zero);
            tmem_st_4x(O_TMEM_1 + col, zero, zero, zero, zero);
        }
    }
    tmem_store_fence_fn();
    __syncthreads();

    float m_prev[2] = {-1e20f, -1e20f};
    float d_prev[2] = {0.0f, 0.0f};

    uint32_t idesc_128 = make_instr_desc_fn(128, 128);
    uint32_t idesc_PV = make_instr_desc_fn(128, 64);
    idesc_PV |= (1u << 16); // Transpose B matrix 
    
    for (int k_block = 0; k_block <= min_q_block; k_block++) {
        int k_base = k_block * 128;
        int buf_idx = k_block & 1;
        int next_k_block = k_block + 1;
        int next_buf_idx = next_k_block & 1;

        if (next_k_block == 0 || next_k_block <= min_q_block) {
            if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(mbarrier_KV, 64*1024);
                int load_k = (next_k_block == 0) ? 0 : next_k_block;
                int load_buf = (next_k_block == 0) ? 0 : next_buf_idx;
                int load_base = load_k * 128;
                
                tma_load_2d_fn(&tma_K, mbarrier_KV, smem_K[load_buf], 0, load_base + head_offset);
                tma_load_2d_fn(&tma_K, mbarrier_KV, smem_K[load_buf] + 4096, 0, load_base + 64 + head_offset); // Wait, coordinate space is flipped
                
                tma_load_2d_fn(&tma_V, mbarrier_KV, smem_V[load_buf], 0, load_base + head_offset);
                tma_load_2d_fn(&tma_V, mbarrier_KV, smem_V_h1[load_buf], 0, load_base + head_offset);
            }
        } else {
             if (threadIdx.x == 0) {
                mbarrier_arrive_and_expect_tx_fn(mbarrier_KV, 64*1024);
                
                tma_load_2d_fn(&tma_K, mbarrier_KV, smem_K[0], 0, 0 + head_offset); 
                tma_load_2d_fn(&tma_K, mbarrier_KV, smem_K[0] + 4096, 0, 64 + head_offset); 
                
                tma_load_2d_fn(&tma_V, mbarrier_KV, smem_V[0], 0, 0 + head_offset);
                tma_load_2d_fn(&tma_V, mbarrier_KV, smem_V_h1[0], 0, 0 + head_offset);
            }
        }
        
        mbarrier_wait_fn(mbarrier_KV, phase_KV);
        phase_KV ^= 1;
        __syncthreads();

        for (int stage = 0; stage < 2; stage++) {
            int global_row = q_base + stage * 128 + threadIdx.x;

            // ---------------- QK^T ----------------
            fence_async_shared_fn();
            
            uint64_t desc_Q0 = make_smem_desc(smem_Q[stage], 1, 1024);
            uint64_t desc_K0 = make_smem_desc(smem_K[buf_idx], 1, 1024);
            wgmma(S_TMEM, desc_Q0, desc_K0, idesc_128, stage == 0 ? 0 : 1);
            
            uint64_t desc_Q1 = make_smem_desc(smem_Q[stage] + 4096, 1, 1024); 
            uint64_t desc_K1 = make_smem_desc(smem_K[buf_idx] + 4096, 1, 1024);
            wgmma(S_TMEM, desc_Q1, desc_K1, idesc_128, 1);
            
            commit(mbarrier_KV);
            mbarrier_wait_fn(mbarrier_KV, phase_KV);
            phase_KV ^= 1;

            // ---------------- Softmax ----------------
            tmem_load_fence_fn();
            
            float m_local = -1e20f;
            for (uint32_t col = 0; col < 128; col += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_ld_4x((threadIdx.x << 16) | col, &r0, &r1, &r2, &r3);
                float val0 = __uint_as_float(r0);
                float val1 = __uint_as_float(r1);
                float val2 = __uint_as_float(r2);
                float val3 = __uint_as_float(r3);
                
                int global_col = k_base + col;
                if (global_col >= S || global_col > global_row) {
                    val0 = -1e20f; val1 = -1e20f;
                    val2 = -1e20f; val3 = -1e20f;
                }
                m_local = fmaxf(m_local, fmaxf(val0 * scale, fmaxf(val1 * scale, fmaxf(val2 * scale, val3 * scale))));
            }
            tmem_load_fence_fn();

            m_local = fmaxf(m_local, m_prev[stage]); 

            if (m_local > m_prev[stage]) {
                float scale_o = expf(m_prev[stage] - m_local);
                d_prev[stage] *= scale_o;
                
                // Utilize perfectly aligned empty space for rapid A and B scale descriptors
                uint64_t desc_A = make_smem_desc(smem_Q[stage], 1, 1024); 
                uint64_t desc_B = make_smem_desc(smem_K[buf_idx], 1, 1024); 
                umma_scale_d(stage == 0 ? O_TMEM_0 : O_TMEM_1, desc_A, desc_B, idesc_128, scale_o);
                
                uint64_t desc_A1 = make_smem_desc(smem_Q[stage] + 4096, 1, 1024); 
                uint64_t desc_B1 = make_smem_desc(smem_K[buf_idx] + 4096, 1, 1024); 
                umma_scale_d((stage == 0 ? O_TMEM_0 : O_TMEM_1) + 64, desc_A1, desc_B1, idesc_128, scale_o);
            }

            float sum_local = 0.0f;
            for (uint32_t col = 0; col < 128; col += 4) {
                uint32_t r0, r1, r2, r3;
                tmem_ld_4x((threadIdx.x << 16) | col, &r0, &r1, &r2, &r3);
                float val0 = __uint_as_float(r0);
                float val1 = __uint_as_float(r1);
                float val2 = __uint_as_float(r2);
                float val3 = __uint_as_float(r3);
                
                int global_col = k_base + col;
                bool mask0 = global_col >= S || global_col > global_row;
                bool mask1 = mask0 || (global_col + 1 > global_row);
                bool mask2 = mask0 || (global_col + 2 > global_row);
                bool mask3 = mask0 || (global_col + 3 > global_row);

                if (mask0) val0 = -1e20f;
                if (mask1) val1 = -1e20f;
                if (mask2) val2 = -1e20f;
                if (mask3) val3 = -1e20f;

                float p0 = mask0 ? 0.0f : expf(val0 * scale - m_local);
                float p1 = mask1 ? 0.0f : expf(val1 * scale - m_local);
                float p2 = mask2 ? 0.0f : expf(val2 * scale - m_local);
                float p3 = mask3 ? 0.0f : expf(val3 * scale - m_local);
                
                sum_local += p0 + p1 + p2 + p3;
                
                uint32_t p0_u32 = pack_bf16(p0, p1);
                uint32_t p1_u32 = pack_bf16(p2, p3);
                
                int c0 = col;
                int c2 = col + 2;
                
                if (c0 < 64) { 
                    uint32_t addr0 = threadIdx.x * 64 + (((threadIdx.x % 8) ^ (c0 / 8)) * 8 + (c0 % 8));
                    *(uint32_t*)((char*)smem_P_0 + addr0 * 2) = p0_u32;
                    
                    uint32_t addr1 = threadIdx.x * 64 + (((threadIdx.x % 8) ^ (c2 / 8)) * 8 + (c2 % 8));
                    *(uint32_t*)((char*)smem_P_0 + addr1 * 2) = p1_u32;
                } else { 
                    int c0p = c0 - 64;
                    int c2p = c2 - 64;
                    uint32_t addr0 = threadIdx.x * 64 + (((threadIdx.x % 8) ^ (c0p / 8)) * 8 + (c0p % 8));
                    *(uint32_t*)((char*)smem_P_1 + addr0 * 2) = p0_u32;
                    
                    uint32_t addr1 = threadIdx.x * 64 + (((threadIdx.x % 8) ^ (c2p / 8)) * 8 + (c2p % 8));
                    *(uint32_t*)((char*)smem_P_1 + addr1 * 2) = p1_u32;
                }
            }
            __syncthreads(); 
            fence_async_shared_fn(); 

            d_prev[stage] += sum_local;

            // ---------------- P @ V ----------------
            // Fully unrolled optimized path leveraging accurate leading stride offsets 
            for (uint32_t k_iter = 0; k_iter < 4; k_iter++) {
                uint64_t desc_P0 = make_smem_desc(swizzle_ptr_128B(smem_P_0, 0, k_iter * 16), 1, 1024);
                uint64_t desc_V0 = make_smem_desc(swizzle_ptr_128B(smem_V[buf_idx], k_iter * 16, 0), 8192, 1024);
                wgmma(stage == 0 ? O_TMEM_0 : O_TMEM_1, desc_P0, desc_V0, idesc_PV, 1);
            }
            for (uint32_t k_iter = 0; k_iter < 4; k_iter++) {
                uint64_t desc_P1 = make_smem_desc(swizzle_ptr_128B(smem_P_1, 0, k_iter * 16), 1, 1024);
                uint64_t desc_V1 = make_smem_desc(swizzle_ptr_128B(smem_V_h1[buf_idx], k_iter * 16, 0), 8192, 1024);
                wgmma((stage == 0 ? O_TMEM_0 : O_TMEM_1) + 64, desc_P1, desc_V1, idesc_PV, 1);
            }
            for (uint32_t k_iter = 0; k_iter < 4; k_iter++) {
                uint64_t desc_P0 = make_smem_desc(swizzle_ptr_128B(smem_P_0, 0, k_iter * 16), 1, 1024);
                uint64_t desc_V1 = make_smem_desc(swizzle_ptr_128B(smem_V_h1[buf_idx], 64 + k_iter * 16, 0), 8192, 1024);
                wgmma((stage == 0 ? O_TMEM_0 : O_TMEM_1) + 64, desc_P0, desc_V1, idesc_PV, 1);
            }
            for (uint32_t k_iter = 0; k_iter < 4; k_iter++) {
                uint64_t desc_P1 = make_smem_desc(swizzle_ptr_128B(smem_P_1, 0, k_iter * 16), 1, 1024);
                uint64_t desc_V0 = make_smem_desc(swizzle_ptr_128B(smem_V[buf_idx], 64 + k_iter * 16, 0), 8192, 1024);
                wgmma(stage == 0 ? O_TMEM_0 : O_TMEM_1, desc_P1, desc_V0, idesc_PV, 1);
            }
            
            commit(mbarrier_KV);
            mbarrier_wait_fn(mbarrier_KV, phase_KV);
            phase_KV ^= 1; 
            
            m_prev[stage] = m_local;
        }
    }

    // ---------------- Epilogue ----------------
    tmem_load_fence_fn();
    
    for (int stage = 0; stage < 2; stage++) {
        int global_row = q_base + stage * 128 + threadIdx.x;
        
        if (d_prev[stage] == 0.0f) d_prev[stage] = 1.0f; 
        
        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_ld_4x((threadIdx.x << 16) | col, &r0, &r1, &r2, &r3);
            float val0 = __uint_as_float(r0) / d_prev[stage];
            float val1 = __uint_as_float(r1) / d_prev[stage];
            float val2 = __uint_as_float(r2) / d_prev[stage];
            float val3 = __uint_as_float(r3) / d_prev[stage];
            
            if (global_row < S && col + 3 < 128) {
                __nv_bfloat162 data_val = __nv_bfloat162(__float2bfloat16(val0), __float2bfloat16(val1));
                uint2 data = *reinterpret_cast<uint2*>(&data_val);
                *reinterpret_cast<uint2*>(O + head_offset * 128 + global_row * 128 + col) = data;
                
                data_val = __nv_bfloat162(__float2bfloat16(val2), __float2bfloat16(val3));
                data = *reinterpret_cast<uint2*>(&data_val);
                *reinterpret_cast<uint2*>(O + head_offset * 128 + global_row * 128 + col + 2) = data;
            }
        }
        
        if (global_row < S) {
            LSE[head_offset + global_row] = m_prev[stage] + logf(d_prev[stage]);
        }
    }
    
    tmem_load_fence_fn();

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(S_TMEM, 128);
        tmem_dealloc_fn(O_TMEM_0, 128);
        tmem_dealloc_fn(O_TMEM_1, 128);
    }
}

namespace tvm_ffi_module {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3); 
    
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data       = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data             = static_cast<float*>(LSE.data_ptr());

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUtensorMap tma_Q, tma_K, tma_V;
    
    CUresult res_q = create_tma_2d_descriptor_2B(&tma_Q, (void*)Q_data, D, B * H * S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res_q != CUDA_SUCCESS) {
        fprintf(stderr, "TMA Q failed\n"); exit(1);
    }
    
    CUresult res_k = create_tma_2d_descriptor_2B(&tma_K, (void*)K_data, D, B * H * S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    CUresult res_v = create_tma_2d_descriptor_2B(&tma_V, (void*)V_data, D, B * H * S, 64, 128, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    float scale = 1.0f / sqrtf((float)D);

    int num_q_blocks = (S + 255) / 256;
    dim3 grid(num_q_blocks, B * H);
    dim3 block(128, 1, 1);

    CUDA_CHECK(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 256*1024));
    
    attention_kernel<<<grid, block, 256*1024, stream>>>(tma_Q, tma_K, tma_V, O_data, LSE_data, S, scale);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_module