#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>
#include <stdlib.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CU_CHECK(call) do { \
    CUresult _e = (call); \
    if (_e != CUDA_SUCCESS) { \
        fprintf(stderr, "CU error %d at %s:%d\n", (int)_e, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

namespace tvm_ffi_kernel {

// -------------------------------------------------------------------------
// Device Helper Functions
// -------------------------------------------------------------------------

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
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase & 1));
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)), "l"((uint64_t)d),
           "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_2d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d), "r"((uint32_t)__cvta_generic_to_shared(smem)),
           "r"(c0), "r"(c1) : "memory");
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
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
        :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t col, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(col));
}

__device__ __forceinline__ void tmem_store_4x_fn(uint32_t col, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};"
        :: "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_store_fence_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
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

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ uint64_t make_smem_desc_128B(void* smem_ptr, bool major) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    
    // For SM100 WGMMA under SWIZZLE_128B:
    // Major=True (K-major):  LBO=1 (unused structurally), SBO=1024
    // Major=False (MN-major): LBO=8192 (structurally dictated), SBO=1024
    uint32_t lbo = major ? 1 : 8192;
    uint32_t sbo = 1024;
    
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool b_transpose = false) {
    uint32_t d = 0;
    d |= (1u << 4);          // c_format = FP32
    d |= (1u << 7);          // a_format = BF16
    d |= (1u << 10);         // b_format = BF16
    d |= (0u << 15);         // a_major = 0 (A mapped intrinsically K-major)
    d |= (b_transpose << 16); // b_major = 1 (B mapped intrinsically N-major / transposed)
    d |= ((N / 8) << 17);    // n_dim
    d |= ((M / 16) << 24);   // m_dim
    return d;
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

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t swizzle_128B_byte(uint32_t row, uint32_t col_bytes) {
    uint32_t x = col_bytes / 16;
    uint32_t rem = col_bytes % 16;
    uint32_t swizzled_x = (row % 8) ^ x;
    return row * 128 + swizzled_x * 16 + rem;
}

// -------------------------------------------------------------------------
// Causal Attention Kernel
// -------------------------------------------------------------------------

__global__ void __launch_bounds__(64, 2) causal_attention_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* LSE, int S_len)
{
    extern __shared__ __align__(1024) uint8_t smem_pool[];
    
    uint8_t* smem_Q_0 = smem_pool;              // 8192 bytes
    uint8_t* smem_Q_1 = smem_pool + 8192;       // 8192 bytes
    uint8_t* smem_K_0 = smem_pool + 16384;      // 8192 bytes
    uint8_t* smem_K_1 = smem_pool + 24576;      // 8192 bytes
    uint8_t* smem_V_0 = smem_pool + 32768;      // 8192 bytes
    uint8_t* smem_V_1 = smem_pool + 40960;      // 8192 bytes
    uint8_t* smem_P   = smem_pool + 49152;      // 8192 bytes
    uint8_t* smem_D_0 = smem_pool + 57344;      // 8192 bytes
    uint8_t* smem_D_1 = smem_pool + 65536;      // 8192 bytes
    
    uint64_t* mbar_Q  = (uint64_t*)(smem_pool + 73728);
    uint64_t* mbar_KV = (uint64_t*)(smem_pool + 73736);

    int batch_head_offset = blockIdx.y * S_len;
    int seq_idx = blockIdx.x * 64;
    if (seq_idx >= S_len) return;

    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_KV, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t* tmem_S_p = (uint32_t*)(smem_pool + 73744);
        uint32_t* tmem_O_p = (uint32_t*)(smem_pool + 73752);
        
        tmem_alloc_fn(tmem_S_p, 64);
        tmem_alloc_fn(tmem_O_p, 64);
    }
    __syncthreads();
    
    uint32_t* tmem_S_p = (uint32_t*)(smem_pool + 73744);
    uint32_t* tmem_O_p = (uint32_t*)(smem_pool + 73752);
    
    uint32_t tmem_S = tmem_S_p[0];
    uint32_t tmem_O = tmem_O_p[0];

    if (threadIdx.x == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 16384);
        tma_load_2d_fn(&tma_Q, mbar_Q, smem_Q_0, 0, batch_head_offset + seq_idx);
        tma_load_2d_fn(&tma_Q, mbar_Q, smem_Q_1, 64, batch_head_offset + seq_idx);
    }
    mbarrier_wait_fn(mbar_Q, 0);

    float m_val = -INFINITY;
    float l_val = 0.0f;

    int phase = 0;

    for (int j_blk = 0; j_blk <= blockIdx.x; j_blk++) {
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_KV, 16384);
            tma_load_2d_fn(&tma_K, mbar_KV, smem_K_0, 0, batch_head_offset + j_blk * 64);
            tma_load_2d_fn(&tma_K, mbar_KV, smem_K_1, 64, batch_head_offset + j_blk * 64);
            tma_load_2d_fn(&tma_V, mbar_KV, smem_V_0, 0, batch_head_offset + j_blk * 64);
            tma_load_2d_fn(&tma_V, mbar_KV, smem_V_1, 64, batch_head_offset + j_blk * 64);
        }
        mbarrier_wait_fn(mbar_KV, phase);

        if (threadIdx.x == 0) {
            uint32_t idesc_Q = make_instr_desc_fn(64, 64);
            
            for (int k = 0; k < 4; k++) {
                uint64_t desc_Q = make_smem_desc_128B(smem_Q_0 + k * 32, true);
                uint64_t desc_K = make_smem_desc_128B(smem_K_0 + k * 32, true);
                umma_f16_cg1_fn(tmem_S, desc_Q, desc_K, idesc_Q, k == 0 ? 0 : 1);
            }
            for (int k = 0; k < 4; k++) {
                uint64_t desc_Q = make_smem_desc_128B(smem_Q_1 + k * 32, true);
                uint64_t desc_K = make_smem_desc_128B(smem_K_1 + k * 32, true);
                umma_f16_cg1_fn(tmem_S, desc_Q, desc_K, idesc_Q, 1);
            }
            umma_commit_1sm_fn(mbar_KV);
        }
        mbarrier_wait_fn(mbar_KV, phase ^ 1);

        float rowmax_val = -INFINITY;
        
        for (int col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S + col, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            float scale = 0.08838834764831845f;
            float val0 = f0 * scale;
            float val1 = f1 * scale;
            float val2 = f2 * scale;
            float val3 = f3 * scale;
            
            int q_pos = seq_idx + threadIdx.x;
            int k0 = j_blk * 64 + col;
            int k1 = j_blk * 64 + col + 1;
            int k2 = j_blk * 64 + col + 2;
            int k3 = j_blk * 64 + col + 3;

            if (k0 <= q_pos && k0 < S_len && val0 > rowmax_val) rowmax_val = val0;
            if (k1 <= q_pos && k1 < S_len && val1 > rowmax_val) rowmax_val = val1;
            if (k2 <= q_pos && k2 < S_len && val2 > rowmax_val) rowmax_val = val2;
            if (k3 <= q_pos && k3 < S_len && val3 > rowmax_val) rowmax_val = val3;
        }
        
        float old_m = m_val;
        float new_m = fmaxf(old_m, rowmax_val);
        
        float rowsum_val = 0.0f;
        float exp_rowmax = fast_exp2f_fn((rowmax_val - new_m) * 1.44269504f);
        
        for (int col = 0; col < 64; col += 4) {
            uint32_t r0, r1, r2, r3;
            tmem_load_4x_fn(tmem_S + col, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            float f0 = __uint_as_float(r0);
            float f1 = __uint_as_float(r1);
            float f2 = __uint_as_float(r2);
            float f3 = __uint_as_float(r3);
            
            float scale = 0.08838834764831845f;
            float val0 = f0 * scale;
            float val1 = f1 * scale;
            float val2 = f2 * scale;
            float val3 = f3 * scale;
            
            int q_pos = seq_idx + threadIdx.x;
            int k0 = j_blk * 64 + col;
            int k1 = j_blk * 64 + col + 1;
            int k2 = j_blk * 64 + col + 2;
            int k3 = j_blk * 64 + col + 3;
            
            bool valid0 = (k0 <= q_pos && k0 < S_len);
            bool valid1 = (k1 <= q_pos && k1 < S_len);
            bool valid2 = (k2 <= q_pos && k2 < S_len);
            bool valid3 = (k3 <= q_pos && k3 < S_len);
            
            float p0 = valid0 ? fast_exp2f_fn((val0 - new_m) * 1.44269504f) * exp_rowmax : 0.0f;
            float p1 = valid1 ? fast_exp2f_fn((val1 - new_m) * 1.44269504f) * exp_rowmax : 0.0f;
            float p2 = valid2 ? fast_exp2f_fn((val2 - new_m) * 1.44269504f) * exp_rowmax : 0.0f;
            float p3 = valid3 ? fast_exp2f_fn((val3 - new_m) * 1.44269504f) * exp_rowmax : 0.0f;
            
            rowsum_val += (p0 + p1 + p2 + p3);
            
            uint32_t pr0 = pack_bf16(p0, p1);
            uint32_t pr1 = pack_bf16(p2, p3);
            
            uint32_t byte_col = col * 2;
            uint32_t offset = swizzle_128B_byte(threadIdx.x, byte_col);
            *(uint32_t*)&smem_P[offset] = pr0;
            
            byte_col = (col + 2) * 2;
            offset = swizzle_128B_byte(threadIdx.x, byte_col);
            *(uint32_t*)&smem_P[offset] = pr1;
        }
        
        float thread_alpha = fast_exp2f_fn((old_m - new_m) * 1.44269504f);
        
        m_val = new_m;
        l_val = l_val * thread_alpha + rowsum_val * exp_rowmax;

        __syncthreads();

        if (threadIdx.x == 0) {
            uint32_t idesc_PV_0 = make_instr_desc_fn(64, 64, true);
            uint32_t idesc_PV_1 = make_instr_desc_fn(64, 64, true);
            
            for (int k = 0; k < 4; k++) {
                uint64_t desc_P0 = make_smem_desc_128B(smem_P + k * 32, true);
                uint64_t desc_V0 = make_smem_desc_128B(smem_V_0 + k * 2048, false); // V mapped intrinsically N-major
                
                umma_f16_cg1_fn(tmem_O, desc_P0, desc_V0, idesc_PV_0, k == 0 ? 0 : 1);
            }
            
            for (int k = 0; k < 4; k++) {
                uint64_t desc_P1 = make_smem_desc_128B(smem_P + 4096 + k * 32, true);
                uint64_t desc_V1 = make_smem_desc_128B(smem_V_1 + k * 2048, false); // V mapped intrinsically N-major
                
                umma_f16_cg1_fn(tmem_O + 4096, desc_P1, desc_V1, idesc_PV_1, k == 0 ? 0 : 1);
            }
            umma_commit_1sm_fn(mbar_KV);
        }
        mbarrier_wait_fn(mbar_KV, phase ^ 1);

        phase ^= 1;
    }

    for(int col = 0; col < 64; col+=4) {
        uint32_t r0, r1, r2, r3;
        tmem_load_4x_fn(tmem_O + col, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        
        float f0 = __uint_as_float(r0) / l_val;
        float f1 = __uint_as_float(r1) / l_val;
        float f2 = __uint_as_float(r2) / l_val;
        float f3 = __uint_as_float(r3) / l_val;
        
        uint32_t p0 = pack_bf16(f0, f1);
        uint32_t p1 = pack_bf16(f2, f3);
        
        uint32_t byte_col = col * 2;
        uint32_t offset = swizzle_128B_byte(threadIdx.x, byte_col);
        *(uint32_t*)&smem_D_0[offset] = p0;
        
        byte_col = (col + 2) * 2;
        offset = swizzle_128B_byte(threadIdx.x, byte_col);
        *(uint32_t*)&smem_D_1[offset] = p1;
    }
    
    __syncthreads();

    if (threadIdx.x == 0) {
        tma_store_fence_fn();
        tma_store_2d_fn(&tma_O, smem_D_0, 0, batch_head_offset + seq_idx);
        tma_store_2d_fn(&tma_O, smem_D_1, 64, batch_head_offset + seq_idx);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    __syncthreads();

    if (threadIdx.x < 64) {
        float lse = m_val + logf(l_val);
        LSE[batch_head_offset + seq_idx + threadIdx.x] = lse;
    }

    if (threadIdx.x == 0) {
        tmem_dealloc_fn(tmem_S, 64);
        tmem_dealloc_fn(tmem_O, 64);
    }
}

// -------------------------------------------------------------------------
// TMA Descriptor Creation
// -------------------------------------------------------------------------

CUresult create_tma_2d_descriptor_2B(
    CUtensorMap* d, void* globalAddress, 
    uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, 
    uint32_t smem_inner_dim, uint32_t smem_outer_dim, 
    CUtensorMapDataType dataType,
    CUtensorMapSwizzle swizzle, 
    CUtensorMapL2promotion l2Promotion, 
    CUtensorMapFloatOOBfill oobFill) 
{
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, dataType, 2, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        swizzle, l2Promotion, oobFill
    );
}

// -------------------------------------------------------------------------
// TVM-FFI Binding
// -------------------------------------------------------------------------

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, 
         tvm::ffi::TensorView V, tvm::ffi::TensorView O, 
         tvm::ffi::TensorView LSE) 
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3); 
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    void* q_ptr = Q.data_ptr();
    void* k_ptr = K.data_ptr();
    void* v_ptr = V.data_ptr();
    void* o_ptr = O.data_ptr();

    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, q_ptr, D, B*H*S, 64, 64, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, k_ptr, D, B*H*S, 64, 64, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, v_ptr, D, B*H*S, 64, 64, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_O, o_ptr, D, B*H*S, 64, 64, 
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, CU_TENSOR_MAP_SWIZZLE_128B, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    float* lse_ptr = static_cast<float*>(LSE.data_ptr());

    dim3 grid((S + 63) / 64, B * H);
    dim3 block(64);
    
    int smem_size = 80 * 1024;
    CUDA_CHECK(cudaFuncSetAttribute(causal_attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    causal_attention_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, lse_ptr, S);
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel