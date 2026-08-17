#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <mma.h>
#include <math_constants.h>
#include <stdio.h>
#include <cuda.h>
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
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha {

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

__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
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
        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
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
                 :: "r"(a), "r"(ncols) : "memory");
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols) : "memory");
}

__device__ __forceinline__ void tcgen05_wait_st_fn() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(bar);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a) : "memory");
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

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo, uint32_t swizzle) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)swizzle << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, int b_major) {
    uint32_t d = 0;
    d |= (1u << 4);    // FP32
    d |= (1u << 7);    // BF16
    d |= (1u << 10);   // BF16
    d |= (0u << 15);   // K-Major
    d |= (b_major << 16); 
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

#define TMEM_LOAD_32(col, r) asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 " \
    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15," \
    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];" \
    : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]),"=r"(r[4]),"=r"(r[5]),"=r"(r[6]),"=r"(r[7]), \
      "=r"(r[8]),"=r"(r[9]),"=r"(r[10]),"=r"(r[11]),"=r"(r[12]),"=r"(r[13]),"=r"(r[14]),"=r"(r[15]), \
      "=r"(r[16]),"=r"(r[17]),"=r"(r[18]),"=r"(r[19]),"=r"(r[20]),"=r"(r[21]),"=r"(r[22]),"=r"(r[23]), \
      "=r"(r[24]),"=r"(r[25]),"=r"(r[26]),"=r"(r[27]),"=r"(r[28]),"=r"(r[29]),"=r"(r[30]),"=r"(r[31]) \
    : "r"(col))

#define TMEM_STORE_32(col, r) asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 " \
    "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15," \
    "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];" \
    :: "r"(r[0]),"r"(r[1]),"r"(r[2]),"r"(r[3]),"r"(r[4]),"r"(r[5]),"r"(r[6]),"r"(r[7]), \
       "r"(r[8]),"r"(r[9]),"r"(r[10]),"r"(r[11]),"r"(r[12]),"r"(r[13]),"r"(r[14]),"r"(r[15]), \
       "r"(r[16]),"r"(r[17]),"r"(r[18]),"r"(r[19]),"r"(r[20]),"r"(r[21]),"r"(r[22]),"r"(r[23]), \
       "r"(r[24]),"r"(r[25]),"r"(r[26]),"r"(r[27]),"r"(r[28]),"r"(r[29]),"r"(r[30]),"r"(r[31]), \
       "r"(col))

__global__ void __launch_bounds__(128) flash_attn_fwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    float* __restrict__ LSE,
    int B, int H, int S, int D)
{
    setmaxnreg_inc_sync_fn<240>();
    
    int bh = blockIdx.x;
    int q_start = blockIdx.y * 128;
    if (q_start >= S) return;
    
    int tid = threadIdx.x;
    
    extern __shared__ char smem_raw[];
    char* smem_buf = (char*)(((uintptr_t)smem_raw + 1023) & ~1023);
    
    auto Q_smem = (__nv_bfloat16*)(smem_buf);                   
    auto K_smem = (__nv_bfloat16*)(smem_buf + 32768);         
    auto V_smem = (__nv_bfloat16*)(smem_buf + 65536);         
    auto P_smem = (__nv_bfloat16*)(smem_buf + 98304);         
    auto O_smem = P_smem;

    uint64_t* mbar = (uint64_t*)(smem_buf + 131072);
    uint64_t* umma_mbar = (uint64_t*)(smem_buf + 131072 + 8);
    uint32_t* tmem_smem = (uint32_t*)(smem_buf + 131072 + 16);
    
    if (tid == 0) {
        init_smem_barrier_fn(mbar, 128);
        init_smem_barrier_fn(umma_mbar, 1);
    }
    if (tid < 32) {
        tmem_alloc_cg1_fn(&tmem_smem[0], 128); // tmem_O
        tmem_alloc_cg1_fn(&tmem_smem[1], 128); // tmem_P
    }
    __syncthreads();
    
    uint32_t tmem_O = tmem_smem[0];
    uint32_t tmem_P = tmem_smem[1];
    
    // Non-swizzled descriptors for 128x128 tiles. Row stride = 256 bytes.
    uint64_t desc_Q = make_smem_desc_sm100_fn(Q_smem, 32768, 2048, 0); // K-Major
    uint64_t desc_K = make_smem_desc_sm100_fn(K_smem, 32768, 2048, 0); // K-Major
    uint64_t desc_V = make_smem_desc_sm100_fn(V_smem, 2048, 32768, 0); // MN-Major
    uint64_t desc_P = make_smem_desc_sm100_fn(P_smem, 32768, 2048, 0); // K-Major

    uint32_t idesc_QK = make_instr_desc_fn(128, 128, 0); // K-Major (B)
    uint32_t idesc_PV = make_instr_desc_fn(128, 128, 1); // MN-Major (B)
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar, 32768);
        tma_load_3d_fn(&tma_Q, mbar, Q_smem, 0, q_start, bh);
    } else {
        mbarrier_arrive_fn(mbar);
    }
    mbarrier_wait_fn(mbar, 0);

    float m_val = -CUDART_INF_F;
    float l_val = 0.0f;
    float scale = 1.0f / sqrtf((float)D);
    int phase = 1;
    int umma_phase = 0;
    
    int num_kv_chunks = (S + 127) / 128;
    for (int tc = 0; tc < num_kv_chunks; tc++) {
        int kv_start = tc * 128;
        int kv_len = min(128, S - kv_start);
        if (kv_len <= 0) break;
        
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 65536);
            tma_load_3d_fn(&tma_K, mbar, K_smem, 0, kv_start, bh);
            tma_load_3d_fn(&tma_V, mbar, V_smem, 0, kv_start, bh);
        } else {
            mbarrier_arrive_fn(mbar);
        }
        mbarrier_wait_fn(mbar, phase & 1);
        
        // GEMM1: P = Q @ K^T
        for (int k = 0; k < 128; k += 16) {
            uint64_t dQ = desc_Q + 2 * (k / 16); 
            uint64_t dK = desc_K + 2 * (k / 16);
            int accum = (k == 0) ? 0 : 1;
            umma_f16_cg1_fn(tmem_P, dQ, dK, idesc_QK, accum);
        }
        
        if (tid == 0) umma_commit_cg1_fn(umma_mbar);
        mbarrier_wait_fn(umma_mbar, umma_phase & 1);
        umma_phase++;
        
        // Softmax
        float row_max = -CUDART_INF_F;
        uint32_t p_regs[4][32]; 
        
        TMEM_LOAD_32(tmem_P, p_regs[0]);
        TMEM_LOAD_32(tmem_P + 32, p_regs[1]);
        TMEM_LOAD_32(tmem_P + 64, p_regs[2]);
        TMEM_LOAD_32(tmem_P + 96, p_regs[3]);
        tmem_load_fence_fn();
        
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 32; j++) {
                int col_idx = i * 32 + j;
                float val = __uint_as_float(p_regs[i][j]);
                val *= scale;
                p_regs[i][j] = __float_as_uint(val);
                if (col_idx < kv_len) row_max = max(row_max, val);
            }
        }
        
        float m_old = m_val;
        float m_new = max(m_old, row_max);
        float row_sum = 0;
        
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 32; j++) {
                int col_idx = i * 32 + j;
                float val = __uint_as_float(p_regs[i][j]);
                float p = 0.0f;
                if (col_idx < kv_len) p = expf(val - m_new);
                row_sum += p;
                p_regs[i][j] = __float_as_uint(p);
            }
        }
        
        float l_new = l_val * expf(m_old - m_new) + row_sum;
        m_val = m_new;
        l_val = l_new;
        float o_scale = (m_old == -CUDART_INF_F) ? 0.0f : expf(m_old - m_new);
        
        for (int i = 0; i < 4; i++) { 
            for (int j = 0; j < 32; j += 2) {
                float p0 = __uint_as_float(p_regs[i][j]);
                float p1 = __uint_as_float(p_regs[i][j+1]);
                ((uint32_t*)P_smem)[tid * 64 + i * 16 + j / 2] = 
                    *(uint32_t*)&__floats2bfloat162_rn(p0, p1);
            }
        }
        
        if (o_scale != 1.0f && tc > 0) {
            uint32_t o_regs[4][32];
            TMEM_LOAD_32(tmem_O, o_regs[0]);
            TMEM_LOAD_32(tmem_O + 32, o_regs[1]);
            TMEM_LOAD_32(tmem_O + 64, o_regs[2]);
            TMEM_LOAD_32(tmem_O + 96, o_regs[3]);
            tmem_load_fence_fn();
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 32; j++) {
                    float val = __uint_as_float(o_regs[i][j]);
                    val *= o_scale;
                    o_regs[i][j] = __float_as_uint(val);
                }
                TMEM_STORE_32(tmem_O + i * 32, o_regs[i]);
            }
            tcgen05_wait_st_fn();
        }
        
        // GEMM2: O = P @ V
        for (int k = 0; k < 128; k += 16) {
            uint64_t dP = desc_P + 2 * (k / 16); 
            uint64_t dV = desc_V + 256 * (k / 16); 
            int accum = (tc == 0 && k == 0) ? 0 : 1;
            umma_f16_cg1_fn(tmem_O, dP, dV, idesc_PV, accum);
        }
        
        if (tid == 0) umma_commit_cg1_fn(umma_mbar);
        mbarrier_wait_fn(umma_mbar, umma_phase & 1);
        umma_phase++;
        
        phase++;
        __syncthreads();
    }
    
    // Store O
    uint32_t o_regs[4][32];
    TMEM_LOAD_32(tmem_O, o_regs[0]);
    TMEM_LOAD_32(tmem_O + 32, o_regs[1]);
    TMEM_LOAD_32(tmem_O + 64, o_regs[2]);
    TMEM_LOAD_32(tmem_O + 96, o_regs[3]);
    tmem_load_fence_fn();
    
    for (int i = 0; i < 4; i++) { 
        for (int j = 0; j < 32; j += 2) {
            float val0 = __uint_as_float(o_regs[i][j]) / l_val;
            float val1 = __uint_as_float(o_regs[i][j+1]) / l_val;
            ((uint32_t*)O_smem)[tid * 64 + i * 16 + j / 2] = 
                *(uint32_t*)&__floats2bfloat162_rn(val0, val1);
        }
    }
    
    tma_store_fence_fn();
    __syncthreads();
    
    if (tid == 0) {
        tma_store_3d_fn(&tma_O, O_smem, 0, q_start, bh);
        tma_store_commit_fn();
        tma_store_wait_fn<0>();
    }
    __syncthreads();
    
    if (tid < 128 && (q_start + tid) < S) {
        LSE[bh * S + q_start + tid] = m_val + logf(l_val);
    }
    
    if (tid < 32) {
        tmem_dealloc_cg1_fn(tmem_O, 128);
        tmem_dealloc_cg1_fn(tmem_P, 128);
    }
}

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t dim0, uint64_t dim1, uint64_t dim2, uint32_t box0, uint32_t box1, uint32_t box2, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[3] = {dim0, dim1, dim2};
    cuuint64_t globalStrides[2] = {dim0 * 2, dim0 * dim1 * 2};
    cuuint32_t boxDim[3] = {box0, box1, box2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int D = Q.size(3);

    auto q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    auto k_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    auto v_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    auto o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    auto lse_ptr = static_cast<float*>(LSE.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_Q, q_ptr, D, S, B * H, 128, 128, 1, CU_TENSOR_MAP_SWIZZLE_NONE));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_K, k_ptr, D, S, B * H, 128, 128, 1, CU_TENSOR_MAP_SWIZZLE_NONE));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_V, v_ptr, D, S, B * H, 128, 128, 1, CU_TENSOR_MAP_SWIZZLE_NONE));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_O, o_ptr, D, S, B * H, 128, 128, 1, CU_TENSOR_MAP_SWIZZLE_NONE));

    dim3 grid(B * H, (S + 127) / 128, 1);
    dim3 block(128, 1, 1);
    int smem_size = 131072 + 1024;

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(flash_attn_fwd_kernel, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    flash_attn_fwd_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, lse_ptr, B, H, S, D);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha::run);

} // namespace tvm_ffi_mha