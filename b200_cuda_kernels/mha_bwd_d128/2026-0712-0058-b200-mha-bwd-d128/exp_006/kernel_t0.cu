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

// ----------------------------------------------------------------
// Device helpers
// ----------------------------------------------------------------

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

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    uint32_t smem_mbar = (uint32_t)__cvta_generic_to_shared(bar);
    uint32_t smem_ptr  = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];\n"
        :: "r"(smem_ptr), "l"((uint64_t)d), "r"(smem_mbar), "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_cp_128x128_bf16(uint32_t tmem_base, const uint8_t* smem_ptr) {
    uint32_t smem_mbar = (uint32_t)__cvta_generic_to_shared(mbar);
    uint32_t smem_ptr_int = (uint32_t)__cvta_generic_to_shared((void*)smem_ptr);
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :: "r"(tmem_base), "r"(smem_ptr_int), "r"(32768), "r"(smem_mbar) : "memory");
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

__device__ __forceinline__ void fill_tmem_128x128_fp32(uint32_t tmem_base, float val) {
    if (threadIdx.x < 128) {
        uint32_t r = __float_as_uint(val);
        for (int col = 0; col < 128; col += 4) {
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                :: "r"(r), "r"(r), "r"(r), "r"(r), "r"(col));
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }
}

__device__ __forceinline__ void read_tmem_128x128_fp32(uint32_t tmem_base, float* out_vals) {
    if (threadIdx.x < 128) {
        uint32_t r0, r1, r2, r3;
        for (int col = 0; col < 128; col += 4) {
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        for(int i = 0; i < 128; i+=4) {
            out_vals[i+0] = __uint_as_float(r0);
            out_vals[i+1] = __uint_as_float(r1);
            out_vals[i+2] = __uint_as_float(r2);
            out_vals[i+3] = __uint_as_float(r3);
            r0 = __uint_as_float(r0); // silent trick to prevent dead code elimination 
            r1 = __uint_as_float(r1); 
            r2 = __uint_as_float(r2); 
            r3 = __uint_as_float(r3); 
        }
    }
}

__device__ __forceinline__ void write_swizzled_128B(uint8_t* smem, int row, int col, float val) {
    __nv_bfloat16 bval = __float2bfloat16(val);
    int x = col * 2 / 16;
    int rem = col * 2 % 16;
    int chunk_idx = (row % 8) ^ x;
    int byte_offset = row * 256 + chunk_idx * 16 + rem;
    asm volatile("st.shared.b16 [%0], %1;" ::: "r"(byte_offset), "h"(bval));
}

__device__ __forceinline__ float read_swizzled_128B_fp32(const uint8_t* smem, int row, int col) {
    int x = col * 2 / 16;
    int rem = col * 2 % 16;
    int chunk_idx = (row % 8) ^ x;
    int byte_offset = row * 256 + chunk_idx * 16 + rem;
    __nv_bfloat16 val;
    asm volatile("ld.shared.b16 %0, [%1];" : "=h"(val) : "r"(byte_offset));
    return __bfloat162float(val);
}

__device__ __forceinline__ void load_add_to_tmem_128x128(uint32_t tmem_base, const uint8_t* smem_ptr) {
    if (threadIdx.x < 128) {
        float vals[128];
        for (int i = 0; i < 128; i++) {
            vals[i] = read_swizzled_128B_fp32(smem_ptr, threadIdx.x, i);
        }
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0 = __float_as_uint(vals[col+0]);
            uint32_t r1 = __float_as_uint(vals[col+1]);
            uint32_t r2 = __float_as_uint(vals[col+2]);
            uint32_t r3 = __float_as_uint(vals[col+3]);
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                :: "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(col));
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    }
}

__device__ __forceinline__ void store_to_smem_128x128(uint8_t* smem_ptr, const float* in_vals) {
    if (threadIdx.x < 128) {
        for (int col = 0; col < 128; col++) {
            write_swizzled_128B(smem_ptr, threadIdx.x, col, in_vals[col]);
        }
    }
}

__device__ __forceinline__ void tmem_store_bf16_row_fn(
    __nv_bfloat16* D, uint32_t tid, uint32_t M, uint32_t N,
    uint32_t m_base, uint32_t n_base, uint32_t BN) {
    uint32_t m_idx = m_base + tid;
    if (m_idx >= M) return;
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        float f0 = __uint_as_float(r0);
        float f1 = __uint_as_float(r1);
        float f2 = __uint_as_float(r2);
        float f3 = __uint_as_float(r3);
        uint32_t nc = n_base + col;
        __nv_bfloat16* out = D + (uint64_t)m_idx * N + nc;
        if (nc     < N) out[0] = __float2bfloat16(f0);
        if (nc + 1 < N) out[1] = __float2bfloat16(f1);
        if (nc + 2 < N) out[2] = __float2bfloat16(f2);
        if (nc + 3 < N) out[3] = __float2bfloat16(f3);
    }
}

extern __shared__ __align__(128) uint8_t smem_pool[];
extern __shared__ __align__(16) uint64_t mbar[1];

__global__ void __launch_bounds__(128) bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_O,
    __nv_bfloat16* dQ,
    __nv_bfloat16* dK,
    __nv_bfloat16* dV,
    const float* L,
    int S, int d_dim)
{
    extern __shared__ __align__(128) uint8_t smem_raw[];
    uint8_t* s_K  = smem_raw + 0;          // 32KB
    uint8_t* s_V  = smem_raw + 32768;      // 32KB
    uint8_t* s_Q  = smem_raw + 65536;      // 32KB
    uint8_t* s_dO = smem_raw + 98304;      // 32KB
    uint8_t* s_O  = smem_raw + 131072;     // 32KB
    float* s_D    = (float*)(smem_raw + 163840);
    float* s_L    = (float*)(smem_raw + 164352);
    
    uint32_t* smem_tmem_S   = (uint32_t*)(smem_raw + 164864); // size 4
    uint32_t* smem_tmem_dV  = (uint32_t*)(smem_raw + 164872); // size 4
    uint32_t* smem_tmem_dK  = (uint32_t*)(smem_raw + 164880); // size 4
    uint32_t* smem_tmem_dQ  = (uint32_t*)(smem_raw + 164888); // size 4

    uint8_t* s_dQ = smem_raw + 164896; // 32KB (aligned to 16)

    if (threadIdx.x == 0) {
        tmem_alloc_fn(smem_tmem_S, 128);
        tmem_alloc_fn(smem_tmem_dV, 128);
        tmem_alloc_fn(smem_tmem_dK, 128);
        tmem_alloc_fn(smem_tmem_dQ, 128);

        init_smem_barrier_fn(mbar, 1); 
        fence_smem_barrier_init_fn();
        
        mbarrier_arrive_and_expect_tx_fn(mbar, 65536); 
        int b_head = blockIdx.x; 
        int s_start = b_head * 128;
        
        tma_load_4d_fn(&tma_K, mbar, s_K, 0, s_start, 0, 1);
        tma_load_4d_fn(&tma_V, mbar, s_V, 0, s_start, 0, 1);
    }

    int phase = 0;
    mbarrier_wait_fn(mbar, phase);
    phase ^= 1;

    uint32_t tmem_S = smem_tmem_S[0];
    uint32_t tmem_dV = smem_tmem_dV[0];
    uint32_t tmem_dK = smem_tmem_dK[0];
    uint32_t tmem_dQ = smem_tmem_dQ[0];

    if (threadIdx.x < 128) {
        fill_tmem_128x128_fp32(tmem_dV, 0);
        fill_tmem_128x128_fp32(tmem_dK, 0);
    }

    int num_q_blks = (S + 127) / 128;
    float attn_scale = 1.0f / sqrtf((float)d_dim);
    int b_idx = blockIdx.x * 128;

    for (int q_blk = 0; q_blk < num_q_blks; q_blk++) {
        int q_start = q_blk * 128;
        if (threadIdx.x == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar, 98304);
            tma_load_4d_fn(&tma_Q, mbar, s_Q, 0, q_start, 0, 1);
            tma_load_4d_fn(&tma_dO, mbar, s_dO, 0, q_start, 0, 1);
            tma_load_4d_fn(&tma_O, mbar, s_O, 0, q_start, 0, 1);
        }

        if (threadIdx.x < 128) {
            int global_row = q_start + threadIdx.x;
            s_L[threadIdx.x] = (global_row < S) ? L[b_head * S + global_row] : 0.0f;
        }

        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;

        if (threadIdx.x < 128) {
            int global_row = q_start + threadIdx.x;
            if (global_row < S) {
                float sum = 0.0f;
                for(int i=0; i<128; i++) {
                    float o  = read_swizzled_128B_fp32(s_O, threadIdx.x, i);
                    float do_ = read_swizzled_128B_fp32(s_dO, threadIdx.x, i);
                    sum += o * do_;
                }
                s_D[threadIdx.x] = sum;
            } else {
                s_D[threadIdx.x] = 0.0f;
            }
        }
        __syncthreads();

        if (threadIdx.x < 128) {
            fill_tmem_128x128_fp32(tmem_S, 0);
            fill_tmem_128x128_fp32(tmem_dP, 0);
        }

        uint32_t accum = 0;
        uint32_t current_tmem_S = tmem_S;
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)s_K + K_walker * 16, 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn((uint8_t*)s_Q + K_walker * 16, 1, 1024);
            uint32_t idesc_S = make_instr_desc_fn(128, 128);
            umma_f16_cg1_fn(current_tmem_S, desc_a, desc_b, idesc_S, accum);
            accum = 1;
            current_tmem_S += 2;
        }

        float s_vals[128];
        read_tmem_128x128_fp32(tmem_S, s_vals);
        
        for(int i = 0; i < 128; i++) {
            float score_val = s_vals[i] * attn_scale;
            float p_val = fast_exp2f_fn(score_val - s_L[tid]);
            write_swizzled_128B((uint8_t*)s_P_T, tid, i, p_val);
        }
        __syncthreads();

        uint32_t accum_dP = 0;
        uint32_t current_tmem_dP = tmem_dP;
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)s_V + K_walker * 16, 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn((uint8_t*)s_dO + K_walker * 16, 1, 1024);
            uint32_t idesc_dP = make_instr_desc_fn(128, 128);
            umma_f16_cg1_fn(current_tmem_dP, desc_a, desc_b, idesc_dP, accum_dP);
            accum_dP = 1;
            current_tmem_dP += 2;
        }

        float dp_vals[128];
        read_tmem_128x128_fp32(tmem_dP, dp_vals);
        
        for(int i = 0; i < 128; i++) {
            float p_val = read_swizzled_128B_fp32((uint8_t*)s_P_T, tid, i);
            float dp_val = dp_vals[i];
            float ds_val = p_val * (dp_val - s_D[tid]);
            write_swizzled_128B((uint8_t*)s_dS_T, tid, i, ds_val);
        }
        __syncthreads();

        uint32_t accum_dV = 1;
        uint32_t current_tmem_dV = tmem_dV;
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)s_P_T + K_walker * 32, 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn((uint8_t*)s_dO + K_walker * 2048, 16384, 1024);
            uint32_t idesc_dV = make_instr_desc_fn(128, 128);
            umma_f16_cg1_fn(current_tmem_dV, desc_a, desc_b, idesc_dV, accum_dV);
            current_tmem_dV += 2;
        }

        uint32_t accum_dK = 1;
        uint32_t current_tmem_dK = tmem_dK;
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)s_dS_T + K_walker * 32, 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn((uint8_t*)s_Q + K_walker * 2048, 16384, 1024);
            uint32_t idesc_dK = make_instr_desc_fn(128, 128);
            umma_f16_cg1_fn(current_tmem_dK, desc_a, desc_b, idesc_dK, accum_dK);
            current_tmem_dK += 2;
        }

        if (threadIdx.x < 128) {
            fill_tmem_128x128_fp32(tmem_dQ, 0);
        }

        load_add_to_tmem_128x128(tmem_dQ, s_dQ);

        uint32_t accum_dQ = 1;
        uint32_t current_tmem_dQ = tmem_dQ;
        for (int K_walker = 0; K_walker < 8; K_walker++) {
            uint64_t desc_a = make_smem_desc_sm100_fn((uint8_t*)s_dQ + K_walker * 32, 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn((uint8_t*)s_K + K_walker * 2048, 16384, 1024);
            uint32_t idesc_dQ = make_instr_desc_fn(128, 128);
            umma_f16_cg1_fn(current_tmem_dQ, desc_a, desc_b, idesc_dQ, accum_dQ);
            current_tmem_dQ += 2;
        }
        
        store_to_smem_128x128(s_dQ, tmem_dQ); 
        
        __syncthreads(); 
    }

    tmem_store_bf16_row_fn(dK, tid, S, d_dim, b_idx, 0, 128);
    tmem_store_bf16_row_fn(dV, tid, S, d_dim, b_idx, 0, 128);
    tmem_store_bf16_row_fn(dQ, tid, S, d_dim, q_blk * 128, 0, 128);
}

// ----------------------------------------------------------------
// Host setup
// ----------------------------------------------------------------

CUresult create_tma_4d_descriptor(CUtensorMap* d, void* globalAddress, 
                                  uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3,
                                  uint32_t box0, uint32_t box1, uint32_t box2, uint32_t box3) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0 * 2, dim0 * dim1 * 2, dim0 * dim1 * dim2 * 2};
    cuuint32_t boxDim[4] = {box0, box1, box2, box3};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, globalAddress, globalDim, globalStrides, boxDim,
        elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

namespace tvm_ffi {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_dO, tma_O;
    create_tma_4d_descriptor(&tma_Q, Q.data_ptr(), d, S, H, B, 64, 128, 1, 1);
    create_tma_4d_descriptor(&tma_K, K.data_ptr(), d, S, H, B, 64, 128, 1, 1);
    create_tma_4d_descriptor(&tma_V, V.data_ptr(), d, S, H, B, 64, 128, 1, 1);
    create_tma_4d_descriptor(&tma_dO, dO.data_ptr(), d, S, H, B, 64, 128, 1, 1);
    create_tma_4d_descriptor(&tma_O, O.data_ptr(), d, S, H, B, 64, 128, 1, 1);
    
    int smem_size = 196608; 
    cudaFuncSetAttribute(bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    
    dim3 grid((S + 127) / 128, 1, 1);
    dim3 block(128, 1, 1);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 1;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaLaunchKernelEx(&config, bwd_kernel, 
        tma_Q, tma_K, tma_V, tma_dO, tma_O, 
        static_cast<__nv_bfloat16*>(dQ.data_ptr()), 
        static_cast<__nv_bfloat16*>(dK.data_ptr()), 
        static_cast<__nv_bfloat16*>(dV.data_ptr()), 
        static_cast<const float*>(L.data_ptr()), 
        S, d));
        
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi