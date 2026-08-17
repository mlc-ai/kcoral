#include <cuda_bf16.h>
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

namespace tvm_ffi_attention_bwd {

// ---------------------- Hardware Helper Functions ----------------------

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
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
        :: "r"(a), "r"(ncols));
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

__device__ __forceinline__ uint64_t advance_desc(uint64_t desc, int step) {
    return desc + (step * 16);
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

__device__ __forceinline__ void umma_f16_cg1_tmem_a(
    uint32_t tmem_c, uint32_t tmem_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
        :: "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

#define tmem_addr(col, row) ((uint32_t)((row << 16) | (col & 0xFFFF)))
__device__ __forceinline__ float tmem_read_f32(uint32_t col, int row) {
    float val;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(val) : "r"(tmem_addr(col, row)));
    return val;
}
__device__ __forceinline__ void tmem_write_f32(uint32_t col, int row, float val) {
    asm volatile("st.shared.f32 [%0], %1;" :: "r"(tmem_addr(col, row)), "f"(val));
}

// ---------------------- Kernel Logic ----------------------

__device__ __forceinline__ void transpose_128x128(const __nv_bfloat16* src, __nv_bfloat16* dst) {
    int tid = threadIdx.x;
    for (int i = 0; i < 128; i++) {
        dst[tid * 128 + i] = src[i * 128 + tid];
    }
}

__device__ __forceinline__ void read_tmem_S_P_T(int tid) {
    float s_val0 = tmem_read_f32(tmem_S_P_T + tid, 0);
    float s_val1 = tmem_read_f32(tmem_S_P_T + tid + 1, 0);
    int j0 = tid;
    int j1 = tid + 1;
    
    float s0 = s_val0 * scale;
    float p0 = fast_exp2f_fn((s0 - smem_LSE[tid]) * 1.4426950408889634f);
    if (q_base + tid < k_base + j0 || q_base + tid >= S_len || k_base + j0 >= S_len) {
        p0 = 0;
    }
    
    float s1 = s_val1 * scale;
    float p1 = fast_exp2f_fn((s1 - smem_LSE[tid]) * 1.4426950408889634f);
    if (q_base + tid < k_base + j1 || q_base + tid >= S_len || k_base + j1 >= S_len) {
        p1 = 0;
    }
    
    uint16_t bp0 = __float2bfloat16(p0);
    uint16_t bp1 = __float2bfloat16(p1);
    uint32_t packed = bp0 | (bp1 << 16);
    *tmem_addr(tmem_S_P_T + tid, 0) = packed;
}

__device__ __forceinline__ void read_tmem_dP_ST_T(int tid) {
    float dp_val0 = tmem_read_f32(tmem_dP_ST_T + tid, 0);
    float dp_val1 = tmem_read_f32(tmem_dP_ST_T + tid + 1, 0);
    
    float s_val0 = tmem_read_f32(tmem_S_P_T + tid, 0);
    float s_val1 = tmem_read_f32(tmem_S_P_T + tid + 1, 0);
    
    float s0 = s_val0 * scale;
    float p0 = fast_exp2f_fn((s0 - smem_LSE[tid]) * 1.4426950408889634f);
    int j0 = tid;
    if (q_base + tid < k_base + j0 || q_base + tid >= S_len || k_base + j0 >= S_len) {
        p0 = 0;
    }
    
    float s1 = s_val1 * scale;
    float p1 = fast_exp2f_fn((s1 - smem_LSE[tid]) * 1.4426950408889634f);
    int j1 = tid + 1;
    if (q_base + tid < k_base + j1 || q_base + tid >= S_len || k_base + j1 >= S_len) {
        p1 = 0;
    }
    
    float ds0 = p0 * (dp_val0 - smem_D[tid]);
    float ds1 = p1 * (dp_val1 - smem_D[tid]);
    
    uint16_t bds0 = __float2bfloat16(ds0);
    uint16_t bds1 = __float2bfloat16(ds1);
    uint32_t packed = bds0 | (bds1 << 16);
    
    *tmem_addr(tmem_dP_ST_T + tid, 0) = packed;
}

__device__ __forceinline__ void write_transposed_to_smem(__nv_bfloat16* dst, int tid) {
    float s_val0 = tmem_read_f32(tmem_dP_ST_T + tid, 0);
    float s_val1 = tmem_read_f32(tmem_dP_ST_T + tid + 1, 0);
    
    int r0 = tid;
    int c0 = 0;
    int swizzled_x0 = ((r0 % 8) ^ (c0 / 8)) * 8 + (c0 % 8);
    dst[r0 * 128 + swizzled_x0] = __float2bfloat16(s_val0);
    
    int r1 = tid + 1;
    int c1 = 0;
    int swizzled_x1 = ((r1 % 8) ^ (c1 / 8)) * 8 + (c1 % 8);
    dst[r1 * 128 + swizzled_x1] = __float2bfloat16(s_val1);
}

extern __shared__ __align__(1024) uint8_t smem_pool[];
__nv_bfloat16* smem_Q = (__nv_bfloat16*)smem_pool;
__nv_bfloat16* smem_K = smem_Q + 128*128;
__nv_bfloat16* smem_V = smem_K + 128*128;
__nv_bfloat16* smem_O = smem_V + 128*128;
__nv_bfloat16* smem_dO = smem_O + 128*128;
__nv_bfloat16* smem_Q_T = smem_dO + 128*128;
__nv_bfloat16* smem_K_T = smem_Q_T + 128*128;
__nv_bfloat16* smem_V_T = smem_K_T + 128*128;
__nv_bfloat16* smem_O_T = smem_V_T + 128*128;
__nv_bfloat16* smem_dO_T = smem_O_T + 128*128;

uint64_t* mbar_Q = (uint64_t*)(smem_dO_T + 128*128);
uint64_t* mbar_K = mbar_Q + 1;
uint64_t* mbar_V = mbar_K + 1;
uint64_t* mbar_O = mbar_V + 1;
uint64_t* mbar_dO = mbar_O + 1;

float* smem_D = (float*)(mbar_dO + 1);
float* smem_LSE = smem_D + 128;

__device__ float scale;
__device__ int q_base;
__device__ int k_base;
__device__ int S_len;

uint32_t tmem_S_P_T;
uint32_t tmem_dP_ST_T;
uint32_t tmem_dQ;
uint32_t tmem_dV;

__global__ void __launch_bounds__(128, 1) bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* L, float* dK_fp32, float* dV_fp32,
    __nv_bfloat16* dQ, int S_len_val, float scale_val)
{
    S_len = S_len_val;
    scale = scale_val;
    
    int q_tile = blockIdx.x;
    int head_idx = blockIdx.y;
    int num_tiles = (S_len + 127) / 128;
    if (q_tile >= num_tiles) return;
    
    q_base = q_tile * 128;
    uint64_t head_offset = head_idx * S_len * 128;
    int tid = threadIdx.x;
    
    if (tid == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_K, 1);
        init_smem_barrier_fn(mbar_V, 1);
        init_smem_barrier_fn(mbar_O, 1);
        init_smem_barrier_fn(mbar_dO, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    if (tid == 0) {
        tmem_alloc_fn(&tmem_S_P_T, 128);
        tmem_alloc_fn(&tmem_dP_ST_T, 128);
        tmem_alloc_fn(&tmem_dQ, 128);
        tmem_alloc_fn(&tmem_dV, 128);
    }
    __syncthreads();
    
    uint32_t phase_Q = 0, phase_K = 0, phase_V = 0, phase_O = 0, phase_dO = 0;
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 16384);
        tma_load_2d_fn(&tma_Q, mbar_Q, smem_Q, 0, head_idx * S_len + q_base);
        tma_load_2d_fn(&tma_Q, mbar_Q, (char*)smem_Q + 8192, 64, head_idx * S_len + q_base);
        
        mbarrier_arrive_and_expect_tx_fn(mbar_O, 16384);
        tma_load_2d_fn(&tma_O, mbar_O, smem_O, 0, head_idx * S_len + q_base);
        tma_load_2d_fn(&tma_O, mbar_O, (char*)smem_O + 8192, 64, head_idx * S_len + q_base);
        
        mbarrier_arrive_and_expect_tx_fn(mbar_dO, 16384);
        tma_load_2d_fn(&tma_dO, mbar_dO, smem_dO, 0, head_idx * S_len + q_base);
        tma_load_2d_fn(&tma_dO, mbar_dO, (char*)smem_dO + 8192, 64, head_idx * S_len + q_base);
    }
    
    mbarrier_wait_fn(mbar_Q, phase_Q);
    mbarrier_wait_fn(mbar_O, phase_O);
    mbarrier_wait_fn(mbar_dO, phase_dO);
    phase_Q ^= 1; phase_O ^= 1; phase_dO ^= 1;
    
    if (tid < 128) {
        float sum = 0;
        if (q_base + tid < S_len) {
            for (int d = 0; d < 128; d++) {
                sum += __bfloat162float(smem_dO[tid*128+d]) * __bfloat162float(smem_O[tid*128+d]);
            }
        }
        smem_D[tid] = sum;
        smem_LSE[tid] = (q_base + tid < S_len) ? L[head_idx * S_len + q_base + tid] : 0;
    }
    __syncthreads();
    
    transpose_128x128(smem_Q, smem_Q_T);
    transpose_128x128(smem_O, smem_O_T);
    transpose_128x128(smem_dO, smem_dO_T);
    
    uint32_t idesc_S = make_instr_desc_fn(128, 128);
    uint32_t idesc_dP = make_instr_desc_fn(128, 128);
    uint32_t idesc_dV = make_instr_desc_fn(128, 128);
    uint32_t idesc_dK = make_instr_desc_fn(128, 128);
    uint32_t idesc_dQ = make_instr_desc_fn(128, 128);
    
    for (int k_tile = 0; k_tile <= q_tile && k_tile < num_tiles; k_tile++) {
        k_base = k_tile * 128;
        
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_K, 16384);
            tma_load_2d_fn(&tma_K, mbar_K, smem_K, 0, head_idx * S_len + k_base);
            tma_load_2d_fn(&tma_K, mbar_K, (char*)smem_K + 8192, 64, head_idx * S_len + k_base);
            
            mbarrier_arrive_and_expect_tx_fn(mbar_V, 16384);
            tma_load_2d_fn(&tma_V, mbar_V, smem_V, 0, head_idx * S_len + k_base);
            tma_load_2d_fn(&tma_V, mbar_V, (char*)smem_V + 8192, 64, head_idx * S_len + k_base);
        }
        
        mbarrier_wait_fn(mbar_K, phase_K);
        mbarrier_wait_fn(mbar_V, phase_V);
        phase_K ^= 1; phase_V ^= 1;
        
        __syncthreads();
        transpose_128x128(smem_K, smem_K_T);
        transpose_128x128(smem_V, smem_V_T);
        __syncthreads();
        
        if (tid == 0) {
            uint64_t desc_a = make_smem_desc_sm100_fn(smem_K_T, 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn(smem_Q_T, 16384, 1024);
            for (int k_step = 0; k_step < 8; k_step++) {
                uint32_t accum = (k_step == 0) ? 0 : 1;
                uint64_t da = advance_desc(desc_a, k_step);
                uint64_t db = advance_desc(desc_b, k_step);
                umma_f16_cg1_fn(tmem_S_P_T, da, db, idesc_S, accum);
            }
        }
        __syncthreads();
        
        read_tmem_S_P_T(tid);
        
        if (tid == 0) {
            uint64_t desc_a = make_smem_desc_sm100_fn(smem_V_T, 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn(smem_dO_T, 16384, 1024);
            for (int k_step = 0; k_step < 8; k_step++) {
                uint32_t accum = (k_step == 0) ? 0 : 1;
                uint64_t da = advance_desc(desc_a, k_step);
                uint64_t db = advance_desc(desc_b, k_step);
                umma_f16_cg1_fn(tmem_dP_ST_T, da, db, idesc_dP, accum);
            }
        }
        __syncthreads();
        
        read_tmem_dP_ST_T(tid);
        
        if (tid == 0) {
            uint64_t desc_b = make_smem_desc_sm100_fn(smem_O_T, 16384, 1024);
            for (int k_step = 0; k_step < 8; k_step++) {
                uint32_t accum = (k_step == 0) ? 0 : 1;
                uint64_t db = advance_desc(desc_b, k_step);
                umma_f16_cg1_tmem_a(tmem_dV, tmem_S_P_T, db, idesc_dV, accum);
            }
        }
        __syncthreads();
        
        if (tid < 128) {
            float dv0 = tmem_read_f32(tmem_dV + tid, 0);
            float dv1 = tmem_read_f32(tmem_dV + tid + 1, 0);
            if (k_base + tid < S_len) {
                atomicAdd(&dV_fp32[head_offset + (k_base + tid) * 128 + 0], dv0);
                atomicAdd(&dV_fp32[head_offset + (k_base + tid) * 128 + 1], dv1);
            }
        }
        __syncthreads();
        
        if (tid == 0) {
            uint64_t desc_b = make_smem_desc_sm100_fn(smem_Q_T, 16384, 1024);
            for (int k_step = 0; k_step < 8; k_step++) {
                uint32_t accum = (k_step == 0) ? 0 : 1;
                uint64_t db = advance_desc(desc_b, k_step);
                umma_f16_cg1_tmem_a(tmem_dV, tmem_dP_ST_T, db, idesc_dK, accum);
            }
        }
        __syncthreads();
        
        if (tid < 128) {
            float dk0 = tmem_read_f32(tmem_dV + tid, 0);
            float dk1 = tmem_read_f32(tmem_dV + tid + 1, 0);
            if (k_base + tid < S_len) {
                atomicAdd(&dK_fp32[head_offset + (k_base + tid) * 128 + 0], dk0);
                atomicAdd(&dK_fp32[head_offset + (k_base + tid) * 128 + 1], dk1);
            }
        }
        __syncthreads();
        
        write_transposed_to_smem(smem_O, tid); 
        
        if (tid == 0) {
            uint64_t desc_a = make_smem_desc_sm100_fn(smem_O, 1, 1024);
            uint64_t desc_b = make_smem_desc_sm100_fn(smem_K_T, 16384, 1024);
            for (int k_step = 0; k_step < 8; k_step++) {
                uint64_t da = advance_desc(desc_a, k_step);
                uint64_t db = advance_desc(desc_b, k_step);
                umma_f16_cg1_fn(tmem_dQ, da, db, idesc_dQ, 1);
            }
        }
        __syncthreads();
    }
    
    if (tid < 128) {
        float dq0 = tmem_read_f32(tmem_dQ + tid, 0);
        float dq1 = tmem_read_f32(tmem_dQ + tid + 1, 0);
        
        int row = tid;
        if (q_base + row < S_len) {
            int col = 0;
            int swizzled_x = ((row % 8) ^ (col / 8)) * 8 + (col % 8);
            smem_O[row * 128 + swizzled_x] = __float2bfloat16(dq0);
            
            col = 1;
            swizzled_x = ((row % 8) ^ (col / 8)) * 8 + (col % 8);
            smem_O[row * 128 + swizzled_x] = __float2bfloat16(dq1);
        }
    }
    __syncthreads();
    
    for (int idx = threadIdx.x; idx < 128*128/8; idx += 128) {
        int row = (idx * 8) / 128;
        if (q_base + row < S_len) {
            int col = (idx * 8) % 128;
            *reinterpret_cast<float4*>(&dQ[head_offset + (q_base + row) * 128 + col]) = *reinterpret_cast<float4*>(&smem_O[row * 128 + col]);
        }
    }
}

__global__ void fp32_to_bf16_kernel(const float* in, __nv_bfloat16* out, size_t n) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2bfloat16(in[idx]);
    }
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

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);
    
    float scale = 1.0f / sqrtf((float)d);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    size_t num_elements = B * H * S * d;
    float* dV_fp32 = nullptr;
    float* dK_fp32 = nullptr;
    
    CUDA_CHECK(cudaMallocAsync(&dV_fp32, num_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&dK_fp32, num_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dV_fp32, 0, num_elements * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(dK_fp32, 0, num_elements * sizeof(float), stream));
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    constexpr uint32_t ATOM_KMODE_DIM = 64;
    constexpr uint32_t ATOM_MMODE_DIM = 128;
    
    create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), d, B*H*S, ATOM_KMODE_DIM, ATOM_MMODE_DIM, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), d, B*H*S, ATOM_KMODE_DIM, ATOM_MMODE_DIM, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), d, B*H*S, ATOM_KMODE_DIM, ATOM_MMODE_DIM, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_O, O.data_ptr(), d, B*H*S, ATOM_KMODE_DIM, ATOM_MMODE_DIM, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    create_tma_2d_descriptor_2B(&tma_dO, dO.data_ptr(), d, B*H*S, ATOM_KMODE_DIM, ATOM_MMODE_DIM, 
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    CUDA_CHECK(cudaFuncSetAttribute(bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 360000));
    
    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128);
    
    bwd_kernel<<<grid, block, 360000, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, tma_dO,
        static_cast<const float*>(L.data_ptr()),
        dV_fp32, dK_fp32,
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        S, scale);
    
    CUDA_CHECK(cudaGetLastError());
    
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    fp32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(dV_fp32, static_cast<__nv_bfloat16*>(dV.data_ptr()), num_elements);
    fp32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(dK_fp32, static_cast<__nv_bfloat16*>(dK.data_ptr()), num_elements);
    
    CUDA_CHECK(cudaFreeAsync(dV_fp32, stream));
    CUDA_CHECK(cudaFreeAsync(dK_fp32, stream));
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_attention_bwd::run);

}  // namespace tvm_ffi_attention_bwd