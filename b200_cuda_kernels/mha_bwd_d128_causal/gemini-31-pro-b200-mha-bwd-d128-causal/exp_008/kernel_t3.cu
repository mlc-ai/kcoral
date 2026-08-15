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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        const char* err_str;                                       \
        cuGetErrorString(_e, &err_str);                            \
        fprintf(stderr, "CU error %s at %s:%d\n",                  \
                err_str, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

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

__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
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

__device__ __forceinline__ void tmem_alloc_1cta_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_1cta_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ uint64_t make_smem_desc_unswizzled(void* ptr, bool is_major_k, uint32_t dim_m_or_n, uint32_t dim_k) {
    uint32_t lbo, sbo;
    if (is_major_k) {
        sbo = 128;
        lbo = (dim_m_or_n / 8) * 128;
    } else {
        lbo = 128;
        sbo = (dim_k / 8) * 128;
    }
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)0 << 49; 
    d |= (uint64_t)0 << 61; // SWIZZLE_NONE
    return d;
}

__device__ __forceinline__ uint32_t make_idesc(bool trans_a, bool trans_b) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (trans_a ? 1u : 0u) << 15;
    d |= (trans_b ? 1u : 0u) << 16;
    d |= (16u) << 17;
    d |= (8u) << 24;
    return d;
}

__device__ __forceinline__ void umma_f16_128x128(
    uint32_t tmem_d, 
    void* smem_a, void* smem_b,
    uint32_t idesc, uint32_t accum) {
    
    bool trans_a = (idesc >> 15) & 1;
    bool trans_b = (idesc >> 16) & 1;
    
    for (int k = 0; k < 128; k += 16) {
        uint32_t current_accum = (k == 0) ? accum : 1;
        
        uint32_t offset_a = trans_a ? (k * 256) : (k * 2);
        uint64_t desc_a = make_smem_desc_unswizzled((uint8_t*)smem_a + offset_a, !trans_a, 128, 128);
        
        uint32_t offset_b = trans_b ? (k * 256) : (k * 2);
        uint64_t desc_b = make_smem_desc_unswizzled((uint8_t*)smem_b + offset_b, !trans_b, 128, 128);
        
        asm volatile(
            "{\n.reg .pred p;\n"
            "setp.ne.b32 p, %4, 0;\n"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
            :: "r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(current_accum));
    }
}

__device__ __forceinline__ uint32_t float_to_bf16_uint32(float x) {
    __nv_bfloat16 bf = __float2bfloat16(x);
    return (uint32_t)(*reinterpret_cast<uint16_t*>(&bf));
}

__device__ __forceinline__ void pointwise_P_dS(
    uint32_t tmem_base, int q_start, int kv_start, float scale, 
    float* smem_L, float* smem_D, uint8_t* smem_dS, uint8_t* smem_P) {
    
    int lane = threadIdx.x;
    int global_j = kv_start + lane;
    int key_idx = lane;
    
    for (uint32_t c = 0; c < 128; c += 4) {
        uint32_t s_regs[4], dp_regs[4];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(s_regs[0]),"=r"(s_regs[1]),"=r"(s_regs[2]),"=r"(s_regs[3]) : "r"(tmem_base + c));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(dp_regs[0]),"=r"(dp_regs[1]),"=r"(dp_regs[2]),"=r"(dp_regs[3]) : "r"(tmem_base + c + 256));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        __nv_bfloat16 ds_bf[4];
        __nv_bfloat16 p_bf[4];
        
        for (int i = 0; i < 4; ++i) {
            int col = c + i;
            int global_i = q_start + col;
            
            float s_val = __uint_as_float(s_regs[i]);
            float dp_val = __uint_as_float(dp_regs[i]);
            
            float p_val, ds_val;
            if (global_j > global_i) {
                p_val = 0.0f;
                ds_val = 0.0f;
            } else {
                float l_val = smem_L[col];
                p_val = fast_exp2f_fn((s_val * scale - l_val) * 1.44269504f);
                float d_val = smem_D[col];
                ds_val = p_val * (dp_val - d_val) * scale;
            }
            
            ds_bf[i] = __float2bfloat16(ds_val);
            p_bf[i] = __float2bfloat16(p_val);
        }
        
        *(uint64_t*)(&((__nv_bfloat16*)smem_dS)[key_idx * 128 + c]) = *(uint64_t*)ds_bf;
        *(uint64_t*)(&((__nv_bfloat16*)smem_P)[key_idx * 128 + c]) = *(uint64_t*)p_bf;
    }
}

__device__ __forceinline__ void tmem_epilogue_atomic_4w_fn(
    uint32_t tmem_col_offset,
    __nv_bfloat16* D_global, __nv_bfloat16* smem_out,
    uint32_t M, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN) {
    
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_col_offset + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * BN + col;
        smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        uint32_t global_row = m_block * BM + row;
        uint32_t col_start = lane_id * 4;
        uint32_t global_col = n_block * BN + col_start;
        if (global_row < M && global_col + 3 < N) {
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN + col_start]);
            atomicAdd((__nv_bfloat162*)(D_global + (uint64_t)global_row * N + global_col), *(__nv_bfloat162*)&data.x);
            atomicAdd((__nv_bfloat162*)(D_global + (uint64_t)global_row * N + global_col + 2), *(__nv_bfloat162*)&data.y);
        }
    }
    __syncthreads(); 
}

__device__ __forceinline__ void tmem_epilogue_direct_4w_fn(
    uint32_t tmem_col_offset,
    __nv_bfloat16* D_global, __nv_bfloat16* smem_out,
    uint32_t M, uint32_t N, uint32_t m_block, uint32_t n_block,
    uint32_t BM, uint32_t BN) {
    
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_col_offset + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        uint32_t base = threadIdx.x * BN + col;
        smem_out[base + 0] = __float2bfloat16(__uint_as_float(r0));
        smem_out[base + 1] = __float2bfloat16(__uint_as_float(r1));
        smem_out[base + 2] = __float2bfloat16(__uint_as_float(r2));
        smem_out[base + 3] = __float2bfloat16(__uint_as_float(r3));
    }
    __syncthreads();
    
    uint32_t warp_id = threadIdx.x / 32;
    uint32_t lane_id = threadIdx.x % 32;
    uint32_t num_steps = (BM + 3) / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        if (row >= BM) continue;
        uint32_t global_row = m_block * BM + row;
        uint32_t col_start = lane_id * 4;
        uint32_t global_col = n_block * BN + col_start;
        if (global_row < M && global_col + 3 < N) {
            uint2 data = *reinterpret_cast<uint2*>(&smem_out[row * BN + col_start]);
            *reinterpret_cast<uint2*>(D_global + (uint64_t)global_row * N + global_col) = data;
        }
    }
    __syncthreads();
}

__global__ void precompute_D_kernel(
    const __nv_bfloat16* __restrict__ dO,
    const __nv_bfloat16* __restrict__ O,
    float* __restrict__ D_out,
    int S, int D) 
{
    int b = blockIdx.z;
    int h = blockIdx.y;
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (s < S) {
        float sum = 0.0f;
        int64_t base = ((int64_t)b * gridDim.y + h) * S * D + (int64_t)s * D;
        for (int d = 0; d < D; ++d) {
            float do_val = __bfloat162float(dO[base + d]);
            float o_val = __bfloat162float(O[base + d]);
            sum += do_val * o_val;
        }
        D_out[((int64_t)b * gridDim.y + h) * S + s] = sum;
    }
}

__global__ void mha_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q, 
    const __grid_constant__ CUtensorMap tma_K, 
    const __grid_constant__ CUtensorMap tma_V, 
    const __grid_constant__ CUtensorMap tma_dO,
    const float* __restrict__ L, 
    const float* __restrict__ D_g,
    __nv_bfloat16* __restrict__ dQ_out,
    __nv_bfloat16* __restrict__ dK_out,
    __nv_bfloat16* __restrict__ dV_out,
    int S, int D_dim, float scale) 
{
    setmaxnreg_inc_sync_fn<256>();
    
    int kv_idx = blockIdx.x;
    int head = blockIdx.y;
    int batch = blockIdx.z;
    int kv_start = kv_idx * 128;
    int num_q = S / 128;
    
    extern __shared__ __align__(1024) uint8_t smem[];
    uint8_t* smem_K = smem + 0;
    uint8_t* smem_V = smem + 32768;
    uint8_t* smem_Q = smem + 65536;
    uint8_t* smem_dO = smem + 98304;
    uint8_t* smem_dS = smem + 131072;
    uint8_t* smem_P = smem + 163840;
    float* smem_L = (float*)(smem + 196608);
    float* smem_D = (float*)(smem + 196608 + 512);
    uint64_t* mbar_kv = (uint64_t*)(smem + 196608 + 1024);
    uint64_t* mbar_q = (uint64_t*)(smem + 196608 + 1024 + 8);
    uint64_t* mbar_mma = (uint64_t*)(smem + 196608 + 1024 + 16);
    uint32_t* smem_tmem_base = (uint32_t*)(smem + 196608 + 1024 + 24);
    
    if (threadIdx.x < 32) {
        tmem_alloc_1cta_fn(smem_tmem_base, 512);
    }
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(mbar_kv, 128);
        init_smem_barrier_fn(mbar_q, 128);
        init_smem_barrier_fn(mbar_mma, 128);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();
    
    uint32_t tmem_base = *smem_tmem_base;
    uint32_t kv_phase = 0;
    uint32_t q_phase = 0;
    uint32_t mma_phase = 0;
    
    if (threadIdx.x == 0) {
        int outer_coord = (batch * gridDim.y + head) * S + kv_start;
        tma_load_2d_fn(&tma_K, mbar_kv, smem_K, 0, outer_coord);
        tma_load_2d_fn(&tma_V, mbar_kv, smem_V, 0, outer_coord);
        mbarrier_arrive_and_expect_tx_fn(mbar_kv, 65536);
    } else {
        mbarrier_arrive_fn(mbar_kv);
    }
    mbarrier_wait_fn(mbar_kv, kv_phase);
    kv_phase ^= 1;
    __syncthreads();
    
    for (int q_idx = kv_idx; q_idx < num_q; ++q_idx) {
        int q_start = q_idx * 128;
        
        if (threadIdx.x == 0) {
            int outer_coord = (batch * gridDim.y + head) * S + q_start;
            tma_load_2d_fn(&tma_Q, mbar_q, smem_Q, 0, outer_coord);
            tma_load_2d_fn(&tma_dO, mbar_q, smem_dO, 0, outer_coord);
            mbarrier_arrive_and_expect_tx_fn(mbar_q, 65536);
        } else {
            mbarrier_arrive_fn(mbar_q);
        }
        
        int tid = threadIdx.x;
        int64_t base_L = ((int64_t)batch * gridDim.y + head) * S + q_start;
        smem_L[tid] = L[base_L + tid];
        smem_D[tid] = D_g[base_L + tid];
        
        mbarrier_wait_fn(mbar_q, q_phase);
        q_phase ^= 1;
        __syncthreads();
        
        if (threadIdx.x == 0) {
            umma_f16_128x128(tmem_base + 0, smem_K, smem_Q, make_idesc(false, false), 0);
            umma_f16_128x128(tmem_base + 256, smem_V, smem_dO, make_idesc(false, false), 0);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(mbar_mma)));
        } else {
            mbarrier_arrive_fn(mbar_mma);
        }
        mbarrier_wait_fn(mbar_mma, mma_phase);
        mma_phase ^= 1;
        
        pointwise_P_dS(tmem_base, q_start, kv_start, scale, smem_L, smem_D, smem_dS, smem_P);
        __syncthreads();
        
        uint32_t accum_dkv = (q_idx == kv_idx) ? 0 : 1;
        
        if (threadIdx.x == 0) {
            umma_f16_128x128(tmem_base + 128, smem_P, smem_dO, make_idesc(false, true), accum_dkv); // dV
            umma_f16_128x128(tmem_base + 384, smem_dS, smem_Q, make_idesc(false, true), accum_dkv); // dK
            umma_f16_128x128(tmem_base + 0, smem_dS, smem_K, make_idesc(true, true), 0); // dQ, reuse col 0
            
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(mbar_mma)));
        } else {
            mbarrier_arrive_fn(mbar_mma);
        }
        mbarrier_wait_fn(mbar_mma, mma_phase);
        mma_phase ^= 1;
        
        int64_t block_base_global = ((int64_t)batch * gridDim.y + head) * S * D_dim;
        tmem_epilogue_atomic_4w_fn(tmem_base + 0, dQ_out + block_base_global, (__nv_bfloat16*)smem_Q, S, D_dim, q_idx, 0, 128, 128);
    }
    
    int64_t block_base_global = ((int64_t)batch * gridDim.y + head) * S * D_dim;
    tmem_epilogue_direct_4w_fn(tmem_base + 128, dV_out + block_base_global, (__nv_bfloat16*)smem_Q, S, D_dim, kv_idx, 0, 128, 128);
    tmem_epilogue_direct_4w_fn(tmem_base + 384, dK_out + block_base_global, (__nv_bfloat16*)smem_Q, S, D_dim, kv_idx, 0, 128, 128);
    
    if (threadIdx.x < 32) {
        tmem_dealloc_1cta_fn(tmem_base, 512);
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
    int64_t D = 128;
    
    float* D_g;
    CUDA_CHECK(cudaMalloc(&D_g, B * H * S * sizeof(float)));
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    dim3 grid_D((S + 127) / 128, H, B);
    dim3 block_D(128);
    precompute_D_kernel<<<grid_D, block_D, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(dO.data_ptr()),
        static_cast<const __nv_bfloat16*>(O.data_ptr()),
        D_g, S, D
    );
    CUDA_CHECK(cudaGetLastError());
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_dO;
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, Q.data_ptr(), D, B * H * S, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, K.data_ptr(), D, B * H * S, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, V.data_ptr(), D, B * H * S, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_dO, dO.data_ptr(), D, B * H * S, 128, 128, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    CUDA_CHECK(cudaMemsetAsync(dQ.data_ptr(), 0, B * H * S * D * sizeof(uint16_t), stream));
    
    dim3 grid((S + 127) / 128, H, B);
    dim3 block(128);
    float scale = 1.0f / sqrtf((float)D);
    
    int smem_size = 200 * 1024;
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    mha_bwd_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_dO,
        static_cast<const float*>(L.data_ptr()),
        D_g,
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        S, D, scale
    );
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaFreeAsync(D_g, stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

} // namespace tvm_ffi_example_cuda