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

#define CU_CHECK(call) do {                                        \
    CUresult _e = (call);                                          \
    if (_e != CUDA_SUCCESS) {                                      \
        const char* errStr;                                        \
        cuGetErrorName(_e, &errStr);                               \
        fprintf(stderr, "CU error %s at %s:%d\n", errStr,          \
                __FILE__, __LINE__);                               \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace flash_attn_causal_v3 {

#define Q_BLK 128
#define KV_BLK 128
#define D_DIM 128

struct __align__(1024) SharedStorage {
    __align__(1024) __nv_bfloat16 Q[Q_BLK * D_DIM];
    __align__(1024) __nv_bfloat16 K[KV_BLK * D_DIM];
    __align__(1024) __nv_bfloat16 V[KV_BLK * D_DIM];
    __align__(1024) __nv_bfloat16 P[Q_BLK * KV_BLK];
    __align__(1024) float O[Q_BLK * D_DIM];
    float m[Q_BLK];
    float l[Q_BLK];
    float scale[Q_BLK];
    uint64_t mbar_q;
    uint64_t mbar_k;
    uint64_t mbar_v;
    uint64_t mbar_umma;
    uint32_t tmem_S;
    uint32_t tmem_O_new;
};

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
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

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_swizzle_fn(uint32_t addr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, int a_major, int b_major) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= ((uint32_t)a_major << 15);   
    d |= ((uint32_t)b_major << 16);   
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
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
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.b64"
        " [%0];"
        :: "r"(a));
}

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
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
    asm("mov.b32 %0, {%1, %2};" : "=r"(result) : "h"(*reinterpret_cast<uint16_t*>(&a)), "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}


extern __shared__ __align__(1024) char smem_buf[];

__global__ void __launch_bounds__(128) flash_attn_causal_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* __restrict__ O_ptr,
    float* __restrict__ LSE_ptr,
    int B, int H, int S) 
{
    setmaxnreg_inc_sync_fn<256>();

    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_buf);

    int b = blockIdx.y / H;
    int h = blockIdx.y % H;
    int block_q = blockIdx.x;
    int tid = threadIdx.x;

    if (tid == 0) {
        init_smem_barrier_fn(&smem.mbar_q, 1);
        init_smem_barrier_fn(&smem.mbar_k, 1);
        init_smem_barrier_fn(&smem.mbar_v, 1);
        init_smem_barrier_fn(&smem.mbar_umma, 1);
        fence_smem_barrier_init_fn();
    }
    __syncthreads();

    for (int i = tid; i < Q_BLK * D_DIM; i += 128) smem.O[i] = 0.0f;
    if (tid < Q_BLK) {
        smem.m[tid] = -INFINITY;
        smem.l[tid] = 0.0f;
    }
    __syncthreads();

    if (tid < 32) {
        tmem_alloc_cg1_fn(&smem.tmem_S, 128);
        tmem_alloc_cg1_fn(&smem.tmem_O_new, 128);
    }
    __syncthreads();
    uint32_t tmem_S = smem.tmem_S;
    uint32_t tmem_O_new = smem.tmem_O_new;

    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(&smem.mbar_q, Q_BLK * D_DIM * 2);
        tma_load_4d_fn(&tma_Q, &smem.mbar_q, smem.Q, 0, block_q * Q_BLK, h, b);
    }

    int max_kv = block_q;
    uint32_t q_phase = 0;
    uint32_t k_phase = 0;
    uint32_t v_phase = 0;
    uint32_t umma_phase = 0;

    for (int kv_idx = 0; kv_idx <= max_kv; ++kv_idx) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(&smem.mbar_k, KV_BLK * D_DIM * 2);
            tma_load_4d_fn(&tma_K, &smem.mbar_k, smem.K, 0, kv_idx * KV_BLK, h, b);
            
            mbarrier_arrive_and_expect_tx_fn(&smem.mbar_v, KV_BLK * D_DIM * 2);
            tma_load_4d_fn(&tma_V, &smem.mbar_v, smem.V, 0, kv_idx * KV_BLK, h, b);
        }
        
        if (kv_idx == 0) {
            mbarrier_wait_fn(&smem.mbar_q, q_phase);
            q_phase ^= 1;
        }
        
        mbarrier_wait_fn(&smem.mbar_k, k_phase);
        k_phase ^= 1;
        
        __syncthreads();
        fence_async_shared_fn(); 
        __syncthreads();
        
        if (tid == 0) {
            uint32_t idesc_qk = make_instr_desc_fn(128, 128, 0, 0); 
            for (int k = 0; k < 128; k += 16) {
                uint32_t q_addr = (uint32_t)__cvta_generic_to_shared(smem.Q) + k * 2;
                uint32_t k_addr = (uint32_t)__cvta_generic_to_shared(smem.K) + k * 2;
                uint64_t desc_q = make_smem_desc_sm100_swizzle_fn(q_addr, 16, 1024);
                uint64_t desc_k = make_smem_desc_sm100_swizzle_fn(k_addr, 16, 1024);
                uint32_t accum = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_S, desc_q, desc_k, idesc_qk, accum);
            }
            umma_commit_cg1_fn(&smem.mbar_umma);
        }
        mbarrier_wait_fn(&smem.mbar_umma, umma_phase);
        umma_phase ^= 1;
        
        uint32_t r_S[32][4];
        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t taddr = tmem_S + col;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r_S[col/4][0]),"=r"(r_S[col/4][1]),"=r"(r_S[col/4][2]),"=r"(r_S[col/4][3]) : "r"(taddr));
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        float row_max = -INFINITY;
        for (uint32_t col = 0; col < 128; col += 4) {
            int global_q = block_q * Q_BLK + tid;
            int global_k0 = kv_idx * KV_BLK + col;
            
            float f0 = __uint_as_float(r_S[col/4][0]) * 0.08838834764f; 
            float f1 = __uint_as_float(r_S[col/4][1]) * 0.08838834764f;
            float f2 = __uint_as_float(r_S[col/4][2]) * 0.08838834764f;
            float f3 = __uint_as_float(r_S[col/4][3]) * 0.08838834764f;
            
            if (global_k0 + 0 > global_q || global_k0 + 0 >= S) f0 = -INFINITY;
            if (global_k0 + 1 > global_q || global_k0 + 1 >= S) f1 = -INFINITY;
            if (global_k0 + 2 > global_q || global_k0 + 2 >= S) f2 = -INFINITY;
            if (global_k0 + 3 > global_q || global_k0 + 3 >= S) f3 = -INFINITY;
            
            row_max = fmaxf(row_max, fmaxf(f0, fmaxf(f1, fmaxf(f2, f3))));
            
            r_S[col/4][0] = __float_as_uint(f0);
            r_S[col/4][1] = __float_as_uint(f1);
            r_S[col/4][2] = __float_as_uint(f2);
            r_S[col/4][3] = __float_as_uint(f3);
        }
        
        float prev_m = smem.m[tid];
        float new_m = fmaxf(prev_m, row_max);
        smem.m[tid] = new_m;
        
        float row_sum = 0.0f;
        for (uint32_t col = 0; col < 128; col += 4) {
            float f0 = __uint_as_float(r_S[col/4][0]);
            float f1 = __uint_as_float(r_S[col/4][1]);
            float f2 = __uint_as_float(r_S[col/4][2]);
            float f3 = __uint_as_float(r_S[col/4][3]);
            
            float p0 = (f0 == -INFINITY) ? 0.0f : fast_exp2f_fn((f0 - new_m) * 1.44269504089f);
            float p1 = (f1 == -INFINITY) ? 0.0f : fast_exp2f_fn((f1 - new_m) * 1.44269504089f);
            float p2 = (f2 == -INFINITY) ? 0.0f : fast_exp2f_fn((f2 - new_m) * 1.44269504089f);
            float p3 = (f3 == -INFINITY) ? 0.0f : fast_exp2f_fn((f3 - new_m) * 1.44269504089f);
            
            row_sum += p0 + p1 + p2 + p3;
            
            uint32_t p01 = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
            uint32_t p23 = pack_bf16_fn(__float_as_uint(p2), __float_as_uint(p3));
            
            int x_chunk = col / 8;
            int x_offset = (col % 8) * 2;
            int swizzled_chunk = (tid % 8) ^ x_chunk;
            int byte_offset = tid * 256 + swizzled_chunk * 16 + x_offset;
            *reinterpret_cast<uint2*>((char*)smem.P + byte_offset) = make_uint2(p01, p23);
        }
        
        float prev_l = smem.l[tid];
        float scale = (prev_m == -INFINITY && new_m == -INFINITY) ? 1.0f : fast_exp2f_fn((prev_m - new_m) * 1.44269504089f);
        smem.l[tid] = prev_l * scale + row_sum;
        smem.scale[tid] = scale;
        
        mbarrier_wait_fn(&smem.mbar_v, v_phase);
        v_phase ^= 1;
        
        __syncthreads();
        fence_async_shared_fn();
        __syncthreads();
        
        if (tid == 0) {
            uint32_t idesc_pv = make_instr_desc_fn(128, 128, 0, 1);
            for (int k = 0; k < 128; k += 16) {
                uint32_t p_addr = (uint32_t)__cvta_generic_to_shared(smem.P) + k * 2;
                uint32_t v_addr = (uint32_t)__cvta_generic_to_shared(smem.V) + k * D_DIM * 2;
                uint64_t desc_p = make_smem_desc_sm100_swizzle_fn(p_addr, 16, 1024);
                uint64_t desc_v = make_smem_desc_sm100_swizzle_fn(v_addr, 16384, 1024);
                uint32_t accum = (k == 0) ? 0 : 1;
                umma_f16_cg1_fn(tmem_O_new, desc_p, desc_v, idesc_pv, accum);
            }
            umma_commit_cg1_fn(&smem.mbar_umma);
        }
        mbarrier_wait_fn(&smem.mbar_umma, umma_phase);
        umma_phase ^= 1;
        
        uint32_t r_O[32][4];
        for (uint32_t col = 0; col < 128; col += 4) {
            uint32_t taddr = tmem_O_new + col;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r_O[col/4][0]),"=r"(r_O[col/4][1]),"=r"(r_O[col/4][2]),"=r"(r_O[col/4][3]) : "r"(taddr));
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        for (uint32_t col = 0; col < 128; col += 4) {
            int base = tid * 128 + col;
            float4 o_vec = *reinterpret_cast<float4*>(&smem.O[base]);
            float s = smem.scale[tid];
            o_vec.x = o_vec.x * s + __uint_as_float(r_O[col/4][0]);
            o_vec.y = o_vec.y * s + __uint_as_float(r_O[col/4][1]);
            o_vec.z = o_vec.z * s + __uint_as_float(r_O[col/4][2]);
            o_vec.w = o_vec.w * s + __uint_as_float(r_O[col/4][3]);
            *reinterpret_cast<float4*>(&smem.O[base]) = o_vec;
        }
    }

    if (tid < 32) {
        tmem_dealloc_cg1_fn(tmem_S, 128);
        tmem_dealloc_cg1_fn(tmem_O_new, 128);
    }
    __syncthreads();
    
    for (int i = tid * 2; i < Q_BLK * D_DIM; i += 128 * 2) {
        int r = i / D_DIM;
        int c = i % D_DIM;
        int global_q = block_q * Q_BLK + r;
        if (global_q < S) {
            float out0 = smem.O[i] / smem.l[r];
            float out1 = smem.O[i+1] / smem.l[r];
            uint32_t packed = pack_bf16_fn(__float_as_uint(out0), __float_as_uint(out1));
            int64_t out_idx = (int64_t(b) * H * S + int64_t(h) * S + global_q) * D_DIM + c;
            *reinterpret_cast<uint32_t*>(&O_ptr[out_idx]) = packed;
        }
    }

    if (tid < Q_BLK) {
        int global_q = block_q * Q_BLK + tid;
        if (global_q < S) {
            int64_t lse_idx = int64_t(b) * H * S + int64_t(h) * S + global_q;
            LSE_ptr[lse_idx] = smem.m[tid] + logf(smem.l[tid]);
        }
    }
}

CUresult create_tma_4d_descriptor_swizzled(CUtensorMap* d, void* globalAddress, 
                                  uint64_t dim0, uint64_t dim1, uint64_t dim2, uint64_t dim3,
                                  uint32_t box0, uint32_t box1, uint32_t box2, uint32_t box3) {
    cuuint64_t globalDim[4] = {dim0, dim1, dim2, dim3};
    cuuint64_t globalStrides[3] = {dim0 * 2, dim1 * dim0 * 2, dim2 * dim1 * dim0 * 2};
    cuuint32_t boxDim[4] = {box0, box1, box2, box3};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, 
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) 
{
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);

    if (S == 0) return;

    const __nv_bfloat16* q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* k_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* v_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());

    CUtensorMap tma_Q, tma_K, tma_V;
    CU_CHECK(create_tma_4d_descriptor_swizzled(&tma_Q, (void*)q_ptr, 128, S, H, B, 128, 128, 1, 1));
    CU_CHECK(create_tma_4d_descriptor_swizzled(&tma_K, (void*)k_ptr, 128, S, H, B, 128, 128, 1, 1));
    CU_CHECK(create_tma_4d_descriptor_swizzled(&tma_V, (void*)v_ptr, 128, S, H, B, 128, 128, 1, 1));

    int64_t num_q_blocks = (S + 127) / 128;
    dim3 grid(num_q_blocks, B * H);
    dim3 block(128); 

    int smem_size = sizeof(SharedStorage);

    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    CUDA_CHECK(cudaFuncSetAttribute(flash_attn_causal_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    flash_attn_causal_kernel<<<grid, block, smem_size, stream>>>(tma_Q, tma_K, tma_V, o_ptr, lse_ptr, B, H, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}