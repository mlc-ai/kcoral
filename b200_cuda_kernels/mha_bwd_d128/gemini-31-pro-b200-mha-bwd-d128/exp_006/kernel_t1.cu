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
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                _e, __FILE__, __LINE__);                           \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_mha_bwd {

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
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

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(addr), "r"(ncols));
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

__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen05_fence_before_fn() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_linear_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   // version = 1 (SM100)
    d |= (uint64_t)0 << 61;   // layout_type = SWIZZLE_NONE
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_f32_bf16(uint32_t M, uint32_t N, int trans_a, int trans_b) {
    uint32_t d = 0;
    d |= (1u << 4);    // dtype = FP32
    d |= (1u << 7);    // atype = BF16
    d |= (1u << 10);   // btype = BF16
    if (trans_a) d |= (1u << 15);
    if (trans_b) d |= (1u << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

CUresult create_tma_4d_descriptor(CUtensorMap* d, void* globalAddress, uint64_t S, uint64_t H, uint64_t B, uint32_t box_S) {
    cuuint64_t globalDim[4] = {128, S, H, B};
    cuuint64_t globalStrides[3] = {128*2, 128*2*S, 128*2*S*H};
    cuuint32_t boxDim[4] = {128, box_S, 1, 1};
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, globalAddress,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

__global__ void mha_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, float scale
) {
    setmaxnreg_inc_sync_fn<256>();
    
    int b = blockIdx.x;
    int h = blockIdx.y;
    int start_k = blockIdx.z * 128;
    int tid = threadIdx.x;
    int r = tid; 
    
    extern __shared__ uint8_t dynamic_smem[];
    uint8_t* smem = dynamic_smem;
    
    uint32_t smem_K = 0;           // 128x128 * 2 = 32 KB
    uint32_t smem_V = 32768;       // 128x128 * 2 = 32 KB
    uint32_t smem_Q = 65536;       // 64x128 * 2 = 16 KB
    uint32_t smem_O = 81920;       // 64x128 * 2 = 16 KB
    uint32_t smem_dO = 98304;      // 64x128 * 2 = 16 KB
    uint32_t smem_dS = 114688;     // 128x64 * 2 = 16 KB
    
    __align__(8) __shared__ uint64_t mbar_KV[1];
    __align__(8) __shared__ uint64_t mbar_Q[1];
    __align__(8) __shared__ uint64_t mbar_mma[1];
    __align__(16) __shared__ uint32_t tmem_alloc[1];
    __align__(16) __shared__ float s_D[64];
    
    if (tid == 0) {
        init_smem_barrier_fn(mbar_KV, 1);
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_mma, 1);
        tmem_alloc_cg1_fn(tmem_alloc, 512);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t tmem_base = tmem_alloc[0];
    uint32_t tmem_dK = tmem_base + 0;
    uint32_t tmem_dV = tmem_base + 128;
    uint32_t tmem_S  = tmem_base + 256;
    uint32_t tmem_dP = tmem_base + 320;
    uint32_t tmem_dQ = tmem_base + 256; 
    uint32_t tmem_P  = tmem_base + 384;
    uint32_t tmem_dS = tmem_base + 416;
    
    uint64_t desc_K  = make_smem_desc_linear_fn(smem + smem_K, 16, 2048);
    uint64_t desc_V  = make_smem_desc_linear_fn(smem + smem_V, 16, 2048);
    uint64_t desc_Q  = make_smem_desc_linear_fn(smem + smem_Q, 16, 2048);
    uint64_t desc_O  = make_smem_desc_linear_fn(smem + smem_O, 16, 2048);
    uint64_t desc_dO = make_smem_desc_linear_fn(smem + smem_dO, 16, 2048);
    uint64_t desc_dS = make_smem_desc_linear_fn(smem + smem_dS, 16, 1024);
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_KV, 32768 * 2);
        tma_load_4d_fn(&tma_K, mbar_KV, smem + smem_K, 0, start_k, h, b);
        tma_load_4d_fn(&tma_V, mbar_KV, smem + smem_V, 0, start_k, h, b);
    }
    mbarrier_wait_fn(mbar_KV, 0);
    fence_proxy_async_fn();
    __syncthreads();
    
    uint32_t phase_Q = 0;
    uint32_t phase_mma = 0;
    
    const float* l_base = L + (b * H + h) * S;
    
    for (int start_q = 0; start_q < S; start_q += 64) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_Q, 16384 * 3);
            tma_load_4d_fn(&tma_Q, mbar_Q, smem + smem_Q, 0, start_q, h, b);
            tma_load_4d_fn(&tma_O, mbar_Q, smem + smem_O, 0, start_q, h, b);
            tma_load_4d_fn(&tma_dO, mbar_Q, smem + smem_dO, 0, start_q, h, b);
        }
        mbarrier_wait_fn(mbar_Q, phase_Q);
        fence_proxy_async_fn();
        __syncthreads();
        
        uint32_t accum_K = (start_q == 0) ? 0 : 1;
        uint32_t accum_V = (start_q == 0) ? 0 : 1;
        
        uint32_t idesc_S = make_instr_desc_f32_bf16(128, 64, 0, 0);
        for (int k = 0; k < 128; k += 16) {
            uint64_t dk = desc_K + (k / 8);
            uint64_t dq = desc_Q + (k / 8);
            asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, %4;"
                         :: "r"(tmem_S), "l"(dk), "l"(dq), "r"(idesc_S), "r"(0)); 
        }
        
        uint32_t idesc_dP = make_instr_desc_f32_bf16(128, 64, 0, 0);
        for (int k = 0; k < 128; k += 16) {
            uint64_t dv = desc_V + (k / 8);
            uint64_t ddo = desc_dO + (k / 8);
            asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, %4;"
                         :: "r"(tmem_dP), "l"(dv), "l"(ddo), "r"(idesc_dP), "r"(0)); 
        }
        
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                     :: "r"((uint32_t)__cvta_generic_to_shared(&mbar_mma[0])));
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;
        
        float d_val = 0.0f;
        if (tid < 64) {
            for (int c = 0; c < 128; c += 2) {
                uint32_t o_addr = smem_O + tid * 256 + c * 2;
                uint32_t do_addr = smem_dO + tid * 256 + c * 2;
                uint32_t o_val = *(uint32_t*)(smem + o_addr);
                uint32_t do_val = *(uint32_t*)(smem + do_addr);
                float2 o2 = __bfloat1622float2(*(__nv_bfloat162*)&o_val);
                float2 do2 = __bfloat1622float2(*(__nv_bfloat162*)&do_val);
                d_val += o2.x * do2.x + o2.y * do2.y;
            }
            s_D[tid] = d_val;
        }
        __syncthreads();
        
        const float* l_ptr = l_base + start_q;
        for (int c = 0; c < 64; c += 8) {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) 
                         : "r"(tmem_S + c));
            uint32_t dp0, dp1, dp2, dp3, dp4, dp5, dp6, dp7;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                         : "=r"(dp0),"=r"(dp1),"=r"(dp2),"=r"(dp3),"=r"(dp4),"=r"(dp5),"=r"(dp6),"=r"(dp7) 
                         : "r"(tmem_dP + c));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float ds[8];
            uint32_t p_packed[4];
            uint32_t ds_packed[4];
            
            float s_arr[8] = {
                __uint_as_float(r0), __uint_as_float(r1), __uint_as_float(r2), __uint_as_float(r3),
                __uint_as_float(r4), __uint_as_float(r5), __uint_as_float(r6), __uint_as_float(r7)
            };
            float dp_arr[8] = {
                __uint_as_float(dp0), __uint_as_float(dp1), __uint_as_float(dp2), __uint_as_float(dp3),
                __uint_as_float(dp4), __uint_as_float(dp5), __uint_as_float(dp6), __uint_as_float(dp7)
            };
            
            for (int i = 0; i < 8; ++i) {
                float l_val = l_ptr[c + i];
                float p = expf(s_arr[i] * scale - l_val);
                float D_v = s_D[c + i];
                ds[i] = p * (dp_arr[i] - D_v);
                s_arr[i] = p; 
            }
            
            p_packed[0] = pack_bf16(s_arr[0], s_arr[1]);
            p_packed[1] = pack_bf16(s_arr[2], s_arr[3]);
            p_packed[2] = pack_bf16(s_arr[4], s_arr[5]);
            p_packed[3] = pack_bf16(s_arr[6], s_arr[7]);
            
            ds_packed[0] = pack_bf16(ds[0], ds[1]);
            ds_packed[1] = pack_bf16(ds[2], ds[3]);
            ds_packed[2] = pack_bf16(ds[4], ds[5]);
            ds_packed[3] = pack_bf16(ds[6], ds[7]);
            
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                         :: "r"(p_packed[0]),"r"(p_packed[1]),"r"(p_packed[2]),"r"(p_packed[3]), "r"(tmem_P + c/2));
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                         :: "r"(ds_packed[0]),"r"(ds_packed[1]),"r"(ds_packed[2]),"r"(ds_packed[3]), "r"(tmem_dS + c/2));
            
            uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem + smem_dS) + r * 128 + c * 2;
            st_shared_128_fn(smem_addr, ds_packed[0], ds_packed[1], ds_packed[2], ds_packed[3]);
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        tcgen05_fence_before_fn();
        
        uint32_t idesc_dV = make_instr_desc_f32_bf16(128, 128, 0, 1);
        for (int k = 0; k < 64; k += 16) {
            uint32_t tp = tmem_P + (k / 2);
            uint64_t ddo = desc_dO + (k / 16) * 256;
            asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, %4;"
                         :: "r"(tmem_dV), "r"(tp), "l"(ddo), "r"(idesc_dV), "r"(accum_V));
        }
        
        uint32_t idesc_dK = make_instr_desc_f32_bf16(128, 128, 0, 1);
        for (int k = 0; k < 64; k += 16) {
            uint32_t tds = tmem_dS + (k / 2);
            uint64_t dq = desc_Q + (k / 16) * 256;
            asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, %4;"
                         :: "r"(tmem_dK), "r"(tds), "l"(dq), "r"(idesc_dK), "r"(accum_K));
        }
        
        uint32_t idesc_dQ = make_instr_desc_f32_bf16(64, 128, 1, 1);
        for (int k = 0; k < 128; k += 16) {
            uint64_t dds = desc_dS + (k / 16) * 128;
            uint64_t dk = desc_K + (k / 16) * 256;
            asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, %4;"
                         :: "r"(tmem_dQ), "l"(dds), "l"(dk), "r"(idesc_dQ), "r"(0));
        }
        
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                     :: "r"((uint32_t)__cvta_generic_to_shared(&mbar_mma[0])));
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;
        
        long long q_base = ((long long)b * H + h) * S * 128 + start_q * 128;
        if (r < 64) {
            for (int c = 0; c < 128; c += 8) {
                uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                             : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) 
                             : "r"(tmem_dQ + c));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                __nv_bfloat162 b01 = __floats2bfloat162_rn(__uint_as_float(r0), __uint_as_float(r1));
                __nv_bfloat162 b23 = __floats2bfloat162_rn(__uint_as_float(r2), __uint_as_float(r3));
                __nv_bfloat162 b45 = __floats2bfloat162_rn(__uint_as_float(r4), __uint_as_float(r5));
                __nv_bfloat162 b67 = __floats2bfloat162_rn(__uint_as_float(r6), __uint_as_float(r7));
                
                __nv_bfloat162* out_ptr = (__nv_bfloat162*)(dQ + q_base + r * 128 + c);
                atomicAdd(out_ptr + 0, b01);
                atomicAdd(out_ptr + 1, b23);
                atomicAdd(out_ptr + 2, b45);
                atomicAdd(out_ptr + 3, b67);
            }
        }
        __syncthreads();
        phase_Q ^= 1;
    }
    
    long long k_base = ((long long)b * H + h) * S * 128 + start_k * 128;
    for (int c = 0; c < 128; c += 8) {
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) 
                     : "r"(tmem_dK + c));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        __nv_bfloat162 b01 = __floats2bfloat162_rn(__uint_as_float(r0), __uint_as_float(r1));
        __nv_bfloat162 b23 = __floats2bfloat162_rn(__uint_as_float(r2), __uint_as_float(r3));
        __nv_bfloat162 b45 = __floats2bfloat162_rn(__uint_as_float(r4), __uint_as_float(r5));
        __nv_bfloat162 b67 = __floats2bfloat162_rn(__uint_as_float(r6), __uint_as_float(r7));
        
        __nv_bfloat162* out_ptr = (__nv_bfloat162*)(dK + k_base + r * 128 + c);
        out_ptr[0] = b01;
        out_ptr[1] = b23;
        out_ptr[2] = b45;
        out_ptr[3] = b67;
    }
    
    for (int c = 0; c < 128; c += 8) {
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) 
                     : "r"(tmem_dV + c));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        __nv_bfloat162 b01 = __floats2bfloat162_rn(__uint_as_float(r0), __uint_as_float(r1));
        __nv_bfloat162 b23 = __floats2bfloat162_rn(__uint_as_float(r2), __uint_as_float(r3));
        __nv_bfloat162 b45 = __floats2bfloat162_rn(__uint_as_float(r4), __uint_as_float(r5));
        __nv_bfloat162 b67 = __floats2bfloat162_rn(__uint_as_float(r6), __uint_as_float(r7));
        
        __nv_bfloat162* out_ptr = (__nv_bfloat162*)(dV + k_base + r * 128 + c);
        out_ptr[0] = b01;
        out_ptr[1] = b23;
        out_ptr[2] = b45;
        out_ptr[3] = b67;
    }
    
    if (tid == 0) {
        tmem_dealloc_cg1_fn(tmem_base, 512);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    float scale = 1.0f / sqrtf(128.0f);
    
    CUDA_CHECK(cudaMemsetAsync(dQ.data_ptr(), 0, B * H * S * 128 * sizeof(__nv_bfloat16), stream));
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    CU_CHECK(create_tma_4d_descriptor(&tma_Q, (void*)Q.data_ptr(), S, H, B, 64));
    CU_CHECK(create_tma_4d_descriptor(&tma_K, (void*)K.data_ptr(), S, H, B, 128));
    CU_CHECK(create_tma_4d_descriptor(&tma_V, (void*)V.data_ptr(), S, H, B, 128));
    CU_CHECK(create_tma_4d_descriptor(&tma_O, (void*)O.data_ptr(), S, H, B, 64));
    CU_CHECK(create_tma_4d_descriptor(&tma_dO, (void*)dO.data_ptr(), S, H, B, 64));
    
    dim3 grid(B, H, (S + 127) / 128);
    dim3 block(128);
    int smem_size = 131072;
    
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    mha_bwd_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_K, tma_V, tma_O, tma_dO,
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        B, H, S, scale
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_mha_bwd