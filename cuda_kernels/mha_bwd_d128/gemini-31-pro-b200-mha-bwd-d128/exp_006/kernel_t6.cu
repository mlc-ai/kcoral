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

__device__ __forceinline__ void tma_load_4d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2, int32_t c3) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5, %6}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2), "r"(c3) : "memory");
}

__device__ __forceinline__ uint64_t make_smem_desc_K_major(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t lbo = 1; 
    uint32_t sbo = 1024;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint64_t make_smem_desc_MN_major(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint32_t sbo = 1024;
    uint32_t lbo = 16384; // (128/8) * 1024
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_f32_bf16(uint32_t M, uint32_t N, int trans_a, int trans_b) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    if (trans_a) d |= (1u << 15);
    if (trans_b) d |= (1u << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void umma_smem_smem(uint32_t tmem_d, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_tmem_smem(uint32_t tmem_d, uint32_t tmem_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n}\n"
        :: "r"(tmem_d), "r"(tmem_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ uint32_t read_swizzled_128B(const uint8_t* smem_base, int row, int col) {
    int byte_idx = col * 2;
    int chunk_x = byte_idx / 16;
    int chunk_off = byte_idx % 16;
    int swizzled_chunk_x = (row % 8) ^ chunk_x;
    int swizzled_byte_idx = swizzled_chunk_x * 16 + chunk_off;
    return *(uint32_t*)(smem_base + row * 128 + swizzled_byte_idx);
}

CUresult create_tma_4d_descriptor_128B(CUtensorMap* d, void* globalAddress, uint64_t S, uint64_t H, uint64_t B) {
    cuuint64_t globalDim[4] = {128, S, H, B};
    cuuint64_t globalStrides[3] = {128*2, 128*2*S, 128*2*S*H};
    cuuint32_t boxDim[4] = {64, 128, 1, 1}; 
    cuuint32_t elementStrides[4] = {1, 1, 1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 4, globalAddress,
        globalDim, globalStrides, boxDim, elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

__global__ __launch_bounds__(128) void pass1_dQ_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    int B, int H, int S, float scale
) {
    setmaxnreg_inc_sync_fn<256>();
    
    int b = blockIdx.x;
    int h = blockIdx.y;
    int start_q = blockIdx.z * 128;
    int tid = threadIdx.x;
    
    if (start_q >= S) return;
    
    extern __shared__ uint8_t dynamic_smem[];
    uintptr_t smem_addr = (uintptr_t)dynamic_smem;
    smem_addr = (smem_addr + 127) & ~127;
    uint8_t* smem = (uint8_t*)smem_addr;
    
    uint32_t smem_Q = 0;       
    uint32_t smem_O = 32768;   
    uint32_t smem_dO = 65536;  
    uint32_t smem_K = 98304;   
    uint32_t smem_V = 131072;  
    
    __align__(8) __shared__ uint64_t mbar_Q[1];
    __align__(8) __shared__ uint64_t mbar_KV[1];
    __align__(8) __shared__ uint64_t mbar_mma[1];
    __align__(16) __shared__ uint32_t tmem_alloc[1];
    __align__(16) __shared__ float s_D[128];
    
    if (tid == 0) {
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_KV, 1);
        init_smem_barrier_fn(mbar_mma, 1);
    }
    if (tid < 32) {
        tmem_alloc_cg1_fn(tmem_alloc, 320);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t tmem_base = tmem_alloc[0];
    uint32_t tmem_dQ = tmem_base + 0;   
    uint32_t tmem_S  = tmem_base + 64; 
    uint32_t tmem_dP = tmem_base + 128; 
    uint32_t tmem_dS = tmem_base + 192; 
    uint32_t tmem_P  = tmem_base + 256; 
    
    uint64_t desc_Q_0  = make_smem_desc_K_major(smem + smem_Q);
    uint64_t desc_Q_1  = make_smem_desc_K_major(smem + smem_Q + 16384);
    
    uint64_t desc_K_0  = make_smem_desc_K_major(smem + smem_K);
    uint64_t desc_K_1  = make_smem_desc_K_major(smem + smem_K + 16384);
    uint64_t desc_V_0  = make_smem_desc_K_major(smem + smem_V);
    uint64_t desc_V_1  = make_smem_desc_K_major(smem + smem_V + 16384);
    
    uint64_t desc_dO_0 = make_smem_desc_K_major(smem + smem_dO);
    uint64_t desc_dO_1 = make_smem_desc_K_major(smem + smem_dO + 16384);
    
    uint64_t desc_K_0_MN = make_smem_desc_MN_major(smem + smem_K);
    uint64_t desc_K_1_MN = make_smem_desc_MN_major(smem + smem_K + 16384);
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_Q, 98304);
        tma_load_4d_fn(&tma_Q, mbar_Q, smem + smem_Q, 0, start_q, h, b);
        tma_load_4d_fn(&tma_Q, mbar_Q, smem + smem_Q + 16384, 64, start_q, h, b);
        tma_load_4d_fn(&tma_O, mbar_Q, smem + smem_O, 0, start_q, h, b);
        tma_load_4d_fn(&tma_O, mbar_Q, smem + smem_O + 16384, 64, start_q, h, b);
        tma_load_4d_fn(&tma_dO, mbar_Q, smem + smem_dO, 0, start_q, h, b);
        tma_load_4d_fn(&tma_dO, mbar_Q, smem + smem_dO + 16384, 64, start_q, h, b);
    }
    mbarrier_wait_fn(mbar_Q, 0);
    fence_proxy_async_fn();
    __syncthreads();
    
    float d_val = 0.0f;
    for (int c = 0; c < 64; c += 2) {
        uint32_t o_val = read_swizzled_128B(smem + smem_O, tid, c);
        uint32_t do_val = read_swizzled_128B(smem + smem_dO, tid, c);
        float2 o2 = __bfloat1622float2(*(__nv_bfloat162*)&o_val);
        float2 do2 = __bfloat1622float2(*(__nv_bfloat162*)&do_val);
        d_val += o2.x * do2.x + o2.y * do2.y;
        
        o_val = read_swizzled_128B(smem + smem_O + 16384, tid, c);
        do_val = read_swizzled_128B(smem + smem_dO + 16384, tid, c);
        o2 = __bfloat1622float2(*(__nv_bfloat162*)&o_val);
        do2 = __bfloat1622float2(*(__nv_bfloat162*)&do_val);
        d_val += o2.x * do2.x + o2.y * do2.y;
    }
    s_D[tid] = d_val;
    __syncthreads();
    
    uint32_t phase_KV = 0;
    uint32_t phase_mma = 0;
    const float* l_ptr = L + (b * H + h) * S + start_q;
    
    uint32_t idesc_S = make_instr_desc_f32_bf16(128, 128, 0, 0);
    uint32_t idesc_dP = make_instr_desc_f32_bf16(128, 128, 0, 0);
    uint32_t idesc_dQ = make_instr_desc_f32_bf16(128, 64, 0, 1); // trans_b = 1
    
    for (int start_k = 0; start_k < S; start_k += 128) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_KV, 65536);
            tma_load_4d_fn(&tma_K, mbar_KV, smem + smem_K, 0, start_k, h, b);
            tma_load_4d_fn(&tma_K, mbar_KV, smem + smem_K + 16384, 64, start_k, h, b);
            tma_load_4d_fn(&tma_V, mbar_KV, smem + smem_V, 0, start_k, h, b);
            tma_load_4d_fn(&tma_V, mbar_KV, smem + smem_V + 16384, 64, start_k, h, b);
        }
        mbarrier_wait_fn(mbar_KV, phase_KV);
        fence_proxy_async_fn();
        __syncthreads();
        
        if (tid == 0) {
            for (int k = 0; k < 64; k += 16) {
                uint64_t da = desc_Q_0 + (k / 8);
                uint64_t db = desc_K_0 + (k / 8);
                uint32_t accum = (k == 0) ? 0 : 1;
                umma_smem_smem(tmem_S, da, db, idesc_S, accum);
            }
            for (int k = 0; k < 64; k += 16) {
                uint64_t da = desc_Q_1 + (k / 8);
                uint64_t db = desc_K_1 + (k / 8);
                umma_smem_smem(tmem_S, da, db, idesc_S, 1);
            }
            
            for (int k = 0; k < 64; k += 16) {
                uint64_t da = desc_V_0 + (k / 8);
                uint64_t db = desc_dO_0 + (k / 8);
                uint32_t accum = (k == 0) ? 0 : 1;
                umma_smem_smem(tmem_dP, da, db, idesc_dP, accum);
            }
            for (int k = 0; k < 64; k += 16) {
                uint64_t da = desc_V_1 + (k / 8);
                uint64_t db = desc_dO_1 + (k / 8);
                umma_smem_smem(tmem_dP, da, db, idesc_dP, 1);
            }
            
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                         :: "r"((uint32_t)__cvta_generic_to_shared(&mbar_mma[0])));
        }
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;
        
        float l_val = l_ptr[tid];
        float d_v = s_D[tid];
        
        for (int c = 0; c < 128; c += 8) {
            uint32_t r_S[8], r_dP[8];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                         : "=r"(r_S[0]), "=r"(r_S[1]), "=r"(r_S[2]), "=r"(r_S[3]),
                           "=r"(r_S[4]), "=r"(r_S[5]), "=r"(r_S[6]), "=r"(r_S[7]) 
                         : "r"(tmem_S + c/2)); 
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                         : "=r"(r_dP[0]), "=r"(r_dP[1]), "=r"(r_dP[2]), "=r"(r_dP[3]),
                           "=r"(r_dP[4]), "=r"(r_dP[5]), "=r"(r_dP[6]), "=r"(r_dP[7]) 
                         : "r"(tmem_dP + c/2));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            uint32_t r_dS[4];
            for (int i = 0; i < 8; i+=2) {
                float s0 = __uint_as_float(r_S[i]);
                float s1 = __uint_as_float(r_S[i+1]);
                float dp0 = __uint_as_float(r_dP[i]);
                float dp1 = __uint_as_float(r_dP[i+1]);
                
                float p0 = expf(s0 * scale - l_val);
                float p1 = expf(s1 * scale - l_val);
                
                float ds0 = p0 * (dp0 - d_v);
                float ds1 = p1 * (dp1 - d_v);
                
                r_dS[i/2] = pack_bf16(ds0, ds1);
            }
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                         :: "r"(r_dS[0]),"r"(r_dS[1]),"r"(r_dS[2]),"r"(r_dS[3]), "r"(tmem_dS + c/2));
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        tcgen05_fence_before_fn();
        
        if (tid == 0) {
            for (int k = 0; k < 128; k += 16) {
                uint32_t ta = tmem_dS + (k / 2);
                uint64_t db = desc_K_0_MN + (k / 16) * 128; 
                uint32_t accum = (start_k == 0 && k == 0) ? 0 : 1;
                umma_tmem_smem(tmem_dQ, ta, db, idesc_dQ, accum);
            }
            for (int k = 0; k < 128; k += 16) {
                uint32_t ta = tmem_dS + (k / 2);
                uint64_t db = desc_K_1_MN + (k / 16) * 128;
                umma_tmem_smem(tmem_dQ + 32, ta, db, idesc_dQ, (start_k == 0 && k == 0) ? 0 : 1);
            }
            
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                         :: "r"((uint32_t)__cvta_generic_to_shared(&mbar_mma[0])));
        }
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;
        
        phase_KV ^= 1;
    }
    
    long long q_base = ((long long)b * H + h) * S * 128 + start_q * 128;
    for (int c = 0; c < 128; c += 8) {
        uint32_t r_dQ[8];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r_dQ[0]), "=r"(r_dQ[1]), "=r"(r_dQ[2]), "=r"(r_dQ[3]),
                       "=r"(r_dQ[4]), "=r"(r_dQ[5]), "=r"(r_dQ[6]), "=r"(r_dQ[7]) 
                     : "r"(tmem_dQ + c/2));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        __nv_bfloat162* out_ptr = (__nv_bfloat162*)(dQ + q_base + tid * 128 + c);
        out_ptr[0] = __floats2bfloat162_rn(__uint_as_float(r_dQ[0]), __uint_as_float(r_dQ[1]));
        out_ptr[1] = __floats2bfloat162_rn(__uint_as_float(r_dQ[2]), __uint_as_float(r_dQ[3]));
        out_ptr[2] = __floats2bfloat162_rn(__uint_as_float(r_dQ[4]), __uint_as_float(r_dQ[5]));
        out_ptr[3] = __floats2bfloat162_rn(__uint_as_float(r_dQ[6]), __uint_as_float(r_dQ[7]));
    }
    
    if (tid < 32) {
        tmem_dealloc_cg1_fn(tmem_base, 320);
    }
}

__global__ __launch_bounds__(128) void pass2_dK_dV_kernel(
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    int B, int H, int S, float scale
) {
    setmaxnreg_inc_sync_fn<256>();
    
    int b = blockIdx.x;
    int h = blockIdx.y;
    int start_k = blockIdx.z * 128;
    int tid = threadIdx.x;
    
    if (start_k >= S) return;
    
    extern __shared__ uint8_t dynamic_smem[];
    uintptr_t smem_addr = (uintptr_t)dynamic_smem;
    smem_addr = (smem_addr + 127) & ~127;
    uint8_t* smem = (uint8_t*)smem_addr;
    
    uint32_t smem_K = 0;       
    uint32_t smem_V = 32768;   
    uint32_t smem_Q = 65536;   
    uint32_t smem_O = 98304;   
    uint32_t smem_dO = 131072;  
    
    __align__(8) __shared__ uint64_t mbar_KV[1];
    __align__(8) __shared__ uint64_t mbar_Q[1];
    __align__(8) __shared__ uint64_t mbar_mma[1];
    __align__(16) __shared__ uint32_t tmem_alloc[1];
    __align__(16) __shared__ float s_D[128];
    
    if (tid == 0) {
        init_smem_barrier_fn(mbar_KV, 1);
        init_smem_barrier_fn(mbar_Q, 1);
        init_smem_barrier_fn(mbar_mma, 1);
    }
    if (tid < 32) {
        tmem_alloc_cg1_fn(tmem_alloc, 384);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t tmem_base = tmem_alloc[0];
    uint32_t tmem_dK = tmem_base + 0;   
    uint32_t tmem_dV = tmem_base + 64; 
    uint32_t tmem_S  = tmem_base + 128; 
    uint32_t tmem_dP = tmem_base + 192; 
    uint32_t tmem_dS = tmem_base + 256; 
    uint32_t tmem_P  = tmem_base + 320; 
    
    uint64_t desc_K_0  = make_smem_desc_K_major(smem + smem_K);
    uint64_t desc_K_1  = make_smem_desc_K_major(smem + smem_K + 16384);
    uint64_t desc_V_0  = make_smem_desc_K_major(smem + smem_V);
    uint64_t desc_V_1  = make_smem_desc_K_major(smem + smem_V + 16384);
    
    uint64_t desc_Q_0  = make_smem_desc_K_major(smem + smem_Q);
    uint64_t desc_Q_1  = make_smem_desc_K_major(smem + smem_Q + 16384);
    uint64_t desc_dO_0 = make_smem_desc_K_major(smem + smem_dO);
    uint64_t desc_dO_1 = make_smem_desc_K_major(smem + smem_dO + 16384);
    
    uint64_t desc_Q_0_MN = make_smem_desc_MN_major(smem + smem_Q);
    uint64_t desc_Q_1_MN = make_smem_desc_MN_major(smem + smem_Q + 16384);
    uint64_t desc_dO_0_MN = make_smem_desc_MN_major(smem + smem_dO);
    uint64_t desc_dO_1_MN = make_smem_desc_MN_major(smem + smem_dO + 16384);
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(mbar_KV, 65536);
        tma_load_4d_fn(&tma_K, mbar_KV, smem + smem_K, 0, start_k, h, b);
        tma_load_4d_fn(&tma_K, mbar_KV, smem + smem_K + 16384, 64, start_k, h, b);
        tma_load_4d_fn(&tma_V, mbar_KV, smem + smem_V, 0, start_k, h, b);
        tma_load_4d_fn(&tma_V, mbar_KV, smem + smem_V + 16384, 64, start_k, h, b);
    }
    mbarrier_wait_fn(mbar_KV, 0);
    fence_proxy_async_fn();
    __syncthreads();
    
    uint32_t phase_Q = 0;
    uint32_t phase_mma = 0;
    const float* l_base = L + (b * H + h) * S;
    
    uint32_t idesc_S = make_instr_desc_f32_bf16(128, 128, 0, 0);
    uint32_t idesc_dP = make_instr_desc_f32_bf16(128, 128, 0, 0);
    uint32_t idesc_dK = make_instr_desc_f32_bf16(128, 64, 0, 1);
    uint32_t idesc_dV = make_instr_desc_f32_bf16(128, 64, 0, 1);
    
    for (int start_q = 0; start_q < S; start_q += 128) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(mbar_Q, 98304);
            tma_load_4d_fn(&tma_Q, mbar_Q, smem + smem_Q, 0, start_q, h, b);
            tma_load_4d_fn(&tma_Q, mbar_Q, smem + smem_Q + 16384, 64, start_q, h, b);
            tma_load_4d_fn(&tma_O, mbar_Q, smem + smem_O, 0, start_q, h, b);
            tma_load_4d_fn(&tma_O, mbar_Q, smem + smem_O + 16384, 64, start_q, h, b);
            tma_load_4d_fn(&tma_dO, mbar_Q, smem + smem_dO, 0, start_q, h, b);
            tma_load_4d_fn(&tma_dO, mbar_Q, smem + smem_dO + 16384, 64, start_q, h, b);
        }
        mbarrier_wait_fn(mbar_Q, phase_Q);
        fence_proxy_async_fn();
        __syncthreads();
        
        float d_val = 0.0f;
        for (int c = 0; c < 64; c += 2) {
            uint32_t o_val = read_swizzled_128B(smem + smem_O, tid, c);
            uint32_t do_val = read_swizzled_128B(smem + smem_dO, tid, c);
            float2 o2 = __bfloat1622float2(*(__nv_bfloat162*)&o_val);
            float2 do2 = __bfloat1622float2(*(__nv_bfloat162*)&do_val);
            d_val += o2.x * do2.x + o2.y * do2.y;
            
            o_val = read_swizzled_128B(smem + smem_O + 16384, tid, c);
            do_val = read_swizzled_128B(smem + smem_dO + 16384, tid, c);
            o2 = __bfloat1622float2(*(__nv_bfloat162*)&o_val);
            do2 = __bfloat1622float2(*(__nv_bfloat162*)&do_val);
            d_val += o2.x * do2.x + o2.y * do2.y;
        }
        s_D[tid] = d_val;
        __syncthreads();
        
        if (tid == 0) {
            for (int k = 0; k < 64; k += 16) {
                uint64_t da = desc_K_0 + (k / 8);
                uint64_t db = desc_Q_0 + (k / 8);
                uint32_t accum = (k == 0) ? 0 : 1;
                umma_smem_smem(tmem_S, da, db, idesc_S, accum);
            }
            for (int k = 0; k < 64; k += 16) {
                uint64_t da = desc_K_1 + (k / 8);
                uint64_t db = desc_Q_1 + (k / 8);
                umma_smem_smem(tmem_S, da, db, idesc_S, 1);
            }
            
            for (int k = 0; k < 64; k += 16) {
                uint64_t da = desc_V_0 + (k / 8);
                uint64_t db = desc_dO_0 + (k / 8);
                uint32_t accum = (k == 0) ? 0 : 1;
                umma_smem_smem(tmem_dP, da, db, idesc_dP, accum);
            }
            for (int k = 0; k < 64; k += 16) {
                uint64_t da = desc_V_1 + (k / 8);
                uint64_t db = desc_dO_1 + (k / 8);
                umma_smem_smem(tmem_dP, da, db, idesc_dP, 1);
            }
            
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                         :: "r"((uint32_t)__cvta_generic_to_shared(&mbar_mma[0])));
        }
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;
        
        const float* l_ptr = l_base + start_q;
        for (int c = 0; c < 128; c += 8) {
            uint32_t r_S[8], r_dP[8];
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                         : "=r"(r_S[0]), "=r"(r_S[1]), "=r"(r_S[2]), "=r"(r_S[3]),
                           "=r"(r_S[4]), "=r"(r_S[5]), "=r"(r_S[6]), "=r"(r_S[7]) 
                         : "r"(tmem_S + c/2));
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                         : "=r"(r_dP[0]), "=r"(r_dP[1]), "=r"(r_dP[2]), "=r"(r_dP[3]),
                           "=r"(r_dP[4]), "=r"(r_dP[5]), "=r"(r_dP[6]), "=r"(r_dP[7]) 
                         : "r"(tmem_dP + c/2));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float l_v[8], d_v[8];
            for (int i=0; i<8; ++i) {
                l_v[i] = l_ptr[c + i];
                d_v[i] = s_D[c + i];
            }
            
            uint32_t r_dS[4], r_P[4];
            for (int i = 0; i < 8; i+=2) {
                float s0 = __uint_as_float(r_S[i]);
                float s1 = __uint_as_float(r_S[i+1]);
                float dp0 = __uint_as_float(r_dP[i]);
                float dp1 = __uint_as_float(r_dP[i+1]);
                
                float p0 = expf(s0 * scale - l_v[i]);
                float p1 = expf(s1 * scale - l_v[i+1]);
                
                float ds0 = p0 * (dp0 - d_v[i]);
                float ds1 = p1 * (dp1 - d_v[i+1]);
                
                r_dS[i/2] = pack_bf16(ds0, ds1);
                r_P[i/2]  = pack_bf16(p0, p1);
            }
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                         :: "r"(r_dS[0]),"r"(r_dS[1]),"r"(r_dS[2]),"r"(r_dS[3]), "r"(tmem_dS + c/2));
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 [%4], {%0,%1,%2,%3};"
                         :: "r"(r_P[0]),"r"(r_P[1]),"r"(r_P[2]),"r"(r_P[3]), "r"(tmem_P + c/2));
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        tcgen05_fence_before_fn();
        
        if (tid == 0) {
            for (int k = 0; k < 128; k += 16) {
                uint32_t ta = tmem_dS + (k / 2);
                uint64_t db = desc_Q_0_MN + (k / 16) * 128;
                uint32_t accum = (start_q == 0 && k == 0) ? 0 : 1;
                umma_tmem_smem(tmem_dK, ta, db, idesc_dK, accum);
            }
            for (int k = 0; k < 128; k += 16) {
                uint32_t ta = tmem_dS + (k / 2);
                uint64_t db = desc_Q_1_MN + (k / 16) * 128;
                umma_tmem_smem(tmem_dK + 32, ta, db, idesc_dK, (start_q == 0 && k == 0) ? 0 : 1);
            }
            
            for (int k = 0; k < 128; k += 16) {
                uint32_t ta = tmem_P + (k / 2);
                uint64_t db = desc_dO_0_MN + (k / 16) * 128;
                uint32_t accum = (start_q == 0 && k == 0) ? 0 : 1;
                umma_tmem_smem(tmem_dV, ta, db, idesc_dV, accum);
            }
            for (int k = 0; k < 128; k += 16) {
                uint32_t ta = tmem_P + (k / 2);
                uint64_t db = desc_dO_1_MN + (k / 16) * 128;
                umma_tmem_smem(tmem_dV + 32, ta, db, idesc_dV, (start_q == 0 && k == 0) ? 0 : 1);
            }
            
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                         :: "r"((uint32_t)__cvta_generic_to_shared(&mbar_mma[0])));
        }
        mbarrier_wait_fn(mbar_mma, phase_mma);
        phase_mma ^= 1;
        
        phase_Q ^= 1;
    }
    
    long long k_base = ((long long)b * H + h) * S * 128 + start_k * 128;
    for (int c = 0; c < 128; c += 8) {
        uint32_t r_dK[8], r_dV[8];
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r_dK[0]), "=r"(r_dK[1]), "=r"(r_dK[2]), "=r"(r_dK[3]),
                       "=r"(r_dK[4]), "=r"(r_dK[5]), "=r"(r_dK[6]), "=r"(r_dK[7]) 
                     : "r"(tmem_dK + c/2));
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                     : "=r"(r_dV[0]), "=r"(r_dV[1]), "=r"(r_dV[2]), "=r"(r_dV[3]),
                       "=r"(r_dV[4]), "=r"(r_dV[5]), "=r"(r_dV[6]), "=r"(r_dV[7]) 
                     : "r"(tmem_dV + c/2));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        
        __nv_bfloat162* outK = (__nv_bfloat162*)(dK + k_base + tid * 128 + c);
        outK[0] = __floats2bfloat162_rn(__uint_as_float(r_dK[0]), __uint_as_float(r_dK[1]));
        outK[1] = __floats2bfloat162_rn(__uint_as_float(r_dK[2]), __uint_as_float(r_dK[3]));
        outK[2] = __floats2bfloat162_rn(__uint_as_float(r_dK[4]), __uint_as_float(r_dK[5]));
        outK[3] = __floats2bfloat162_rn(__uint_as_float(r_dK[6]), __uint_as_float(r_dK[7]));
        
        __nv_bfloat162* outV = (__nv_bfloat162*)(dV + k_base + tid * 128 + c);
        outV[0] = __floats2bfloat162_rn(__uint_as_float(r_dV[0]), __uint_as_float(r_dV[1]));
        outV[1] = __floats2bfloat162_rn(__uint_as_float(r_dV[2]), __uint_as_float(r_dV[3]));
        outV[2] = __floats2bfloat162_rn(__uint_as_float(r_dV[4]), __uint_as_float(r_dV[5]));
        outV[3] = __floats2bfloat162_rn(__uint_as_float(r_dV[6]), __uint_as_float(r_dV[7]));
    }
    
    if (tid < 32) {
        tmem_dealloc_cg1_fn(tmem_base, 384);
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
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O, tma_dO;
    CU_CHECK(create_tma_4d_descriptor_128B(&tma_Q, (void*)Q.data_ptr(), S, H, B));
    CU_CHECK(create_tma_4d_descriptor_128B(&tma_O, (void*)O.data_ptr(), S, H, B));
    CU_CHECK(create_tma_4d_descriptor_128B(&tma_dO, (void*)dO.data_ptr(), S, H, B));
    CU_CHECK(create_tma_4d_descriptor_128B(&tma_K, (void*)K.data_ptr(), S, H, B));
    CU_CHECK(create_tma_4d_descriptor_128B(&tma_V, (void*)V.data_ptr(), S, H, B));
    
    CUtensorMap tma_Q_p2, tma_K_p2, tma_V_p2, tma_O_p2, tma_dO_p2;
    CU_CHECK(create_tma_4d_descriptor_128B(&tma_K_p2, (void*)K.data_ptr(), S, H, B));
    CU_CHECK(create_tma_4d_descriptor_128B(&tma_V_p2, (void*)V.data_ptr(), S, H, B));
    CU_CHECK(create_tma_4d_descriptor_128B(&tma_Q_p2, (void*)Q.data_ptr(), S, H, B));
    CU_CHECK(create_tma_4d_descriptor_128B(&tma_O_p2, (void*)O.data_ptr(), S, H, B));
    CU_CHECK(create_tma_4d_descriptor_128B(&tma_dO_p2, (void*)dO.data_ptr(), S, H, B));
    
    dim3 grid1(B, H, (S + 127) / 128);
    dim3 block(128);
    int smem_size = 163840 + 128;
    
    CUDA_CHECK(cudaFuncSetAttribute(pass1_dQ_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    CUDA_CHECK(cudaFuncSetAttribute(pass2_dK_dV_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    pass1_dQ_kernel<<<grid1, block, smem_size, stream>>>(
        tma_Q, tma_O, tma_dO, tma_K, tma_V,
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        B, H, S, scale
    );
    CUDA_CHECK(cudaGetLastError());
    
    pass2_dK_dV_kernel<<<grid1, block, smem_size, stream>>>(
        tma_K_p2, tma_V_p2, tma_Q_p2, tma_O_p2, tma_dO_p2,
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        B, H, S, scale
    );
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_mha_bwd