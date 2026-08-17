#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
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

CUresult create_tma_2d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, l2Promotion, oobFill);
}

template <int BM, int BN>
__global__ void flash_attn_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_dO,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    const float* L, int32_t S, float attn_scale)
{
    extern __shared__ __align__(128) uint8_t smem_pool[];
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)smem_pool;
    __nv_bfloat16* smem_K = (__nv_bfloat16*)(smem_pool + 32768);
    __nv_bfloat16* smem_V = (__nv_bfloat16*)(smem_pool + 65536);
    __nv_bfloat16* smem_dO = (__nv_bfloat16*)(smem_pool + 98304);
    __nv_bfloat16* smem_O = (__nv_bfloat16*)(smem_pool + 131072);
    __nv_bfloat16* smem_P = (__nv_bfloat16*)(smem_pool + 163840);
    __nv_bfloat16* smem_dS = (__nv_bfloat16*)(smem_pool + 196608);
    float* smem_D = (float*)(smem_pool + 229376);
    uint64_t* mbar = (uint64_t*)(smem_pool + 230400);

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;"
            :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])), "r"(1));
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    uint32_t c_S[128], c_dP[128], c_dV[128], c_dK[128], c_dQ[128];
    if (threadIdx.x == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(c_S)), "r"(128));
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(c_dP)), "r"(128));
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(c_dV)), "r"(128));
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(c_dK)), "r"(128));
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(c_dQ)), "r"(128));
    }
    __syncthreads();

    uint32_t kv_tile = blockIdx.x;
    uint32_t head_idx = blockIdx.y;
    uint32_t kv_off = kv_tile * 128;
    uint32_t h_S_offset = head_idx * S;

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
            :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])), "r"(2 * 32768));
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
            :: "r"((uint32_t)__cvta_generic_to_shared(smem_K)), "l"((uint64_t)&tma_K),
            "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])), "r"(0), "r"(h_S_offset + kv_off) : "memory");
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
            :: "r"((uint32_t)__cvta_generic_to_shared(smem_V)), "l"((uint64_t)&tma_V),
            "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])), "r"(0), "r"(h_S_offset + kv_off) : "memory");
    }
    
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])), "r"(0));

    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(c_S[0]), "l"(make_smem_desc_unswizzled(smem_Q, 128, 8)), "l"(make_smem_desc_unswizzled(smem_K, 128, 8)), "r"(make_instr_desc_fn<0, 0>(128, 128)), "r"(0));

    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(c_dP[0]), "l"(make_smem_desc_unswizzled(smem_V, 128, 8)), "l"(make_smem_desc_unswizzled(smem_dO, 128, 8)), "r"(make_instr_desc_fn<0, 0>(128, 128)), "r"(0));
    
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared.b64 [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])));
    
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])), "r"(1));

    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    
    for (uint32_t c = 0; c < 128; c += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(c_S[0] + c));
        float f0 = __uint_as_float(r0) * attn_scale;
        float f1 = __uint_as_float(r1) * attn_scale;
        float f2 = __uint_as_float(r2) * attn_scale;
        float f3 = __uint_as_float(r3) * attn_scale;
        
        float p0 = fast_exp2f_fn((f0 - L[h_S_offset + q_off + tid]) * 1.4426950408889634f);
        float p1 = fast_exp2f_fn((f1 - L[h_S_offset + q_off + tid]) * 1.4426950408889634f);
        float p2 = fast_exp2f_fn((f2 - L[h_S_offset + q_off + tid]) * 1.4426950408889634f);
        float p3 = fast_exp2f_fn((f3 - L[h_S_offset + q_off + tid]) * 1.4426950408889634f);
        
        smem_P[tid * 128 + c] = __float2bfloat16(p0);
        smem_P[tid * 128 + c + 1] = __float2bfloat16(p1);
        smem_P[tid * 128 + c + 2] = __float2bfloat16(p2);
        smem_P[tid * 128 + c + 3] = __float2bfloat16(p3);
    }
    __syncthreads();
    fence_async_shared_fn();
    
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(c_dP[0]), "l"(make_smem_desc_unswizzled(smem_V, 128, 8)), "l"(make_smem_desc_unswizzled(smem_dO, 128, 8)), "r"(make_instr_desc_fn<0, 0>(128, 128)), "r"(1));
    
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared.b64 [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])));
    
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])), "r"(1));
    
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    
    for (uint32_t c = 0; c < 128; c += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(c_dP[0] + c));
        float dp0 = __uint_as_float(r0);
        float dp1 = __uint_as_float(r1);
        float dp2 = __uint_as_float(r2);
        float dp3 = __uint_as_float(r3);
        
        float p0 = __bfloat162float(smem_P[tid * 128 + c]);
        float p1 = __bfloat162float(smem_P[tid * 128 + c + 1]);
        float p2 = __bfloat162float(smem_P[tid * 128 + c + 2]);
        float p3 = __bfloat162float(smem_P[tid * 128 + c + 3]);
        
        float ds0 = p0 * (dp0 - smem_D[tid]);
        float ds1 = p1 * (dp1 - smem_D[tid]);
        float ds2 = p2 * (dp2 - smem_D[tid]);
        float ds3 = p3 * (dp3 - smem_D[tid]);
        
        smem_dS[tid * 128 + c] = __float2bfloat16(ds0);
        smem_dS[tid * 128 + c + 1] = __float2bfloat16(ds1);
        smem_dS[tid * 128 + c + 2] = __float2bfloat16(ds2);
        smem_dS[tid * 128 + c + 3] = __float2bfloat16(ds3);
    }
    __syncthreads();
    fence_async_shared_fn();
    
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(c_dV[0]), "l"(make_smem_desc_unswizzled(smem_P, 8, 128)), "l"(make_smem_desc_unswizzled(smem_dO, 8, 128)), "r"(make_instr_desc_fn<1, 1>(128, 128)), "r"(1));
        
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(c_dK[0]), "l"(make_smem_desc_unswizzled(smem_dS, 8, 128)), "l"(make_smem_desc_unswizzled(smem_Q, 8, 128)), "r"(make_instr_desc_fn<1, 1>(128, 128)), "r"(1));
        
    // Load existing dQ from global mem into c_dQ
    for (uint32_t c = 0; c < 128; c += 4) {
        for (uint32_t i = 0; i < 4; ++i) {
            float fq = 0;
            if (q_off + tid < S && c + i < 128) {
                fq = __bfloat162float(dQ[head_idx * S * 128 + (q_off + tid) * 128 + (c + i)]);
            }
            asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 {%0}, [%1];" :: "f"(fq), "r"(c_dQ[c + i]));
        }
    }
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
    fence_async_shared_fn();
    
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(c_dQ[0]), "l"(make_smem_desc_unswizzled(smem_dS, 128, 8)), "l"(make_smem_desc_unswizzled(smem_K, 8, 128)), "r"(make_instr_desc_fn<0, 1>(128, 128)), "r"(1));
    
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared.b64 [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])));
    
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])), "r"(1));
    
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    
    // Atomic add dQ back to global
    for (uint32_t c = 0; c < 128; c += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(c_dQ[c]));
        if (q_off + tid < S) {
            uint32_t base_idx = head_idx * S * 128 + (q_off + tid) * 128;
            __nv_bfloat162 val0 = __halff2bfloat1622(__float2bfloat16(__uint_as_float(r0)), __float2bfloat16(__uint_as_float(r1)));
            atomicAdd((__nv_bfloat162*)(dQ + base_idx + c), val0);
            
            __nv_bfloat162 val1 = __halff2bfloat1622(__float2bfloat16(__uint_as_float(r2)), __float2bfloat16(__uint_as_float(r3)));
            atomicAdd((__nv_bfloat162*)(dQ + base_idx + c + 2), val1);
        }
    }
}

__global__ void store_epilogue_kernel(
    uint32_t c_dV, uint32_t c_dK,
    __nv_bfloat16* dV, __nv_bfloat16* dK,
    uint32_t kv_off, uint32_t head_idx, int32_t S)
{
    uint32_t tid = threadIdx.x;
    for (uint32_t c = 0; c < 128; c += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(c_dV + c));
        if (kv_off + tid < S) {
            uint32_t base_idx_v = head_idx * S * 128 + (kv_off + tid) * 128;
            *reinterpret_cast<uint32_t*>(dV + base_idx_v + c) = pack_bf16_fn(r0, r1);
            *reinterpret_cast<uint32_t*>(dV + base_idx_v + c + 2) = pack_bf16_fn(r2, r3);
        }
        
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(c_dK + c));
        if (kv_off + tid < S) {
            uint32_t base_idx_k = head_idx * S * 128 + (kv_off + tid) * 128;
            *reinterpret_cast<uint32_t*>(dK + base_idx_k + c) = pack_bf16_fn(r0, r1);
            *reinterpret_cast<uint32_t*>(dK + base_idx_k + c + 2) = pack_bf16_fn(r2, r3);
        }
    }
}