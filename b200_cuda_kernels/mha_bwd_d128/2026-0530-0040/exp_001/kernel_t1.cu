#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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
        const char* err_str;                                       \
        cuGetErrorString(_e, &err_str);                            \
        fprintf(stderr, "CU error %s at %s:%d\n",                  \
                err_str, __FILE__, __LINE__);                      \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_example_cuda {

CUresult create_tma_2d_descriptor_2B(
    CUtensorMap* d, void* globalAddress, 
    uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, 
    uint32_t smem_inner_dim, uint32_t smem_outer_dim, 
    CUtensorMapSwizzle swizzle, 
    CUtensorMapL2promotion l2Promotion, 
    CUtensorMapFloatOOBfill oobFill) 
{
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

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
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
    asm("mov.b32 %0, {%1, %2};"
        : "=r"(result)
        : "h"(*reinterpret_cast<uint16_t*>(&a)),
          "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ uint16_t float2bfloat16_u16(float f) {
    __nv_bfloat16 b = __float2bfloat16(f);
    return *reinterpret_cast<uint16_t*>(&b);
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

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
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

__device__ __forceinline__ void umma_f16_cg1(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_cg1_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
        :: "r"(a) : "memory");
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

__device__ __forceinline__ uint64_t offset_desc(uint64_t desc, uint32_t bytes) {
    return desc + (bytes >> 4);
}

__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N, int a_trans, int b_trans) {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= ((a_trans & 1) << 15);
    d |= ((b_trans & 1) << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void atomic_add_bf16x2(void* ptr, uint32_t val) {
    atomicAdd((__nv_bfloat162*)ptr, *(__nv_bfloat162*)&val);
}

__device__ __forceinline__ void st_shared_swizzle_128B(void* smem_base, int row, int col, uint16_t val) {
    int block_idx = col / 64;
    int c = col % 64;
    int r = row;
    int chunk_idx = (c * 2) / 16;
    int swizzled_chunk = chunk_idx ^ (r % 8);
    int final_offset = r * 128 + swizzled_chunk * 16 + (c * 2) % 16;
    char* ptr = (char*)smem_base + block_idx * 16384 + final_offset;
    *(uint16_t*)ptr = val;
}

__device__ __forceinline__ void st_shared_swizzle_128B_vec8(void* smem_base, int row, int col_start, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    int block_idx = col_start / 64;
    int c = col_start % 64;
    int r = row;
    int chunk_idx = (c * 2) / 16; 
    int swizzled_chunk = chunk_idx ^ (r % 8);
    int final_offset = r * 128 + swizzled_chunk * 16;
    char* ptr = (char*)smem_base + block_idx * 16384 + final_offset;
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"((uint32_t)__cvta_generic_to_shared(ptr)), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}

__device__ __forceinline__ void tmem_epilogue_normal_4w_fn(
    __nv_bfloat16* dV_out, __nv_bfloat16* smem_out, uint32_t tmem_base,
    uint32_t global_row_offset, uint32_t global_col_offset, uint32_t BN) 
{
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
       : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_base + col));
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
    uint32_t num_steps = 128 / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        uint32_t col_start = lane_id * 4;
        
        uint2 data = *(uint2*)(&smem_out[row * 128 + col_start]);
        
        uint64_t global_idx = ((uint64_t)global_row_offset + row) * 128 + (global_col_offset + col_start);
        *(uint2*)(dV_out + global_idx) = data;
    }
}

__device__ __forceinline__ void tmem_epilogue_transpose_4w_fn(
    __nv_bfloat16* dK_out, __nv_bfloat16* smem_out, uint32_t tmem_base,
    uint32_t global_row_offset, uint32_t global_col_offset, uint32_t BN) 
{
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
       : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_base + col));
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
    uint32_t num_steps = 128 / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        uint32_t col_start = lane_id * 4;
        
        uint16_t v0 = *reinterpret_cast<uint16_t*>(&smem_out[(col_start + 0) * 128 + row]);
        uint16_t v1 = *reinterpret_cast<uint16_t*>(&smem_out[(col_start + 1) * 128 + row]);
        uint16_t v2 = *reinterpret_cast<uint16_t*>(&smem_out[(col_start + 2) * 128 + row]);
        uint16_t v3 = *reinterpret_cast<uint16_t*>(&smem_out[(col_start + 3) * 128 + row]);
        
        uint2 data;
        data.x = (v1 << 16) | v0;
        data.y = (v3 << 16) | v2;
        
        uint64_t global_idx = ((uint64_t)global_row_offset + row) * 128 + (global_col_offset + col_start);
        *(uint2*)(dK_out + global_idx) = data;
    }
}

__device__ __forceinline__ void tmem_epilogue_transpose_atomic_4w_fn(
    __nv_bfloat162* dQ_out, __nv_bfloat16* smem_out, uint32_t tmem_base,
    uint32_t global_row_offset, uint32_t global_col_offset, uint32_t BN) 
{
    for (uint32_t col = 0; col < BN; col += 4) {
        uint32_t r0, r1, r2, r3;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
       : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_base + col));
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
    uint32_t num_steps = 128 / 4;
    for (uint32_t step = 0; step < num_steps; ++step) {
        uint32_t row = step * 4 + warp_id;
        uint32_t col_start = lane_id * 4;
        
        uint16_t v0 = *reinterpret_cast<uint16_t*>(&smem_out[(col_start + 0) * 128 + row]);
        uint16_t v1 = *reinterpret_cast<uint16_t*>(&smem_out[(col_start + 1) * 128 + row]);
        uint16_t v2 = *reinterpret_cast<uint16_t*>(&smem_out[(col_start + 2) * 128 + row]);
        uint16_t v3 = *reinterpret_cast<uint16_t*>(&smem_out[(col_start + 3) * 128 + row]);
        
        uint32_t b01 = (v1 << 16) | v0;
        uint32_t b23 = (v3 << 16) | v2;
        
        uint64_t global_idx = ((uint64_t)global_row_offset + row) * 64 + (global_col_offset + col_start) / 2;
        atomic_add_bf16x2(&dQ_out[global_idx], b01);
        atomic_add_bf16x2(&dQ_out[global_idx + 1], b23);
    }
}

__global__ void precompute_D_kernel(const __nv_bfloat16* O, const __nv_bfloat16* dO, float* D, int S) {
    int b = blockIdx.x; 
    int s_blk = blockIdx.y;
    int tid = threadIdx.x;
    
    int row = s_blk * 128 + tid;
    if (row >= S) return;
    
    int offset = b * S * 128 + row * 128;
    
    float d_val = 0.0f;
    for (int k = 0; k < 128; k += 2) { 
        float o1 = __bfloat162float(O[offset + k]);
        float o2 = __bfloat162float(O[offset + k + 1]);
        float do1 = __bfloat162float(dO[offset + k]);
        float do2 = __bfloat162float(dO[offset + k + 1]);
        d_val += o1 * do1 + o2 * do2;
    }
    D[b * S + row] = d_val;
}

__global__ void bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q, 
    const __grid_constant__ CUtensorMap tma_K, 
    const __grid_constant__ CUtensorMap tma_V, 
    const __grid_constant__ CUtensorMap tma_dO,
    const float* L, const float* D, 
    __nv_bfloat162* dQ, __nv_bfloat16* dV, __nv_bfloat16* dK,
    int B, int H, int S) 
{
    setmaxnreg_inc_sync_fn<240>();
    
    int j = blockIdx.x; 
    int b = blockIdx.y;
    int h = blockIdx.z;
    int tid = threadIdx.x;
    
    int num_q_blocks = S / 128;
    
    extern __shared__ __align__(1024) char smem_buffer[];
    void* Q_smem = smem_buffer + 0;
    void* dO_smem = smem_buffer + 32768;
    void* K_smem = smem_buffer + 65536;
    void* V_smem = smem_buffer + 98304;
    void* P_smem = smem_buffer + 131072;
    void* dS_K_smem = smem_buffer + 163840;
    void* dS_Q_smem = smem_buffer + 196608;

    __shared__ __align__(8) uint64_t bar_KV[1];
    __shared__ __align__(8) uint64_t bar_QdO[1];
    __shared__ __align__(8) uint64_t bar_UMMA[1];

    if (tid == 0) {
        init_smem_barrier_fn(bar_KV, 1);
        init_smem_barrier_fn(bar_QdO, 1);
        init_smem_barrier_fn(bar_UMMA, 1);
    }
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    uint32_t tmem_base;
    tmem_alloc_cg1_fn(&tmem_base, 512);
    
    uint32_t tmem_S  = tmem_base;
    uint32_t tmem_dQ = tmem_base; 
    uint32_t tmem_dP = tmem_base + 128;
    uint32_t tmem_dV = tmem_base + 256;
    uint32_t tmem_dK = tmem_base + 384;
    
    uint64_t desc_K_Kmajor = make_smem_desc_sm100_fn(K_smem, 1, 1024);
    uint64_t desc_Q_Kmajor = make_smem_desc_sm100_fn(Q_smem, 1, 1024);
    uint64_t desc_V_Kmajor = make_smem_desc_sm100_fn(V_smem, 1, 1024);
    uint64_t desc_dO_Kmajor = make_smem_desc_sm100_fn(dO_smem, 1, 1024);

    uint64_t desc_K_MNmajor = make_smem_desc_sm100_fn(K_smem, 16384, 1024);
    uint64_t desc_Q_MNmajor = make_smem_desc_sm100_fn(Q_smem, 16384, 1024);
    uint64_t desc_dO_MNmajor = make_smem_desc_sm100_fn(dO_smem, 16384, 1024);

    uint64_t desc_P_MNmajor = make_smem_desc_sm100_fn(P_smem, 16384, 1024);
    uint64_t desc_dS_K_MNmajor = make_smem_desc_sm100_fn(dS_K_smem, 16384, 1024);
    uint64_t desc_dS_Q_MNmajor = make_smem_desc_sm100_fn(dS_Q_smem, 16384, 1024);
    
    uint32_t idesc_128x128_K = make_instr_desc(128, 128, 0, 0); 
    uint32_t idesc_64x64_MN = make_instr_desc(64, 64, 1, 1); 
    
    int umma_phase = 0;
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(bar_KV, 32768 * 2);
        tma_load_2d_fn(&tma_K, bar_KV, K_smem, 0, (b * H + h) * S + j * 128);
        tma_load_2d_fn(&tma_K, bar_KV, (char*)K_smem + 16384, 64, (b * H + h) * S + j * 128);
        tma_load_2d_fn(&tma_V, bar_KV, V_smem, 0, (b * H + h) * S + j * 128);
        tma_load_2d_fn(&tma_V, bar_KV, (char*)V_smem + 16384, 64, (b * H + h) * S + j * 128);
    }
    mbarrier_wait_fn(bar_KV, 0);
    
    float scale = 1.0f / sqrtf(128.0f);
    float log2e = 1.44269504f;
    
    for (int i = 0; i < num_q_blocks; i++) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(bar_QdO, 32768 * 2);
            tma_load_2d_fn(&tma_Q, bar_QdO, Q_smem, 0, (b * H + h) * S + i * 128);
            tma_load_2d_fn(&tma_Q, bar_QdO, (char*)Q_smem + 16384, 64, (b * H + h) * S + i * 128);
            tma_load_2d_fn(&tma_dO, bar_QdO, dO_smem, 0, (b * H + h) * S + i * 128);
            tma_load_2d_fn(&tma_dO, bar_QdO, (char*)dO_smem + 16384, 64, (b * H + h) * S + i * 128);
        }
        mbarrier_wait_fn(bar_QdO, i & 1);
        
        float l_val = L[(b * H + h) * S + i * 128 + tid];
        float d_val = D[(b * H + h) * S + i * 128 + tid];
        
        umma_f16_cg1(tmem_S, desc_K_Kmajor, desc_Q_Kmajor, idesc_128x128_K, 0);
        umma_f16_cg1(tmem_S, offset_desc(desc_K_Kmajor, 16384), offset_desc(desc_Q_Kmajor, 16384), idesc_128x128_K, 1);
        
        umma_f16_cg1(tmem_dP, desc_V_Kmajor, desc_dO_Kmajor, idesc_128x128_K, 0);
        umma_f16_cg1(tmem_dP, offset_desc(desc_V_Kmajor, 16384), offset_desc(desc_dO_Kmajor, 16384), idesc_128x128_K, 1);
        
        if (tid == 0) umma_commit_cg1_fn(bar_UMMA);
        mbarrier_wait_fn(bar_UMMA, umma_phase & 1);
        umma_phase++;
        
        for (uint32_t col = 0; col < 128; col += 8) {
            uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) : "r"(tmem_S + col));
            uint32_t dr0, dr1, dr2, dr3, dr4, dr5, dr6, dr7;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                         : "=r"(dr0),"=r"(dr1),"=r"(dr2),"=r"(dr3),"=r"(dr4),"=r"(dr5),"=r"(dr6),"=r"(dr7) : "r"(tmem_dP + col));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            float p0 = fast_exp2f_fn((__uint_as_float(r0) * scale - l_val) * log2e);
            float p1 = fast_exp2f_fn((__uint_as_float(r1) * scale - l_val) * log2e);
            float p2 = fast_exp2f_fn((__uint_as_float(r2) * scale - l_val) * log2e);
            float p3 = fast_exp2f_fn((__uint_as_float(r3) * scale - l_val) * log2e);
            float p4 = fast_exp2f_fn((__uint_as_float(r4) * scale - l_val) * log2e);
            float p5 = fast_exp2f_fn((__uint_as_float(r5) * scale - l_val) * log2e);
            float p6 = fast_exp2f_fn((__uint_as_float(r6) * scale - l_val) * log2e);
            float p7 = fast_exp2f_fn((__uint_as_float(r7) * scale - l_val) * log2e);
            
            st_shared_swizzle_128B(P_smem, col + 0, tid, float2bfloat16_u16(p0));
            st_shared_swizzle_128B(P_smem, col + 1, tid, float2bfloat16_u16(p1));
            st_shared_swizzle_128B(P_smem, col + 2, tid, float2bfloat16_u16(p2));
            st_shared_swizzle_128B(P_smem, col + 3, tid, float2bfloat16_u16(p3));
            st_shared_swizzle_128B(P_smem, col + 4, tid, float2bfloat16_u16(p4));
            st_shared_swizzle_128B(P_smem, col + 5, tid, float2bfloat16_u16(p5));
            st_shared_swizzle_128B(P_smem, col + 6, tid, float2bfloat16_u16(p6));
            st_shared_swizzle_128B(P_smem, col + 7, tid, float2bfloat16_u16(p7));
            
            float ds0 = p0 * (__uint_as_float(dr0) - d_val) * scale;
            float ds1 = p1 * (__uint_as_float(dr1) - d_val) * scale;
            float ds2 = p2 * (__uint_as_float(dr2) - d_val) * scale;
            float ds3 = p3 * (__uint_as_float(dr3) - d_val) * scale;
            float ds4 = p4 * (__uint_as_float(dr4) - d_val) * scale;
            float ds5 = p5 * (__uint_as_float(dr5) - d_val) * scale;
            float ds6 = p6 * (__uint_as_float(dr6) - d_val) * scale;
            float ds7 = p7 * (__uint_as_float(dr7) - d_val) * scale;
            
            st_shared_swizzle_128B(dS_K_smem, col + 0, tid, float2bfloat16_u16(ds0));
            st_shared_swizzle_128B(dS_K_smem, col + 1, tid, float2bfloat16_u16(ds1));
            st_shared_swizzle_128B(dS_K_smem, col + 2, tid, float2bfloat16_u16(ds2));
            st_shared_swizzle_128B(dS_K_smem, col + 3, tid, float2bfloat16_u16(ds3));
            st_shared_swizzle_128B(dS_K_smem, col + 4, tid, float2bfloat16_u16(ds4));
            st_shared_swizzle_128B(dS_K_smem, col + 5, tid, float2bfloat16_u16(ds5));
            st_shared_swizzle_128B(dS_K_smem, col + 6, tid, float2bfloat16_u16(ds6));
            st_shared_swizzle_128B(dS_K_smem, col + 7, tid, float2bfloat16_u16(ds7));
            
            uint32_t b01 = pack_bf16_fn(__float_as_uint(ds0), __float_as_uint(ds1));
            uint32_t b23 = pack_bf16_fn(__float_as_uint(ds2), __float_as_uint(ds3));
            uint32_t b45 = pack_bf16_fn(__float_as_uint(ds4), __float_as_uint(ds5));
            uint32_t b67 = pack_bf16_fn(__float_as_uint(ds6), __float_as_uint(ds7));
            
            st_shared_swizzle_128B_vec8(dS_Q_smem, tid, col, b01, b23, b45, b67);
        }
        __syncthreads();
        fence_proxy_async_fn();
        
        uint32_t accum = (i == 0) ? 0 : 1;
        
        umma_f16_cg1(tmem_dV, desc_P_MNmajor, desc_dO_MNmajor, idesc_64x64_MN, accum);
        umma_f16_cg1(tmem_dV + (64 << 16), offset_desc(desc_P_MNmajor, 16384), desc_dO_MNmajor, idesc_64x64_MN, accum);
        umma_f16_cg1(tmem_dV + 64, desc_P_MNmajor, offset_desc(desc_dO_MNmajor, 16384), idesc_64x64_MN, accum);
        umma_f16_cg1(tmem_dV + (64 << 16) + 64, offset_desc(desc_P_MNmajor, 16384), offset_desc(desc_dO_MNmajor, 16384), idesc_64x64_MN, accum);
        
        umma_f16_cg1(tmem_dK, desc_Q_MNmajor, desc_dS_K_MNmajor, idesc_64x64_MN, accum);
        umma_f16_cg1(tmem_dK + (64 << 16), offset_desc(desc_Q_MNmajor, 16384), desc_dS_K_MNmajor, idesc_64x64_MN, accum);
        umma_f16_cg1(tmem_dK + 64, desc_Q_MNmajor, offset_desc(desc_dS_K_MNmajor, 16384), idesc_64x64_MN, accum);
        umma_f16_cg1(tmem_dK + (64 << 16) + 64, offset_desc(desc_Q_MNmajor, 16384), offset_desc(desc_dS_K_MNmajor, 16384), idesc_64x64_MN, accum);
        
        umma_f16_cg1(tmem_dQ, desc_K_MNmajor, desc_dS_Q_MNmajor, idesc_64x64_MN, 0);
        umma_f16_cg1(tmem_dQ + (64 << 16), offset_desc(desc_K_MNmajor, 16384), desc_dS_Q_MNmajor, idesc_64x64_MN, 0);
        umma_f16_cg1(tmem_dQ + 64, desc_K_MNmajor, offset_desc(desc_dS_Q_MNmajor, 16384), idesc_64x64_MN, 0);
        umma_f16_cg1(tmem_dQ + (64 << 16) + 64, offset_desc(desc_K_MNmajor, 16384), offset_desc(desc_dS_Q_MNmajor, 16384), idesc_64x64_MN, 0);
        
        if (tid == 0) umma_commit_cg1_fn(bar_UMMA);
        mbarrier_wait_fn(bar_UMMA, umma_phase & 1);
        umma_phase++;
        
        __syncthreads();
        tmem_epilogue_transpose_atomic_4w_fn(dQ, (__nv_bfloat16*)Q_smem, tmem_dQ, (b * H + h) * S + i * 128, 0, 128);
        __syncthreads();
    }
    
    __syncthreads();
    tmem_epilogue_normal_4w_fn(dV, (__nv_bfloat16*)Q_smem, tmem_dV, (b * H + h) * S + j * 128, 0, 128);
    __syncthreads();
    
    tmem_epilogue_transpose_4w_fn(dK, (__nv_bfloat16*)Q_smem, tmem_dK, (b * H + h) * S + j * 128, 0, 128);
    
    tmem_dealloc_cg1_fn(tmem_base, 512);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int d = Q.size(3); 
    
    __nv_bfloat16* Q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* K_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* V_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    __nv_bfloat16* dO_ptr = static_cast<__nv_bfloat16*>(dO.data_ptr());
    float* L_ptr = static_cast<float*>(L.data_ptr());
    
    __nv_bfloat162* dQ_ptr = static_cast<__nv_bfloat162*>(dQ.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());

    CUDA_CHECK(cudaMemsetAsync(dQ_ptr, 0, B * H * S * d * sizeof(__nv_bfloat16), stream));

    float* D_ptr;
    CUDA_CHECK(cudaMallocAsync(&D_ptr, B * H * S * sizeof(float), stream));

    dim3 grid_D(B * H, S / 128);
    dim3 block_D(128);
    precompute_D_kernel<<<grid_D, block_D, 0, stream>>>(O_ptr, dO_ptr, D_ptr, S);

    CUtensorMap tma_Q, tma_K, tma_V, tma_dO;
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_Q, Q_ptr, 128, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_K, K_ptr, 128, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_V, V_ptr, 128, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_2d_descriptor_2B(&tma_dO, dO_ptr, 128, B * H * S, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    dim3 grid(S / 128, B, H);
    dim3 block(128);
    
    CUDA_CHECK(cudaFuncSetAttribute(bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 229376));
    bwd_kernel<<<grid, block, 229376, stream>>>(
        tma_Q, tma_K, tma_V, tma_dO,
        L_ptr, D_ptr, dQ_ptr, dV_ptr, dK_ptr,
        B, H, S
    );
    
    CUDA_CHECK(cudaFreeAsync(D_ptr, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda