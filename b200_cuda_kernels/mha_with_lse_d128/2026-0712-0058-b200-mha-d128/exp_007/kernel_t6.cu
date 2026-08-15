#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CU_CHECK(call) do {                                      \
    CUresult _e = (call);                                        \
    if (_e != CUDA_SUCCESS) {                                    \
        fprintf(stderr, "CU error %d at %s:%d\n",                 \
                (int)_e, __FILE__, __LINE__);                    \
        exit(1);                                                 \
    }                                                            \
} while(0)

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace tvm_ffi_kernel {

CUresult create_tma_3d_descriptor_2B(CUtensorMap* d, void* globalAddress, uint64_t gmem_dim0, uint64_t gmem_dim1, uint64_t gmem_dim2, uint32_t smem_dim0, uint32_t smem_dim1, uint32_t smem_dim2, CUtensorMapSwizzle swizzle, CUtensorMapL2promotion l2Promotion, CUtensorMapFloatOOBfill oobFill) {
    cuuint64_t globalDim[3] = {gmem_dim0, gmem_dim1, gmem_dim2};
    cuuint64_t globalStrides[2] = {gmem_dim0 * 2, gmem_dim0 * gmem_dim1 * 2};
    cuuint32_t boxDim[3] = {smem_dim0, smem_dim1, smem_dim2};
    cuuint32_t elementStrides[3] = {1, 1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3,
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

__device__ __forceinline__ void tma_load_3d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, int32_t c2) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4, %5}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1), "r"(c2) : "memory");
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

__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
   :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
   :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void tmem_load_4x_fn(uint32_t addr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3) : "r"(addr));
}

__device__ __forceinline__ void tmem_load_fence_fn() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_store_4x_fn(uint32_t addr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("tcgen05.st.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
   :: "r"(*r0), "r"(*r1), "r"(*r2), "r"(*r3), "r"(addr));
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

__device__ __forceinline__ void umma_commit_1sm(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile(
        "tcgen05.commit.cta_group::1"
        ".mbarrier::arrive::one.shared::cta.b64"
        " [%0];"
        :: "r"(a));
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

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, uint32_t a_major, uint32_t b_major) {
    uint32_t d = 0;
    d |= (1u << 4);
    d |= (1u << 7);
    d |= (1u << 10);
    d |= (a_major << 15);
    d |= (b_major << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

struct SharedStorage {
    alignas(1024) __nv_bfloat16 s_Q[16384];
    alignas(1024) __nv_bfloat16 s_K[16384];
    alignas(1024) __nv_bfloat16 s_V0[8192];
    alignas(1024) __nv_bfloat16 s_V1[8192];
    alignas(1024) __nv_bfloat16 s_P[16384];
    alignas(1024) __nv_bfloat16 s_O[16384];
    alignas(128) float s_M[128];
    alignas(128) float s_L[128];
};

__shared__ alignas(8) uint64_t bar_Q;
__shared__ alignas(8) uint64_t bar_KV;
__shared__ alignas(8) uint64_t bar_S;
__shared__ alignas(8) uint64_t bar_PV;

__global__ __launch_bounds__(128) void run_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    __nv_bfloat16* O,
    float* LSE,
    uint32_t S)
{
    extern __shared__ char smem_pool[];
    SharedStorage* smem = (SharedStorage*)smem_pool;
    
    if (threadIdx.x == 0) {
        init_smem_barrier_fn(&bar_Q, 1);
        init_smem_barrier_fn(&bar_KV, 1);
        init_smem_barrier_fn(&bar_S, 1);
        init_smem_barrier_fn(&bar_PV, 1);
    }
    __syncthreads();
    fence_smem_barrier_init_fn();
    
    uint32_t row_base = blockIdx.x * 128;
    uint32_t bh = blockIdx.y;
    uint32_t tid = threadIdx.x;
    
    if (tid < 128) {
        smem->s_M[tid] = -INFINITY;
        smem->s_L[tid] = 0.0f;
    }
    __syncthreads();
    
    uint32_t s_tmem, o_tmem;
    if (tid == 0) {
        tmem_alloc_fn(&s_tmem, 128);
        tmem_alloc_fn(&o_tmem, 128);
    }
    __syncthreads();
    
    if (tid == 0) {
        mbarrier_arrive_and_expect_tx_fn(&bar_Q, 32768);
        tma_load_3d_fn(&tma_Q, &bar_Q, smem->s_Q, 0, bh * S + row_base, bh);
        tma_load_3d_fn(&tma_Q, &bar_Q, smem->s_Q + 8192, 64, bh * S + row_base, bh);
    }
    mbarrier_wait_fn(&bar_Q, 0);
    fence_proxy_async_fn();
    
    float scale_factor = 1.0f / sqrtf(128);
    uint32_t idesc_QKT = make_instr_desc_fn(128, 16, 0, 0);
    uint32_t idesc_PV = make_instr_desc_fn(128, 128, 0, 1);
    
    uint64_t desc_Q0 = make_smem_desc_sm100_fn(smem->s_Q, 1, 1024);
    uint64_t desc_Q0_p1 = make_smem_desc_sm100_fn(smem->s_Q + 8192, 1, 1024);
    uint64_t desc_K = make_smem_desc_sm100_fn(smem->s_K, 1, 1024);
    uint64_t desc_K_p1 = make_smem_desc_sm100_fn(smem->s_K + 8192, 1, 1024);
    uint64_t desc_P = make_smem_desc_sm100_fn(smem->s_P, 1, 1024);
    uint64_t desc_V0 = make_smem_desc_sm100_fn(smem->s_V0, 16384, 1024);
    uint64_t desc_V1 = make_smem_desc_sm100_fn(smem->s_V1, 16384, 1024);
    
    uint32_t num_iters = (S + 127) / 128;
    uint32_t phase_KV = 0, phase_S = 0, phase_PV = 0;
    
    for (uint32_t j = 0; j < num_iters; j++) {
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(&bar_KV, 65536);
            tma_load_3d_fn(&tma_K, &bar_KV, smem->s_K, 0, bh * S + j * 128, bh);
            tma_load_3d_fn(&tma_K, &bar_KV, smem->s_K + 8192, 64, bh * S + j * 128, bh);
            
            tma_load_3d_fn(&tma_V, &bar_KV, smem->s_V0, 0, bh * S + j * 128, bh);
            tma_load_3d_fn(&tma_V, &bar_KV, smem->s_V1, 64, bh * S + j * 128, bh);
        }
        mbarrier_wait_fn(&bar_KV, phase_KV);
        phase_KV ^= 1;
        fence_proxy_async_fn();
        
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(&bar_S, 0);
            for (int n_tile = 0; n_tile < 8; n_tile++) {
                for (int k = 0; k < 8; k++) {
                    int k_half = k / 2;
                    uint64_t desc_Q0_k = (k_half < 2) ? (desc_Q0 + k_half * 16) : (desc_Q0_p1 + (k_half - 2) * 16);
                    uint64_t desc_K_k = (k_half < 2) ? (desc_K + k_half * 16) : (desc_K_p1 + (k_half - 2) * 16);
                    
                    uint32_t s_addr = s_tmem + (n_tile * 16) * 128;
                    umma_f16_cg1_fn(s_addr, desc_Q0_k, desc_K_k, idesc_QKT, (k==0)?0:1);
                }
            }
            umma_commit_1sm(&bar_S);
        }
        mbarrier_wait_fn(&bar_S, phase_S);
        phase_S ^= 1;
        
        float max_val = -INFINITY;
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            uint32_t addr_s = s_tmem + tid * 128 + col;
            tmem_load_4x_fn(addr_s, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            float val0 = __uint_as_float(r0); float p0 = val0 * scale_factor; if (j * 128 + col + 0 >= S) p0 = -INFINITY;
            float val1 = __uint_as_float(r1); float p1 = val1 * scale_factor; if (j * 128 + col + 1 >= S) p1 = -INFINITY;
            float val2 = __uint_as_float(r2); float p2 = val2 * scale_factor; if (j * 128 + col + 2 >= S) p2 = -INFINITY;
            float val3 = __uint_as_float(r3); float p3 = val3 * scale_factor; if (j * 128 + col + 3 >= S) p3 = -INFINITY;
            max_val = fmaxf(max_val, fmaxf(fmaxf(p0, p1), fmaxf(p2, p3)));
        }

        float sum_val = 0;
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            uint32_t addr_s = s_tmem + tid * 128 + col;
            tmem_load_4x_fn(addr_s, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            float val0 = __uint_as_float(r0); float p0 = val0 * scale_factor; if (j * 128 + col + 0 >= S) p0 = -INFINITY; else p0 = expf(p0 - max_val);
            float val1 = __uint_as_float(r1); float p1 = val1 * scale_factor; if (j * 128 + col + 1 >= S) p1 = -INFINITY; else p1 = expf(p1 - max_val);
            float val2 = __uint_as_float(r2); float p2 = val2 * scale_factor; if (j * 128 + col + 2 >= S) p2 = -INFINITY; else p2 = expf(p2 - max_val);
            float val3 = __uint_as_float(r3); float p3 = val3 * scale_factor; if (j * 128 + col + 3 >= S) p3 = -INFINITY; else p3 = expf(p3 - max_val);
            
            sum_val += p0 + p1 + p2 + p3;
            
            int c_x0 = col / 8; int c_rem0 = col % 8; int sc_x0 = (tid % 8) ^ c_x0; int sc0 = sc_x0 * 8 + c_rem0; int idx0 = tid * 128 + sc0;
            smem->s_P[idx0] = __float2bfloat16(p0);
            
            int c_x1 = (col+1) / 8; int c_rem1 = (col+1) % 8; int sc_x1 = (tid % 8) ^ c_x1; int sc1 = sc_x1 * 8 + c_rem1; int idx1 = tid * 128 + sc1;
            smem->s_P[idx1] = __float2bfloat16(p1);
            
            int c_x2 = (col+2) / 8; int c_rem2 = (col+2) % 8; int sc_x2 = (tid % 8) ^ c_x2; int sc2 = sc_x2 * 8 + c_rem2; int idx2 = tid * 128 + sc2;
            smem->s_P[idx2] = __float2bfloat16(p2);
            
            int c_x3 = (col+3) / 8; int c_rem3 = (col+3) % 8; int sc_x3 = (tid % 8) ^ c_x3; int sc3 = sc_x3 * 8 + c_rem3; int idx3 = tid * 128 + sc3;
            smem->s_P[idx3] = __float2bfloat16(p3);
        }
        
        float global_M = smem->s_M[tid];
        float nm = fmaxf(global_M, max_val);
        float alpha = expf(global_M - nm);
        smem->s_M[tid] = nm;
        smem->s_L[tid] = smem->s_L[tid] * alpha + sum_val;
        
        for (int col = 0; col < 128; col += 4) {
            uint32_t r0, r1, r2, r3;
            uint32_t addr_o = o_tmem + tid * 128 + col;
            tmem_load_4x_fn(addr_o, &r0, &r1, &r2, &r3);
            tmem_load_fence_fn();
            
            float f0 = __uint_as_float(r0) * alpha;
            float f1 = __uint_as_float(r1) * alpha;
            float f2 = __uint_as_float(r2) * alpha;
            float f3 = __uint_as_float(r3) * alpha;
            
            uint32_t nf0 = __float_as_uint(f0);
            uint32_t nf1 = __float_as_uint(f1);
            uint32_t nf2 = __float_as_uint(f2);
            uint32_t nf3 = __float_as_uint(f3);
            
            tmem_store_4x_fn(addr_o, &nf0, &nf1, &nf2, &nf3);
        }
        tmem_store_fence_fn();
        fence_proxy_async_fn();
        __syncthreads(); 
        
        if (tid == 0) {
            mbarrier_arrive_and_expect_tx_fn(&bar_PV, 0);
            for (int k = 0; k < 8; k++) {
                uint64_t desc_P_k = desc_P + k * 16;
                uint64_t desc_V0_k = desc_V0 + k * 128;
                uint64_t desc_V1_k = desc_V1 + k * 128;
                
                uint32_t o_addr0 = o_tmem + (k * 16) * 128;
                uint32_t o_addr1 = o_tmem + (k * 16) * 128 + 8192;
                
                umma_f16_cg1_fn(o_addr0, desc_P_k, desc_V0_k, idesc_PV, (k==0)?0:1);
                umma_f16_cg1_fn(o_addr1, desc_P_k, desc_V1_k, idesc_PV, (k==0)?0:1);
            }
            umma_commit_1sm(&bar_PV);
        }
        mbarrier_wait_fn(&bar_PV, phase_PV);
        phase_PV ^= 1;
        
        __syncthreads();
    }
    
    for (int col = 0; col < 128; col += 4) {
        uint32_t r0, r1, r2, r3;
        uint32_t addr_o = o_tmem + tid * 128 + col;
        tmem_load_4x_fn(addr_o, &r0, &r1, &r2, &r3);
        tmem_load_fence_fn();
        
        float o0 = __uint_as_float(r0) / smem->s_L[tid];
        float o1 = __uint_as_float(r1) / smem->s_L[tid];
        float o2 = __uint_as_float(r2) / smem->s_L[tid];
        float o3 = __uint_as_float(r3) / smem->s_L[tid];
        
        int c_x0 = col / 8; int c_rem0 = col % 8; int sc_x0 = (tid % 8) ^ c_x0; int sc0 = sc_x0 * 8 + c_rem0; int idx0 = tid * 128 + sc0;
        smem->s_O[idx0] = __float2bfloat16(o0);
        
        int c_x1 = (col+1) / 8; int c_rem1 = (col+1) % 8; int sc_x1 = (tid % 8) ^ c_x1; int sc1 = sc_x1 * 8 + c_rem1; int idx1 = tid * 128 + sc1;
        smem->s_O[idx1] = __float2bfloat16(o1);
        
        int c_x2 = (col+2) / 8; int c_rem2 = (col+2) % 8; int sc_x2 = (tid % 8) ^ c_x2; int sc2 = sc_x2 * 8 + c_rem2; int idx2 = tid * 128 + sc2;
        smem->s_O[idx2] = __float2bfloat16(o2);
        
        int c_x3 = (col+3) / 8; int c_rem3 = (col+3) % 8; int sc_x3 = (tid % 8) ^ c_x3; int sc3 = sc_x3 * 8 + c_rem3; int idx3 = tid * 128 + sc3;
        smem->s_O[idx3] = __float2bfloat16(o3);
    }
    __syncthreads();
    
    for (int i = tid; i < 128 * 128; i += 128) {
        int row = i / 128;
        int col = i % 128;
        int global_row = row_base + row;
        if (global_row < S && col < 128) {
            int c_x = col / 8;
            int c_rem = col % 8;
            int swizzled_c_x = (row % 8) ^ c_x;
            int sc = c_x * 8 + c_rem;
            int idx = row * 128 + sc;
            
            int out_idx = (bh * S + global_row) * 128 + col;
            O[out_idx] = smem->s_O[idx];
        }
    }
    
    if (tid < 128) {
        int global_row = row_base + tid;
        if (global_row < S) {
            LSE[(uint64_t)bh * S + global_row] = smem->s_M[tid] + logf(smem->s_L[tid]);
        }
    }
    
    if (tid == 0) {
        tmem_dealloc_fn(s_tmem, 128);
        tmem_dealloc_fn(o_tmem, 128);
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V, tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    uint32_t B = Q.size(0);
    uint32_t H = Q.size(1);
    uint32_t S = Q.size(2);
    uint32_t D = Q.size(3);
    
    CUtensorMap tma_Q, tma_K, tma_V;
    __nv_bfloat16* Q_ptr = static_cast<__nv_bfloat16*>(Q.data_ptr());
    __nv_bfloat16* K_ptr = static_cast<__nv_bfloat16*>(K.data_ptr());
    __nv_bfloat16* V_ptr = static_cast<__nv_bfloat16*>(V.data_ptr());
    
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_Q, Q_ptr, D, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_K, K_ptr, D, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    CU_CHECK(create_tma_3d_descriptor_2B(&tma_V, V_ptr, D, S, B * H, 64, 128, 1, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    
    dim3 grid((S + 127) / 128, B * H);
    dim3 block(128);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUDA_CHECK(cudaFuncSetAttribute(run_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 140 * 1024));
    run_kernel<<<grid, block, 140 * 1024, stream>>>(tma_Q, tma_K, tma_V, static_cast<__nv_bfloat16*>(O.data_ptr()), static_cast<float*>(LSE.data_ptr()), S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_kernel