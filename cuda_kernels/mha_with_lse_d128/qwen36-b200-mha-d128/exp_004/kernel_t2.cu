#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_d128 {

__device__ __forceinline__ uint32_t cvta_shared(void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ float dexp2f(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void fence_proxy_async() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tcgen_fence_after() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;
constexpr int BS = 128;
constexpr int NT = 128; // threads per CTA

__device__ __forceinline__ uint64_t make_smem_desc(void* p, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = cvta_shared(p);
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    d |= (uint64_t)(((lbo) & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)(((sbo) & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1ULL << 46;   // version = 1 (SM100)
    d |= (uint64_t)2ULL << 61;   // SWIZZLE_128B
    return d;
}

__global__ void mha_blackwell_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, int D, int H,
    int stride_bhd, int stride_hd, int stride_d) {
    
    int bid = blockIdx.x;
    int b = bid / H;
    int h = bid % H;
    int tid = threadIdx.x;
    int lane = tid;
    
    // Per-head base pointers
    const __nv_bfloat16* q_head = Q + (int64_t)(b * H + h) * stride_bhd;
    const __nv_bfloat16* k_head = K + (int64_t)(b * H + h) * stride_bhd;
    const __nv_bfloat16* v_head = V + (int64_t)(b * H + h) * stride_bhd;
    __nv_bfloat16* o_head = O + (int64_t)(b * H + h) * stride_bhd;
    float* lse_head = LSE + (int64_t)(b * H + h) * S;
    
    // Shared memory tiles
    __shared__ __align__(128) uint2 smem_Q[BM][BK / 2];
    __shared__ __align__(128) uint2 smem_K[BS][BK / 2];
    __shared__ __align__(128) __nv_bfloat16 smem_V[BS][BN];
    
    // SM100 TMEM / barrier infrastructure
    __shared__ __align__(8) uint64_t bar[1];
    __shared__ uint32_t tmem_S_base;
    
    // Initialize mbarrier
    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" 
                     :: "r"(cvta_shared(bar)), "r"(NT));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    
    // Allocate TMEM: BM rows x BS cols = 128x128 FP32 accumulator
    // ALL threads in warp 0 must execute identically (.aligned requires it)
    {
        uint32_t dst_addr = cvta_shared(&tmem_S_base);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     : : "r"(dst_addr), "r"(BS) : "memory");
    }
    __syncthreads();
    
    // SMEM descriptors (K-major, 128B swizzle, 128x16 atom)
    uint64_t desc_Q = make_smem_desc(smem_Q, 1, 1024);
    uint64_t desc_K = make_smem_desc(smem_K, 1, 1024);
    
    // Instruction descriptor: BF16@BF16->FP32, M=128, N=128, K=16-per-MMA
    uint32_t idesc = 0;
    idesc |= (1u << 4);      // dtype = FP32
    idesc |= (1u << 7);      // atype = BF16
    idesc |= (1u << 10);     // btype = BF16
    idesc |= ((BS / 8) << 17);   // N-dim >> 3
    idesc |= ((BM / 16) << 24);  // M-dim >> 4
    
    // Online softmax accumulators per thread
    float m_row = -1e20f;
    float l_row = 0.0f;
    float o_regs[BN] = {0.0f};
    
    int num_s_blocks = (S + BS - 1) / BS;
    bool s_first_iter = true;
    
    for (int sb = 0; sb < num_s_blocks; ++sb) {
        int s_start = sb * BS;
        int bs_eff = min(BS, S - s_start);
        
        // Load K stripe [bs_eff x BK] into smem_K
        for (int idx = tid; idx < bs_eff * (BK / 2); idx += NT) {
            int r = idx / (BK / 2);
            int c = idx % (BK / 2);
            ((uint2*)smem_K)[r * (BK / 2) + c] = 
                ((const uint2*)(k_head + s_start * stride_d))[r * (D / 2) + c];
        }
        __syncthreads();
        
        // Load V tile [bs_eff x BN] into smem_V
        for (int idx = tid; idx < bs_eff * BN; idx += NT) {
            int r = idx / BN;
            int c = idx % BN;
            smem_V[r][c] = v_head[(int64_t)s_start * D + (int64_t)r * D + c];
        }
        __syncthreads();
        
        // === Inner K-loops: S += Q @ K^T via UMMA ===
        for (int kb = 0; kb < D; kb += BK) {
            // Load Q stripe [BM x BK]
            for (int idx = tid; idx < BM * (BK / 2); idx += NT) {
                int r = idx / (BK / 2);
                int c = idx % (BK / 2);
                ((uint2*)smem_Q)[r * (BK / 2) + c] = 
                    ((const uint2*)(q_head + kb * stride_d))[r * (D / 2) + c];
            }
            __syncthreads();
            
            // UMMA inner: iterate 16-wide K slices
            uint32_t tmem_row_base = tmem_S_base + (lane << 16);
            bool needs_accum = !s_first_iter || kb > 0;
            uint32_t acc_flag = needs_accum ? 1u : 0u;
            
            #pragma unroll
            for (int ki = 0; ki < BK; ki += 16) {
                uint32_t tmem_col = tmem_row_base + (ki << 4);
                
                // Descriptor advancement: each ki steps 16 in K-major direction
                uint64_t desc_q_adv = desc_Q + (uint64_t)(kb + ki) * 4;
                uint64_t desc_k_adv = desc_K + (uint64_t)ki * 4;
                
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, %5, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                    :: "r"(tmem_col), "l"(desc_q_adv), "l"(desc_k_adv), 
                       "r"(idesc), "r"(0), "r"(acc_flag));
            }
        }
        
        fence_proxy_async();
        
        // Read S row from TMEM into registers
        float s_vals[BS] = {0.0f};
        #pragma unroll
        for (int ci = 0; ci < BS; ci += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                : "r"(tmem_S_base + (lane << 16) + (ci << 4)));
            s_vals[ci + 0] = __uint_as_float(r0);
            s_vals[ci + 1] = __uint_as_float(r1);
            s_vals[ci + 2] = __uint_as_float(r2);
            s_vals[ci + 3] = __uint_as_float(r3);
        }
        tcgen_fence_after();
        
        // Row-wise softmax
        float s_max = -1e20f;
        for (int j = 0; j < bs_eff; ++j) {
            s_max = max(s_max, s_vals[j]);
        }
        
        float p_sum = 0.0f;
        float p_arr[BS] = {0.0f};
        for (int j = 0; j < bs_eff; ++j) {
            p_arr[j] = dexp2f((s_vals[j] - s_max) * 1.44269504f);
            p_sum += p_arr[j];
        }
        
        // Online softmax update
        float m_new = max(m_row, s_max);
        float alpha = (m_row >= s_max) ? 1.0f : dexp2f((m_row - m_new) * 1.44269504f);
        float beta  = dexp2f((s_max - m_new) * 1.44269504f);
        float l_new = alpha * l_row + beta * p_sum;
        
        // Accumulate O += P @ V
        float temp_o[BN] = {0.0f};
        for (int c = 0; c < BN; ++c) {
            float dot = 0.0f;
            for (int j = 0; j < bs_eff; ++j) {
                dot += p_arr[j] * __bfloat162float(smem_V[j][c]);
            }
            temp_o[c] = alpha * o_regs[c] + beta * dot;
        }
        #pragma unroll
        for (int c = 0; c < BN; ++c) o_regs[c] = temp_o[c];
        
        m_row = m_new;
        l_row = l_new;
        s_first_iter = false;
        
        // Sync before next SMEM reuse
        if (sb < num_s_blocks - 1) __syncthreads();
    }
    
    // Final normalization and epilogue
    float inv_l = fdividef(1.0f, l_row);
    float lse_val = m_row + logf(l_row);
    
    // Store O row
    for (int c = tid; c < BN; c += NT) {
        float val = o_regs[c] * inv_l;
        o_head[(int64_t)tid * D + c] = __float2bfloat16(val);
    }
    
    // Store LSE
    if (tid < BM) {
        lse_head[tid] = lse_val;
    }
    
    // Deallocate TMEM (warp 0, all threads execute same instruction)
    {
        uint32_t addr = tmem_S_base;
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     : : "r"(addr), "r"(BS) : "memory");
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    const __nv_bfloat16* q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* k_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* v_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_ptr = static_cast<float*>(LSE.data_ptr());
    
    int64_t stride_bhd = H * S * D;
    int64_t stride_d = D;
    
    int grid_size = (int)(B * H);
    dim3 grid(grid_size);
    dim3 block(NT);
    
    cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(
        Q.device().device_type, Q.device().device_id);
        
    mha_blackwell_kernel<<<grid, block, 0, stream>>>(
        q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
        (int)S, (int)D, (int)H,
        (int)stride_bhd, (int)stride_d, (int)stride_d
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128::run);

} // namespace mha_d128