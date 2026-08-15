#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

namespace mha_d128 {

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 64;
constexpr int BS = 128;
constexpr int NUM_THREADS = 128;

__device__ __forceinline__ uint32_t cvta_shared(void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
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
    int lane = tid; // Each thread owns one row of the 128x128 tile
    
    // Base pointers for current head
    const __nv_bfloat16* q_base = Q + (b * H + h) * stride_bhd;
    const __nv_bfloat16* k_base = K + (b * H + h) * stride_bhd;
    const __nv_bfloat16* v_base = V + (b * H + h) * stride_bhd;
    __nv_bfloat16* o_base = O + (b * H + h) * stride_bhd;
    float* lse_base = LSE + (b * H + h) * S;
    
    // Shared memory tiles
    __shared__ __align__(128) uint2 smem_Q[BM][BK/2];
    __shared__ __align__(128) uint2 smem_K[BS][BK/2];
    __shared__ __align__(128) uint2 smem_V[BS][BN/2];
    __shared__ __align__(8) uint64_t bar[1];
    __shared__ uint32_t tmem_S_base;
    
    // Initialize mbarrier
    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" 
                     :: "r"(cvta_shared(bar)), "r"(NUM_THREADS));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    
    // Allocate Tensor Memory for S accumulator (BM x BS)
    if (tid == 0) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     : "=r"(tmem_S_base) : "r"(cvta_shared(&tmem_S_base)), "r"(BS));
    }
    __syncthreads();
    
    // Descriptor builders
    auto make_smem_desc = [](void* p, uint32_t lbo, uint32_t sbo) {
        uint64_t d = 0;
        uint32_t addr = cvta_shared(p);
        d |= (uint64_t)((addr & 0x3FFFF) >> 4);
        d |= (uint64_t)(((lbo) & 0x3FFFF) >> 4) << 16;
        d |= (uint64_t)(((sbo) & 0x3FFFF) >> 4) << 32;
        d |= (uint64_t)1ULL << 46;   // version = 1
        d |= (uint64_t)2ULL << 61;   // SWIZZLE_128B
        return d;
    };
    
    uint64_t desc_Q_base = make_smem_desc(smem_Q, 1, 1024);
    uint64_t desc_K_base = make_smem_desc(smem_K, 1, 1024);
    
    // Instruction descriptor: BF16 x BF16 -> FP32, M=128, N=128
    uint32_t idesc = 0;
    idesc |= (1u << 4);     // dtype = FP32
    idesc |= (1u << 7);     // atype = BF16
    idesc |= (1u << 10);    // btype = BF16
    idesc |= ((BS / 8) << 17);
    idesc |= ((BM / 16) << 24);
    
    // Online softmax state per thread (each thread handles 1 row)
    float m_local = -1e20f;
    float l_local = 0.0f;
    float o_acc[BN] = {0.0f};
    
    int num_s_blocks = (S + BS - 1) / BS;
    bool first_s_iter = true;
    
    for (int sb = 0; sb < num_s_blocks; ++sb) {
        int s_off = sb * BS;
        int bs_eff = min(BS, S - s_off);
        
        // Load K tile [bs_eff x BK]
        for (int i = tid; i < bs_eff * (BK/2); i += NUM_THREADS) {
            int r = i / (BK/2);
            int c = i % (BK/2);
            ((uint2*)smem_K)[r * (BK/2) + c] = ((const uint2*)(k_base + s_off * stride_d))[r * (D/2) + c];
        }
        __syncthreads();
        
        // Load V tile [bs_eff x BN]
        for (int i = tid; i < bs_eff * (BN/2); i += NUM_THREADS) {
            int r = i / (BN/2);
            int c = i % (BN/2);
            ((uint2*)smem_V)[r * (BN/2) + c] = ((const uint2*)(v_base + s_off * stride_d))[r * (D/2) + c];
        }
        __syncthreads();
        
        // Compute S += Q @ K^T
        // TMEM base for this thread's row
        uint32_t tmem_c = tmem_S_base + (lane << 16);
        
        for (int kb = 0; kb < D; kb += BK) {
            // Load Q stripe [BM x BK]
            for (int i = tid; i < BM * (BK/2); i += NUM_THREADS) {
                int r = i / (BK/2);
                int c = i % (BK/2);
                ((uint2*)smem_Q)[r * (BK/2) + c] = ((const uint2*)(q_base + kb * stride_d))[r * (D/2) + c];
            }
            __syncthreads();
            
            bool acc_flag = !(first_s_iter && kb == 0);
            uint32_t accum = acc_flag ? 1 : 0;
            
            // Advance descriptor bases for current K block
            uint64_t desc_Q = desc_Q_base + ((uint64_t)kb << 4);
            uint64_t desc_K = desc_K_base; // K starts at tile beginning
            
            // Issue UMMA: each iteration handles K=16. We loop BK/16 times.
            #pragma unroll
            for (int ki = 0; ki < BK; ki += 16) {
                uint32_t tmem_dst = tmem_c + (ki << 4); // Column offset in TMEM
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, %5, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                    :: "r"(tmem_dst), "l"(desc_Q + ((uint64_t)ki << 4)), 
                       "l"(desc_K + ((uint64_t)ki << 4)), "r"(idesc), "r"(0), "r"(accum)
                    : "memory");
            }
        }
        
        fence.proxy.async();
        
        // Read S row from TMEM into registers
        float s_row[BS] = {0.0f};
        #pragma unroll
        for (int ci = 0; ci < BS; ci += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) 
                         : "r"(tmem_S_base + (lane << 16) + (ci << 4)));
            s_row[ci]   = __uint_as_float(r0);
            s_row[ci+1] = __uint_as_float(r1);
            s_row[ci+2] = __uint_as_float(r2);
            s_row[ci+3] = __uint_as_float(r3);
        }
        tcgen05.fence::after_thread_sync(); // Ensure ld completed
        
        // Row-wise Softmax
        float s_max = -1e20f;
        for (int j = 0; j < bs_eff; ++j) {
            if (s_row[j] > s_max) s_max = s_row[j];
        }
        // Warp reduction for max (optional, but per-thread max is fine for online softmax)
        // Actually online softmax computes per-row max independently. Correct.
        
        float p_sum = 0.0f;
        float p_vals[BS];
        #pragma unroll
        for (int j = 0; j < bs_eff; ++j) {
            float p = fast_exp2f_fn((s_row[j] - s_max) * 1.44269504f); // log2(e) approx
            // Use native exp2 for precision if available, else fallback
            asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(p_vals[j]) : "f__((s_row[j] - s_max) * 1.44269504f));
            p_sum += p_vals[j];
            // Mask out-of-bounds
            if (j >= bs_eff) p_vals[j] = 0.0f;
        }
        
        // Online softmax update
        float m_new = max(m_local, s_max);
        float alpha = fast_exp2f_fn((m_local - m_new) * 1.44269504f);
        float beta = fast_exp2f_fn((s_max - m_new) * 1.44269504f);
        
        float l_new = alpha * l_local + beta * p_sum;
        
        // Accumulate O += P @ V
        // Each thread computes its row of O
        for (int c = 0; c < BN; ++c) {
            float dot = 0.0f;
            for (int j = 0; j < bs_eff; ++j) {
                dot += p_vals[j] * ((const float*)(&((const uint2*)smem_V)[j]))[c]; // unsafe cast for speed, better: reinterpret
            }
            o_acc[c] = alpha * o_acc[c] + beta * dot;
        }
        
        // Safer V load in loop above:
        // Rewrite V access properly:
        float temp_o[BN] = {0};
        for(int c=0; c<BN; ++c) {
            float dot = 0;
            for(int j=0; j<bs_eff; ++j) {
                __nv_bfloat16 v_val = ((__nv_bfloat16*)smem_V)[j * (BN/2) * 2 + c];
                dot += p_vals[j] * __bfloat162float(v_val);
            }
            temp_o[c] = alpha * o_acc[c] + beta * dot;
        }
        #pragma unroll
        for(int c=0; c<BN; ++c) o_acc[c] = temp_o[c];
        
        m_local = m_new;
        l_local = l_new;
        first_s_iter = false;
        
        __syncthreads(); // Ensure next SMEM loads don't race
    }
    
    // Final normalization and store
    float inv_l = 1.0f / l_local;
    float lse_val = m_local + __logf(l_local);
    
    // Store O row
    for (int c = tid; c < BN; c += NUM_THREADS) {
        int global_c = c;
        float val = o_acc[c] * inv_l;
        if (tid < BM && global_c < D) {
            o_base[tid * stride_d + global_c] = __float2bfloat16(val);
        }
    }
    
    // Store LSE
    if (tid < BM) {
        lse_base[tid] = lse_val;
    }
    
    // Deallocate TMEM
    if (tid == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(tmem_S_base), "r"(BS));
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    const __nv_bfloat16* q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* k_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* v_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* o_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* lse_data = static_cast<float*>(LSE.data_ptr());
    
    int64_t stride_bhd = H * S * D;
    int64_t stride_hd = S * D;
    int64_t stride_d = D;
    
    int grid_size = B * H;
    dim3 grid(grid_size);
    dim3 block(NUM_THREADS);
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
        
    mha_blackwell_kernel<<<grid, block, 0, stream>>>(
        q_data, k_data, v_data, o_data, lse_data,
        (int)S, (int)D, (int)H,
        (int)stride_bhd, (int)stride_hd, (int)stride_d
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128::run);

} // namespace mha_d128