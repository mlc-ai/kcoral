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

__device__ __forceinline__ float fast_exp2f(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void fence_proxy_async() {
    asm volatile("fence.proxy.async;\n" ::: "memory");
}

__device__ __forceinline__ void tmem_fence_after() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;
constexpr int BS = 128;
constexpr int NUM_THREADS = 128;

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
    
    const __nv_bfloat16* q_base = Q + (b * H + h) * stride_bhd;
    const __nv_bfloat16* k_base = K + (b * H + h) * stride_bhd;
    const __nv_bfloat16* v_base = V + (b * H + h) * stride_bhd;
    __nv_bfloat16* o_base = O + (b * H + h) * stride_bhd;
    float* lse_base = LSE + (b * H + h) * S;
    
    __shared__ __align__(128) uint2 smem_Q[BM][BK/2];
    __shared__ __align__(128) uint2 smem_K[BS][BK/2];
    __shared__ __align__(128) uint2 smem_V[BS][BN/2];
    __shared__ __align__(8) uint64_t bar[1];
    __shared__ uint32_t tmem_S_base;
    
    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" 
                     :: "r"(cvta_shared(bar)), "r"(NUM_THREADS));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    
    if (tid == 0) {
        uint32_t addr;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     : "=r"(addr) : "r"(cvta_shared(&tmem_S_base)), "r"(BS));
        tmem_S_base = addr;
    }
    __syncthreads();
    
    auto make_smem_desc = [](void* p, uint32_t lbo, uint32_t sbo) {
        uint64_t d = 0;
        uint32_t addr = cvta_shared(p);
        d |= (uint64_t)((addr & 0x3FFFF) >> 4);
        d |= (uint64_t)(((lbo) & 0x3FFFF) >> 4) << 16;
        d |= (uint64_t)(((sbo) & 0x3FFFF) >> 4) << 32;
        d |= (uint64_t)1ULL << 46;
        d |= (uint64_t)2ULL << 61;
        return d;
    };
    
    uint64_t desc_Q_base = make_smem_desc(smem_Q, 1, 1024);
    uint64_t desc_K_base = make_smem_desc(smem_K, 1, 1024);
    
    uint32_t idesc = 0;
    idesc |= (1u << 4);
    idesc |= (1u << 7);
    idesc |= (1u << 10);
    idesc |= ((BS / 8) << 17);
    idesc |= ((BM / 16) << 24);
    
    float m_local = -1e20f;
    float l_local = 1.0f;
    float o_acc[BN] = {0.0f};
    
    int num_s_blocks = (S + BS - 1) / BS;
    
    for (int sb = 0; sb < num_s_blocks; ++sb) {
        int s_off = sb * BS;
        int bs_eff = min(BS, S - s_off);
        
        for (int i = tid; i < bs_eff * (BK/2); i += NUM_THREADS) {
            int r = i / (BK/2);
            int c = i % (BK/2);
            ((uint2*)smem_K)[r * (BK/2) + c] = ((const uint2*)(k_base + s_off * stride_d))[r * (D/2) + c];
        }
        __syncthreads();
        
        for (int i = tid; i < bs_eff * (BN/2); i += NUM_THREADS) {
            int r = i / (BN/2);
            int c = i % (BN/2);
            ((uint2*)smem_V)[r * (BN/2) + c] = ((const uint2*)(v_base + s_off * stride_d))[r * (D/2) + c];
        }
        __syncthreads();
        
        bool first_s_iter = (sb == 0);
        
        for (int kb = 0; kb < D; kb += BK) {
            for (int i = tid; i < BM * (BK/2); i += NUM_THREADS) {
                int r = i / (BK/2);
                int c = i % (BK/2);
                ((uint2*)smem_Q)[r * (BK/2) + c] = ((const uint2*)(q_base + kb * stride_d))[r * (D/2) + c];
            }
            __syncthreads();
            
            bool acc_flag = !first_s_iter || kb > 0;
            uint32_t accum = acc_flag ? 1 : 0;
            
            #pragma unroll
            for (int ki = 0; ki < BK; ki += 16) {
                uint32_t tmem_dst = tmem_S_base + (lane << 16) + (ki << 4);
                uint64_t desc_Q = desc_Q_base + ((uint64_t)(kb + ki) << 4);
                uint64_t desc_K = desc_K_base + ((uint64_t)ki << 4);
                
                asm volatile(
                    "{\n.reg .pred p;\n"
                    "setp.ne.b32 p, %5, 0;\n"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                    :: "r"(tmem_dst), "l"(desc_Q), 
                       "l"(desc_K), "r"(idesc), "r"(0), "r"(accum));
            }
        }
        
        fence_proxy_async();
        
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
        tmem_fence_after();
        
        float s_max = -1e20f;
        for (int j = 0; j < bs_eff; ++j) {
            if (s_row[j] > s_max) s_max = s_row[j];
        }
        
        float p_sum = 0.0f;
        float p_vals[BS] = {0.0f};
        for (int j = 0; j < bs_eff; ++j) {
            float p = fast_exp2f((s_row[j] - s_max) * 1.44269504f);
            p_vals[j] = p;
            p_sum += p;
        }
        
        float m_new = max(m_local, s_max);
        float alpha = (m_local >= s_max) ? 1.0f : fast_exp2f((m_local - m_new) * 1.44269504f);
        float beta = fast_exp2f((s_max - m_new) * 1.44269504f);
        
        float l_new = alpha * l_local + beta * p_sum;
        
        float temp_o[BN] = {0.0f};
        for(int c = 0; c < BN; ++c) {
            float dot = 0.0f;
            for(int j = 0; j < bs_eff; ++j) {
                __nv_bfloat16 v_val = ((__nv_bfloat16*)smem_V)[j * BN + c];
                dot += p_vals[j] * __bfloat162float(v_val);
            }
            temp_o[c] = alpha * o_acc[c] + beta * dot;
        }
        #pragma unroll
        for(int c = 0; c < BN; ++c) o_acc[c] = temp_o[c];
        
        m_local = m_new;
        l_local = l_new;
        
        if (sb < num_s_blocks - 1) __syncthreads();
    }
    
    float inv_l = fdividef(1.0f, l_local);
    float lse_val = m_local + logf(l_local);
    
    for (int c = tid; c < BN; c += NUM_THREADS) {
        float val = o_acc[c] * inv_l;
        o_base[tid * stride_d + c] = __float2bfloat16(val);
    }
    
    if (tid < BM) {
        lse_base[tid] = lse_val;
    }
    
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
    
    int grid_size = (int)(B * H);
    dim3 grid(grid_size);
    dim3 block(NUM_THREADS);
    
    cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id);
        
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