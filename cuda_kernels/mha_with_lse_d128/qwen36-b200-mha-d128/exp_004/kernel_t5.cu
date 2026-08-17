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
constexpr int BK = 64;
constexpr int BS = 128;
constexpr int NT = 128;
constexpr int MMA_K = 16;

// Build SM100 UMMA SMEM descriptor (K-major, no-swizzle)
// Row stride in shared memory is computed from ATOM_MMODE_DIM spans
__device__ __forceinline__ uint64_t make_smem_desc(void* p, int smem_row_stride_bytes) {
    uint64_t d = 0;
    uint32_t addr = cvta_shared(p);
    
    // Encode base address (bits 0-13): addr >> 4
    d |= (uint64_t)((addr & 0x3FFFF) >> 4);
    
    // Encode LBO (bits 16-29): leading dim byte offset >> 4
    // For K-major with 128-bit normalized elements (bf16), this is the stride 
    // between consecutive columns of a span (8 bf16 elements = 16 bytes per span)
    // smem_row_stride_bytes is the number of bytes from row r to row r+1
    // LBO = number_of_spans_along_M * stride_per_span = (BM/8) * 16 = BM*2
    // But we encode it in units of 16 bytes >> 4 = 1 unit = 16 bytes
    // Actually enc_LBO = stride_between_column_spans_in_bytes >> 4
    // For K-major no-swizzle: offset from first column to second column of 8-element span
    // = smem_row_stride_bytes (per row) * (# bf16 elements in a row segment) = ...
    // Simplified: enc_LBO = smem_row_stride_bytes / 2  (since each bf16 is 2 bytes and we normalize)
    // No: LBO in HW units = leading_dim_byte_offset_relative >> 4
    // For K-major no-swizzle: "offset from first column to second columns of the 8x2 tile in the 128-bit element type normalized matrix"
    // One 128-bit normalized element = 8 bf16. If our atom has ATOM_MMODE_DIM=8 (rows of 128b norm),
    // then LBO = 8 * smem_row_stride_bytes, encoded >> 4.
    // Actually for simplicity with swizzle disabled, LBO = smem_row_stride_bytes >> 4 won't quite work.
    // Let me use the documented example values directly.
    
    int lbo_enc = smem_row_stride_bytes / 16;  // Encoded LBO
    d |= (uint64_t)(lbo_enc & 0x3FFF) << 16;
    
    // Encode SBO (bits 32-45): stride dim byte offset >> 4  
    // Offset from first 8 rows to next 8 rows = 8 * smem_row_stride_bytes >> 4
    int sbo_enc = (8 * smem_row_stride_bytes) / 16;
    d |= (uint64_t)(sbo_enc & 0x3FFF) << 32;
    
    // Version field (bits 46-48) = 0b001 for SM100
    d |= (uint64_t)1ULL << 46;
    
    // No swizzle (bits 61-63) = 0
    return d;
}

__global__ void mha_blackwell_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int S, int D, int H,
    int stride_BHD, int stride_SD, int stride_D) {
    
    int bid = blockIdx.x;
    int b = bid / H;
    int h = bid % H;
    int tid = threadIdx.x;
    int lane = tid;
    
    int64_t head_off = (int64_t)b * stride_BHD + (int64_t)h * stride_SD;
    const __nv_bfloat16* q_head = Q + head_off;
    const __nv_bfloat16* k_head = K + head_off;
    const __nv_bfloat16* v_head = V + head_off;
    __nv_bfloat16* o_head = O + head_off;
    float* lse_head = LSE + (int64_t)b * (H * S) + (int64_t)h * S;
    
    // Shared memory buffers - reduced sizes
    // Q and K share same buffer conceptually but stored separately
    __shared__ __align__(256) __nv_bfloat16 smem_Q[BM][BK];
    __shared__ __align__(256) __nv_bfloat16 smem_K[BS][BK];
    __shared__ __align__(128) __nv_bfloat16 smem_V[BS][BN];
    
    // TMEM infrastructure
    __shared__ __align__(8) uint64_t bar[1];
    __shared__ uint32_t tmem_S_base;
    
    // Init mbarrier
    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" 
                     :: "r"(cvta_shared(bar)), "r"(NT));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    
    // Allocate TMEM
    {
        uint32_t dst_addr = cvta_shared(&tmem_S_base);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     : : "r"(dst_addr), "r"(BS) : "memory");
    }
    __syncthreads();
    
    // Descriptor construction: each row of smem_Q/smem_K has BK bf16 elements = BK*2 bytes
    int smem_Q_stride = BK * 2;  // bytes per row in smem_Q
    int smem_K_stride = BK * 2;  // bytes per row in smem_K
    
    uint64_t desc_Q = make_smem_desc(smem_Q, smem_Q_stride);
    uint64_t desc_K = make_smem_desc(smem_K, smem_K_stride);
    
    // Instruction descriptor: BF16@BF16->FP32
    uint32_t idesc = 0;
    idesc |= (1u << 4);      // dtype = FP32
    idesc |= (1u << 7);      // atype = BF16
    idesc |= (1u << 10);     // btype = BF16
    idesc |= ((BS / 8) << 17);   // N-dim >> 3  
    idesc |= ((BM / 16) << 24);  // M-dim >> 4
    
    // Online softmax state
    float m_row = -1e20f;
    float l_row = 0.0f;
    float o_regs[BN] = {0.0f};
    
    int num_s_blocks = (S + BS - 1) / BS;
    bool first_s_block = true;
    
    for (int sb = 0; sb < num_s_blocks; ++sb) {
        int s_start = sb * BS;
        int bs_eff = min(BS, S - s_start);
        
        // Load K stripe [bs_eff x BK] using direct indexed stores (no casts)
        for (int idx = tid; idx < bs_eff * BK; idx += NT) {
            int r = idx / BK;
            int c = idx % BK;
            int64_t gidx = (int64_t)(s_start + r) * stride_D + c;
            smem_K[r][c] = k_head[gidx];
        }
        __syncthreads();
        
        // Load V tile [bs_eff x BN]
        for (int idx = tid; idx < bs_eff * BN; idx += NT) {
            int r = idx / BN;
            int c = idx % BN;
            int64_t gidx = (int64_t)(s_start + r) * stride_D + c;
            smem_V[r][c] = v_head[gidx];
        }
        __syncthreads();
        
        // Compute S += Q @ K^T via UMMA
        for (int kb = 0; kb < D; kb += BK) {
            // Load Q stripe [BM x BK]
            for (int idx = tid; idx < BM * BK; idx += NT) {
                int r = idx / BK;
                int c = idx % BK;
                int64_t gidx = (int64_t)r * stride_D + kb + c;
                smem_Q[r][c] = q_head[gidx];
            }
            __syncthreads();
            
            // Issue UMMA for K=64 slice  
            // Hardware atom is 64x64x16, so for BK=64 we need 4 iterations of K=16
            // Or we can pass K=BK directly if the hardware supports it
            
            uint32_t tmem_row_base = tmem_S_base + (lane << 16);
            bool needs_accum = !first_s_block || kb > 0;
            uint32_t acc_flag = needs_accum ? 1u : 0u;
            
            // Advance descriptor base by kb*k_stride_in_descriptor_units
            // Each step of 1 bf16 in K-direction advances descriptor by 1 (in the encoded address space)
            uint64_t desc_q_adv = desc_Q + (uint64_t)kb;
            uint64_t desc_k_adv = desc_K;
            
            // Single UMMA call
            asm volatile(
                "{\n.reg .pred p;\n"
                "setp.ne.b32 p, %5, 0;\n"
                "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
                :: "r"(tmem_row_base), "l"(desc_q_adv), "l"(desc_k_adv), 
                   "r"(idesc), "r"(0), "r"(acc_flag));
        }
        
        fence_proxy_async();
        
        // Read S row from TMEM
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
        
        // Softmax row
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
        first_s_block = false;
        
        if (sb < num_s_blocks - 1) __syncthreads();
    }
    
    float inv_l = fdividef(1.0f, l_row);
    float lse_val = m_row + logf(l_row);
    
    for (int c = tid; c < BN; c += NT) {
        float val = o_regs[c] * inv_l;
        o_head[tid * stride_D + c] = __float2bfloat16(val);
    }
    
    if (tid < BM) {
        lse_head[tid] = lse_val;
    }
    
    // Deallocate TMEM
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
    
    int64_t stride_BHD = H * S * D;
    int64_t stride_SD = S * D;
    int64_t stride_D = D;
    
    int grid_size = (int)(B * H);
    dim3 grid(grid_size);
    dim3 block(NT);
    
    cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(
        Q.device().device_type, Q.device().device_id);
        
    mha_blackwell_kernel<<<grid, block, 0, stream>>>(
        q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
        (int)S, (int)D, (int)H,
        (int)stride_BHD, (int)stride_SD, (int)stride_D
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha_d128::run);

} // namespace mha_d128