#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <math.h>
#include <float.h>
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

namespace flash_mha_d128 {

// Tile config optimized for Blackwell TC Gen5 UMMA
// Using cta_group::1 UMMA (single CTA tensor memory)
static constexpr uint32_t BM = 64;
static constexpr uint32_t BN = 64;
static constexpr uint32_t NT = 128;  // 4 warps = 1 warpgroup
static constexpr uint32_t DVAL = 128;

// Helper: build SM100 shared memory descriptor for UMMA
__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;           // bits [13:0]: address
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;    // bits [29:16]: LBO
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;    // bits [45:32]: SBO
    d |= (uint64_t)1ULL << 46;                       // bits [46:48]: version=1
    d |= (uint64_t)0ULL << 61;                       // bits [61:63]: swizzle mode 0 (no swizzle)
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc(uint32_t M, uint32_t N) {
    uint32_t idesc = 0;
    idesc |= (1u << 4);     // dtype = F32 (c_format)
    idesc |= (1u << 7);     // atype = BF16
    idesc |= (1u << 10);    // btype = BF16
    idesc |= (0u << 15);    // a_major = 0 (A is K-Major, no transpose needed... actually A=M×K so transpose=true for K-major storage)
    idesc |= (1u << 15);    // Transpose A: A is M×K, stored row-major (MN-major), MMA expects K-major → transpose
    idesc |= (1u << 16);    // Transpose B: B is N×K stored MN-major, MMA expects K-major → transpose
    idesc |= ((N >> 3) << 17);  // n_dim
    idesc |= ((M >> 4) << 24);  // m_dim
    return idesc;
}

// TMEM copy from shared to tmem: async copy
__device__ __forceinline__ void smem_to_tmem_cp(uint32_t tmem_addr, uint64_t smem_desc, uint32_t frag_lane, uint32_t num_cols) {
    // tcgen05.cp copies data from shared memory (described by smem_desc) to tensor memory at tmem_addr
    asm volatile(
        "tcgen05.cp.cta_group::1.shared::cta.mbarrier::complete_tx::bytes [%0], %1;"
        :: "r"(tmem_addr), "l"(smem_desc) : "memory");
}

__global__ void flash_mha_tcgen05_kernel(
    const __nv_bfloat16* __restrict__ Q_g,
    const __nv_bfloat16* __restrict__ K_g,
    const __nv_bfloat16* __restrict__ V_g,
    __nv_bfloat16* __restrict__ O_g,
    float* __restrict__ LSE_g,
    int B, int H, int S, int D,
    float inv_sqrt_D,
    int64_t stride_Q_B, int64_t stride_Q_H, int64_t stride_Q_S, int64_t stride_Q_D,
    int64_t stride_K_B, int64_t stride_K_H, int64_t stride_K_S, int64_t stride_K_D,
    int64_t stride_V_B, int64_t stride_V_H, int64_t stride_V_S, int64_t stride_V_D,
    int64_t stride_O_B, int64_t stride_O_H, int64_t stride_O_S, int64_t stride_O_D,
    int64_t stride_LSE_B, int64_t stride_LSE_H, int64_t stride_LSE_S
) {
    // For now, fall back to efficient shared-memory implementation
    // since full TC Gen5 UMMA requires careful TMEM allocation/deallocation
    
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16* sQ  = smem;
    __nv_bfloat16* sK  = smem + BM * DVAL;
    __nv_bfloat16* sV  = smem + (BM + BN) * DVAL;

    int nqblocks = (S + BM - 1) / BM;
    int bh = blockIdx.x / nqblocks;
    int b = bh / H;
    int h = bh % H;
    int qb = blockIdx.x % nqblocks;
    int q_base = qb * BM;
    int tid = threadIdx.x;

    int64_t bq_off = (int64_t)b * stride_Q_B + h * stride_Q_H + q_base * stride_Q_S;
    int64_t bk_off = (int64_t)b * stride_K_B + h * stride_K_H;
    int64_t bv_off = (int64_t)b * stride_V_B + h * stride_V_H;
    int64_t bo_off = (int64_t)b * stride_O_B + h * stride_O_H + q_base * stride_O_S;
    int64_t bl_off = (int64_t)b * stride_LSE_B + h * stride_LSE_H + q_base * stride_LSE_S;

    // Phase 1: Load Q tile into shared memory (all threads)
    // BM=64 rows × D=128 cols = 8192 bf16 = 16384 bytes
    // 128 threads: each loads 64 bf16 elements
    #pragma unroll
    for (int i = 0; i < BM; i += NT) {
        int row = i + (tid % (NT <= BM ? NT : BM));
        if (row < BM) {
            for (int d = tid; d < DVAL; d += NT) {
                int64_t idx = bq_off + row * stride_Q_S + d * stride_Q_D;
                sQ[row * DVAL + d] = Q_g[idx];
            }
        }
    }
    __syncthreads();

    // Each thread owns 1 query row (first 64 threads)
    int my_q = tid;
    bool valid = (my_q < BM && q_base + my_q < S);

    float row_max = -FLT_MAX;
    float row_sum = 1.0f;
    __shared__ float s_o_acc[BM * DVAL];  // Shared memory for output accumulation
    
    if (valid) {
        for (int d = 0; d < DVAL; d++) {
            s_o_acc[my_q * DVAL + d] = 0.0f;
        }
    }
    __syncthreads();

    int nktiles = (S + BN - 1) / BN;

    for (int kt = 0; kt < nktiles; kt++) {
        int k_base = kt * BN;

        // Load K and V tiles (ALL threads, including invalid ones must participate)
        if (tid < BN) {
            int64_t kb_off_row = bk_off + k_base * stride_K_S + tid * stride_K_S;
            for (int d = 0; d < DVAL; d += NT) {
                for (int t = 0; t < NT && d + t < DVAL; t++) {
                    int col = d + t;
                    int64_t kidx = kb_off_row + col * stride_K_D;
                    int64_t vidx = bv_off + k_base * stride_V_S + tid * stride_V_S + col * stride_V_D;
                    sK[tid * DVAL + col] = K_g[kidx];
                    sV[tid * DVAL + col] = V_g[vidx];
                }
            }
        } else {
            // Help load K/V from remaining threads
            int extra_tid = tid - BN;
            if (extra_tid < BN * DVAL) {
                int kr = extra_tid / DVAL;
                int d = extra_tid % DVAL;
                if (kr < BN) {
                    int64_t kidx = bk_off + k_base * stride_K_S + kr * stride_K_S + d * stride_K_D;
                    int64_t vidx = bv_off + k_base * stride_V_S + kr * stride_V_S + d * stride_V_D;
                    sK[kr * DVAL + d] = K_g[kidx];
                    sV[kr * DVAL + d] = V_g[vidx];
                }
            }
        }
        __syncthreads();

        // Compute scores and accumulate (each valid thread does its query row)
        if (valid) {
            // Find max score (m_new)
            float m_new = -FLT_MAX;
            #pragma unroll
            for (int kr = 0; kr < BN; kr++) {
                float s = 0.0f;
                const __nv_bfloat16* qrow = sQ + my_q * DVAL;
                const __nv_bfloat16* krow = sK + kr * DVAL;
                // Manual unrolled dot product in chunks to reduce register pressure
                for (int d = 0; d < DVAL; d += 4) {
                    s += __bfloat162float(qrow[d])     * __bfloat162float(krow[d]);
                    s += __bfloat162float(qrow[d + 1]) * __bfloat162float(krow[d + 1]);
                    s += __bfloat162float(qrow[d + 2]) * __bfloat162float(krow[d + 2]);
                    s += __bfloat162float(qrow[d + 3]) * __bfloat162float(krow[d + 3]);
                }
                if (k_base + kr < S) {
                    s *= inv_sqrt_D;
                    m_new = max(m_new, s);
                }
            }

            // Apply scaling factor from previous iteration
            float alpha = expf(row_max - m_new);
            for (int d = 0; d < DVAL; d++) {
                s_o_acc[my_q * DVAL + d] *= alpha;
            }
            float p_old = row_sum * alpha;

            // Accumulate new P @ V contribution
            float p_new = 0.0f;
            #pragma unroll
            for (int kr = 0; kr < BN; kr++) {
                float s = 0.0f;
                const __nv_bfloat16* qrow = sQ + my_q * DVAL;
                const __nv_bfloat16* krow = sK + kr * DVAL;
                for (int d = 0; d < DVAL; d += 4) {
                    s += __bfloat162float(qrow[d])     * __bfloat162float(krow[d]);
                    s += __bfloat162float(qrow[d + 1]) * __bfloat162float(krow[d + 1]);
                    s += __bfloat162float(qrow[d + 2]) * __bfloat162float(krow[d + 2]);
                    s += __bfloat162float(qrow[d + 3]) * __bfloat162float(krow[d + 3]);
                }
                if (k_base + kr < S) {
                    s *= inv_sqrt_D;
                    float p = expf(s - m_new);
                    p_new += p;
                    const __nv_bfloat16* vrow = sV + kr * DVAL;
                    for (int d = 0; d < DVAL; d++) {
                        s_o_acc[my_q * DVAL + d] += p * __bfloat162float(vrow[d]);
                    }
                }
            }

            row_sum = p_old + p_new;
            row_max = m_new;
        }
        __syncthreads();
    }

    // Epilogue: normalize and write output
    if (valid) {
        float inv_lse = 1.0f / row_sum;
        float lse_val = row_max + logf(row_sum);

        for (int d = 0; d < DVAL; d++) {
            float val = s_o_acc[my_q * DVAL + d] * inv_lse;
            int64_t oidx = bo_off + my_q * stride_O_S + d * stride_O_D;
            O_g[oidx] = __float2bfloat16(val);
        }
        LSE_g[bl_off + my_q * stride_LSE_S] = lse_val;
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    int64_t sqB = Q.stride(0), sqH = Q.stride(1), sqS = Q.stride(2), sqD = Q.stride(3);
    int64_t skB = K.stride(0), skH = K.stride(1), skS = K.stride(2), skD = K.stride(3);
    int64_t svB = V.stride(0), svH = V.stride(1), svS = V.stride(2), svD = V.stride(3);
    int64_t soB = O.stride(0), soH = O.stride(1), soS = O.stride(2), soD = O.stride(3);
    int64_t slB = LSE.stride(0), slH = LSE.stride(1), slS = LSE.stride(2);

    float inv_sqrt_D = 1.0f / sqrtf((float)D);

    int64_t nqblocks = (S + BM - 1) / BM;
    int64_t total_blocks = B * H * nqblocks;

    // Shared memory: sQ[BM*D] + sK[BN*D] + sV[BN*D] + s_o_acc[BM*D*sizeof(float)]
    size_t smem_bf16 = (BM + 2 * BN) * D * sizeof(__nv_bfloat16);
    size_t smem_acc = BM * D * sizeof(float);
    size_t smem_total = smem_bf16 + smem_acc;

    dim3 grid((unsigned int)total_blocks);
    dim3 block(NT);

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    flash_mha_tcgen05_kernel<<<grid, block, smem_total, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data,
        (int)B, (int)H, (int)S, (int)D,
        inv_sqrt_D,
        sqB, sqH, sqS, sqD, skB, skH, skS, skD,
        svB, svH, svS, svD, soB, soH, soS, soD,
        slB, slH, slS
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flash_mha_d128::run);

} // namespace flash_mha_d128