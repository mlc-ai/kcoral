#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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

namespace gemm_blackwell {

__device__ __forceinline__ uint64_t make_smem_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo, uint32_t swizzle_mode) {
    uint64_t d = 0;
    d |= ((uint64_t)(saddr & 0x3FFFFu) >> 4);
    d |= ((uint64_t)((lbo & 0x3FFFFu) >> 4) << 16);
    d |= ((uint64_t)((sbo & 0x3FFFFu) >> 4) << 32);
    d |= (uint64_t)1 << 46;  // version=1
    d |= (uint64_t)swizzle_mode << 61;
    return d;
}

template <int BM, int BN, int BK>
__global__ void gemm_kernel(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M, int N, int K) 
{
    // Warpgroup-based: 4 warps x 32 = 128 threads minimum
    // We use 256 threads: warps 0-3 are compute warpgroup, warps 4-7 handle loads/stores
    constexpr int WARPGROUP_SIZE = 128;
    constexpr int NTOTAL = 256;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Shared memory layout
    extern __shared__ char smem_bytes[];
    
    size_t off_As = 0;
    size_t off_Bs = BM * BK * sizeof(__nv_bfloat16);
    size_t off_Cacc = off_Bs + BN * BK * sizeof(__nv_bfloat16);
    size_t off_tmema = (off_Cacc + BM * BN * sizeof(float) + 15) & ~15ULL;
    size_t off_barrier_a = off_tmema + 4;
    size_t off_barrier_b = off_barrier_a + 8;

    __nv_bfloat16* __restrict__ As = reinterpret_cast<__nv_bfloat16*>(smem_bytes + off_As);
    __nv_bfloat16* __restrict__ Bs = reinterpret_cast<__nv_bfloat16*>(smem_bytes + off_Bs);
    float* __restrict__ Cacc = reinterpret_cast<float*>(smem_bytes + off_Cacc);
    uint32_t* __restrict__ tmema_slot = reinterpret_cast<uint32_t*>(smem_bytes + off_tmema);
    uint64_t* __restrict__ barA = reinterpret_cast<uint64_t*>(smem_bytes + off_barrier_a);
    uint64_t* __restrict__ barB = reinterpret_cast<uint64_t*>(smem_bytes + off_barrier_b);

    int m_start = blockIdx.x * BM;
    int n_start = blockIdx.y * BN;

    // Init mbarriers (thread 0 only, then fence)
    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;"
            :: "r"((uint32_t)__cvta_generic_to_shared(barA)), "r"(1));
        asm volatile("mbarrier.init.shared.b64 [%0], %1;"
            :: "r"((uint32_t)__cvta_generic_to_shared(barB)), "r"(1));
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    // Allocate Tensor Memory: BM cols for the C accumulator (warp-group sync required)
    // tcgen05.alloc requires a full warp in warpgroup. All warps in warpgroup execute together.
    if (tid < WARPGROUP_SIZE) {
        uint32_t ncols = (BN + 31) / 32 * 32;  // Round up to multiple of 32, power of 2
        // Find smallest power-of-2 multiple of 32 >= BN
        while (ncols > 0 && (ncols & (ncols - 1)) != 0) ncols &= ncols - 1;
        ncols *= 2;
        if (ncols < 32) ncols = 32;
        if (ncols > 512) ncols = 512;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"((uint32_t)__cvta_generic_to_shared(tmema_slot)), "r"(ncols));
    }
    __syncthreads();
    uint32_t tmem_base = tmema_slot[0];

    // Instruction descriptor: BF16*BF16->FP32, K-major, cta_group::1
    uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10) |
                     ((BN >> 3) & 0x3Fu) << 17 |
                     ((BM >> 4) & 0x1Fu) << 24;

    // Descriptor params for K-major non-swizzled: SBO=128, LBO=(BM/8)*128
    uint32_t sbo_val = 128u;
    uint32_t lbo_val = (uint32_t)((BM >> 3) * 128u);

    int num_k_tiles = K / BK;

    for (int ki = 0; ki < num_k_tiles; ki++) {
        int kg = ki * BK;

        // ---- Load A tile ----
        if (tid < BM) {
            int m = tid;
            if (m_start + m < M) {
                const __nv_bfloat16* row = A + (size_t)(m_start + m) * K + kg;
                #pragma unroll
                for (int k = 0; k < BK; k += 4) {
                    As[m * BK + k]     = row[k];
                    As[m * BK + k + 1] = row[k + 1];
                    As[m * BK + k + 2] = row[k + 2];
                    As[m * BK + k + 3] = row[k + 3];
                }
            }
        }

        // ---- Load B tile ----
        if (tid >= BM) {
            int btid = tid - BM;
            int nbthreads = NTOTAL - BM;
            int nelem = BN * BK;
            #pragma unroll
            for (int e = btid; e < nelem; e += nbthreads) {
                int n = e / BK;
                int k = e % BK;
                if (n_start + n < N) {
                    Bs[n * BK + k] = B[(size_t)(n_start + n) * K + kg + k];
                }
            }
        }
        __syncthreads();

        // ---- Issue UMMA: 2 sub-iterations per K tile (each handles K=8) ----
        bool first_umma = (ki == 0);
        
        #pragma unroll
        for (int sub = 0; sub < BK / 8; sub++) {
            uint32_t k_off = sub * 8;
            
            __nv_bfloat16* a_base = As + k_off * BM;
            __nv_bfloat16* b_base = Bs + k_off * BN;
            
            uint32_t asaddr = (uint32_t)__cvta_generic_to_shared(a_base);
            uint32_t bsaddr = (uint32_t)__cvta_generic_to_shared(b_base);
            
            uint64_t adesc = make_smem_desc(asaddr, lbo_val, sbo_val, 0);
            uint64_t bdesc = make_smem_desc(bsaddr, lbo_val, sbo_val, 0);
            
            uint32_t accum_val = (first_umma && sub == 0) ? 0u : 1u;
            
            if (tid == 0) {
                asm volatile(
                    "{\n\t"
                    ".reg .pred p;\n\t"
                    "setp.ne.s32 p, %4, 0;\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
                    "}\n\t"
                    : : "r"(tmem_base), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accum_val));
            }
        }

        // Commit and wait for UMMA completion via mbarrier
        if (tid == 0) {
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cta.b64 [%0];"
                :: "r"((uint32_t)__cvta_generic_to_shared(barA)));
        }
        
        // Wait for UMMA
        {
            asm volatile(
                "{\n\t.reg .pred P;\n\t"
                "WAIT_UMA%=: ;\n\t"
                "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n\t"
                "@!P bra WAIT_UMA%=;\n\t"
                "}"
                :: "r"((uint32_t)__cvta_generic_to_shared(barA)), "r"(ki & 1));
        }

        // ---- Read C accumulator from TMEM and store to shared memory ----
        // Warpgroup: warp i accesses lanes i*32..(i+1)*32-1 of TMEM
        // Each thread loads from its TMEM column position
        if (tid < WARPGROUP_SIZE) {
            int wg_warp_id = warp_id;           // 0-3
            int lane_in_tmem = wg_warp_id * 32 + lane_id;  // 0-127
            
            // Check bounds
            if (lane_in_tmem < BN) {
                // Number of FP32 values per TMEM column to load depends on BM
                // Each TMEM column has 128 lanes (rows). We allocated BN columns.
                // TMEM address = tmem_base + (lane << 16) | col
                // Row index in output = lane_in_tmem maps to output column (BN dimension)
                // We need to load BM values for this column
                
                uint32_t tmem_col = lane_in_tmem;
                
                #pragma unroll
                for (int row = tid; row < BM; row += WARPGROUP_SIZE) {
                    // Build TMEM address: (row << 16) | col
                    uint32_t taddr = (uint32_t)((row << 16) | tmem_col);
                    taddr += tmem_base;
                    
                    float val;
                    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
                        : "=r"(val) : "r"(taddr));
                    
                    if (m_start + row < M && n_start + lane_in_tmem < N) {
                        Cacc[row * BN + lane_in_tmem] = val;
                    }
                }
            }
        }
        
        // Fence for LD completion
        if (tid < WARPGROUP_SIZE) {
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        }
        
        __syncthreads();
        
        // ---- Write Cacc to global memory (all threads) ----
        if (ki == num_k_tiles - 1) {
            int total_out = BM * BN;
            #pragma unroll
            for (int idx = tid; idx < total_out; idx += NTOTAL) {
                int lm = idx / BN;
                int ln = idx % BN;
                int gm = m_start + lm;
                int gn = n_start + ln;
                if (gm < M && gn < N) {
                    C[(size_t)gm * N + gn] = __float2bfloat16(Cacc[idx]);
                }
            }
        } else {
            // Zero Cacc for next iteration's accumulation... 
            // Actually UMMA with accum=1 will add to TMEM which persists
            // We need to track Cacc separately or reset TMEM
            // For simplicity, accumulate in Cacc SMEM and pass to TMEM on next iter
            // This complicates things. Let me restructure: accumulate Cacc in-place
            // and only write out on final iteration.
        }
        
        __syncthreads();
    }

    // Deallocate TMEM
    if (tid < WARPGROUP_SIZE) {
        uint32_t ncols = (BN + 31) / 32 * 32;
        while (ncols > 0 && (ncols & (ncols - 1)) != 0) ncols &= ncols - 1;
        ncols *= 2;
        if (ncols < 32) ncols = 32;
        if (ncols > 512) ncols = 512;
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :: "r"(tmem_base), "r"(ncols));
    }
}

void run(tvm::ffi::TensorView A, tvm::ffi::TensorView B, tvm::ffi::TensorView C) {
    CUDA_CHECK(cudaSetDevice(A.device().device_id));
    
    int64_t M = A.size(0);
    constexpr int64_t N = 7168;
    constexpr int64_t K = 5120;
    
    constexpr int BM = 64;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int NTOTAL = 256;
    
    dim3 block(NTOTAL);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    
    size_t smem_size = BM * BK * sizeof(__nv_bfloat16) +
                       BN * BK * sizeof(__nv_bfloat16) +
                       BM * BN * sizeof(float) +
                       4 + 8 + 8;  // tmema slot + 2 barriers
    
    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(A.device().device_type, A.device().device_id));
    
    gemm_kernel<BM, BN, BK><<<grid, block, smem_size, stream>>>(
        static_cast<const __nv_bfloat16*>(A.data_ptr()),
        static_cast<const __nv_bfloat16*>(B.data_ptr()),
        static_cast<__nv_bfloat16*>(C.data_ptr()),
        static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, gemm_blackwell::run);

} // namespace gemm_blackwell