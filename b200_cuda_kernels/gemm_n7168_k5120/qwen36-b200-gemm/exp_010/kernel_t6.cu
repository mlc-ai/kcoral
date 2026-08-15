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

__device__ __forceinline__ uint64_t make_smem_desc_kmajor_noswizzle(uint32_t saddr, int rows) {
    // K-major, no swizzle: SBO=128, LBO=(rows/8)*128
    // Each UMMA consumes 8 bf16 elements of K (K=8 atom)
    uint32_t sbo_val = 128u;
    uint32_t lbo_val = (uint32_t)((rows >> 3) * 128u);
    uint64_t d = 0;
    d |= ((uint64_t)(saddr & 0x3FFFFu) >> 4);
    d |= ((uint64_t)((lbo_val & 0x3FFFFu) >> 4) << 16);
    d |= ((uint64_t)((sbo_val & 0x3FFFFu) >> 4) << 32);
    d |= (uint64_t)1 << 46;   // version=1
    // swizzle mode = 0 (no swizzle)
    return d;
}

template <int BM, int BN, int BK>
__global__ void gemm_kernel(
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    __nv_bfloat16* C,
    int M, int N, int K) 
{
    constexpr int NTOTAL = 256;
    constexpr int WG_SIZE = 128;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Shared memory: As[BM][BK] + Bs[BN][BK] + barA + barB + tmem_addr
    extern __shared__ char smem_bytes[];

    size_t off_As = 0;
    size_t off_Bs = BM * BK * sizeof(__nv_bfloat16);
    size_t off_barA = off_Bs + BN * BK * sizeof(__nv_bfloat16);
    size_t off_barB = off_barA + 8;
    size_t off_tmemb = (off_barB + 8 + 15) & ~15ULL;

    __nv_bfloat16* __restrict__ As = reinterpret_cast<__nv_bfloat16*>(smem_bytes + off_As);
    __nv_bfloat16* __restrict__ Bs = reinterpret_cast<__nv_bfloat16*>(smem_bytes + off_Bs);
    uint64_t* __restrict__ barA = reinterpret_cast<uint64_t*>(smem_bytes + off_barA);
    uint64_t* __restrict__ barB = reinterpret_cast<uint64_t*>(smem_bytes + off_barB);
    uint32_t* __restrict__ tmem_slot = reinterpret_cast<uint32_t*>(smem_bytes + off_tmemb);

    int m_start = blockIdx.x * BM;
    int n_start = blockIdx.y * BN;

    // Init mbarriers
    if (tid == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;"
            :: "r"((uint32_t)__cvta_generic_to_shared(barA)), "r"(1));
        asm volatile("mbarrier.init.shared.b64 [%0], %1;"
            :: "r"((uint32_t)__cvta_generic_to_shared(barB)), "r"(1));
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    // Allocate TMEM: BN=128 columns (power of 2, multiple of 32). Warpgroup sync required.
    if (tid < WG_SIZE) {
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
            :: "r"((uint32_t)__cvta_generic_to_shared(tmem_slot)), "r"(128));
    }
    __syncthreads();
    
    uint32_t tmem_base = tmem_slot[0];

    // Instruction descriptor: BF16*BF16->FP32, K-major
    uint32_t idesc = 0;
    idesc |= (1u << 4);           // dtype=F32
    idesc |= (1u << 7);           // a_format=BF16
    idesc |= (1u << 10);          // b_format=BF16
    idesc |= ((BN >> 3) & 0x3Fu) << 17;  // N>>3
    idesc |= ((BM >> 4) & 0x1Fu) << 24;  // M>>4

    int num_k_tiles = K / BK;

    for (int ki = 0; ki < num_k_tiles; ki++) {
        int kg = ki * BK;

        // ---- Load A tile [BM][BK] into SMEM ----
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

        // ---- Load B tile [BN][BK] into SMEM ----
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

        // ---- Issue UMMA sub-tiles ----
        // Each UMMA contracts 8 bf16-cols of K. BK=16 => 2 sub-iters.
        bool first_mma = (ki == 0);

        #pragma unroll
        for (int sub = 0; sub < BK / 8; sub++) {
            int k_off = sub * 8;
            
            __nv_bfloat16* aptr = As + k_off * BM;
            __nv_bfloat16* bptr = Bs + k_off * BN;
            
            uint32_t asaddr = (uint32_t)__cvta_generic_to_shared(aptr);
            uint32_t bsaddr = (uint32_t)__cvta_generic_to_shared(bptr);
            
            uint64_t adesc = make_smem_desc_kmajor_noswizzle(asaddr, BM);
            uint64_t bdesc = make_smem_desc_kmajor_noswizzle(bsaddr, BN);
            
            // enable-input-d: false=c-clear accumulator, true=accumulate
            uint32_t accum_flag = (first_mma && sub == 0) ? 0u : 1u;
            
            if (tid == 0) {
                asm volatile(
                    "{\n\t"
                    ".reg .pred p;\n\t"
                    "setp.ne.s32 p, %4, 0;\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
                    "}\n\t"
                    : : "r"(tmem_base), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(accum_flag)
                    : "memory");
            }
        }
        
        // Commit UMMA async ops -> mbarrier
        if (tid == 0) {
            uint32_t ba = (uint32_t)__cvta_generic_to_shared(barA);
            asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared.b64 [%0];"
                :: "r"(ba) : "memory");
        }
        
        // Wait for UMMA via mbarrier
        {
            uint32_t parity = ki & 1;
            uint32_t ba = (uint32_t)__cvta_generic_to_shared(barA);
            asm volatile(
                "{\n\t.reg .pred P;\n\t"
                "WAIT_U%=: ;\n\t"
                "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n\t"
                "@!P bra WAIT_U%=;\n\t"
                "}"
                :: "r"(ba), "r"(parity) : "memory");
        }
        
        // ---- Epilogue: read TMEM -> convert -> write global ----
        // Only on last K iteration
        if (ki == num_k_tiles - 1) {
            __syncthreads();
            
            // Warpgroup reads TMEM row by row
            // Each warp handles 32 consecutive output rows
            // Within a warp, each lane loads data for its assigned row-column
            if (tid < WG_SIZE) {
                // wg_tid = 0..127 maps to TMEM lanes 0..127 (rows)
                // But we only have BM=64 rows, so warps 0-1 fully participate, warps 2-3 partial
                
                int my_row = tid;  // 0..127 but valid for 0..63
                
                if (my_row < BM) {
                    int gm = m_start + my_row;
                    if (gm < M) {
                        // Load entire row of BN cols from TMEM lane=my_row
                        // Use .64x128b shape = 64 lanes x 128 bits, needs collective warp
                        // Instead use smaller shapes iterated
                        
                        // For simplicity: load 2 values at a time (.16x32bx2 shape or two .32x32b)
                        // Actually .32x32b.x1 loads 32-bits from 1 location
                        // Shape .16x64b could work too
                        
                        // Let's just iterate column by column with simple ld
                        for (int col = lane_id; col < BN; col += 32) {
                            if (n_start + col >= N) break;
                            
                            // TMEM address = base + (row<<16)|col
                            uint32_t taddr = tmem_base + ((my_row << 16) | col);
                            
                            uint32_t reg_val;
                            // .32x32b = single 32-bit value from specified lane+col
                            asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
                                : "=r"(reg_val) : "r"(taddr));
                            
                            float fval = __uint_as_float(reg_val);
                            C[(size_t)gm * N + (size_t)(n_start + col)] = __float2bfloat16(fval);
                        }
                    }
                }
            }
            
            // Wait for all LD completions within warpgroup
            if (tid < WG_SIZE) {
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            }
            __syncthreads();
        }
    }

    // Deallocate TMEM
    if (tid < WG_SIZE) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
            :: "r"(tmem_base), "r"(128));
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
                       8 + 8 + 16;  // barA + barB + aligned tmem slot
    
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