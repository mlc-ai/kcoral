#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <stdio.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>
#include <math.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error %s at %s:%d\n",               \
                cudaGetErrorString(_e), __FILE__, __LINE__);       \
        exit(1);                                                   \
    }                                                              \
} while(0)

// Helper: 128B Software Swizzle Address
__device__ __forceinline__ int get_swizzled_col_128B(int row, int col) {
    int chunk = col / 8;
    int chunk_mod = chunk % 8;
    int swizzled_chunk = (chunk - chunk_mod) + ((row % 8) ^ chunk_mod);
    return swizzled_chunk * 8 + (col % 8);
}

// SM100 descriptor builders
__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)(1) << 16;
    d |= (uint64_t)(1024 >> 4) << 32;
    d |= (uint64_t)1 << 46;
    uint64_t base_offset = (addr >> 7) & 0x7;
    d |= base_offset << 49;
    d |= (uint64_t)2 << 61;
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn() {
    uint32_t d = 0;
    d |= (1u << 4);    // FP32 accum
    d |= (1u << 7);    // BF16 A
    d |= (1u << 10);   // BF16 B
    d |= (0u << 15);   // transA = 0
    d |= (0u << 16);   // transB = 0
    d |= ((128 / 8) << 17); // N = 128
    d |= ((128 / 16) << 24); // M = 128
    return d;
}

// MMA 128x128 Issuer (Advances K by 128 elements over 8 MMAs)
__device__ void issue_mma_128(uint32_t tmem_c, void* ptr_a, void* ptr_b, uint32_t idesc, bool accumulate) {
    for (int k = 0; k < 8; ++k) {
        void* p_a = (void*)((char*)ptr_a + k * 32);     // A advances in columns (K=16) -> 32 bytes
        void* p_b = (void*)((char*)ptr_b + k * 4096);   // B advances in rows (K=16) -> 16 * 128 * 2 = 4096 bytes
        uint64_t a_k = make_smem_desc_sm100_fn(p_a);
        uint64_t b_k = make_smem_desc_sm100_fn(p_b);
        uint32_t acc = (k == 0 && !accumulate) ? 0 : 1;
        
        asm volatile(
            "{\n.reg .pred p;\n"
            "setp.ne.b32 p, %4, 0;\n"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
            :: "r"(tmem_c), "l"(a_k), "l"(b_k), "r"(idesc), "r"(acc));
    }
}

__device__ void load_tmem_row_to_regs(uint32_t start_col, float* regs) {
    for (int col = 0; col < 128; col += 8) {
        uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
           : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3),"=r"(r4),"=r"(r5),"=r"(r6),"=r"(r7) 
           : "r"(start_col + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        regs[col+0] = __uint_as_float(r0);
        regs[col+1] = __uint_as_float(r1);
        regs[col+2] = __uint_as_float(r2);
        regs[col+3] = __uint_as_float(r3);
        regs[col+4] = __uint_as_float(r4);
        regs[col+5] = __uint_as_float(r5);
        regs[col+6] = __uint_as_float(r6);
        regs[col+7] = __uint_as_float(r7);
    }
}

__device__ void load_gmem_to_smem_swizzle(const __nv_bfloat16* gmem, __nv_bfloat16* smem, int stride) {
    int tid = threadIdx.x;
    for (int i = 0; i < 128; i += 8) {
        int row = (tid / 16) + i;
        int col = (tid % 16) * 8;
        if (row < 128) {
            ulonglong2 val = *(ulonglong2*)&gmem[row * stride + col];
            __nv_bfloat16* vals = (__nv_bfloat16*)&val;
            for (int k = 0; k < 8; ++k) {
                smem[row * 128 + get_swizzled_col_128B(row, col + k)] = vals[k];
            }
        }
    }
}

__device__ void load_gmem_to_smem_swizzle_transposed(const __nv_bfloat16* gmem, __nv_bfloat16* smem, int stride) {
    int tid = threadIdx.x;
    for (int i = 0; i < 128; i += 8) {
        int row = (tid / 16) + i;
        int col = (tid % 16) * 8;
        if (row < 128) {
            ulonglong2 val = *(ulonglong2*)&gmem[row * stride + col];
            __nv_bfloat16* vals = (__nv_bfloat16*)&val;
            for (int k = 0; k < 8; ++k) {
                int t_row = col + k;
                int t_col = row;
                smem[t_row * 128 + get_swizzled_col_128B(t_row, t_col)] = vals[k];
            }
        }
    }
}

__device__ void write_regs_to_smem_P_and_P_T(float* regs, __nv_bfloat16* smem_P, __nv_bfloat16* smem_P_T) {
    int row = threadIdx.x; 
    for (int col = 0; col < 128; ++col) {
        __nv_bfloat16 val = __float2bfloat16(regs[col]);
        smem_P[row * 128 + get_swizzled_col_128B(row, col)] = val;
        int t_row = col;
        int t_col = row;
        smem_P_T[t_row * 128 + get_swizzled_col_128B(t_row, t_col)] = val;
    }
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

template <int NUM_REGS>
__device__ __forceinline__ void setmaxnreg_inc_sync_fn() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" :: "n"(NUM_REGS) : "memory");
}

__global__ void compute_D_kernel(const __nv_bfloat16* dO, const __nv_bfloat16* O, float* D, int B, int H, int S, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * H * S) {
        float sum = 0;
        for (int i = 0; i < d; ++i) {
            float o_val = __bfloat162float(O[idx * d + i]);
            float do_val = __bfloat162float(dO[idx * d + i]);
            sum += o_val * do_val;
        }
        D[idx] = sum;
    }
}

__global__ void mha_bwd_kernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* dO, const float* L, const float* D,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int B, int H, int S_seq) 
{
    setmaxnreg_inc_sync_fn<248>();
    
    int b = blockIdx.z;
    int h = blockIdx.y;
    int n_idx = blockIdx.x;

    extern __shared__ char shared_memory[];
    __nv_bfloat16* smem_K = (__nv_bfloat16*)shared_memory;
    __nv_bfloat16* smem_K_T = smem_K + 128*128;
    __nv_bfloat16* smem_V_T = smem_K_T + 128*128;
    __nv_bfloat16* smem_Q = smem_V_T + 128*128;
    __nv_bfloat16* smem_dO = smem_Q + 128*128;
    __nv_bfloat16* smem_P = smem_dO + 128*128;
    __nv_bfloat16* smem_P_T = smem_P + 128*128;
    float* smem_L = (float*)(smem_P_T + 128*128); 
    float* smem_D = smem_L + 128; 
    uint64_t* mbar = (uint64_t*)(smem_D + 128);

    uint32_t tmem_addr;
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" : : "r"((uint32_t)__cvta_generic_to_shared(&tmem_addr)));

    uint64_t b_stride = H * S_seq * 128;
    uint64_t h_stride = S_seq * 128;
    
    const __nv_bfloat16* K_ptr = K + b * b_stride + h * h_stride + n_idx * 128 * 128;
    const __nv_bfloat16* V_ptr = V + b * b_stride + h * h_stride + n_idx * 128 * 128;
    
    load_gmem_to_smem_swizzle(K_ptr, smem_K, 128);
    load_gmem_to_smem_swizzle_transposed(K_ptr, smem_K_T, 128);
    load_gmem_to_smem_swizzle_transposed(V_ptr, smem_V_T, 128);
    
    uint32_t idesc = make_instr_desc_fn();
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])));
    }
    __syncthreads();

    int phase = 0;
    float scale = 1.0f / sqrtf(128.0f);
    
    for (int m_idx = n_idx; m_idx < S_seq / 128; ++m_idx) {
        bool first_m = (m_idx == n_idx);
        
        const __nv_bfloat16* Q_ptr = Q + b * b_stride + h * h_stride + m_idx * 128 * 128;
        const __nv_bfloat16* dO_ptr = dO + b * b_stride + h * h_stride + m_idx * 128 * 128;
        
        load_gmem_to_smem_swizzle(Q_ptr, smem_Q, 128);
        load_gmem_to_smem_swizzle(dO_ptr, smem_dO, 128);
        
        if (threadIdx.x < 128) {
            smem_L[threadIdx.x] = L[b * H * S_seq + h * S_seq + m_idx * 128 + threadIdx.x];
            smem_D[threadIdx.x] = D[b * H * S_seq + h * S_seq + m_idx * 128 + threadIdx.x];
        }
        __syncthreads();
        
        // MMA 1: S = Q_m @ K_n_T
        issue_mma_128(256, smem_Q, smem_K_T, idesc, false);
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])));
        mbarrier_wait_fn(mbar, phase); phase ^= 1;
        
        float regs[128];
        load_tmem_row_to_regs(256, regs);
        
        float lse = smem_L[threadIdx.x];
        for (int col = 0; col < 128; ++col) {
            int pos_q = m_idx * 128 + threadIdx.x;
            int pos_k = n_idx * 128 + col;
            float val = regs[col] * scale;
            if (pos_q < pos_k) val = -INFINITY;
            float p_val = exp2f((val - lse) * 1.44269504f);
            if (pos_q < pos_k) p_val = 0.0f;
            regs[col] = p_val;
        }
        
        write_regs_to_smem_P_and_P_T(regs, smem_P, smem_P_T);
        __syncthreads();
        
        // MMA 2 and 3: dV_n += P_T @ dO_m | dP = dO_m @ V_n_T
        issue_mma_128(128, smem_P_T, smem_dO, idesc, !first_m); 
        issue_mma_128(256, smem_dO, smem_V_T, idesc, false); 
        
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])));
        mbarrier_wait_fn(mbar, phase); phase ^= 1;
        
        float regs_dP[128];
        load_tmem_row_to_regs(256, regs_dP);
        
        float d_val = smem_D[threadIdx.x];
        for (int col = 0; col < 128; ++col) {
            regs[col] = (regs[col] * (regs_dP[col] - d_val)) * scale;
        }
        
        write_regs_to_smem_P_and_P_T(regs, smem_P, smem_P_T); 
        __syncthreads();
        
        // MMA 4 and 5: dK_n += dS_T @ Q_m | dQ_m += dS @ K_n
        issue_mma_128(0, smem_P_T, smem_Q, idesc, !first_m);
        issue_mma_128(384, smem_P, smem_K, idesc, false); 
        
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(&mbar[0])));
        mbarrier_wait_fn(mbar, phase); phase ^= 1;
        
        float regs_dQ[128];
        load_tmem_row_to_regs(384, regs_dQ);
        
        __nv_bfloat16* dQ_ptr = dQ + b * b_stride + h * h_stride + m_idx * 128 * 128;
        for (int col = 0; col < 128; ++col) {
            __nv_bfloat16 val = __float2bfloat16(regs_dQ[col]);
            atomicAdd(&dQ_ptr[threadIdx.x * 128 + col], val);
        }
    }
    
    float regs_dK[128], regs_dV[128];
    load_tmem_row_to_regs(0, regs_dK);
    load_tmem_row_to_regs(128, regs_dV);
    
    __nv_bfloat16* dK_ptr = dK + b * b_stride + h * h_stride + n_idx * 128 * 128;
    __nv_bfloat16* dV_ptr = dV + b * b_stride + h * h_stride + n_idx * 128 * 128;
    for (int col = 0; col < 128; ++col) {
        dK_ptr[threadIdx.x * 128 + col] = __float2bfloat16(regs_dK[col]);
        dV_ptr[threadIdx.x * 128 + col] = __float2bfloat16(regs_dV[col]);
    }
    
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" :: "r"(tmem_addr));
}

namespace tvm_ffi_mha_bwd_d128_causal {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id)); 
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = Q.size(3);

    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    const __nv_bfloat16* O_ptr = static_cast<const __nv_bfloat16*>(O.data_ptr());
    const __nv_bfloat16* dO_ptr = static_cast<const __nv_bfloat16*>(dO.data_ptr());
    const float* L_ptr = static_cast<const float*>(L.data_ptr());
    __nv_bfloat16* dQ_ptr = static_cast<__nv_bfloat16*>(dQ.data_ptr());
    __nv_bfloat16* dK_ptr = static_cast<__nv_bfloat16*>(dK.data_ptr());
    __nv_bfloat16* dV_ptr = static_cast<__nv_bfloat16*>(dV.data_ptr());

    CUDA_CHECK(cudaMemsetAsync(dQ_ptr, 0, B * H * S * d * sizeof(__nv_bfloat16), stream));

    float* D_buf;
    CUDA_CHECK(cudaMallocAsync(&D_buf, B * H * S * sizeof(float), stream));
    
    int num_elements = B * H * S;
    int blocks_D = (num_elements + 255) / 256;
    compute_D_kernel<<<blocks_D, 256, 0, stream>>>(dO_ptr, O_ptr, D_buf, B, H, S, d);

    dim3 grid(S / 128, H, B);
    dim3 block(128); 
    size_t dynamic_smem = 230000;
    CUDA_CHECK(cudaFuncSetAttribute(mha_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_smem));
    
    mha_bwd_kernel<<<grid, block, dynamic_smem, stream>>>(
        Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_buf,
        dQ_ptr, dK_ptr, dV_ptr, B, H, S);
        
    CUDA_CHECK(cudaFreeAsync(D_buf, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

} // namespace tvm_ffi_mha_bwd_d128_causal