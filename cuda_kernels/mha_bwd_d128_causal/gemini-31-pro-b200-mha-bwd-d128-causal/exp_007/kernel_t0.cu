#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

__device__ __forceinline__ void tmem_alloc_cg1_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_cg1_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ uint64_t make_smem_desc_sm100_fn(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;
    d |= (uint64_t)0 << 61; // SWIZZLE_NONE
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

__device__ __forceinline__ void umma_f16_cg1_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void umma_commit_1sm_fn(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(a));
}

__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
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

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    __nv_bfloat16 a = __float2bfloat16(__uint_as_float(fp32_a));
    __nv_bfloat16 b = __float2bfloat16(__uint_as_float(fp32_b));
    uint32_t result;
    asm("mov.b32 %0, {%1, %2};" : "=r"(result) : "h"(*reinterpret_cast<uint16_t*>(&a)), "h"(*reinterpret_cast<uint16_t*>(&b)));
    return result;
}

__device__ __forceinline__ void load_global_to_smem_64x128(const __nv_bfloat16* global_ptr, __nv_bfloat16* smem_ptr, int valid_rows, long stride_row) {
    int tid = threadIdx.x;
    for (int idx = tid; idx < 64 * 16; idx += blockDim.x) {
        int r = idx / 16;
        int c = (idx % 16) * 8;
        if (r < valid_rows) {
            float4 val = *reinterpret_cast<const float4*>(global_ptr + r * stride_row + c);
            *reinterpret_cast<float4*>(smem_ptr + r * 128 + c) = val;
        } else {
            *reinterpret_cast<float4*>(smem_ptr + r * 128 + c) = make_float4(0, 0, 0, 0);
        }
    }
}

__device__ __forceinline__ void load_global_to_smem_64(const float* global_ptr, float* smem_ptr, int valid_rows, long stride_row) {
    int tid = threadIdx.x;
    for (int idx = tid; idx < 64; idx += blockDim.x) {
        if (idx < valid_rows) {
            smem_ptr[idx] = global_ptr[idx * stride_row];
        } else {
            smem_ptr[idx] = 0.0f;
        }
    }
}

__global__ void PrecomputeDKernel(const __nv_bfloat16* O, const __nv_bfloat16* dO, float* D, int S, int d,
                                  long o_s0, long o_s1, long o_s2, long o_s3,
                                  long do_s0, long do_s1, long do_s2, long do_s3) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int seq = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq < S) {
        float sum = 0;
        const __nv_bfloat16* o_ptr = O + b * o_s0 + h * o_s1 + seq * o_s2;
        const __nv_bfloat16* do_ptr = dO + b * do_s0 + h * do_s1 + seq * do_s2;
        for (int i = 0; i < d; ++i) {
            sum += __bfloat162float(o_ptr[i * o_s3]) * __bfloat162float(do_ptr[i * do_s3]);
        }
        D[b * gridDim.y * S + h * S + seq] = sum;
    }
}

extern __shared__ char smem[];

__global__ void MhaBwdKernel(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* dO, const float* L, const float* D,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int S, int d,
    long q_s0, long q_s1, long q_s2, long q_s3,
    long k_s0, long k_s1, long k_s2, long k_s3,
    long v_s0, long v_s1, long v_s2, long v_s3,
    long do_s0, long do_s1, long do_s2, long do_s3,
    long l_s0, long l_s1, long l_s2,
    long dq_s0, long dq_s1, long dq_s2, long dq_s3,
    long dk_s0, long dk_s1, long dk_s2, long dk_s3,
    long dv_s0, long dv_s1, long dv_s2, long dv_s3
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int j = blockIdx.x;
    
    int num_j_blocks = (S + 63) / 64;
    if (j >= num_j_blocks) return;
    
    int valid_j = min(64, S - j * 64);
    
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)smem;
    __nv_bfloat16* smem_K = smem_Q + 64 * 128;
    __nv_bfloat16* smem_V = smem_K + 64 * 128;
    __nv_bfloat16* smem_dO = smem_V + 64 * 128;
    __nv_bfloat16* smem_P = smem_dO + 64 * 128;
    __nv_bfloat16* smem_dS = smem_P + 64 * 64;
    float* smem_L = (float*)(smem_dS + 64 * 64);
    float* smem_D = smem_L + 64;
    uint64_t* mbar = (uint64_t*)(smem_D + 64);
    uint32_t* tmem_alloc_addr = (uint32_t*)(mbar + 1);

    const __nv_bfloat16* K_base = K + b * k_s0 + h * k_s1 + j * 64 * k_s2;
    const __nv_bfloat16* V_base = V + b * v_s0 + h * v_s1 + j * 64 * v_s2;
    
    load_global_to_smem_64x128(K_base, smem_K, valid_j, k_s2);
    load_global_to_smem_64x128(V_base, smem_V, valid_j, v_s2);
    
    uint32_t tmem_base;
    if (threadIdx.x == 0) tmem_alloc_cg1_fn(tmem_alloc_addr, 512);
    __syncthreads();
    tmem_base = *tmem_alloc_addr;
    
    uint32_t tmem_S  = tmem_base;
    uint32_t tmem_dP = tmem_base + 64;
    uint32_t tmem_dQ = tmem_base + 128;
    uint32_t tmem_dV = tmem_base + 256;
    uint32_t tmem_dK = tmem_base + 384;
    
    if (threadIdx.x == 0) init_smem_barrier_fn(mbar, 1);
    fence_smem_barrier_init_fn();
    __syncthreads();
    
    float scale = 1.0f / sqrtf((float)d);
    int phase = 0;
    
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int row_idx = warp_id * 32 + lane_id;

    for (int i = j; i < num_j_blocks; ++i) {
        int valid_i = min(64, S - i * 64);
        
        const __nv_bfloat16* Q_base = Q + b * q_s0 + h * q_s1 + i * 64 * q_s2;
        const __nv_bfloat16* dO_base = dO + b * do_s0 + h * do_s1 + i * 64 * do_s2;
        const float* L_base = L + b * l_s0 + h * l_s1 + i * 64 * l_s2;
        const float* D_base = D + b * gridDim.y * S + h * S + i * 64;
        
        __syncthreads();
        load_global_to_smem_64x128(Q_base, smem_Q, valid_i, q_s2);
        load_global_to_smem_64x128(dO_base, smem_dO, valid_i, do_s2);
        load_global_to_smem_64(L_base, smem_L, valid_i, l_s2);
        load_global_to_smem_64(D_base, smem_D, valid_i, 1);
        __syncthreads();
        
        if (threadIdx.x == 0) {
            uint32_t a_ptr = (uint32_t)__cvta_generic_to_shared(smem_Q);
            uint32_t b_ptr = (uint32_t)__cvta_generic_to_shared(smem_K);
            for (int k = 0; k < 128; k += 16) {
                uint64_t desc_a = make_smem_desc_sm100_fn((void*)a_ptr, 1024, 128);
                uint64_t desc_b = make_smem_desc_sm100_fn((void*)b_ptr, 1024, 128);
                uint32_t idesc = make_instr_desc_fn(64, 64, 0, 0);
                umma_f16_cg1_fn(tmem_S, desc_a, desc_b, idesc, k > 0 ? 1 : 0);
                a_ptr += 32;
                b_ptr += 32;
            }
            
            a_ptr = (uint32_t)__cvta_generic_to_shared(smem_dO);
            b_ptr = (uint32_t)__cvta_generic_to_shared(smem_V);
            for (int k = 0; k < 128; k += 16) {
                uint64_t desc_a = make_smem_desc_sm100_fn((void*)a_ptr, 1024, 128);
                uint64_t desc_b = make_smem_desc_sm100_fn((void*)b_ptr, 1024, 128);
                uint32_t idesc = make_instr_desc_fn(64, 64, 0, 0);
                umma_f16_cg1_fn(tmem_dP, desc_a, desc_b, idesc, k > 0 ? 1 : 0);
                a_ptr += 32;
                b_ptr += 32;
            }
            umma_commit_1sm_fn(mbar);
        }
        
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        if (warp_id < 2) {
            float L_val = smem_L[row_idx];
            float D_val = smem_D[row_idx];
            for (int c = 0; c < 64; c += 4) {
                uint32_t sr0, sr1, sr2, sr3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(sr0),"=r"(sr1),"=r"(sr2),"=r"(sr3) : "r"(tmem_S + c));
                uint32_t dpr0, dpr1, dpr2, dpr3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(dpr0),"=r"(dpr1),"=r"(dpr2),"=r"(dpr3) : "r"(tmem_dP + c));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                float s0 = __uint_as_float(sr0) * scale;
                float s1 = __uint_as_float(sr1) * scale;
                float s2 = __uint_as_float(sr2) * scale;
                float s3 = __uint_as_float(sr3) * scale;
                
                int global_row = i * 64 + row_idx;
                int global_col = j * 64 + c;
                
                if (row_idx >= valid_i || c + 0 >= valid_j || global_col + 0 > global_row) s0 = -INFINITY;
                if (row_idx >= valid_i || c + 1 >= valid_j || global_col + 1 > global_row) s1 = -INFINITY;
                if (row_idx >= valid_i || c + 2 >= valid_j || global_col + 2 > global_row) s2 = -INFINITY;
                if (row_idx >= valid_i || c + 3 >= valid_j || global_col + 3 > global_row) s3 = -INFINITY;
                
                float p0 = fast_exp2f_fn((s0 - L_val) * 1.44269504089f);
                float p1 = fast_exp2f_fn((s1 - L_val) * 1.44269504089f);
                float p2 = fast_exp2f_fn((s2 - L_val) * 1.44269504089f);
                float p3 = fast_exp2f_fn((s3 - L_val) * 1.44269504089f);
                
                uint32_t p_packed0 = pack_bf16_fn(__float_as_uint(p0), __float_as_uint(p1));
                uint32_t p_packed1 = pack_bf16_fn(__float_as_uint(p2), __float_as_uint(p3));
                
                uint32_t p_offset = (row_idx * 64 + c) * 2;
                *reinterpret_cast<uint32_t*>((char*)smem_P + p_offset) = p_packed0;
                *reinterpret_cast<uint32_t*>((char*)smem_P + p_offset + 4) = p_packed1;
                
                float dp0 = __uint_as_float(dpr0);
                float dp1 = __uint_as_float(dpr1);
                float dp2 = __uint_as_float(dpr2);
                float dp3 = __uint_as_float(dpr3);
                
                float ds0 = p0 * (dp0 - D_val) * scale;
                float ds1 = p1 * (dp1 - D_val) * scale;
                float ds2 = p2 * (dp2 - D_val) * scale;
                float ds3 = p3 * (dp3 - D_val) * scale;
                
                uint32_t ds_packed0 = pack_bf16_fn(__float_as_uint(ds0), __float_as_uint(ds1));
                uint32_t ds_packed1 = pack_bf16_fn(__float_as_uint(ds2), __float_as_uint(ds3));
                
                uint32_t ds_offset = (row_idx * 64 + c) * 2;
                *reinterpret_cast<uint32_t*>((char*)smem_dS + ds_offset) = ds_packed0;
                *reinterpret_cast<uint32_t*>((char*)smem_dS + ds_offset + 4) = ds_packed1;
            }
        }
        
        __syncthreads();
        
        if (threadIdx.x == 0) {
            uint32_t a_ptr = (uint32_t)__cvta_generic_to_shared(smem_P);
            uint32_t b_ptr = (uint32_t)__cvta_generic_to_shared(smem_dO);
            for (int k = 0; k < 64; k += 16) {
                uint64_t desc_a = make_smem_desc_sm100_fn((void*)a_ptr, 128, 1024);
                uint64_t desc_b = make_smem_desc_sm100_fn((void*)b_ptr, 128, 1024);
                uint32_t idesc = make_instr_desc_fn(64, 128, 1, 1);
                umma_f16_cg1_fn(tmem_dV, desc_a, desc_b, idesc, (i == j && k == 0) ? 0 : 1);
                a_ptr += 2048;
                b_ptr += 4096;
            }
            
            a_ptr = (uint32_t)__cvta_generic_to_shared(smem_dS);
            b_ptr = (uint32_t)__cvta_generic_to_shared(smem_Q);
            for (int k = 0; k < 64; k += 16) {
                uint64_t desc_a = make_smem_desc_sm100_fn((void*)a_ptr, 128, 1024);
                uint64_t desc_b = make_smem_desc_sm100_fn((void*)b_ptr, 128, 1024);
                uint32_t idesc = make_instr_desc_fn(64, 128, 1, 1);
                umma_f16_cg1_fn(tmem_dK, desc_a, desc_b, idesc, (i == j && k == 0) ? 0 : 1);
                a_ptr += 2048;
                b_ptr += 4096;
            }
            
            a_ptr = (uint32_t)__cvta_generic_to_shared(smem_dS);
            b_ptr = (uint32_t)__cvta_generic_to_shared(smem_K);
            for (int k = 0; k < 64; k += 16) {
                uint64_t desc_a = make_smem_desc_sm100_fn((void*)a_ptr, 1024, 128);
                uint64_t desc_b = make_smem_desc_sm100_fn((void*)b_ptr, 128, 1024);
                uint32_t idesc = make_instr_desc_fn(64, 128, 0, 1);
                umma_f16_cg1_fn(tmem_dQ, desc_a, desc_b, idesc, (k == 0) ? 0 : 1);
                a_ptr += 32;
                b_ptr += 4096;
            }
            
            umma_commit_1sm_fn(mbar);
        }
        
        mbarrier_wait_fn(mbar, phase);
        phase ^= 1;
        
        if (warp_id < 2) {
            for (int c = 0; c < 128; c += 4) {
                uint32_t dqr0, dqr1, dqr2, dqr3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(dqr0),"=r"(dqr1),"=r"(dqr2),"=r"(dqr3) : "r"(tmem_dQ + c));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                int global_row = i * 64 + row_idx;
                if (global_row < S && c < 128 && row_idx < valid_i) {
                    __nv_bfloat162* out_ptr = reinterpret_cast<__nv_bfloat162*>(dQ + b * dq_s0 + h * dq_s1 + global_row * dq_s2 + c);
                    atomicAdd(out_ptr, __floats2bfloat162_rn(__uint_as_float(dqr0), __uint_as_float(dqr1)));
                    atomicAdd(out_ptr + 1, __floats2bfloat162_rn(__uint_as_float(dqr2), __uint_as_float(dqr3)));
                }
            }
        }
        __syncthreads();
    }
    
    if (warp_id < 2) {
        for (int c = 0; c < 128; c += 4) {
            uint32_t dkr0, dkr1, dkr2, dkr3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(dkr0),"=r"(dkr1),"=r"(dkr2),"=r"(dkr3) : "r"(tmem_dK + c));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            int global_row = j * 64 + row_idx;
            if (global_row < S && c < 128 && row_idx < valid_j) {
                __nv_bfloat162* out_ptr = reinterpret_cast<__nv_bfloat162*>(dK + b * dk_s0 + h * dk_s1 + global_row * dk_s2 + c);
                *out_ptr = __floats2bfloat162_rn(__uint_as_float(dkr0), __uint_as_float(dkr1));
                *(out_ptr + 1) = __floats2bfloat162_rn(__uint_as_float(dkr2), __uint_as_float(dkr3));
            }
        }
        
        for (int c = 0; c < 128; c += 4) {
            uint32_t dvr0, dvr1, dvr2, dvr3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                         : "=r"(dvr0),"=r"(dvr1),"=r"(dvr2),"=r"(dvr3) : "r"(tmem_dV + c));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            int global_row = j * 64 + row_idx;
            if (global_row < S && c < 128 && row_idx < valid_j) {
                __nv_bfloat162* out_ptr = reinterpret_cast<__nv_bfloat162*>(dV + b * dv_s0 + h * dv_s1 + global_row * dv_s2 + c);
                *out_ptr = __floats2bfloat162_rn(__uint_as_float(dvr0), __uint_as_float(dvr1));
                *(out_ptr + 1) = __floats2bfloat162_rn(__uint_as_float(dvr2), __uint_as_float(dvr3));
            }
        }
    }
    __syncthreads();
    
    if (threadIdx.x == 0) tmem_dealloc_cg1_fn(tmem_base, 512);
}

namespace tvm_ffi_mha_bwd {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    int B = Q.size(0);
    int H = Q.size(1);
    int S = Q.size(2);
    int d = Q.size(3);
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    
    CUDA_CHECK(cudaMemsetAsync(dQ.data_ptr(), 0, B * H * S * d * sizeof(uint16_t), stream));
    CUDA_CHECK(cudaMemsetAsync(dK.data_ptr(), 0, B * H * S * d * sizeof(uint16_t), stream));
    CUDA_CHECK(cudaMemsetAsync(dV.data_ptr(), 0, B * H * S * d * sizeof(uint16_t), stream));

    float* D_workspace = nullptr;
    CUDA_CHECK(cudaMallocAsync(&D_workspace, B * H * S * sizeof(float), stream));

    dim3 grid_D((S + 255) / 256, H, B);
    PrecomputeDKernel<<<grid_D, 256, 0, stream>>>(
        (const __nv_bfloat16*)O.data_ptr(), (const __nv_bfloat16*)dO.data_ptr(), D_workspace, S, d,
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3)
    );
    CUDA_CHECK(cudaGetLastError());
    
    dim3 grid_mha((S + 63) / 64, H, B);
    dim3 block_mha(128);
    int smem_size = 96 * 1024;
    CUDA_CHECK(cudaFuncSetAttribute(MhaBwdKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    
    MhaBwdKernel<<<grid_mha, block_mha, smem_size, stream>>>(
        (const __nv_bfloat16*)Q.data_ptr(), (const __nv_bfloat16*)K.data_ptr(), (const __nv_bfloat16*)V.data_ptr(),
        (const __nv_bfloat16*)dO.data_ptr(), (const float*)L.data_ptr(), D_workspace,
        (__nv_bfloat16*)dQ.data_ptr(), (__nv_bfloat16*)dK.data_ptr(), (__nv_bfloat16*)dV.data_ptr(),
        S, d,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3)
    );
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaFreeAsync(D_workspace, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_mha_bwd::run);

} // namespace tvm_ffi_mha_bwd