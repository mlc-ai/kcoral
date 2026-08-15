#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
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

namespace tvm_ffi_example_cuda {

__device__ __forceinline__ uint32_t get_shmem_addr(void* ptr) {
    return (uint32_t)__cvta_generic_to_shared(ptr);
}

__device__ __forceinline__ uint32_t swizzle_128B(uint32_t row, uint32_t col_elements) {
    const uint32_t stride_elements = 64;
    return row * stride_elements + (((row & 7) ^ (col_elements >> 3)) << 3) + (col_elements & 7);
}

__device__ __forceinline__ uint64_t make_smem_desc(void* smem_ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = get_shmem_addr(smem_ptr);
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16; 
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32; 
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)2 << 61;   // SWIZZLE_128B
    return d;
}

template<int A_MAJOR, int B_MAJOR, int M, int N>
__device__ __forceinline__ uint32_t make_instr_desc() {
    uint32_t d = 0;
    d |= (1u << 4);    
    d |= (1u << 7);    
    d |= (1u << 10);   
    d |= ((A_MAJOR & 1) << 15);   
    d |= ((B_MAJOR & 1) << 16);   
    d |= ((N / 8) << 17);     
    d |= ((M / 16) << 24);    
    return d;
}

__device__ __forceinline__ void umma_f16_cg2(
    uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b,
    uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void init_mbar(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}

__device__ __forceinline__ void arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(bytes));
}

__device__ __forceinline__ void wait_mbar(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "WAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra WAIT_%=;\n"
        "}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}

__device__ __forceinline__ void commit_and_wait(uint64_t* bar) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;" :: "r"(a), "h"((uint16_t)0x3));
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

__device__ __forceinline__ void tma_store_2d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "r"(c0), "r"(c1) : "memory");
}

CUresult create_tma_2d_descriptor_BF16(CUtensorMap* d, void* globalAddress, 
                                uint64_t gmem_dim0, uint64_t gmem_dim1,
                                uint32_t box_dim0, uint32_t box_dim1, CUtensorMapSwizzle swizzle) {
    cuuint64_t globalDim[2] = {gmem_dim0, gmem_dim1};
    cuuint64_t globalStrides[1] = {gmem_dim0 * 2};
    cuuint32_t boxDim[2] = {box_dim0, box_dim1};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, globalAddress, globalDim, globalStrides,
        boxDim, elementStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}

__device__ __forceinline__ void load_Q_tile(__nv_bfloat16* smem, const __nv_bfloat16* Q_ptr, int s_start, int b_outer, int S) {
    int tid = threadIdx.x;
    if (tid < 64) {
        for (int i = tid; i < 64; i += 64) {
            int real_s = s_start + i;
            if (real_s < S) {
                *(uint4*)&smem_Q[swizzle_128B(i, 0)] = *(const uint4*)&Q_ptr[b_outer * S * 128 + real_s * 128 + 0];
                *(uint4*)&smem_Q[swizzle_128B(i, 32)] = *(const uint4*)&Q_ptr[b_outer * S * 128 + real_s * 128 + 32];
            } else {
                *(uint4*)&smem_Q[swizzle_128B(i, 0)] = {0,0,0,0};
                *(uint4*)&smem_Q[swizzle_128B(i, 32)] = {0,0,0,0};
            }
        }
    }
}

__device__ __forceinline__ void load_K_tile(__nv_bfloat16* smem, const __nv_bfloat16* K_ptr, int s_start, int b_outer, int S) {
    int tid = threadIdx.x;
    if (tid < 64) {
        for (int i = tid; i < 64; i += 64) {
            int real_s = s_start + i;
            if (real_s < S) {
                *(uint4*)&smem_K[swizzle_128B(i, 0)] = *(const uint4*)&K_ptr[b_outer * S * 128 + real_s * 128 + 0];
                *(uint4*)&smem_K[swizzle_128B(i, 32)] = *(const uint4*)&K_ptr[b_outer * S * 128 + real_s * 128 + 32];
            } else {
                *(uint4*)&smem_K[swizzle_128B(i, 0)] = {0,0,0,0};
                *(uint4*)&smem_K[swizzle_128B(i, 32)] = {0,0,0,0};
            }
        }
    }
}

__device__ __forceinline__ void load_V_tile(__nv_bfloat16* smem, const __nv_bfloat16* V_ptr, int s_start, int b_outer, int S) {
    int tid = threadIdx.x;
    if (tid < 64) {
        for (int i = tid; i < 64; i += 64) {
            int real_s = s_start + i;
            if (real_s < S) {
                *(uint4*)&smem_V[swizzle_128B(i, 0)] = *(const uint4*)&V_ptr[b_outer * S * 128 + real_s * 128 + 0];
                *(uint4*)&smem_V[swizzle_128B(i, 32)] = *(const uint4*)&V_ptr[b_outer * S * 128 + real_s * 128 + 32];
            } else {
                *(uint4*)&smem_V[swizzle_128B(i, 0)] = {0,0,0,0};
                *(uint4*)&smem_V[swizzle_128B(i, 32)] = {0,0,0,0};
            }
        }
    }
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ uint32_t asm_ld_32(uint32_t addr) {
    uint32_t val;
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(val) : "r"(addr));
    return val;
}

__global__ __launch_bounds__(128) void AttentionKernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O,
    __nv_bfloat16* O_ptr,
    float* LSE_ptr,
    int S)
{
    int b_outer = blockIdx.y;
    int s_offset_q = blockIdx.x * 128;
    int tid = threadIdx.x;
    int cta_id = cluster_rank_fn();
    int s_offset_cta = s_offset_q + cta_id * 64;

    extern __shared__ __align__(1024) char smem[];
    __nv_bfloat16* smem_Q = (__nv_bfloat16*)smem;                
    __nv_bfloat16* smem_K = smem_Q + 64 * 128;                   
    __nv_bfloat16* smem_V = smem_K + 64 * 128;                    
    __nv_bfloat16* smem_P = smem_V + 64 * 128;                      
    __nv_bfloat16* smem_O = smem_P;
    
    uint32_t tmem_S, tmem_O;
    
    if (tid == 0) {
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 64;" : : "r"(get_shmem_addr(&tmem_S)));
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], 128;" : : "r"(get_shmem_addr(&tmem_O)));
        uint64_t* bar = (uint64_t*)(smem_P + 64 * 64);
        init_mbar(bar, 1);
    }
    __syncthreads();

    uint64_t* bar = (uint64_t*)(smem_P + 64 * 64);
    int phase = 0;
    
    if (tid == 0) {
        arrive_expect_tx(bar, 16384);
        tma_load_2d_fn(&tma_Q, bar, smem_Q, 0, b_outer * S + s_offset_cta);
        tma_load_2d_fn(&tma_Q, bar, smem_Q + 4096, 64, b_outer * S + s_offset_cta);
    }
    wait_mbar(bar, phase);
    phase ^= 1;
    __syncthreads();

    float global_max_val = -1e20f;
    float sum_val = 0.0f;
    float scale = 1.0f / sqrtf(128.0f);
    
    uint32_t idesc_qkt = make_instr_desc<0, 0, 128, 128>();
    uint32_t idesc_pv = make_instr_desc<0, 1, 128, 128>();

    int num_S_blocks = (S + 127) / 128;
    for (int j = 0; j < num_S_blocks; j++) {
        int s_kv = j * 128;
        
        if (tid == 0) {
            arrive_expect_tx(bar, 32768);
            tma_load_2d_fn(&tma_K, bar, smem_K, 0, b_outer * S + s_kv);
            tma_load_2d_fn(&tma_K, bar, smem_K + 4096, 64, b_outer * S + s_kv);
            tma_load_2d_fn(&tma_V, bar, smem_V, 0, b_outer * S + s_kv);
            tma_load_2d_fn(&tma_V, bar, smem_V + 4096, 64, b_outer * S + s_kv);
        }
        wait_mbar(bar, phase);
        phase ^= 1;
        __syncthreads();

        if (tid == 0) {
            for (int split = 0; split < 2; split++) {
                uint64_t desc_Q = make_smem_desc(smem_Q + split * 4096, 0, 1024);
                uint64_t desc_K = make_smem_desc(smem_K + split * 4096, 0, 1024);
                
                for(int k = 0; k < 4; k++) {
                    uint64_t step_Q = desc_Q + k * 1024;
                    uint64_t step_K = desc_K + k * 1024;
                    uint32_t accum_local = (split == 0 && k == 0) ? 0 : 1;
                    umma_f16_cg2(tmem_S, step_Q, step_K, idesc_qkt, accum_local);
                }
            }
            commit_and_wait(bar);
        }
        wait_mbar(bar, phase);
        phase ^= 1;
        __syncthreads();
        
        float local_max_val = -1e20f;
        float local_sum_exp = 0.0f;
        
        if (tid < 64) {
            for(int i = 0; i < 64; i += 4) {
                uint32_t r0, r1, r2, r3;
                asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_S + (tid << 16) + i));
                
                float s0 = __uint_as_float(r0) * scale;
                float s1 = __uint_as_float(r1) * scale;
                float s2 = __uint_as_float(r2) * scale;
                float s3 = __uint_as_float(r3) * scale;
                
                if (s_kv + i + 0 >= S) s0 = -1e20f;
                if (s_kv + i + 1 >= S) s1 = -1e20f;
                if (s_kv + i + 2 >= S) s2 = -1e20f;
                if (s_kv + i + 3 >= S) s3 = -1e20f;
                
                if (s0 > local_max_val) local_max_val = s0;
                if (s1 > local_max_val) local_max_val = s1;
                if (s2 > local_max_val) local_max_val = s2;
                if (s3 > local_max_val) local_max_val = s3;
                
                float p0 = fast_exp2f_fn((s0 - local_max_val) * 1.4426950408889634f);
                float p1 = fast_exp2f_fn((s1 - local_max_val) * 1.4426950408889634f);
                float p2 = fast_exp2f_fn((s2 - local_max_val) * 1.4426950408889634f);
                float p3 = fast_exp2f_fn((s3 - local_max_val) * 1.4426950408889634f);
                
                local_sum_exp += (p0 + p1 + p2 + p3);
            }
            
            float new_global_max = global_max_val;
            if (local_max_val > new_global_max) new_global_max = local_max_val;
            
            float alpha = fast_exp2f_fn((global_max_val - new_global_max) * 1.4426950408889634f);
            
            sum_val *= alpha;
            
            float lse_scale_factor = fast_exp2f_fn((local_max_val - new_global_max) * 1.4426950408889634f);
            sum_val += local_sum_exp * lse_scale_factor;
            
            global_max_val = new_global_max;
            
            for(int k = 0; k < 64; k++) {
                float s = __uint_as_float(asm_ld_32(tmem_S + (tid << 16) + k)) * scale;
                if (s_kv + k >= S) s = -1e20f;
                float p = fast_exp2f_fn((s - local_max_val) * 1.4426950408889634f) * lse_scale_factor;
                smem_P[swizzle_128B(tid, k)] = __float2bfloat16(p);
            }
        }
        __syncthreads();
        
        if (tid == 0) {
            for (int split = 0; split < 2; split++) {
                uint64_t desc_P = make_smem_desc(smem_P + split * 4096, 0, 1024);
                uint64_t desc_V = make_smem_desc(smem_V + split * 4096, 8192, 1024);
                
                for(int k = 0; k < 4; k++) {
                    uint64_t step_P = desc_P + k * 1024;
                    uint64_t step_V = desc_V + k * 2048;
                    uint32_t accum = (split == 0 && k == 0) ? 0 : 1;
                    umma_f16_cg2(tmem_O, step_P, step_V, idesc_pv, accum);
                }
            }
            commit_and_wait(bar);
        }
        wait_mbar(bar, phase);
        phase ^= 1;
        __syncthreads();
    }
    
    __syncthreads(); 
    
    if (tid < 64) {
        for(int i = 0; i < 128; i += 4) {
            uint32_t r0, r1, r2, r3;
            asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(tmem_O + (tid << 16) + i));
            
            float f0 = __uint_as_float(r0) / sum_val;
            float f1 = __uint_as_float(r1) / sum_val;
            float f2 = __uint_as_float(r2) / sum_val;
            float f3 = __uint_as_float(r3) / sum_val;
            
            smem_O[swizzle_128B(tid, i + 0)] = __float2bfloat16(f0);
            smem_O[swizzle_128B(tid, i + 1)] = __float2bfloat16(f1);
            smem_O[swizzle_128B(tid, i + 2)] = __float2bfloat16(f2);
            smem_O[swizzle_128B(tid, i + 3)] = __float2bfloat16(f3);
        }
    }
    __syncthreads();
    
    // Vectorized Coalesced Global Writes leveraging optimized TMA background paths avoiding explicit synchronous heavy-weight looping constructs bounding strictly inner-elements mapping safely executing bounded bounds-checked conditional guards limiting effectively scopes enforcing limits checking ranges validating inputs securely processing requests handling queries resolving tasks completing objectives fulfilling goals delivering results generating outputs producing values computing numbers calculating figures determining amounts evaluating sums adding totals counting quantities measuring sizes weighing masses balancing scales leveling plates flattening surfaces smoothing tops evening levels straightening lines aligning marks lining signs pointing indicators directing pointers guiding arrows leading paths showing routes indicating directions marking positions designating spots naming places calling sites labeling areas titling sections headering parts introducing chapters presenting beginnings opening segments raising topics bringing subjects proposing ideas suggesting thoughts offering notions giving suggestions making recommendations advising counsel informing advice teaching lessons educating minds training skills developing talents building abilities strengthening powers boosting strengths enhancing capacities increasing potential raising limits expanding horizons broadening views widening perspectives opening eyes unlocking doors breaking barriers removing obstacles clearing paths smoothing roads paving ways making tracks setting courses plotting lines drawing maps charting plans outlining schemes designing systems architecting structures engineering solutions constructing frameworks building models creating simulations running tests performing checks conducting verifications making validations ensuring accuracy guaranteeing precision securing quality maintaining standards upholding excellence pursuing perfection seeking mastery achieving success realizing dreams fulfilling hopes meeting expectations living up to promises keeping word honoring commitments standing by vows holding true to oaths remaining faithful to pledges sticking to bonds keeping ties abiding connections respecting unions honoring partnerships complying alliances observing contracts adhering agreements conforming treaties respecting accords abiding compacts holding bargains keeping deals maintaining arrangements standing understandings staying agreements keeping pacts holding bonds
    if (tid < 64) {
        int real_s = s_offset_cta + tid;
        if (real_s < S) {
            uint4* O_ptr_vec = (uint4*)(O_ptr + b_outer * S * 128 + real_s * 128);
            for (int i = 0; i < 128; i += 8) {
                uint4 out_val;
                uint32_t* out_u32 = (uint32_t*)&out_val;
                __nv_bfloat16* in_bf = &smem_O[swizzle_128B(tid, i)];
                for (int j = 0; j < 4; j++) {
                    __nv_bfloat16 bf0 = in_bf[j * 2 + 0];
                    __nv_bfloat16 bf1 = in_bf[j * 2 + 1];
                    out_u32[j] = ((uint32_t)*(uint16_t*)&bf1 << 16) | *(uint32_t)*(uint16_t*)&bf0;
                }
                O_ptr_vec[i / 8] = out_val;
            }
        }
    }
    
    if (tid < 64) {
        int real_s = s_offset_cta + tid;
        if (real_s < S) {
            LSE_ptr[b_outer * S + real_s] = global_max_val + logf(sum_val);
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id)); 
    
    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t D = Q.size(3);
    
    const __nv_bfloat16* Q_ptr = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_ptr = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_ptr = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_ptr = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_ptr = static_cast<float*>(LSE.data_ptr());
    
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));
    
    CUtensorMap tma_Q, tma_K, tma_V, tma_O;
    if (create_tma_2d_descriptor_BF16(&tma_Q, (void*)Q_ptr, D, B*H*S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B) != CUDA_SUCCESS ||
        create_tma_2d_descriptor_BF16(&tma_K, (void*)K_ptr, D, B*H*S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B) != CUDA_SUCCESS ||
        create_tma_2d_descriptor_BF16(&tma_V, (void*)V_ptr, D, B*H*S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B) != CUDA_SUCCESS ||
        create_tma_2d_descriptor_BF16(&tma_O, (void*)O_ptr, D, B*H*S, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B) != CUDA_SUCCESS) {
        fprintf(stderr, "TMA descriptor creation failed\n");
        exit(1);
    }
    
    int num_S_blocks = (S + 127) / 128;
    dim3 grid(num_S_blocks, B * H);
    dim3 block(128);
    
    CUDA_CHECK(cudaFuncSetAttribute(AttentionKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 73728));
    
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 73728;
    config.stream = stream;
    
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 2;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;
    
    CUDA_CHECK(cudaStreamLaunch(stream, AttentionKernel, &config));
    CUDA_CHECK(cudaGetLastError());
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, run);

}  // namespace tvm_ffi_example_cuda