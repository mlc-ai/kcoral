#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
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

// Helper functions (Hardware Instructions)
__device__ __forceinline__ void tmem_alloc_fn(uint32_t* dst_smem, int ncols) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(a), "r"(ncols));
}

__device__ __forceinline__ void tmem_dealloc_fn(uint32_t addr, int ncols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(ncols));
}

__device__ __forceinline__ void umma_f16_cg1_fn(uint32_t tmem_c, uint64_t desc_a, uint64_t desc_b, uint32_t idesc, uint32_t accum) {
    asm volatile(
        "{\n.reg .pred p;\n"
        "setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accum));
}

__device__ __forceinline__ void tmem_load_8x_fn(uint32_t col,
    uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
    uint32_t* r4, uint32_t* r5, uint32_t* r6, uint32_t* r7) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
   : "=r"(*r0),"=r"(*r1),"=r"(*r2),"=r"(*r3), "=r"(*r4),"=r"(*r5),"=r"(*r6),"=r"(*r7) : "r"(col));
}

__device__ __forceinline__ float fast_exp2f_fn(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void st_shared_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}

__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
        "l"((uint64_t)d),
        "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
        "r"(c0), "r"(c1) : "memory");
}

// SM100 Descriptors Build
__device__ __forceinline__ uint64_t make_smem_desc_swizzle128(void* ptr, uint32_t lbo, uint32_t sbo) {
    uint64_t d = 0;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
    uint32_t base_offset = (addr >> 7) & 0x7;
    d |= (uint64_t)(addr & 0x3FFFF) >> 4;
    d |= (uint64_t)((lbo & 0x3FFFF) >> 4) << 16;
    d |= (uint64_t)((sbo & 0x3FFFF) >> 4) << 32;
    d |= (uint64_t)1 << 46;   
    d |= (uint64_t)base_offset << 49;
    d |= (uint64_t)2 << 61;   
    return d;
}

__device__ __forceinline__ uint32_t make_instr_desc_fn(uint32_t M, uint32_t N, bool a_kmajor, bool b_kmajor) {
    uint32_t d = 0;
    d |= (1u << 4);    // FP32 dest
    d |= (1u << 7);    // BF16 A
    d |= (1u << 10);   // BF16 B
    d |= ((a_kmajor ? 0u : 1u) << 15);
    d |= ((b_kmajor ? 0u : 1u) << 16);
    d |= ((N / 8) << 17);
    d |= ((M / 16) << 24);
    return d;
}

__device__ __forceinline__ void mma_64x64_k64(
    uint32_t tmem_D, void* smem_A, void* smem_B, 
    uint32_t idesc, bool a_kmajor, bool b_kmajor,
    uint32_t step_A, uint32_t step_B, bool overwrite) {
    uint32_t sbo_A = 1024, sbo_B = 1024;
    uint32_t lbo_A = a_kmajor ? 1 : 8192;
    uint32_t lbo_B = b_kmajor ? 1 : 8192;
    for (int k = 0; k < 4; ++k) {
        uint64_t desc_a = make_smem_desc_swizzle128((char*)smem_A + k * step_A, lbo_A, sbo_A);
        uint64_t desc_b = make_smem_desc_swizzle128((char*)smem_B + k * step_B, lbo_B, sbo_B);
        uint32_t accum = (overwrite && k == 0) ? 0 : 1;
        umma_f16_cg1_fn(tmem_D, desc_a, desc_b, idesc, accum);
    }
}


__global__ void sdpa_bwd_kernel(
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_O,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const float* __restrict__ L,
    __nv_bfloat16* __restrict__ dQ,
    __nv_bfloat16* __restrict__ dK,
    __nv_bfloat16* __restrict__ dV,
    uint32_t seq_len
) {
    extern __shared__ uint8_t smem[];
    uint64_t* mbar = (uint64_t*)smem;
    uint32_t* tmem_addr = (uint32_t*)(smem + 8);
    uint8_t* smem_base = smem + 128; // Offset to avoid misalignment

    uint8_t* smem_Q_L  = smem_base;
    uint8_t* smem_Q_R  = smem_base + 16384;
    uint8_t* smem_dO_L = smem_base + 32768;
    uint8_t* smem_dO_R = smem_base + 49152;
    uint8_t* smem_O_L  = smem_base + 65536;
    uint8_t* smem_O_R  = smem_base + 81920;
    uint8_t* smem_K_L  = smem_base + 98304;
    uint8_t* smem_K_R  = smem_base + 106496;
    uint8_t* smem_V_L  = smem_base + 114688;
    uint8_t* smem_V_R  = smem_base + 122880;
    uint8_t* smem_dS   = smem_base + 131072;
    uint8_t* smem_S    = smem_base + 147456;

    uint32_t block_i = blockIdx.x;
    uint32_t h = blockIdx.y;
    uint32_t b = blockIdx.z;
    uint32_t t = threadIdx.x;
    uint32_t global_row = block_i * 128 + t;
    uint32_t batch_offset = (b * gridDim.y + h) * seq_len;
    
    // TMEM Allocation
    if (t == 0) {
        tmem_alloc_fn(tmem_addr, 512); // Fit perfectly exactly 512 cols for gradients
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(128));
    }
    __syncthreads();
    uint32_t tmem_base = *tmem_addr;
    uint32_t tmem_dQ_L = tmem_base + 0;
    uint32_t tmem_dQ_R = tmem_base + 64;
    uint32_t tmem_dK_L = tmem_base + 128;
    uint32_t tmem_dK_R = tmem_base + 192;
    uint32_t tmem_dV_L = tmem_base + 256;
    uint32_t tmem_dV_R = tmem_base + 320;
    uint32_t tmem_P    = tmem_base + 384;
    uint32_t tmem_dP   = tmem_base + 448;

    // Load outer loop tensors (Q, dO, O)
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(98304));
    tma_load_2d_fn(&tma_Q,  mbar, smem_Q_L,  0,  batch_offset + block_i * 128);
    tma_load_2d_fn(&tma_Q,  mbar, smem_Q_R,  64, batch_offset + block_i * 128);
    tma_load_2d_fn(&tma_dO, mbar, smem_dO_L, 0,  batch_offset + block_i * 128);
    tma_load_2d_fn(&tma_dO, mbar, smem_dO_R, 64, batch_offset + block_i * 128);
    tma_load_2d_fn(&tma_O,  mbar, smem_O_L,  0,  batch_offset + block_i * 128);
    tma_load_2d_fn(&tma_O,  mbar, smem_O_R,  64, batch_offset + block_i * 128);

    float L_val = 0.0f;
    if (global_row < seq_len) {
        L_val = L[batch_offset + global_row];
    }
    
    // Wait for outer loop TMA
    asm volatile(
        "{\n.reg .pred P;\nWAIT_%=:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], 0;\n"
        "@!P bra WAIT_%=;\n}\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));

    // D_i dot product execution internally from un-swizzled parsing logic directly on SWIZZLE_128B chunk pattern layout
    float D_val = 0.0f;
    for (int c = 0; c < 8; ++c) {
        uint32_t swizzled_c = (t % 8) ^ c;
        uint32_t offset = (t * 8 + swizzled_c) * 16;
        float4 dO_vec = *(float4*)(smem_dO_L + offset);
        float4 O_vec  = *(float4*)(smem_O_L + offset);
        
        __nv_bfloat162* dO_bf = (__nv_bfloat162*)&dO_vec;
        __nv_bfloat162* O_bf  = (__nv_bfloat162*)&O_vec;
        for(int i=0; i<4; i++) {
            float2 f_dO = __bfloat1622float2(dO_bf[i]);
            float2 f_O  = __bfloat1622float2(O_bf[i]);
            D_val += f_dO.x * f_O.x + f_dO.y * f_O.y;
        }
        
        dO_vec = *(float4*)(smem_dO_R + offset);
        O_vec  = *(float4*)(smem_O_R + offset);
        dO_bf = (__nv_bfloat162*)&dO_vec;
        O_bf  = (__nv_bfloat162*)&O_vec;
        for(int i=0; i<4; i++) {
            float2 f_dO = __bfloat1622float2(dO_bf[i]);
            float2 f_O  = __bfloat1622float2(O_bf[i]);
            D_val += f_dO.x * f_O.x + f_dO.y * f_O.y;
        }
    }

    uint32_t idesc_128x64 = make_instr_desc_fn(128, 64, true, true);
    uint32_t idesc_128x64_bMN = make_instr_desc_fn(128, 64, true, false);
    uint32_t idesc_64x64 = make_instr_desc_fn(64, 64, false, false);
    
    uint32_t max_j = (block_i * 128 + 127) / 64;
    for (uint32_t j = 0; j <= max_j; ++j) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"(32768));
        tma_load_2d_fn(&tma_K, mbar, smem_K_L, 0,  batch_offset + j * 64);
        tma_load_2d_fn(&tma_K, mbar, smem_K_R, 64, batch_offset + j * 64);
        tma_load_2d_fn(&tma_V, mbar, smem_V_L, 0,  batch_offset + j * 64);
        tma_load_2d_fn(&tma_V, mbar, smem_V_R, 64, batch_offset + j * 64);
        
        asm volatile(
            "{\n.reg .pred P;\nWAIT_%=:\n"
            "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
            "@!P bra WAIT_%=;\n}\n"
            :: "r"((uint32_t)__cvta_generic_to_shared(mbar)), "r"((j+1)&1));

        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        
        // Compute P_ij and dP_ij in TMEM
        mma_64x64_k64(tmem_P, smem_Q_L, smem_K_L, idesc_128x64, true, true, 32, 2048, true);
        mma_64x64_k64(tmem_P, smem_Q_R, smem_K_R, idesc_128x64, true, true, 32, 2048, false);
        
        mma_64x64_k64(tmem_dP, smem_dO_L, smem_V_L, idesc_128x64, true, true, 32, 2048, true);
        mma_64x64_k64(tmem_dP, smem_dO_R, smem_V_R, idesc_128x64, true, true, 32, 2048, false);
        
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
        
        // Mathematical evaluation & SWIZZLE_128B packaging for SMEM
        for(int c = 0; c < 64; c += 8) {
            uint32_t rP[8], rDP[8];
            tmem_load_8x_fn(tmem_P + c, &rP[0], &rP[1], &rP[2], &rP[3], &rP[4], &rP[5], &rP[6], &rP[7]);
            tmem_load_8x_fn(tmem_dP + c, &rDP[0], &rDP[1], &rDP[2], &rDP[3], &rDP[4], &rDP[5], &rDP[6], &rDP[7]);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            
            uint32_t chunk_dS[4]; 
            uint32_t chunk_S[4];
            
            float ds_prev = 0, s_prev = 0;
            for(int i=0; i<8; i++) {
                uint32_t global_col = j * 64 + c + i;
                float s = 0.0f;
                float ds = 0.0f;
                
                if (global_row < seq_len && global_col <= global_row && global_col < seq_len) {
                    float scaled_p = __uint_as_float(rP[i]) * 0.08838834764f; // 1 / sqrt(128)
                    s = fast_exp2f_fn((scaled_p - L_val) * 1.44269504088f); // * log2(e)
                    float ds_fp32 = (__uint_as_float(rDP[i]) - D_val) * s;
                    ds = ds_fp32 * 0.08838834764f;
                }
                
                if (i % 2 == 1) {
                    __nv_bfloat162 packed_dS = __floats2bfloat162_rn(ds_prev, ds);
                    __nv_bfloat162 packed_S  = __floats2bfloat162_rn(s_prev, s);
                    chunk_dS[i/2] = *(uint32_t*)&packed_dS;
                    chunk_S[i/2]  = *(uint32_t*)&packed_S;
                } else {
                    ds_prev = ds;
                    s_prev = s;
                }
            }
            uint32_t swx = c / 8;
            uint32_t swizzled_c = (t % 8) ^ swx;
            uint32_t offset = (t * 8 + swizzled_c) * 16;
            st_shared_128_fn((uint32_t)__cvta_generic_to_shared(smem_dS + offset), chunk_dS[0], chunk_dS[1], chunk_dS[2], chunk_dS[3]);
            st_shared_128_fn((uint32_t)__cvta_generic_to_shared(smem_S + offset),  chunk_S[0], chunk_S[1], chunk_S[2], chunk_S[3]);
        }
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        __syncthreads();
        
        // Gradient compute onto TMEM natively supporting transposed operands via SM100 Descriptors
        mma_64x64_k64(tmem_dQ_L, smem_dS, smem_K_L, idesc_128x64_bMN, true, false, 32, 2048, j == 0);
        mma_64x64_k64(tmem_dQ_R, smem_dS, smem_K_R, idesc_128x64_bMN, true, false, 32, 2048, j == 0);
        
        mma_64x64_k64(tmem_dK_L, smem_dS, smem_Q_L, idesc_64x64, false, false, 2048, 2048, true);
        mma_64x64_k64(tmem_dK_R, smem_dS, smem_Q_R, idesc_64x64, false, false, 2048, 2048, true);
        
        mma_64x64_k64(tmem_dV_L, smem_S,  smem_dO_L, idesc_64x64, false, false, 2048, 2048, true);
        mma_64x64_k64(tmem_dV_R, smem_S,  smem_dO_R, idesc_64x64, false, false, 2048, 2048, true);
        
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(mbar)));
        
        // Atomic GMEM accumulates natively supporting bf162
        uint32_t row = t % 64;
        uint32_t k_row_global = j * 64 + row;
        uint32_t half_offset = (t < 64) ? 0 : 64;
        
        if (k_row_global < seq_len) {
            for (int c = 0; c < 64; c += 8) {
                uint32_t rK[8], rV[8];
                tmem_load_8x_fn((t < 64 ? tmem_dK_L : tmem_dK_R) + c, &rK[0], &rK[1], &rK[2], &rK[3], &rK[4], &rK[5], &rK[6], &rK[7]);
                tmem_load_8x_fn((t < 64 ? tmem_dV_L : tmem_dV_R) + c, &rV[0], &rV[1], &rV[2], &rV[3], &rV[4], &rV[5], &rV[6], &rV[7]);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                __nv_bfloat162* ptr_K = (__nv_bfloat162*)(dK + batch_offset + k_row_global * 128 + half_offset + c);
                __nv_bfloat162* ptr_V = (__nv_bfloat162*)(dV + batch_offset + k_row_global * 128 + half_offset + c);
                for(int i=0; i<4; i++) {
                    atomicAdd(ptr_K + i, __floats2bfloat162_rn(__uint_as_float(rK[2*i]), __uint_as_float(rK[2*i+1])));
                    atomicAdd(ptr_V + i, __floats2bfloat162_rn(__uint_as_float(rV[2*i]), __uint_as_float(rV[2*i+1])));
                }
            }
        }
        __syncthreads();
    }
    
    // Direct store dQ
    if (global_row < seq_len) {
        for(int half=0; half<2; half++) {
            uint32_t tmem_src = (half == 0) ? tmem_dQ_L : tmem_dQ_R;
            uint32_t col_offset = (half == 0) ? 0 : 64;
            for (int c = 0; c < 64; c += 8) {
                uint32_t r[8];
                tmem_load_8x_fn(tmem_src + c, &r[0], &r[1], &r[2], &r[3], &r[4], &r[5], &r[6], &r[7]);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                
                __nv_bfloat162 vals[4];
                for(int i=0; i<4; i++) {
                    vals[i] = __floats2bfloat162_rn(__uint_as_float(r[2*i]), __uint_as_float(r[2*i+1]));
                }
                float4* out_ptr = (float4*)(dQ + batch_offset + global_row * 128 + col_offset + c);
                *out_ptr = *(float4*)&vals[0];
            }
        }
    }
    
    if (t == 0) tmem_dealloc_fn(tmem_base, 512);
}


CUresult create_tma_2d_descriptor(CUtensorMap* d, void* globalAddress, uint64_t gmem_inner_dim, uint64_t gmem_outer_dim, uint32_t smem_inner_dim, uint32_t smem_outer_dim) {
    cuuint64_t globalDim[2] = {gmem_inner_dim, gmem_outer_dim};
    cuuint64_t globalStrides[1] = {gmem_inner_dim * 2};
    cuuint32_t boxDim[2] = {smem_inner_dim, smem_outer_dim};
    cuuint32_t elementStrides[2] = {1, 1};
    return cuTensorMapEncodeTiled(
        d,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2, 
        globalAddress,
        globalDim,
        globalStrides,
        boxDim, 
        elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
}


namespace tvm_ffi_sdpa_causal {

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView dO, tvm::ffi::TensorView L,
         tvm::ffi::TensorView dQ, tvm::ffi::TensorView dK, tvm::ffi::TensorView dV) {
    
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    int64_t B = Q.size(0);
    int64_t H = Q.size(1);
    int64_t S = Q.size(2);
    int64_t d = 128;
    
    CUDA_CHECK(cudaMemsetAsync(dK.data_ptr(), 0, B * H * S * d * 2, stream));
    CUDA_CHECK(cudaMemsetAsync(dV.data_ptr(), 0, B * H * S * d * 2, stream));

    CUtensorMap tma_Q, tma_dO, tma_O, tma_K, tma_V;
    
    create_tma_2d_descriptor(&tma_Q, Q.data_ptr(), 128, B * H * S, 64, 128);
    create_tma_2d_descriptor(&tma_dO, dO.data_ptr(), 128, B * H * S, 64, 128);
    create_tma_2d_descriptor(&tma_O, O.data_ptr(), 128, B * H * S, 64, 128);
    create_tma_2d_descriptor(&tma_K, K.data_ptr(), 128, B * H * S, 64, 64);
    create_tma_2d_descriptor(&tma_V, V.data_ptr(), 128, B * H * S, 64, 64);

    dim3 grid((S + 127) / 128, H, B);
    dim3 block(128);

    int smem_size = 196608; 
    CUDA_CHECK(cudaFuncSetAttribute(sdpa_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    sdpa_bwd_kernel<<<grid, block, smem_size, stream>>>(
        tma_Q, tma_dO, tma_O, tma_K, tma_V, 
        static_cast<const float*>(L.data_ptr()),
        static_cast<__nv_bfloat16*>(dQ.data_ptr()),
        static_cast<__nv_bfloat16*>(dK.data_ptr()),
        static_cast<__nv_bfloat16*>(dV.data_ptr()),
        S
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_sdpa_causal::run);

}