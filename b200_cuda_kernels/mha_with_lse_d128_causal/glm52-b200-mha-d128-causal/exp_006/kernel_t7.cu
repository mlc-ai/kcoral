#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cstdio>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

using namespace nvcuda;

#define CUDA_CHECK(call) do {                                    \
    cudaError_t _e = (call);                                     \
    if (_e != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error %s at %s:%d\n",              \
                cudaGetErrorString(_e), __FILE__, __LINE__);     \
        exit(1);                                                 \
    }                                                            \
} while(0)

namespace tvm_ffi_example_cuda {

constexpr int BM = 128;
constexpr int BN = 64;
constexpr int D = 128;
constexpr int THREADS = 256;
constexpr int WM = 16, WN = 16, WK = 16;
constexpr int D_TILES = D / WN;   // 8
constexpr int COL_TILES = BN / WN; // 4
constexpr int K_STEPS = D / WK;    // 8

__device__ __forceinline__ void cp_async_16(void* dst_smem, const void* src_gmem) {
    uint32_t dst = (uint32_t)__cvta_generic_to_shared(dst_smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(dst), "l"(src_gmem));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

__global__ void attention_kernel(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int B, int H, int S) {

    int bh = blockIdx.y;
    int b = bh / H;
    int h = bh % H;
    int q_block = blockIdx.x;
    int q_start = q_block * BM;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    const int64_t base_offset = (int64_t)(b * H + h) * S * D;
    const float scale = 0.08838834764831845f;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_raw);         // 128*128*2 = 32KB
    __nv_bfloat16* sK = sQ + BM * D;                                         // 64*128*2 = 16KB
    __nv_bfloat16* sV = sK + BN * D;                                         // 64*128*2 = 16KB
    // sP reuses sK space (16KB) after QK^T is done
    __nv_bfloat16* sP = sK;
    float* s_rowmax = reinterpret_cast<float*>(sV + BN * D);                 // 128*4 = 512B
    float* s_rowsum = s_rowmax + BM;                                         // 512B

    // Load Q tile [BM, D] using cp.async
    #pragma unroll
    for (int i = tid; i < BM * D / 8; i += THREADS) {
        int row = i / (D / 8);
        int col8 = i % (D / 8);
        int q_pos = q_start + row;
        if (q_pos < S) {
            cp_async_16(&sQ[row * D + col8 * 8],
                        &Q[base_offset + (int64_t)q_pos * D + col8 * 8]);
        } else {
            *(int4*)&sQ[row * D + col8 * 8] = make_int4(0, 0, 0, 0);
        }
    }
    cp_async_commit();

    // Init rowmax, rowsum
    for (int i = tid; i < BM; i += THREADS) {
        s_rowmax[i] = -INFINITY;
        s_rowsum[i] = 0.0f;
    }

    // O accumulators in registers
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> o_frag[D_TILES];
    #pragma unroll
    for (int tn = 0; tn < D_TILES; tn++)
        wmma::fill_fragment(o_frag[tn], 0.0f);

    cp_async_wait_all();
    __syncthreads();

    int max_k = min(S, q_start + BM);
    int num_k_blocks = (max_k + BN - 1) / BN;

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        int k_start = k_block * BN;

        // Load K, V tiles using cp.async
        #pragma unroll
        for (int i = tid; i < BN * D / 8; i += THREADS) {
            int row = i / (D / 8);
            int col8 = i % (D / 8);
            int k_pos = k_start + row;
            if (k_pos < S) {
                cp_async_16(&sK[row * D + col8 * 8],
                            &K[base_offset + (int64_t)k_pos * D + col8 * 8]);
                cp_async_16(&sV[row * D + col8 * 8],
                            &V[base_offset + (int64_t)k_pos * D + col8 * 8]);
            } else {
                *(int4*)&sK[row * D + col8 * 8] = make_int4(0, 0, 0, 0);
                *(int4*)&sV[row * D + col8 * 8] = make_int4(0, 0, 0, 0);
            }
        }
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();

        // QK^T: compute S in fragments (4 col tiles per warp)
        wmma::fragment<wmma::accumulator, WM, WN, WK, float> s_frag[COL_TILES];
        #pragma unroll
        for (int t = 0; t < COL_TILES; t++)
            wmma::fill_fragment(s_frag[t], 0.0f);

        #pragma unroll
        for (int col_tile = 0; col_tile < COL_TILES; col_tile++) {
            #pragma unroll
            for (int kk = 0; kk < K_STEPS; kk++) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::col_major> b_frag;

                wmma::load_matrix_sync(a_frag, sQ + warp_id * WM * D + kk * WK, D);
                wmma::load_matrix_sync(b_frag, sK + col_tile * WN * D + kk * WK, D);
                wmma::mma_sync(s_frag[col_tile], a_frag, b_frag, s_frag[col_tile]);
            }
            #pragma unroll
            for (int i = 0; i < s_frag[col_tile].num_elements; i++)
                s_frag[col_tile].x[i] *= scale;
        }

        // Need syncthreads here: all warps must finish reading sK before
        // any warp writes P to sP (which reuses sK's memory)
        __syncthreads();

        // Softmax from fragments
        // Fragment layout: x[0,1,4,5]->row r, x[2,3,6,7]->row r+8
        // Col: (lane%4)*2 + {0,1,8,9} for x[0,1,4,5]
        int r = lane_id / 4;
        int c = (lane_id % 4) * 2;
        int row0 = warp_id * WM + r;
        int row1 = warp_id * WM + r + 8;
        int q_pos0 = q_start + row0;
        int q_pos1 = q_start + row1;

        // Apply causal mask and compute row max
        float max0 = -INFINITY, max1 = -INFINITY;
        #pragma unroll
        for (int t = 0; t < COL_TILES; t++) {
            int bc = t * WN;
            int kp0 = k_start + bc + c;
            int kp1 = k_start + bc + c + 1;
            int kp2 = k_start + bc + c + 8;
            int kp3 = k_start + bc + c + 9;

            if (kp0 > q_pos0 || kp0 >= S) s_frag[t].x[0] = -INFINITY;
            if (kp1 > q_pos0 || kp1 >= S) s_frag[t].x[1] = -INFINITY;
            if (kp2 > q_pos0 || kp2 >= S) s_frag[t].x[4] = -INFINITY;
            if (kp3 > q_pos0 || kp3 >= S) s_frag[t].x[5] = -INFINITY;
            if (kp0 > q_pos1 || kp0 >= S) s_frag[t].x[2] = -INFINITY;
            if (kp1 > q_pos1 || kp1 >= S) s_frag[t].x[3] = -INFINITY;
            if (kp2 > q_pos1 || kp2 >= S) s_frag[t].x[6] = -INFINITY;
            if (kp3 > q_pos1 || kp3 >= S) s_frag[t].x[7] = -INFINITY;

            max0 = fmaxf(max0, fmaxf(fmaxf(s_frag[t].x[0], s_frag[t].x[1]),
                                     fmaxf(s_frag[t].x[4], s_frag[t].x[5])));
            max1 = fmaxf(max1, fmaxf(fmaxf(s_frag[t].x[2], s_frag[t].x[3]),
                                     fmaxf(s_frag[t].x[6], s_frag[t].x[7])));
        }

        // Reduce max across 4 threads in group (shfl_xor with offsets 1, 2)
        #pragma unroll
        for (int offset = 1; offset <= 2; offset <<= 1) {
            max0 = fmaxf(max0, __shfl_xor_sync(0xFFFFFFFF, max0, offset));
            max1 = fmaxf(max1, __shfl_xor_sync(0xFFFFFFFF, max1, offset));
        }

        float old_max0 = (q_pos0 < S) ? s_rowmax[row0] : -INFINITY;
        float old_max1 = (q_pos1 < S) ? s_rowmax[row1] : -INFINITY;
        float new_max0 = fmaxf(old_max0, max0);
        float new_max1 = fmaxf(old_max1, max1);
        float exp_old0 = (old_max0 == -INFINITY) ? 0.0f : __expf(old_max0 - new_max0);
        float exp_old1 = (old_max1 == -INFINITY) ? 0.0f : __expf(old_max1 - new_max1);

        // Compute P (exp) and row sum
        float sum0 = 0.0f, sum1 = 0.0f;
        #pragma unroll
        for (int t = 0; t < COL_TILES; t++) {
            s_frag[t].x[0] = (s_frag[t].x[0] == -INFINITY) ? 0.0f : __expf(s_frag[t].x[0] - new_max0);
            s_frag[t].x[1] = (s_frag[t].x[1] == -INFINITY) ? 0.0f : __expf(s_frag[t].x[1] - new_max0);
            s_frag[t].x[4] = (s_frag[t].x[4] == -INFINITY) ? 0.0f : __expf(s_frag[t].x[4] - new_max0);
            s_frag[t].x[5] = (s_frag[t].x[5] == -INFINITY) ? 0.0f : __expf(s_frag[t].x[5] - new_max0);
            s_frag[t].x[2] = (s_frag[t].x[2] == -INFINITY) ? 0.0f : __expf(s_frag[t].x[2] - new_max1);
            s_frag[t].x[3] = (s_frag[t].x[3] == -INFINITY) ? 0.0f : __expf(s_frag[t].x[3] - new_max1);
            s_frag[t].x[6] = (s_frag[t].x[6] == -INFINITY) ? 0.0f : __expf(s_frag[t].x[6] - new_max1);
            s_frag[t].x[7] = (s_frag[t].x[7] == -INFINITY) ? 0.0f : __expf(s_frag[t].x[7] - new_max1);

            sum0 += s_frag[t].x[0] + s_frag[t].x[1] + s_frag[t].x[4] + s_frag[t].x[5];
            sum1 += s_frag[t].x[2] + s_frag[t].x[3] + s_frag[t].x[6] + s_frag[t].x[7];
        }

        // Reduce sum across 4 threads in group
        #pragma unroll
        for (int offset = 1; offset <= 2; offset <<= 1) {
            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, offset);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, offset);
        }

        // Update rowmax, rowsum (only thread 0 of each group)
        if (lane_id % 4 == 0) {
            if (q_pos0 < S) {
                s_rowmax[row0] = new_max0;
                s_rowsum[row0] = s_rowsum[row0] * exp_old0 + sum0;
            }
            if (q_pos1 < S) {
                s_rowmax[row1] = new_max1;
                s_rowsum[row1] = s_rowsum[row1] * exp_old1 + sum1;
            }
        }

        // Rescale O fragments
        #pragma unroll
        for (int tn = 0; tn < D_TILES; tn++) {
            o_frag[tn].x[0] *= exp_old0;
            o_frag[tn].x[1] *= exp_old0;
            o_frag[tn].x[4] *= exp_old0;
            o_frag[tn].x[5] *= exp_old0;
            o_frag[tn].x[2] *= exp_old1;
            o_frag[tn].x[3] *= exp_old1;
            o_frag[tn].x[6] *= exp_old1;
            o_frag[tn].x[7] *= exp_old1;
        }

        // Write P to shared memory as BF16 (directly from fragments)
        __syncwarp();
        #pragma unroll
        for (int t = 0; t < COL_TILES; t++) {
            int bc = t * WN;
            __nv_bfloat16* p = sP + warp_id * WM * BN + r * BN + bc + c;
            // Pack pairs into bfloat162 for efficiency
            __nv_bfloat162 p01 = __floats2bfloat162_rn(s_frag[t].x[0], s_frag[t].x[1]);
            __nv_bfloat162 p45 = __floats2bfloat162_rn(s_frag[t].x[4], s_frag[t].x[5]);
            __nv_bfloat162 p23 = __floats2bfloat162_rn(s_frag[t].x[2], s_frag[t].x[3]);
            __nv_bfloat162 p67 = __floats2bfloat162_rn(s_frag[t].x[6], s_frag[t].x[7]);
            *reinterpret_cast<__nv_bfloat162*>(p) = p01;
            *reinterpret_cast<__nv_bfloat162*>(p + 8) = p45;
            *reinterpret_cast<__nv_bfloat162*>(p + BN) = p23;
            *reinterpret_cast<__nv_bfloat162*>(p + BN + 8) = p67;
        }
        __syncwarp();

        // PV: O += P @ V
        #pragma unroll
        for (int tile_n = 0; tile_n < D_TILES; tile_n++) {
            #pragma unroll
            for (int kk = 0; kk < BN / WK; kk++) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, __nv_bfloat16, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, __nv_bfloat16, wmma::row_major> b_frag;

                wmma::load_matrix_sync(a_frag, sP + warp_id * WM * BN + kk * WK, BN);
                wmma::load_matrix_sync(b_frag, sV + kk * WK * D + tile_n * WN, D);
                wmma::mma_sync(o_frag[tile_n], a_frag, b_frag, o_frag[tile_n]);
            }
        }
        __syncthreads();
    }

    // Epilogue: store O fragments to shared memory for normalization
    float* sO_first = reinterpret_cast<float*>(sQ);   // 128*64*4 = 32KB
    float* sO_second = reinterpret_cast<float*>(sK);   // 128*64*4 = 32KB (sK+sV contiguous)

    #pragma unroll
    for (int tn = 0; tn < 4; tn++) {
        wmma::store_matrix_sync(sO_first + warp_id * WM * 64 + tn * WN,
            o_frag[tn], 64, wmma::mem_row_major);
    }
    #pragma unroll
    for (int tn = 4; tn < 8; tn++) {
        wmma::store_matrix_sync(sO_second + warp_id * WM * 64 + (tn - 4) * WN,
            o_frag[tn], 64, wmma::mem_row_major);
    }
    __syncthreads();

    // Store LSE
    for (int i = tid; i < BM; i += THREADS) {
        int q_pos = q_start + i;
        if (q_pos < S) {
            LSE[(int64_t)(b * H + h) * S + q_pos] =
                s_rowmax[i] + logf(s_rowsum[i] + 1e-30f);
        }
    }

    // Final normalization and coalesced store to global O
    for (int i = tid; i < BM * D / 8; i += THREADS) {
        int row = i / (D / 8);
        int col8 = i % (D / 8);
        int col = col8 * 8;
        int q_pos = q_start + row;
        if (q_pos < S) {
            float* sO_half = (col < 64) ? sO_first : sO_second;
            int local_col = col % 64;
            float sum = s_rowsum[row] + 1e-30f;

            int4 out;
            __nv_bfloat16* bf16_out = reinterpret_cast<__nv_bfloat16*>(&out);
            #pragma unroll
            for (int dd = 0; dd < 8; dd++) {
                float val = sO_half[row * 64 + local_col + dd] / sum;
                bf16_out[dd] = __float2bfloat16(val);
            }
            *reinterpret_cast<int4*>(O + base_offset + (int64_t)q_pos * D + col) = out;
        }
    }
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K,
         tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE) {
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));

    const int B = 4;
    const int H = 48;
    const int S = static_cast<int>(Q.size(2));

    const __nv_bfloat16* Q_data = static_cast<const __nv_bfloat16*>(Q.data_ptr());
    const __nv_bfloat16* K_data = static_cast<const __nv_bfloat16*>(K.data_ptr());
    const __nv_bfloat16* V_data = static_cast<const __nv_bfloat16*>(V.data_ptr());
    __nv_bfloat16* O_data = static_cast<__nv_bfloat16*>(O.data_ptr());
    float* LSE_data = static_cast<float*>(LSE.data_ptr());

    dim3 grid((S + BM - 1) / BM, B * H);
    dim3 block(THREADS);

    // sQ(32KB) + sK(16KB) + sV(16KB) + s_rowmax(512B) + s_rowsum(512B)
    int smem_size = BM * D * 2        // sQ (32KB)
                  + BN * D * 2        // sK (16KB)
                  + BN * D * 2        // sV (16KB)
                  + BM * 4            // s_rowmax
                  + BM * 4;           // s_rowsum

    cudaStream_t stream = static_cast<cudaStream_t>(
        TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id));

    cudaFuncSetAttribute(
        attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    attention_kernel<<<grid, block, smem_size, stream>>>(
        Q_data, K_data, V_data, O_data, LSE_data, B, H, S);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, tvm_ffi_example_cuda::run);

}  // namespace tvm_ffi_example_cuda