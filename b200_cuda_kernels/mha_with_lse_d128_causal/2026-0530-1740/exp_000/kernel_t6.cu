#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#define CUDA_CHECK(call) do { cudaError_t _e=(call); if(_e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(_e),__FILE__,__LINE__); exit(1);} } while(0)
#define CU_CHECK(call) do { CUresult _e=(call); if(_e!=CUDA_SUCCESS){ const char* s; cuGetErrorString(_e,&s); \
    fprintf(stderr,"CU error %s at %s:%d\n", s, __FILE__,__LINE__); exit(1);} } while(0)

namespace mha {

constexpr int BM=128, BN=64, ND=128;
constexpr float SCALE = 0.08838834764831843f;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2   = 0.6931471805599453f;
constexpr float SCALE_L2 = SCALE*LOG2E;
constexpr float NEG = -1e30f;

// smem byte offsets (1024-aligned base)
constexpr int O_Qs0=0,      O_Qs1=16384;
constexpr int O_Ks000=32768,O_Ks001=40960, O_Ks010=49152, O_Ks011=57344;
constexpr int O_Vs000=65536,O_Vs001=73728, O_Vs010=81920, O_Vs011=90112;
constexpr int O_Ps=98304;
constexpr int O_BARS=114688;
constexpr int SMEM_REQ = 115840;

__device__ __forceinline__ uint32_t cvtsh(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ float ex2(float x){ float y; asm volatile("ex2.approx.ftz.f32 %0,%1;":"=f"(y):"f"(x)); return y; }

__device__ __forceinline__ uint64_t desc_k(const __nv_bfloat16* region, int sub){
    uint32_t a = cvtsh(region) + (uint32_t)(sub*32);
    uint64_t d=0; d |= (uint64_t)((a & 0x3FFFF) >> 4);
    d |= ((uint64_t)1)<<16; d |= ((uint64_t)64)<<32; d |= ((uint64_t)1)<<46; d |= ((uint64_t)2)<<61; return d;
}
__device__ __forceinline__ uint64_t desc_mn(const __nv_bfloat16* region, int nstep){
    uint32_t a = cvtsh(region) + (uint32_t)(nstep*2048);
    uint64_t d=0; d |= (uint64_t)((a & 0x3FFFF) >> 4);
    d |= ((uint64_t)128)<<16; d |= ((uint64_t)64)<<32; d |= ((uint64_t)1)<<46; d |= ((uint64_t)2)<<61; return d;
}
__device__ __forceinline__ uint32_t idesc_qk(){ uint32_t d=0; d|=(1u<<4); d|=(1u<<7); d|=(1u<<10); d|=((64u/8)<<17); d|=((128u/16)<<24); return d; }
__device__ __forceinline__ uint32_t idesc_pv(){ uint32_t d=0; d|=(1u<<4); d|=(1u<<7); d|=(1u<<10); d|=(1u<<16); d|=((64u/8)<<17); d|=((128u/16)<<24); return d; }

__device__ __forceinline__ void mma1(uint32_t td, uint64_t da, uint64_t db, uint32_t id, uint32_t acc){
    asm volatile("{\n.reg .pred p;\n setp.ne.b32 p, %4, 0;\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
        :: "r"(td), "l"(da), "l"(db), "r"(id), "r"(acc));
}
__device__ __forceinline__ void mma_commit(uint64_t* bar){ asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];" :: "r"(cvtsh(bar))); }
__device__ __forceinline__ void tmem_alloc1(uint32_t* dst, int n){ asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(cvtsh(dst)), "r"(n)); }
__device__ __forceinline__ void tmem_relinquish(){ asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"); }
__device__ __forceinline__ void tmem_dealloc1(uint32_t addr, int n){ asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(addr), "r"(n)); }
__device__ __forceinline__ void wait_ld(){ asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wait_st(){ asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void fence_before(){ asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory"); }
__device__ __forceinline__ void fence_after(){ asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory"); }

__device__ __forceinline__ void tmem_ld32(uint32_t a, float* o){
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=f"(o[0]),"=f"(o[1]),"=f"(o[2]),"=f"(o[3]),"=f"(o[4]),"=f"(o[5]),"=f"(o[6]),"=f"(o[7]),
        "=f"(o[8]),"=f"(o[9]),"=f"(o[10]),"=f"(o[11]),"=f"(o[12]),"=f"(o[13]),"=f"(o[14]),"=f"(o[15]),
        "=f"(o[16]),"=f"(o[17]),"=f"(o[18]),"=f"(o[19]),"=f"(o[20]),"=f"(o[21]),"=f"(o[22]),"=f"(o[23]),
        "=f"(o[24]),"=f"(o[25]),"=f"(o[26]),"=f"(o[27]),"=f"(o[28]),"=f"(o[29]),"=f"(o[30]),"=f"(o[31])
      : "r"(a));
}
__device__ __forceinline__ void tmem_st32(uint32_t a, const float* o){
    asm volatile("tcgen05.st.sync.aligned.32x32b.x32.b32 [%32], "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31};"
      :: "f"(o[0]),"f"(o[1]),"f"(o[2]),"f"(o[3]),"f"(o[4]),"f"(o[5]),"f"(o[6]),"f"(o[7]),
         "f"(o[8]),"f"(o[9]),"f"(o[10]),"f"(o[11]),"f"(o[12]),"f"(o[13]),"f"(o[14]),"f"(o[15]),
         "f"(o[16]),"f"(o[17]),"f"(o[18]),"f"(o[19]),"f"(o[20]),"f"(o[21]),"f"(o[22]),"f"(o[23]),
         "f"(o[24]),"f"(o[25]),"f"(o[26]),"f"(o[27]),"f"(o[28]),"f"(o[29]),"f"(o[30]),"f"(o[31]),
         "r"(a) : "memory");
}

__device__ __forceinline__ void init_bar(uint64_t* b, uint32_t c){ asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(cvtsh(b)), "r"(c)); }
__device__ __forceinline__ void arrive_expect(uint64_t* b, uint32_t tx){ asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(cvtsh(b)), "r"(tx) : "memory"); }
__device__ __forceinline__ void bar_wait(uint64_t* b, uint32_t ph){
    asm volatile("{\n.reg .pred P;\nW%=:\n mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n@!P bra W%=;\n}\n":: "r"(cvtsh(b)), "r"(ph));
}
__device__ __forceinline__ void tma2d(const CUtensorMap* d, uint64_t* bar, void* smem, int c0, int c1){
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
      :: "r"(cvtsh(smem)), "l"((uint64_t)d), "r"(cvtsh(bar)), "r"(c0), "r"(c1) : "memory");
}
__device__ __forceinline__ void st_bf16(char* region, uint32_t row, uint32_t col, __nv_bfloat16 v){
    uint32_t off = row*128 + (((col>>3)^(row&7))<<4) + ((col&7)<<1);
    *reinterpret_cast<__nv_bfloat16*>(region + off) = v;
}

__global__ __launch_bounds__(128,2) void attn(
    const __grid_constant__ CUtensorMap tmaQ, const __grid_constant__ CUtensorMap tmaK, const __grid_constant__ CUtensorMap tmaV,
    __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, int S)
{
    extern __shared__ char smem_raw[];
    char* base = (char*)(((uintptr_t)smem_raw + 1023) & ~((uintptr_t)1023));
    __nv_bfloat16* Qs[2] = {(__nv_bfloat16*)(base+O_Qs0),(__nv_bfloat16*)(base+O_Qs1)};
    __nv_bfloat16* Ks[2][2] = {{(__nv_bfloat16*)(base+O_Ks000),(__nv_bfloat16*)(base+O_Ks001)},{(__nv_bfloat16*)(base+O_Ks010),(__nv_bfloat16*)(base+O_Ks011)}};
    __nv_bfloat16* Vs[2][2] = {{(__nv_bfloat16*)(base+O_Vs000),(__nv_bfloat16*)(base+O_Vs001)},{(__nv_bfloat16*)(base+O_Vs010),(__nv_bfloat16*)(base+O_Vs011)}};
    char* Ps = base+O_Ps;
    uint64_t* bar_k[2] = {(uint64_t*)(base+O_BARS),  (uint64_t*)(base+O_BARS+8)};
    uint64_t* bar_v[2] = {(uint64_t*)(base+O_BARS+16),(uint64_t*)(base+O_BARS+24)};
    uint64_t* bar_qk[2]= {(uint64_t*)(base+O_BARS+32),(uint64_t*)(base+O_BARS+40)};
    uint64_t* bar_pv   = (uint64_t*)(base+O_BARS+48);
    uint32_t* tmem_ptr = (uint32_t*)(base+O_BARS+56);

    int tid = threadIdx.x;
    int warp = tid>>5;
    bool leader = (tid==0);
    int qtile = blockIdx.x, bh = blockIdx.y;
    int qbase = qtile*BM;
    int qrow = qbase + tid;
    int row_off = bh*S;

    if (warp==0){ tmem_alloc1(tmem_ptr, 256); tmem_relinquish(); }
    if (leader){
        init_bar(bar_k[0],1); init_bar(bar_k[1],1);
        init_bar(bar_v[0],1); init_bar(bar_v[1],1);
        init_bar(bar_qk[0],1);init_bar(bar_qk[1],1);
        init_bar(bar_pv,1);
    }
    __syncthreads();
    uint32_t tbase = *tmem_ptr;
    uint32_t S_addr = tbase + 0;     // S0@+0, S1@+64
    uint32_t O_addr = tbase + 128;   // O@+128 (128 cols)
    uint32_t id_qk = idesc_qk(), id_pv = idesc_pv();

    uint32_t ph_k[2]={0,0}, ph_v[2]={0,0}, ph_qk[2]={0,0}, ph_pv=0;

    int total_kb = (S+BN-1)/BN;
    int n = (qbase+BM-1)/BN + 1; if(n>total_kb) n=total_kb;

    // ---- Prologue ----
    if (leader){
        arrive_expect(bar_k[0], 49152);                 // Q(32768) + K0(16384)
        tma2d(&tmaQ, bar_k[0], Qs[0], 0,  row_off+qbase);
        tma2d(&tmaQ, bar_k[0], Qs[1], 64, row_off+qbase);
        tma2d(&tmaK, bar_k[0], Ks[0][0], 0,  row_off+0);
        tma2d(&tmaK, bar_k[0], Ks[0][1], 64, row_off+0);
        arrive_expect(bar_v[0], 16384);
        tma2d(&tmaV, bar_v[0], Vs[0][0], 0,  row_off+0);
        tma2d(&tmaV, bar_v[0], Vs[0][1], 64, row_off+0);
        if (n>1){
            arrive_expect(bar_k[1], 16384);
            tma2d(&tmaK, bar_k[1], Ks[1][0], 0,  row_off+BN);
            tma2d(&tmaK, bar_k[1], Ks[1][1], 64, row_off+BN);
        }
    }
    bar_wait(bar_k[0], ph_k[0]); ph_k[0]^=1;
    if (leader){
        for (int ks=0; ks<8; ks++){ int r=ks>>2, sub=ks&3;
            mma1(S_addr, desc_k(Qs[r],sub), desc_k(Ks[0][r],sub), id_qk, ks==0?0:1); }
        mma_commit(bar_qk[0]);
    }

    float m_run = NEG, l_run = 0.f;

    for (int i=0;i<n;i++){
        int cur=i&1, nxt=(i+1)&1;
        uint32_t S_cur = S_addr + cur*64;
        int kbase=i*BN;

        bar_wait(bar_qk[cur], ph_qk[cur]); ph_qk[cur]^=1;
        float s[64];
        tmem_ld32(S_cur+0, s); tmem_ld32(S_cur+32, s+32);
        wait_ld();

        // QK(i+1) -> S[nxt], overlaps softmax/rescale
        if (i+1<n){
            bar_wait(bar_k[nxt], ph_k[nxt]); ph_k[nxt]^=1;
            if (leader){
                uint32_t S_nxt = S_addr + nxt*64;
                for (int ks=0; ks<8; ks++){ int r=ks>>2, sub=ks&3;
                    mma1(S_nxt, desc_k(Qs[r],sub), desc_k(Ks[nxt][r],sub), id_qk, ks==0?0:1); }
                mma_commit(bar_qk[nxt]);
                if (i+2<n){
                    arrive_expect(bar_k[cur], 16384);
                    tma2d(&tmaK, bar_k[cur], Ks[cur][0], 0,  row_off+(i+2)*BN);
                    tma2d(&tmaK, bar_k[cur], Ks[cur][1], 64, row_off+(i+2)*BN);
                }
            }
        }

        // softmax(i): max
        float rowmax=NEG;
        #pragma unroll
        for (int nn=0;nn<64;nn++){
            int kpos=kbase+nn;
            float u=s[nn]*SCALE_L2;
            if (kpos>qrow||kpos>=S) u=NEG;
            s[nn]=u; rowmax=fmaxf(rowmax,u);
        }
        float m_new=fmaxf(m_run,rowmax);
        float corr=(m_run<=-1e29f)?0.f:ex2(m_run-m_new);

        // wait PV(i-1): frees O for rescale, frees Ps for overwrite
        if (i>0){ bar_wait(bar_pv, ph_pv); ph_pv^=1; }

        // write P
        float lsum=0.f;
        #pragma unroll
        for (int nn=0;nn<64;nn++){
            float p=ex2(s[nn]-m_new);
            lsum+=p;
            st_bf16(Ps,(uint32_t)tid,(uint32_t)nn,__float2bfloat16(p));
        }
        l_run=l_run*corr+lsum;

        // rescale O by corr (i>0)
        if (i>0){
            #pragma unroll
            for (int c=0;c<4;c++){
                float o[32]; tmem_ld32(O_addr+c*32,o); wait_ld();
                #pragma unroll
                for (int j=0;j<32;j++) o[j]*=corr;
                tmem_st32(O_addr+c*32,o);
            }
            wait_st();
            fence_before();
        }
        __syncthreads();
        asm volatile("fence.proxy.async.shared::cta;\n":::"memory");

        // PV(i): Ps x V(cur) accumulate -> O
        bar_wait(bar_v[cur], ph_v[cur]); ph_v[cur]^=1;
        if (leader){
            fence_after();
            for (int hd=0;hd<2;hd++){
                for (int ns=0;ns<4;ns++){
                    uint32_t acc = (i==0 && ns==0)?0:1;
                    mma1(O_addr+hd*64, desc_k((const __nv_bfloat16*)Ps,ns),
                         desc_mn(Vs[cur][hd],ns), id_pv, acc);
                }
            }
            mma_commit(bar_pv);
            // prefetch V(i+1) into Vs[nxt]
            if (i+1<n){
                arrive_expect(bar_v[nxt], 16384);
                tma2d(&tmaV, bar_v[nxt], Vs[nxt][0], 0,  row_off+(i+1)*BN);
                tma2d(&tmaV, bar_v[nxt], Vs[nxt][1], 64, row_off+(i+1)*BN);
            }
        }

        m_run=m_new;
    }

    // epilogue
    bar_wait(bar_pv, ph_pv); ph_pv^=1;
    float o[128];
    tmem_ld32(O_addr+0,o); tmem_ld32(O_addr+32,o+32); tmem_ld32(O_addr+64,o+64); tmem_ld32(O_addr+96,o+96);
    wait_ld();
    if (qrow<S){
        float inv=1.f/l_run;
        __nv_bfloat16* op = O + (int64_t)(row_off+qrow)*128;
        #pragma unroll
        for (int d=0; d<128; d++) op[d]=__float2bfloat16(o[d]*inv);
        LSE[(int64_t)row_off+qrow] = m_run*LN2 + logf(l_run);
    }

    __syncthreads();
    if (warp==0) tmem_dealloc1(tbase, 256);
}

static CUresult make_tma(CUtensorMap* d, void* ptr, uint64_t inner, uint64_t outer,
                         uint32_t binner, uint32_t bouter, CUtensorMapSwizzle sw){
    uint64_t gdim[2]={inner, outer}; uint64_t gstr[1]={inner*2};
    uint32_t bdim[2]={binner, bouter}; uint32_t estr[2]={1,1};
    return cuTensorMapEncodeTiled(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, ptr, gdim, gstr,
        bdim, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

void run(tvm::ffi::TensorView Q, tvm::ffi::TensorView K, tvm::ffi::TensorView V,
         tvm::ffi::TensorView O, tvm::ffi::TensorView LSE){
    CUDA_CHECK(cudaSetDevice(Q.device().device_id));
    int B=Q.size(0), H=Q.size(1), S=Q.size(2);
    int64_t BH=(int64_t)B*H; int64_t outer=BH*S;

    __nv_bfloat16* Qp=(__nv_bfloat16*)Q.data_ptr();
    __nv_bfloat16* Kp=(__nv_bfloat16*)K.data_ptr();
    __nv_bfloat16* Vp=(__nv_bfloat16*)V.data_ptr();
    __nv_bfloat16* Op=(__nv_bfloat16*)O.data_ptr();
    float* LSEp=(float*)LSE.data_ptr();

    CUtensorMap tmaQ,tmaK,tmaV;
    CU_CHECK(make_tma(&tmaQ, Qp, ND, outer, 64, 128, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(make_tma(&tmaK, Kp, ND, outer, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B));
    CU_CHECK(make_tma(&tmaV, Vp, ND, outer, 64, 64, CU_TENSOR_MAP_SWIZZLE_128B));

    dim3 grid((S+BM-1)/BM, BH); dim3 block(128);
    cudaStream_t stream=(cudaStream_t)TVMFFIEnvGetStream(Q.device().device_type, Q.device().device_id);
    cudaFuncSetAttribute((const void*)attn, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_REQ);
    attn<<<grid, block, SMEM_REQ, stream>>>(tmaQ, tmaK, tmaV, Op, LSEp, S);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, mha::run);

} // namespace mha