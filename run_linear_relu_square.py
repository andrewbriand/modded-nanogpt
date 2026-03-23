"""
Self-contained harness for the linear_relu_square kernel.

Generates readable PTX programmatically, then compiles and runs it.

Kernel semantics:
    C[m,n]   = A[m,:] @ B[n,:]^T          (bf16 inputs, bf16 output)
    aux[m,n] = relu(C[m,n])^2             (fused activation, bf16 output)

Tile shape  : M=128, N=256, K=64 per block
CTA shape   : 256 threads (8 warps), sm_90a
Pipeline    : 3-stage TMA prefetch with mbarriers
SMEM layout : 213,016 bytes
  smem_base + 0       : B pipeline  3 x 256x64 bf16 = 3 x 32768 B
  smem_base + 98304   : A pipeline  3 x 128x64 bf16 = 3 x 16384 B
  smem_base + 163840  : A wgmma window (= 98304 + stage*16384, stage in 0..2)
  smem_base + 212992  : mbarrier[0]  (8 bytes)
  smem_base + 213000  : mbarrier[1]  (8 bytes)
  smem_base + 213008  : mbarrier[2]  (8 bytes)

The C/aux output staging buffer reuses the A pipeline region (98304..163839),
written via stmatrix then evicted via TMA store to global.
"""
import struct

# ============================================================
# PTX generator
# ============================================================

def I(s):
    """Indent a PTX instruction line by one tab."""
    return "\t" + s


def generate_ptx():
    lines = []
    def L(s=""):
        lines.append(s)

    # ---- Header ----------------------------------------------------------------
    L(".version 8.7")
    L(".target sm_90a")
    L(".address_size 64")
    L()
    L(".extern .shared .align 16 .b8 global_smem[];")
    L()

    # ---- Kernel signature ------------------------------------------------------
    # 25 parameters matching the Triton-generated ABI exactly.
    # Descriptors are 128-byte structs passed by value (.param .align 64 .b8 [128]).
    L(".visible .entry linear_relu_square_kernel(")
    L("\t.param .align 64 .b8  param_desc_A[128],")    # 0
    L("\t.param .u32           param_A_row_step,")      # 1  (unused at call site)
    L("\t.param .u32           param_A_col_step,")      # 2  (unused at call site)
    L("\t.param .u64           param_A_stride_bytes,")  # 3
    L("\t.param .u64           param_A_base_ptr,")      # 4
    L("\t.param .align 64 .b8  param_desc_B[128],")    # 5
    L("\t.param .u32           param_B_row_step,")      # 6
    L("\t.param .u32           param_B_col_step,")      # 7
    L("\t.param .u64           param_B_stride_bytes,")  # 8
    L("\t.param .u64           param_B_base_ptr,")      # 9
    L("\t.param .align 64 .b8  param_desc_C[128],")    # 10
    L("\t.param .u32           param_C_row_step,")      # 11
    L("\t.param .u32           param_C_col_step,")      # 12
    L("\t.param .u64           param_C_stride_bytes,")  # 13
    L("\t.param .u64           param_C_base_ptr,")      # 14
    L("\t.param .align 64 .b8  param_desc_aux[128],")  # 15
    L("\t.param .u32           param_aux_row_step,")    # 16
    L("\t.param .u32           param_aux_col_step,")    # 17
    L("\t.param .u64           param_aux_stride_bytes,")# 18
    L("\t.param .u64           param_aux_base_ptr,")    # 19
    L("\t.param .u32           param_M,")               # 20
    L("\t.param .u32           param_N,")               # 21
    L("\t.param .u32           param_K,")               # 22
    L("\t.param .u64 .ptr .global .align 1 param_unused0,") # 23
    L("\t.param .u64 .ptr .global .align 1 param_unused1")  # 24
    L(")")
    L(".reqntid 256")
    L("{")

    # ---- Register declarations -------------------------------------------------
    # We keep the same pool sizes as the original to preserve all register numbers.
    # Named meaning for key registers is documented in comments below.
    L("\t.reg .pred \t%p<53>;")
    L("\t.reg .b16  \t%rs<129>;")
    L("\t.reg .b32  \t%r<821>;")
    L("\t.reg .b64  \t%rd<23>;")
    L()

    # Key register names (all regs keep their original numeric IDs)
    L("\t// ---- Scalar parameters ----")
    L("\t// %r32  = K")
    L("\t// %r33  = M,  %r39 = N")
    L("\t// %r38  = num_M_tiles = cdiv(M,128)")
    L("\t// %r1   = num_N_tiles = cdiv(N,256)")
    L("\t// %r2   = num_tiles   = num_M_tiles * num_N_tiles")
    L("\t// %r54  = num_K_blocks = cdiv(K,64)")
    L("\t// %r6   = total_loop_iters,  %r7 = num_K_blocks - 1")
    L("\t// ---- Thread/CTA identifiers ----")
    L("\t// %r679 = tile_id (current CTA's tile, persistent scheduler)")
    L("\t// %r8   = tid,  %r9 = tid&255,  %r11 = warp_id = tid>>5")
    L("\t// %p45  = is_leader_thread (tid&255 == 0)")
    L("\t// %p11  = is_tma_warp (warp 0, lanes 0-31)")
    L("\t// ---- Shared memory ----")
    L("\t// %r49  = smem_base = &global_smem[0]")
    L("\t// %r45  = &mbarrier[0]  = smem_base + 212992")
    L("\t// %r72  = &mbarrier[1]  = smem_base + 213000")
    L("\t// %r46  = &mbarrier[2]  = smem_base + 213008")
    L("\t// ---- Output tile coordinates ----")
    L("\t// %r681 = offs_am  (current tile row offset into A/C, = pid_m * 128)")
    L("\t// %r680 = offs_bn  (current tile col offset into B/C, = pid_n * 256)")
    L("\t// %r688 = prefetch_offs_am  (next tile to prefetch)")
    L("\t// %r687 = prefetch_offs_bn")
    L("\t// ---- Pipeline ring state ----")
    L("\t// %r685 = consume_stage  (0..2, which mbarrier/smem slot to read from)")
    L("\t// %r686 = produce_stage  (0..2, which slot to prefetch into)")
    L("\t// %r684 = barrier_parity (flips each time ring wraps)")
    L("\t// %r683 = k_block_in_tile (position within current tile's K loop)")
    L("\t// %r682 = k_consume_offset = k_block_in_tile * 64")
    L("\t// ---- Accumulator (128 x f32, registers acc[0..127]) ----")
    L("\t// acc[i] = %r{689+i} for i in 0..127")
    L("\t// Each thread holds a fragment of the wgmma.m64n256k16 output.")
    L("\t// ---- Epilogue temporaries ----")
    L("\t// For each acc pair i in 0..63 (processing acc[2i] and acc[2i+1]):")
    L("\t//   c_reg[i]   = %r{139+8*i}  bf16x2 packed linear output")
    L("\t//   aux_reg[i] = %r{146+8*i}  bf16x2 packed relu^2 output")
    L("\t//   intermediates at %r{140..145+8*i}: f32 round-trip for relu^2")
    L("\t// stmatrix writes 16 groups of 4 c_reg/aux_reg to the staging buffer.")
    L()

    # ---- Helpers for generating typed register names --------------------------
    def acc(i):        return f"%r{689 + i}"          # accumulator f32
    def c_reg(i):      return f"%r{139 + 8*i}"        # bf16x2 linear output
    def aux_reg(i):    return f"%r{146 + 8*i}"        # bf16x2 relu^2 output
    def f_lo(i):       return f"%r{140 + 8*i}"        # f32 round-trip lo
    def f_hi(i):       return f"%r{141 + 8*i}"        # f32 round-trip hi
    def relu_hi(i):    return f"%r{142 + 8*i}"
    def relu_lo(i):    return f"%r{143 + 8*i}"
    def sq_lo(i):      return f"%r{144 + 8*i}"
    def sq_hi(i):      return f"%r{145 + 8*i}"
    def rs_hi(i):      return f"%rs{1  + 2*i}"        # bf16 hi element
    def rs_lo(i):      return f"%rs{2  + 2*i}"        # bf16 lo element

    # stmatrix address registers: %r12..%r27 (16 addresses)
    # Mapping: addr_k → c_reg indices = [col_group*4+j + row_group*16 for j in 0..3]
    # where col_group = k//4, row_group = k%4
    def stmatrix_indices(k):
        col_group = k // 4
        row_group = k % 4
        return [col_group * 4 + j + row_group * 16 for j in range(4)]

    # All 128 accumulator registers as a comma-separated string (for wgmma inline asm)
    all_acc = ",".join(acc(i) for i in range(128))

    # ===================================================================
    # Entry block
    # ===================================================================
    L("$L__func_begin0:")
    L()
    L("// %bb.0: entry — load scalar params, compute grid dimensions")
    L(I("ld.param.b32 \t%r32, [param_K];"))
    L(I("mov.b64      \t%rd5, param_desc_A;"))
    L(I("mov.b64      \t%rd6, param_desc_aux;"))
    L("$L__tmp0:")
    L(I("cvta.param.u64 \t%rd1, %rd6;              // rd1 = generic ptr to aux descriptor"))
    L(I("mov.b64        \t%rd7, param_desc_C;"))
    L(I("cvta.param.u64 \t%rd2, %rd7;              // rd2 = generic ptr to C descriptor"))
    L(I("mov.b64        \t%rd8, param_desc_B;"))
    L(I("cvta.param.u64 \t%rd3, %rd8;              // rd3 = generic ptr to B descriptor"))
    L(I("cvta.param.u64 \t%rd4, %rd5;              // rd4 = generic ptr to A descriptor"))
    L(I("mov.u32        \t%r679, %ctaid.x;          // tile_id = blockIdx.x"))
    L(I("ld.param.b32   \t%r33, [param_M];"))
    L("$L__tmp1:")
    L(I("// num_M_tiles = cdiv(M, 128)  via arithmetic right-shift trick"))
    L(I("add.s32 \t%r34, %r33, 127;"))
    L(I("shr.s32 \t%r35, %r34, 31;"))
    L(I("shr.u32 \t%r36, %r35, 25;"))
    L(I("add.s32 \t%r37, %r34, %r36;"))
    L(I("shr.s32 \t%r38, %r37, 7;                  // r38 = num_M_tiles"))
    L(I("ld.param.b32 \t%r39, [param_N];"))
    L("$L__tmp2:")
    L(I("// num_N_tiles = cdiv(N, 256)"))
    L(I("add.s32 \t%r40, %r39, 255;"))
    L(I("shr.s32 \t%r41, %r40, 31;"))
    L(I("shr.u32 \t%r42, %r41, 24;"))
    L(I("add.s32 \t%r43, %r40, %r42;"))
    L(I("shr.s32 \t%r1,  %r43, 8;                  // r1 = num_N_tiles"))
    L("$L__tmp3:")
    L(I("mul.lo.s32 \t%r2, %r1, %r38;              // num_tiles = num_M_tiles * num_N_tiles"))
    L(I("// Branch to main GEMM path if K > 0"))
    L(I("add.s32         \t%r44, %r32, 126;"))
    L(I("setp.gt.u32     \t%p1, %r44, 126;          // p1 = (K > 0)"))
    L(I("@%p1 bra        \t$L__BB0_5;"))

    # ===================================================================
    # K=0 path: write zeros to C and aux outputs
    # ===================================================================
    L()
    L("// K==0 path: no GEMM needed, zero-fill C and aux via TMA store")
    L("// %bb.1:")
    L(I("setp.le.s32 \t%p46, %r2, %r679;           // skip if tile_id >= num_tiles"))
    L(I("@%p46 bra   \t$L__BB0_4;"))
    L()
    L("// %bb.2: compute per-thread store address in the staging buffer")
    L(I("mov.u32 \t%r653, %tid.x;"))
    L(I("and.b32 \t%r3,   %r653, 128;              // r3 = tid & 128 (wg selector bit)"))
    L(I("// Swizzled smem offset for this thread (matches 128B-swizzle layout)"))
    L(I("shl.b32 \t%r654, %r653, 11;"))
    L(I("shl.b32 \t%r655, %r653, 4;"))
    L(I("or.b32  \t%r656, %r654, %r655;"))
    L(I("and.b32 \t%r657, %r653, 224;"))
    L(I("shl.b32 \t%r658, %r657, 2;"))
    L(I("shr.u32 \t%r659, %r657, 1;"))
    L(I("and.b32 \t%r660, %r656, 49264;"))
    L(I("or.b32  \t%r661, %r658, %r659;"))
    L(I("xor.b32 \t%r662, %r660, %r661;            // swizzled_thread_offset"))
    L(I("mov.b32 \t%r663, global_smem;"))
    L(I("add.s32 \t%r4,   %r663, %r662;            // smem store address for this thread"))
    L(I("shr.u32 \t%r5,   %r653, 5;                // warp_id = tid >> 5"))
    L(I("shl.b32 \t%r677, %r679, 8;                // tile_id * 256 (C col base scaled)"))
    L(I("setp.eq.b32 \t%p49, %r3, 0;               // p49 = (tid < 128), wg0"))
    L()
    L("$L__BB0_3:  // K==0 tile loop — one iteration per output tile")
    L(I("div.s32 \t%r667, %r679, %r1;              // pid_m = tile_id / num_N_tiles"))
    L(I("shl.b32 \t%r665, %r667, 7;               // offs_am = pid_m * 128"))
    L()
    L(I("// Zero-fill the C staging buffer, then TMA store zeros to global C"))
    L(I("cp.async.bulk.wait_group.read \t0;        // wait for prior TMA store"))
    L(I("bar.sync \t0;"))
    for offset in range(0, 16384, 1024):  # 16 stores × v4.b32 × 16B = 16384 bytes / thread
        L(I(f"st.shared.v4.b32 \t[%r4+{offset}], {{0, 0, 0, 0}};"))
    L(I("// begin inline asm"))
    L(I("fence.proxy.async.shared::cta;"))
    L(I("// end inline asm"))
    L(I("bar.sync \t0;"))
    L(I("elect.sync    \t%r668|%p50, -1;"))
    L(I("shfl.sync.idx.b32 \t%r669, %r5, 0, 31, -1;  // broadcast warp_id"))
    L(I("and.pred      \t%p47, %p49, %p50;           // wg0 & elected"))
    L(I("and.b32       \t%r670, %r669, 3;"))
    L(I("shl.b32       \t%r671, %r670, 14;"))
    L(I("add.s32       \t%r666, %r663, %r671;        // smem source addr for TMA"))
    L(I("shl.b32       \t%r672, %r670, 6;"))
    L(I("mul.lo.s32    \t%r673, %r1, %r667;"))
    L(I("shl.b32       \t%r674, %r673, 8;"))
    L(I("sub.s32       \t%r675, %r672, %r674;"))
    L(I("add.s32       \t%r664, %r677, %r675;        // C global col coordinate"))
    L(I("// begin inline asm"))
    L(I("@%p47 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd2, {%r664, %r665}], [%r666];"))
    L(I("// end inline asm"))
    L(I("cp.async.bulk.commit_group;"))
    L()
    L(I("// Same for aux output"))
    L(I("cp.async.bulk.wait_group.read \t0;"))
    L(I("bar.sync \t0;"))
    for offset in range(0, 16384, 1024):
        L(I(f"st.shared.v4.b32 \t[%r4+{offset}], {{0, 0, 0, 0}};"))
    L(I("// begin inline asm"))
    L(I("fence.proxy.async.shared::cta;"))
    L(I("// end inline asm"))
    L(I("bar.sync \t0;"))
    L(I("elect.sync \t%r676|%p51, -1;"))
    L(I("and.pred   \t%p48, %p49, %p51;"))
    L(I("// begin inline asm"))
    L(I("@%p48 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd1, {%r664, %r665}], [%r666];"))
    L(I("// end inline asm"))
    L(I("cp.async.bulk.commit_group;"))
    L()
    L(I("add.s32       \t%r679, %r679, 132;          // advance tile_id (grid stride = 132)"))
    L(I("add.s32       \t%r677, %r677, 33792;        // 132 * 256"))
    L(I("setp.lt.s32   \t%p52, %r679, %r2;"))
    L(I("@%p52 bra     \t$L__BB0_3;"))
    L()
    L("$L__BB0_4:  // K==0 path exit")
    L(I("cp.async.bulk.wait_group.read \t0;"))
    L(I("bar.sync \t0;"))
    L(I("bra.uni \t$L__BB0_15;"))

    # ===================================================================
    # Main GEMM path (K > 0)
    # ===================================================================
    L()
    L("$L__BB0_5:  // K > 0 — pipelined GEMM path")
    L("$L__tmp4:")
    L(I("// num_K_blocks = cdiv(K, 64)"))
    L(I("add.s32 \t%r50, %r32, 63;"))
    L(I("shr.s32 \t%r51, %r50, 31;"))
    L(I("shr.u32 \t%r52, %r51, 26;"))
    L(I("add.s32 \t%r53, %r50, %r52;"))
    L(I("shr.s32 \t%r54, %r53, 6;                  // r54 = num_K_blocks"))
    L("$L__tmp5:")
    L()
    L(I("// Compute total loop iterations for this CTA's persistent tile slice"))
    L(I("sub.s32       \t%r55, %r2, %r679;          // tiles remaining from this CTA"))
    L(I("mul.hi.s32    \t%r56, %r55, 1041204193;    // cdiv(r55, 132) via magic multiply"))
    L(I("shr.u32       \t%r57, %r56, 31;"))
    L(I("shr.s32       \t%r58, %r56, 5;"))
    L(I("add.s32       \t%r59, %r58, %r57;"))
    L(I("mul.lo.s32    \t%r60, %r59, 132;"))
    L(I("setp.ne.b32   \t%p5,  %r55, %r60;"))
    L(I("setp.gt.s32   \t%p6,  %r55, -1;"))
    L(I("and.pred      \t%p7,  %p6, %p5;"))
    L(I("selp.b32      \t%r61, 1, 0, %p7;"))
    L(I("add.s32       \t%r62, %r59, %r61;"))
    L(I("max.s32       \t%r63, %r54, 1;             // max(num_K_blocks, 1)"))
    L(I("mul.lo.s32    \t%r6,  %r62, %r63;          // total_loop_iters"))
    L(I("add.s32       \t%r7,  %r63, -1;            // last k-index within a tile"))
    L()
    L(I("mov.u32       \t%r8,  %tid.x;"))
    L(I("and.b32       \t%r9,  %r8,  255;"))
    L(I("setp.eq.b32   \t%p45, %r9,  0;             // p45 = leader thread"))
    L(I("mov.b32       \t%r49, global_smem;          // smem_base"))
    L()
    L(I("// Init 3 mbarriers (one per pipeline stage) — leader thread only"))
    for stage, offset in enumerate([212992, 213000, 213008]):
        reg = [45, 72, 46][stage]
        L(I(f"add.s32 \t%r{reg}, %r49, {offset};   // &mbarrier[{stage}]"))
        L(I(f"// begin inline asm"))
        L(I(f"@%p45 mbarrier.init.shared::cta.b64 [%r{reg}], 1;"))
        L(I(f"// end inline asm"))
        if stage < 2:
            L(I("bar.sync \t0;"))
    L()
    L(I("setp.gt.s32   \t%p8, %r6, 0;              // p8 = has work"))
    L()
    L(I("// Initial output tile coordinates for this CTA"))
    L(I("div.s32       \t%r64, %r679, %r1;          // pid_m"))
    L(I("mul.lo.s32    \t%r65, %r64, %r1;"))
    L(I("sub.s32       \t%r66, %r679, %r65;         // pid_n"))
    L(I("shl.b32       \t%r681, %r64, 7;            // offs_am = pid_m * 128"))
    L(I("shl.b32       \t%r680, %r66, 8;            // offs_bn = pid_n * 256"))
    L()

    # ---- Prologue: issue stage 0 prefetch -------------------------------------
    L(I("// --- Prologue: prefetch stage 0 (A[k=0], B[k=0]) ---"))
    L(I("bar.sync \t0;"))
    L(I("and.pred \t%p2, %p45, %p8;"))
    L(I("// begin inline asm"))
    L(I("@%p2 mbarrier.arrive.expect_tx.shared.b64 _, [%r45], 49152; // expect 48KB = A+B"))
    L(I("// end inline asm"))
    L(I("// TMA load A[stage=0] at smem offset 163840"))
    L(I("bar.sync \t0;"))
    L(I("elect.sync    \t%r67|%p9, -1;"))
    L(I("and.pred      \t%p10, %p8, %p9;"))
    L(I("setp.lt.u32   \t%p11, %r9, 32;             // p11 = first warp (TMA warp)"))
    L(I("and.pred      \t%p3,  %p11, %p10;"))
    L(I("add.s32       \t%r47, %r49, 163840;"))
    L(I("mov.b32       \t%r48, 0;                    // k_offset = 0"))
    L(I("// begin inline asm"))
    L(I("@%p3 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r47], [%rd4, {%r48, %r681}], [%r45];"))
    L(I("// end inline asm"))
    L(I("// TMA load B[stage=0] at smem offset 0"))
    L(I("bar.sync \t0;"))
    L(I("elect.sync    \t%r68|%p12, -1;"))
    L(I("and.pred      \t%p13, %p8, %p12;"))
    L(I("and.pred      \t%p4,  %p11, %p13;"))
    L(I("// begin inline asm"))
    L(I("@%p4 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r49], [%rd3, {%r48, %r680}], [%r45];"))
    L(I("// end inline asm"))
    L()
    L(I("// If stage 0 is the last k-block, the next prefetch is for the first block"))
    L(I("// of the next output tile; otherwise it's k=1 of the same tile."))
    L(I("setp.ne.b32   \t%p14, %r7, 0;              // p14 = (num_K_blocks > 1)"))
    L(I("mov.b32       \t%r678, 64;                  // default: next k_offset = 64"))
    L(I("mov.b32       \t%r688, %r681;              // prefetch_offs_am = current"))
    L(I("mov.b32       \t%r687, %r680;"))
    L(I("@%p14 bra     \t$L__BB0_7;"))
    L()
    L("// %bb.6: k=0 is last block → next prefetch is k=0 of next tile")
    L(I("add.s32       \t%r679, %r679, 132;"))
    L(I("div.s32       \t%r69, %r679, %r1;"))
    L(I("mul.lo.s32    \t%r70, %r69, %r1;"))
    L(I("sub.s32       \t%r71, %r679, %r70;"))
    L(I("shl.b32       \t%r688, %r69, 7;"))
    L(I("shl.b32       \t%r687, %r71, 8;"))
    L(I("mov.b32       \t%r678, %r48;               // next k_offset = 0"))
    L()

    # ---- Prologue: issue stage 1 prefetch -------------------------------------
    L("$L__BB0_7:")
    L(I("// --- Prologue: prefetch stage 1 ---"))
    L(I("setp.gt.s32   \t%p18, %r6, 1;              // p18 = total_iters > 1"))
    L(I("setp.lt.s32   \t%p19, %r6, 1;              // p19 = total_iters < 1 (empty)"))
    L(I("bar.sync \t0;"))
    L(I("and.pred      \t%p15, %p45, %p18;"))
    L(I("// begin inline asm"))
    L(I("@%p15 mbarrier.arrive.expect_tx.shared.b64 _, [%r72], 49152;"))
    L(I("// end inline asm"))
    L(I("// TMA load A[stage=1] at smem offset 180224 = 163840 + 16384"))
    L(I("bar.sync \t0;"))
    L(I("elect.sync    \t%r75|%p20, -1;"))
    L(I("and.pred      \t%p21, %p18, %p20;"))
    L(I("and.pred      \t%p16, %p11, %p21;"))
    L(I("add.s32       \t%r73, %r49, 180224;"))
    L(I("// begin inline asm"))
    L(I("@%p16 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r73], [%rd4, {%r678, %r688}], [%r72];"))
    L(I("// end inline asm"))
    L(I("// TMA load B[stage=1] at smem offset 32768"))
    L(I("bar.sync \t0;"))
    L(I("elect.sync    \t%r76|%p22, -1;"))
    L(I("and.pred      \t%p23, %p18, %p22;"))
    L(I("and.pred      \t%p17, %p11, %p23;"))
    L(I("add.s32       \t%r74, %r49, 32768;"))
    L(I("// begin inline asm"))
    L(I("@%p17 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r74], [%rd3, {%r678, %r687}], [%r72];"))
    L(I("// end inline asm"))
    L()

    # ---- Initialize accumulator to zero ----------------------------------------
    L(I("// Initialize 128-element f32 accumulator to 0.0"))
    L(I(f"mov.b32 \t{acc(0)}, 0f00000000;"))
    for i in range(1, 128):
        L(I(f"mov.b32 \t{acc(i)}, {acc(0)};"))
    L()
    L(I("// Pipeline ring state init"))
    L(I("mov.b32 \t%r686, 1;   // produce_stage = 1 (stage 0 and 1 already issued)"))
    L(I("mov.b32 \t%r685, -1;  // consume_stage starts at -1 (pre-increments to 0)"))
    L(I("mov.b32 \t%r682, 0;   // k_consume_offset = 0"))
    L(I("mov.b32 \t%r684, %r682;  // barrier_parity = 0"))
    L()
    L(I("@%p19 bra \t$L__BB0_14;  // skip loop entirely if total_iters < 1"))

    # ===================================================================
    # Loop setup block (runs once before loop entry)
    # ===================================================================
    L()
    L("// %bb.8: pre-loop — compute per-thread stmatrix addresses into staging buffer")
    L(I("selp.b32 \t%r683, 1, 0, %p14;  // k_block_in_tile initial value"))
    L(I("add.s32  \t%r10,  %r6, -2;     // prefetch lookahead bound"))
    L(I("shr.u32  \t%r11,  %r8, 5;      // warp_id = tid >> 5"))
    L()
    L(I("// Compute swizzled per-thread offset into the C/aux output staging buffer."))
    L(I("// The staging buffer overlaps with the A smem region (offset 98304)."))
    L(I("// This swizzle matches the 128B-swizzled nvmma_shared layout for stmatrix."))
    L(I("shl.b32 \t%r77, %r8, 7;"))
    L(I("and.b32 \t%r78, %r77, 1920;"))
    L(I("shl.b32 \t%r79, %r8, 6;"))
    L(I("and.b32 \t%r80, %r79, 14336;"))
    L(I("shl.b32 \t%r81, %r8, 4;"))
    L(I("and.b32 \t%r82, %r81, 112;"))
    L(I("and.b32 \t%r83, %r8,  16;"))
    L(I("or.b32  \t%r84, %r78, %r82;"))
    L(I("xor.b32 \t%r85, %r84, %r83;"))
    L(I("or.b32  \t%r86, %r85, %r80;   // swizzled_offset"))
    L(I("add.s32 \t%r87, %r49, %r86;   // smem_base + swizzled_offset"))
    L()
    L(I("// 16 stmatrix destination addresses — 4 groups × 4 row-strides of 16384 bytes."))
    L(I("// Group g (g=0..3) is xor'd from swizzled_offset by g*32."))
    L(I("// Within each group, 4 addresses are at +98304, +114688, +131072, +147456."))
    # Group 0: base = r87 (no xor)
    for row in range(4):
        off = 98304 + row * 16384
        L(I(f"add.s32 \t%r{12 + row}, %r87, {off};   // stmatrix_addr[grp=0][row={row}]"))
    # Groups 1..3: xor by 32, 64, 96
    for g in range(1, 4):
        xor_val = g * 32
        L(I(f"xor.b32 \t%r{88 + (g-1)*2}, %r86, {xor_val};"))
        L(I(f"add.s32 \t%r{89 + (g-1)*2}, %r49, %r{88 + (g-1)*2};"))
        base_r = 89 + (g-1)*2
        for row in range(4):
            off = 98304 + row * 16384
            L(I(f"add.s32 \t%r{12 + g*4 + row}, %r{base_r}, {off};   // stmatrix_addr[grp={g}][row={row}]"))
    L()

    # Re-initialize accumulators and loop state (matches original PTX structure)
    L(I(f"mov.b32 \t{acc(0)}, 0f00000000;"))
    L(I("mov.b32 \t%r686, 1;"))
    L(I("mov.b32 \t%r685, -1;"))
    L(I("mov.b32 \t%r682, 0;"))
    L(I("mov.b32 \t%r684, %r682;"))
    for i in range(1, 128):
        L(I(f"mov.b32 \t{acc(i)}, {acc(0)};"))
    L(I("mov.b32 \t%r817, %r683;   // loop copy of k_block_in_tile"))
    L(I("mov.b32 \t%r818, %r682;   // loop_iter counter"))
    L(I("mov.b32 \t%r819, %r688;   // loop copy of prefetch_offs_am"))
    L(I("mov.b32 \t%r820, %r687;   // loop copy of prefetch_offs_bn"))
    L(I("bra.uni \t$L__BB0_9;"))

    # ===================================================================
    # Loop back-edge
    # ===================================================================
    L()
    L("$L__BB0_13:  // loop back-edge — update live-out loop-carried values")
    L(I("add.s32       \t%r818, %r818, 1;           // loop_iter++"))
    L(I("setp.ne.b32   \t%p44, %r6, %r818;          // p44 = not last iteration"))
    L(I("mov.b32       \t%r680, %r687;              // offs_bn = prefetch_offs_bn"))
    L(I("mov.b32       \t%r681, %r688;              // offs_am = prefetch_offs_am"))
    L(I("mov.b32       \t%r682, %r817;"))
    L(I("mov.b32       \t%r687, %r820;"))
    L(I("mov.b32       \t%r688, %r819;"))
    L(I("mov.b32       \t%r817, %r28;               // k_block_in_tile = next value"))
    L(I("@%p44 bra     \t$L__BB0_9;"))
    L(I("bra.uni       \t$L__BB0_14;"))

    # ===================================================================
    # Main loop body
    # ===================================================================
    L()
    L("$L__BB0_9:  // loop body — consume one pipeline stage, run wgmma, prefetch next")
    L()
    L(I("// Advance k_block_in_tile ring: wraps at num_K_blocks"))
    L(I("add.s32       \t%r94, %r817, 1;"))
    L(I("setp.eq.b32   \t%p24, %r817, %r7;          // p24 = this was last k-block"))
    L(I("selp.b32      \t%r28, 0, %r94, %p24;       // r28 = next k_block_in_tile (0 on wrap)"))
    L(I("setp.ne.b32   \t%p25, %r28, 0;"))
    L(I("@%p25 bra     \t$L__BB0_11;"))
    L()
    L("// %bb.10: tile boundary — advance tile_id and compute next tile coordinates")
    L(I("add.s32       \t%r679, %r679, 132;"))
    L(I("div.s32       \t%r95, %r679, %r1;"))
    L(I("mul.lo.s32    \t%r96, %r95, %r1;"))
    L(I("sub.s32       \t%r97, %r679, %r96;"))
    L(I("shl.b32       \t%r819, %r95, 7;            // new prefetch_offs_am"))
    L(I("shl.b32       \t%r820, %r97, 8;            // new prefetch_offs_bn"))
    L()
    L("$L__BB0_11:  // wait on consume stage, then run wgmma")
    L(I("setp.eq.b32   \t%p30, %r28, 0;             // p30 = next k_block == 0 (tile boundary)"))
    L(I("setp.lt.s32   \t%p31, %r818, %r10;         // p31 = room for one more prefetch"))
    L()
    L(I("// Advance consume ring: consume_stage = (consume_stage + 1) % 3"))
    L(I("add.s32       \t%r107, %r685, 1;"))
    L(I("setp.gt.s32   \t%p32, %r107, 2;"))
    L(I("selp.b32      \t%r685, 0, %r107, %p32;"))
    L(I("selp.b32      \t%r108, 1, 0, %p32;"))
    L(I("xor.b32       \t%r684, %r684, %r108;       // flip parity on ring wrap"))
    L()
    L(I("// Wait until TMA load for consume_stage is complete"))
    L(I("shl.b32       \t%r109, %r685, 3;"))
    L(I("add.s32       \t%r110, %r49,  %r109;"))
    L(I("add.s32       \t%r98,  %r110, 212992;       // &mbarrier[consume_stage]"))
    L(I("bar.sync \t0;"))
    L(I("// begin inline asm"))
    L(I("{"))
    L(I("\t.reg .pred complete;"))
    L(I("waitLoop:"))
    L(I("\tmbarrier.try_wait.parity.shared.b64 complete, [%r98], %r684;"))
    L(I("\t@!complete bra.uni waitLoop;"))
    L(I("}"))
    L(I("// end inline asm"))
    L()
    L(I("// Compute A and B smem addresses for this consume stage"))
    L(I("shl.b32       \t%r111, %r685, 15;"))
    L(I("add.s32       \t%r29, %r49, %r111;          // B_smem = base + consume_stage*32768"))
    L(I("shl.b32       \t%r112, %r685, 14;"))
    L(I("add.s32       \t%r113, %r49, %r112;"))
    L(I("add.s32       \t%r30, %r113, 163840;         // A_smem = base + 163840 + stage*16384"))
    L()
    L(I("// Build wgmma matrix descriptors from smem pointers"))
    L(I("shr.u32       \t%r114, %r30, 4;"))
    L(I("shfl.sync.idx.b32 \t%r31, %r11, 0, 31, -1; // broadcast warp_id to all lanes"))
    L(I("shl.b32       \t%r115, %r31, 7;"))
    L(I("and.b32       \t%r116, %r115, 512;"))
    L(I("add.s32       \t%r117, %r116, %r114;"))
    L(I("and.b32       \t%r118, %r117, 16383;"))
    L(I("cvt.u64.u32   \t%rd19, %r118;               // A descriptor address bits"))
    L(I("bfe.u32       \t%r119, %r29, 4, 14;"))
    L(I("cvt.u64.u32   \t%rd20, %r119;               // B descriptor address bits"))
    L(I("wgmma.fence.sync.aligned;"))
    L(I("or.b64        \t%rd9,  %rd19, 4611686293305294848;  // A desc with TMA metadata"))
    L(I("or.b64        \t%rd10, %rd20, 4611686293305294848;  // B desc with TMA metadata"))
    L(I("mov.pred      \t%p26, -1;                   // always-true predicate"))
    L()

    # 4 × wgmma.m64n256k16 = k64 per pipeline stage
    L(I("// 4 × wgmma.mma_async.m64n256k16 covers the full 64-element K block"))
    for wgmma_idx in range(4):
        if wgmma_idx == 0:
            desc_a_arg, desc_b_arg = "%rd9", "%rd10"
        else:
            offset = wgmma_idx * 2
            a_rd = 9  + wgmma_idx * 2 - 1
            b_rd = 10 + wgmma_idx * 2 - 1
            L(I(f"add.s64 \t%rd{a_rd}, %rd19, {4611686293305294848 + offset};  // A k-slice {wgmma_idx}"))
            L(I(f"add.s64 \t%rd{b_rd}, %rd20, {4611686293305294848 + offset};  // B k-slice {wgmma_idx}"))
            desc_a_arg, desc_b_arg = f"%rd{a_rd}", f"%rd{b_rd}"
        L(I("// begin inline asm"))
        L(I(f"wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
            f"{{{all_acc}}}, {desc_a_arg}, {desc_b_arg}, %p26, 1, 1, 0, 0;"))
        L(I("// end inline asm"))
    L(I("wgmma.commit_group.sync.aligned;"))
    L()
    L(I("// begin inline asm"))
    L(I(f"// wait for regs: {all_acc}"))
    L(I("wgmma.wait_group.sync.aligned 1;  // allow 1 in-flight (overlap with prefetch)"))
    L(I("// end inline asm"))
    L()

    # Advance produce ring and issue next prefetch
    L(I("// Advance produce ring: produce_stage = (produce_stage + 1) % 3"))
    L(I("add.s32       \t%r120, %r683, 1;"))
    L(I("add.s32       \t%r121, %r686, 1;"))
    L(I("setp.gt.s32   \t%p33, %r121, 2;"))
    L(I("selp.b32      \t%r686, 0, %r121, %p33;"))
    L(I("selp.b32      \t%r683, 0, %r120, %p30;     // reset k_block_in_tile at tile boundary"))
    L(I("shl.b32       \t%r105, %r683, 6;           // prefetch k_offset = k_block * 64"))
    L()
    L(I("// Issue prefetch for produce stage (if more iters remain)"))
    L(I("shl.b32       \t%r122, %r686, 3;"))
    L(I("add.s32       \t%r123, %r49,  %r122;"))
    L(I("add.s32       \t%r103, %r123, 212992;       // &mbarrier[produce_stage]"))
    L(I("bar.sync \t0;"))
    L(I("and.pred      \t%p27, %p45, %p31;"))
    L(I("// begin inline asm"))
    L(I("@%p27 mbarrier.arrive.expect_tx.shared.b64 _, [%r103], 49152;"))
    L(I("// end inline asm"))
    L(I("// TMA load A into produce stage slot"))
    L(I("shl.b32       \t%r124, %r686, 14;"))
    L(I("add.s32       \t%r125, %r49,  %r124;"))
    L(I("add.s32       \t%r104, %r125, 163840;"))
    L(I("bar.sync \t0;"))
    L(I("elect.sync    \t%r126|%p34, -1;"))
    L(I("and.pred      \t%p35, %p31, %p34;"))
    L(I("and.pred      \t%p28, %p11, %p35;"))
    L(I("// begin inline asm"))
    L(I("@%p28 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r104], [%rd4, {%r105, %r819}], [%r103];"))
    L(I("// end inline asm"))
    L(I("// TMA load B into produce stage slot"))
    L(I("shl.b32       \t%r127, %r686, 15;"))
    L(I("add.s32       \t%r106, %r49,  %r127;"))
    L(I("bar.sync \t0;"))
    L(I("elect.sync    \t%r128|%p36, -1;"))
    L(I("and.pred      \t%p37, %p31, %p36;"))
    L(I("and.pred      \t%p29, %p11, %p37;"))
    L(I("// begin inline asm"))
    L(I("@%p29 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%r106], [%rd3, {%r105, %r820}], [%r103];"))
    L(I("// end inline asm"))
    L()
    L(I("// If not the last k-block of the tile, loop back without epilogue"))
    L(I("setp.ne.b32   \t%p38, %r682, %r7;          // p38 = not last k-block"))
    L(I("@%p38 bra     \t$L__BB0_13;"))

    # ===================================================================
    # Epilogue: last k-block → write C and aux to global
    # ===================================================================
    L()
    L("// %bb.12: epilogue — last k-block of tile, drain wgmma, write outputs")
    L(I("setp.lt.u32   \t%p41, %r9, 128;            // p41 = (tid < 128), i.e. wg0"))
    L()
    L(I("// Drain all pending wgmma before reading accumulator registers"))
    L(I("mov.b32 \t%r130, %r129;"))
    L(I("mov.b32 \t%r131, %r129;"))
    L(I("mov.b32 \t%r132, %r129;"))
    L(I("// begin inline asm"))
    L(I(f"// wait for regs: {all_acc}"))
    L(I("wgmma.wait_group.sync.aligned 0;"))
    L(I("// end inline asm"))
    L()
    L(I("// Wait for any prior TMA store to finish before reusing staging buffer"))
    L(I("cp.async.bulk.wait_group.read \t0;"))
    L(I("bar.sync \t0;"))
    L()
    L(I("// Compute staging buffer address and C global col coordinate for this tile"))
    L(I("and.b32       \t%r135, %r31, 3;"))
    L(I("shl.b32       \t%r136, %r135, 14;"))
    L(I("add.s32       \t%r137, %r49,  %r136;"))
    L(I("add.s32       \t%r134, %r137, 98304;        // smem staging addr for TMA source"))
    L(I("shl.b32       \t%r138, %r135, 6;"))
    L(I("add.s32       \t%r133, %r138, %r680;        // C output col coordinate"))
    L()

    # ---- Convert acc pairs to bf16 C and relu²→bf16 aux -----------------------
    L(I("// --- Convert 128 f32 accumulator registers to bf16 ---"))
    L(I("// For each of 64 pairs i: process acc[2i] (lo) and acc[2i+1] (hi):"))
    L(I("//   1. Convert to bf16, pack into c_reg[i]"))
    L(I("//   2. Round-trip back to f32, apply relu, square, pack into aux_reg[i]"))
    L()
    for i in range(64):
        acc_lo_idx = 2 * i
        acc_hi_idx = 2 * i + 1
        L(I(f"// pair {i}: acc[{acc_lo_idx}]={acc(acc_lo_idx)}, acc[{acc_hi_idx}]={acc(acc_hi_idx)}"))
        L(I(f"cvt.rn.bf16.f32 \t{rs_hi(i)}, {acc(acc_hi_idx)};"))
        L(I(f"cvt.rn.bf16.f32 \t{rs_lo(i)}, {acc(acc_lo_idx)};"))
        L(I(f"mov.b32         \t{c_reg(i)}, {{{rs_lo(i)}, {rs_hi(i)}}};"))
        L(I(f"cvt.f32.bf16    \t{f_lo(i)},  {rs_lo(i)};"))
        L(I(f"cvt.f32.bf16    \t{f_hi(i)},  {rs_hi(i)};"))
        L(I(f"max.f32         \t{relu_hi(i)}, {f_hi(i)}, 0f00000000;"))
        L(I(f"max.f32         \t{relu_lo(i)}, {f_lo(i)}, 0f00000000;"))
        L(I(f"mul.f32         \t{sq_lo(i)},   {relu_lo(i)}, {relu_lo(i)};"))
        L(I(f"mul.f32         \t{sq_hi(i)},   {relu_hi(i)}, {relu_hi(i)};"))
        L(I(f"cvt.rn.bf16x2.f32 \t{aux_reg(i)}, {sq_hi(i)}, {sq_lo(i)};"))
        L()

    # ---- stmatrix: write C to staging buffer ----------------------------------
    # 16 stmatrix.x4 calls; addr k → c_regs at stmatrix_indices(k)
    L(I("// stmatrix: scatter 64 c_reg bf16x2 values to shared memory"))
    L(I("// 16 calls × 4 registers = 64 total, one per acc pair"))
    L(I("// Address interleaving follows wgmma m8n8 tile layout:"))
    L(I("//   addr_k: col_group = k//4, row_group = k%4"))
    L(I("//   c_reg indices: [col_group*4 + j + row_group*16 for j in 0..3]"))
    for k in range(16):
        idxs  = stmatrix_indices(k)
        cregs = ", ".join(c_reg(i) for i in idxs)
        L(I(f"stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r{12+k}], {{{cregs}}};  // C group {k} → indices {idxs}"))
    L()

    # TMA store C → global
    L(I("// TMA store: flush C staging buffer → global memory"))
    L(I("// begin inline asm"))
    L(I("fence.proxy.async.shared::cta;"))
    L(I("// end inline asm"))
    L(I("bar.sync \t0;"))
    L(I("elect.sync    \t%r650|%p42, -1;"))
    L(I("and.pred      \t%p39, %p41, %p42;           // wg0 elected thread issues store"))
    L(I("// begin inline asm"))
    L(I("@%p39 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd2, {%r133, %r681}], [%r134];"))
    L(I("// end inline asm"))
    L(I("cp.async.bulk.commit_group;"))
    L()

    # ---- stmatrix: write aux to staging buffer --------------------------------
    L(I("// Wait for C store then reuse staging buffer for aux"))
    L(I("cp.async.bulk.wait_group.read \t0;"))
    L(I("bar.sync \t0;"))
    L(I("// stmatrix: scatter 64 aux_reg bf16x2 values — same layout as C"))
    for k in range(16):
        idxs  = stmatrix_indices(k)
        aregs = ", ".join(aux_reg(i) for i in idxs)
        L(I(f"stmatrix.sync.aligned.m8n8.x4.shared.b16 [%r{12+k}], {{{aregs}}};  // aux group {k} → indices {idxs}"))
    L()

    # TMA store aux → global
    L(I("// TMA store: flush aux staging buffer → global memory"))
    L(I("// begin inline asm"))
    L(I("fence.proxy.async.shared::cta;"))
    L(I("// end inline asm"))
    L(I("bar.sync \t0;"))
    L(I("elect.sync    \t%r652|%p43, -1;"))
    L(I("and.pred      \t%p40, %p41, %p43;"))
    L(I("// begin inline asm"))
    L(I("@%p40 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%rd1, {%r133, %r681}], [%r134];"))
    L(I("// end inline asm"))
    L(I("cp.async.bulk.commit_group;"))
    L()

    # Reset accumulator for next tile and jump back to loop top
    L(I("// Reset accumulator to zero for the next output tile"))
    L(I(f"mov.b32 \t{acc(0)}, 0f00000000;"))
    for i in range(1, 128):
        L(I(f"mov.b32 \t{acc(i)}, {acc(0)};"))
    L(I("bra.uni \t$L__BB0_13;"))

    # ===================================================================
    # Loop exit / cleanup
    # ===================================================================
    L()
    L("$L__BB0_14:  // loop exit — final drain and mbarrier invalidation")
    L(I("cp.async.bulk.wait_group.read \t0;"))
    L(I("bar.sync \t0;"))
    L(I("// begin inline asm"))
    L(I(f"// wait for regs: {all_acc}"))
    L(I("wgmma.wait_group.sync.aligned 0;"))
    L(I("// end inline asm"))
    L()
    L(I("// Invalidate all 3 mbarriers"))
    for reg, addr in [(45, 212992), (72, 213000), (46, 213008)]:
        L(I("// begin inline asm"))
        L(I(f"@%p45 mbarrier.inval.shared::cta.b64 [%r{reg}];"))
        L(I("// end inline asm"))
        L(I("bar.sync \t0;"))

    L()
    L("$L__BB0_15:")
    L(I("ret;"))
    L("$L__func_end0:")
    L("}")
    L()
    L('\t.file\t1 "relu_fusion_2.py"')

    return "\n".join(lines)


# ============================================================
# TMA descriptor construction
# ============================================================

def make_tma_descriptor(tensor, box_shape):
    """Build a 128-byte CUtensorMap for a 2-D bf16 tensor with 128B swizzle."""
    import ctypes
    from cuda.bindings import driver as cuda

    rows, cols = tensor.shape
    rank = 2
    global_dim     = (ctypes.c_uint64 * rank)(cols, rows)
    global_strides = (ctypes.c_uint64 * (rank - 1))(cols * 2)
    box_dim        = (ctypes.c_uint32 * rank)(box_shape[1], box_shape[0])
    elem_strides   = (ctypes.c_uint32 * rank)(1, 1)

    print("box_shape:", box_shape)

    status, desc = cuda.cuTensorMapEncodeTiled(
        cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        rank,
        tensor.data_ptr(),
        (cuda.cuuint64_t(cols), cuda.cuuint64_t(rows)),
        (cuda.cuuint64_t(cols*2),),
        (cuda.cuuint32_t(box_shape[1]), cuda.cuuint32_t(box_shape[0])),
        (cuda.cuuint32_t(1), cuda.cuuint32_t(1)),
        cuda.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
        cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B,
        cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE,
        cuda.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )
    return struct.pack('16Q', *[int(v) for v in desc.opaque])


# ============================================================
# Kernel loading and launch
# ============================================================

SMEM_BYTES      = 213_016
M_TILE, N_TILE, K_BLOCK = 128, 256, 64


def load_kernel(ptx_source):
    import ctypes
    from cuda.bindings import driver as cuda
    cuda.cuInit(0)
    _, dev = cuda.cuDeviceGet(0)
    _, ctx = cuda.cuCtxCreate(0, dev)
    _, mod = cuda.cuModuleLoadData(ptx_source.encode())
    _, fn = cuda.cuModuleGetFunction(mod, b"linear_relu_square_kernel")
    cuda.cuFuncSetAttribute(
        fn,
        cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        SMEM_BYTES,
    )
    return ctx, mod, fn


def run_kernel(fn, A, B, C, aux, M, N, K, stream):
    import ctypes
    from cuda.bindings import driver as cuda

    print("A desc:")
    desc_A   = make_tma_descriptor(A,   (M_TILE,  K_BLOCK))
    print("B desc:")
    desc_B   = make_tma_descriptor(B,   (N_TILE,  K_BLOCK))
    print("C desc:")
    desc_C   = make_tma_descriptor(C,   (M_TILE // 2,  N_TILE // 4))
    print("Aux desc:")
    desc_aux = make_tma_descriptor(aux, (M_TILE // 2,  N_TILE // 4))

    bufs = []  # keep byte-buffer ctypes objects alive
    def desc_buf(b):
        buf = (ctypes.c_uint8 * 128)(*b)
        bufs.append(buf)
        return buf

    dummy = A.data_ptr()
    raw_params = [
        desc_buf(desc_A),               # 0  param_desc_A
        ctypes.c_uint32(0),             # 1  param_A_row_step   (unused)
        ctypes.c_uint32(0),             # 2  param_A_col_step   (unused)
        ctypes.c_uint64(K * 2),         # 3  param_A_stride_bytes
        ctypes.c_uint64(A.data_ptr()),  # 4  param_A_base_ptr
        desc_buf(desc_B),               # 5  param_desc_B
        ctypes.c_uint32(0),             # 6
        ctypes.c_uint32(0),             # 7
        ctypes.c_uint64(K * 2),         # 8  param_B_stride_bytes
        ctypes.c_uint64(B.data_ptr()),  # 9
        desc_buf(desc_C),               # 10 param_desc_C
        ctypes.c_uint32(0),             # 11
        ctypes.c_uint32(0),             # 12
        ctypes.c_uint64(N * 2),         # 13 param_C_stride_bytes
        ctypes.c_uint64(C.data_ptr()),  # 14
        desc_buf(desc_aux),             # 15 param_desc_aux
        ctypes.c_uint32(0),             # 16
        ctypes.c_uint32(0),             # 17
        ctypes.c_uint64(N * 2),         # 18 param_aux_stride_bytes
        ctypes.c_uint64(aux.data_ptr()),# 19
        ctypes.c_uint32(M),             # 20 param_M
        ctypes.c_uint32(N),             # 21 param_N
        ctypes.c_uint32(K),             # 22 param_K
        ctypes.c_uint64(dummy),         # 23 param_unused0
        ctypes.c_uint64(dummy),         # 24 param_unused1
    ]

    param_ptrs = (ctypes.c_void_p * len(raw_params))()
    for i, p in enumerate(raw_params):
        param_ptrs[i] = ctypes.cast(ctypes.pointer(p), ctypes.c_void_p)

    grid_x = ((M + M_TILE - 1) // M_TILE) * ((N + N_TILE - 1) // N_TILE)
    status = cuda.cuLaunchKernel(fn, grid_x, 1, 1, 256, 1, 1, SMEM_BYTES, 0, param_ptrs, 0)
    print("STATUS:", status)
    cuda.cuStreamSynchronize(stream)


# ============================================================
# Reference implementation and correctness check
# ============================================================

def reference(A, B):
    import torch
    C    = A @ B.T
    aux = (C.clamp(min=0) ** 2).to(torch.bfloat16)
    return C, aux


def check(C_kernel, C_ref, aux_kernel, aux_ref):
    import torch
    ok = True
    for name, got, want in [("C", C_kernel, C_ref), ("aux", aux_kernel, aux_ref)]:
        err    = (got.float() - want.float()).abs()
        passed = torch.allclose(got.float(), want.float(), atol=1e-2, rtol=1e-2)
        ok     = ok and passed
        print(f"  {name}: {'PASS ✓' if passed else 'FAIL ✗'}"
              f"  max_err={err.max():.5f}  mean_err={err.mean():.5f}")
        if not passed:
            flat = err.flatten()
            for idx in flat.topk(5).indices:
                r, c = divmod(idx.item(), want.shape[1])
                print(f"    [{r},{c}]  got={got[r,c].item():.4f}  want={want[r,c].item():.4f}")
    return ok


# ============================================================
# Main
# ============================================================

def main():
    import torch
    from cuda.bindings import driver as cuda

    M, N, K = 8 * 2048, 3072, 768

    print("Generating PTX...")
    ptx = generate_ptx()
    print(f"  {ptx.count(chr(10))} lines")

    print("Loading kernel...")
    ctx, mod, fn = load_kernel(ptx)

    _, stream = cuda.cuStreamCreate(0)

    torch.manual_seed(42)
    device = torch.device("cuda")
    A   = (torch.randn(M, K) * 0.1).to(torch.bfloat16).to(device).contiguous()
    B   = (torch.randn(N, K) * 0.1).to(torch.bfloat16).to(device).contiguous()
    C   = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
    aux = torch.zeros(M, N, dtype=torch.bfloat16, device=device)

    print("Computing reference (torch.matmul fp32)...")
    C_ref, aux_ref = reference(A, B)

    print("Warmup (2 runs)...")
    for _ in range(2):
        C.zero_(); aux.zero_()
        run_kernel(fn, A, B, C, aux, M, N, K, stream)

    print("Correctness check...")
    C.zero_(); aux.zero_()
    run_kernel(fn, A, B, C, aux, M, N, K, stream)
    all_ok = check(C, C_ref, aux, aux_ref)

    exit()

    print("Timing (10 iters)...")
    start = cuda.CUevent(); end = cuda.CUevent()
    cuda.cuEventCreate(start, 0); cuda.cuEventCreate(end, 0)
    cuda.cuEventRecord(start, stream)
    for _ in range(10):
        C.zero_(); aux.zero_()
        run_kernel(fn, A, B, C, aux, M, N, K, stream)
    cuda.cuEventRecord(end, stream)
    cuda.cuEventSynchronize(end)
    ms     = cuda.cuEventElapsedTime(start, end) / 10
    tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
    print(f"  {ms:.3f} ms  |  {tflops:.2f} TFLOP/s (GEMM FLOPs only)")
    print(f"\nOverall: {'PASS ✓' if all_ok else 'FAIL ✗'}")

    cuda.cuEventDestroy(start); cuda.cuEventDestroy(end)
    cuda.cuStreamDestroy(stream)
    cuda.cuModuleUnload(mod)
    cuda.cuCtxDestroy(ctx)


if __name__ == "__main__":
    main()
