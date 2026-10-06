#ifndef PS2_VU1_NATIVE_H
#define PS2_VU1_NATIVE_H

// VU1 native programs.
//
// A native program is the interpreter's run() loop body generated once per
// VU1 code image, one specialisation of Vu1NativeAccess::stepPair<U, L> per
// instruction pair, so that everything the interpreter derives from the
// instruction words at run time (usage, latencies, hazard register set, write
// lanes, branch/halt kind, FMAC form) is a compile-time constant. Semantics
// come from the interpreter's own code: decodePairWords (constexpr),
// execLower, the FMAC fast path (fmacFast, same code as execUpper's) with
// execUpper as the fallback for every other upper op, and the same timing
// calls (commitReadyPipelines / advanceTo / advanceOneCycle) in the same
// order. What differs is only the absence of decoding, table lookups and the
// data-dependent loops over usage lists.

#include "runtime/ps2_vu1.h"
#include "ps2_vu1_detail.h"
#include "ps2_vu1_decode_inl.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <emmintrin.h>

#if defined(_MSC_VER)
#define PS2X_VU1N_INLINE __forceinline
#define PS2X_VU1N_NOINLINE __declspec(noinline)
#else
#define PS2X_VU1N_INLINE inline __attribute__((always_inline))
#define PS2X_VU1N_NOINLINE __attribute__((noinline))
#endif

class GS;
class PS2Memory;

namespace vu1n
{
enum class FmacKind : uint8_t
{
    None,
    Add,
    Sub,
    Madd,
    Msub,
    Mul
};
enum class FmacRight : uint8_t
{
    Bc,
    Q,
    I,
    Vt
};
struct FmacForm
{
    FmacKind kind;
    FmacRight right;
};

// `sel` is the op (< 0x3C) or the upper special selector; both groups use
// the same numbering for these forms.
constexpr FmacForm fmacForm(uint8_t sel)
{
    if (sel <= 0x03u)
        return {FmacKind::Add, FmacRight::Bc};
    if (sel <= 0x07u)
        return {FmacKind::Sub, FmacRight::Bc};
    if (sel <= 0x0Bu)
        return {FmacKind::Madd, FmacRight::Bc};
    if (sel <= 0x0Fu)
        return {FmacKind::Msub, FmacRight::Bc};
    if (sel >= 0x18u && sel <= 0x1Bu)
        return {FmacKind::Mul, FmacRight::Bc};
    switch (sel)
    {
    case 0x1Cu:
        return {FmacKind::Mul, FmacRight::Q};
    case 0x1Eu:
        return {FmacKind::Mul, FmacRight::I};
    case 0x20u:
        return {FmacKind::Add, FmacRight::Q};
    case 0x21u:
        return {FmacKind::Madd, FmacRight::Q};
    case 0x22u:
        return {FmacKind::Add, FmacRight::I};
    case 0x23u:
        return {FmacKind::Madd, FmacRight::I};
    case 0x24u:
        return {FmacKind::Sub, FmacRight::Q};
    case 0x25u:
        return {FmacKind::Msub, FmacRight::Q};
    case 0x26u:
        return {FmacKind::Sub, FmacRight::I};
    case 0x27u:
        return {FmacKind::Msub, FmacRight::I};
    case 0x28u:
        return {FmacKind::Add, FmacRight::Vt};
    case 0x29u:
        return {FmacKind::Madd, FmacRight::Vt};
    case 0x2Au:
        return {FmacKind::Mul, FmacRight::Vt};
    case 0x2Cu:
        return {FmacKind::Sub, FmacRight::Vt};
    case 0x2Du:
        return {FmacKind::Msub, FmacRight::Vt};
    default:
        return {FmacKind::None, FmacRight::Vt};
    }
}

// normalizeOperand() on four lanes: denormal -> signed zero, Inf/NaN ->
// signed FLT_MAX, everything else unchanged.
inline __m128i vuNormalizeBits(__m128i b)
{
    const __m128i expMask = _mm_set1_epi32(0x7F800000);
    const __m128i signMask = _mm_set1_epi32(static_cast<int>(0x80000000u));
    const __m128i e = _mm_and_si128(b, expMask);
    const __m128i zeroExp = _mm_cmpeq_epi32(e, _mm_setzero_si128());
    const __m128i maxExp = _mm_cmpeq_epi32(e, expMask);
    const __m128i sign = _mm_and_si128(b, signMask);
    const __m128i fmax = _mm_or_si128(sign, _mm_set1_epi32(0x7F7FFFFF));
    b = _mm_or_si128(_mm_andnot_si128(zeroExp, b), _mm_and_si128(zeroExp, sign));
    return _mm_or_si128(_mm_andnot_si128(maxExp, b), _mm_and_si128(maxExp, fmax));
}

inline __m128i vuLoadNormalized(const float *v)
{
    return vuNormalizeBits(_mm_loadu_si128(reinterpret_cast<const __m128i *>(v)));
}

inline __m128i vuBroadcastNormalized(float v)
{
    uint32_t bits = 0u;
    std::memcpy(&bits, &v, sizeof(bits));
    return vuNormalizeBits(_mm_set1_epi32(static_cast<int>(bits)));
}

// normalizeFmacExactResult()'s classification for four exact (double)
// results held as two pairs; bit c of each mask = component c.
struct VuLaneClass
{
    uint32_t zero, neg, over, under;
};
inline VuLaneClass vuClassify(__m128d lo, __m128d hi)
{
    const __m128d signBit = _mm_set1_pd(-0.0);
    const __m128d zero = _mm_setzero_pd();
    const __m128d fmax = _mm_set1_pd(static_cast<double>(std::numeric_limits<float>::max()));
    const __m128d fmin = _mm_set1_pd(static_cast<double>(std::numeric_limits<float>::min()));
    const __m128d aLo = _mm_andnot_pd(signBit, lo);
    const __m128d aHi = _mm_andnot_pd(signBit, hi);
    VuLaneClass c;
    c.zero = static_cast<uint32_t>(_mm_movemask_pd(_mm_cmpeq_pd(aLo, zero)) |
                                   (_mm_movemask_pd(_mm_cmpeq_pd(aHi, zero)) << 2));
    c.neg = static_cast<uint32_t>(_mm_movemask_pd(lo) | (_mm_movemask_pd(hi) << 2));
    c.over = static_cast<uint32_t>(_mm_movemask_pd(_mm_cmpgt_pd(aLo, fmax)) |
                                   (_mm_movemask_pd(_mm_cmpgt_pd(aHi, fmax)) << 2));
    c.under = static_cast<uint32_t>(_mm_movemask_pd(_mm_cmplt_pd(aLo, fmin)) |
                                    (_mm_movemask_pd(_mm_cmplt_pd(aHi, fmin)) << 2)) &
              ~c.zero;
    return c;
}

// DEST field (x=8 y=4 z=2 w=1) -> component bit mask (x=1 y=2 z=4 w=8).
constexpr uint8_t kDestToLanes[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};

// Lanes whose single-precision magnitude sits strictly inside the normal
// float range, i.e. FLT_MIN <= |x| < FLT_MAX.
//
// Why that settles the FMAC flags without the double-precision pass: the VU
// rounds toward zero (run() sets MXCSR RTZ), so the float result is the
// largest representable value not exceeding the exact result in magnitude.
//   * |float| >= FLT_MIN  =>  |exact| >= |float| >= FLT_MIN, so the lane is
//     neither zero nor underflow.
//   * |float| <  FLT_MAX  =>  |exact| < the next float above |float|, which
//     is <= FLT_MAX, so the lane cannot be overflow (an exact value above
//     FLT_MAX rounds toward zero to exactly FLT_MAX, or to inf).
// So for such a lane the only flag is the sign, and the result needs no
// clamping. Every other case (zero, denormal, FLT_MAX, inf) still takes the
// exact double path below, unchanged.
inline uint32_t vuOrdinaryLanes(__m128 v)
{
    const __m128i mag = _mm_and_si128(_mm_castps_si128(v), _mm_set1_epi32(0x7FFFFFFF));
    const __m128i aboveDenormal = _mm_cmpgt_epi32(mag, _mm_set1_epi32(0x007FFFFF));
    const __m128i belowMax = _mm_cmpgt_epi32(_mm_set1_epi32(0x7F7FFFFF), mag);
    return static_cast<uint32_t>(_mm_movemask_ps(_mm_castsi128_ps(_mm_and_si128(aboveDenormal, belowMax))));
}

// Lanes whose value is exactly +0 or -0.
inline uint32_t vuZeroLanes(__m128 v)
{
    const __m128i mag = _mm_and_si128(_mm_castps_si128(v), _mm_set1_epi32(0x7FFFFFFF));
    return static_cast<uint32_t>(_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(mag, _mm_setzero_si128()))));
}

} // namespace vu1n

enum class Vu1NativeStep : uint8_t
{
    Continue, // pair executed, keep going
    Ended,    // program ended (E-bit / halt): run() does its epilogue
    Stop,     // budget exhausted or reserved instruction: run() stops as the loop would
    Fallback  // pc not covered: run() continues in the interpreter loop
};

struct Vu1NativeCtx
{
    uint8_t *vuData;
    uint32_t dataSize;
    uint32_t codeSize;
    GS *gs;
    PS2Memory *memory;
    uint64_t budgetEnd;
    uint64_t instrCount;
};

using Vu1NativeProgram = Vu1NativeStep (*)(VU1Interpreter &, Vu1NativeCtx &);
// Pair trace hook for the PS2X_VU1_DIFF harness. Diagnostic builds only:
// it is an out-of-line call plus an EFU scan on EVERY pair, which cost
// ~19 ns per instruction when it was compiled into the play build.
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
void ps2xVu1TracePair(uint32_t pc, uint64_t cycle, uint64_t maxReady, uint64_t fdiv, uint64_t efu,
                      float q, uint32_t mac, uint32_t clip, uint32_t status);
#define PS2X_VU1_TRACE_PAIR(v)                                                          \
    do {                                                                               \
        uint64_t efuReady_ = 0u;                                                        \
        for (const I::ScalarPipelineEntry &e_ : (v).m_efu)                              \
            if (e_.valid)                                                              \
                efuReady_ = std::max(efuReady_, e_.readyCycle);                         \
        ps2xVu1TracePair((v).m_state.pc, (v).m_cycle, (v).m_maxReadyCycle,              \
                         (v).m_fdiv.valid ? (v).m_fdiv.readyCycle : 0u, efuReady_,      \
                         (v).m_state.q, (v).m_state.mac, (v).m_state.clip, (v).m_state.status); \
    } while (0)
#else
#define PS2X_VU1_TRACE_PAIR(v) ((void)0)
#endif
void ps2xVu1NativeRegister(uint64_t imageHash, Vu1NativeProgram program);
Vu1NativeProgram ps2xVu1NativeLookup(uint64_t imageHash);

struct Vu1NativeAccess
{
    using I = VU1Interpreter;
    using D = VU1Interpreter::DecodedInstructionPair;
    // Vu0: native VU0 micro-programs (same instruction set; 12-bit micro
    // address mask, XGKICK/EFU reserved).
    template <uint32_t U, uint32_t L, bool Vu0 = false>
    struct PairDecode
    {
        static constexpr D d = I::decodePairWords(U, L, Vu0 ? I::Unit::VU0 : I::Unit::VU1);
    };

    static bool stopRequested(I &v) { return v.m_stopRequested; }
    // High-level kernels (ps2_vu1_heavy_block.cpp): one whole loop iteration
    // with registers held in locals; exact results, false = not applicable
    // (nothing changed, run the native pairs).
    static bool hleFight1028(I &v, Vu1NativeCtx &c);
    static bool hleSameState(const I &a, const I &b, const char **what);
    static bool hleShader40(I &v, Vu1NativeCtx &c);
    static uint64_t cycle(I &v) { return v.m_cycle; }
    static uint32_t pc(I &v) { return v.m_state.pc; }
    static PS2X_VU1N_INLINE void setPc(I &v, uint32_t pc) { v.m_state.pc = pc; }

    // applyDest with the destination mask a constant.
    template <uint8_t dest>
    static PS2X_VU1N_INLINE void applyDestC(float *dst, const float *result)
    {
        if constexpr ((dest & 0x8u) != 0u)
            dst[0] = result[0];
        if constexpr ((dest & 0x4u) != 0u)
            dst[1] = result[1];
        if constexpr ((dest & 0x2u) != 0u)
            dst[2] = result[2];
        if constexpr ((dest & 0x1u) != 0u)
            dst[3] = result[3];
    }

    // Relaxed VU1: tick the clock until Q/P/store/XGKICK pipelines are empty.
    static PS2X_VU1N_NOINLINE void relaxSettle(I &v)
    {
        while (v.m_pipelineNextReady != ~0ull || v.m_xgkick.active)
            v.advanceOneCycle();
    }

    // Apply every queued flag entry in order (immediate-flag path). Cold:
    // out of line so the hot pair body stays small.
    static PS2X_VU1N_NOINLINE void drainFlagFifo(I &v)
    {
    while (v.m_flagOrderCount != 0u)
    {
        const uint32_t best = v.m_flagOrder[v.m_flagOrderHead];
        I::FlagPipelineEntry &e = v.m_flagPipeline[static_cast<size_t>(best)];
        v.m_flagOrderHead = (v.m_flagOrderHead + 1u) % I::kMaxFlagEntries;
        --v.m_flagOrderCount;
        if (e.writesMac)
            v.m_state.mac = e.mac;
        if (e.writesStatus)
        {
            const uint32_t cur = e.status & 0xFu;
            v.m_state.status = (v.m_state.status & 0xFF0u) | cur | ((cur | e.extraSticky) << 6);
        }
        if (e.writesSticky)
            v.m_state.status = (v.m_state.status & 0x03Fu) | (e.status & 0xFC0u);
        if (e.writesClip)
            v.m_state.clip = e.clip;
        e.valid = false;
        v.m_flagMask &= ~(1u << best);
    }
    }

    // updateFmacFlags with the destination mask a constant: same mac/status
    // packing, same entry allocation (retire-on-full, FIFO order), no loops.
    // NoFlags: gen_vu1_native.py proved no MAC/STATUS reader can see this
    // update before MAC is overwritten (relaxed block compiler, dead flags).
    template <uint8_t dest, bool Imm = false, bool NoFlags = false>
    static PS2X_VU1N_INLINE void updateFmacFlagsC(I &v, const uint8_t laneFlags[4], uint32_t extraSticky)
    {
        if constexpr (dest == 0u || NoFlags)
        {
            (void)v;
            (void)laneFlags;
            (void)extraSticky;
            return;
        }
        else
        {
            uint32_t mac = 0u;
            uint32_t status = 0u;
            auto lane = [&](uint32_t component, uint8_t laneBit)
            {
                const uint32_t flags = laneFlags[component];
                if ((flags & 0x1u) != 0u)
                    mac |= laneBit;
                if ((flags & 0x2u) != 0u)
                    mac |= static_cast<uint32_t>(laneBit) << 4;
                if ((flags & 0x4u) != 0u)
                    mac |= static_cast<uint32_t>(laneBit) << 8;
                if ((flags & 0x8u) != 0u)
                    mac |= static_cast<uint32_t>(laneBit) << 12;
                status |= flags;
            };
            if constexpr ((dest & 0x8u) != 0u)
                lane(0u, 0x8u);
            if constexpr ((dest & 0x4u) != 0u)
                lane(1u, 0x4u);
            if constexpr ((dest & 0x2u) != 0u)
                lane(2u, 0x2u);
            if constexpr ((dest & 0x1u) != 0u)
                lane(3u, 0x1u);

            if constexpr (Imm)
            {
                // The generator proved no flag read (or FSSET/FCSET, JR/JALR,
                // E-bit) can run within kFmacLatency pairs of here, so nothing
                // can observe the flags before this update would retire.
                // Apply everything still queued (in order), then this update.
                if (v.m_flagOrderCount != 0u)
                    drainFlagFifo(v);
                v.m_state.mac = mac;
                const uint32_t cur = status & 0xFu;
                v.m_state.status = (v.m_state.status & 0xFF0u) | cur | ((cur | extraSticky) << 6);
                v.m_flagTailReady = v.m_cycle + I::kFmacLatency;
                return;
            }
            uint32_t freeBits = ~v.m_flagMask;
            if (freeBits == 0u)
            {
                v.retireFlags();
                freeBits = ~v.m_flagMask;
            }
            if (freeBits == 0u)
            {
                v.reportReservedInstruction(true, 0xFFFFFFFFu);
                return;
            }
            const uint32_t freeSlot = static_cast<uint32_t>(std::countr_zero(freeBits));
            I::FlagPipelineEntry *entry = &v.m_flagPipeline[static_cast<size_t>(freeSlot)];
            v.m_flagMask |= 1u << freeSlot;
            // pushFlagOrder(freeSlot), inlined (it is out of line in the .cpp).
            v.m_flagOrder[(v.m_flagOrderHead + v.m_flagOrderCount) % I::kMaxFlagEntries] = static_cast<uint8_t>(freeSlot);
            ++v.m_flagOrderCount;
            *entry = {};
            entry->valid = true;
            entry->issueCycle = v.m_cycle;
            entry->readyCycle = v.m_cycle + I::kFmacLatency;
            entry->mac = mac;
            entry->status = status;
            entry->extraSticky = extraSticky;
            entry->writesMac = true;
            entry->writesStatus = true;
        }
    }

    // execUpper's FMAC fast path with the instruction a constant. Returns
    // false for every non-FMAC form (caller then runs execUpper).
    // fmacFast's exact (double) path, out of line: recomputes the float result
    // from the unchanged state, then classifies the double-precision result.
    template <uint32_t U, bool Imm, bool NoFlags>
    static PS2X_VU1N_NOINLINE void fmacSlow(I &v)
    {
        constexpr uint8_t dest = DEST(U);
        constexpr uint8_t ft = FT(U);
        constexpr uint8_t fs = FS(U);
        constexpr uint8_t fd = FD(U);
        constexpr uint8_t op = static_cast<uint8_t>(U & 0x3Fu);
        constexpr bool upperSpecial = op >= 0x3Cu;
        constexpr uint8_t sel = upperSpecial ? static_cast<uint8_t>((U & 0x3u) | ((U >> 4) & 0x7Cu)) : op;
        constexpr vu1n::FmacForm form = vu1n::fmacForm(sel);
        (void)ft;
        {
                v.m_currentUpperInstruction = U;
                const __m128i vsBits = vu1n::vuLoadNormalized(v.m_state.vf[fs]);
                __m128i rBits;
                if constexpr (form.right == vu1n::FmacRight::Bc)
                    rBits = vu1n::vuBroadcastNormalized(v.m_state.vf[ft][sel & 3u]);
                else if constexpr (form.right == vu1n::FmacRight::Q)
                    rBits = vu1n::vuBroadcastNormalized(v.m_state.q);
                else if constexpr (form.right == vu1n::FmacRight::I)
                    rBits = vu1n::vuBroadcastNormalized(v.m_state.i);
                else
                    rBits = vu1n::vuLoadNormalized(v.m_state.vf[ft]);
                constexpr bool productSum = form.kind == vu1n::FmacKind::Madd || form.kind == vu1n::FmacKind::Msub;
                const __m128 vs = _mm_castsi128_ps(vsBits);
                const __m128 r = _mm_castsi128_ps(rBits);
                const __m128 acc = productSum ? _mm_castsi128_ps(vu1n::vuLoadNormalized(v.m_state.acc)) : _mm_setzero_ps();

                const __m128 prod = _mm_mul_ps(vs, r);
                __m128 res;
                if constexpr (form.kind == vu1n::FmacKind::Add)
                    res = _mm_add_ps(vs, r);
                else if constexpr (form.kind == vu1n::FmacKind::Sub)
                    res = _mm_sub_ps(vs, r);
                else if constexpr (form.kind == vu1n::FmacKind::Madd)
                    res = _mm_add_ps(acc, prod);
                else if constexpr (form.kind == vu1n::FmacKind::Msub)
                    res = _mm_sub_ps(acc, prod);
                else
                    res = prod;
                float result[4];
                _mm_storeu_ps(result, res);

                constexpr uint32_t destLanes = vu1n::kDestToLanes[dest & 0xFu];
                // Ordinary-magnitude lanes need no exact (double) pass: see
                // vuOrdinaryLanes. Only the sign bit can be set, and the float
                // result stands as computed.
                const __m128d sLo = _mm_cvtps_pd(vs);
                const __m128d sHi = _mm_cvtps_pd(_mm_movehl_ps(vs, vs));
                const __m128d rLo = _mm_cvtps_pd(r);
                const __m128d rHi = _mm_cvtps_pd(_mm_movehl_ps(r, r));
                const __m128d pLo = _mm_mul_pd(sLo, rLo);
                const __m128d pHi = _mm_mul_pd(sHi, rHi);
                __m128d eLo, eHi;
                if constexpr (form.kind == vu1n::FmacKind::Add)
                {
                    eLo = _mm_add_pd(sLo, rLo);
                    eHi = _mm_add_pd(sHi, rHi);
                }
                else if constexpr (form.kind == vu1n::FmacKind::Sub)
                {
                    eLo = _mm_sub_pd(sLo, rLo);
                    eHi = _mm_sub_pd(sHi, rHi);
                }
                else if constexpr (form.kind == vu1n::FmacKind::Madd)
                {
                    eLo = _mm_add_pd(_mm_cvtps_pd(acc), pLo);
                    eHi = _mm_add_pd(_mm_cvtps_pd(_mm_movehl_ps(acc, acc)), pHi);
                }
                else if constexpr (form.kind == vu1n::FmacKind::Msub)
                {
                    eLo = _mm_sub_pd(_mm_cvtps_pd(acc), pLo);
                    eHi = _mm_sub_pd(_mm_cvtps_pd(_mm_movehl_ps(acc, acc)), pHi);
                }
                else
                {
                    eLo = pLo;
                    eHi = pHi;
                }

                const vu1n::VuLaneClass ec = vu1n::vuClassify(eLo, eHi);
                uint8_t laneFlags[4] = {};
                for (uint32_t c = 0; c < 4u; ++c)
                {
                    const uint32_t bit = 1u << c;
                    if ((destLanes & bit) == 0u)
                        continue;
                    const bool neg = (ec.neg & bit) != 0u;
                    uint8_t f = neg ? 0x2u : 0u;
                    if ((ec.zero & bit) != 0u)
                    {
                        f |= 0x1u;
                        const uint32_t bits = neg ? 0x80000000u : 0u;
                        std::memcpy(&result[c], &bits, sizeof(bits));
                    }
                    else if ((ec.over & bit) != 0u)
                    {
                        f |= 0x8u;
                        const uint32_t bits = neg ? 0xFF7FFFFFu : 0x7F7FFFFFu;
                        std::memcpy(&result[c], &bits, sizeof(bits));
                    }
                    else if ((ec.under & bit) != 0u)
                    {
                        f |= 0x5u;
                        const uint32_t bits = neg ? 0x80000000u : 0u;
                        std::memcpy(&result[c], &bits, sizeof(bits));
                    }
                    laneFlags[c] = f;
                }
                uint32_t extraSticky = 0u;
                if constexpr (productSum)
                {
                    const vu1n::VuLaneClass pcl = vu1n::vuClassify(pLo, pHi);
                    if ((pcl.zero & destLanes) != 0u)
                        extraSticky |= 0x1u;
                    if ((pcl.neg & destLanes) != 0u)
                        extraSticky |= 0x2u;
                    if ((pcl.over & destLanes) != 0u)
                        extraSticky |= 0x8u;
                    if ((pcl.under & destLanes) != 0u)
                        extraSticky |= 0x5u;
                }
                updateFmacFlagsC<dest, Imm, NoFlags>(v, laneFlags, extraSticky);
                if constexpr (upperSpecial)
                    applyDestC<dest>(v.m_state.acc, result);
                else
                    applyDestC<dest>(v.m_state.vf[fd], result);
        }
    }

    template <uint32_t U, bool Imm = false, bool NoFlags = false>
    static PS2X_VU1N_INLINE bool fmacFast(I &v)
    {
        constexpr uint8_t dest = DEST(U);
        constexpr uint8_t ft = FT(U);
        constexpr uint8_t fs = FS(U);
        constexpr uint8_t fd = FD(U);
        constexpr uint8_t op = static_cast<uint8_t>(U & 0x3Fu);
        constexpr bool upperSpecial = op >= 0x3Cu;
        constexpr uint8_t sel = upperSpecial ? static_cast<uint8_t>((U & 0x3u) | ((U >> 4) & 0x7Cu)) : op;
        if constexpr (upperSpecial && (sel == 0x2Fu || sel == 0x30u))
        {
            v.m_currentUpperInstruction = U; // NOP, as execUpper
            return true;
        }
        else
        {
            constexpr vu1n::FmacForm form = vu1n::fmacForm(sel);
            if constexpr (form.kind == vu1n::FmacKind::None)
            {
                return false;
            }
            else
            {
                v.m_currentUpperInstruction = U;
                const __m128i vsBits = vu1n::vuLoadNormalized(v.m_state.vf[fs]);
                __m128i rBits;
                if constexpr (form.right == vu1n::FmacRight::Bc)
                    rBits = vu1n::vuBroadcastNormalized(v.m_state.vf[ft][sel & 3u]);
                else if constexpr (form.right == vu1n::FmacRight::Q)
                    rBits = vu1n::vuBroadcastNormalized(v.m_state.q);
                else if constexpr (form.right == vu1n::FmacRight::I)
                    rBits = vu1n::vuBroadcastNormalized(v.m_state.i);
                else
                    rBits = vu1n::vuLoadNormalized(v.m_state.vf[ft]);
                constexpr bool productSum = form.kind == vu1n::FmacKind::Madd || form.kind == vu1n::FmacKind::Msub;
                const __m128 vs = _mm_castsi128_ps(vsBits);
                const __m128 r = _mm_castsi128_ps(rBits);
                const __m128 acc = productSum ? _mm_castsi128_ps(vu1n::vuLoadNormalized(v.m_state.acc)) : _mm_setzero_ps();

                const __m128 prod = _mm_mul_ps(vs, r);
                __m128 res;
                if constexpr (form.kind == vu1n::FmacKind::Add)
                    res = _mm_add_ps(vs, r);
                else if constexpr (form.kind == vu1n::FmacKind::Sub)
                    res = _mm_sub_ps(vs, r);
                else if constexpr (form.kind == vu1n::FmacKind::Madd)
                    res = _mm_add_ps(acc, prod);
                else if constexpr (form.kind == vu1n::FmacKind::Msub)
                    res = _mm_sub_ps(acc, prod);
                else
                    res = prod;
                float result[4];
                _mm_storeu_ps(result, res);

                constexpr uint32_t destLanes = vu1n::kDestToLanes[dest & 0xFu];
                // Ordinary-magnitude lanes need no exact (double) pass: see
                // vuOrdinaryLanes. Only the sign bit can be set, and the float
                // result stands as computed.
                // A lane may skip the exact (double) pass when its float result
                // provably equals the exact one and is ordinary or zero:
                //  * ordinary magnitude (vuOrdinaryLanes): only the sign flag;
                //  * Add/Sub result +-0: float cancellation of two floats is exact;
                //  * a zero operand: the product is exactly +-0 (operands are
                //    normalised, so never Inf/NaN), sign as in double;
                //  * MADD/MSUB: the product must be exact (ordinary or zero
                //    operand); the result must be ordinary, or zero with a zero
                //    product (acc +- 0 is exact).
                // Zero lanes report Z (0x1) plus the sign, as
                // normalizeFmacExactResult does, and the value is already +-0.
                const uint32_t resZero = vu1n::vuZeroLanes(res);
                uint32_t ordinary;
                if constexpr (form.kind == vu1n::FmacKind::Add || form.kind == vu1n::FmacKind::Sub)
                {
                    ordinary = vu1n::vuOrdinaryLanes(res) | resZero;
                }
                else
                {
                    const uint32_t opZero = vu1n::vuZeroLanes(vs) | vu1n::vuZeroLanes(r);
                    if constexpr (productSum)
                        ordinary = (vu1n::vuOrdinaryLanes(prod) | opZero) &
                                   (vu1n::vuOrdinaryLanes(res) | (opZero & resZero));
                    else
                        ordinary = vu1n::vuOrdinaryLanes(res) | (opZero & resZero);
                }
                if ((destLanes & ~ordinary) == 0u)
                {
                    const uint32_t negMask = static_cast<uint32_t>(_mm_movemask_ps(res));
                    uint8_t fastFlags[4] = {};
                    if constexpr ((destLanes & 0x1u) != 0u)
                        fastFlags[0] = static_cast<uint8_t>(((negMask & 0x1u) ? 0x2u : 0u) | ((resZero & 0x1u) ? 0x1u : 0u));
                    if constexpr ((destLanes & 0x2u) != 0u)
                        fastFlags[1] = static_cast<uint8_t>(((negMask & 0x2u) ? 0x2u : 0u) | ((resZero & 0x2u) ? 0x1u : 0u));
                    if constexpr ((destLanes & 0x4u) != 0u)
                        fastFlags[2] = static_cast<uint8_t>(((negMask & 0x4u) ? 0x2u : 0u) | ((resZero & 0x4u) ? 0x1u : 0u));
                    if constexpr ((destLanes & 0x8u) != 0u)
                        fastFlags[3] = static_cast<uint8_t>(((negMask & 0x8u) ? 0x2u : 0u) | ((resZero & 0x8u) ? 0x1u : 0u));
                    uint32_t fastSticky = 0u;
                    if constexpr (productSum)
                    {
                        if ((static_cast<uint32_t>(_mm_movemask_ps(prod)) & destLanes) != 0u)
                            fastSticky |= 0x2u;
                        if ((vu1n::vuZeroLanes(prod) & destLanes) != 0u)
                            fastSticky |= 0x1u;
                    }
                    updateFmacFlagsC<dest, Imm, NoFlags>(v, fastFlags, fastSticky);
                    if constexpr (upperSpecial)
                        applyDestC<dest>(v.m_state.acc, result);
                    else
                        applyDestC<dest>(v.m_state.vf[fd], result);
                    return true;
                }

                // Some lane is zero / denormal / huge: exact (double) pass, cold.
                fmacSlow<U, Imm, NoFlags>(v);
                return true;
            }
        }
    }

    // VU1Interpreter::normalizeOperand: denormal -> signed zero, Inf/NaN ->
    // signed FLT_MAX, anything else unchanged.
    static PS2X_VU1N_INLINE float normOp(float value)
    {
        uint32_t bits = 0u;
        std::memcpy(&bits, &value, sizeof(bits));
        const uint32_t exponent = (bits >> 23) & 0xFFu;
        if (exponent == 0u)
            bits &= 0x80000000u;
        else if (exponent == 0xFFu)
            bits = (bits & 0x80000000u) | 0x7F7FFFFFu;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    // ps2_vu1_upper.cpp vuFloatToInt, verbatim.
    static PS2X_VU1N_INLINE int32_t floatToInt(float value, float scale)
    {
        const double scaled = static_cast<double>(value) * static_cast<double>(scale);
        if (scaled >= static_cast<double>(std::numeric_limits<int32_t>::max()))
            return std::numeric_limits<int32_t>::max();
        if (scaled <= static_cast<double>(std::numeric_limits<int32_t>::min()))
            return std::numeric_limits<int32_t>::min();
        return static_cast<int32_t>(scaled);
    }

    // The per-vertex upper instructions that are not FMAC forms, so fmacFast
    // does not take them: MAX/MINI (vs VT lane, broadcast VT or I), ITOF,
    // FTOI, ABS and CLIP. Bodies mirror execUpper's cases line for line --
    // same operand normalisation (none for ITOF and CLIP, which read raw
    // bits), same comparison form (a > b ? a : b), same destinations (FD for
    // MAX/MINI, FT for the special group). Returns false for anything else.
    template <uint32_t U>
    static PS2X_VU1N_INLINE bool nonFmacFast(I &v)
    {
        constexpr uint8_t dest = DEST(U);
        constexpr uint8_t ft = FT(U);
        constexpr uint8_t fs = FS(U);
        constexpr uint8_t fd = FD(U);
        constexpr uint8_t op = static_cast<uint8_t>(U & 0x3Fu);
        constexpr bool upperSpecial = op >= 0x3Cu;
        constexpr uint8_t sel = upperSpecial ? static_cast<uint8_t>((U & 0x3u) | ((U >> 4) & 0x7Cu)) : op;

        constexpr bool isMaxMini = !upperSpecial &&
            ((op >= 0x10u && op <= 0x17u) || op == 0x1Du || op == 0x1Fu || op == 0x2Bu || op == 0x2Fu);
        constexpr bool isItof = upperSpecial && sel >= 0x10u && sel <= 0x13u;
        constexpr bool isFtoi = upperSpecial && sel >= 0x14u && sel <= 0x17u;
        constexpr bool isAbs = upperSpecial && sel == 0x1Du;
        constexpr bool isClip = upperSpecial && sel == 0x1Fu;

        if constexpr (isMaxMini)
        {
            v.m_currentUpperInstruction = U;
            const float *sv = v.m_state.vf[fs];
            const float *tv = v.m_state.vf[ft];
            float result[4];
            if constexpr (op >= 0x10u && op <= 0x13u) // MAXbc
            {
                const float bc = normOp(tv[op & 3u]);
                for (int c = 0; c < 4; ++c)
                {
                    const float a = normOp(sv[c]);
                    result[c] = (a > bc) ? a : bc;
                }
            }
            else if constexpr (op >= 0x14u && op <= 0x17u) // MINIbc
            {
                const float bc = normOp(tv[op & 3u]);
                for (int c = 0; c < 4; ++c)
                {
                    const float a = normOp(sv[c]);
                    result[c] = (a < bc) ? a : bc;
                }
            }
            else if constexpr (op == 0x1Du) // MAXi
            {
                const float i = normOp(v.m_state.i);
                for (int c = 0; c < 4; ++c)
                {
                    const float a = normOp(sv[c]);
                    result[c] = (a > i) ? a : i;
                }
            }
            else if constexpr (op == 0x1Fu) // MINIi
            {
                const float i = normOp(v.m_state.i);
                for (int c = 0; c < 4; ++c)
                {
                    const float a = normOp(sv[c]);
                    result[c] = (a < i) ? a : i;
                }
            }
            else if constexpr (op == 0x2Bu) // MAX
            {
                for (int c = 0; c < 4; ++c)
                {
                    const float a = normOp(sv[c]);
                    const float b = normOp(tv[c]);
                    result[c] = (a > b) ? a : b;
                }
            }
            else // 0x2F MINI
            {
                for (int c = 0; c < 4; ++c)
                {
                    const float a = normOp(sv[c]);
                    const float b = normOp(tv[c]);
                    result[c] = (a < b) ? a : b;
                }
            }
            applyDestC<dest>(v.m_state.vf[fd], result);
            return true;
        }
        else if constexpr (isItof)
        {
            v.m_currentUpperInstruction = U;
            constexpr float scale = sel == 0x10u ? 1.0f : sel == 0x11u ? 16.0f : sel == 0x12u ? 4096.0f : 32768.0f;
            float result[4];
            for (int c = 0; c < 4; ++c)
            {
                int32_t iv;
                std::memcpy(&iv, &v.m_state.vf[fs][c], 4);
                if constexpr (sel == 0x10u)
                    result[c] = static_cast<float>(iv);
                else
                    result[c] = static_cast<float>(iv) / scale;
            }
            applyDestC<dest>(v.m_state.vf[ft], result);
            return true;
        }
        else if constexpr (isFtoi)
        {
            v.m_currentUpperInstruction = U;
            constexpr float scale = sel == 0x14u ? 1.0f : sel == 0x15u ? 16.0f : sel == 0x16u ? 4096.0f : 32768.0f;
            float result[4];
            for (int c = 0; c < 4; ++c)
            {
                const int32_t iv = floatToInt(normOp(v.m_state.vf[fs][c]), scale);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDestC<dest>(v.m_state.vf[ft], result);
            return true;
        }
        else if constexpr (isAbs)
        {
            v.m_currentUpperInstruction = U;
            float result[4];
            for (int c = 0; c < 4; ++c)
                result[c] = std::fabs(normOp(v.m_state.vf[fs][c]));
            applyDestC<dest>(v.m_state.vf[ft], result);
            return true;
        }
        else if constexpr (isClip)
        {
            v.m_currentUpperInstruction = U;
            uint32_t wBits = 0u;
            std::memcpy(&wBits, &v.m_state.vf[ft][3], sizeof(wBits));
            const int32_t limit = (wBits & 0x7F800000u) != 0u ? static_cast<int32_t>(wBits & 0x7FFFFFFFu) : 0x007FFFFF;
            const auto exceeds = [limit](float value, uint32_t signMask)
            {
                uint32_t bits = 0u;
                std::memcpy(&bits, &value, sizeof(bits));
                bits ^= signMask;
                int32_t orderedBits = 0;
                std::memcpy(&orderedBits, &bits, sizeof(orderedBits));
                return orderedBits > limit;
            };
            const float *sv = v.m_state.vf[fs];
            uint32_t flags = 0u;
            if (exceeds(sv[0], 0x00000000u))
                flags |= 0x01u;
            if (exceeds(sv[0], 0x80000000u))
                flags |= 0x02u;
            if (exceeds(sv[1], 0x00000000u))
                flags |= 0x04u;
            if (exceeds(sv[1], 0x80000000u))
                flags |= 0x08u;
            if (exceeds(sv[2], 0x00000000u))
                flags |= 0x10u;
            if (exceeds(sv[2], 0x80000000u))
                flags |= 0x20u;
            v.queueClip(flags);
            return true;
        }
        else
        {
            return false;
        }
    }

    template <uint32_t U, bool Imm = false, bool NoFlags = false>
    static PS2X_VU1N_INLINE void execUpperN(I &v)
    {
        if (!fmacFast<U, Imm, NoFlags>(v) && !nonFmacFast<U>(v))
            v.execUpper(U);
    }

    // Mirrors VU1Interpreter::advanceOneCycle(), with the two callees' own
    // early-out conditions hoisted so a quiet cycle costs two compares.
    static PS2X_VU1N_INLINE void advanceOneCycleFast(I &v)
    {
        ++v.m_cycle;
        v.m_state.cycles = v.m_cycle;
        if (v.m_cycle >= v.m_pipelineNextReady)
            v.commitReadyPipelines();
        if (v.m_xgkick.active)
            v.progressXgkick();
    }

    static PS2X_VU1N_INLINE void setBranch(I &v, uint32_t target)
    {
        v.m_state.branchPending = true;
        v.m_state.branchTarget = target;
        v.m_state.branchDelay = 1u;
    }

    // execLower with the instruction a constant: the hot ops (loads, stores,
    // integer ALU, moves, branches -- bodies copied from execLower) inline;
    // everything else (DIV/SQRT/RSQRT, XGKICK, XTOP/XITOP, flag reads, LQI/
    // SQI/LQD/SQD, ILWR/ISWR, EFU, R*) still goes through execLower.
    template <uint32_t U, uint32_t L, bool Vu0 = false>
    static PS2X_VU1N_INLINE void execLowerN(I &v, Vu1NativeCtx &c)
    {
        if constexpr (L == 0x00000000u || L == 0x8000033Cu)
        {
            return; // NOP
        }
        else
        {
            constexpr uint8_t opHi = static_cast<uint8_t>((L >> 25) & 0x7Fu);
            constexpr uint8_t dest = static_cast<uint8_t>((L >> 21) & 0xFu);
            constexpr uint8_t vit = VIT(L);
            constexpr uint8_t vis = VIS(L);
            constexpr uint8_t vfT = FT(L);
            constexpr uint8_t vfS = FS(L);
            constexpr int16_t imm11 = IMM11(L);
            constexpr uint32_t pcMask = Vu0 ? 0x0FFFu : 0x3FFFu; // microAddressMask() of the unit
            uint8_t *vuData = c.vuData;
            const uint32_t dataSize = c.dataSize;
            if constexpr (opHi == 0x00u) // LQ
            {
                uint32_t addr = (static_cast<uint32_t>(static_cast<int32_t>(v.m_state.vi[vis] + imm11))) * 16u;
                addr &= (dataSize - 1u);
                if (addr + 16u <= dataSize)
                {
                    float tmp[4];
                    std::memcpy(tmp, vuData + addr, 16);
                    applyDestC<dest>(v.m_state.vf[vfT], tmp); // VF destination is the 5-bit FT field
                }
            }
            else if constexpr (opHi == 0x01u) // SQ
            {
                uint32_t addr = (static_cast<uint32_t>(static_cast<int32_t>(v.m_state.vi[vit] + imm11))) * 16u;
                addr &= (dataSize - 1u);
                if (addr + 16u <= dataSize)
                {
                    uint32_t words[4]{};
                    std::memcpy(words, v.m_state.vf[vfS], sizeof(words));
                    v.queueStore(addr, words, dest);
                }
            }
            else if constexpr (opHi == 0x04u) // ILW
            {
                uint32_t addr = (static_cast<uint32_t>(static_cast<int32_t>(v.m_state.vi[vis] + imm11))) * 16u;
                addr &= (dataSize - 1u);
                if (addr + 16u <= dataSize)
                {
                    constexpr int comp = (dest & 0x8u) ? 0 : (dest & 0x4u) ? 1 : (dest & 0x2u) ? 2 : 3;
                    uint32_t w;
                    std::memcpy(&w, vuData + addr + comp * 4, 4);
                    if constexpr (vit != 0u)
                        v.m_state.vi[vit] = static_cast<int32_t>(static_cast<int16_t>(w & 0xFFFFu));
                }
            }
            else if constexpr (opHi == 0x05u) // ISW
            {
                uint32_t addr = (static_cast<uint32_t>(static_cast<int32_t>(v.m_state.vi[vis] + imm11))) * 16u;
                addr &= (dataSize - 1u);
                if (addr + 16u <= dataSize)
                {
                    const uint32_t val = static_cast<uint32_t>(static_cast<uint16_t>(v.m_state.vi[vit] & 0xFFFF));
                    const uint32_t words[4] = {val, val, val, val};
                    v.queueStore(addr, words, dest);
                }
            }
            else if constexpr (opHi == 0x08u || opHi == 0x09u) // IADDIU / ISUBIU
            {
                constexpr int16_t imm = static_cast<int16_t>((L & 0x7FFu) | ((L >> 10) & 0x7800u));
                if constexpr (vit != 0u)
                {
                    if constexpr (opHi == 0x08u)
                        v.m_state.vi[vit] = static_cast<int16_t>(v.m_state.vi[vis] + imm);
                    else
                        v.m_state.vi[vit] = static_cast<int16_t>(v.m_state.vi[vis] - imm);
                }
            }
            else if constexpr (opHi == 0x20u) // B
            {
                setBranch(v, (v.m_state.pc + 8u + imm11 * 8) & pcMask);
            }
            else if constexpr (opHi == 0x21u) // BAL
            {
                const uint32_t target = (v.m_state.pc + 8u + imm11 * 8) & pcMask;
                if constexpr (vit != 0u)
                    v.m_state.vi[vit] = static_cast<int32_t>((v.m_state.pc + 16u) / 8u);
                setBranch(v, target);
            }
            else if constexpr (opHi == 0x24u) // JR
            {
                setBranch(v, (static_cast<uint32_t>(static_cast<uint16_t>(v.readBranchVi(vis))) * 8u) & pcMask);
            }
            else if constexpr (opHi == 0x25u) // JALR
            {
                const uint32_t target = (static_cast<uint32_t>(static_cast<uint16_t>(v.readBranchVi(vis))) * 8u) & pcMask;
                if constexpr (vit != 0u)
                    v.m_state.vi[vit] = static_cast<int32_t>((v.m_state.pc + 16u) / 8u);
                setBranch(v, target);
            }
            else if constexpr (opHi == 0x28u) // IBEQ
            {
                if (static_cast<int16_t>(v.readBranchVi(vis)) == static_cast<int16_t>(v.readBranchVi(vit)))
                    setBranch(v, (v.m_state.pc + 8u + imm11 * 8) & pcMask);
            }
            else if constexpr (opHi == 0x29u) // IBNE
            {
                if (static_cast<int16_t>(v.readBranchVi(vis)) != static_cast<int16_t>(v.readBranchVi(vit)))
                    setBranch(v, (v.m_state.pc + 8u + imm11 * 8) & pcMask);
            }
            else if constexpr (opHi == 0x2Cu) // IBLTZ
            {
                if (static_cast<int16_t>(v.readBranchVi(vis)) < 0)
                    setBranch(v, (v.m_state.pc + 8u + imm11 * 8) & pcMask);
            }
            else if constexpr (opHi == 0x2Du) // IBGTZ
            {
                if (static_cast<int16_t>(v.readBranchVi(vis)) > 0)
                    setBranch(v, (v.m_state.pc + 8u + imm11 * 8) & pcMask);
            }
            else if constexpr (opHi == 0x2Eu) // IBLEZ
            {
                if (static_cast<int16_t>(v.readBranchVi(vis)) <= 0)
                    setBranch(v, (v.m_state.pc + 8u + imm11 * 8) & pcMask);
            }
            else if constexpr (opHi == 0x2Fu) // IBGEZ
            {
                if (static_cast<int16_t>(v.readBranchVi(vis)) >= 0)
                    setBranch(v, (v.m_state.pc + 8u + imm11 * 8) & pcMask);
            }
            else if constexpr (opHi == 0x40u && (L & 0x3Fu) == 0x30u) // IADD
            {
                constexpr uint8_t vid = VID(L);
                if constexpr (vid != 0u)
                    v.m_state.vi[vid] = static_cast<int16_t>(v.m_state.vi[vis] + v.m_state.vi[vit]);
            }
            else if constexpr (opHi == 0x40u && (L & 0x3Fu) == 0x31u) // ISUB
            {
                constexpr uint8_t vid = VID(L);
                if constexpr (vid != 0u)
                    v.m_state.vi[vid] = static_cast<int16_t>(v.m_state.vi[vis] - v.m_state.vi[vit]);
            }
            else if constexpr (opHi == 0x40u && (L & 0x3Fu) == 0x32u) // IADDI
            {
                constexpr int16_t imm5 = static_cast<int16_t>(static_cast<int32_t>((L >> 6) & 0x1Fu) << 27 >> 27);
                if constexpr (vit != 0u)
                    v.m_state.vi[vit] = static_cast<int16_t>(v.m_state.vi[vis] + imm5);
            }
            else if constexpr (opHi == 0x40u && (L & 0x3Fu) == 0x34u) // IAND
            {
                constexpr uint8_t vid = VID(L);
                if constexpr (vid != 0u)
                    v.m_state.vi[vid] = v.m_state.vi[vis] & v.m_state.vi[vit];
            }
            else if constexpr (opHi == 0x40u && (L & 0x3Fu) == 0x35u) // IOR
            {
                constexpr uint8_t vid = VID(L);
                if constexpr (vid != 0u)
                    v.m_state.vi[vid] = v.m_state.vi[vis] | v.m_state.vi[vit];
            }
            // Flag reads, Q ops and post/pre-increment quadword moves: bodies
            // copied from execLower (they were ~2.4M interpreter fallbacks/s in a
            // fight). Same helpers, same order, so results are identical.
            else if constexpr (opHi == 0x10u) // FCEQ
            {
                v.syncFlagsForRead("FCEQ");
                v.m_state.vi[1] = ((v.m_state.clip & 0xFFFFFFu) == (L & 0xFFFFFFu)) ? 1 : 0;
            }
            else if constexpr (opHi == 0x12u) // FCAND
            {
                v.syncFlagsForRead("FCAND");
                v.m_state.vi[1] = ((v.m_state.clip & (L & 0xFFFFFFu)) != 0u) ? 1 : 0;
            }
            else if constexpr (opHi == 0x13u) // FCOR
            {
                v.syncFlagsForRead("FCOR");
                v.m_state.vi[1] = ((v.m_state.clip | (L & 0xFFFFFFu)) == 0xFFFFFFu) ? 1 : 0;
            }
            else if constexpr (opHi == 0x16u) // FSAND
            {
                v.syncFlagsForRead("FSAND");
                constexpr uint16_t imm12 = static_cast<uint16_t>((((L >> 21) & 0x1u) << 11) | (L & 0x7FFu));
                if constexpr (vit != 0u)
                    v.m_state.vi[vit] = static_cast<int32_t>((v.m_state.status & 0xFFFu) & imm12);
            }
            else if constexpr (opHi == 0x1Au) // FMAND
            {
                v.syncFlagsForRead("FMAND");
                if constexpr (vit != 0u)
                    v.m_state.vi[vit] = static_cast<int32_t>(v.m_state.mac & static_cast<uint32_t>(static_cast<uint16_t>(v.m_state.vi[vis])));
            }
            else if constexpr (opHi == 0x1Cu) // FCGET
            {
                v.syncFlagsForRead("FCGET");
                if constexpr (vit != 0u)
                    v.m_state.vi[vit] = static_cast<int32_t>(v.m_state.clip & 0x0FFFu);
            }
            else if constexpr (opHi == 0x40u && (L & 0x3Fu) >= 0x3Cu && ((L & 0x3u) | ((L >> 4) & 0x7Cu)) == 0x34u) // LQI
            {
                uint32_t addr = (static_cast<uint32_t>(static_cast<uint16_t>(v.m_state.vi[vis]))) * 16u;
                addr &= (dataSize - 1u);
                if (addr + 16u <= dataSize)
                {
                    float tmp[4];
                    std::memcpy(tmp, vuData + addr, 16);
                    applyDestC<dest>(v.m_state.vf[vfT], tmp);
                }
                if constexpr (vis != 0u)
                    v.m_state.vi[vis] = static_cast<int16_t>(v.m_state.vi[vis] + 1);
            }
            else if constexpr (opHi == 0x40u && (L & 0x3Fu) >= 0x3Cu && ((L & 0x3u) | ((L >> 4) & 0x7Cu)) == 0x35u) // SQI
            {
                uint32_t addr = (static_cast<uint32_t>(static_cast<uint16_t>(v.m_state.vi[vit]))) * 16u;
                addr &= (dataSize - 1u);
                if (addr + 16u <= dataSize)
                {
                    uint32_t words[4]{};
                    std::memcpy(words, v.m_state.vf[vfS], sizeof(words));
                    v.queueStore(addr, words, dest);
                }
                if constexpr (vit != 0u)
                    v.m_state.vi[vit] = static_cast<int16_t>(v.m_state.vi[vit] + 1);
            }
            else if constexpr (opHi == 0x40u && (L & 0x3Fu) >= 0x3Cu && ((L & 0x3u) | ((L >> 4) & 0x7Cu)) == 0x36u) // LQD
            {
                if constexpr (vis != 0u)
                    v.m_state.vi[vis] = static_cast<int16_t>(v.m_state.vi[vis] - 1);
                uint32_t addr = (static_cast<uint32_t>(static_cast<uint16_t>(v.m_state.vi[vis]))) * 16u;
                addr &= (dataSize - 1u);
                if (addr + 16u <= dataSize)
                {
                    float tmp[4];
                    std::memcpy(tmp, vuData + addr, 16);
                    applyDestC<dest>(v.m_state.vf[vfT], tmp);
                }
            }
            else if constexpr (opHi == 0x40u && (L & 0x3Fu) >= 0x3Cu && ((L & 0x3u) | ((L >> 4) & 0x7Cu)) == 0x37u) // SQD
            {
                if constexpr (vit != 0u)
                    v.m_state.vi[vit] = static_cast<int16_t>(v.m_state.vi[vit] - 1);
                uint32_t addr = (static_cast<uint32_t>(static_cast<uint16_t>(v.m_state.vi[vit]))) * 16u;
                addr &= (dataSize - 1u);
                if (addr + 16u <= dataSize)
                {
                    uint32_t words[4]{};
                    std::memcpy(words, v.m_state.vf[vfS], sizeof(words));
                    v.queueStore(addr, words, dest);
                }
            }
            else if constexpr (opHi == 0x40u && (L & 0x3Fu) >= 0x3Cu && ((L & 0x3u) | ((L >> 4) & 0x7Cu)) == 0x38u) // DIV
            {
                constexpr int fsf = (L >> 21) & 0x3;
                constexpr int ftf = (L >> 23) & 0x3;
                const float num = v.normalizeOperand(v.m_state.vf[vfS][fsf]);
                const float den = v.normalizeOperand(v.m_state.vf[vfT][ftf]);
                uint32_t statusDi = 0u;
                float result = 0.0f;
                if (den == 0.0f)
                {
                    statusDi = num == 0.0f ? 0x10u : 0x20u;
                    result = std::signbit(num) != std::signbit(den) ? -std::numeric_limits<float>::max()
                                                                    : std::numeric_limits<float>::max();
                }
                else
                {
                    result = num / den;
                }
                uint32_t ignoredFlags = 0u;
                result = v.normalizeResult(result, ignoredFlags);
                v.queueQ(result, 7u, statusDi);
            }
            else if constexpr (opHi == 0x40u && (L & 0x3Fu) >= 0x3Cu && ((L & 0x3u) | ((L >> 4) & 0x7Cu)) == 0x39u) // SQRT
            {
                constexpr int ftf = (L >> 23) & 0x3;
                const float val = v.normalizeOperand(v.m_state.vf[vfT][ftf]);
                v.queueQ(std::sqrt(std::fabs(val)), 7u, val < 0.0f ? 0x10u : 0u);
            }
            else if constexpr (opHi == 0x40u && (L & 0x3Fu) >= 0x3Cu && ((L & 0x3u) | ((L >> 4) & 0x7Cu)) == 0x3Au) // RSQRT
            {
                constexpr int fsf = (L >> 21) & 0x3;
                constexpr int ftf = (L >> 23) & 0x3;
                const float num = v.normalizeOperand(v.m_state.vf[vfS][fsf]);
                const float radicand = v.normalizeOperand(v.m_state.vf[vfT][ftf]);
                const float den = std::sqrt(std::fabs(radicand));
                uint32_t statusDi = radicand < 0.0f ? 0x10u : 0u;
                float result = 0.0f;
                if (den != 0.0f)
                    result = num / den;
                else
                {
                    statusDi = num == 0.0f ? 0x10u : 0x20u;
                    result = std::signbit(num) ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max();
                }
                uint32_t ignoredFlags = 0u;
                result = v.normalizeResult(result, ignoredFlags);
                v.queueQ(result, 13u, statusDi);
            }
            else if constexpr (opHi == 0x40u && (L & 0x3Fu) >= 0x3Cu &&
                               ((L & 0x3u) | ((L >> 4) & 0x7Cu)) == 0x30u) // MOVE
            {
                float tmp[4];
                std::memcpy(tmp, v.m_state.vf[vfS], 16);
                applyDestC<dest>(v.m_state.vf[vfT], tmp);
            }
            else if constexpr (opHi == 0x40u && (L & 0x3Fu) >= 0x3Cu &&
                               ((L & 0x3u) | ((L >> 4) & 0x7Cu)) == 0x31u) // MR32
            {
                float tmp[4] = {v.m_state.vf[vfS][1], v.m_state.vf[vfS][2], v.m_state.vf[vfS][3], v.m_state.vf[vfS][0]};
                applyDestC<dest>(v.m_state.vf[vfT], tmp);
            }
            else if constexpr (opHi == 0x40u && (L & 0x3Fu) >= 0x3Cu &&
                               ((L & 0x3u) | ((L >> 4) & 0x7Cu)) == 0x3Cu) // MTIR
            {
                constexpr uint32_t comp = (L >> 21) & 0x3u;
                uint32_t fval;
                std::memcpy(&fval, &v.m_state.vf[vfS][comp], 4);
                if constexpr (vit != 0u)
                    v.m_state.vi[vit] = static_cast<int32_t>(static_cast<int16_t>(fval & 0xFFFFu));
            }
            else if constexpr (opHi == 0x40u && (L & 0x3Fu) >= 0x3Cu &&
                               ((L & 0x3u) | ((L >> 4) & 0x7Cu)) == 0x3Du) // MFIR
            {
                float result[4];
                const int32_t val = static_cast<int32_t>(static_cast<int16_t>(v.m_state.vi[vis] & 0xFFFF));
                std::memcpy(&result[0], &val, 4);
                result[1] = result[0];
                result[2] = result[0];
                result[3] = result[0];
                applyDestC<dest>(v.m_state.vf[vfT], result);
            }
            else
            {
                v.execLower(L, c.vuData, c.dataSize, *c.gs, c.memory, U);
            }
        }
    }

    // max() of the ready cycles of one VF register's components in `lanes`.
    template <uint8_t Reg, uint8_t Lanes>
    static PS2X_VU1N_INLINE uint64_t vfReadyMax(const I &v, uint64_t ready)
    {
        if constexpr ((Lanes & laneForComponent(0u)) != 0u)
            ready = std::max(ready, v.m_vfReady[Reg][0]);
        if constexpr ((Lanes & laneForComponent(1u)) != 0u)
            ready = std::max(ready, v.m_vfReady[Reg][1]);
        if constexpr ((Lanes & laneForComponent(2u)) != 0u)
            ready = std::max(ready, v.m_vfReady[Reg][2]);
        if constexpr ((Lanes & laneForComponent(3u)) != 0u)
            ready = std::max(ready, v.m_vfReady[Reg][3]);
        return ready;
    }

    // calculatePairReadyCycle with the usage lists constant (loops unrolled
    // at compile time: MSVC would otherwise walk the decode tables at run time).
    // NoReady: gen_vu1_native.py proved no VF/VI/ACC write that this pair
    // reads can still be in flight here, so the table scan cannot raise
    // `ready` and is compiled out.
    template <uint32_t U, uint32_t L, bool NoReady = false, bool Vu0 = false>
    static PS2X_VU1N_INLINE uint64_t readyCycle(const I &v)
    {
        constexpr const D &d = PairDecode<U, L, Vu0>::d;
        uint64_t ready = v.m_cycle;
        if (!NoReady && v.m_maxReadyCycle > v.m_cycle)
        {
            ready = vfReadyMax<d.upperUsage.vfRead[0].reg, (d.upperUsage.vfReadCount > 0u ? d.upperUsage.vfRead[0].lanes : uint8_t(0))>(v, ready);
            ready = vfReadyMax<d.upperUsage.vfRead[1].reg, (d.upperUsage.vfReadCount > 1u ? d.upperUsage.vfRead[1].lanes : uint8_t(0))>(v, ready);
            ready = vfReadyMax<d.lowerUsage.vfRead[0].reg, (d.lowerUsage.vfReadCount > 0u ? d.lowerUsage.vfRead[0].lanes : uint8_t(0))>(v, ready);
            ready = vfReadyMax<d.lowerUsage.vfRead[1].reg, (d.lowerUsage.vfReadCount > 1u ? d.lowerUsage.vfRead[1].lanes : uint8_t(0))>(v, ready);
            constexpr uint32_t viBits = (static_cast<uint32_t>(d.upperUsage.viRead) | d.lowerUsage.viRead) & 0xFFFEu;
            if constexpr ((viBits >> 1u) & 1u)
                ready = std::max(ready, v.m_viReady[1]);
            if constexpr ((viBits >> 2u) & 1u)
                ready = std::max(ready, v.m_viReady[2]);
            if constexpr ((viBits >> 3u) & 1u)
                ready = std::max(ready, v.m_viReady[3]);
            if constexpr ((viBits >> 4u) & 1u)
                ready = std::max(ready, v.m_viReady[4]);
            if constexpr ((viBits >> 5u) & 1u)
                ready = std::max(ready, v.m_viReady[5]);
            if constexpr ((viBits >> 6u) & 1u)
                ready = std::max(ready, v.m_viReady[6]);
            if constexpr ((viBits >> 7u) & 1u)
                ready = std::max(ready, v.m_viReady[7]);
            if constexpr ((viBits >> 8u) & 1u)
                ready = std::max(ready, v.m_viReady[8]);
            if constexpr ((viBits >> 9u) & 1u)
                ready = std::max(ready, v.m_viReady[9]);
            if constexpr ((viBits >> 10u) & 1u)
                ready = std::max(ready, v.m_viReady[10]);
            if constexpr ((viBits >> 11u) & 1u)
                ready = std::max(ready, v.m_viReady[11]);
            if constexpr ((viBits >> 12u) & 1u)
                ready = std::max(ready, v.m_viReady[12]);
            if constexpr ((viBits >> 13u) & 1u)
                ready = std::max(ready, v.m_viReady[13]);
            if constexpr ((viBits >> 14u) & 1u)
                ready = std::max(ready, v.m_viReady[14]);
            if constexpr ((viBits >> 15u) & 1u)
                ready = std::max(ready, v.m_viReady[15]);
            constexpr uint32_t accBits = d.upperUsage.accRead | d.lowerUsage.accRead;
            if constexpr ((accBits & laneForComponent(0u)) != 0u)
                ready = std::max(ready, v.m_accReady[0]);
            if constexpr ((accBits & laneForComponent(1u)) != 0u)
                ready = std::max(ready, v.m_accReady[1]);
            if constexpr ((accBits & laneForComponent(2u)) != 0u)
                ready = std::max(ready, v.m_accReady[2]);
            if constexpr ((accBits & laneForComponent(3u)) != 0u)
                ready = std::max(ready, v.m_accReady[3]);
        }
        if constexpr (d.lowerUsage.pipeline == I::PipelineFdiv)
            if (v.m_fdiv.valid)
                ready = std::max(ready, v.m_fdiv.readyCycle);
        if constexpr (d.lowerUsage.pipeline == I::PipelineEfu)
            ready = std::max(ready, v.m_efuResourceReady);
        if constexpr (d.lowerUsage.waitQ)
            if (v.m_fdiv.valid)
                ready = std::max(ready, v.m_fdiv.readyCycle);
        if constexpr (d.lowerUsage.waitP)
        {
            for (const I::ScalarPipelineEntry &entry : v.m_efu)
                if (entry.valid)
                    ready = std::max(ready, entry.readyCycle);
        }
        if constexpr (d.lowerUsage.pipeline == I::PipelineXgkick)
            if (v.m_xgkick.active)
                ready = std::max(ready, v.m_cycle + 1u);
        return ready;
    }

    // markPairWrites with the write lists constant.
    template <uint32_t U, uint32_t L, bool Vu0 = false>
    static PS2X_VU1N_INLINE void markWrites(I &v)
    {
        constexpr const D &d = PairDecode<U, L, Vu0>::d;
        constexpr auto lowerWrite = d.lowerUsage.vfWrite;
        if constexpr (lowerWrite.reg != 0u && d.suppressedLowerVf != lowerWrite.reg)
        {
            constexpr uint32_t latency = d.lowerUsage.vfLatency != 0u ? d.lowerUsage.vfLatency : d.lowerUsage.latency;
            const uint64_t readyAt = v.m_cycle + latency;
            for (uint32_t c = 0; c < 4u; ++c)
                if ((lowerWrite.lanes & laneForComponent(c)) != 0u)
                    v.m_vfReady[lowerWrite.reg][c] = readyAt;
            v.m_vfPendingMask |= 1u << lowerWrite.reg;
            if (readyAt > v.m_maxReadyCycle)
                v.m_maxReadyCycle = readyAt;
        }
        constexpr auto upperWrite = d.upperUsage.vfWrite;
        if constexpr (upperWrite.reg != 0u)
        {
            constexpr uint32_t latency = d.upperUsage.vfLatency != 0u ? d.upperUsage.vfLatency : d.upperUsage.latency;
            const uint64_t readyAt = v.m_cycle + latency;
            for (uint32_t c = 0; c < 4u; ++c)
                if ((upperWrite.lanes & laneForComponent(c)) != 0u)
                    v.m_vfReady[upperWrite.reg][c] = readyAt;
            v.m_vfPendingMask |= 1u << upperWrite.reg;
            if (readyAt > v.m_maxReadyCycle)
                v.m_maxReadyCycle = readyAt;
        }
        constexpr uint32_t viWriteBits = static_cast<uint32_t>(d.lowerUsage.viWrite) & 0xFFFEu;
        if constexpr (viWriteBits != 0u)
        {
            constexpr uint32_t latency = d.lowerUsage.viLatency != 0u ? d.lowerUsage.viLatency : d.lowerUsage.latency;
            const uint64_t readyAt = v.m_cycle + latency;
            for (uint32_t bits = viWriteBits; bits != 0u; bits &= bits - 1u)
            {
                const uint32_t reg = static_cast<uint32_t>(std::countr_zero(bits));
                v.m_viReady[reg] = readyAt;
                v.m_viPendingMask |= static_cast<uint16_t>(1u << reg);
            }
            if (readyAt > v.m_maxReadyCycle)
                v.m_maxReadyCycle = readyAt;
        }
        if constexpr (d.upperUsage.accWrite != 0u)
        {
            const uint64_t readyAt = v.m_cycle + I::kAccForwardLatency;
            for (uint32_t c = 0; c < 4u; ++c)
                if ((d.upperUsage.accWrite & laneForComponent(c)) != 0u)
                    v.m_accReady[c] = readyAt;
            v.m_accPending = true;
            if (readyAt > v.m_maxReadyCycle)
                v.m_maxReadyCycle = readyAt;
        }
    }

    // One iteration of VU1Interpreter::run()'s loop (fast mode: immediate
    // register writes), with the decoded pair a constant.
    // Inlined into the generated chunk functions (gen_vu1_native.py splits
    // each image into CHUNK-pair functions so MSVC can cope: all 2048 in one
    // function exhausted the compiler's heap in pass 2).
    template <uint32_t U, uint32_t L, bool Imm = false, bool NoReady = false, bool Relax = false, bool NoFlags = false, bool Mid = false, bool Vu0 = false>
    static Vu1NativeStep stepPair(I &v, Vu1NativeCtx &c)
    {
        constexpr const D &d = PairDecode<U, L, Vu0>::d;
        PS2X_VU1_TRACE_PAIR(v);
        // Pipelines (Q from DIV/SQRT/RSQRT, P, stores, XGKICK) keep real cycle
        // timing in relaxed mode too: WotM's microcode starts the next DIV
        // before multiplying by the previous Q (d166 0x1bd0/0x1c08/0x1c10), so an
        // instant Q put the wrong 1/w on building vertices.
        if (v.m_cycle >= v.m_pipelineNextReady)
            v.commitReadyPipelines();
        if constexpr (d.upperUsage.reserved || d.lowerUsage.reserved)
        {
            v.reportReservedInstruction(d.upperUsage.reserved, d.upperUsage.reserved ? U : L);
            return Vu1NativeStep::Stop;
        }
        else
        {
            if constexpr (!Relax)
            {
                uint64_t ready = readyCycle<U, L, NoReady, Vu0>(v);
                while (ready > v.m_cycle)
                {
                    if (ready >= c.budgetEnd)
                    {
                        v.advanceTo(c.budgetEnd);
                        break;
                    }
                    v.advanceTo(ready);
                    ready = readyCycle<U, L, NoReady, Vu0>(v);
                }
            }
            // Mid (relaxed block compiler): a pair inside a straight run. The
            // budget is checked, and PC/branch/halt state updated, only by the
            // run's non-Mid pairs (the generator sets PC before each of them).
            if constexpr (!Mid)
            {
                if (v.m_cycle >= c.budgetEnd)
                    return Vu1NativeStep::Stop;
            }

            constexpr uint32_t viBits = static_cast<uint32_t>(d.lowerUsage.viWrite) & 0xFFFEu;
            constexpr uint8_t writtenVi = viBits != 0u ? static_cast<uint8_t>(std::countr_zero(viBits)) : 0u;
            const int32_t oldVi = writtenVi != 0u ? v.m_state.vi[writtenVi] : 0;

            if constexpr (d.iBit)
            {
                execUpperN<U, Imm, NoFlags>(v);
                float immediate = 0.0f;
                constexpr uint32_t bits = L;
                std::memcpy(&immediate, &bits, sizeof(immediate));
                v.m_state.i = v.normalizeOperand(immediate);
            }
            else if constexpr (d.upperVfShadowReg != 0u)
            {
                constexpr uint8_t sh = d.upperVfShadowReg;
                float oldVf[4];
                float upperVf[4];
                std::memcpy(oldVf, v.m_state.vf[sh], sizeof(oldVf));
                execUpperN<U, Imm, NoFlags>(v);
                std::memcpy(upperVf, v.m_state.vf[sh], sizeof(upperVf));
                std::memcpy(v.m_state.vf[sh], oldVf, sizeof(oldVf));
                execLowerN<U, L, Vu0>(v, c);
                std::memcpy(v.m_state.vf[sh], upperVf, sizeof(upperVf));
            }
            else
            {
                execUpperN<U, Imm, NoFlags>(v);
                execLowerN<U, L, Vu0>(v, c);
            }

            v.m_viBranchBackupValid = false;
            ++c.instrCount;
            if constexpr (Relax)
            {
                // Relaxed VU1: results take effect at once. The rare pipeline
                // ops (DIV/SQRT/RSQRT Q, EFU P, stores, XGKICK) are run to
                // completion before the next pair, and any queued flag update
                // is applied now. No VF/VI/ACC latency tracking.
                // An XGKICK issued by this pair finishes after the NEXT pair (so a
                // store in the following instruction still lands in the packet,
                // as with the streamed transfer); everything else settles now.
                // Relaxed = no hazard stalls only. Pipelines tick with the clock
                // (commit at the top of the pair) and FMAC flags keep their
                // 4-cycle delay: a lower FMAND in the same pair as an upper FMAC
                // reads the older MAC (7d7e 0x188), as PCSX2's microVU also models.
            }
            else
            {
                markWrites<U, L, Vu0>(v);
            }
            if constexpr (writtenVi != 0u && d.lowerUsage.delaysNextBranchRead)
                v.recordViWriteForBranch(writtenVi, oldVi);

            v.m_state.vf[0][0] = 0.0f;
            v.m_state.vf[0][1] = 0.0f;
            v.m_state.vf[0][2] = 0.0f;
            v.m_state.vf[0][3] = 1.0f;
            v.m_state.vi[0] = 0;

            if constexpr (Mid)
            {
                advanceOneCycleFast(v);
                return Vu1NativeStep::Continue;
            }

            uint32_t nextPc = v.m_state.pc + 8u;
            if (nextPc >= c.codeSize)
                nextPc = 0u;
            v.m_state.pc = nextPc;
            if (v.m_state.branchPending)
            {
                if (v.m_state.branchDelay == 0u)
                {
                    v.m_state.pc = v.m_state.branchTarget & (Vu0 ? 0x0FFFu : 0x3FFFu); // microAddressMask()
                    v.m_state.branchPending = false;
                }
                else
                {
                    --v.m_state.branchDelay;
                }
            }

            const bool dHalt = d.dBit && v.m_state.dBitEnabled;
            const bool tHalt = d.tBit && v.m_state.tBitEnabled;
            const bool haltBit = dHalt || tHalt;
            constexpr bool isBranch = d.lowerUsage.pipeline == I::PipelineBranch;
            const bool haltBranch = haltBit && isBranch;
            Vu1NativeStep result = Vu1NativeStep::Continue;
            if (v.m_state.haltAfterDelaySlot)
            {
                v.m_state.stoppedByD = v.m_pendingHaltD;
                v.m_state.stoppedByT = v.m_pendingHaltT;
                result = Vu1NativeStep::Ended;
            }
            else if (v.m_state.ebit)
            {
                result = Vu1NativeStep::Ended;
            }
            else if (haltBit && !haltBranch)
            {
                v.m_state.stoppedByD = dHalt;
                v.m_state.stoppedByT = tHalt;
                result = Vu1NativeStep::Ended;
            }
            else if (d.eBit)
            {
                v.m_state.ebit = true;
            }
            else if (haltBranch)
            {
                v.m_state.haltAfterDelaySlot = true;
                v.m_pendingHaltD = dHalt;
                v.m_pendingHaltT = tHalt;
            }
            advanceOneCycleFast(v);
            return result;
        }
    }
};

#endif
