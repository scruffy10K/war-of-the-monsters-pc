#include "runtime/ps2_vu1.h"
#include "ps2_vu1_detail.h"
#include "ps2_vu1_native.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <emmintrin.h>

using namespace vu1n;

namespace
{
    int32_t vuFloatToInt(float value, float scale)
    {
        const double scaled = static_cast<double>(value) * static_cast<double>(scale);
        if (scaled >= static_cast<double>(std::numeric_limits<int32_t>::max()))
            return std::numeric_limits<int32_t>::max();
        if (scaled <= static_cast<double>(std::numeric_limits<int32_t>::min()))
            return std::numeric_limits<int32_t>::min();
        return static_cast<int32_t>(scaled);
    }

    // Fast FMAC path (see execUpper). Every flag-producing FMAC op is one of
    // five forms over (vs, right[, acc]); `right` is a broadcast VT lane, Q, I,
    // or VT per lane. Mirrors the op lists in calculateFmacExactResult and
    // calculateFmacProductSticky (product-sum <=> Madd/Msub). OPMSUB/OPMULA and
    // everything that is not an FMAC flag op return None and take the old path.

    // PS2X_VU1_FMAC_SLOW=1 turns the fast path off (A/B timing, any build).
    const bool s_fmacSlow = std::getenv("PS2X_VU1_FMAC_SLOW") != nullptr;


#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    // PS2X_VU1_FMAC_CHECK=1: run the old per-lane path next to the fast one on
    // every FMAC op and report any difference in results or flags.
    const bool s_fmacCheck = std::getenv("PS2X_VU1_FMAC_CHECK") != nullptr;
    uint64_t s_fmacChecked = 0u;
    uint64_t s_fmacMismatched = 0u;
#endif
}

// ============================================================================
// Upper instructions (FMAC pipeline)
// ============================================================================
#if defined(_MSC_VER)
__declspec(safebuffers) // hot path: no /GS cookie (fixed-size local arrays only)
#endif
void VU1Interpreter::execUpper(uint32_t instr)
{
    m_currentUpperInstruction = instr;
    uint8_t dest = DEST(instr);
    uint8_t ft = FT(instr);
    uint8_t fs = FS(instr);
    uint8_t fd = FD(instr);
    uint8_t op = instr & 0x3F;

    const bool upperSpecial = op >= 0x3Cu;
    const uint8_t sel = upperSpecial ? static_cast<uint8_t>((instr & 0x3u) | ((instr >> 4) & 0x7Cu)) : op;
    if (upperSpecial && (sel == 0x2Fu || sel == 0x30u))
        return; // NOP: the generic path below has no side effects for it.

    // Fast FMAC path. Same float arithmetic, same exact (long double) result
    // per lane and same flags as applyFmacDest/applyFmacDestAcc, but computed
    // once per lane from operands normalized once, instead of re-decoding the
    // instruction and re-normalizing every operand per lane in
    // calculateFmacExactResult / calculateFmacProductSticky (~450 cycles per
    // op measured, the bulk of VU1 time in a level).
    if (const FmacForm form = fmacForm(sel); form.kind != FmacKind::None && !s_fmacSlow)
    {
        // Four lanes at once with SSE: the same float ops (packed mulps/addps
        // round exactly like the scalar ones under the VU round-to-zero MXCSR
        // run() sets, and MSVC does not contract them into FMA) and the same
        // double 'exact' results, so results and flags are bit-identical to the
        // per-lane code (checked by PS2X_VU1_FMAC_CHECK against the old path).
        const __m128i vsBits = vuLoadNormalized(m_state.vf[fs]);
        __m128i rBits;
        switch (form.right)
        {
        case FmacRight::Bc:
            rBits = vuBroadcastNormalized(m_state.vf[ft][sel & 3u]);
            break;
        case FmacRight::Q:
            rBits = vuBroadcastNormalized(m_state.q);
            break;
        case FmacRight::I:
            rBits = vuBroadcastNormalized(m_state.i);
            break;
        default:
            rBits = vuLoadNormalized(m_state.vf[ft]);
            break;
        }
        const bool productSum = form.kind == FmacKind::Madd || form.kind == FmacKind::Msub;
        const __m128 vs = _mm_castsi128_ps(vsBits);
        const __m128 r = _mm_castsi128_ps(rBits);
        const __m128 acc = productSum ? _mm_castsi128_ps(vuLoadNormalized(m_state.acc)) : _mm_setzero_ps();

        __m128 res;
        switch (form.kind)
        {
        case FmacKind::Add:
            res = _mm_add_ps(vs, r);
            break;
        case FmacKind::Sub:
            res = _mm_sub_ps(vs, r);
            break;
        case FmacKind::Madd:
            res = _mm_add_ps(acc, _mm_mul_ps(vs, r));
            break;
        case FmacKind::Msub:
            res = _mm_sub_ps(acc, _mm_mul_ps(vs, r));
            break;
        default:
            res = _mm_mul_ps(vs, r);
            break;
        }
        float result[4];
        _mm_storeu_ps(result, res);
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
        float rawResult[4];
        std::memcpy(rawResult, result, sizeof(rawResult));
#endif

        const __m128d sLo = _mm_cvtps_pd(vs);
        const __m128d sHi = _mm_cvtps_pd(_mm_movehl_ps(vs, vs));
        const __m128d rLo = _mm_cvtps_pd(r);
        const __m128d rHi = _mm_cvtps_pd(_mm_movehl_ps(r, r));
        const __m128d pLo = _mm_mul_pd(sLo, rLo);
        const __m128d pHi = _mm_mul_pd(sHi, rHi);
        __m128d eLo, eHi;
        switch (form.kind)
        {
        case FmacKind::Add:
            eLo = _mm_add_pd(sLo, rLo);
            eHi = _mm_add_pd(sHi, rHi);
            break;
        case FmacKind::Sub:
            eLo = _mm_sub_pd(sLo, rLo);
            eHi = _mm_sub_pd(sHi, rHi);
            break;
        case FmacKind::Madd:
            eLo = _mm_add_pd(_mm_cvtps_pd(acc), pLo);
            eHi = _mm_add_pd(_mm_cvtps_pd(_mm_movehl_ps(acc, acc)), pHi);
            break;
        case FmacKind::Msub:
            eLo = _mm_sub_pd(_mm_cvtps_pd(acc), pLo);
            eHi = _mm_sub_pd(_mm_cvtps_pd(_mm_movehl_ps(acc, acc)), pHi);
            break;
        default:
            eLo = pLo;
            eHi = pHi;
            break;
        }

        const uint32_t destLanes = kDestToLanes[dest & 0xFu];
        const VuLaneClass ec = vuClassify(eLo, eHi);
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
        if (productSum)
        {
            // Product-sum ops accumulate the product's own Z/S/U/O into the
            // sticky flags (calculateFmacProductSticky).
            const VuLaneClass pc = vuClassify(pLo, pHi);
            if ((pc.zero & destLanes) != 0u)
                extraSticky |= 0x1u;
            if ((pc.neg & destLanes) != 0u)
                extraSticky |= 0x2u;
            if ((pc.over & destLanes) != 0u)
                extraSticky |= 0x8u;
            if ((pc.under & destLanes) != 0u)
                extraSticky |= 0x5u;
        }

#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
        if (s_fmacCheck)
        {
            // Reference: the old path on the same, still unmodified, state.
            float ref[4];
            std::memcpy(ref, rawResult, sizeof(ref));
            uint8_t refFlags[4] = {};
            normalizeFmacResult(ref, dest, refFlags);
            const uint32_t refSticky = calculateFmacProductSticky(dest);
            bool same = refSticky == extraSticky;
            for (uint32_t c = 0; c < 4u && same; ++c)
            {
                if ((dest & (1u << (3u - c))) == 0u)
                    continue;
                same = std::memcmp(&ref[c], &result[c], sizeof(float)) == 0 && refFlags[c] == laneFlags[c];
            }
            ++s_fmacChecked;
            if (!same && ++s_fmacMismatched <= 20u)
                std::fprintf(stderr, "[vu1:fmaccheck] MISMATCH upper=%08x dest=%x sticky fast=%x ref=%x flags fast=%x,%x,%x,%x ref=%x,%x,%x,%x\n",
                             instr, dest, extraSticky, refSticky, laneFlags[0], laneFlags[1], laneFlags[2], laneFlags[3],
                             refFlags[0], refFlags[1], refFlags[2], refFlags[3]);
            if ((s_fmacChecked % 4000000u) == 0u)
                std::fprintf(stderr, "[vu1:fmaccheck] checked=%llu mismatched=%llu\n",
                             static_cast<unsigned long long>(s_fmacChecked),
                             static_cast<unsigned long long>(s_fmacMismatched));
        }
#endif

        updateFmacFlags(laneFlags, dest, extraSticky);
        if (upperSpecial)
            applyDestAcc(result, dest);
        else
            applyDest(m_state.vf[fd], result, dest);
        return;
    }

    float *vd = m_state.vf[fd];
    float normalizedVs[4] = {};
    float normalizedVt[4] = {};
    float normalizedAcc[4] = {};
    // Only normalize what the remaining (non fast-path) ops read: ITOF and CLIP
    // read raw VF bits, and OPMSUB is the only op left here that reads ACC
    // (unless PS2X_VU1_FMAC_SLOW sends every FMAC op down this path).
    const bool rawOperandsOnly = upperSpecial && ((sel >= 0x10u && sel <= 0x13u) || sel == 0x1Fu);
    if (!rawOperandsOnly)
    {
        for (uint32_t component = 0; component < 4u; ++component)
        {
            normalizedVs[component] = normalizeOperand(m_state.vf[fs][component]);
            normalizedVt[component] = normalizeOperand(m_state.vf[ft][component]);
        }
    }
    if (s_fmacSlow || (!upperSpecial && op == 0x2Eu))
        for (uint32_t component = 0; component < 4u; ++component)
            normalizedAcc[component] = normalizeOperand(m_state.acc[component]);
    const float *vs = normalizedVs;
    const float *vt = normalizedVt;
    const float *acc = normalizedAcc;
    const float q = normalizeOperand(m_state.q);
    const float i = normalizeOperand(m_state.i);
    float result[4];

    // Upper opcode decoding (bits 5:0 of upper word)
    switch (op)
    {
    case 0x00:
    case 0x01:
    case 0x02:
    case 0x03: // ADDbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + bc;
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x04:
    case 0x05:
    case 0x06:
    case 0x07: // SUBbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - bc;
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x08:
    case 0x09:
    case 0x0A:
    case 0x0B: // MADDbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] + vs[c] * bc;
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x0C:
    case 0x0D:
    case 0x0E:
    case 0x0F: // MSUBbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] - vs[c] * bc;
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x10:
    case 0x11:
    case 0x12:
    case 0x13: // MAXbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] > bc) ? vs[c] : bc;
        applyDest(vd, result, dest);
        return;
    }
    case 0x14:
    case 0x15:
    case 0x16:
    case 0x17: // MINIbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] < bc) ? vs[c] : bc;
        applyDest(vd, result, dest);
        return;
    }
    case 0x18:
    case 0x19:
    case 0x1A:
    case 0x1B: // MULbc
    {
        float bc = broadcast(vt, op & 3);
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * bc;
        applyFmacDest(vd, result, dest);
        return;
    }
    case 0x1C: // MULq
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * q;
        applyFmacDest(vd, result, dest);
        return;
    case 0x1D: // MAXi
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] > i) ? vs[c] : i;
        applyDest(vd, result, dest);
        return;
    case 0x1E: // MULi
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * i;
        applyFmacDest(vd, result, dest);
        return;
    case 0x1F: // MINIi
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] < i) ? vs[c] : i;
        applyDest(vd, result, dest);
        return;
    case 0x20: // ADDq
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + q;
        applyFmacDest(vd, result, dest);
        return;
    case 0x21: // MADDq
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] + vs[c] * q;
        applyFmacDest(vd, result, dest);
        return;
    case 0x22: // ADDi
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + i;
        applyFmacDest(vd, result, dest);
        return;
    case 0x23: // MADDi
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] + vs[c] * i;
        applyFmacDest(vd, result, dest);
        return;
    case 0x24: // SUBq
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - q;
        applyFmacDest(vd, result, dest);
        return;
    case 0x25: // MSUBq
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] - vs[c] * q;
        applyFmacDest(vd, result, dest);
        return;
    case 0x26: // SUBi
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - i;
        applyFmacDest(vd, result, dest);
        return;
    case 0x27: // MSUBi
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] - vs[c] * i;
        applyFmacDest(vd, result, dest);
        return;
    case 0x28: // ADD
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] + vt[c];
        applyFmacDest(vd, result, dest);
        return;
    case 0x29: // MADD
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] + vs[c] * vt[c];
        applyFmacDest(vd, result, dest);
        return;
    case 0x2A: // MUL
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] * vt[c];
        applyFmacDest(vd, result, dest);
        return;
    case 0x2B: // MAX
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] > vt[c]) ? vs[c] : vt[c];
        applyDest(vd, result, dest);
        return;
    case 0x2C: // SUB
        for (int c = 0; c < 4; c++)
            result[c] = vs[c] - vt[c];
        applyFmacDest(vd, result, dest);
        return;
    case 0x2D: // MSUB
        for (int c = 0; c < 4; c++)
            result[c] = acc[c] - vs[c] * vt[c];
        applyFmacDest(vd, result, dest);
        return;
    case 0x2E: // OPMSUB
        result[0] = acc[0] - vs[1] * vt[2];
        result[1] = acc[1] - vs[2] * vt[0];
        result[2] = acc[2] - vs[0] * vt[1];
        result[3] = 0.0f;
        applyFmacDest(vd, result, dest);
        return;
    case 0x2F: // MINI
        for (int c = 0; c < 4; c++)
            result[c] = (vs[c] < vt[c]) ? vs[c] : vt[c];
        applyDest(vd, result, dest);
        return;

    // Upper special group (low op 0x3C..0x3F).
    // Like lower1 special, the real selector is not just bits 5:0.  Dobie decodes:
    //   op = (instr & 0x3) | ((instr >> 4) & 0x7C)
    // Several instructions in this group also use FT as the destination, not FD.
    case 0x3C:
    case 0x3D:
    case 0x3E:
    case 0x3F:
    {
        const uint8_t specialOp = static_cast<uint8_t>((instr & 0x3u) | ((instr >> 4) & 0x7Cu));
        float *vtDest = m_state.vf[ft];

        switch (specialOp)
        {
        case 0x00:
        case 0x01:
        case 0x02:
        case 0x03: // ADDAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + bc;
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x04:
        case 0x05:
        case 0x06:
        case 0x07: // SUBAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - bc;
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x08:
        case 0x09:
        case 0x0A:
        case 0x0B: // MADDAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] + vs[c] * bc;
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x0C:
        case 0x0D:
        case 0x0E:
        case 0x0F: // MSUBAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] - vs[c] * bc;
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x10: // ITOF0
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x11: // ITOF4
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv) / 16.0f;
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x12: // ITOF12
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv) / 4096.0f;
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x13: // ITOF15
            for (int c = 0; c < 4; c++)
            {
                int32_t iv;
                std::memcpy(&iv, &m_state.vf[fs][c], 4);
                result[c] = static_cast<float>(iv) / 32768.0f;
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x14: // FTOI0
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 1.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x15: // FTOI4
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 16.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x16: // FTOI12
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 4096.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x17: // FTOI15
            for (int c = 0; c < 4; c++)
            {
                int32_t iv = vuFloatToInt(vs[c], 32768.0f);
                std::memcpy(&result[c], &iv, 4);
            }
            applyDest(vtDest, result, dest);
            return;
        case 0x18:
        case 0x19:
        case 0x1A:
        case 0x1B: // MULAbc
        {
            float bc = broadcast(vt, specialOp & 3);
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * bc;
            applyFmacDestAcc(result, dest);
            return;
        }
        case 0x1C: // MULAq
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * q;
            applyFmacDestAcc(result, dest);
            return;
        case 0x1D: // ABS
            for (int c = 0; c < 4; c++)
                result[c] = std::fabs(vs[c]);
            applyDest(vtDest, result, dest);
            return;
        case 0x1E: // MULAi
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * i;
            applyFmacDestAcc(result, dest);
            return;
        case 0x1F: // CLIP
        {
            uint32_t wBits = 0u;
            std::memcpy(&wBits, &m_state.vf[ft][3], sizeof(wBits));
            const int32_t limit = (wBits & 0x7F800000u) != 0u ? static_cast<int32_t>(wBits & 0x7FFFFFFFu) : 0x007FFFFF;

            const auto exceedsClipPlane = [limit](float value, uint32_t signMask)
            {
                uint32_t bits = 0u;
                std::memcpy(&bits, &value, sizeof(bits));
                bits ^= signMask;
                int32_t orderedBits = 0;
                std::memcpy(&orderedBits, &bits, sizeof(orderedBits));
                return orderedBits > limit;
            };

            uint32_t flags = 0u;
            if (exceedsClipPlane(m_state.vf[fs][0], 0x00000000u))
                flags |= 0x01u;
            if (exceedsClipPlane(m_state.vf[fs][0], 0x80000000u))
                flags |= 0x02u;
            if (exceedsClipPlane(m_state.vf[fs][1], 0x00000000u))
                flags |= 0x04u;
            if (exceedsClipPlane(m_state.vf[fs][1], 0x80000000u))
                flags |= 0x08u;
            if (exceedsClipPlane(m_state.vf[fs][2], 0x00000000u))
                flags |= 0x10u;
            if (exceedsClipPlane(m_state.vf[fs][2], 0x80000000u))
                flags |= 0x20u;
            queueClip(flags);
            return;
        }
        case 0x20: // ADDAq
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + q;
            applyFmacDestAcc(result, dest);
            return;
        case 0x21: // MADDAq
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] + vs[c] * q;
            applyFmacDestAcc(result, dest);
            return;
        case 0x22: // ADDAi
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + i;
            applyFmacDestAcc(result, dest);
            return;
        case 0x23: // MADDAi
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] + vs[c] * i;
            applyFmacDestAcc(result, dest);
            return;
        case 0x24: // SUBAq
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - q;
            applyFmacDestAcc(result, dest);
            return;
        case 0x25: // MSUBAq
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] - vs[c] * q;
            applyFmacDestAcc(result, dest);
            return;
        case 0x26: // SUBAi
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - i;
            applyFmacDestAcc(result, dest);
            return;
        case 0x27: // MSUBAi
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] - vs[c] * i;
            applyFmacDestAcc(result, dest);
            return;
        case 0x28: // ADDA
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] + vt[c];
            applyFmacDestAcc(result, dest);
            return;
        case 0x29: // MADDA
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] + vs[c] * vt[c];
            applyFmacDestAcc(result, dest);
            return;
        case 0x2A: // MULA
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] * vt[c];
            applyFmacDestAcc(result, dest);
            return;
        case 0x2C: // SUBA
            for (int c = 0; c < 4; c++)
                result[c] = vs[c] - vt[c];
            applyFmacDestAcc(result, dest);
            return;
        case 0x2D: // MSUBA
            for (int c = 0; c < 4; c++)
                result[c] = acc[c] - vs[c] * vt[c];
            applyFmacDestAcc(result, dest);
            return;
        case 0x2E: // OPMULA
            result[0] = vs[1] * vt[2];
            result[1] = vs[2] * vt[0];
            result[2] = vs[0] * vt[1];
            result[3] = 0.0f;
            applyFmacDestAcc(result, dest);
            return;
        case 0x2F:
        case 0x30: // NOP
            return;
        default:
            reportReservedInstruction(true, instr);
            return;
        }
    }

    case 0x30:
    case 0x31:
    case 0x32:
    case 0x33:
    default:
        reportReservedInstruction(true, instr);
        return;
    }
}
