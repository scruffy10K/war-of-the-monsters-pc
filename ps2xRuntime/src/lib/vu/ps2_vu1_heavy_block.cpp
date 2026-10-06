// Compact control flow for the shared 7d7e/d166 ff0/1ea0 workload.
// Step-pair words and timing flags are copied unchanged from the native reference.
#include "ps2_vu1_native.h"
#include <fused_parameters.inc>
#include <cfenv>
#include <cstdlib>
#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>
extern Vu1NativeStep (*g_ps2xVu1ShaderTransform)(VU1Interpreter &,Vu1NativeCtx &);
extern bool (*g_ps2xVu1MenuVertex)(VU1Interpreter &,Vu1NativeCtx &);
extern bool (*g_ps2xVu1MenuLoop)(VU1Interpreter &,Vu1NativeCtx &);
extern Vu1NativeStep (*g_ps2xVu1GameMatrix[3])(VU1Interpreter &,Vu1NativeCtx &);
extern Vu1NativeStep (*g_ps2xVu1ConvertPair[2])(VU1Interpreter &, Vu1NativeCtx &);
#include <heavyBlockImpl.inc>

// Compact menu shader paths. In 1a48..1ae0, MAC/STATUS writes are dead:
// only CLIP is read, and 1ae8/1af0 or 1b40 replace MAC before its next read.
// The analogous cc0..d88 region is overwritten by d90/d98 or df0.
// Keep CLIP, arithmetic, stores and cycle timing; native continuation tests
// cover stops inside the region.

// ---------------------------------------------------------------------------
// High-level kernel: shader 0x1028 per-vertex loop (0x10f8..0x1200, 34 pairs).
// The instruction sequence is generated from fightShaderBlock's pair words by
// gen_hle1028.py (ps2_vu1_hle1028.inc). Same float operations in the same
// order, same operand normalisation, same Q/clip/store/flag timing; what goes
// away is the per-pair bookkeeping and the register reloads forced by stores
// into VU memory. Declines (returns false, nothing modified) on a vertex that
// takes the clip path, a lane needing the exact double pass, a store/load
// alias, or any pipeline state it does not model. PS2X_VU1_HLE=0 disables;
// PS2X_VU1_HLE_VERIFY=1 re-runs sampled iterations natively and compares.
#include "ps2_vu1_alias_kernels.inc"
namespace {
const bool s_fixedAliases=[] { const char *p=std::getenv("PS2X_VU1_FIXED_ALIASES"); return !p || p[0]!='0'; }();
const bool s_verifyAliases=[] { const char *p=std::getenv("PS2X_VU1_ALIAS_VERIFY"); return p && p[0]=='1'; }();
bool hleReferenceAlias(const uint32_t *storeAddr, const uint32_t *storeIdx, unsigned nStore,
                       const uint32_t *loadAddr, const uint32_t *loadIdx, unsigned nLoad)
{
    for (unsigned st=0; st<nStore; ++st)
        for (unsigned ld=0; ld<nLoad; ++ld)
            if (loadIdx[ld]>storeIdx[st] && loadAddr[ld]==storeAddr[st]) return true;
    return false;
}
inline bool hleCheckedAlias(bool fixed, const uint32_t *storeAddr, const uint32_t *storeIdx, unsigned nStore,
                            const uint32_t *loadAddr, const uint32_t *loadIdx, unsigned nLoad)
{
    if (!s_fixedAliases || s_verifyAliases) {
        const bool reference=hleReferenceAlias(storeAddr,storeIdx,nStore,loadAddr,loadIdx,nLoad);
        if (s_verifyAliases && fixed!=reference) {
            std::fprintf(stderr,"[vu1:alias] MISMATCH fixed=%u reference=%u\n",fixed,reference);
            std::abort();
        }
        if (s_verifyAliases) {
            static thread_local uint64_t checked=0;
            if (++checked==1 || checked%1000000u==0)
                std::fprintf(stderr,"[vu1:alias] checked=%llu mismatches=0\n",checked);
        }
        return reference;
    }
    return fixed;
}
}

namespace hle1028
{
inline __m128 hn(__m128 x) { return _mm_castsi128_ps(vu1n::vuNormalizeBits(_mm_castps_si128(x))); }
inline float hlane(__m128 x, int c) { float t[4]; _mm_storeu_ps(t, x); return t[c]; }
inline __m128 hbs(float f) { return _mm_castsi128_ps(vu1n::vuBroadcastNormalized(f)); }
inline __m128 hbc(__m128 x, int c) { return hbs(hlane(x, c)); }
inline uint32_t hlqaddr(int32_t vi, int32_t imm) { return (static_cast<uint32_t>(static_cast<int32_t>(vi + imm)) * 16u) & 16383u; }
// Vu1NativeAccess::floatToInt(normOp(lane), scale) on four lanes: exact
// double product, clamp to the int32 range, truncate.
const bool s_fastFtoi=[] {
    const char *p=std::getenv("PS2X_VU1_FLOAT_FTOI"); return !p || p[0]!='0';
}();
inline __m128 hftoi(__m128 x, float scale)
{
    if (s_fastFtoi && (scale == 1.0f || scale == 16.0f)) {
        // Power-of-two scaling is exact throughout the int32 range. CVTT
        // already returns INT_MIN for negative overflow; select INT_MAX for
        // positive overflow, including float overflow to infinity. Normalize
        // VU exponent-zero/255 inputs before using host floating-point math.
        const __m128 scaled = _mm_mul_ps(hn(x), _mm_set1_ps(scale));
        const __m128 overflow = _mm_cmpge_ps(scaled, _mm_set1_ps(2147483648.0f));
        return _mm_castsi128_ps(_mm_blendv_epi8(_mm_cvttps_epi32(scaled),
            _mm_set1_epi32(std::numeric_limits<int32_t>::max()), _mm_castps_si128(overflow)));
    }
    const __m256d d = _mm256_mul_pd(_mm256_cvtps_pd(hn(x)), _mm256_set1_pd(static_cast<double>(scale)));
    const __m256d hiD = _mm256_cmp_pd(d, _mm256_set1_pd(static_cast<double>(std::numeric_limits<int32_t>::max())), _CMP_GE_OQ);
    const __m256d loD = _mm256_cmp_pd(d, _mm256_set1_pd(static_cast<double>(std::numeric_limits<int32_t>::min())), _CMP_LE_OQ);
    const __m256i pick = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    const __m128i hi = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(hiD), pick));
    const __m128i lo = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(loD), pick));
    __m128i iv = _mm256_cvttpd_epi32(d);
    iv = _mm_blendv_epi8(iv, _mm_set1_epi32(std::numeric_limits<int32_t>::max()), hi);
    iv = _mm_blendv_epi8(iv, _mm_set1_epi32(std::numeric_limits<int32_t>::min()), lo);
    return _mm_castsi128_ps(iv);
}
// VU1Interpreter::normalizeResult without the flags.
inline float hnormres(float value)
{
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t exponent = (bits >> 23) & 0xFFu;
    if ((bits & 0x7FFFFFFFu) != 0u)
    {
        if (exponent == 0u)
            bits &= 0x80000000u;
        else if (exponent == 0xFFu)
            bits = (bits & 0x80000000u) | 0x7F7FFFFFu;
    }
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
// Reverse SSE lane order (xyzw bits 0..3) into VU MAC lane order.
inline uint32_t hreverse4(uint32_t mask)
{
    return ((mask & 1u) << 3) | ((mask & 2u) << 1) |
           ((mask & 4u) >> 1) | ((mask & 8u) >> 3);
}
inline uint32_t hclip(__m128 vs, __m128 vt)
{
    const uint32_t w = uint32_t(_mm_extract_epi32(_mm_castps_si128(vt), 3));
    const __m128i limit = _mm_set1_epi32((w & 0x7f800000u) ? int32_t(w & 0x7fffffffu) : 0x007fffff);
    const __m128i bits = _mm_castps_si128(vs);
    const uint32_t positive = uint32_t(_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(bits, limit))));
    const uint32_t negative = uint32_t(_mm_movemask_ps(_mm_castsi128_ps(
        _mm_cmpgt_epi32(_mm_xor_si128(bits, _mm_set1_epi32(int32_t(0x80000000u))), limit))));
    // Integer comparisons preserve VU treatment of denormals and exponent255.
    return (positive & 1u) | ((negative & 1u) << 1) |
           ((positive & 2u) << 1) | ((negative & 2u) << 2) |
           ((positive & 4u) << 2) | ((negative & 4u) << 3);
}
// updateFmacFlagsC's mac/status packing for an ordinary-magnitude result.
// All kernel call sites have constant destination masks. Inlining lets the
// compiler fold lane packing and avoids an out-of-line call per VU operation.
__forceinline void hflags(uint32_t destLanes, uint32_t negMask, uint32_t resZero, uint32_t sticky, uint32_t &mac, uint32_t &st)
{
    const uint32_t zero = resZero & destLanes;
    const uint32_t negative = negMask & destLanes;
    mac = hreverse4(zero) | (hreverse4(negative) << 4);
    const uint32_t cur = (zero != 0u ? 1u : 0u) | (negative != 0u ? 2u : 0u);
    st = (st & 0xFF0u) | cur | ((cur | sticky) << 6);
}
}

namespace {
const bool s_kernelProfile = std::getenv("PS2X_VU1_KERNEL_PROFILE") != nullptr;
const bool s_nativeBoundary=[] {const char *p=std::getenv("PS2X_VU1_NATIVE_BOUNDARY");return !p || p[0]!='0';}();
struct KernelProfile {
    uint64_t guards[512]{}, bodyFailures=0, aliasFailures=0, chunks=0, vertices=0;
    ~KernelProfile() {dump();}
    void dump() const {
        if (!s_kernelProfile) return;
        std::fprintf(stderr,"[vu1:kernel-profile] kernel=%s chunks=%llu vertices=%llu body=%llu alias=%llu\n",
            name,chunks,vertices,bodyFailures,aliasFailures);
        for(unsigned i=0;i<512;++i) if(guards[i])
            std::fprintf(stderr,"[vu1:kernel-guard] kernel=%s mask=%x calls=%llu\n",name,i,guards[i]);
    }
    const char *name;
};
KernelProfile s_kernel1028{{},0,0,0,0,"1028"},s_kernel40{{},0,0,0,0,"40"};
}

bool Vu1NativeAccess::hleFight1028(I &v, Vu1NativeCtx &c)
{
    using namespace hle1028;
    if (v.m_cycle >= v.m_pipelineNextReady)
        v.commitReadyPipelines();
    static const bool pendingFlags = [] {
        const char *p = std::getenv("PS2X_VU1_HLE_PENDING_FLAGS");
        return !p || p[0] != '0';
    }();
    // The first old-flag observer is FCAND at pair22. DIV retires at pair8;
    // its DI update commutes with MAC/status/clip retirement. FSSET commutes
    // only for DI=0, so zero-denominator sticky cases keep the reference path.
    // Preview in locals: a declined first vertex must leave the FIFO intact.
    unsigned incomingFlags = 0;
    uint32_t incomingMac = v.m_state.mac, incomingStatus = v.m_state.status;
    uint32_t incomingClip = v.m_state.clip;
    if (pendingFlags && v.m_flagOrderCount && v.m_flagOrderCount <= I::kMaxFlagEntries - 3u &&
        !(v.m_storeMask | v.m_vfWriteMask | v.m_viWriteMask | v.m_accWriteMask) &&
        v.m_pipelineNextReady == ~0ull) {
        uint32_t seen = 0;
        bool valid = true, sticky = false;
        for (unsigned n = 0; n < v.m_flagOrderCount; ++n) {
            const unsigned slot = v.m_flagOrder[(v.m_flagOrderHead + n) % I::kMaxFlagEntries];
            if (slot >= I::kMaxFlagEntries || (seen & (1u << slot))) { valid = false; break; }
            const auto &entry = v.m_flagPipeline[slot];
            if (!entry.valid || entry.readyCycle > v.m_cycle + 7u) { valid = false; break; }
            seen |= 1u << slot;
            if (entry.writesMac) incomingMac = entry.mac;
            if (entry.writesStatus) {
                const uint32_t current = entry.status & 0xFu;
                incomingStatus = (incomingStatus & 0xFF0u) | current | ((current | entry.extraSticky) << 6);
            }
            if (entry.writesSticky) {
                sticky = true;
                incomingStatus = (incomingStatus & 0x03Fu) | (entry.status & 0xFC0u);
            }
            if (entry.writesClip) incomingClip = entry.clip;
        }
        if (seen != v.m_flagMask) valid = false;
        if (sticky && normOp(v.m_state.vf[25][3]) == 0.0f) valid = false;
        if (valid) incomingFlags = v.m_flagOrderCount;

    }
    if ((v.m_storeMask | v.m_vfWriteMask | v.m_viWriteMask | v.m_accWriteMask) != 0u ||
        ((v.m_flagMask != 0u || v.m_flagOrderCount != 0u) && !incomingFlags) ||
        v.m_fdiv.valid || v.m_efu[0].valid || v.m_efu[1].valid || v.m_xgkick.active || v.m_state.branchPending ||
        v.m_state.haltAfterDelaySlot || v.m_state.ebit || v.m_stopRequested || c.dataSize != 16384u ||
        v.m_cycle + 34u > c.budgetEnd) {
        if(s_kernelProfile) {
            const unsigned why = (v.m_storeMask ? 1u : 0u) |
                ((v.m_vfWriteMask | v.m_viWriteMask | v.m_accWriteMask) ? 2u : 0u) |
                ((v.m_flagMask | v.m_flagOrderCount) ? 4u : 0u) |
                (v.m_fdiv.valid ? 8u : 0u) | ((v.m_efu[0].valid || v.m_efu[1].valid) ? 16u : 0u) |
                (v.m_xgkick.active ? 32u : 0u) | (v.m_state.branchPending ? 64u : 0u) |
                ((v.m_state.haltAfterDelaySlot || v.m_state.ebit || v.m_stopRequested || c.dataSize!=16384u) ? 128u : 0u) |
                (v.m_cycle + 34u > c.budgetEnd ? 256u : 0u);
            if(++s_kernel1028.guards[why]%100000u==0u) s_kernel1028.dump();
        }
        return false;
    }
    const uint64_t c0 = v.m_cycle;
    static const bool meshKernel = [] { const char *p = std::getenv("PS2X_VU1_MESH_KERNEL"); return !p || p[0] != '0'; }();
    static const bool valueMath = [] { const char *p = std::getenv("PS2X_VU1_VALUE_MATH"); return p && p[0] == '1'; }();
    // run() selects toward-zero rounding. Direct diagnostic callers can use
    // other modes, whose underflow boundary must retain the original checks.
    const bool valueMathAllowed=valueMath && std::fegetround()==FE_TOWARDZERO;
    const unsigned maxIterations = meshKernel ? 64u : 1u;
    // Keep the live vector registers local across vertices. Save only registers
    // written by the verified 10f8..1200 kernel before trying a later vertex;
    // clipping then returns the completed prefix, without redoing its geometry.
    unsigned iterations = 0;
    bool loopTaken = false, branched = false;
    uint8_t *const mem = c.vuData;
    __m128 R[32];
    for (int i = 0; i < 32; ++i)
        R[i] = _mm_loadu_ps(v.m_state.vf[i]);
    __m128 ACC = _mm_loadu_ps(v.m_state.acc);
    float Q = v.m_state.q, IR = v.m_state.i, qPending = 0.0f;
    uint32_t diPending = 0u;
    int32_t VI[16];
    std::memcpy(VI, v.m_state.vi, sizeof(VI));
    uint32_t MAC = incomingFlags ? incomingMac : v.m_state.mac;
    uint32_t ST = incomingFlags ? incomingStatus : v.m_state.status;
    uint32_t CLIPR = incomingFlags ? incomingClip : v.m_state.clip, WCLIP = v.m_workingClip;
    int32_t oldBranchVi = v.m_viBranchBackupValue;
    uint8_t branchViReg = v.m_viBranchBackupReg;
    do {
        uint32_t loadAddr[16], loadIdx[16];
        unsigned nLoad = 0u;
        uint32_t storeAddr[4], storeLanes[4], storeIdx[4];
        float storeWords[4][4];
        unsigned nStore = 0u;
        constexpr unsigned dirty[] = {5,6,8,9,10,11,12,16,20,23,24,25,28,29,30};
        __m128 savedR[15], savedAcc;
        int32_t savedVi[16], savedOldBranch = oldBranchVi;
        uint8_t savedBranchReg = branchViReg;
        const float savedQ = Q, savedI = IR;
        const uint32_t savedMac = MAC, savedStatus = ST, savedClip = CLIPR, savedWorkingClip = WCLIP;
        if (iterations) {
            for (unsigned i = 0; i < 15; ++i) savedR[i] = R[dirty[i]];
            savedAcc = ACC;
            std::memcpy(savedVi, VI, sizeof(VI));
        }
        const int result = [&]() -> int {
            if(valueMathAllowed) {
#include <ps2_vu1_hle1028_values.inc>
                return loopTaken ? 2 : 1;
            }
#include <ps2_vu1_hle1028.inc>
            return loopTaken ? 2 : 1;
        }();
        if(s_kernelProfile && result==0) ++s_kernel1028.bodyFailures;
        bool failed = result == 0;
        if (!failed) failed=hleCheckedAlias(wotmVuAlias::shader1028(storeAddr[0],loadAddr[5]),
            storeAddr,storeIdx,nStore,loadAddr,loadIdx,nLoad);
        if(s_kernelProfile && failed && result!=0) ++s_kernel1028.aliasFailures;
        if (failed) {
            if (!iterations) return false;
            for (unsigned i = 0; i < 15; ++i) R[dirty[i]] = savedR[i];
            ACC = savedAcc;
            std::memcpy(VI, savedVi, sizeof(VI));
            oldBranchVi = savedOldBranch; branchViReg = savedBranchReg;
            Q = savedQ; IR = savedI; MAC = savedMac; ST = savedStatus;
            CLIPR = savedClip; WCLIP = savedWorkingClip;
            break;
        }
        for (unsigned st = 0; st < nStore; ++st) {
            if (storeLanes[st] == 0xFu) std::memcpy(mem + storeAddr[st], storeWords[st], 16u);
            else for (uint32_t comp = 0; comp < 4u; ++comp)
                if (storeLanes[st] & (1u << (3u - comp)))
                    std::memcpy(mem + storeAddr[st] + comp * 4u, &storeWords[st][comp], 4u);
        }
        ++iterations;
        loopTaken = result == 2;
        branched |= loopTaken;
    } while (loopTaken && iterations < maxIterations && c0 + (iterations + 1u) * 34u <= c.budgetEnd);

    if(s_kernelProfile) {++s_kernel1028.chunks;s_kernel1028.vertices+=iterations;}
    // Only these registers are written by this native loop.
    constexpr unsigned written[] = {5,6,8,9,10,11,12,16,20,23,24,25,28,29,30};
    for (unsigned i : written)
        _mm_storeu_ps(v.m_state.vf[i], R[i]);
    _mm_storeu_ps(v.m_state.acc, ACC);
    v.m_state.q = Q;
    v.m_state.i = IR;
    VI[0] = 0;
    std::memcpy(v.m_state.vi, VI, sizeof(VI));
    v.m_state.mac = MAC;
    v.m_state.status = ST;
    v.m_state.clip = CLIPR;
    v.m_workingClip = WCLIP;
    v.m_cycle = c0 + 34u * iterations;
    v.m_state.cycles = v.m_cycle;
    if (incomingFlags) {
        for (uint32_t live = v.m_flagMask; live; live &= live - 1u)
            v.m_flagPipeline[std::countr_zero(live)].valid = false;
        v.m_flagMask = 0;
        v.m_flagOrderCount = 0;
        if (s_kernelProfile) {
            static uint64_t accepted = 0, vertices = 0;
            vertices += iterations;
            if ((++accepted % 100000u) == 1u)
                std::fprintf(stderr, "[vu1:pending-flags] chunks=%llu vertices=%llu\n",
                    (unsigned long long)accepted, (unsigned long long)vertices);
        }
    }
    v.m_flagOrderHead = (v.m_flagOrderHead + incomingFlags + 3u * iterations) % I::kMaxFlagEntries;
    v.m_flagTailReady = v.m_cycle - 1u + I::kFmacLatency;
    v.m_currentUpperInstruction = locallyGeneratedVu::hle1028LastUpper;
    v.m_viBranchBackupValue = oldBranchVi;
    v.m_viBranchBackupReg = branchViReg;
    v.m_viBranchBackupValid = false;
    v.m_state.branchPending = false;
    if (branched) {
        v.m_state.branchTarget = 0x10f8u;
        v.m_state.branchDelay = 0u;
    }
    if (loopTaken)
    {
        v.m_state.pc = 0x10f8u;
        v.m_state.branchTarget = 0x10f8u;
        v.m_state.branchDelay = 0u;
    }
    else
    {
        v.m_state.pc = 0x1208u;
    }
    c.instrCount += 34u * iterations;
    return true;
}

bool Vu1NativeAccess::hleSameState(const I &a, const I &b, const char **what)
{
    const VU1State &x = a.m_state, &y = b.m_state;
    auto bad = [what](const char *w) { *what = w; return false; };
    if (std::memcmp(x.vf, y.vf, sizeof(x.vf)) != 0) return bad("vf");
    if (std::memcmp(x.vi, y.vi, sizeof(x.vi)) != 0) return bad("vi");
    if (std::memcmp(x.acc, y.acc, sizeof(x.acc)) != 0) return bad("acc");
    if (std::memcmp(&x.q, &y.q, 4) || std::memcmp(&x.p, &y.p, 4) || std::memcmp(&x.i, &y.i, 4)) return bad("q/p/i");
    if (x.pc != y.pc) return bad("pc");
    if (x.mac != y.mac) return bad("mac");
    if (x.clip != y.clip) return bad("clip");
    if (x.status != y.status) return bad("status");
    if (x.cycles != y.cycles || a.m_cycle != b.m_cycle) return bad("cycle");
    if (x.branchPending != y.branchPending || x.branchTarget != y.branchTarget || x.branchDelay != y.branchDelay) return bad("branch");
    if (a.m_workingClip != b.m_workingClip) return bad("workingClip");
    if (a.m_flagOrderCount != b.m_flagOrderCount || a.m_flagMask != b.m_flagMask || a.m_flagOrderHead != b.m_flagOrderHead) return bad("flagfifo");
    if (a.m_flagTailReady != b.m_flagTailReady) return bad("flagTail");
    if (a.m_storeMask != b.m_storeMask || a.m_pipelineNextReady != b.m_pipelineNextReady)
    {
        static thread_local char buf[160];
        std::snprintf(buf, sizeof(buf), "storeMask/nextReady hle=%x/%llu native=%x/%llu cycle=%llu", a.m_storeMask,
                      (unsigned long long)a.m_pipelineNextReady, b.m_storeMask, (unsigned long long)b.m_pipelineNextReady,
                      (unsigned long long)a.m_cycle);
        return bad(buf);
    }
    for (uint32_t live = a.m_storeMask; live != 0u; live &= live - 1u)
    {
        const auto &p = a.m_storePipeline[std::countr_zero(live)];
        const auto &q = b.m_storePipeline[std::countr_zero(live)];
        if (p.readyCycle != q.readyCycle || p.address != q.address || p.words != q.words || p.laneMask != q.laneMask) return bad("store");
    }
    if (a.m_fdiv.valid != b.m_fdiv.valid) return bad("fdiv");
    if (a.m_currentUpperInstruction != b.m_currentUpperInstruction) return bad("upperInstr");
    if (a.m_viBranchBackupValid != b.m_viBranchBackupValid || a.m_viBranchBackupReg != b.m_viBranchBackupReg ||
        a.m_viBranchBackupValue != b.m_viBranchBackupValue) return bad("branchBackup");
    return true;
}

static Vu1NativeStep fightShaderBlock(VU1Interpreter &v, Vu1NativeCtx &c, Vu1NativeProgram nativeProgram);
static thread_local bool t_hleVerifyNative = false;
// Runtime switch for the high-level kernels (PS2X_VU1_HLE_ALTERNATE flips it
// every perf window for an in-process A/B).
std::atomic<bool> g_ps2xHleOn{true};

// true: a native vertex chunk ran (pc is 0x10f8 or 0x1208).
static bool hleFight1028Try(VU1Interpreter &v, Vu1NativeCtx &c, Vu1NativeProgram nativeProgram)
{
    static const bool enabled = [] { const char *p = std::getenv("PS2X_VU1_HLE"); return !p || p[0] != '0'; }();
    static const bool verify = [] { const char *p = std::getenv("PS2X_VU1_HLE_VERIFY"); return p && p[0] == '1'; }();
    if (!enabled || t_hleVerifyNative || !g_ps2xHleOn.load(std::memory_order_relaxed))
        return false;
    if (!verify)
        return Vu1NativeAccess::hleFight1028(v, c);
    static uint64_t s_calls = 0u, s_checked = 0u, s_bad = 0u, s_declined = 0u;
    if ((++s_calls % (std::getenv("PS2X_VU1_HLE_VERIFY_ALL") ? 1u : 32u)) != 0u)
    {
        const bool ok = Vu1NativeAccess::hleFight1028(v, c);
        s_declined += ok ? 0u : 1u;
        return ok;
    }
    auto before = std::make_unique<VU1Interpreter>(v);
    std::vector<uint8_t> memBefore(c.vuData, c.vuData + c.dataSize);
    const uint64_t icBefore = c.instrCount;
    if (!Vu1NativeAccess::hleFight1028(v, c))
    {
        ++s_declined;
        return false;
    }
    auto hle = std::make_unique<VU1Interpreter>(v);
    std::vector<uint8_t> memHle(c.vuData, c.vuData + c.dataSize);
    const uint64_t icHle = c.instrCount;
    v = *before;
    // M10f8 can be reached inside a Mid run whose stored PC is still10a8.
    // Start the reference at the actual kernel label, not that publication.
    // No compared output or pending state is normalized or excluded.
    Vu1NativeAccess::setPc(v, 0x10f8u);
    std::memcpy(c.vuData, memBefore.data(), c.dataSize);
    Vu1NativeCtx cn = c;
    cn.instrCount = icBefore;
    cn.budgetEnd = std::min<uint64_t>(c.budgetEnd, Vu1NativeAccess::cycle(*hle));
    t_hleVerifyNative = true;
    (void)fightShaderBlock(v, cn, nativeProgram);
    t_hleVerifyNative = false;
    c.instrCount = cn.instrCount;
    ++s_checked;
    const char *what = "";
    bool same = Vu1NativeAccess::hleSameState(*hle, v, &what);
    if (same && std::memcmp(memHle.data(), c.vuData, c.dataSize) != 0) { same = false; what = "memory"; }
    if (same && icHle != cn.instrCount) { same = false; what = "instrCount"; }
    if (!same && ++s_bad <= 20u)
        std::fprintf(stderr, "[vu1:hle1028] MISMATCH #%llu field=%s cycle=%llu pc hle=0x%x native=0x%x\n",
                     (unsigned long long)s_bad, what, (unsigned long long)Vu1NativeAccess::cycle(v), hle->state().pc, v.state().pc);
    if (s_checked == 1u || (s_checked % 2000u) == 0u)
        std::fprintf(stderr, "[vu1:hle1028] checked=%llu mismatched=%llu declined=%llu calls=%llu\n",
                     (unsigned long long)s_checked, (unsigned long long)s_bad, (unsigned long long)s_declined, (unsigned long long)s_calls);
    return true; // native result kept
}


// ---------------------------------------------------------------------------
// High-level kernel: shader 0x40 per-vertex loop (0x128..0x200, 28 pairs, 27
// when the IBEQ at 0x198 skips 0x1a8). Body generated by gen_hle_loop.py
// (ps2_vu1_hle40.inc) from the pair words and flag metadata of the generated
// native program. Unlike 0x1028 this loop reads MAC (FMAND@0x188), so the
// flag FIFO (queued / immediate updates, retire on read) is modelled exactly,
// and OPMULA/OPMSUB take the interpreter's exact double pass. Declines on the
// clip path (IBNE@0x1b0), a non-ordinary FMAC lane, a store/load alias or any
// pipeline state it does not model. PS2X_VU1_HLE=0 disables; verify with
// PS2X_VU1_HLE_VERIFY=1.
namespace hle40
{
// Lane flags -> mac bits / status OR, as updateFmacFlags packs them.
inline void hflaglanes(uint32_t destLanes, const uint8_t lf[4], uint32_t &mac, uint32_t &status)
{
    uint32_t m = 0u, s = 0u;
    for (uint32_t c = 0; c < 4u; ++c)
    {
        if ((destLanes & (1u << c)) == 0u)
            continue;
        const uint32_t flags = lf[c];
        const uint32_t laneBit = 8u >> c;
        if (flags & 0x1u) m |= laneBit;
        if (flags & 0x2u) m |= laneBit << 4;
        if (flags & 0x4u) m |= laneBit << 8;
        if (flags & 0x8u) m |= laneBit << 12;
        s |= flags;
    }
    mac = m;
    status = s;
}
// Ordinary-magnitude FMAC result: only Z and S per lane.
inline void hflagbits(uint32_t destLanes, uint32_t negMask, uint32_t resZero, uint32_t &mac, uint32_t &status)
{
    const uint32_t zero = resZero & destLanes;
    const uint32_t negative = negMask & destLanes;
    mac = hle1028::hreverse4(zero) | (hle1028::hreverse4(negative) << 4);
    status = (zero != 0u ? 1u : 0u) | (negative != 0u ? 2u : 0u);
}
// VU1Interpreter::normalizeFmacExactResult.
inline uint8_t hexact(float &value, double exact)
{
    const bool negative = std::signbit(exact);
    const double magnitude = std::fabs(exact);
    uint8_t flags = negative ? 0x2u : 0u;
    uint32_t bits = negative ? 0x80000000u : 0u;
    if (magnitude == 0.0)
    {
        flags |= 0x1u;
        std::memcpy(&value, &bits, sizeof(value));
    }
    else if (magnitude > static_cast<double>(std::numeric_limits<float>::max()))
    {
        flags |= 0x8u;
        bits |= 0x7F7FFFFFu;
        std::memcpy(&value, &bits, sizeof(value));
    }
    else if (magnitude < static_cast<double>(std::numeric_limits<float>::min()))
    {
        flags |= 0x5u;
        std::memcpy(&value, &bits, sizeof(value));
    }
    return flags;
}
// execUpper OPMSUB (opmsub) / OPMULA with normalised operands: float result,
// normalizeFmacResult per dest lane, calculateFmacProductSticky (OPMSUB only).
inline void hcrossScalar(__m128 vsN, __m128 vtN, __m128 accN, uint32_t destLanes, bool opmsub, float res[4], uint8_t lf[4], uint32_t &sticky)
{
    static constexpr uint32_t left[4] = {1u, 2u, 0u, 3u};
    static constexpr uint32_t right[4] = {2u, 0u, 1u, 3u};
    float vs[4], vt[4], acc[4];
    _mm_storeu_ps(vs, vsN);
    _mm_storeu_ps(vt, vtN);
    _mm_storeu_ps(acc, accN);
    for (uint32_t c = 0; c < 3u; ++c)
        res[c] = opmsub ? acc[c] - vs[left[c]] * vt[right[c]] : vs[left[c]] * vt[right[c]];
    res[3] = 0.0f;
    sticky = 0u;
    for (uint32_t c = 0; c < 4u; ++c)
    {
        lf[c] = 0u;
        if ((destLanes & (1u << c)) == 0u)
            continue;
        const double exact = c == 3u ? 0.0
                                     : (opmsub ? static_cast<double>(acc[c]) - static_cast<double>(vs[left[c]]) * static_cast<double>(vt[right[c]])
                                               : static_cast<double>(vs[left[c]]) * static_cast<double>(vt[right[c]]));
        lf[c] = hexact(res[c], exact);
        if (opmsub)
        {
            float product = vs[left[c]] * vt[right[c]];
            const double exactProduct = static_cast<double>(vs[left[c]]) * static_cast<double>(vt[right[c]]);
            sticky |= hexact(product, exactProduct) & 0xFu;
        }
    }
}
// Same rounded float product/subtraction and exact double classification as
// the scalar path. No fused multiply-add and no approximate flag shortcut:
// cancellation, signed zero and exponent boundary behavior stay observable.
inline void hcrossPacked(__m128 vsN, __m128 vtN, __m128 accN, uint32_t destLanes, bool opmsub, float res[4], uint8_t lf[4], uint32_t &sticky)
{
    const __m128 left=_mm_shuffle_ps(vsN,vsN,_MM_SHUFFLE(3,0,2,1));
    const __m128 right=_mm_shuffle_ps(vtN,vtN,_MM_SHUFFLE(3,1,0,2));
    const __m128 product=_mm_mul_ps(left,right);
    __m128 result=opmsub?_mm_sub_ps(accN,product):product;
    result=_mm_blend_ps(result,_mm_setzero_ps(),8);
    const __m256d exactProduct=_mm256_mul_pd(_mm256_cvtps_pd(left),_mm256_cvtps_pd(right));
    __m256d exact=opmsub?_mm256_sub_pd(_mm256_cvtps_pd(accN),exactProduct):exactProduct;
    exact=_mm256_blend_pd(exact,_mm256_setzero_pd(),8);
    const auto classify=[](__m256d value) {
        const __m256d magnitude=_mm256_andnot_pd(_mm256_set1_pd(-0.0),value);
        vu1n::VuLaneClass flags;
        flags.zero=uint32_t(_mm256_movemask_pd(_mm256_cmp_pd(magnitude,_mm256_setzero_pd(),_CMP_EQ_OQ)));
        flags.neg=uint32_t(_mm256_movemask_pd(value));
        flags.over=uint32_t(_mm256_movemask_pd(_mm256_cmp_pd(magnitude,
            _mm256_set1_pd(double(std::numeric_limits<float>::max())),_CMP_GT_OQ)));
        flags.under=uint32_t(_mm256_movemask_pd(_mm256_cmp_pd(magnitude,
            _mm256_set1_pd(double(std::numeric_limits<float>::min())),_CMP_LT_OQ)))&~flags.zero;
        return flags;
    };
    const auto flags=classify(exact);
    const auto productSticky=[&] {
        if(!opmsub)return 0u;
        // An ordinary rounded product has only its sign flag. Everything
        // outside the normal finite range still gets exact classification.
        // Exclude the FLT_MIN endpoint too: round-up/nearest can round an
        // exact underflow to FLT_MIN. Normal gameplay uses round-toward-zero,
        // but this helper keeps the scalar result in other rounding modes.
        const auto magnitude=_mm_and_si128(_mm_castps_si128(product),_mm_set1_epi32(0x7fffffff));
        const auto interior=_mm_and_si128(_mm_cmpgt_epi32(magnitude,_mm_set1_epi32(0x00800000)),
            _mm_cmpgt_epi32(_mm_set1_epi32(0x7f7fffff),magnitude));
        if((destLanes&~uint32_t(_mm_movemask_ps(_mm_castsi128_ps(interior))))==0u)
            return (uint32_t(_mm_movemask_ps(product))&destLanes)?2u:0u;
        const auto p=classify(exactProduct);
        return ((p.zero|p.under)&destLanes?1u:0u)|(p.neg&destLanes?2u:0u)|
            (p.under&destLanes?4u:0u)|(p.over&destLanes?8u:0u);
    };
    if(((flags.zero|flags.under|flags.over)&destLanes)==0u) {
        _mm_storeu_ps(res,result);
        for(unsigned lane=0;lane<4;++lane)
            lf[lane]=uint8_t(((flags.neg&destLanes)>>lane&1u)*2u);
        sticky=productSticky();
        return;
    }
    const __m128i sign=_mm_setr_epi32((flags.neg&1u)?int32_t(0x80000000u):0,
        (flags.neg&2u)?int32_t(0x80000000u):0,(flags.neg&4u)?int32_t(0x80000000u):0,
        (flags.neg&8u)?int32_t(0x80000000u):0);
    const unsigned exceptional=(flags.zero|flags.under|flags.over)&destLanes;
    const __m128i replace=_mm_setr_epi32((exceptional&1u)?-1:0,(exceptional&2u)?-1:0,
        (exceptional&4u)?-1:0,(exceptional&8u)?-1:0);
    const __m128i overflow=_mm_setr_epi32((flags.over&1u)?0x7f7fffff:0,(flags.over&2u)?0x7f7fffff:0,
        (flags.over&4u)?0x7f7fffff:0,(flags.over&8u)?0x7f7fffff:0);
    result=_mm_castsi128_ps(_mm_blendv_epi8(_mm_castps_si128(result),_mm_or_si128(sign,overflow),replace));
    _mm_storeu_ps(res,result);
    for(unsigned lane=0;lane<4;++lane) {
        const unsigned bit=1u<<lane;
        lf[lane]=(destLanes&bit)?uint8_t(((flags.zero|flags.under)&bit?1u:0u)|
            (flags.neg&bit?2u:0u)|(flags.under&bit?4u:0u)|(flags.over&bit?8u:0u)):0u;
    }
    sticky=productSticky();
}
inline void hcross(__m128 vsN, __m128 vtN, __m128 accN, uint32_t destLanes, bool opmsub, float res[4], uint8_t lf[4], uint32_t &sticky)
{
    static const bool enabled=[] {const char* p=std::getenv("PS2X_VU1_SIMD_CROSS");return !p || p[0]!='0';}();
    if(enabled)hcrossPacked(vsN,vtN,accN,destLanes,opmsub,res,lf,sticky);
    else hcrossScalar(vsN,vtN,accN,destLanes,opmsub,res,lf,sticky);
}

}

bool Vu1NativeAccess::hleShader40(I &v, Vu1NativeCtx &c)
{
    using namespace hle1028;
    using namespace hle40;
    if (v.m_cycle >= v.m_pipelineNextReady)
        v.commitReadyPipelines();
    // Enter through the exact first pair when flags are pending. It performs
    // the reference FIFO drain itself; no pending state is folded or ignored.
    // If the remaining native body declines, the caller re-dispatches at 130.
    bool prefixExecuted=false;
    if(s_nativeBoundary && (v.m_flagMask || v.m_flagOrderCount) &&
       !(v.m_storeMask | v.m_vfWriteMask | v.m_viWriteMask | v.m_accWriteMask) &&
       !v.m_fdiv.valid && !v.m_efu[0].valid && !v.m_efu[1].valid &&
       !v.m_xgkick.active && !v.m_state.branchPending && !v.m_state.haltAfterDelaySlot &&
       !v.m_state.ebit && !v.m_stopRequested && c.dataSize==16384u &&
       v.m_cycle+28u<=c.budgetEnd) {
        const auto result=stepPair<locallyGeneratedVu::hle40FirstUpper,locallyGeneratedVu::hle40FirstLower,true,true,true,false,true>(v,c);
        prefixExecuted=true;
        if(result!=Vu1NativeStep::Continue) return true;
        // Mid pairs leave PC publication to the enclosing native block.
        v.m_state.pc=0x130u;
    }
    if ((v.m_storeMask | v.m_vfWriteMask | v.m_viWriteMask | v.m_accWriteMask | v.m_flagMask) != 0u || v.m_flagOrderCount != 0u ||
        v.m_fdiv.valid || v.m_efu[0].valid || v.m_efu[1].valid || v.m_xgkick.active || v.m_state.branchPending ||
        v.m_state.haltAfterDelaySlot || v.m_state.ebit || v.m_stopRequested || c.dataSize != 16384u ||
        v.m_cycle + 28u > c.budgetEnd) {
        if(s_kernelProfile) {
            const unsigned why = (v.m_storeMask ? 1u : 0u) |
                ((v.m_vfWriteMask | v.m_viWriteMask | v.m_accWriteMask) ? 2u : 0u) |
                ((v.m_flagMask | v.m_flagOrderCount) ? 4u : 0u) |
                (v.m_fdiv.valid ? 8u : 0u) | ((v.m_efu[0].valid || v.m_efu[1].valid) ? 16u : 0u) |
                (v.m_xgkick.active ? 32u : 0u) | (v.m_state.branchPending ? 64u : 0u) |
                ((v.m_state.haltAfterDelaySlot || v.m_state.ebit || v.m_stopRequested || c.dataSize!=16384u) ? 128u : 0u) |
                (v.m_cycle + 28u > c.budgetEnd ? 256u : 0u);
            if(++s_kernel40.guards[why]%100000u==0u) s_kernel40.dump();
        }
        return prefixExecuted;
    }
    const uint64_t chunkStart = v.m_cycle - unsigned(prefixExecuted);
    static const bool meshKernel = [] { const char *p = std::getenv("PS2X_VU1_MESH_KERNEL"); return !p || p[0] != '0'; }();
    const unsigned maxIterations = meshKernel ? 64u : 1u;
    unsigned iterations = 0, totalPairs = 0, fifoPushes = 0;
    uint64_t lastFlagCycle = 0;
    uint32_t lastUpper = 0;
    bool loopTaken = false, skip14 = false, branched = false;
    uint8_t *const mem = c.vuData;
    __m128 R[32];
    for (int i = 0; i < 32; ++i)
        R[i] = _mm_loadu_ps(v.m_state.vf[i]);
    __m128 ACC = _mm_loadu_ps(v.m_state.acc);
    float Q = v.m_state.q, IR = v.m_state.i, qPending = 0.0f;
    uint32_t diPending = 0u;
    int32_t VI[16];
    std::memcpy(VI, v.m_state.vi, sizeof(VI));
    uint32_t MAC = v.m_state.mac, ST = v.m_state.status, CLIPR = v.m_state.clip, WCLIP = v.m_workingClip;
    int32_t oldBranchVi = v.m_viBranchBackupValue;
    uint8_t branchViReg = v.m_viBranchBackupReg;
    do {
        const uint64_t c0 = chunkStart + totalPairs;
        uint32_t loadAddr[16], loadIdx[16], storeAddr[6], storeLanes[6], storeIdx[6];
        unsigned nLoad = 0, nStore = 0;
        float storeWords[6][4];
        constexpr unsigned dirty[] = {5,7,8,9,10,11,12,14,15,16,17,19,20,23,24,25,26,28,29,30};
        __m128 savedR[20], savedAcc;
        int32_t savedVi[16], savedOldBranch = oldBranchVi;
        uint8_t savedBranchReg = branchViReg;
        const float savedQ = Q, savedI = IR;
        const uint32_t savedMac = MAC, savedStatus = ST, savedClip = CLIPR, savedWorkingClip = WCLIP;
        if (iterations) {
            for (unsigned i = 0; i < 20; ++i) savedR[i] = R[dirty[i]];
            savedAcc = ACC; std::memcpy(savedVi, VI, sizeof(VI));
        }
        uint32_t stepPairs = 0, stepPushes = 0, stepUpper = 0;
        uint64_t stepFlag = 0;
        bool stepSkip = false;
        const int result = [&]() -> int {
#include <ps2_vu1_hle40.inc>
            stepPairs = hlePairs; stepPushes = hleFifoPushes; stepUpper = hleLastUpper;
            stepFlag = hleLastFlagCycle; stepSkip = skip14;
            return loopTaken ? 2 : 1;
        }();
        if(s_kernelProfile && result==0) ++s_kernel40.bodyFailures;
        bool failed = result == 0;
        if (!failed) failed=hleCheckedAlias(wotmVuAlias::shader40(),
            storeAddr,storeIdx,nStore,loadAddr,loadIdx,nLoad);
        if(s_kernelProfile && failed && result!=0) ++s_kernel40.aliasFailures;
        if (failed) {
            if (!iterations) return prefixExecuted;
            for (unsigned i = 0; i < 20; ++i) R[dirty[i]] = savedR[i];
            ACC = savedAcc; std::memcpy(VI, savedVi, sizeof(VI));
            oldBranchVi = savedOldBranch; branchViReg = savedBranchReg;
            Q = savedQ; IR = savedI; MAC = savedMac; ST = savedStatus;
            CLIPR = savedClip; WCLIP = savedWorkingClip;
            break;
        }
        for (unsigned st = 0; st < nStore; ++st) {
            if (storeLanes[st] == 0xFu) std::memcpy(mem + storeAddr[st], storeWords[st], 16u);
            else for (uint32_t comp = 0; comp < 4u; ++comp)
                if (storeLanes[st] & (1u << (3u - comp)))
                    std::memcpy(mem + storeAddr[st] + comp * 4u, &storeWords[st][comp], 4u);
        }
        ++iterations; totalPairs += stepPairs; fifoPushes += stepPushes;
        lastFlagCycle = stepFlag; lastUpper = stepUpper; skip14 = stepSkip;
        loopTaken = result == 2; branched |= loopTaken;
    } while (loopTaken && iterations < maxIterations && chunkStart + totalPairs + 28u <= c.budgetEnd);

    if(s_kernelProfile) {++s_kernel40.chunks;s_kernel40.vertices+=iterations;}
    // Only these registers are written by this native loop.
    constexpr unsigned written[] = {5,7,8,9,10,11,12,14,15,16,17,19,20,23,24,25,26,28,29,30};
    for (unsigned i : written)
        _mm_storeu_ps(v.m_state.vf[i], R[i]);
    _mm_storeu_ps(v.m_state.acc, ACC);
    v.m_state.q = Q;
    v.m_state.i = IR;
    VI[0] = 0;
    std::memcpy(v.m_state.vi, VI, sizeof(VI));
    v.m_state.mac = MAC;
    v.m_state.status = ST;
    v.m_state.clip = CLIPR;
    v.m_workingClip = WCLIP;
    v.m_cycle = chunkStart + totalPairs;
    v.m_state.cycles = v.m_cycle;
    v.m_flagOrderHead = (v.m_flagOrderHead + fifoPushes) % I::kMaxFlagEntries;
    v.m_flagTailReady = lastFlagCycle + I::kFmacLatency;
    v.m_currentUpperInstruction = lastUpper;
    v.m_viBranchBackupValue = oldBranchVi;
    v.m_viBranchBackupReg = branchViReg;
    v.m_viBranchBackupValid = false;
    v.m_state.branchPending = false;
    if (branched) {
        v.m_state.branchTarget = 0x128u;
        v.m_state.branchDelay = 0u;
    }
    if (loopTaken)
    {
        v.m_state.pc = 0x128u;
        v.m_state.branchTarget = 0x128u;
        v.m_state.branchDelay = 0u;
    }
    else
    {
        v.m_state.pc = 0x208u;
        if (skip14)
        {
            v.m_state.branchTarget = 0x1b0u;
            v.m_state.branchDelay = 0u;
        }
    }
    c.instrCount += totalPairs - unsigned(prefixExecuted);
    return true;
}

// Called at case 0x0128 of the 7d7edbc78086ac63 native program.
bool hleShader40Try(VU1Interpreter &v, Vu1NativeCtx &c, Vu1NativeProgram nativeProgram)
{
    static const bool enabled = [] { const char *p = std::getenv("PS2X_VU1_HLE"); return !p || p[0] != '0'; }();
    static const bool verify = [] { const char *p = std::getenv("PS2X_VU1_HLE_VERIFY"); return p && p[0] == '1'; }();
    if (!enabled || t_hleVerifyNative || !g_ps2xHleOn.load(std::memory_order_relaxed))
        return false;
    if (!verify)
        return Vu1NativeAccess::hleShader40(v, c);
    static uint64_t s_calls = 0u, s_checked = 0u, s_bad = 0u, s_declined = 0u;
    if ((++s_calls % (std::getenv("PS2X_VU1_HLE_VERIFY_ALL") ? 1u : 32u)) != 0u)
    {
        const bool ok = Vu1NativeAccess::hleShader40(v, c);
        s_declined += ok ? 0u : 1u;
        return ok;
    }
    auto before = std::make_unique<VU1Interpreter>(v);
    std::vector<uint8_t> memBefore(c.vuData, c.vuData + c.dataSize);
    const uint64_t icBefore = c.instrCount;
    const uint64_t cycleBefore = Vu1NativeAccess::cycle(v);
    if (!Vu1NativeAccess::hleShader40(v, c))
    {
        ++s_declined;
        return false;
    }
    auto hle = std::make_unique<VU1Interpreter>(v);
    std::vector<uint8_t> memHle(c.vuData, c.vuData + c.dataSize);
    const uint64_t icHle = c.instrCount;
    const uint64_t hleCycles = Vu1NativeAccess::cycle(v) - cycleBefore;
    v = *before;
    // The generated straight-line chunk reaches case 128 by fallthrough,
    // without publishing PC at every Mid pair. Replay starts at this hook's
    // actual instruction, not the last published checkpoint before it.
    if(Vu1NativeAccess::pc(v)!=0x128u) {
        static uint64_t stalePc=0;
        if(++stalePc==1 || stalePc%100000==0)
            std::fprintf(stderr,"[vu1:hle40-reference] normalized=%llu storedPC=%x actualPC=128\n",stalePc,Vu1NativeAccess::pc(v));
        Vu1NativeAccess::setPc(v,0x128u);
    }
    std::memcpy(c.vuData, memBefore.data(), c.dataSize);
    Vu1NativeCtx cn = c;
    cn.instrCount = icBefore;
    cn.budgetEnd = std::min<uint64_t>(c.budgetEnd, cycleBefore + hleCycles);
    t_hleVerifyNative = true;
    if(hleCycles==1u && Vu1NativeAccess::pc(*hle)==0x130u) {
        // A declined suffix returns the completed boundary pair. The chunk
        // runner checks budget only at non-Mid instructions and would execute
        // past this point; use the exact pair reference for this one-pair slice.
        (void)Vu1NativeAccess::stepPair<locallyGeneratedVu::hle40FirstUpper,locallyGeneratedVu::hle40FirstLower,true,true,true,false,true>(v,cn);
        Vu1NativeAccess::setPc(v,0x130u);
    } else (void)nativeProgram(v, cn);
    t_hleVerifyNative = false;
    c.instrCount = cn.instrCount;
    ++s_checked;
    const char *what = "";
    bool same = Vu1NativeAccess::hleSameState(*hle, v, &what);
    if (same && std::memcmp(memHle.data(), c.vuData, c.dataSize) != 0) { same = false; what = "memory"; }
    if (same && icHle != cn.instrCount) { same = false; what = "instrCount"; }
    if (!same && ++s_bad <= 20u)
        std::fprintf(stderr, "[vu1:hle40] MISMATCH #%llu field=%s cycle=%llu pc hle=0x%x native=0x%x\n",
                     (unsigned long long)s_bad, what, (unsigned long long)Vu1NativeAccess::cycle(v), hle->state().pc, v.state().pc);
    if(!same && std::getenv("PS2X_VU1_HLE_VERIFY_ALL")) std::abort();
    if (s_checked == 1u || (s_checked % 2000u) == 0u)
        std::fprintf(stderr, "[vu1:hle40] checked=%llu mismatched=%llu declined=%llu calls=%llu\n",
                     (unsigned long long)s_checked, (unsigned long long)s_bad, (unsigned long long)s_declined, (unsigned long long)s_calls);
    return true; // native result kept
}

#include <menuShaderBlock.inc>


// Gameplay shaders: dead MAC/STATUS writes in 10a8..1198,
// 30f0..31e8 and 3210..3238. Later FMAC writes replace them before reads;
// CLIP and all original arithmetic, memory and timing behavior are retained.
#include <fightShaderBlock.inc>

#include "ps2_vu1_object_shader.inc"

// Game-specific dead flag elimination, independently selectable for A/B checks.
Vu1NativeStep ps2xVu1HeavyBlock(VU1Interpreter &v, Vu1NativeCtx &c, Vu1NativeProgram nativeProgram) {
    // Strict diagnostic mode retains the reference's inactive flag history.
    static const bool objectNative=[] {
        const char *p=std::getenv("PS2X_VU1_OBJECT_NATIVE");
        const char *math=std::getenv("PS2X_VU1_GAME_MATH");
        return (!p || p[0]!='0') && (!math || math[0]!='0');
    }();
    if(objectNative && (Vu1NativeAccess::pc(v)==0xff0u || Vu1NativeAccess::pc(v)==0x1ea0u) &&
       ((static_cast<uint16_t>(v.state().vi[13])*8u)&0x3fffu)==0x15e8u)
    {
        static const bool deadFlags=[] {const char *p=std::getenv("PS2X_VU1_GAME_MATH");const char *q=std::getenv("PS2X_VU1_OBJECT_FLAGS");return (!p || p[0]!='0') && (!q || q[0]!='0');}();
        return deadFlags?objectShaderBlock<true>(v,c,nativeProgram):objectShaderBlock<false>(v,c,nativeProgram);
    }
    static const bool gameMath=[] {const char *p=std::getenv("PS2X_VU1_GAME_MATH");return !p || p[0]!='0';}();
    static const bool shaderMath=[] {const char *p=std::getenv("PS2X_VU1_SHADER_MATH");return !p || p[0]!='0';}();
    // The verified second geometry entry shares the existing native vertex path.
    // Set the switch to 0 to retain its original entry route for comparisons.
    static const bool geometryEntryNative=[] {
        const char *p=std::getenv("PS2X_VU1_GEOMETRY_ENTRY_NATIVE");
        return !p || p[0]!='0';
    }();
    if(shaderMath && gameMath) {
        const uint32_t pc=Vu1NativeAccess::pc(v);
        if(pc==0xff0u || (geometryEntryNative && pc==0x1ea0u)) {
            const uint32_t shader=(static_cast<uint16_t>(v.state().vi[13])*8u)&0x3fffu;
            if(shader==0x19f8u || (pc==0xff0u && shader==0xc08u))return menuShaderBlock(v,c,nativeProgram);
        }
    }
    static const bool fightShaders=[] {const char *p=std::getenv("PS2X_VU1_FIGHT_SHADERS");return !p || p[0]!='0';}();
    // Both gameplay entries share the verified fight shader. Set the switch to 0 for the original entry route.
    static const bool fightEntryNative=[] {
        const char *p=std::getenv("PS2X_VU1_FIGHT_ENTRY_NATIVE");
        const char *math=std::getenv("PS2X_VU1_GAME_MATH");
        return (!p || p[0]!='0') && (!math || math[0]!='0');
    }();
    if(fightShaders && (Vu1NativeAccess::pc(v)==0xff0u ||
                       (fightEntryNative && Vu1NativeAccess::pc(v)==0x1ea0u))) {
        const uint32_t shader=(static_cast<uint16_t>(v.state().vi[13])*8u)&0x3fffu;
        if(shader==0x1028u || shader==0x3080u)return fightShaderBlock(v,c,nativeProgram);
    }
    // Active-fight shaders outside the compact block. Enter their native
    // reference at ff0 directly, keeping all setup, branch and timing work.
    static const bool gameplayGate=[] {const char *p=std::getenv("PS2X_VU1_GAMEPLAY_GATE");return !p || p[0]!='0';}();
    if(gameplayGate && Vu1NativeAccess::pc(v)==0xff0u) {
        const uint32_t shader=(static_cast<uint16_t>(v.state().vi[13])*8u)&0x3fffu;
        if(shader==0x3080u || shader==0x40u)return nativeProgram(v,c);
    }
    // These shader bodies are not in the compact block. Avoid entering it
    // only to redispatch to native after the ff0 setup instructions.
    static const bool shaderGate=[] {const char *p=std::getenv("PS2X_VU1_SHADER_GATE");return !p || p[0]!='0';}();
    if(shaderGate && Vu1NativeAccess::pc(v)==0xff0u) {
        const uint32_t shader=(static_cast<uint16_t>(v.state().vi[13])*8u)&0x3fffu;
        if(shader==0x1028u || shader==0x1228u || shader==0x19f8u ||
           shader==0xc08u || shader==0x6d8u || shader==0x7d8u)return nativeProgram(v,c);
    }

    static const bool lightingFlags=[] {
        const char *p=std::getenv("PS2X_VU1_LIGHTING_FLAGS");return !p || p[0]!='0';
    }();
    if(lightingFlags && gameMath &&
       (Vu1NativeAccess::pc(v)==0xff0u || Vu1NativeAccess::pc(v)==0x1ea0u) &&
       ((static_cast<uint16_t>(v.state().vi[13])*8u)&0x3fffu)==0x1878u)
        return heavyBlockImpl<true,true>(v,c,nativeProgram);
    return gameMath?heavyBlockImpl<true>(v,c,nativeProgram):heavyBlockImpl<false>(v,c,nativeProgram);
}
