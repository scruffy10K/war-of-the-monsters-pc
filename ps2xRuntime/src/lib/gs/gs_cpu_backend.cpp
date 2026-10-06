#include <map>
#include <chrono>
#if defined(_M_X64) || defined(__SSE2__)
#include <emmintrin.h>
#endif
#include "runtime/gs/gs_cpu_backend.h"
#include "runtime/gs/ps2_gs_common.h"
#include "runtime/gs/ps2_gs_psmct16.h"
#include "runtime/gs/ps2_gs_psmct32.h"
#include "runtime/gs/ps2_gs_psmt4.h"
#include "runtime/gs/ps2_gs_psmt8.h"
#include "runtime/gs/ps2_gs_memory.h"
#include "ps2_log.h"
#include <atomic>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <memory>
#include <type_traits>
#include <iostream>


extern bool ps2xRenderGsGpuSegment(const uint8_t *, uint32_t, const GSPrimitiveBatch *, size_t, size_t, std::vector<uint8_t> &, std::vector<uint8_t> *);

namespace
{
    // Which band this thread rasterizes. The dispatcher runs as band 0 while a
    // parallel run is in flight and with t_bandCount == 1 otherwise, so every
    // non-band caller (inline mode, snapshot backend, barrier commands) sees
    // RowInBand() == true and behaves exactly as before.
    thread_local uint32_t t_bandIndex = 0u;
    thread_local uint32_t t_bandCount = 1u;
    // Consumed by the immediately following WritePixel on this raster thread.
    thread_local bool t_depthAlreadyPassed = false;

    inline bool RowInBand(int y)
    {
        return t_bandCount <= 1u || ((static_cast<uint32_t>(y) >> 3u) % t_bandCount) == t_bandIndex;
    }

    struct BlockRange
    {
        uint32_t begin = 0u;
        uint32_t end = 0u; // exclusive
    };

    // Conservative VRAM footprint in 256-byte blocks, widened to whole 8 KiB
    // pages: swizzled layouts scatter a rectangle across its pages, so page
    // granularity is the tightest bound that is still safe.
    inline BlockRange pageRangeFor(uint32_t baseBlock, uint32_t widthPixels, uint32_t heightPixels, uint32_t bpp)
    {
        const uint64_t bytes = static_cast<uint64_t>(widthPixels) * heightPixels * bpp / 8u;
        const uint64_t blocks = (bytes + 255u) / 256u;
        BlockRange r;
        r.begin = baseBlock & ~31u;
        const uint64_t end = static_cast<uint64_t>(baseBlock) + std::max<uint64_t>(blocks, 1u);
        r.end = static_cast<uint32_t>(std::min<uint64_t>((end + 31u) & ~31ull, 0xFFFFFFFFull));
        return r;
    }

    inline bool overlaps(const BlockRange &a, const BlockRange &b)
    {
        return a.begin < b.end && b.begin < a.end;
    }

    inline bool isColorTargetPsm(uint8_t psm)
    {
        return psm == GS_PSM_CT32 || psm == GS_PSM_CT24 || psm == GS_PSM_CT16 || psm == GS_PSM_CT16S;
    }

    inline bool isDepthPsm(uint8_t psm)
    {
        return psm == GS_PSM_Z32 || psm == GS_PSM_Z24 || psm == GS_PSM_Z16 || psm == GS_PSM_Z16S ||
               isColorTargetPsm(psm);
    }

    inline bool isPalettedPsm(uint8_t psm)
    {
        return psm == GS_PSM_T8 || psm == GS_PSM_T8H || psm == GS_PSM_T4 || psm == GS_PSM_T4HL || psm == GS_PSM_T4HH;
    }

    // Framebuffer + Z footprint a draw may write.
    inline void writeRangesFor(const GSPrimitiveBatch &batch, BlockRange out[2], int &count)
    {
        const GSContext &ctx = batch.state.context;
        const uint32_t height = static_cast<uint32_t>(ctx.scissor.y1) + 1u;
        const uint32_t width = std::max<uint32_t>(ctx.frame.fbw, 1u) * 64u;
        count = 0;
        out[count++] = pageRangeFor(GSInternal::framePageBaseToBlock(ctx.frame.fbp), width, height,
                                    GSInternal::bitsPerPixel(ctx.frame.psm));
        if (!ctx.zbuf.zmask)
            out[count++] = pageRangeFor(GSInternal::framePageBaseToBlock(ctx.zbuf.zbp), width, height,
                                        GSInternal::bitsPerPixel(ctx.zbuf.psm));
    }

    // Texture (+ CLUT) footprint a textured draw may read.
    inline void readRangesFor(const GSPrimitiveBatch &batch, BlockRange out[2], int &count)
    {
        count = 0;
        if (!batch.state.prim.tme)
            return;
        const GSTex0Reg &tex = batch.state.context.tex0;
        const uint32_t texW = 1u << std::min<uint32_t>(tex.tw, 10u);
        const uint32_t texH = 1u << std::min<uint32_t>(tex.th, 10u);
        const uint32_t rowPixels = std::max<uint32_t>(static_cast<uint32_t>(tex.tbw) * 64u, texW);
        out[count++] = pageRangeFor(tex.tbp0, rowPixels, texH, GSInternal::bitsPerPixel(tex.psm));
        if (isPalettedPsm(tex.psm))
            out[count++] = pageRangeFor(tex.cbp, 16u, 16u, 32u); // up to 256 x 32-bit entries
    }

    // A draw can be split across bands only if its targets are formats whose
    // pixels never share bytes across rows.
    inline bool isBandSafeDraw(const GSPrimitiveBatch &batch)
    {
        const GSContext &ctx = batch.state.context;
        if (!isColorTargetPsm(ctx.frame.psm))
            return false;
        if (!ctx.zbuf.zmask && !isDepthPsm(ctx.zbuf.psm))
            return false;
        // Sampling its own target would let one band read rows another band is
        // writing within the same draw.
        BlockRange w[2], r[2];
        int wc = 0, rc = 0;
        writeRangesFor(batch, w, wc);
        readRangesFor(batch, r, rc);
        for (int i = 0; i < rc; ++i)
            for (int j = 0; j < wc; ++j)
                if (overlaps(r[i], w[j]))
                    return false;
        return true;
    }

    constexpr size_t kMinParallelDraws = 32u;
}


using namespace GSInternal;

namespace
{
    float fabsQ(float q)
    {
        return (std::fabs(q) > 1.0e-8f) ? q : 1.0f;
    }

    u16 Rgba8888ToRgba5551(u32 c)
    {
        uint32_t r = ((c >> 0) & 0xFF) >> 3;
        uint32_t g = ((c >> 8) & 0xFF) >> 3;
        uint32_t b = ((c >> 16) & 0xFF) >> 3;
        uint32_t a = ((c >> 24) & 0xFF) >> 7;

        return (r | (g << 5) | (b << 10) | (a << 15));
    }

    u32 Rgba5551ToRgba8888(u16 c)
    {
        u32 r = ((c >> 0) & 0x1F) << 3;
        u32 g = ((c >> 5) & 0x1F) << 3;
        u32 b = ((c >> 10) & 0x1F) << 3;
        u32 a = ((c >> 15) & 0x01) << 7;

        return (r | (g << 8) | (b << 16) | (a << 24));
    }

    u32 pack32(u8 r, u8 g, u8 b, u8 a)
    {
        return static_cast<u32>(r) | (g << 8) | (b << 16) | (a << 24);
    }

    uint32_t applyTexa(const GSTexaReg &texa, uint8_t psm, uint32_t texel)
    {
        if (psm == GS_PSM_CT32)
            return texel;

        const uint8_t r = static_cast<uint8_t>(texel & 0xFFu);
        const uint8_t g = static_cast<uint8_t>((texel >> 8) & 0xFFu);
        const uint8_t b = static_cast<uint8_t>((texel >> 16) & 0xFFu);
        const bool rgbZero = r == 0u && g == 0u && b == 0u;
        uint8_t a = static_cast<uint8_t>((texel >> 24) & 0xFFu);

        switch (psm)
        {
        case GS_PSM_CT24:
            a = (texa.aem && rgbZero) ? 0u : texa.ta0;
            break;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
            if ((a & 0x80u) != 0u)
                a = texa.ta1;
            else
                a = (texa.aem && rgbZero) ? 0u : texa.ta0;
            break;
        default:
            break;
        }

        return (texel & 0x00FFFFFFu) | (static_cast<uint32_t>(a) << 24);
    }

    uint32_t addrPSMCT16Family(uint32_t basePtr, uint32_t width, uint8_t psm, uint32_t x, uint32_t y)
    {
        switch (psm)
        {
        case GS_PSM_CT16:
            return GSPSMCT16::addrPSMCT16(basePtr, width, x, y);
        case GS_PSM_CT16S:
            return GSPSMCT16::addrPSMCT16S(basePtr, width, x, y);
        case GS_PSM_Z16:
            return GSPSMCT16::addrPSMZ16(basePtr, width, x, y);
        case GS_PSM_Z16S:
            return GSPSMCT16::addrPSMZ16S(basePtr, width, x, y);
        default:
            return 0u;
        }
    }

    std::atomic<uint32_t> s_debugPrimitiveCount{0};
    std::atomic<uint32_t> s_debugPixelCount{0};
    std::atomic<uint32_t> s_debugContext1PrimitiveCount{0};
    std::atomic<uint32_t> s_debugFbp150PixelCount{0};
}

// Defined in gs_frontend.cpp — one-shot full-frame GS stream capture budget.
extern std::atomic<int64_t> g_gsFrameDump;

// Per frame-buffer page (indexed by FBP, 9 bits): how tall the game draws into
// it. Written on the raster side, read by the presenter -- which runs on a
// snapshot copy of the backend, so this cannot live in the object.
//
// Two generations, rolled over once per present: an all-time maximum is wrong
// because pages get reused at different heights (the intro draws 448 rows into
// a page, then the title screen reuses it with 224, and a stale 448 made the
// presenter read past the content into garbage). "This frame" is the truth for
// a buffer being redrawn; for one drawn once and displayed for many frames (a
// static logo) the previous generation is the best available answer.
std::atomic<uint16_t> g_ps2xFbDrawnHeight[512]{};
std::atomic<uint16_t> g_ps2xFbDrawnHeightPrev[512]{};

namespace
{
    uint32_t fbDrawnHeight(uint32_t fbp)
    {
        const uint32_t index = fbp & 0x1FFu;
        const uint32_t current = g_ps2xFbDrawnHeight[index].load(std::memory_order_relaxed);
        return current != 0u ? current : g_ps2xFbDrawnHeightPrev[index].load(std::memory_order_relaxed);
    }

    void rollFbDrawnHeights()
    {
        for (uint32_t i = 0; i < 512u; ++i)
        {
            const uint16_t current = g_ps2xFbDrawnHeight[i].exchange(0u, std::memory_order_relaxed);
            if (current != 0u)
                g_ps2xFbDrawnHeightPrev[i].store(current, std::memory_order_relaxed);
        }
    }
}

namespace
{

    int wrapTextureCoordinate(int coordinate,
                              int textureSize,
                              uint8_t mode,
                              uint16_t regionMin,
                              uint16_t regionMax)
    {
        switch (mode & 0x3u)
        {
        case 0: // REPEAT
            return static_cast<int>(static_cast<uint32_t>(coordinate) & static_cast<uint32_t>(textureSize - 1));
        case 1: // CLAMP
            return clampInt(coordinate, 0, textureSize - 1);
        case 2: // REGION_CLAMP
            return std::min(std::max(coordinate, static_cast<int>(regionMin)), static_cast<int>(regionMax));
        case 3: // REGION_REPEAT
            return static_cast<int>((static_cast<uint32_t>(coordinate) & static_cast<uint32_t>(regionMin)) | static_cast<uint32_t>(regionMax));
        default:
            return coordinate;
        }
    }

    bool passesAlphaTest(uint64_t testReg, uint8_t alpha)
    {
        if ((testReg & 0x1u) == 0u)
            return true;

        const uint8_t atst = static_cast<uint8_t>((testReg >> 1) & 0x7u);
        const uint8_t aref = static_cast<uint8_t>((testReg >> 4) & 0xFFu);

        switch (atst)
        {
        case 0:
            return false;
        case 1:
            return true;
        case 2:
            return alpha < aref;
        case 3:
            return alpha <= aref;
        case 4:
            return alpha == aref;
        case 5:
            return alpha >= aref;
        case 6:
            return alpha > aref;
        case 7:
            return alpha != aref;
        default:
            return true;
        }
    }

    struct PixelWriteMask
    {
        bool writeRgb = true;
        bool writeAlpha = true;
        bool writeDepth = true;

        bool writesFramebuffer() const
        {
            return writeRgb || writeAlpha;
        }

        bool writesAnything() const
        {
            return writesFramebuffer() || writeDepth;
        }
    };

    PixelWriteMask classifyAlphaTest(uint64_t testReg, uint8_t alpha, uint8_t framePsm)
    {
        const bool pass = passesAlphaTest(testReg, alpha);
        if (pass)
            return {};

        // TEST.AFAIL controls what happens when the alpha comparison fails.
        switch (static_cast<uint8_t>((testReg >> 12) & 0x3u))
        {
        case 1: // FB_ONLY
            return {true, true, false};
        case 2: // ZB_ONLY
            return {false, false, true};
        case 3: // RGB_ONLY
            // RGB_ONLY is only distinct for RGBA32. The GS treats it as
            // FB_ONLY for RGB24 and RGBA16 framebuffers.
            if (framePsm == GS_PSM_CT32)
                return {true, false, false};
            return {true, true, false};
        case 0: // KEEP
        default:
            return {false, false, false};
        }
    }

    bool passesDestinationAlphaTest(uint64_t testReg, uint8_t framePsm, uint32_t rawFramebufferPixel)
    {
        const bool date = ((testReg >> 14) & 0x1u) != 0u;
        if (!date)
            return true;

        const bool datm = ((testReg >> 15) & 0x1u) != 0u;
        switch (framePsm)
        {
        case GS_PSM_CT32:
            return (((rawFramebufferPixel >> 31) & 0x1u) != 0u) == datm;
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
            return (((rawFramebufferPixel >> 15) & 0x1u) != 0u) == datm;
        case GS_PSM_CT24:
            // RGB24 has no destination alpha, so DATE always passes.
            return true;
        default:
            return true;
        }
    }

    struct TextureCombineResult
    {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;
    };

    TextureCombineResult combineTexture(const GSTex0Reg &tex,
                                        uint8_t vr,
                                        uint8_t vg,
                                        uint8_t vb,
                                        uint8_t va,
                                        uint8_t tr,
                                        uint8_t tg,
                                        uint8_t tb,
                                        uint8_t ta)
    {
        const bool textureHasAlpha = tex.tcc != 0u;
        TextureCombineResult out{tr, tg, tb, textureHasAlpha ? ta : va};

        switch (tex.tfx)
        {
        case 0: // MODULATE
            out.r = clampU8((tr * vr) >> 7);
            out.g = clampU8((tg * vg) >> 7);
            out.b = clampU8((tb * vb) >> 7);
            out.a = textureHasAlpha ? clampU8((ta * va) >> 7) : va;
            break;
        case 1: // DECAL
            out.r = tr;
            out.g = tg;
            out.b = tb;
            out.a = textureHasAlpha ? ta : va;
            break;
        case 2: // HIGHLIGHT
            out.r = clampU8(((tr * vr) >> 7) + va);
            out.g = clampU8(((tg * vg) >> 7) + va);
            out.b = clampU8(((tb * vb) >> 7) + va);
            out.a = textureHasAlpha ? clampU8(ta + va) : va;
            break;
        case 3: // HIGHLIGHT2
            out.r = clampU8(((tr * vr) >> 7) + va);
            out.g = clampU8(((tg * vg) >> 7) + va);
            out.b = clampU8(((tb * vb) >> 7) + va);
            out.a = textureHasAlpha ? ta : va;
            break;
        default:
            out.r = tr;
            out.g = tg;
            out.b = tb;
            out.a = textureHasAlpha ? ta : va;
            break;
        }

        return out;
    }

    uint32_t swizzleClutIndexCSM1(uint32_t index)
    {
        // CSM1 swaps address bits 3 and 4. Preserve the remaining bits:
        // 16-bit CLUTs expose a ninth address bit through CSA[4].
        return (index & ~0x18u) | ((index & 0x08u) << 1u) | ((index & 0x10u) >> 1u);
    }

    // TODO: clut cache
    uint32_t resolveClutIndex(uint8_t index, uint8_t cpsm, uint8_t csm, uint8_t csa, uint8_t sourcePsm)
    {
        uint32_t clutIndex = static_cast<uint32_t>(index);

        // CSM2 addresses the source directly through TEXCLUT. CSA is required
        // to be zero there, so it must not offset the source coordinates.
        if (csm != 0u)
            return (sourcePsm == GS_PSM_T4 ||
                    sourcePsm == GS_PSM_T4HH ||
                    sourcePsm == GS_PSM_T4HL)
                       ? (clutIndex & 0x0Fu)
                       : clutIndex;

        const bool is16BitClut = cpsm == GS_PSM_CT16 || cpsm == GS_PSM_CT16S;
        const uint32_t csaMask = is16BitClut ? 0x1Fu : 0x0Fu;
        const uint32_t clutIndexMask = is16BitClut ? 0x1FFu : 0x0FFu;
        const uint32_t clutBase = (static_cast<uint32_t>(csa) & csaMask) << 4u;

        switch (sourcePsm)
        {
        case GS_PSM_T4:
        case GS_PSM_T4HH:
        case GS_PSM_T4HL:
            clutIndex = clutBase + (clutIndex & 0x0Fu);
            break;
        case GS_PSM_T8:
        case GS_PSM_T8H:
            clutIndex = clutBase + clutIndex;
            break;
        default:
            return clutIndex;
        }

        return swizzleClutIndexCSM1(clutIndex & clutIndexMask);
    }

#if defined(_M_X64) || defined(__SSE2__)
    uint32_t lerpRgbaSimd(uint32_t c00,uint32_t c10,uint32_t c01,uint32_t c11,float fx,float fy)
    {
        const auto unpack=[](uint32_t c) {
            const __m128i zero=_mm_setzero_si128();
            const __m128i bytes=_mm_cvtsi32_si128(static_cast<int>(c));
            return _mm_cvtepi32_ps(_mm_unpacklo_epi16(_mm_unpacklo_epi8(bytes,zero),zero));
        };
        const __m128 a=unpack(c00),b=unpack(c10),c=unpack(c01),d=unpack(c11);
        const __m128 x=_mm_set1_ps(fx),y=_mm_set1_ps(fy);
        const __m128 top=_mm_add_ps(a,_mm_mul_ps(_mm_sub_ps(b,a),x));
        const __m128 bottom=_mm_add_ps(c,_mm_mul_ps(_mm_sub_ps(d,c),x));
        const __m128 value=_mm_add_ps(top,_mm_mul_ps(_mm_sub_ps(bottom,top),y));
        // Channels and filter fractions are nonnegative. Round halves upward
        // exactly as lround; adding 0.5 first would misround values just below
        // a half due to the extra floating-point rounding step.
        const __m128i integral=_mm_cvttps_epi32(value);
        const __m128 fraction=_mm_sub_ps(value,_mm_cvtepi32_ps(integral));
        const __m128i rounded=_mm_sub_epi32(integral,_mm_castps_si128(_mm_cmpge_ps(fraction,_mm_set1_ps(0.5f))));
        const __m128i words=_mm_packs_epi32(rounded,_mm_setzero_si128());
        return static_cast<uint32_t>(_mm_cvtsi128_si32(_mm_packus_epi16(words,_mm_setzero_si128())));
    }

#endif

    uint8_t lerpChannel(uint8_t c00, uint8_t c10, uint8_t c01, uint8_t c11, float fx, float fy)
    {
        const float top = static_cast<float>(c00) + (static_cast<float>(c10) - static_cast<float>(c00)) * fx;
        const float bottom = static_cast<float>(c01) + (static_cast<float>(c11) - static_cast<float>(c01)) * fx;
        return clampU8(static_cast<int>(std::lround(top + (bottom - top) * fy)));
    }
}

namespace
{
    static constexpr uint32_t kDefaultDisplayWidth = 640u;
    static constexpr uint32_t kDefaultDisplayHeight = 448u;
    static constexpr uint32_t kHostFrameWidth = 640u;
    static constexpr uint32_t kHostFrameHeight = 512u;

    uint16_t encodeFramePixelPSMCT16(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    {
        return static_cast<uint16_t>(((r >> 3) & 0x1Fu) |
                                     (((g >> 3) & 0x1Fu) << 5) |
                                     (((b >> 3) & 0x1Fu) << 10) |
                                     ((a >= 0x40u) ? 0x8000u : 0u));
    }

    void decodeDisplaySize(uint64_t display64, uint32_t &outWidth, uint32_t &outHeight)
    {
        const uint32_t dw = static_cast<uint32_t>((display64 >> 32) & 0x0FFFu);
        const uint32_t dh = static_cast<uint32_t>((display64 >> 44) & 0x07FFu);
        const uint32_t magh = static_cast<uint32_t>((display64 >> 23) & 0x0Fu);

        outWidth = (dw + 1u) / (magh + 1u);
        outHeight = dh + 1u;
        if (outWidth < 64u || outHeight < 64u)
        {
            outWidth = kDefaultDisplayWidth;
            outHeight = kDefaultDisplayHeight;
        }
        outWidth = std::min<uint32_t>(outWidth, kHostFrameWidth);
        outHeight = std::min<uint32_t>(outHeight, kHostFrameHeight);
    }

    GSFrameReg decodeDisplayFrame(uint64_t dispfb64)
    {
        GSFrameReg frame{};
        frame.fbp = static_cast<uint32_t>(dispfb64 & 0x1FFu);
        frame.fbw = static_cast<uint32_t>((dispfb64 >> 9) & 0x3Fu);
        frame.psm = static_cast<uint8_t>((dispfb64 >> 15) & 0x1Fu);
        return frame;
    }

    struct GSDisplayReadOrigin
    {
        uint32_t x = 0u;
        uint32_t y = 0u;
    };

    GSDisplayReadOrigin decodeDisplayReadOrigin(uint64_t dispfb64)
    {
        return {
            static_cast<uint32_t>((dispfb64 >> 32) & 0x7FFu),
            static_cast<uint32_t>((dispfb64 >> 43) & 0x7FFu)};
    }

    bool hasDisplaySetup(uint64_t display64, const GSFrameReg &frame)
    {
        const uint32_t dw = static_cast<uint32_t>((display64 >> 32) & 0x0FFFu);
        const uint32_t dh = static_cast<uint32_t>((display64 >> 44) & 0x07FFu);
        const uint32_t magh = static_cast<uint32_t>((display64 >> 23) & 0x0Fu);
        return frame.fbw != 0u || dw != 0u || dh != 0u || magh != 0u;
    }

    struct GSPmodeState
    {
        bool enableCrt1 = false;
        bool enableCrt2 = false;
        bool mmod = false;
        bool amod = false;
        bool slbg = false;
        uint8_t alp = 0u;
    };

    GSPmodeState decodePmode(uint64_t pmode64)
    {
        return {
            (pmode64 & 0x1ull) != 0ull,
            (pmode64 & 0x2ull) != 0ull,
            ((pmode64 >> 5) & 0x1ull) != 0ull,
            ((pmode64 >> 6) & 0x1ull) != 0ull,
            ((pmode64 >> 7) & 0x1ull) != 0ull,
            static_cast<uint8_t>((pmode64 >> 8) & 0xFFu)};
    }

    struct GSSmode2State
    {
        bool interlaced = false;
        bool frameMode = true;
    };

    GSSmode2State decodeSMode2(uint64_t smode2)
    {
        return {(smode2 & 0x1ull) != 0ull, ((smode2 >> 1) & 0x1ull) != 0ull};
    }

    void applyFieldPresentation(std::vector<uint8_t> &pixels, uint32_t width, uint32_t height, bool oddField)
    {
        if (pixels.empty() || width == 0u || height < 2u)
            return;
        const std::vector<uint8_t> source = pixels;
        for (uint32_t y = 0; y < height; ++y)
        {
            uint32_t sourceY = ((y >> 1u) << 1u) + (oddField ? 1u : 0u);
            if (sourceY >= height)
                sourceY = height - 1u;
            std::memcpy(pixels.data() + y * kHostFrameWidth * 4u,
                        source.data() + sourceY * kHostFrameWidth * 4u,
                        width * 4u);
        }
    }

    // Interlaced FRAME mode (SMODE2 INT=1, FFMD=1): every field scans the frame
    // buffer from line 0, so a (DH+1)-line display shows (DH+1)/2 buffer lines,
    // each repeated on both fields. The caller copies the half-height image;
    // this doubles it in place (bottom-up so no source row is clobbered first).
    void applyFrameModeLineDouble(std::vector<uint8_t> &pixels, uint32_t width, uint32_t height)
    {
        if (pixels.empty() || width == 0u || height < 2u)
            return;
        for (uint32_t y = height; y-- > 1u;)
        {
            const uint32_t sourceY = y >> 1u;
            std::memcpy(pixels.data() + y * kHostFrameWidth * 4u,
                        pixels.data() + sourceY * kHostFrameWidth * 4u,
                        width * 4u);
        }
    }

    void normalizePresentationAlpha(std::vector<uint8_t> &pixels, uint32_t width, uint32_t height)
    {
        for (uint32_t y = 0; y < height; ++y)
        {
            uint8_t *row = pixels.data() + y * kHostFrameWidth * 4u;
            for (uint32_t x = 0; x < width; ++x)
                row[x * 4u + 3u] = 255u;
        }
    }

    uint8_t blendPresentationChannel(uint8_t src, uint8_t dst, uint32_t factor)
    {
        const int delta = static_cast<int>(src) - static_cast<int>(dst);
        return GSInternal::clampU8(static_cast<int>(dst) + ((delta * static_cast<int>(factor)) / 255));
    }

    uint32_t countNonBlackPixels(const std::vector<uint8_t> &pixels, uint32_t width, uint32_t height)
    {
        uint32_t count = 0u;
        for (uint32_t y = 0; y < height; ++y)
        {
            const uint8_t *row = pixels.data() + y * kHostFrameWidth * 4u;
            for (uint32_t x = 0; x < width; ++x)
            {
                if (row[x * 4u] != 0u || row[x * 4u + 1u] != 0u || row[x * 4u + 2u] != 0u)
                    ++count;
            }
        }
        return count;
    }
}

GSCpuBackend::GSCpuBackend()
{
    using namespace GSMem;
    static std::once_flag lookupTablesOnce;
    std::call_once(lookupTablesOnce, []()
                   { InitLookupTables(); });
    for (size_t i = 0; i < kPsmHandlerCount; ++i)
    {
        switch (i)
        {
        case GS_PSM_CT32:
            m_readVramFuncs[i] = ReadCT32;
            m_writeVramFuncs[i] = WriteCT32;
            break;
        case GS_PSM_CT24:
            m_readVramFuncs[i] = ReadCT24;
            m_writeVramFuncs[i] = WriteCT24;
            break;
        case GS_PSM_CT16:
            m_readVramFuncs[i] = ReadCT16;
            m_writeVramFuncs[i] = WriteCT16;
            break;
        case GS_PSM_CT16S:
            m_readVramFuncs[i] = ReadCT16S;
            m_writeVramFuncs[i] = WriteCT16S;
            break;
        case GS_PSM_T8:
            m_readVramFuncs[i] = ReadP8;
            m_writeVramFuncs[i] = WriteP8;
            break;
        case GS_PSM_T8H:
            m_readVramFuncs[i] = ReadP8H;
            m_writeVramFuncs[i] = WriteP8H;
            break;
        case GS_PSM_T4:
            m_readVramFuncs[i] = ReadP4;
            m_writeVramFuncs[i] = WriteP4;
            break;
        case GS_PSM_T4HH:
            m_readVramFuncs[i] = ReadP4HH;
            m_writeVramFuncs[i] = WriteP4HH;
            break;
        case GS_PSM_T4HL:
            m_readVramFuncs[i] = ReadP4HL;
            m_writeVramFuncs[i] = WriteP4HL;
            break;
        case GS_PSM_Z32:
            m_readVramFuncs[i] = ReadZ32;
            m_writeVramFuncs[i] = WriteZ32;
            break;
        case GS_PSM_Z24:
            m_readVramFuncs[i] = ReadZ24;
            m_writeVramFuncs[i] = WriteZ24;
            break;
        case GS_PSM_Z16:
            m_readVramFuncs[i] = ReadZ16;
            m_writeVramFuncs[i] = WriteZ16;
            break;
        case GS_PSM_Z16S:
            m_readVramFuncs[i] = ReadZ16S;
            m_writeVramFuncs[i] = WriteZ16S;
            break;
        default:
            m_readVramFuncs[i] = ReadNull;
            m_writeVramFuncs[i] = WriteNull;
            break;
        }
    }
    Reset();
}

void GSCpuBackend::Initialize(uint8_t *vram, uint32_t vramSize)
{
    WaitForRasterIdle();
    std::lock_guard<std::mutex> lock(m_mutex);
    m_vram = vram;
    m_vramSize = vramSize;
    ResetUnlocked();
}

void GSCpuBackend::Reset()
{
    WaitForRasterIdle();
    std::lock_guard<std::mutex> lock(m_mutex);
    ResetUnlocked();
}

void GSCpuBackend::ResetUnlocked()
{
    m_transfer = {};
    m_transfer.direction = 3u;
    m_transferState = {};
    m_transferState.direction = 3u;
    m_localToHostBuffer.clear();
    m_localToHostReadPos = 0u;
}

// PS2X_GS_CENSUS=1: which GS draw states the game actually uses. This is the
// feature list a GPU renderer has to implement (prim type, texture format and
// clamp, blend equation, alpha/depth test, write masks).
static void ps2xGsCensus(const GSPrimitiveBatch &batch)
{
    static const bool enabled = [] { const char *p = std::getenv("PS2X_GS_CENSUS"); return p && p[0] == '1'; }();
    if (!enabled)
        return;
    const GSDrawState &s = batch.state;
    const GSContext &c = s.context;
    struct Key
    {
        uint8_t prim, tme, abe, fge, iip, fst, aa1, filter;
        uint8_t framePsm, texPsm, zPsm, zmsk;
        uint32_t fbmsk;
        uint64_t alpha, test, clamp;
        bool operator<(const Key &o) const { return std::memcmp(this, &o, sizeof(Key)) < 0; }
    };
    Key key{};
    key.prim = static_cast<uint8_t>(s.prim.type);
    key.tme = s.prim.tme;
    key.abe = s.prim.abe;
    key.fge = s.prim.fge;
    key.iip = s.prim.iip;
    key.fst = s.prim.fst;
    key.aa1 = s.prim.aa1;
    key.filter = s.linearFilter;
    key.framePsm = static_cast<uint8_t>(c.frame.psm);
    key.texPsm = static_cast<uint8_t>(c.tex0.psm);
    key.zPsm = static_cast<uint8_t>(c.zbuf.psm);
    key.zmsk = static_cast<uint8_t>(c.zbuf.zmask);
    key.fbmsk = c.frame.fbmsk;
    key.alpha = c.alpha;
    key.test = c.test;
    key.clamp = c.clamp;
    static std::mutex mutex;
    static std::map<Key, uint64_t> counts;
    static auto t0 = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(mutex);
    counts[key] += 1u;
    const auto now = std::chrono::steady_clock::now();
    if (now - t0 < std::chrono::seconds(10))
        return;
    t0 = now;
    std::vector<std::pair<Key, uint64_t>> rows(counts.begin(), counts.end());
    std::sort(rows.begin(), rows.end(), [](const auto &a, const auto &b) { return a.second > b.second; });
    uint64_t total = 0u;
    for (const auto &r : rows)
        total += r.second;
    std::fprintf(stderr, "[gs:census] %zu distinct states, %llu prims\n", rows.size(), (unsigned long long)total);
    for (size_t i = 0; i < rows.size() && i < 16u; ++i)
    {
        const Key &k = rows[i].first;
        std::fprintf(stderr,
                     "[gs:census] %5.1f%% prim=%u tex=%u(psm %02x,filt %u,clamp %llx) abe=%u alpha=%llx test=%llx fge=%u iip=%u fst=%u aa1=%u frame(psm %02x,fbmsk %08x) z(psm %02x,msk %u)\n",
                     100.0 * rows[i].second / total, k.prim, k.tme, k.texPsm, k.filter, (unsigned long long)k.clamp, k.abe,
                     (unsigned long long)k.alpha, (unsigned long long)k.test, k.fge, k.iip, k.fst, k.aa1, k.framePsm, k.fbmsk,
                     k.zPsm, k.zmsk);
    }
    counts.clear();
}

void GSCpuBackend::Submit(const GSPrimitiveBatch &batch)
{
    if (!m_vram || batch.vertexCount == 0u)
        return;
    ps2xGsCensus(batch);
    RasterCommand command;
    command.kind = RasterCommand::Kind::Draw;
    command.batch = batch;
    Enqueue(std::move(command));
}

void GSCpuBackend::Flush()
{
    // CPU backend is immediate. GPU backends may submit command buffers here.
}

void GSCpuBackend::TextureFlush()
{
    // CPU texture reads are coherent with local memory. Future cached/GPU
    // backends use this boundary to invalidate texture views.
}

void GSCpuBackend::Sync(GSSyncReason)
{
    WaitForRasterIdle();
}

uint32_t GSCpuBackend::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    WaitForRasterIdle();
    std::lock_guard<std::mutex> lock(m_mutex);
    return ReadVramUnlocked(psm, base, bw, x, y);
}

uint32_t GSCpuBackend::ReadVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    if (!m_vram)
        return 0u;
    // Guard against absurd texel coords surviving the wrap stage (garbage CLAMP
    // region, un-transformed prim): keep the swizzled address inside the 4 MB
    // VRAM so a bad sample can't fault. 2047 is past any legal GS texture.
    if (x > 2047u)
        x = 2047u;
    if (y > 2047u)
        y = 2047u;
    return m_readVramFuncs[psm & 0x3Fu](m_vram, base, bw, x, y);
}

void GSCpuBackend::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    RasterCommand command;
    command.kind = RasterCommand::Kind::WriteVram;
    command.psm = psm;
    command.base = base;
    command.bw = bw;
    command.x = x;
    command.y = y;
    command.value = value;
    Enqueue(std::move(command));
}

void GSCpuBackend::WriteVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    if (!m_vram)
        return;
    m_writeVramFuncs[psm & 0x3Fu](m_vram, base, bw, x, y, value);
}

void GSCpuBackend::SnapshotVram(std::vector<uint8_t> &out) const
{
    WaitForRasterIdle();
    SnapshotVramUnsynced(out);
}

void GSCpuBackend::SnapshotVramUnsynced(std::vector<uint8_t> &out) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_vram || m_vramSize == 0u)
    {
        out.clear();
        return;
    }
    out.resize(m_vramSize);
    std::memcpy(out.data(), m_vram, m_vramSize);
}

GSTransferSnapshot GSCpuBackend::GetTransferSnapshot() const
{
    WaitForRasterIdle();
    std::lock_guard<std::mutex> lock(m_mutex);
    GSTransferSnapshot result = m_transferState;
    result.localToHostPendingBytes = m_localToHostReadPos < m_localToHostBuffer.size()
                                         ? m_localToHostBuffer.size() - m_localToHostReadPos
                                         : 0u;
    return result;
}

struct Ps2xPerf { std::atomic<uint64_t> rasterNs, presentNs, vu1Ns, vif1Ns, prims; };
extern Ps2xPerf g_ps2xPerf;

void GSCpuBackend::DrawPrimitive(const GSPrimitiveBatch &batch)
{
    struct PerfScope
    {
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~PerfScope()
        {
            // Only band 0 (the dispatcher) reports, so prims/s is not multiplied
            // by the band count and the raster share stays "dispatcher busy".
            if (t_bandIndex != 0u)
                return;
            g_ps2xPerf.rasterNs.fetch_add(
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()),
                std::memory_order_relaxed);
            g_ps2xPerf.prims.fetch_add(1u, std::memory_order_relaxed);
        }
    } perfScope;
    const GSDrawState &state = batch.state;
    const auto &ctx = state.context;
    PS2_IF_AGRESSIVE_LOGS({
        const uint32_t primitiveIndex = s_debugPrimitiveCount.fetch_add(1u, std::memory_order_relaxed);
        if (primitiveIndex < 400u || g_gsFrameDump.load(std::memory_order_relaxed) > 0)
        {
            if (g_gsFrameDump.load(std::memory_order_relaxed) > 0)
                g_gsFrameDump.fetch_sub(1, std::memory_order_relaxed);
            std::cout << "[gs:prim] idx=" << primitiveIndex
                      << " type=" << static_cast<uint32_t>(state.prim.type)
                      << " tme=" << static_cast<uint32_t>(state.prim.tme)
                      << " abe=" << static_cast<uint32_t>(state.prim.abe)
                      << " fst=" << static_cast<uint32_t>(state.prim.fst)
                      << " ctxt=" << static_cast<uint32_t>(state.prim.ctxt)
                      << " fbp=" << ctx.frame.fbp
                      << " fbw=" << ctx.frame.fbw
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.frame.psm) << std::dec
                      << " tex0=("
                      << "tbp0=" << ctx.tex0.tbp0
                      << " tbw=" << static_cast<uint32_t>(ctx.tex0.tbw)
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.psm) << std::dec
                      << " tw=" << static_cast<uint32_t>(ctx.tex0.tw)
                      << " th=" << static_cast<uint32_t>(ctx.tex0.th)
                      << " tcc=" << static_cast<uint32_t>(ctx.tex0.tcc)
                      << " tfx=" << static_cast<uint32_t>(ctx.tex0.tfx)
                      << " cbp=" << ctx.tex0.cbp
                      << " cpsm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.cpsm) << std::dec
                      << " csm=" << static_cast<uint32_t>(ctx.tex0.csm)
                      << " csa=" << static_cast<uint32_t>(ctx.tex0.csa)
                      << ")"
                      << " texclut=("
                      << "cbw=" << static_cast<uint32_t>(state.texclut.cbw)
                      << " cou=" << static_cast<uint32_t>(state.texclut.cou)
                      << " cov=" << state.texclut.cov
                      << ")"
                      << " ofx=" << (ctx.xyoffset.ofx >> 4)
                      << " ofy=" << (ctx.xyoffset.ofy >> 4)
                      << " scissor=(" << ctx.scissor.x0
                      << "," << ctx.scissor.y0
                      << ")-(" << ctx.scissor.x1
                      << "," << ctx.scissor.y1 << ")"
                      << " test=0x" << std::hex << ctx.test
                      << " alpha=0x" << ctx.alpha
                      << " zbuf=(" << ctx.zbuf.zbp << "," << static_cast<uint32_t>(ctx.zbuf.psm) << "," << static_cast<uint32_t>(ctx.zbuf.zmask) << ")"
                      << " z=(" << static_cast<uint32_t>(batch.vertices[0].z) << "," << static_cast<uint32_t>(batch.vertices[1].z) << "," << static_cast<uint32_t>(batch.vertices[2].z) << ")"
                      << std::dec
                      << " v0=(" << batch.vertices[0].x << "," << batch.vertices[0].y << ")"
                      << " uv0=(" << (batch.vertices[0].u >> 4) << "," << (batch.vertices[0].v >> 4) << ")"
                      << " stq0=(" << batch.vertices[0].s << "," << batch.vertices[0].t << "," << batch.vertices[0].q << ")"
                      << " v1=(" << batch.vertices[1].x << "," << batch.vertices[1].y << ")"
                      << " uv1=(" << (batch.vertices[1].u >> 4) << "," << (batch.vertices[1].v >> 4) << ")"
                      << " stq1=(" << batch.vertices[1].s << "," << batch.vertices[1].t << "," << batch.vertices[1].q << ")"
                      << " v2=(" << batch.vertices[2].x << "," << batch.vertices[2].y << ")"
                      << " uv2=(" << (batch.vertices[2].u >> 4) << "," << (batch.vertices[2].v >> 4) << ")"
                      << " stq2=(" << batch.vertices[2].s << "," << batch.vertices[2].t << "," << batch.vertices[2].q << ")"
                      << " rgba0=(" << static_cast<uint32_t>(batch.vertices[0].r) << ","
                      << static_cast<uint32_t>(batch.vertices[0].g) << ","
                      << static_cast<uint32_t>(batch.vertices[0].b) << ","
                      << static_cast<uint32_t>(batch.vertices[0].a) << ")"
                      << " rgba1=(" << static_cast<uint32_t>(batch.vertices[1].r) << ","
                      << static_cast<uint32_t>(batch.vertices[1].g) << ","
                      << static_cast<uint32_t>(batch.vertices[1].b) << ","
                      << static_cast<uint32_t>(batch.vertices[1].a) << ")"
                      << " rgba2=(" << static_cast<uint32_t>(batch.vertices[2].r) << ","
                      << static_cast<uint32_t>(batch.vertices[2].g) << ","
                      << static_cast<uint32_t>(batch.vertices[2].b) << ","
                      << static_cast<uint32_t>(batch.vertices[2].a) << ")"
                      << std::endl;
        }
    });

    PS2_IF_AGRESSIVE_LOGS({
        if ((state.prim.ctxt != 0u || ctx.frame.fbp == 150u) &&
            s_debugContext1PrimitiveCount.fetch_add(1u, std::memory_order_relaxed) < 32u)
        {
            std::cout << "[gs:copy-prim]"
                      << " type=" << static_cast<uint32_t>(state.prim.type)
                      << " tme=" << static_cast<uint32_t>(state.prim.tme)
                      << " abe=" << static_cast<uint32_t>(state.prim.abe)
                      << " fst=" << static_cast<uint32_t>(state.prim.fst)
                      << " ctxt=" << static_cast<uint32_t>(state.prim.ctxt)
                      << " fbp=" << ctx.frame.fbp
                      << " fbw=" << ctx.frame.fbw
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.frame.psm) << std::dec
                      << " tex0=("
                      << "tbp0=" << ctx.tex0.tbp0
                      << " tbw=" << static_cast<uint32_t>(ctx.tex0.tbw)
                      << " psm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.psm) << std::dec
                      << " tcc=" << static_cast<uint32_t>(ctx.tex0.tcc)
                      << " tfx=" << static_cast<uint32_t>(ctx.tex0.tfx)
                      << " cbp=" << ctx.tex0.cbp
                      << " cpsm=0x" << std::hex << static_cast<uint32_t>(ctx.tex0.cpsm) << std::dec
                      << " csm=" << static_cast<uint32_t>(ctx.tex0.csm)
                      << " csa=" << static_cast<uint32_t>(ctx.tex0.csa)
                      << ")"
                      << " texclut=("
                      << "cbw=" << static_cast<uint32_t>(state.texclut.cbw)
                      << " cou=" << static_cast<uint32_t>(state.texclut.cou)
                      << " cov=" << state.texclut.cov
                      << ")"
                      << " ofx=" << (ctx.xyoffset.ofx >> 4)
                      << " ofy=" << (ctx.xyoffset.ofy >> 4)
                      << " scissor=(" << ctx.scissor.x0
                      << "," << ctx.scissor.y0
                      << ")-(" << ctx.scissor.x1
                      << "," << ctx.scissor.y1 << ")"
                      << " test=0x" << std::hex << ctx.test
                      << " alpha=0x" << ctx.alpha
                      << std::dec << std::endl;
        }
    });

    switch (state.prim.type)
    {
    case GS_PRIM_SPRITE:
        DrawSprite(batch);
        break;
    case GS_PRIM_TRIANGLE:
    case GS_PRIM_TRISTRIP:
    case GS_PRIM_TRIFAN:
        DrawTriangle(batch);
        break;
    case GS_PRIM_LINE:
    case GS_PRIM_LINESTRIP:
        DrawLine(batch);
        break;
    case GS_PRIM_POINT:
    {
        const GSVertex &v = batch.vertices[0];
        const auto &ctx = state.context;
        int px = static_cast<int>(v.x) - (ctx.xyoffset.ofx >> 4);
        int py = static_cast<int>(v.y) - (ctx.xyoffset.ofy >> 4);
        WritePixel(state, px, py, static_cast<u32>(v.z), v.r, v.g, v.b, v.a, v.fog);
        break;
    }
    default:
        break;
    }
}

void GSCpuBackend::WritePixel(const GSDrawState &state, int x, int y, int z, uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog)
{
    const bool depthAlreadyPassed=t_depthAlreadyPassed;
    t_depthAlreadyPassed=false;

    const auto &ctx = state.context;
    if (x < ctx.scissor.x0 || x > ctx.scissor.x1 || y < ctx.scissor.y0 || y > ctx.scissor.y1)
        return;
    if (!RowInBand(y))
        return;

    if (state.prim.fge)
    {
        const uint32_t inverseFog = 255u - fog;
        auto applyFog = [&](uint8_t input, uint8_t fogColor) -> uint8_t
        {
            return static_cast<uint8_t>(((static_cast<uint32_t>(fog) * input) >> 8) + ((inverseFog * fogColor) >> 8));
        };

        r = applyFog(r, state.fogR);
        g = applyFog(g, state.fogG);
        b = applyFog(b, state.fogB);
    }

    const u32 fbp = GSInternal::framePageBaseToBlock(ctx.frame.fbp);
    const u32 fbw = std::max<u32>(ctx.frame.fbw, 1u);
    const u32 fpsm = ctx.frame.psm;
    const u32 zbp = GSInternal::framePageBaseToBlock(ctx.zbuf.zbp);
    const u32 zpsm = ctx.zbuf.psm;
    // The GS clamps depth to the Z buffer format's range (Z24: 0xFFFFFF,
    // Z16/Z16S: 0xFFFF) instead of truncating it; near-camera geometry has Z
    // far above those limits and would otherwise wrap to 'far'.
    {
        uint32_t zu = static_cast<uint32_t>(z);
        const uint32_t zfmt = zpsm & 0xFu;
        const uint32_t zLimit = zfmt == 0x1u ? 0xFFFFFFu : (zfmt == 0x2u || zfmt == 0xAu) ? 0xFFFFu : 0xFFFFFFFFu;
        if (zu > zLimit)
            zu = zLimit;
        z = static_cast<int>(zu);
    }

    {
        const uint32_t fbpIndex = ctx.frame.fbp & 0x1FFu;
        const uint16_t drawnTo = static_cast<uint16_t>(ctx.scissor.y1 + 1u);
        uint16_t seen = g_ps2xFbDrawnHeight[fbpIndex].load(std::memory_order_relaxed);
        while (drawnTo > seen &&
               !g_ps2xFbDrawnHeight[fbpIndex].compare_exchange_weak(seen, drawnTo, std::memory_order_relaxed))
        {
        }
    }

    const PixelWriteMask writeMask = classifyAlphaTest(ctx.test, a, static_cast<uint8_t>(fpsm));
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    // [gs:moviepx] — per-pixel verdicts for the drive-in movie-screen tris
    // (texture at tbp0 0x3BA0): why do they end up invisible?
    struct MoviePxStats { uint32_t total, failAlpha, failDate, failZ, written; uint32_t lastZ, lastStoredZ, lastA, lastDest; };
    thread_local MoviePxStats s_mp{};
    static std::atomic<uint32_t> s_mpReports{0u};
    const bool isMoviePx = (ctx.tex0.tbp0 == 0x3BA0u) && state.prim.tme;
    auto mpReport = [&]()
    {
        if (isMoviePx && (++s_mp.total % 50000u) == 0u && s_mpReports.fetch_add(1u, std::memory_order_relaxed) < 40u)
            std::fprintf(stderr, "[gs:moviepx] total=%u failAlpha=%u failDate=%u failZ=%u written=%u lastZ=0x%x lastStoredZ=0x%x lastA=%u lastDest=0x%08x rgb=(%u,%u,%u)\n",
                         s_mp.total, s_mp.failAlpha, s_mp.failDate, s_mp.failZ, s_mp.written, s_mp.lastZ, s_mp.lastStoredZ, s_mp.lastA, s_mp.lastDest, r, g, b);
    };
    mpReport();
#else
    const bool isMoviePx = false;
    auto mpReport = [&]() {};
#endif
    if (!writeMask.writesAnything())
    {
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
        if (isMoviePx) { ++s_mp.failAlpha; s_mp.lastA = a; }
#endif
        return;
    }

    const uint32_t ztestMethod = static_cast<uint32_t>((ctx.test >> 17) & 3u);
    const bool alphaBlendEnabled = state.prim.abe;
    const bool preserveDestinationAlpha = writeMask.writeRgb && !writeMask.writeAlpha && fpsm == GS_PSM_CT32;
    const bool destinationAlphaTestNeedsRead = ((ctx.test >> 14) & 0x1u) != 0u && (fpsm == GS_PSM_CT32 || fpsm == GS_PSM_CT16 || fpsm == GS_PSM_CT16S);

    // small optimization, avoid reading the framebuffer for simple draws
    // TODO: only one address lookup for rmw
    const bool frmw = destinationAlphaTestNeedsRead || (writeMask.writesFramebuffer() && ((ctx.frame.fbmsk != 0) || alphaBlendEnabled || preserveDestinationAlpha));

    u32 rawFramebufferPixel = 0;
    u32 fbrgba = 0;
    if (frmw)
    {
        rawFramebufferPixel = ReadVramUnlocked(fpsm, fbp, fbw, x, y);
        fbrgba = rawFramebufferPixel;

        if (bitsPerPixel(fpsm) == 16)
        {
            fbrgba = Rgba5551ToRgba8888(fbrgba);
        }
        else if (fpsm == GS_PSM_CT24)
        {
            // The GS supplies 0x80 as destination alpha for RGB24 blending.
            fbrgba |= 0x80000000u;
        }
    }

    if (!passesDestinationAlphaTest(ctx.test, static_cast<uint8_t>(fpsm), rawFramebufferPixel))
    {
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
        if (isMoviePx) { ++s_mp.failDate; s_mp.lastDest = rawFramebufferPixel; }
#endif
        return;
    }

    bool zpass = depthAlreadyPassed;
    uint32_t storedZ = 0u;
    if (!depthAlreadyPassed) switch (ztestMethod)
    {
    case 0:
        zpass = false;
        break;
    case 1:
        zpass = true;
        break;
    case 2:
        storedZ = ReadVramUnlocked(zpsm, zbp, fbw, x, y);
        zpass = static_cast<uint32_t>(z) >= storedZ;
        break;
    case 3:
        storedZ = ReadVramUnlocked(zpsm, zbp, fbw, x, y);
        zpass = static_cast<uint32_t>(z) > storedZ;
        break;
    }

    if (!zpass)
    {
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
        if (isMoviePx) { ++s_mp.failZ; s_mp.lastZ = static_cast<uint32_t>(z); s_mp.lastStoredZ = storedZ; }
#endif
        return;
    }
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    if (isMoviePx) { ++s_mp.written; s_mp.lastZ = static_cast<uint32_t>(z); s_mp.lastStoredZ = storedZ; s_mp.lastA = a; }
#endif
    (void)mpReport;

    if (writeMask.writesFramebuffer())
    {
        const u8 srcR = r;
        const u8 srcG = g;
        const u8 srcB = b;

        if (state.prim.abe)
        {
            uint8_t dr = fbrgba & 0xFF;
            uint8_t dg = (fbrgba >> 8) & 0xFF;
            uint8_t db = (fbrgba >> 16) & 0xFF;
            uint8_t da = (fbrgba >> 24) & 0xFF;

            // PABE disables alpha blending when the source alpha MSB is clear.
            if (!(state.pabe && (a & 0x80u) == 0u))
            {
                uint64_t alphaReg = ctx.alpha;
                uint8_t asel = alphaReg & 3;
                uint8_t bsel = (alphaReg >> 2) & 3;
                uint8_t csel = (alphaReg >> 4) & 3;
                uint8_t dsel = (alphaReg >> 6) & 3;
                uint8_t fix = static_cast<uint8_t>((alphaReg >> 32) & 0xFF);

                auto pickRGB = [&](uint8_t sel, int cs, int cd) -> int
                {
                    if (sel == 0)
                        return cs;
                    if (sel == 1)
                        return cd;
                    return 0;
                };
                int cAlpha = (csel == 0) ? a : (csel == 1) ? da
                                                           : fix;

                r = clampU8(((pickRGB(asel, r, dr) - pickRGB(bsel, r, dr)) * cAlpha >> 7) + pickRGB(dsel, r, dr));
                g = clampU8(((pickRGB(asel, g, dg) - pickRGB(bsel, g, dg)) * cAlpha >> 7) + pickRGB(dsel, g, dg));
                b = clampU8(((pickRGB(asel, b, db) - pickRGB(bsel, b, db)) * cAlpha >> 7) + pickRGB(dsel, b, db));
            }
            else
            {
                r = srcR;
                g = srcG;
                b = srcB;
            }
        }

        if (writeMask.writeAlpha && (ctx.fba & 0x1ull) != 0ull && ctx.frame.psm != GS_PSM_CT24)
        {
            a = static_cast<uint8_t>(a | 0x80u);
        }

        u32 pixel = pack32(r, g, b, a);

        if (ctx.frame.fbmsk != 0)
        {
            pixel = (pixel & ~ctx.frame.fbmsk) | (fbrgba & ctx.frame.fbmsk);
        }

        if (preserveDestinationAlpha)
        {
            pixel = (pixel & 0x00FFFFFFu) | (fbrgba & 0xFF000000u);
        }

        // format conversion
        if (bitsPerPixel(fpsm) == 16)
        {
            pixel = Rgba8888ToRgba5551(pixel);
        }

        WriteVramUnlocked(fpsm, fbp, fbw, x, y, pixel);
    }

    if (writeMask.writeDepth && !ctx.zbuf.zmask)
    {
        WriteVramUnlocked(zpsm, zbp, fbw, x, y, z);
    }
}

uint32_t GSCpuBackend::LookupCLUT(const GSDrawState &state,
                                  uint8_t index,
                                  uint32_t cbp,
                                  uint8_t cpsm,
                                  uint8_t csm,
                                  uint8_t csa,
                                  uint8_t sourcePsm)
{
    const uint32_t clutIndex = resolveClutIndex(index, cpsm, csm, csa, sourcePsm);
    const uint32_t clutWidth = (state.texclut.cbw != 0u) ? static_cast<uint32_t>(state.texclut.cbw) : 1u;
    const uint32_t clutX = static_cast<uint32_t>(state.texclut.cou) + (clutIndex & 0x0Fu);
    const uint32_t clutY = static_cast<uint32_t>(state.texclut.cov) + (clutIndex >> 4);

    switch (cpsm)
    {
    case GS_PSM_CT32:
        return applyTexa(state.texa, cpsm, GSMem::ReadCT32(m_vram, cbp, clutWidth, clutX, clutY));
    case GS_PSM_CT24:
        return applyTexa(state.texa, cpsm, GSMem::ReadCT24(m_vram, cbp, clutWidth, clutX, clutY));
    case GS_PSM_CT16:
        return applyTexa(state.texa, cpsm, Rgba5551ToRgba8888(GSMem::ReadCT16(m_vram, cbp, clutWidth, clutX, clutY)));
    case GS_PSM_CT16S:
        return applyTexa(state.texa, cpsm, Rgba5551ToRgba8888(GSMem::ReadCT16S(m_vram, cbp, clutWidth, clutX, clutY)));
    default:
        break;
    }

    return 0xFFFF00FFu;
}

void GSCpuBackend::DecodeTextureRgba(const GSDrawState &state, uint32_t width, uint32_t height, uint8_t *outRgba)
{
    // Sample at texel centres with filtering and region clamping disabled: this
    // produces the raw texture. The GPU applies filtering and wrapping itself,
    // and a CLAMP region is handled in the shader from the same registers.
    GSDrawState raw = state;
    raw.prim.fst = true;
    raw.linearFilter = false;
    raw.context.clamp = 0u;
    raw.textureWidth = static_cast<uint16_t>(width);
    raw.textureHeight = static_cast<uint16_t>(height);

    // Transfers are applied by the raster worker, so local memory is only
    // settled once the queue drains. ReadVram waits for the same reason;
    // without it a texture decoded while its upload was still in flight came
    // out as noise (menu words, character select thumbnails).
    WaitForRasterIdle();
    std::lock_guard<std::mutex> lock(m_mutex);
    for (uint32_t y = 0; y < height; ++y)
    {
        uint8_t *row = outRgba + static_cast<size_t>(y) * width * 4u;
        for (uint32_t x = 0; x < width; ++x)
        {
            const uint32_t texel = SampleTexture(raw, 0.0f, 0.0f, 1.0f, static_cast<uint16_t>(x * 16u),
                                                 static_cast<uint16_t>(y * 16u));
            uint8_t *dst = row + static_cast<size_t>(x) * 4u;
            dst[0] = static_cast<uint8_t>(texel & 0xFFu);
            dst[1] = static_cast<uint8_t>((texel >> 8) & 0xFFu);
            dst[2] = static_cast<uint8_t>((texel >> 16) & 0xFFu);
            dst[3] = static_cast<uint8_t>((texel >> 24) & 0xFFu);
        }
    }
}

uint32_t GSCpuBackend::SampleTexture(const GSDrawState &state, float s, float t, float q, uint16_t u, uint16_t v)
{
    const auto &ctx = state.context;
    const auto &tex = ctx.tex0;

    const int texW = state.textureWidth;
    const int texH = state.textureHeight;
    const uint64_t clamp = ctx.clamp;
    const uint8_t wrapU = static_cast<uint8_t>(clamp & 0x3u);
    const uint8_t wrapV = static_cast<uint8_t>((clamp >> 2) & 0x3u);
    const uint16_t minU = static_cast<uint16_t>((clamp >> 4) & 0x3FFu);
    const uint16_t maxU = static_cast<uint16_t>((clamp >> 14) & 0x3FFu);
    const uint16_t minV = static_cast<uint16_t>((clamp >> 24) & 0x3FFu);
    const uint16_t maxV = static_cast<uint16_t>((clamp >> 34) & 0x3FFu);

    float texUf, texVf;
    if (state.prim.fst)
    {
        texUf = static_cast<float>(u) / 16.0f;
        texVf = static_cast<float>(v) / 16.0f;
    }
    else
    {
        const float invQ = 1.0f / fabsQ(q);
        texUf = s * invQ * static_cast<float>(texW);
        texVf = t * invQ * static_cast<float>(texH);
    }

    // Defensive: malformed VU1 output (or an un-transformed prim) can hand us
    // non-finite or wildly out-of-range S/T/Q here. Casting those to int is UB
    // and downstream VRAM addressing would read out of bounds -> crash. Clamp to
    // a sane texel window so a bad primitive just samples garbage-but-safe.
    constexpr float kTexelCoordLimit = 1.0e6f;
    if (!std::isfinite(texUf) || texUf > kTexelCoordLimit || texUf < -kTexelCoordLimit)
        texUf = 0.0f;
    if (!std::isfinite(texVf) || texVf > kTexelCoordLimit || texVf < -kTexelCoordLimit)
        texVf = 0.0f;

    auto samplePoint = [&](int sampleU, int sampleV) -> uint32_t
    {
        sampleU = wrapTextureCoordinate(sampleU, texW, wrapU, minU, maxU);
        sampleV = wrapTextureCoordinate(sampleV, texH, wrapV, minV, maxV);

        u32 out = ReadVramUnlocked(tex.psm, tex.tbp0, tex.tbw, sampleU, sampleV);

        switch (tex.psm)
        {
        case GS_PSM_CT32:
        case GS_PSM_Z32:
        case GS_PSM_CT24:
        case GS_PSM_Z24:
            return applyTexa(state.texa, tex.psm, out);
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
            return applyTexa(state.texa, tex.psm, Rgba5551ToRgba8888(out));
        case GS_PSM_T8:
        case GS_PSM_T8H:
        case GS_PSM_T4:
        case GS_PSM_T4HL:
        case GS_PSM_T4HH:
            return LookupCLUT(state, static_cast<u8>(out), tex.cbp, tex.cpsm, tex.csm, tex.csa, tex.psm);
        }

        return 0xFFFF00FFu;
    };

    if (!state.linearFilter)
    {
        return samplePoint(static_cast<int>(texUf), static_cast<int>(texVf));
    }

    const float sampleU = texUf - 0.5f;
    const float sampleV = texVf - 0.5f;
    const int u0 = static_cast<int>(std::floor(sampleU));
    const int v0 = static_cast<int>(std::floor(sampleV));
    const int u1 = u0 + 1;
    const int v1 = v0 + 1;
    const float fx = sampleU - static_cast<float>(u0);
    const float fy = sampleV - static_cast<float>(v0);

    const uint32_t c00 = samplePoint(u0, v0);
    const uint32_t c10 = samplePoint(u1, v0);
    const uint32_t c01 = samplePoint(u0, v1);
    const uint32_t c11 = samplePoint(u1, v1);

#if defined(_M_X64) || defined(__SSE2__)
    static const bool vectorFilter=[] {const char *p=std::getenv("PS2X_GS_SIMD_FILTER");return !p || p[0]!='0';}();
    if (vectorFilter) return lerpRgbaSimd(c00,c10,c01,c11,fx,fy);
#endif

    const uint8_t r = lerpChannel(static_cast<uint8_t>(c00 & 0xFFu),
                                  static_cast<uint8_t>(c10 & 0xFFu),
                                  static_cast<uint8_t>(c01 & 0xFFu),
                                  static_cast<uint8_t>(c11 & 0xFFu),
                                  fx, fy);
    const uint8_t g = lerpChannel(static_cast<uint8_t>((c00 >> 8) & 0xFFu),
                                  static_cast<uint8_t>((c10 >> 8) & 0xFFu),
                                  static_cast<uint8_t>((c01 >> 8) & 0xFFu),
                                  static_cast<uint8_t>((c11 >> 8) & 0xFFu),
                                  fx, fy);
    const uint8_t b = lerpChannel(static_cast<uint8_t>((c00 >> 16) & 0xFFu),
                                  static_cast<uint8_t>((c10 >> 16) & 0xFFu),
                                  static_cast<uint8_t>((c01 >> 16) & 0xFFu),
                                  static_cast<uint8_t>((c11 >> 16) & 0xFFu),
                                  fx, fy);
    const uint8_t a = lerpChannel(static_cast<uint8_t>((c00 >> 24) & 0xFFu),
                                  static_cast<uint8_t>((c10 >> 24) & 0xFFu),
                                  static_cast<uint8_t>((c01 >> 24) & 0xFFu),
                                  static_cast<uint8_t>((c11 >> 24) & 0xFFu),
                                  fx, fy);

    return static_cast<uint32_t>(r) |
           (static_cast<uint32_t>(g) << 8) |
           (static_cast<uint32_t>(b) << 16) |
           (static_cast<uint32_t>(a) << 24);
}

void GSCpuBackend::DrawSprite(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const auto &ctx = state.context;

    int ofx = ctx.xyoffset.ofx >> 4;
    int ofy = ctx.xyoffset.ofy >> 4;

    int x0 = static_cast<int>(v0.x) - ofx;
    int y0 = static_cast<int>(v0.y) - ofy;
    int x1 = static_cast<int>(v1.x) - ofx;
    int y1 = static_cast<int>(v1.y) - ofy;
    u32 z1 = static_cast<u32>(v1.z);

    if (x0 > x1)
        std::swap(x0, x1);
    if (y0 > y1)
        std::swap(y0, y1);

    const int unclippedX0 = x0;
    const int unclippedY0 = y0;
    const int spanX = std::max(1, x1 - x0);
    const int spanY = std::max(1, y1 - y0);
    const int unclippedX1 = unclippedX0 + spanX - 1;
    const int unclippedY1 = unclippedY0 + spanY - 1;

    // If the sprite rectangle is fully outside scissor, nothing should render.
    if (unclippedX1 < ctx.scissor.x0 || unclippedX0 > ctx.scissor.x1 ||
        unclippedY1 < ctx.scissor.y0 || unclippedY0 > ctx.scissor.y1)
        return;

    const int drawX0 = clampInt(unclippedX0, ctx.scissor.x0, ctx.scissor.x1);
    const int drawY0 = clampInt(unclippedY0, ctx.scissor.y0, ctx.scissor.y1);
    const int drawX1 = clampInt(unclippedX1, ctx.scissor.x0, ctx.scissor.x1);
    const int drawY1 = clampInt(unclippedY1, ctx.scissor.y0, ctx.scissor.y1);

    const uint64_t alphaReg = ctx.alpha;
    const uint8_t alphaMode = static_cast<uint8_t>(alphaReg & 0xFFu);
    const uint8_t alphaFix = static_cast<uint8_t>((alphaReg >> 32) & 0xFFu);

    uint8_t r = v1.r, g = v1.g, b = v1.b, a = v1.a;

    if (state.prim.tme)
    {
        const auto &tex = ctx.tex0;
        const int texW = state.textureWidth;
        const int texH = state.textureHeight;

        float u0f, v0f, u1f, v1f;
        if (state.prim.fst)
        {
            u0f = static_cast<float>(v0.u >> 4);
            v0f = static_cast<float>(v0.v >> 4);
            u1f = static_cast<float>(v1.u >> 4);
            v1f = static_cast<float>(v1.v >> 4);
        }
        else
        {
            const float q0 = fabsQ(v0.q);
            const float q1 = fabsQ(v1.q);
            u0f = (v0.s / q0) * static_cast<float>(texW);
            v0f = (v0.t / q0) * static_cast<float>(texH);
            u1f = (v1.s / q1) * static_cast<float>(texW);
            v1f = (v1.t / q1) * static_cast<float>(texH);
        }

        float spriteW = static_cast<float>(spanX);
        float spriteH = static_cast<float>(spanY);
        if (spriteW < 1.0f)
            spriteW = 1.0f;
        if (spriteH < 1.0f)
            spriteH = 1.0f;

        for (int y = drawY0; y <= drawY1; ++y)
        {
            if (!RowInBand(y))
                continue;
            float ty = (static_cast<float>(y - unclippedY0) + 0.5f) / spriteH;
            float texVf = v0f + (v1f - v0f) * ty;

            for (int x = drawX0; x <= drawX1; ++x)
            {
                float tx = (static_cast<float>(x - unclippedX0) + 0.5f) / spriteW;
                float texUf = u0f + (u1f - u0f) * tx;
                uint32_t texel = 0xFFFF00FFu;
                if (state.prim.fst)
                {
                    const int fixedU = static_cast<int>((texUf * 16.0f) + 0.5f);
                    const int fixedV = static_cast<int>((texVf * 16.0f) + 0.5f);
                    const uint16_t sampleU = static_cast<uint16_t>(clampInt(fixedU, 0, 0xFFFF));
                    const uint16_t sampleV = static_cast<uint16_t>(clampInt(fixedV, 0, 0xFFFF));
                    texel = SampleTexture(state, 0.0f, 0.0f, 1.0f, sampleU, sampleV);
                }
                else
                {
                    texel = SampleTexture(state, texUf / static_cast<float>(texW), texVf / static_cast<float>(texH), 1.0f, 0u, 0u);
                }

                uint8_t tr = static_cast<uint8_t>(texel & 0xFF);
                uint8_t tg = static_cast<uint8_t>((texel >> 8) & 0xFF);
                uint8_t tb = static_cast<uint8_t>((texel >> 16) & 0xFF);
                uint8_t ta = static_cast<uint8_t>((texel >> 24) & 0xFF);

                const TextureCombineResult color = combineTexture(tex, r, g, b, a, tr, tg, tb, ta);
                WritePixel(state, x, y, z1, color.r, color.g, color.b, color.a, v1.fog);
            }
        }
    }
    else
    {
        for (int y = drawY0; y <= drawY1; ++y)
        {
            if (!RowInBand(y))
                continue;
            for (int x = drawX0; x <= drawX1; ++x)
                WritePixel(state, x, y, z1, r, g, b, a, v1.fog);
        }
    }
}

void GSCpuBackend::DrawTriangle(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const GSVertex &v2 = batch.vertices[2];
    const auto &ctx = state.context;

    int ofx = ctx.xyoffset.ofx >> 4;
    int ofy = ctx.xyoffset.ofy >> 4;

    float fx0 = v0.x - static_cast<float>(ofx);
    float fy0 = v0.y - static_cast<float>(ofy);
    float fx1 = v1.x - static_cast<float>(ofx);
    float fy1 = v1.y - static_cast<float>(ofy);
    float fx2 = v2.x - static_cast<float>(ofx);
    float fy2 = v2.y - static_cast<float>(ofy);

    int minX = static_cast<int>(std::floor(std::min({fx0, fx1, fx2})));
    int maxX = static_cast<int>(std::ceil(std::max({fx0, fx1, fx2})));
    int minY = static_cast<int>(std::floor(std::min({fy0, fy1, fy2})));
    int maxY = static_cast<int>(std::ceil(std::max({fy0, fy1, fy2})));

    minX = clampInt(minX, ctx.scissor.x0, ctx.scissor.x1);
    maxX = clampInt(maxX, ctx.scissor.x0, ctx.scissor.x1);
    minY = clampInt(minY, ctx.scissor.y0, ctx.scissor.y1);
    maxY = clampInt(maxY, ctx.scissor.y0, ctx.scissor.y1);

    float denom = (fy1 - fy2) * (fx0 - fx2) + (fx2 - fx1) * (fy0 - fy2);
    if (std::fabs(denom) < 0.001f)
        return;

    const float winding = (denom < 0.0f) ? -1.0f : 1.0f;
    const float invAbsDenom = 1.0f / std::fabs(denom);
    constexpr float kEdgeEpsilon = 1.0e-4f;

    for (int y = minY; y <= maxY; ++y)
    {
        if (!RowInBand(y))
            continue;
        float py = static_cast<float>(y) + 0.5f;
        for (int x = minX; x <= maxX; ++x)
        {
            float px = static_cast<float>(x) + 0.5f;

            float w0 = (((fy1 - fy2) * (px - fx2) + (fx2 - fx1) * (py - fy2)) * winding) * invAbsDenom;
            float w1 = (((fy2 - fy0) * (px - fx2) + (fx0 - fx2) * (py - fy2)) * winding) * invAbsDenom;
            float w2 = 1.0f - w0 - w1;

            if (w0 < -kEdgeEpsilon || w1 < -kEdgeEpsilon || w2 < -kEdgeEpsilon)
                continue;

            double z = v0.z * w0 + v1.z * w1 + v2.z * w2;

            // Texture reads cannot rescue a fragment that fails the unchanged
            // depth test. Reject before shading, retaining presentation metadata.
            static const bool earlyDepth=[] {const char *p=std::getenv("PS2X_GS_EARLY_DEPTH");return !p || p[0]!='0';}();
            const uint32_t depthMethod=static_cast<uint32_t>((ctx.test>>17)&3u);
            if (earlyDepth && state.prim.tme && depthMethod>=2u)
            {
                uint32_t depth=static_cast<u32>(z+0.5);
                const uint32_t fmt=ctx.zbuf.psm&0xfu;
                const uint32_t limit=fmt==1u?0xffffffu:(fmt==2u || fmt==0xau)?0xffffu:0xffffffffu;
                depth=std::min(depth,limit);
                const uint32_t stored=ReadVramUnlocked(ctx.zbuf.psm,
                    GSInternal::framePageBaseToBlock(ctx.zbuf.zbp),
                    std::max<u32>(ctx.frame.fbw,1u),x,y);
                if (depth<stored || (depthMethod==3u && depth==stored))
                {
                    const uint32_t page=ctx.frame.fbp&0x1ffu;
                    const uint16_t height=static_cast<uint16_t>(ctx.scissor.y1+1u);
                    uint16_t seen=g_ps2xFbDrawnHeight[page].load(std::memory_order_relaxed);
                    while(height>seen && !g_ps2xFbDrawnHeight[page].compare_exchange_weak(seen,height,std::memory_order_relaxed)) {}
                    continue;
                }
            }

            uint8_t r, g, b, a;
            if (state.prim.iip)
            {
                r = clampU8(static_cast<int>(v0.r * w0 + v1.r * w1 + v2.r * w2));
                g = clampU8(static_cast<int>(v0.g * w0 + v1.g * w1 + v2.g * w2));
                b = clampU8(static_cast<int>(v0.b * w0 + v1.b * w1 + v2.b * w2));
                a = clampU8(static_cast<int>(v0.a * w0 + v1.a * w1 + v2.a * w2));
            }
            else
            {
                r = v2.r;
                g = v2.g;
                b = v2.b;
                a = v2.a;
            }

            if (state.prim.tme)
            {
                float is, it, iq;
                uint16_t iu, iv;
                if (state.prim.fst)
                {
                    iu = static_cast<uint16_t>(v0.u * w0 + v1.u * w1 + v2.u * w2);
                    iv = static_cast<uint16_t>(v0.v * w0 + v1.v * w1 + v2.v * w2);
                    is = 0.0f;
                    it = 0.0f;
                    iq = 1.0f;
                }
                else
                {
                    // The GS DDA interpolates the homogeneous S, T and Q
                    // values. Texel coordinates are calculated from S/Q and
                    // T/Q only after interpolation.
                    is = v0.s * w0 + v1.s * w1 + v2.s * w2;
                    it = v0.t * w0 + v1.t * w1 + v2.t * w2;
                    iq = v0.q * w0 + v1.q * w1 + v2.q * w2;
                    iu = 0;
                    iv = 0;
                }

                uint32_t texel = SampleTexture(state, is, it, iq, iu, iv);

                uint8_t tr = static_cast<uint8_t>(texel & 0xFF);
                uint8_t tg = static_cast<uint8_t>((texel >> 8) & 0xFF);
                uint8_t tb = static_cast<uint8_t>((texel >> 16) & 0xFF);
                uint8_t ta = static_cast<uint8_t>((texel >> 24) & 0xFF);

                const auto &tex = ctx.tex0;
                const uint8_t shadeR = r;
                const uint8_t shadeG = g;
                const uint8_t shadeB = b;
                const uint8_t shadeA = a;
                const TextureCombineResult color = combineTexture(tex, shadeR, shadeG, shadeB, shadeA, tr, tg, tb, ta);

                r = color.r;
                g = color.g;
                b = color.b;
                a = color.a;
            }

            const uint8_t fog = clampU8(static_cast<int>(v0.fog * w0 + v1.fog * w1 + v2.fog * w2));
            // Sampling and shading above only read memory; band-safe draws
            // cannot modify this pixel's depth from another raster thread.
            t_depthAlreadyPassed=earlyDepth && state.prim.tme && depthMethod>=2u;
            WritePixel(state, x, y, static_cast<u32>(z + 0.5), r, g, b, a, fog);
        }
    }
}

void GSCpuBackend::DrawLine(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    const auto &ctx = state.context;

    int ofx = ctx.xyoffset.ofx >> 4;
    int ofy = ctx.xyoffset.ofy >> 4;

    int x0 = static_cast<int>(v0.x) - ofx;
    int y0 = static_cast<int>(v0.y) - ofy;
    int x1 = static_cast<int>(v1.x) - ofx;
    int y1 = static_cast<int>(v1.y) - ofy;

    int dx = std::abs(x1 - x0);
    int dy = -std::abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int err = dx + dy;

    int totalSteps = std::max(std::abs(x1 - x0), std::abs(y1 - y0));
    if (totalSteps == 0)
        totalSteps = 1;
    int step = 0;

    for (;;)
    {
        float t = static_cast<float>(step) / static_cast<float>(totalSteps);
        uint8_t r, g, b, a;
        if (state.prim.iip)
        {
            r = clampU8(static_cast<int>(v0.r + (v1.r - v0.r) * t));
            g = clampU8(static_cast<int>(v0.g + (v1.g - v0.g) * t));
            b = clampU8(static_cast<int>(v0.b + (v1.b - v0.b) * t));
            a = clampU8(static_cast<int>(v0.a + (v1.a - v0.a) * t));
        }
        else
        {
            r = v1.r;
            g = v1.g;
            b = v1.b;
            a = v1.a;
        }

        double z = (v0.z + (v1.z - v0.z) * t);
        const uint8_t fog = clampU8(static_cast<int>(v0.fog + (v1.fog - v0.fog) * t));
        WritePixel(state, x0, y0, static_cast<u32>(z), r, g, b, a, fog);

        if (x0 == x1 && y0 == y1)
            break;

        int e2 = 2 * err;
        if (e2 >= dy)
        {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx)
        {
            err += dx;
            y0 += sy;
        }
        ++step;
    }
}

void GSCpuBackend::BeginTransferUnlocked(const GSTransferCommand &command)
{
    m_transfer = command;
    m_transferState.x = command.trxpos.dsax;
    m_transferState.y = command.trxpos.dsay;
    m_transferState.totalPixels = static_cast<uint32_t>(command.trxreg.rrw) * static_cast<uint32_t>(command.trxreg.rrh);
    m_transferState.copiedPixels = 0u;
    m_transferState.direction = command.direction;
    m_transferState.localToHostPendingBytes = 0u;
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    {
        // [gs:trx] — every TRXDIR-started transfer regardless of which path
        // (native chain/packet fast paths or the generic GIF parser) issued it.
        static std::atomic<uint32_t> s_trx{0u};
        if (s_trx.fetch_add(1u, std::memory_order_relaxed) < 600u)
            std::fprintf(stderr, "[gs:trx] dir=%u sbp=0x%x spsm=0x%x dbp=0x%x dbw=%u dpsm=0x%x dst=(%u,%u) size=%ux%u\n",
                         command.direction, command.bitbltbuf.sbp, command.bitbltbuf.spsm,
                         command.bitbltbuf.dbp, command.bitbltbuf.dbw, command.bitbltbuf.dpsm,
                         command.trxpos.dsax, command.trxpos.dsay, command.trxreg.rrw, command.trxreg.rrh);
    }
#endif

    if (command.direction == 2u)
        PerformLocalToLocalTransfer();
    else if (command.direction == 1u)
        PerformLocalToHostTransfer();
}

void GSCpuBackend::UploadImageUnlocked(const uint8_t *data, uint32_t sizeBytes)
{
    if (!data || sizeBytes == 0u || !m_vram || m_transferState.direction != 0u)
        return;
    if (m_transfer.trxreg.rrw == 0u || m_transfer.trxreg.rrh == 0u || m_transferState.totalPixels == 0u)
        return;

    const uint32_t dbp = m_transfer.bitbltbuf.dbp;
    const uint32_t dbw = std::max<uint32_t>(m_transfer.bitbltbuf.dbw, 1u);
    const uint8_t dpsm = m_transfer.bitbltbuf.dpsm;
    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t dsax = m_transfer.trxpos.dsax;
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    if (dbp == 0x3BA0u)
    {
        // [gs:movieblk] pixel content of the movie-screen macroblock uploads.
        static std::atomic<uint32_t> s_mb{0u};
        const uint32_t n = s_mb.fetch_add(1u, std::memory_order_relaxed);
        if (n < 40u || (n % 512u) == 0u)
        {
            uint32_t nonBlack = 0u;
            for (uint32_t i = 0; i + 3u < sizeBytes; i += 4u)
                if (data[i] | data[i + 1u] | data[i + 2u])
                    ++nonBlack;
            std::fprintf(stderr, "[gs:movieblk] #%u dst=(%u,%u) bytes=%u nonBlackPx=%u first=%02x%02x%02x%02x\n",
                         n, dsax, m_transfer.trxpos.dsay, sizeBytes, nonBlack, data[0], data[1], data[2], data[3]);
        }
    }
#endif
    uint32_t offset = 0u;

    auto advancePixel = [&](uint32_t count)
    {
        const uint32_t totalPixels = m_transferState.totalPixels;
        m_transferState.copiedPixels =
            std::min<uint32_t>(totalPixels, m_transferState.copiedPixels + count);

        if (m_transferState.copiedPixels >= totalPixels)
        {
            m_transferState.direction = 3u;
            m_transferState.totalPixels = 0u;
            return;
        }

        m_transferState.x = dsax + (m_transferState.copiedPixels % rrw);
        m_transferState.y = m_transfer.trxpos.dsay + (m_transferState.copiedPixels / rrw);
    };

    while (offset < sizeBytes && m_transferState.direction == 0u)
    {
        switch (dpsm)
        {
        case GS_PSM_CT32:
        case GS_PSM_Z32:
        {
            if (sizeBytes - offset < 4u)
                return;
            uint32_t value = 0u;
            std::memcpy(&value, data + offset, sizeof(value));
            WriteVramUnlocked(dpsm, dbp, dbw, m_transferState.x, m_transferState.y, value);
            offset += 4u;
            advancePixel(1u);
            break;
        }
        case GS_PSM_CT24:
        case GS_PSM_Z24:
        {
            if (sizeBytes - offset < 3u)
                return;
            const uint32_t value = static_cast<uint32_t>(data[offset]) |
                                   (static_cast<uint32_t>(data[offset + 1u]) << 8u) |
                                   (static_cast<uint32_t>(data[offset + 2u]) << 16u);
            WriteVramUnlocked(dpsm, dbp, dbw, m_transferState.x, m_transferState.y, value);
            offset += 3u;
            advancePixel(1u);
            break;
        }
        case GS_PSM_CT16:
        case GS_PSM_CT16S:
        case GS_PSM_Z16:
        case GS_PSM_Z16S:
        {
            if (sizeBytes - offset < 2u)
                return;
            uint16_t value = 0u;
            std::memcpy(&value, data + offset, sizeof(value));
            WriteVramUnlocked(dpsm, dbp, dbw, m_transferState.x, m_transferState.y, value);
            offset += 2u;
            advancePixel(1u);
            break;
        }
        case GS_PSM_T8:
        case GS_PSM_T8H:
            WriteVramUnlocked(dpsm, dbp, dbw, m_transferState.x, m_transferState.y, data[offset++]);
            advancePixel(1u);
            break;
        case GS_PSM_T4:
        case GS_PSM_T4HL:
        case GS_PSM_T4HH:
        {
            const uint8_t packed = data[offset++];
            const uint32_t firstPixel = m_transferState.copiedPixels;
            WriteVramUnlocked(dpsm, dbp, dbw,
                              dsax + (firstPixel % rrw),
                              m_transfer.trxpos.dsay + (firstPixel / rrw),
                              packed & 0x0Fu);
            if (firstPixel + 1u < m_transferState.totalPixels)
            {
                const uint32_t secondPixel = firstPixel + 1u;
                WriteVramUnlocked(dpsm, dbp, dbw,
                                  dsax + (secondPixel % rrw),
                                  m_transfer.trxpos.dsay + (secondPixel / rrw),
                                  (packed >> 4u) & 0x0Fu);
            }
            advancePixel(std::min<uint32_t>(2u, m_transferState.totalPixels - firstPixel));
            break;
        }
        default:
            return;
        }
    }
}

void GSCpuBackend::PerformLocalToLocalTransfer()
{
    if (!m_vram)
        return;

    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t rrh = m_transfer.trxreg.rrh;
    const uint32_t total = rrw * rrh;
    if (total == 0u)
    {
        m_transferState.direction = 3u;
        return;
    }

    for (uint32_t pixel = 0; pixel < total; ++pixel)
    {
        uint32_t x = pixel % rrw;
        uint32_t y = pixel / rrw;
        if ((m_transfer.trxpos.dir & 0x2u) != 0u)
            x = rrw - x - 1u;
        if ((m_transfer.trxpos.dir & 0x1u) != 0u)
            y = rrh - y - 1u;

        const uint32_t value = ReadVramUnlocked(m_transfer.bitbltbuf.spsm,
                                                m_transfer.bitbltbuf.sbp,
                                                std::max<uint32_t>(m_transfer.bitbltbuf.sbw, 1u),
                                                x + m_transfer.trxpos.ssax,
                                                y + m_transfer.trxpos.ssay);
        WriteVramUnlocked(m_transfer.bitbltbuf.dpsm,
                          m_transfer.bitbltbuf.dbp,
                          std::max<uint32_t>(m_transfer.bitbltbuf.dbw, 1u),
                          x + m_transfer.trxpos.dsax,
                          y + m_transfer.trxpos.dsay,
                          value);
    }

    m_transferState.copiedPixels = total;
    m_transferState.direction = 3u;
}

void GSCpuBackend::PerformLocalToHostTransfer()
{
    m_localToHostBuffer.clear();
    m_localToHostReadPos = 0u;
    if (!m_vram)
        return;

    const uint32_t rrw = m_transfer.trxreg.rrw;
    const uint32_t rrh = m_transfer.trxreg.rrh;
    const uint32_t sbw = std::max<uint32_t>(m_transfer.bitbltbuf.sbw, 1u);
    const uint8_t spsm = m_transfer.bitbltbuf.spsm;
    const uint32_t bpp = static_cast<uint32_t>(GSMem::BitsPerPixel(static_cast<GSMem::PixelStorageMode>(spsm)));
    const uint32_t total = rrw * rrh;
    m_localToHostBuffer.reserve((static_cast<size_t>(total) * bpp + 7u) / 8u);

    for (uint32_t pixel = 0u; pixel < total; ++pixel)
    {
        const uint32_t x = pixel % rrw;
        const uint32_t y = pixel / rrw;
        const uint32_t value = ReadVramUnlocked(spsm,
                                                m_transfer.bitbltbuf.sbp,
                                                sbw,
                                                x + m_transfer.trxpos.ssax,
                                                y + m_transfer.trxpos.ssay);
        switch (bpp)
        {
        case 32:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 16u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 24u));
            break;
        case 24:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 16u));
            break;
        case 16:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value >> 8u));
            break;
        case 8:
            m_localToHostBuffer.push_back(static_cast<uint8_t>(value));
            break;
        case 4:
        {
            if ((pixel & 1u) != 0u)
                break;
            uint32_t next = 0u;
            if (pixel + 1u < total)
            {
                const uint32_t nextPixel = pixel + 1u;
                const uint32_t nextX = nextPixel % rrw;
                const uint32_t nextY = nextPixel / rrw;
                next = ReadVramUnlocked(spsm, m_transfer.bitbltbuf.sbp, sbw,
                                        nextX + m_transfer.trxpos.ssax,
                                        nextY + m_transfer.trxpos.ssay);
            }
            m_localToHostBuffer.push_back(static_cast<uint8_t>((value & 0x0Fu) | ((next & 0x0Fu) << 4u)));
            break;
        }
        default:
            break;
        }
    }

    m_transferState.copiedPixels = total;
    m_transferState.localToHostPendingBytes = m_localToHostBuffer.size();
}

uint32_t GSCpuBackend::ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    WaitForRasterIdle();
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!dst || maxBytes == 0u || m_localToHostReadPos >= m_localToHostBuffer.size())
        return 0u;
    const size_t count = std::min<size_t>(maxBytes, m_localToHostBuffer.size() - m_localToHostReadPos);
    std::memcpy(dst, m_localToHostBuffer.data() + m_localToHostReadPos, count);
    m_localToHostReadPos += count;
    m_transferState.localToHostPendingBytes = m_localToHostBuffer.size() - m_localToHostReadPos;
    return static_cast<uint32_t>(count);
}

bool GSCpuBackend::ClearFramebufferUnlocked(const GSContext &context, uint32_t rgba)
{
    if (!m_vram || context.frame.fbw == 0u)
        return false;

    const uint32_t x0 = context.scissor.x0;
    const uint32_t x1 = std::max<uint32_t>(x0, context.scissor.x1);
    const uint32_t y0 = context.scissor.y0;
    const uint32_t y1 = std::max<uint32_t>(y0, context.scissor.y1);
    uint8_t r = static_cast<uint8_t>(rgba);
    uint8_t g = static_cast<uint8_t>(rgba >> 8u);
    uint8_t b = static_cast<uint8_t>(rgba >> 16u);
    uint8_t a = static_cast<uint8_t>(rgba >> 24u);
    if ((context.fba & 1ull) != 0ull && context.frame.psm != GS_PSM_CT24)
        a |= 0x80u;

    const uint32_t fbp = GSInternal::framePageBaseToBlock(context.frame.fbp);
    const uint32_t fbw = std::max<uint32_t>(context.frame.fbw, 1u);
    if (context.frame.psm == GS_PSM_CT32 || context.frame.psm == GS_PSM_CT24)
    {
        const uint32_t source = static_cast<uint32_t>(r) |
                                (static_cast<uint32_t>(g) << 8u) |
                                (static_cast<uint32_t>(b) << 16u) |
                                (static_cast<uint32_t>(a) << 24u);
        for (uint32_t y = y0; y <= y1; ++y)
            for (uint32_t x = x0; x <= x1; ++x)
            {
                uint32_t pixel = source;
                if (context.frame.fbmsk != 0u)
                {
                    const uint32_t old = ReadVramUnlocked(context.frame.psm, fbp, fbw, x, y);
                    pixel = (pixel & ~context.frame.fbmsk) | (old & context.frame.fbmsk);
                }
                WriteVramUnlocked(context.frame.psm, fbp, fbw, x, y, pixel);
            }
        return true;
    }

    if (context.frame.psm == GS_PSM_CT16 || context.frame.psm == GS_PSM_CT16S)
    {
        const uint16_t source = encodeFramePixelPSMCT16(r, g, b, a);
        const uint16_t mask = static_cast<uint16_t>(context.frame.fbmsk);
        for (uint32_t y = y0; y <= y1; ++y)
            for (uint32_t x = x0; x <= x1; ++x)
            {
                uint16_t pixel = source;
                if (mask != 0u)
                {
                    const uint16_t old = static_cast<uint16_t>(ReadVramUnlocked(context.frame.psm, fbp, fbw, x, y));
                    pixel = static_cast<uint16_t>((pixel & ~mask) | (old & mask));
                }
                WriteVramUnlocked(context.frame.psm, fbp, fbw, x, y, pixel);
            }
        return true;
    }
    return false;
}

bool GSCpuBackend::CopyFrameToHostRgba(const GSFrameReg &frame,
                                       uint32_t width,
                                       uint32_t height,
                                       std::vector<uint8_t> &outPixels,
                                       bool preserveAlpha,
                                       bool useLocalMemoryLayout,
                                       bool frameBaseIsPages,
                                       uint32_t sourceOriginX,
                                       uint32_t sourceOriginY) const
{
    if (!m_vram || m_vramSize == 0u)
        return false;

    outPixels.assign(kHostFrameWidth * kHostFrameHeight * 4u, 0u);
    const uint32_t baseBytes = frameBaseIsPages ? frame.fbp * 8192u : frame.fbp * 256u;
    const uint32_t basePtr = frameBaseIsPages ? GSInternal::framePageBaseToBlock(frame.fbp) : frame.fbp;
    const uint32_t fbw = frame.fbw ? frame.fbw : kHostFrameWidth / 64u;
    const uint32_t bytesPerPixel = (frame.psm == GS_PSM_CT16 || frame.psm == GS_PSM_CT16S) ? 2u : 4u;
    const uint32_t stride = fbw * 64u * bytesPerPixel;

    for (uint32_t y = 0; y < height; ++y)
    {
        uint8_t *dst = outPixels.data() + y * kHostFrameWidth * 4u;
        for (uint32_t x = 0; x < width; ++x)
        {
            const uint32_t sx = sourceOriginX + x;
            const uint32_t sy = sourceOriginY + y;
            if (frame.psm == GS_PSM_CT32 || frame.psm == GS_PSM_CT24)
            {
                uint32_t color = 0u;
                if (useLocalMemoryLayout)
                    color = ReadVramUnlocked(frame.psm, basePtr, fbw, sx, sy);
                else
                {
                    const uint32_t pixelBytes = frame.psm == GS_PSM_CT24 ? 3u : 4u;
                    const uint64_t offset = static_cast<uint64_t>(baseBytes) + static_cast<uint64_t>(sy) * stride + static_cast<uint64_t>(sx) * pixelBytes;
                    if (offset + pixelBytes > m_vramSize)
                        return false;
                    color = m_vram[offset] | (static_cast<uint32_t>(m_vram[offset + 1u]) << 8u) |
                            (static_cast<uint32_t>(m_vram[offset + 2u]) << 16u);
                    if (pixelBytes == 4u)
                        color |= static_cast<uint32_t>(m_vram[offset + 3u]) << 24u;
                }
                dst[x * 4u] = static_cast<uint8_t>(color);
                dst[x * 4u + 1u] = static_cast<uint8_t>(color >> 8u);
                dst[x * 4u + 2u] = static_cast<uint8_t>(color >> 16u);
                dst[x * 4u + 3u] = preserveAlpha && frame.psm != GS_PSM_CT24 ? static_cast<uint8_t>(color >> 24u) : 255u;
            }
            else if (frame.psm == GS_PSM_CT16 || frame.psm == GS_PSM_CT16S)
            {
                uint16_t color = 0u;
                if (useLocalMemoryLayout)
                    color = static_cast<uint16_t>(ReadVramUnlocked(frame.psm, basePtr, fbw, sx, sy));
                else
                {
                    const uint64_t offset = static_cast<uint64_t>(baseBytes) + static_cast<uint64_t>(sy) * stride + static_cast<uint64_t>(sx) * 2u;
                    if (offset + 2u > m_vramSize)
                        return false;
                    std::memcpy(&color, m_vram + offset, sizeof(color));
                }
                const uint32_t r = color & 31u;
                const uint32_t g = (color >> 5u) & 31u;
                const uint32_t b = (color >> 10u) & 31u;
                dst[x * 4u] = static_cast<uint8_t>((r << 3u) | (r >> 2u));
                dst[x * 4u + 1u] = static_cast<uint8_t>((g << 3u) | (g >> 2u));
                dst[x * 4u + 2u] = static_cast<uint8_t>((b << 3u) | (b >> 2u));
                dst[x * 4u + 3u] = preserveAlpha ? ((color & 0x8000u) ? 0x80u : 0u) : 255u;
            }
            else
            {
                outPixels.clear();
                return false;
            }
        }
    }
    return true;
}

PresentationFrame GSCpuBackend::Present(const GSPresentationRequest &request)
{
    // Snapshot local memory under the backend lock, then perform the expensive
    // display conversion without holding the producer-side raster lock.
    thread_local std::vector<uint8_t> snapshot;
    SnapshotVramUnsynced(snapshot);
    if (snapshot.empty())
        return {};

    thread_local GSCpuBackend snapshotBackend;
    snapshotBackend.Initialize(snapshot.data(), static_cast<uint32_t>(snapshot.size()));
    return snapshotBackend.PresentFromLocalMemory(request);
}

PresentationFrame GSCpuBackend::PresentFromLocalMemory(const GSPresentationRequest &request)
{
    PresentationFrame result{};
    const GSPmodeState pmode = decodePmode(request.pmode);
    const GSSmode2State smode2 = decodeSMode2(request.smode2);
    const bool fieldMode = smode2.interlaced && !smode2.frameMode;
    // PS2X_GS_PRESENT_FULLHEIGHT=1 (diagnostic): read the full DISPLAY height
    // from local memory instead of half of it + line doubling in interlaced
    // frame mode. WotM's boot screens lose their bottom half ('presents', the
    // trademark text) with the half-height read; in-game buffers are 640x224.
    static const bool s_presentFullHeight = []
    {
        const char *value = std::getenv("PS2X_GS_PRESENT_FULLHEIGHT");
        return value != nullptr && value[0] == '1';
    }();
    // Whether the source is half height cannot be read off the display
    // registers: the boot logos and the menu program them identically
    // (640x448, interlaced frame mode). What differs is the content, so use
    // that -- g_ps2xFbDrawnHeight records the tallest scissor drawn into each
    // frame buffer. Measured: the menu's buffer is drawn 224 rows (needs the
    // half read + line doubling), the logo buffer 448 (needs the full read;
    // reading half is what cut their bottom off). MAGV is 0 in both, which is
    // why deciding on it broke the menu.
    const bool interlacedFrame = smode2.interlaced && smode2.frameMode && !s_presentFullHeight;
    const bool oddField = (request.vsyncTick & 1ull) != 0ull;
    const GSFrameReg displayFrame1 = decodeDisplayFrame(request.dispfb1);
    const GSFrameReg displayFrame2 = decodeDisplayFrame(request.dispfb2);
    const GSDisplayReadOrigin origin1 = decodeDisplayReadOrigin(request.dispfb1);
    const GSDisplayReadOrigin origin2 = decodeDisplayReadOrigin(request.dispfb2);
    uint32_t width1 = 0u, height1 = 0u, width2 = 0u, height2 = 0u;
    decodeDisplaySize(request.display1, width1, height1);
    decodeDisplaySize(request.display2, width2, height2);
    const uint32_t drawn1 = fbDrawnHeight(displayFrame1.fbp);
    const uint32_t drawn2 = fbDrawnHeight(displayFrame2.fbp);
    rollFbDrawnHeights();
    const bool frameDoubled1 = interlacedFrame && !(drawn1 != 0u && drawn1 >= height1);
    const bool frameDoubled2 = interlacedFrame && !(drawn2 != 0u && drawn2 >= height2);
    // The merged-CRT path below builds its result from display 1.
    const bool frameDoubled = frameDoubled1;
    const bool valid1 = pmode.enableCrt1 && hasDisplaySetup(request.display1, displayFrame1);
    const bool valid2 = pmode.enableCrt2 && hasDisplaySetup(request.display2, displayFrame2);
    if (!valid1 && !valid2)
        return result;
    // Buffer rows actually read from local memory (half the display height in
    // interlaced frame mode; see applyFrameModeLineDouble).
    const uint32_t srcHeight1 = frameDoubled1 ? (height1 + 1u) / 2u : height1;
    const uint32_t srcHeight2 = frameDoubled2 ? (height2 + 1u) / 2u : height2;

    {
        // One line per distinct presentation setup.
        static uint64_t s_lastKey = ~0ull;
        static uint32_t s_logged = 0u;
        const uint64_t key = request.pmode ^ (request.smode2 << 8) ^ (request.display1 * 31u) ^
                             (request.display2 * 131u) ^
                             (static_cast<uint64_t>(g_ps2xFbDrawnHeight[displayFrame1.fbp & 0x1FFu].load(std::memory_order_relaxed)) << 40) ^
                             (static_cast<uint64_t>(g_ps2xFbDrawnHeight[displayFrame2.fbp & 0x1FFu].load(std::memory_order_relaxed)) << 52);
        if (key != s_lastKey && s_logged < 60u)
        {
            s_lastKey = key;
            ++s_logged;
            std::fprintf(stderr,
                         "[present] pmode=%llx smode2=%llx crt1=%d crt2=%d interlaced=%d frameMode=%d"
                         " d1=%llx (w=%u h=%u src=%u dy=%u magv=%u) d2=%llx (w=%u h=%u src=%u)"
                         " fb1=%llx(fbw=%u psm=%u fbp=%u) fb2fbp=%u drawn1=%u drawn2=%u doubled=%d\n",
                         static_cast<unsigned long long>(request.pmode),
                         static_cast<unsigned long long>(request.smode2),
                         pmode.enableCrt1 ? 1 : 0, pmode.enableCrt2 ? 1 : 0,
                         smode2.interlaced ? 1 : 0, smode2.frameMode ? 1 : 0,
                         static_cast<unsigned long long>(request.display1), width1, height1, srcHeight1,
                         static_cast<uint32_t>((request.display1 >> 12) & 0x7FFu),
                         static_cast<uint32_t>((request.display1 >> 27) & 0x3u),
                         static_cast<unsigned long long>(request.display2), width2, height2, srcHeight2,
                         static_cast<unsigned long long>(request.dispfb1),
                         displayFrame1.fbw, displayFrame1.psm, displayFrame1.fbp,
                         displayFrame2.fbp,
                         drawn1, drawn2,
                         frameDoubled ? 1 : 0);
        }
    }

    auto copySource = [&](const GSFrameReg &displayFrame,
                          const GSDisplayReadOrigin &origin,
                          uint32_t width,
                          uint32_t height,
                          bool allowPreferred,
                          bool preserveAlpha,
                          GSFrameReg &selected,
                          std::vector<uint8_t> &pixels,
                          bool &usedPreferred) -> bool
    {
        selected = displayFrame;
        pixels.clear();
        usedPreferred = false;
        if (allowPreferred && request.hasPreferredSource && request.preferredDestFbp == displayFrame.fbp &&
            (request.preferredSource.fbw != 0u || request.preferredSource.fbp != displayFrame.fbp) &&
            CopyFrameToHostRgba(request.preferredSource, width, height, pixels, preserveAlpha, true, false, 0u, 0u))
        {
            selected = request.preferredSource;
            usedPreferred = true;
        }
        if (pixels.empty() && !CopyFrameToHostRgba(displayFrame, width, height, pixels, preserveAlpha, true, true, origin.x, origin.y))
            return false;

        if (!usedPreferred && displayFrame.fbp == 0u && countNonBlackPixels(pixels, width, height) == 0u)
        {
            for (const GSFrameReg &candidate : request.contextFrames)
            {
                if (candidate.fbp == selected.fbp && candidate.fbw == selected.fbw && candidate.psm == selected.psm)
                    continue;
                std::vector<uint8_t> candidatePixels;
                if (!CopyFrameToHostRgba(candidate, width, height, candidatePixels, preserveAlpha, true, true, 0u, 0u))
                    continue;
                if (countNonBlackPixels(candidatePixels, width, height) == 0u)
                    continue;
                selected = candidate;
                pixels.swap(candidatePixels);
                break;
            }
        }
        return true;
    };

    if (valid1 && valid2)
    {
        GSFrameReg selected1{}, selected2{};
        std::vector<uint8_t> crt1, crt2;
        bool preferred1 = false, preferred2 = false;
        if (copySource(displayFrame1, origin1, width1, srcHeight1, false, true, selected1, crt1, preferred1) &&
            copySource(displayFrame2, origin2, width2, srcHeight2, false, true, selected2, crt2, preferred2))
        {
            result.width = std::max(width1, width2);
            result.height = std::max(height1, height2);
            result.pixels.assign(kHostFrameWidth * kHostFrameHeight * 4u, 0u);
            const uint8_t bgR = static_cast<uint8_t>(request.bgcolor);
            const uint8_t bgG = static_cast<uint8_t>(request.bgcolor >> 8u);
            const uint8_t bgB = static_cast<uint8_t>(request.bgcolor >> 16u);
            for (uint32_t y = 0; y < result.height; ++y)
                for (uint32_t x = 0; x < result.width; ++x)
                {
                    uint8_t *dst = result.pixels.data() + (y * kHostFrameWidth + x) * 4u;
                    dst[0] = bgR;
                    dst[1] = bgG;
                    dst[2] = bgB;
                    dst[3] = pmode.alp;
                }
            if (!pmode.slbg)
                for (uint32_t y = 0; y < srcHeight2; ++y)
                    std::memcpy(result.pixels.data() + y * kHostFrameWidth * 4u, crt2.data() + y * kHostFrameWidth * 4u, width2 * 4u);
            for (uint32_t y = 0; y < srcHeight1; ++y)
                for (uint32_t x = 0; x < width1; ++x)
                {
                    const uint8_t *src = crt1.data() + (y * kHostFrameWidth + x) * 4u;
                    uint8_t *dst = result.pixels.data() + (y * kHostFrameWidth + x) * 4u;
                    const uint32_t factor = pmode.mmod ? pmode.alp : std::min<uint32_t>(255u, static_cast<uint32_t>(src[3]) * 2u);
                    dst[0] = blendPresentationChannel(src[0], dst[0], factor);
                    dst[1] = blendPresentationChannel(src[1], dst[1], factor);
                    dst[2] = blendPresentationChannel(src[2], dst[2], factor);
                    dst[3] = pmode.amod ? dst[3] : src[3];
                }
            normalizePresentationAlpha(result.pixels, result.width, result.height);
            if (frameDoubled)
                applyFrameModeLineDouble(result.pixels, result.width, result.height);
            if (fieldMode)
                applyFieldPresentation(result.pixels, result.width, result.height, oddField);
            result.displayFbp = displayFrame1.fbp;
            result.sourceFbp = selected1.fbp;
            return result;
        }
    }

    const GSFrameReg &displayFrame = valid1 ? displayFrame1 : displayFrame2;
    const GSDisplayReadOrigin &origin = valid1 ? origin1 : origin2;
    result.width = valid1 ? width1 : width2;
    result.height = valid1 ? height1 : height2;
    GSFrameReg selected = displayFrame;
    const uint32_t srcHeight = valid1 ? srcHeight1 : srcHeight2;
    if (!copySource(displayFrame, origin, result.width, srcHeight, true, false, selected, result.pixels, result.usedPreferred))
        return {};
    if (valid1 ? frameDoubled1 : frameDoubled2)
        applyFrameModeLineDouble(result.pixels, result.width, result.height);
    if (fieldMode)
        applyFieldPresentation(result.pixels, result.width, result.height, oddField);
    normalizePresentationAlpha(result.pixels, result.width, result.height);
    result.displayFbp = displayFrame.fbp;
    result.sourceFbp = selected.fbp;
    return result;
}

// ---------------------------------------------------------------------------
// Raster worker (see the comment on RasterCommand in gs_cpu_backend.h).
// ---------------------------------------------------------------------------

GSCpuBackend::~GSCpuBackend()
{
    {
        std::lock_guard<std::mutex> queueLock(m_queueMutex);
        m_stopWorker = true;
    }
    m_queueCv.notify_all();
    if (m_worker.joinable())
        m_worker.join();
    {
        std::lock_guard<std::mutex> bandLock(m_bandMutex);
        m_bandStop = true;
    }
    m_bandStartCv.notify_all();
    for (std::thread &band : m_bandThreads)
        if (band.joinable())
            band.join();
}

void GSCpuBackend::BeginTransfer(const GSTransferCommand &command)
{
    RasterCommand queued;
    queued.kind = RasterCommand::Kind::BeginTransfer;
    queued.transfer = command;
    Enqueue(std::move(queued));
}

void GSCpuBackend::UploadImage(const uint8_t *data, uint32_t sizeBytes)
{
    if (!data || sizeBytes == 0u)
        return;
    // The caller's buffer is only valid for the duration of this call.
    RasterCommand queued;
    queued.kind = RasterCommand::Kind::UploadImage;
    queued.bytes.assign(data, data + sizeBytes);
    Enqueue(std::move(queued));
}

bool GSCpuBackend::ClearFramebuffer(const GSContext &context, uint32_t rgba)
{
    // Same acceptance test ClearFramebufferUnlocked applies, answered up front so
    // the caller gets its result without waiting for the raster thread.
    const uint8_t psm = context.frame.psm;
    const bool handled = m_vram != nullptr && context.frame.fbw != 0u &&
                         (psm == GS_PSM_CT32 || psm == GS_PSM_CT24 || psm == GS_PSM_CT16 || psm == GS_PSM_CT16S);
    if (!handled)
        return false;
    RasterCommand queued;
    queued.kind = RasterCommand::Kind::Clear;
    queued.clearContext = context;
    queued.clearRgba = rgba;
    Enqueue(std::move(queued));
    return true;
}

// [perf]: game-thread time blocked because the raster queue was full.
std::atomic<uint64_t> g_ps2xRasterFullNs{0u};

void GSCpuBackend::Enqueue(RasterCommand &&command)
{
    static const bool s_inlineRaster = []
    {
        const char *value = std::getenv("PS2X_GS_SYNC");
        return value != nullptr && value[0] == '1';
    }();
    if (s_inlineRaster)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        ExecuteCommand(command);
        return;
    }

    bool wakeWorker = false;
    {
        std::unique_lock<std::mutex> queueLock(m_queueMutex);
        if (!m_worker.joinable())
            m_worker = std::thread([this] { WorkerMain(); });
        // Backpressure: never let the game run unboundedly far ahead.
        if (m_pending.size() >= kMaxPendingRasterCommands)
        {
            const auto fullT0 = std::chrono::steady_clock::now();
            m_spaceCv.wait(queueLock, [this] { return m_pending.size() < kMaxPendingRasterCommands; });
            g_ps2xRasterFullNs.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                                   std::chrono::steady_clock::now() - fullT0)
                                                                   .count()),
                                         std::memory_order_relaxed);
        }
        m_pending.push_back(std::move(command));
        ++m_submittedSeq;
        // The worker only sleeps on an empty queue, so only the push that makes
        // it non-empty needs to wake it.
        wakeWorker = (m_pending.size() == 1u);
    }
    if (wakeWorker)
        m_queueCv.notify_one();
}

void GSCpuBackend::ExecuteCommand(RasterCommand &command)
{
    switch (command.kind)
    {
    case RasterCommand::Kind::Draw:
        DrawPrimitive(command.batch);
        break;
    case RasterCommand::Kind::BeginTransfer:
        BeginTransferUnlocked(command.transfer);
        break;
    case RasterCommand::Kind::UploadImage:
        UploadImageUnlocked(command.bytes.data(), static_cast<uint32_t>(command.bytes.size()));
        break;
    case RasterCommand::Kind::Clear:
        (void)ClearFramebufferUnlocked(command.clearContext, command.clearRgba);
        break;
    case RasterCommand::Kind::WriteVram:
        WriteVramUnlocked(command.psm, command.base, command.bw, command.x, command.y, command.value);
        break;
    case RasterCommand::Kind::PresentSnapshot:
        // Runs as a barrier (non-draw), in FIFO order: exactly the draws queued
        // before it are in local memory. See EnqueuePresentSnapshot.
        if (m_vram && m_vramSize != 0u)
        {
            std::lock_guard<std::mutex> snapshotLock(m_presentSnapshotMutex);
            m_presentSnapshot.resize(m_vramSize);
            std::memcpy(m_presentSnapshot.data(), m_vram, m_vramSize);
        }
        break;
    }
}

uint64_t GSCpuBackend::EnqueuePresentSnapshot()
{
    RasterCommand command;
    command.kind = RasterCommand::Kind::PresentSnapshot;
    Enqueue(std::move(command));
    // The caller holds the GS state lock, so nothing else was queued after the
    // snapshot command; waiting for this sequence number waits for it. (Inline
    // mode, PS2X_GS_SYNC=1, executed it already and never advances the counters.)
    std::lock_guard<std::mutex> queueLock(m_queueMutex);
    return m_submittedSeq;
}

PresentationFrame GSCpuBackend::PresentSnapshot(uint64_t token, const GSPresentationRequest &request)
{
    {
        std::unique_lock<std::mutex> queueLock(m_queueMutex);
        m_doneCv.wait(queueLock, [this, token] { return m_completedSeq >= token; });
    }
    thread_local std::vector<uint8_t> snapshot;
    {
        std::lock_guard<std::mutex> snapshotLock(m_presentSnapshotMutex);
        snapshot.swap(m_presentSnapshot);
    }
    if (snapshot.empty())
        return {};
    thread_local GSCpuBackend snapshotBackend;
    snapshotBackend.Initialize(snapshot.data(), static_cast<uint32_t>(snapshot.size()));
    return snapshotBackend.PresentFromLocalMemory(request);
}

void GSCpuBackend::WorkerMain()
{
    StartBandThreads();
    std::vector<RasterCommand> work;
    for (;;)
    {
        uint64_t workEndSeq = 0u;
        {
            std::unique_lock<std::mutex> queueLock(m_queueMutex);
            m_queueCv.wait(queueLock, [this] { return m_stopWorker || !m_pending.empty(); });
            if (m_pending.empty())
                return; // stopping, and everything submitted has been drawn
            // Swap rather than copy: the emptied vector keeps its capacity, so the
            // steady state does no reallocation on either side.
            work.swap(m_pending);
            workEndSeq = m_submittedSeq;
        }
        m_spaceCv.notify_all();
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            ExecuteWorkParallel(work);
        }
        work.clear();
        {
            std::lock_guard<std::mutex> queueLock(m_queueMutex);
            m_completedSeq = workEndSeq;
        }
        m_doneCv.notify_all();
    }
}

std::atomic<uint64_t> g_ps2xRasterIdleWaits{0u};
std::atomic<uint64_t> g_ps2xRasterIdleWaitNs{0u};

void GSCpuBackend::WaitForRasterIdle() const
{
    const auto t0 = std::chrono::steady_clock::now();
    {
        std::unique_lock<std::mutex> queueLock(m_queueMutex);
        const uint64_t ticket = m_submittedSeq;
        m_doneCv.wait(queueLock, [this, ticket] { return m_completedSeq >= ticket; });
    }
    // Callers: GS FINISH, VRAM/debug readbacks, presentation Sync ([perf] waits).
    g_ps2xRasterIdleWaits.fetch_add(1u, std::memory_order_relaxed);
    g_ps2xRasterIdleWaitNs.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now() - t0).count()),
                                     std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Banded parallel raster (see the comment on ExecuteWorkParallel in the header).
// ---------------------------------------------------------------------------

void GSCpuBackend::StartBandThreads()
{
    uint32_t bands = 0u;
    if (const char *value = std::getenv("PS2X_GS_THREADS"))
        bands = static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
    if (bands == 0u)
    {
        // Leave one core each for the game thread and the presenter. Measured on
        // an 8-core i7-9700K in a WotM level: hw/3 = 2 bands left the game thread
        // blocked on a full raster queue 13-56% of the time (~86k prims/s in the
        // heaviest scene); 6 bands = 0% blocked and ~160-185k prims/s.
        const uint32_t hw = std::max(1u, std::thread::hardware_concurrency());
        bands = std::clamp(hw > 2u ? hw - 2u : 1u, 1u, 8u);
    }
    bands = std::clamp(bands, 1u, 16u);
    m_bandCount = bands;
    for (uint32_t band = 1u; band < bands; ++band)
        m_bandThreads.emplace_back([this, band] { BandThreadMain(band); });
    std::fprintf(stderr, "[gs:raster] %u raster band thread(s)\n", bands);
}

// [raster] breakdown of the raster worker's time (worker thread only).
namespace
{
    uint64_t g_rwScanNs = 0u, g_rwSegNs = 0u, g_rwSerialNs = 0u, g_rwBarrierNs = 0u;
    uint64_t g_rwSegDraws = 0u, g_rwSerialDraws = 0u, g_rwBarriers = 0u, g_rwSegments = 0u;
    inline uint64_t rwNow()
    {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         std::chrono::steady_clock::now().time_since_epoch())
                                         .count());
    }
    void rwMaybeReport()
    {
        static uint64_t s_last = rwNow();
        const uint64_t now = rwNow();
        if (now - s_last < 5000000000ull)
            return;
        const double wall = double(now - s_last);
        s_last = now;
        std::fprintf(stderr,
                     "[raster] scan %.1f%% parallel %.1f%% (%llu draws in %llu segs) serial %.1f%% (%llu draws) barriers %.1f%% (%llu)\n",
                     100.0 * g_rwScanNs / wall, 100.0 * g_rwSegNs / wall,
                     static_cast<unsigned long long>(g_rwSegDraws), static_cast<unsigned long long>(g_rwSegments),
                     100.0 * g_rwSerialNs / wall, static_cast<unsigned long long>(g_rwSerialDraws),
                     100.0 * g_rwBarrierNs / wall, static_cast<unsigned long long>(g_rwBarriers));
        g_rwScanNs = g_rwSegNs = g_rwSerialNs = g_rwBarrierNs = 0u;
        g_rwSegDraws = g_rwSerialDraws = g_rwBarriers = g_rwSegments = 0u;
    }
}

// Opt-in input/output fixture for a future GPU backend. Captures a contiguous
// draw-only segment while all band workers are idle, not an assumed whole frame.
// Raw GSPrimitiveBatch records are intentionally tied to this build's ABI.
static_assert(std::is_trivially_copyable_v<GSPrimitiveBatch>);
extern std::atomic<uint64_t> g_ps2xWotmCompletedFrames;
namespace
{
    std::unique_ptr<std::ofstream> beginGsCapture(size_t count,
                                 const uint8_t *vram, uint32_t vramSize)
    {
        static const char *path = std::getenv("PS2X_GS_CAPTURE");
        static bool attempted = false;
        if (!path || !*path || attempted || count < 64u || count > 8192u)
            return {};
        static const uint64_t start = [] {
            const char *value = std::getenv("PS2X_GS_CAPTURE_FRAME");
            return value ? std::strtoull(value, nullptr, 10) : 120ull;
        }();
        const uint64_t frame = g_ps2xWotmCompletedFrames.load(std::memory_order_relaxed);
        if (frame < start || !vram || vramSize != 4u * 1024u * 1024u)
            return {};
        attempted = true;
        // Never silently overwrite a prior reference capture.
        std::error_code ec;
        if (std::filesystem::exists(path, ec) || ec)
        {
            std::fprintf(stderr, "[gs:capture] path exists or is inaccessible; capture skipped\n");
            return {};
        }
        auto out = std::make_unique<std::ofstream>(std::string(path) + ".tmp", std::ios::binary | std::ios::trunc);
        const char magic[8] = {'P','S','2','G','S','C','1','\0'};
        const uint32_t header[6] = {1u, vramSize, sizeof(GSPrimitiveBatch),
                                   static_cast<uint32_t>(count), static_cast<uint32_t>(frame),
                                   static_cast<uint32_t>(frame >> 32)};
        out->write(magic, sizeof(magic));
        out->write(reinterpret_cast<const char *>(header), sizeof(header));
        out->write(reinterpret_cast<const char *>(vram), vramSize);
        return out;
    }

    void finishGsCapture(std::ofstream &out, const uint8_t *vram, uint32_t vramSize, size_t count)
    {
        if (!out.is_open())
            return;
        out.write(reinterpret_cast<const char *>(vram), vramSize);
        out.flush();
        const bool ok = out.good();
        out.close();
        const char *path = std::getenv("PS2X_GS_CAPTURE");
        std::error_code ec;
        if (ok)
            std::filesystem::rename(std::string(path) + ".tmp", path, ec);
        std::fprintf(stderr, "[gs:capture] %s draws=%zu bytesPerDraw=%zu path=%s\n",
                     ok && !ec ? "saved" : "FAILED", count, sizeof(GSPrimitiveBatch), path);
    }
}

// Headless deterministic CPU replay. Returns a process exit status to main.
// Validates capture integrity and establishes a baseline for GPU comparisons.
int ps2xReplayGsCapture(const char *path)
{
    std::ifstream in(path, std::ios::binary);
    char magic[8]{};
    uint32_t header[6]{};
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char *>(header), sizeof(header));
    if (!in || std::memcmp(magic, "PS2GSC1\0", 8) != 0 || header[0] != 1u ||
        header[1] != 4u * 1024u * 1024u || header[2] != sizeof(GSPrimitiveBatch) ||
        header[3] == 0u || header[3] > 8192u)
    {
        std::fprintf(stderr, "[gs:replay] invalid header or incompatible build ABI\n");
        return 2;
    }
    std::vector<uint8_t> vram(header[1]), expected(header[1]), actual;
    std::vector<GSPrimitiveBatch> draws(header[3]);
    in.read(reinterpret_cast<char *>(vram.data()), vram.size());
    in.read(reinterpret_cast<char *>(draws.data()), draws.size() * sizeof(GSPrimitiveBatch));
    in.read(reinterpret_cast<char *>(expected.data()), expected.size());
    if (!in || in.peek() != std::char_traits<char>::eof())
    {
        std::fprintf(stderr, "[gs:replay] truncated capture or trailing bytes\n");
        return 2;
    }
    uint64_t textured = 0u, blended = 0u, destinationAlpha = 0u, masked = 0u;
    std::array<uint64_t, 64> textureFormats{}, targetFormats{};
    for (const auto &draw : draws)
    {
        textured += draw.state.prim.tme;
        blended += draw.state.prim.abe;
        destinationAlpha += (draw.state.context.test >> 14) & 1u;
        masked += draw.state.context.frame.fbmsk != 0u;
        ++targetFormats[draw.state.context.frame.psm & 63u];
        if (draw.state.prim.tme) ++textureFormats[draw.state.context.tex0.psm & 63u];
    }
    std::fprintf(stderr, "[gs:replay] textured=%llu blended=%llu destinationAlpha=%llu framebufferMasked=%llu\n",
                 static_cast<unsigned long long>(textured), static_cast<unsigned long long>(blended),
                 static_cast<unsigned long long>(destinationAlpha), static_cast<unsigned long long>(masked));
    for (size_t psm = 0; psm < 64; ++psm)
        if (textureFormats[psm] || targetFormats[psm])
            std::fprintf(stderr, "[gs:replay] psm=0x%02zx textureDraws=%llu targetDraws=%llu\n", psm,
                         static_cast<unsigned long long>(textureFormats[psm]),
                         static_cast<unsigned long long>(targetFormats[psm]));
#if defined(_WIN32)
    if (std::getenv("PS2X_GS_SUBMIT_BENCH")) {
        extern int ps2xBenchmarkGlSubmission(std::vector<uint8_t> &, const std::vector<GSPrimitiveBatch> &);
        return ps2xBenchmarkGlSubmission(vram, draws);
    }
#endif
    GSCpuBackend backend;
    backend.Initialize(vram.data(), static_cast<uint32_t>(vram.size()));
    const auto replayStart=std::chrono::steady_clock::now();
    for (const auto &draw : draws)
        backend.Submit(draw);
    backend.SnapshotVram(actual);
    std::fprintf(stderr,"[gs:replay:time] ns=%llu\n",static_cast<unsigned long long>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-replayStart).count()));
    size_t mismatches = 0u, first = 0u;
    for (size_t i = 0; i < expected.size(); ++i)
        if (actual[i] != expected[i])
        {
            if (mismatches == 0u) first = i;
            ++mismatches;
        }
    std::fprintf(stderr, "[gs:replay] draws=%zu mismatchedBytes=%zu firstOffset=0x%zx\n",
                 draws.size(), mismatches, first);
    return mismatches == 0u ? 0 : 1;
}


void GSCpuBackend::ExecuteWorkParallel(std::vector<RasterCommand> &work)
{
    static const bool splitGpuSegments = [] { const char *v=std::getenv("PS2X_GS_BACKEND"); const char *split=std::getenv("PS2X_GS_GPU_SPLIT"); return v && std::strcmp(v,"gpu")==0 && split && std::atoi(split)!=0; }();
    auto gpuCandidate = [](const GSPrimitiveBatch &batch) {
        const auto &ctx=batch.state.context;
        return batch.vertexCount==3 && batch.state.prim.type>=GS_PRIM_TRIANGLE && batch.state.prim.type<=GS_PRIM_TRIFAN &&
            (ctx.frame.psm==GS_PSM_CT24 || ctx.frame.psm==GS_PSM_CT32) &&
            (ctx.zbuf.psm==GS_PSM_Z24 || ctx.zbuf.psm==GS_PSM_Z32);
    };
    size_t i = 0u;
    while (i < work.size())
    {
        // Longest run of band-safe draws with no render-to-texture dependency
        // on a target written earlier in the same run.
        std::vector<BlockRange> written;
        const uint64_t scanT0 = rwNow();
        size_t j = i;
        while (j < work.size() && work[j].kind == RasterCommand::Kind::Draw && isBandSafeDraw(work[j].batch))
        {
            // A sprite or target change should not force surrounding supported
            // triangles onto the CPU. Existing dependency barriers still apply.
            if (splitGpuSegments && j>i)
            {
                const auto &a=work[i].batch;const auto &b=work[j].batch;
                const bool candidate=gpuCandidate(a);
                if(candidate!=gpuCandidate(b))break;
                if(candidate)
                {
                    const auto &x=a.state.context;const auto &y=b.state.context;
                    if(j-i>=8192 || x.frame.fbp!=y.frame.fbp || x.frame.fbw!=y.frame.fbw || x.frame.psm!=y.frame.psm ||
                       x.zbuf.zbp!=y.zbuf.zbp || x.zbuf.psm!=y.zbuf.psm)break;
                }
            }
            BlockRange r[2];
            int rc = 0;
            readRangesFor(work[j].batch, r, rc);
            bool dependent = false;
            for (int k = 0; k < rc && !dependent; ++k)
                for (const BlockRange &w : written)
                    if (overlaps(r[k], w))
                    {
                        dependent = true;
                        break;
                    }
            if (dependent)
                break;
            BlockRange w[2];
            int wc = 0;
            writeRangesFor(work[j].batch, w, wc);
            for (int k = 0; k < wc; ++k)
            {
                bool covered = false;
                for (BlockRange &x : written)
                {
                    if (w[k].begin >= x.begin && w[k].end <= x.end)
                    {
                        covered = true;
                        break;
                    }
                    if (x.begin >= w[k].begin && x.end <= w[k].end)
                    {
                        x = w[k]; // widen: the old entry is contained in the new one
                        covered = true;
                        break;
                    }
                }
                if (!covered)
                    written.push_back(w[k]);
            }
            ++j;
        }

        const uint64_t scanT1 = rwNow();
        g_rwScanNs += scanT1 - scanT0;
        if (j > i)
        {
            auto capture = beginGsCapture(j - i, m_vram, m_vramSize);
            if (capture && capture->is_open())
                for (size_t k = i; k < j; ++k)
                    capture->write(reinterpret_cast<const char *>(&work[k].batch), sizeof(GSPrimitiveBatch));
            // This worker owns VRAM; all CPU bands are idle at this boundary.
            // GPU failure leaves it untouched. Captures always retain a CPU reference.
            static const bool gpuVerify = [] { const char *v = std::getenv("PS2X_GS_GPU_VERIFY"); return v && std::strcmp(v, "1") == 0; }();
            static thread_local bool gpuMismatch = false;
            std::vector<uint8_t> gpuResult;
            std::vector<uint8_t> gpuBefore;

            const bool gpuRendered = !capture && !gpuMismatch &&
                ps2xRenderGsGpuSegment(m_vram, m_vramSize, &work[i].batch, sizeof(RasterCommand), j - i, gpuResult, gpuVerify ? &gpuBefore : nullptr);
            if (gpuRendered && !gpuVerify)
            {
                std::memcpy(m_vram, gpuResult.data(), m_vramSize);
                g_ps2xPerf.prims.fetch_add(j - i, std::memory_order_relaxed);
                for (size_t k = i; k < j; ++k)
                {
                    const auto &ctx = work[k].batch.state.context;
                    auto &height = g_ps2xFbDrawnHeight[ctx.frame.fbp & 0x1FFu];
                    uint16_t seen = height.load(std::memory_order_relaxed);
                    const uint16_t drawnTo = static_cast<uint16_t>(ctx.scissor.y1 + 1u);
                    while (drawnTo > seen && !height.compare_exchange_weak(seen, drawnTo, std::memory_order_relaxed)) {}
                }
                g_rwSegNs += rwNow() - scanT1;
                g_rwSegDraws += j - i;
                ++g_rwSegments;
            }
            else if (m_bandCount > 1u && (j - i) >= kMinParallelDraws)
            {
                RunSegmentOnAllBands(&work[i], j - i);
                g_rwSegNs += rwNow() - scanT1;
                g_rwSegDraws += j - i;
                ++g_rwSegments;
            }
            else
            {
                for (size_t k = i; k < j; ++k)
                    ExecuteCommand(work[k]);
                g_rwSerialNs += rwNow() - scanT1;
                g_rwSerialDraws += j - i;
            }
            if (gpuRendered && gpuVerify)
            {
                static thread_local uint64_t verified = 0;
                size_t different = 0, firstDifference = 0;
                for (size_t b = 0; b < m_vramSize; ++b)
                    if (gpuResult[b] != m_vram[b]) { if (!different) firstDifference = b; ++different; }
                ++verified;
                if (different || verified <= 3 || verified % 1000 == 0)
                    std::fprintf(stderr, "[gs:gpu:verify] segments=%llu mismatchedBytes=%zu first=0x%zx\n",
                        static_cast<unsigned long long>(verified), different, firstDifference);
                if (different)
                {
                    gpuMismatch = true; // Keep CPU output and disable further offload.
                    const char *path = std::getenv("PS2X_GS_GPU_MISMATCH");
                    if (path && *path && !std::filesystem::exists(path))
                    {
                        std::ofstream dump(path, std::ios::binary);
                        const uint32_t header[6] = {1, m_vramSize, sizeof(GSPrimitiveBatch), static_cast<uint32_t>(j-i), 0, 0};
                        dump.write("PS2GSC1\0", 8);
                        dump.write(reinterpret_cast<const char*>(header), sizeof(header));
                        dump.write(reinterpret_cast<const char*>(gpuBefore.data()), gpuBefore.size());
                        for(size_t k=i;k<j;++k) dump.write(reinterpret_cast<const char*>(&work[k].batch), sizeof(GSPrimitiveBatch));
                        dump.write(reinterpret_cast<const char*>(m_vram), m_vramSize);
                        dump.flush();
                        std::fprintf(stderr,"[gs:gpu:verify] mismatch capture %s: %s\n",path,dump?"saved":"write failed");
                    }
                }
            }
            if (capture)
                finishGsCapture(*capture, m_vram, m_vramSize, j - i);
            i = j;
            continue;
        }

        // Barrier: a non-draw command or a draw that can't be split. All band
        // threads are idle here, so it runs alone and in order.
        ExecuteCommand(work[i]);
        g_rwBarrierNs += rwNow() - scanT1;
        ++g_rwBarriers;
        ++i;
    }
    rwMaybeReport();
}

void GSCpuBackend::RunSegmentOnAllBands(const RasterCommand *commands, size_t count)
{
    {
        std::lock_guard<std::mutex> bandLock(m_bandMutex);
        m_bandSegment = commands;
        m_bandSegmentCount = count;
        m_bandsRemaining = m_bandCount - 1u;
        ++m_bandGeneration;
    }
    m_bandStartCv.notify_all();

    t_bandIndex = 0u;
    t_bandCount = m_bandCount;
    for (size_t k = 0; k < count; ++k)
        DrawPrimitive(commands[k].batch);
    t_bandCount = 1u;

    std::unique_lock<std::mutex> bandLock(m_bandMutex);
    m_bandDoneCv.wait(bandLock, [this] { return m_bandsRemaining == 0u; });
}

void GSCpuBackend::BandThreadMain(uint32_t band)
{
    t_bandIndex = band;
    t_bandCount = m_bandCount;
    uint64_t seenGeneration = 0u;
    for (;;)
    {
        const RasterCommand *commands = nullptr;
        size_t count = 0u;
        {
            std::unique_lock<std::mutex> bandLock(m_bandMutex);
            m_bandStartCv.wait(bandLock, [this, seenGeneration]
                               { return m_bandStop || m_bandGeneration != seenGeneration; });
            if (m_bandStop)
                return;
            seenGeneration = m_bandGeneration;
            commands = m_bandSegment;
            count = m_bandSegmentCount;
        }
        for (size_t k = 0; k < count; ++k)
            DrawPrimitive(commands[k].batch);
        bool last = false;
        {
            std::lock_guard<std::mutex> bandLock(m_bandMutex);
            last = (--m_bandsRemaining == 0u);
        }
        if (last)
            m_bandDoneCv.notify_one();
    }
}
