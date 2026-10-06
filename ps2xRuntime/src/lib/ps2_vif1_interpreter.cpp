#include "motion_provenance.inc"
// Based on Blackline Interactive implementation
#include "runtime/ps2_perf_clock.h"
#include "runtime/ps2_memory.h"
#include <cstring>
#include <array>
#include <atomic>
#include <cstdio>
#include <iostream>
#include <ps2_log.h>
#if defined(_M_X64) || defined(__x86_64__)
#include <emmintrin.h>
#endif

enum VIFCmd : uint8_t
{
    VIF_NOP = 0x00,
    VIF_STCYCL = 0x01,
    VIF_OFFSET = 0x02,
    VIF_BASE = 0x03,
    VIF_ITOP = 0x04,
    VIF_STMOD = 0x05,
    VIF_MSKPATH3 = 0x06,
    VIF_MARK = 0x07,
    VIF_FLUSHE = 0x10,
    VIF_FLUSH = 0x11,
    VIF_FLUSHA = 0x13,
    VIF_MSCAL = 0x14,
    VIF_MSCALF = 0x15,
    VIF_MSCNT = 0x17,
    VIF_STMASK = 0x20,
    VIF_STROW = 0x30,
    VIF_STCOL = 0x31,
    VIF_MPG = 0x4A,
    VIF_DIRECT = 0x50,
    VIF_DIRECTHL = 0x51,
};

namespace
{
const bool s_unpackProfile=std::getenv("PS2X_VIF1_UNPACK_PROFILE")!=nullptr;

#if defined(_M_X64) || defined(__x86_64__)
const bool s_unpackMasked=[] {const char *p=std::getenv("PS2X_VIF1_UNPACK_MASKED");return !p || p[0]!='0';}();
const bool s_unpackPartial=[] {
    const char *p=std::getenv("PS2X_VIF1_UNPACK_PARTIAL");return !p || p[0]!='0';
}();
// Unmasked S/V2/V3 packets: select format once, preserve unwritten lanes.
// The caller excludes fill cycles, add modes, aliasing and truncated input.
template<unsigned Components, unsigned Vl, bool Unsigned>
void unpackVif1Partial(uint8_t *memory, const uint8_t *src, uint32_t count,
                       uint32_t dest, uint32_t cl, uint32_t wl)
{
    static_assert(Components>=1 && Components<=3 && Vl<=2);
    auto write=[&](uint32_t address) {
        uint32_t lanes[4];
        for(unsigned c=0;c<Components;++c) {
            if constexpr(Vl==0) std::memcpy(&lanes[c],src+c*4u,4u);
            else if constexpr(Vl==1) {
                uint16_t raw;std::memcpy(&raw,src+c*2u,2u);
                lanes[c]=Unsigned ? uint32_t(raw) : uint32_t(int32_t(int16_t(raw)));
            } else {
                const uint8_t raw=src[c];
                lanes[c]=Unsigned ? uint32_t(raw) : uint32_t(int32_t(int8_t(raw)));
            }
        }
        if constexpr(Components==1) lanes[1]=lanes[2]=lanes[3]=lanes[0];
        std::memcpy(memory+address*16u,lanes,(Components==1 ? 4u : Components)*4u);
        src+=Components*(4u>>Vl);
    };
    if(wl==1u) {
        for(uint32_t i=0;i<count;++i) {write(dest);dest=(dest+cl)&1023u;}
    } else {
        uint32_t remaining=wl;
        for(uint32_t i=0;i<count;++i) {
            write(dest);dest=(dest+1u)&1023u;
            if(--remaining==0u) {dest=(dest+cl-wl)&1023u;remaining=wl;}
        }
    }
}
template<unsigned Components>
void unpackVif1PartialFormat(uint8_t *memory, const uint8_t *src, uint32_t count,
                             uint32_t dest, uint32_t cl, uint32_t wl,
                             unsigned vl, bool zeroExtend)
{
    if(vl==0u) unpackVif1Partial<Components,0,true>(memory,src,count,dest,cl,wl);
    else if(vl==1u) {
        if(zeroExtend) unpackVif1Partial<Components,1,true>(memory,src,count,dest,cl,wl);
        else unpackVif1Partial<Components,1,false>(memory,src,count,dest,cl,wl);
    } else {
        if(zeroExtend) unpackVif1Partial<Components,2,true>(memory,src,count,dest,cl,wl);
        else unpackVif1Partial<Components,2,false>(memory,src,count,dest,cl,wl);
    }
}

// Common masked packets: prepare lane selection once per STCYCL row, then
// merge whole vectors. No fill, add/accumulate, overlap or truncated inputs.
// Each mask byte selects data/row/column/protected independently per lane.
// Decode all 256 combinations once at compile time, not for every tiny packet.
struct alignas(16) VifMaskRow { uint32_t lanes[4][4]{}; };
static constexpr auto s_vifMaskRows=[] {
    std::array<VifMaskRow,256> table{};
    for(unsigned mask=0;mask<256;++mask)
        for(unsigned lane=0;lane<4;++lane) table[mask].lanes[(mask>>(lane*2))&3u][lane]=~0u;
    return table;
}();
template<bool Scalar8, bool Unsigned>
void unpackVif1Masked(uint8_t *memory,const uint8_t *src,uint32_t count,
                      uint32_t dest,uint32_t cl,uint32_t wl,uint32_t mask,
                      const uint32_t *row,const uint32_t *col) {
    struct Selection {__m128i data,keep,fixed;};
    Selection selections[4];
    const __m128i rowValues=_mm_loadu_si128(reinterpret_cast<const __m128i*>(row));
    const uint32_t rows=std::min({wl,4u,count});
    for(uint32_t cycle=0;cycle<rows;++cycle) {
        const auto &m=s_vifMaskRows[(mask>>(cycle*8u))&255u];
        __m128i data=_mm_load_si128(reinterpret_cast<const __m128i*>(m.lanes[0]));
        __m128i keep=_mm_load_si128(reinterpret_cast<const __m128i*>(m.lanes[3]));
        if constexpr(!Scalar8) {
            const __m128i low=_mm_set_epi32(0,0,-1,-1);
            keep=_mm_or_si128(keep,_mm_andnot_si128(low,data));
            data=_mm_and_si128(data,low);
        }
        const __m128i fixed=_mm_or_si128(
            _mm_and_si128(rowValues,_mm_load_si128(reinterpret_cast<const __m128i*>(m.lanes[1]))),
            _mm_and_si128(_mm_set1_epi32(int32_t(col[cycle])),_mm_load_si128(reinterpret_cast<const __m128i*>(m.lanes[2]))));
        selections[cycle]={data,keep,fixed};
    }
    auto write=[&](const Selection &select) {
        __m128i values;
        if constexpr(Scalar8) {
            const uint8_t raw=*src++;
            values=_mm_set1_epi32(Unsigned ? int32_t(raw) : int32_t(int8_t(raw)));
        } else {
            uint32_t packed;std::memcpy(&packed,src,4u);src+=4u;
            values=_mm_unpacklo_epi16(_mm_cvtsi32_si128(int32_t(packed)),_mm_setzero_si128());
            if constexpr(!Unsigned) values=_mm_srai_epi32(_mm_slli_epi32(values,16),16);
        }
        auto *target=reinterpret_cast<__m128i*>(memory+dest*16u);
        const __m128i prior=_mm_loadu_si128(target);
        const __m128i result=_mm_or_si128(select.fixed,_mm_or_si128(
            _mm_and_si128(values,select.data),_mm_and_si128(prior,select.keep)));
        _mm_storeu_si128(target,result);
    };
    if(wl==1u) {
        for(uint32_t i=0;i<count;++i) {write(selections[0]);dest=(dest+cl)&1023u;}
    } else {
        uint32_t cycle=0;
        for(uint32_t i=0;i<count;++i) {
            write(selections[std::min(cycle,3u)]);dest=(dest+1u)&1023u;
            if(++cycle==wl) {dest=(dest+cl-wl)&1023u;cycle=0;}
        }
    }
}

// Full V4 writes only: no mask, add/accumulate, or source/destination overlap.
// Split at STCYCL gaps and VU memory wrapping; read exactly one source vector.
template<unsigned Vl, bool Unsigned>
void unpackVif1V4(uint8_t *memory, const uint8_t *src, uint32_t count,
                 uint32_t dest, uint32_t cl, uint32_t wl)
{
    if constexpr(Vl==0u) {
        if(cl==wl) {
            const uint32_t first=std::min(count,1024u-dest);
            std::memcpy(memory+dest*16u,src,first*16u);
            if(first<count) std::memcpy(memory,src+first*16u,(count-first)*16u);
            return;
        }
    }
    auto write=[&](uint32_t address) {
        __m128i lanes;
        if constexpr(Vl==0u) {
            lanes=_mm_loadu_si128(reinterpret_cast<const __m128i*>(src));src+=16u;
        } else if constexpr(Vl==1u) {
            lanes=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(src));
            lanes=_mm_unpacklo_epi16(lanes,_mm_setzero_si128());
            if constexpr(!Unsigned) lanes=_mm_srai_epi32(_mm_slli_epi32(lanes,16),16);
            src+=8u;
        } else {
            uint32_t packed; std::memcpy(&packed,src,4u);
            lanes=_mm_cvtsi32_si128(static_cast<int32_t>(packed));
            lanes=_mm_unpacklo_epi8(lanes,_mm_setzero_si128());
            lanes=_mm_unpacklo_epi16(lanes,_mm_setzero_si128());
            if constexpr(!Unsigned) lanes=_mm_srai_epi32(_mm_slli_epi32(lanes,24),24);
            src+=4u;
        }
        _mm_storeu_si128(reinterpret_cast<__m128i*>(memory+address*16u),lanes);
    };
    if(wl==1u) {
        for(uint32_t i=0;i<count;++i) {write(dest);dest=(dest+cl)&1023u;}
    } else {
        uint32_t remaining=wl;
        for(uint32_t i=0;i<count;++i) {
            write(dest);dest=(dest+1u)&1023u;
            if(--remaining==0u) {dest=(dest+cl-wl)&1023u;remaining=wl;}
        }
    }
}
#endif

    constexpr uint8_t kGifFmtPacked = 0u;
    constexpr uint8_t kGifFmtReglist = 1u;
    constexpr uint8_t kGifFmtImage = 2u;

    // Walk the GIFtags in a DIRECT/DIRECTHL block. If the block ends part-way
    // through an IMAGE-format payload, return the number of image qwords still
    // owed to GIF PATH2 (which will arrive in a later VIF1 packet as raw data,
    // NOT as VIFcodes). Returns 0 if the block contains no unfinished IMAGE
    // payload. Handles an IMAGE tag that is preceded by PACKED/REGLIST tags
    // (e.g. a BITBLTBUF/TRXPOS/TRXREG/TRXDIR A+D setup, as War of the Monsters
    // sends before every texture upload).
    uint32_t pendingImageQwcAfterDirectBlock(const uint8_t *data, uint32_t sizeBytes)
    {
        if (!data)
            return 0u;

        uint32_t offset = 0u;
        while (offset + 16u <= sizeBytes)
        {
            uint64_t tagLo = 0u;
            std::memcpy(&tagLo, data + offset, sizeof(tagLo));
            offset += 16u;

            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
            const uint8_t flg = static_cast<uint8_t>((tagLo >> 58) & 0x3u);
            uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;

            if (flg == kGifFmtImage)
            {
                const uint64_t imageBytes = static_cast<uint64_t>(nloop) * 16ull;
                if (static_cast<uint64_t>(offset) + imageBytes <= sizeBytes)
                {
                    offset += static_cast<uint32_t>(imageBytes); // fully inline
                    continue;
                }
                const uint32_t inlineQw = (sizeBytes - offset) / 16u;
                return (nloop > inlineQw) ? (nloop - inlineQw) : 0u;
            }
            else if (flg == kGifFmtReglist)
            {
                uint64_t bytes = static_cast<uint64_t>(nloop) * nreg * 8ull;
                bytes = (bytes + 15ull) & ~15ull; // qword aligned
                offset += static_cast<uint32_t>(bytes);
            }
            else // PACKED (0) or DISABLE (3): nloop*nreg qwords of data
            {
                offset += nloop * nreg * 16u;
            }
        }
        return 0u;
    }
}

void PS2Memory::processVIF0Data(uint32_t srcPhys, uint32_t sizeBytes)
{
    if (sizeBytes == 0u || srcPhys >= PS2_RAM_SIZE)
        return;

    const uint64_t requestedEnd = static_cast<uint64_t>(srcPhys) + static_cast<uint64_t>(sizeBytes);
    if (requestedEnd > static_cast<uint64_t>(PS2_RAM_SIZE))
        sizeBytes = PS2_RAM_SIZE - srcPhys;

    processVIF0Data(m_rdram + srcPhys, sizeBytes);
}

void PS2Memory::processVIF0Data(const uint8_t *data, uint32_t sizeBytes)
{
    if (sizeBytes == 0u)
        return;

    uint32_t pos = 0;
    while (pos + 4 <= sizeBytes)
    {
        uint32_t cmd = 0u;
        std::memcpy(&cmd, data + pos, sizeof(cmd));
        pos += 4u;

        const uint8_t opcode = static_cast<uint8_t>((cmd >> 24) & 0x7Fu);
        const uint16_t imm = static_cast<uint16_t>(cmd & 0xFFFFu);
        const uint8_t num = static_cast<uint8_t>((cmd >> 16) & 0xFFu);
        const bool irq = (cmd & 0x80000000u) != 0u;

        vif0_regs.code = cmd;
        vif0_regs.num = num;
        if (irq)
            vif0_regs.stat |= (1u << 11);

        if (opcode == VIF_NOP)
        {
            continue;
        }
        else if (opcode == VIF_STCYCL)
        {
            vif0_regs.cycle = imm;
            continue;
        }
        else if (opcode == VIF_ITOP)
        {
            vif0_regs.itops = imm & 0x3FFu;
            continue;
        }
        else if (opcode == VIF_STMOD)
        {
            vif0_regs.mode = imm & 3u;
            continue;
        }
        else if (opcode == VIF_MARK)
        {
            vif0_regs.mark = imm;
            vif0_regs.stat |= (1u << 6);
            continue;
        }
        else if (opcode == VIF_FLUSHE || opcode == VIF_FLUSH || opcode == VIF_FLUSHA)
        {
            continue;
        }
        else if (opcode == VIF_STMASK)
        {
            if (pos + 4u > sizeBytes)
                break;
            std::memcpy(&vif0_regs.mask, data + pos, sizeof(vif0_regs.mask));
            pos += 4u;
            continue;
        }
        else if (opcode == VIF_STROW)
        {
            if (pos + 16u > sizeBytes)
                break;
            std::memcpy(vif0_regs.row, data + pos, 16u);
            pos += 16u;
            continue;
        }
        else if (opcode == VIF_STCOL)
        {
            if (pos + 16u > sizeBytes)
                break;
            std::memcpy(vif0_regs.col, data + pos, 16u);
            pos += 16u;
            continue;
        }
        else if (opcode == VIF_MPG)
        {
            const uint32_t destAddr = static_cast<uint32_t>(imm & 0x1FFu) * 8u;
            const uint32_t instructionCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);
            const uint32_t mpgBytes = instructionCount * 8u;
            uint32_t copyBytes = 0u;
            if (m_vu0Code && destAddr < PS2_VU0_CODE_SIZE && mpgBytes > 0u)
            {
                copyBytes = mpgBytes;
                if (destAddr + copyBytes > PS2_VU0_CODE_SIZE)
                    copyBytes = PS2_VU0_CODE_SIZE - destAddr;
                if (pos + copyBytes <= sizeBytes)
                {
                    std::memcpy(m_vu0Code + destAddr, data + pos, copyBytes);
                    markVU0CodeModified();
                }
            }

            pos += mpgBytes;
            if (pos > sizeBytes)
                break;
            continue;
        }
        else if ((opcode & 0x60u) == 0x60u)
        {
            const uint8_t vn = static_cast<uint8_t>((opcode >> 2) & 0x3u);
            const uint8_t vl = static_cast<uint8_t>(opcode & 0x3u);
            const int components = static_cast<int>(vn) + 1;
            int bitsPerComponent = 32;
            switch (vl)
            {
            case 0:
                bitsPerComponent = 32;
                break;
            case 1:
                bitsPerComponent = 16;
                break;
            case 2:
                bitsPerComponent = 8;
                break;
            case 3:
                bitsPerComponent = (vn == 3u) ? 4 : 16;
                break;
            default:
                break;
            }
            const int bitsPerVector = (vl == 3u && vn == 3u) ? 16 : (components * bitsPerComponent);
            uint32_t bytesPerVector = static_cast<uint32_t>((bitsPerVector + 7) / 8);
            const uint32_t writeVectorCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);
            uint32_t cl = vif0_regs.cycle & 0xFFu;
            uint32_t wl = (vif0_regs.cycle >> 8) & 0xFFu;
            if (cl == 0u)
                cl = 1u;
            if (wl == 0u)
                wl = 1u;
            uint32_t sourceVectorCount = writeVectorCount;
            if (cl < wl)
            {
                const uint32_t fullBlocks = writeVectorCount / wl;
                uint32_t remainder = writeVectorCount % wl;
                if (remainder > cl)
                    remainder = cl;
                sourceVectorCount = fullBlocks * cl + remainder;
            }
            uint32_t totalBytes = sourceVectorCount * bytesPerVector;
            totalBytes = (totalBytes + 3u) & ~3u;

            if (m_vu0Data && pos + totalBytes <= sizeBytes && vl == 0u)
            {
                uint32_t vuAddr = static_cast<uint32_t>(imm & 0x3FFu);
                if ((imm & 0x8000u) != 0u)
                    vuAddr = (vuAddr + (vif0_regs.tops & 0x3FFu)) & 0x3FFu;
                const uint8_t *srcBase = data + pos;
                uint32_t srcIndex = 0u;
                for (uint32_t writeIndex = 0; writeIndex < writeVectorCount; ++writeIndex)
                {
                    const uint32_t cyclePos = writeIndex % wl;
                    const bool sourceAvailable = (cl >= wl) || (cyclePos < cl);
                    uint32_t destVec = (cl >= wl) ? ((vuAddr + (writeIndex / wl) * cl + cyclePos) & 0x3FFu)
                                                  : ((vuAddr + writeIndex) & 0x3FFu);
                    const uint32_t destOff = destVec * 16u;
                    if (destOff + 16u > PS2_VU0_DATA_SIZE)
                    {
                        if (sourceAvailable && srcIndex < sourceVectorCount)
                            ++srcIndex;
                        continue;
                    }
                    if (!sourceAvailable || srcIndex >= sourceVectorCount)
                        continue;
                    const uint8_t *srcVec = srcBase + srcIndex * bytesPerVector;
                    ++srcIndex;
                    uint32_t lanes[4] = {0u, 0u, 0u, 0u};
                    std::memcpy(lanes, m_vu0Data + destOff, sizeof(lanes));
                    const uint32_t limit = (components > 4) ? 4u : static_cast<uint32_t>(components);
                    for (uint32_t c = 0; c < limit; ++c)
                    {
                        uint32_t scalar = 0u;
                        std::memcpy(&scalar, srcVec + c * 4u, sizeof(scalar));
                        lanes[c] = scalar;
                    }
                    _mm_storeu_si128(reinterpret_cast<__m128i *>(m_vu0Data + destOff), _mm_loadu_si128(reinterpret_cast<const __m128i *>(lanes)));
                }
            }
            pos += totalBytes;
            if (pos > sizeBytes)
                break;
            continue;
        }
        else
        {
            break;
        }
    }
}

void PS2Memory::processVIF1Data(uint32_t srcPhys, uint32_t sizeBytes)
{
    if (sizeBytes == 0u || srcPhys >= PS2_RAM_SIZE)
        return;

    const uint64_t requestedEnd = static_cast<uint64_t>(srcPhys) + static_cast<uint64_t>(sizeBytes);
    if (requestedEnd > static_cast<uint64_t>(PS2_RAM_SIZE))
        sizeBytes = PS2_RAM_SIZE - srcPhys;

    processVIF1Data(m_rdram + srcPhys, sizeBytes);
}

struct Ps2xPerf { std::atomic<uint64_t> rasterNs, presentNs, vu1Ns, vif1Ns, prims; };
extern Ps2xPerf g_ps2xPerf;

void PS2Memory::processVIF1Data(const uint8_t *data, uint32_t sizeBytes)
{
    struct PerfScope
    {
        Ps2xTscClock::time_point t0 = Ps2xTscClock::now();
        ~PerfScope()
        {
            g_ps2xPerf.vif1Ns.fetch_add(
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Ps2xTscClock::now() - t0).count()),
                std::memory_order_relaxed);
        }
    } perfScope;
    if (sizeBytes == 0u)
        return;

#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    {
        static std::atomic<uint32_t> s_v1{0u};
        const uint32_t i = s_v1.fetch_add(1u, std::memory_order_relaxed);
        if (i < 30u)
        {
            char b[64] = {0};
            const uint32_t n = std::min<uint32_t>(sizeBytes, 24u);
            for (uint32_t k = 0; k < n; ++k)
                std::snprintf(b + k * 2, 3, "%02x", data[k]);
            std::cerr << "[vif1:data] #" << i << " size=" << sizeBytes
                      << " first=" << b << std::endl;
        }
    }
#endif

    const bool traceMotion = MotionProvenance::enabled();
    const auto motionSpans = MotionProvenance::take(data);
    uint32_t pos = 0;

    while (pos + 4 <= sizeBytes)
    {
        if (m_vif1PendingPath2ImageQwc != 0u)
        {
            const uint32_t availableQw = (sizeBytes - pos) / 16u;
            if (availableQw == 0u)
            {
                break;
            }

            const uint32_t chunkQw = std::min<uint32_t>(m_vif1PendingPath2ImageQwc, availableQw);
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
            {
                static std::atomic<uint32_t> s_ic{0u};
                if (s_ic.fetch_add(1u, std::memory_order_relaxed) < 64u)
                    std::fprintf(stderr, "[vif1:imgcont] owedQw=%u chunkQw=%u availQw=%u pktBytes=%u pos=%u\n",
                                 m_vif1PendingPath2ImageQwc, chunkQw, availableQw, sizeBytes, pos);
            }
#endif
            std::vector<uint8_t> imagePacket(16u + static_cast<size_t>(chunkQw) * 16u, 0u);
            const uint64_t imageTag =
                static_cast<uint64_t>(chunkQw & 0x7FFFu) |
                ((m_vif1PendingPath2ImageQwc == chunkQw) ? (1ull << 15) : 0ull) |
                (static_cast<uint64_t>(kGifFmtImage) << 58);
            std::memcpy(imagePacket.data(), &imageTag, sizeof(imageTag));
            std::memcpy(imagePacket.data() + 16u, data + pos, static_cast<size_t>(chunkQw) * 16u);
            submitGifPacket(GifPathId::Path2,
                            imagePacket.data(),
                            static_cast<uint32_t>(imagePacket.size()),
                            true,
                            m_vif1PendingPath2DirectHl);

            pos += chunkQw * 16u;
            m_vif1PendingPath2ImageQwc -= chunkQw;
            if (m_vif1PendingPath2ImageQwc == 0u)
            {
                m_vif1PendingPath2DirectHl = false;
            }
            continue;
        }

        uint32_t cmd;
        memcpy(&cmd, data + pos, 4);
        pos += 4;

        uint8_t opcode = (cmd >> 24) & 0x7F;
        uint16_t imm = cmd & 0xFFFF;
        uint8_t num = (cmd >> 16) & 0xFF;
        const bool irq = (cmd & 0x80000000u) != 0u;


        // Track most-recent command for VIFn_CODE emulation.
        vif1_regs.code = cmd;
        vif1_regs.num = num;
        if (irq)
            vif1_regs.stat |= (1u << 11); // INT

        if (opcode == VIF_NOP)
        {
            continue;
        }
        else if (opcode == VIF_STCYCL)
        {
            vif1_regs.cycle = imm;
            continue;
        }
        else if (opcode == VIF_OFFSET)
        {
            // VIF double-buffer setup. OFFSET clears DBF and resets TOPS to BASE.
            // Do not rewrite BASE from the previous TOPS value.
            vif1_regs.ofst = imm & 0x3FFu;
            vif1_regs.tops = vif1_regs.base & 0x3FFu;
            vif1_regs.stat &= ~(1u << 7); // clear DBF
            continue;
        }
        else if (opcode == VIF_BASE)
        {
            // BASE only updates the base register. TOPS changes on OFFSET/MSCAL.
            vif1_regs.base = imm & 0x3FFu;
            continue;
        }
        else if (opcode == VIF_ITOP)
        {
            // ITOP VIFcode writes pending ITOPS; VU XITOP observes it after MSCAL/MSCNT.
            vif1_regs.itops = imm & 0x3FFu;
            continue;
        }
        else if (opcode == VIF_STMOD)
        {
            vif1_regs.mode = imm & 3u;
            continue;
        }
        else if (opcode == VIF_MSKPATH3)
        {
            // VIF command docs: MSKPATH3 uses IMMEDIATE bit 15.
            const bool wasMasked = m_path3Masked;
            m_path3Masked = (imm & 0x8000u) != 0u;
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
            {
                static std::atomic<uint32_t> s_msk{0u};
                if (s_msk.fetch_add(1u, std::memory_order_relaxed) < 64u)
                    std::fprintf(stderr, "[vif1:mskpath3] masked %d -> %d fifo=%zu\n",
                                 wasMasked ? 1 : 0, m_path3Masked ? 1 : 0, m_path3MaskedFifo.size());
            }
#endif
            if (wasMasked && !m_path3Masked)
                flushMaskedPath3Packets();
            continue;
        }
        else if (opcode == VIF_MARK)
        {
            vif1_regs.mark = imm;
            vif1_regs.stat |= (1u << 6); // MRK
            continue;
        }
        else if (opcode == VIF_FLUSHE || opcode == VIF_FLUSH || opcode == VIF_FLUSHA)
        {
            continue;
        }
        else if (opcode == VIF_MSCAL || opcode == VIF_MSCALF)
        {
            uint32_t startPC = (uint32_t)imm * 8u;

            // Values visible to the VU program for this MSCAL.
            // DobieStation semantics: ITOP = ITOPS; TOP = current TOPS;
            // then TOPS/DBF are prepared for the next buffer.
            const uint32_t runTop = vif1_regs.tops & 0x3FFu;
            const uint32_t runItop = vif1_regs.itops & 0x3FFu;
            vif1_regs.top = runTop;
            vif1_regs.itop = runItop;

            const bool dbf = (vif1_regs.stat & (1u << 7)) != 0u;
            if (dbf)
                vif1_regs.tops = vif1_regs.base & 0x3FFu;
            else
                vif1_regs.tops = (vif1_regs.base + vif1_regs.ofst) & 0x3FFu;
            vif1_regs.stat ^= (1u << 7); // toggle DBF

#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
            {
                static std::atomic<uint32_t> s_ms{0u};
                if (s_ms.fetch_add(1u, std::memory_order_relaxed) < 16u)
                    std::cerr << "[vif1:mscal] startPC=0x" << std::hex << startPC << std::dec
                              << " top=" << runTop << " itop=" << runItop
                              << " cbSet=" << (m_vu1MscalCallback ? 1 : 0) << std::endl;
            }
#endif
            if (m_vu1MscalCallback) {
                if(traceMotion) {
                    MotionProvenance::Run provenance(motionSpans,pos-4,opcode,startPC);
                    m_vu1MscalCallback(startPC, runTop, runItop);
                } else m_vu1MscalCallback(startPC, runTop, runItop);
            }
            continue;
        }
        else if (opcode == VIF_MSCNT)
        {
            const uint32_t runTop = vif1_regs.tops & 0x3FFu;
            const uint32_t runItop = vif1_regs.itops & 0x3FFu;
            vif1_regs.top = runTop;
            vif1_regs.itop = runItop;

            const bool dbf = (vif1_regs.stat & (1u << 7)) != 0u;
            if (dbf)
                vif1_regs.tops = vif1_regs.base & 0x3FFu;
            else
                vif1_regs.tops = (vif1_regs.base + vif1_regs.ofst) & 0x3FFu;
            vif1_regs.stat ^= (1u << 7); // toggle DBF

            if (m_vu1MscntCallback) {
                if(traceMotion) {
                    MotionProvenance::Run provenance(motionSpans,pos-4,opcode,0xffffffffu);
                    m_vu1MscntCallback(runTop, runItop);
                } else m_vu1MscntCallback(runTop, runItop);
            }
            continue;
        }
        else if (opcode == VIF_STMASK)
        {
            if (pos + 4 > sizeBytes)
                break;
            uint32_t maskValue = 0;
            std::memcpy(&maskValue, data + pos, sizeof(maskValue));
            vif1_regs.mask = maskValue;
            pos += 4;
            continue;
        }
        else if (opcode == VIF_STROW)
        {
            if (pos + 16 > sizeBytes)
                break;
            std::memcpy(vif1_regs.row, data + pos, 16);
            pos += 16;
            continue;
        }
        else if (opcode == VIF_STCOL)
        {
            if (pos + 16 > sizeBytes)
                break;
            std::memcpy(vif1_regs.col, data + pos, 16);
            pos += 16;
            continue;
        }
        else if (opcode == VIF_MPG)
        {
            uint32_t destAddr = (uint32_t)imm * 8u;
            // VIF MPG semantics: NUM==0 means 256 instructions (2048 bytes).
            // MPG payload is instruction-packed and should not be QW-aligned.
            const uint32_t instructionCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);
            const uint32_t mpgBytes = instructionCount * 8u;
            bool mpgOk = false;
            if (m_vu1Code && destAddr < PS2_VU1_CODE_SIZE && mpgBytes > 0)
            {
                uint32_t copyBytes = mpgBytes;
                if (destAddr + copyBytes > PS2_VU1_CODE_SIZE)
                    copyBytes = PS2_VU1_CODE_SIZE - destAddr;
                if (pos + copyBytes <= sizeBytes)
                {
                    std::memcpy(m_vu1Code + destAddr, data + pos, copyBytes);
                    markVU1CodeModified();
                    mpgOk = true;
                }
            }
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
            {
                static std::atomic<uint32_t> s_mpg{0u};
                if (s_mpg.fetch_add(1u, std::memory_order_relaxed) < 16u)
                    std::cerr << "[vif1:mpg] dest=0x" << std::hex << destAddr << std::dec
                              << " instrs=" << instructionCount << " bytes=" << mpgBytes
                              << " avail=" << (sizeBytes - pos) << " ok=" << (mpgOk ? 1 : 0)
                              << " vu1Code=" << (m_vu1Code ? 1 : 0)
                              << " codeSize=" << PS2_VU1_CODE_SIZE << std::endl;
            }
#endif
            pos += mpgBytes;
            if (pos > sizeBytes)
                break;
            continue;
        }
        else if (opcode == VIF_DIRECT || opcode == VIF_DIRECTHL)
        {
            uint32_t qwCount = imm;
            if (qwCount == 0)
                qwCount = 65536;
            const uint32_t availableQw = (sizeBytes - pos) / 16u;
            const bool truncated = qwCount > availableQw;
            if (qwCount > availableQw)
                qwCount = availableQw;

            if (qwCount > 0)
            {
                const bool directHl = (opcode == VIF_DIRECTHL);
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
                {
                    static std::atomic<uint32_t> s_dir{0u};
                    if (s_dir.fetch_add(1u, std::memory_order_relaxed) < 96u)
                    {
                        uint64_t q0 = 0u;
                        std::memcpy(&q0, data + pos, 8u);
                        uint32_t before[4] = {0, 0, 0, 0};
                        const uint32_t nb = (pos >= 20u) ? 4u : (pos - 4u) / 4u;
                        if (nb)
                            std::memcpy(before + (4u - nb), data + pos - 4u - nb * 4u, nb * 4u);
                        std::fprintf(stderr, "[vif1:direct] %s imm=%u qw=%u availQw=%u truncated=%d pktBytes=%u pos=%u before=[%08x %08x %08x %08x] q0=%016llx\n",
                                     directHl ? "DIRECTHL" : "DIRECT", (unsigned)imm, qwCount, availableQw,
                                     truncated ? 1 : 0, sizeBytes, pos, before[0], before[1], before[2], before[3],
                                     (unsigned long long)q0);
                    }
                }
#endif
                submitGifPacket(GifPathId::Path2, data + pos, qwCount * 16, true, directHl);

                // If this DIRECT transfer is larger than what the current VIF1
                // packet holds and it ends mid-IMAGE-payload, the remaining image
                // qwords arrive in later packets as raw PATH2 data, not VIFcodes.
                if (truncated)
                {
                    const uint32_t owedImageQw =
                        pendingImageQwcAfterDirectBlock(data + pos, qwCount * 16u);
                    if (owedImageQw != 0u)
                    {
                        m_vif1PendingPath2ImageQwc = owedImageQw;
                        m_vif1PendingPath2DirectHl = directHl;
                    }
                }
            }

            pos += qwCount * 16;
            if (truncated)
            {
                pos = sizeBytes;
                break;
            }
            continue;
        }
        else if ((opcode & 0x60) == 0x60)
        {
            uint8_t vn = (opcode >> 2) & 0x3;
            uint8_t vl = opcode & 0x3;
            const bool maskEnable = (opcode & 0x10u) != 0u;
            int components = vn + 1;
            int bitsPerComponent = 32;
            switch (vl)
            {
            case 0:
                bitsPerComponent = 32;
                break;
            case 1:
                bitsPerComponent = 16;
                break;
            case 2:
                bitsPerComponent = 8;
                break;
            case 3:
                bitsPerComponent = (vn == 3) ? 4 : 16;
                break;
            default:
                break;
            }
            int bitsPerVector = (vl == 3 && vn == 3) ? 16 : (components * bitsPerComponent);
            uint32_t bytesPerVector = (bitsPerVector + 7) / 8;
            // UNPACK semantics: NUM is 8-bit and NUM==0 means 256 vectors (writes).
            const uint32_t writeVectorCount = (num == 0u) ? 256u : static_cast<uint32_t>(num);

            // STCYCL controls write cycles for UNPACK.
            uint32_t cl = vif1_regs.cycle & 0xFFu;
            uint32_t wl = (vif1_regs.cycle >> 8) & 0xFFu;
            if (cl == 0u)
                cl = 1u;
            if (wl == 0u)
                wl = 1u;

            uint32_t sourceVectorCount = writeVectorCount;
            if (cl < wl)
            {
                const uint32_t fullBlocks = writeVectorCount / wl;
                uint32_t remainder = writeVectorCount % wl;
                if (remainder > cl)
                    remainder = cl;
                sourceVectorCount = fullBlocks * cl + remainder;
            }

            uint32_t totalBytes = sourceVectorCount * bytesPerVector;
            totalBytes = (totalBytes + 3) & ~3u;

            uint32_t vuAddr = (uint32_t)imm & 0x3FFu;
            if ((imm & 0x8000u) != 0u)
                vuAddr = (vuAddr + (vif1_regs.tops & 0x3FFu)) & 0x3FFu;

            const bool zeroExtend = (imm & 0x4000u) != 0u;

#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
            {
                const bool unpSettled = (read32(0x6F7E8Cu) == 0u && read32(0x6F8464u) == 1u);
                static std::atomic<uint32_t> s_unp{0u};
                if (unpSettled && s_unp.fetch_add(1u, std::memory_order_relaxed) < 500u)
                {
                    // First source qword (16 raw bytes) so we can see WHAT is being
                    // UNPACKed to each VU1 matrix slot and whether it changes per frame.
                    char row[40] = "<oob>";
                    const bool fits = (pos + 16u <= sizeBytes);
                    if (fits)
                    {
                        const uint8_t *p = data + pos;
                        std::snprintf(row, sizeof(row),
                                      "%02x%02x%02x%02x_%02x%02x%02x%02x_%02x%02x%02x%02x_%02x%02x%02x%02x",
                                      p[0],p[1],p[2],p[3], p[4],p[5],p[6],p[7],
                                      p[8],p[9],p[10],p[11], p[12],p[13],p[14],p[15]);
                    }
                    std::fprintf(stderr, "[vif1:unpack] op=%02x vuAddr=0x%x num=%u wvc=%u imm=%04x tops=%x bytes=%u avail=%u src0=%s\n",
                                 (unsigned)opcode, vuAddr, (unsigned)num, writeVectorCount,
                                 (unsigned)imm, vif1_regs.tops & 0x3FFu, totalBytes,
                                 (unsigned)(sizeBytes - pos), row);
                }
                // [vif1:objmtx] — the per-object MVP unpack (vuAddr 0, V4-32, 4 vec).
                // Dump all 4 source rows so we can see if a real matrix ever reaches q0-q3.
                if (unpSettled && vuAddr == 0u && writeVectorCount == 4u && vl == 0u && vn == 3u)
                {
                    static std::atomic<uint32_t> s_om{0u};
                    if (s_om.fetch_add(1u, std::memory_order_relaxed) < 400u)
                    {
                        char rr[4][40] = {{0}};
                        for (int r = 0; r < 4; ++r)
                        {
                            const uint32_t o = pos + (uint32_t)r * 16u;
                            if (o + 16u <= sizeBytes)
                            {
                                const uint8_t *q = data + o;
                                std::snprintf(rr[r], sizeof(rr[r]), "%02x%02x%02x%02x_%02x%02x%02x%02x_%02x%02x%02x%02x_%02x%02x%02x%02x",
                                    q[0],q[1],q[2],q[3], q[4],q[5],q[6],q[7], q[8],q[9],q[10],q[11], q[12],q[13],q[14],q[15]);
                            }
                        }
                        std::fprintf(stderr, "[vif1:objmtx] avail=%u r0=%s r1=%s r2=%s r3=%s\n",
                                     (unsigned)(sizeBytes - pos), rr[0], rr[1], rr[2], rr[3]);
                    }
                }
            }
#endif

            // Fast path for the common UNPACK: no mask, no add/accumulate mode,
            // every write cycle has source data (CL >= WL), known format. Writes
            // exactly what the generic loop below writes. PS2X_VIF1_UNPACK_VERIFY=1
            // runs both and compares VU1 data memory.
            const uint32_t unpackMode = vif1_regs.mode & 3u;
            bool unpackFastOk = m_vu1Data && totalBytes > 0 && pos + totalBytes <= sizeBytes && !maskEnable &&
                                      unpackMode != 1u && unpackMode != 2u && cl >= wl && !(vl == 3u && vn != 3u);
            if(s_unpackProfile) {
                static uint64_t packets[256]{},vectors[256]{},total=0;
                const unsigned key=vn*4u+vl+(maskEnable?16u:0u)+unpackMode*32u+(cl<wl?128u:0u);
                ++packets[key];vectors[key]+=writeVectorCount;
                if(++total%1000000u==0u) for(unsigned i=0;i<256;++i) if(packets[i])
                    std::fprintf(stderr,"[vif1:formats] total=%llu format=%x mask=%u mode=%u fill=%u packets=%llu vectors=%llu\n",
                        total,i&15u,(i>>4)&1u,(i>>5)&3u,i>>7,packets[i],vectors[i]);
            }
            static const bool s_unpackVerify = []
            {
                const char *value = std::getenv("PS2X_VIF1_UNPACK_VERIFY");
                return value != nullptr && value[0] == '1';
            }();
            static const bool s_unpackFastOff = []
            {
                const char *value = std::getenv("PS2X_VIF1_UNPACK_FAST");
                return value != nullptr && value[0] == '0';
            }();
            static const bool s_unpackNative=[] {const char *p=std::getenv("PS2X_VIF1_UNPACK_NATIVE");return !p || p[0]!='0';}();
            bool maskedFastOk=false;
#if defined(_M_X64) || defined(__x86_64__)
            const uintptr_t maskedInput=reinterpret_cast<uintptr_t>(data+pos),maskedOutput=reinterpret_cast<uintptr_t>(m_vu1Data);
            maskedFastOk=s_unpackMasked && s_unpackNative && maskEnable && m_vu1Data && totalBytes>0 && pos+totalBytes<=sizeBytes &&
                unpackMode!=1u && unpackMode!=2u && cl>=wl && ((vn==0u && vl==2u) || (vn==1u && vl==1u)) &&
                (maskedInput+totalBytes<=maskedOutput || maskedOutput+PS2_VU1_DATA_SIZE<=maskedInput);
            unpackFastOk=unpackFastOk || maskedFastOk;
#endif
            bool unpackFastDone = false;
            thread_local std::vector<uint8_t> unpackBefore, unpackFastResult;
            if (unpackFastOk && !s_unpackFastOff)
            {
                if (s_unpackVerify)
                    unpackBefore.assign(m_vu1Data, m_vu1Data + PS2_VU1_DATA_SIZE);
                const uint8_t *src = data + pos;
                bool vectorDone=false;
#if defined(_M_X64) || defined(__x86_64__)
                const uintptr_t input=reinterpret_cast<uintptr_t>(src), output=reinterpret_cast<uintptr_t>(m_vu1Data);
                const bool separate=input+totalBytes<=output || output+PS2_VU1_DATA_SIZE<=input;
                if(maskedFastOk) {
                    if(vn==0u) {
                        if(zeroExtend) unpackVif1Masked<true,true>(m_vu1Data,src,writeVectorCount,vuAddr,cl,wl,vif1_regs.mask,vif1_regs.row,vif1_regs.col);
                        else unpackVif1Masked<true,false>(m_vu1Data,src,writeVectorCount,vuAddr,cl,wl,vif1_regs.mask,vif1_regs.row,vif1_regs.col);
                    } else {
                        if(zeroExtend) unpackVif1Masked<false,true>(m_vu1Data,src,writeVectorCount,vuAddr,cl,wl,vif1_regs.mask,vif1_regs.row,vif1_regs.col);
                        else unpackVif1Masked<false,false>(m_vu1Data,src,writeVectorCount,vuAddr,cl,wl,vif1_regs.mask,vif1_regs.row,vif1_regs.col);
                    }
                    vectorDone=true;
                }
                if(!vectorDone && s_unpackNative && components==4 && vl<=2u && separate) {
                    if(vl==0u) unpackVif1V4<0,true>(m_vu1Data,src,writeVectorCount,vuAddr,cl,wl);
                    else if(vl==1u) {
                        if(zeroExtend) unpackVif1V4<1,true>(m_vu1Data,src,writeVectorCount,vuAddr,cl,wl);
                        else unpackVif1V4<1,false>(m_vu1Data,src,writeVectorCount,vuAddr,cl,wl);
                    } else {
                        if(zeroExtend) unpackVif1V4<2,true>(m_vu1Data,src,writeVectorCount,vuAddr,cl,wl);
                        else unpackVif1V4<2,false>(m_vu1Data,src,writeVectorCount,vuAddr,cl,wl);
                    }
                    vectorDone=true;
                } else if(!vectorDone && s_unpackNative && s_unpackPartial && components<=3 && vl<=2u && separate) {
                    if(components==1) unpackVif1PartialFormat<1>(m_vu1Data,src,writeVectorCount,vuAddr,cl,wl,vl,zeroExtend);
                    else if(components==2) unpackVif1PartialFormat<2>(m_vu1Data,src,writeVectorCount,vuAddr,cl,wl,vl,zeroExtend);
                    else unpackVif1PartialFormat<3>(m_vu1Data,src,writeVectorCount,vuAddr,cl,wl,vl,zeroExtend);
                    vectorDone=true;
                }
#endif
                uint32_t destCursor=vuAddr, cycleRemaining=wl;
                for (uint32_t writeIndex = 0; !vectorDone && writeIndex < writeVectorCount; ++writeIndex, src += bytesPerVector)
                {
                    const uint32_t destVec = s_unpackNative ? destCursor : (cl == wl)
                                                 ? ((vuAddr + writeIndex) & 0x3FFu)
                                                 : ((vuAddr + (writeIndex / wl) * cl + (writeIndex % wl)) & 0x3FFu);
                    destCursor=(destCursor+1u)&1023u;
                    if(--cycleRemaining==0u) {destCursor=(destCursor+cl-wl)&1023u;cycleRemaining=wl;}

                    const uint32_t destOff = destVec * 16u;
                    if (destOff + 16u > PS2_VU1_DATA_SIZE)
                        continue;
                    uint8_t *dst = m_vu1Data + destOff;
                    if (vl == 0u)
                    {
                        if (components == 1)
                        {
                            for (uint32_t c = 0; c < 4u; ++c)
                                std::memcpy(dst + c * 4u, src, 4u);
                        }
                        else
                        {
                            std::memcpy(dst, src, static_cast<size_t>(components) * 4u);
                        }
                    }
                    else if (vl == 1u)
                    {
                        const uint32_t limit = static_cast<uint32_t>(components);
                        for (uint32_t c = 0; c < (components == 1 ? 1u : limit); ++c)
                        {
                            uint16_t raw = 0;
                            std::memcpy(&raw, src + c * 2u, 2u);
                            const uint32_t v = zeroExtend ? static_cast<uint32_t>(raw)
                                                          : static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(raw)));
                            if (components == 1)
                            {
                                for (uint32_t k = 0; k < 4u; ++k)
                                    std::memcpy(dst + k * 4u, &v, 4u);
                            }
                            else
                            {
                                std::memcpy(dst + c * 4u, &v, 4u);
                            }
                        }
                    }
                    else if (vl == 2u)
                    {
                        const uint32_t limit = static_cast<uint32_t>(components);
                        for (uint32_t c = 0; c < (components == 1 ? 1u : limit); ++c)
                        {
                            const uint8_t raw = src[c];
                            const uint32_t v = zeroExtend ? static_cast<uint32_t>(raw)
                                                          : static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(raw)));
                            if (components == 1)
                            {
                                for (uint32_t k = 0; k < 4u; ++k)
                                    std::memcpy(dst + k * 4u, &v, 4u);
                            }
                            else
                            {
                                std::memcpy(dst + c * 4u, &v, 4u);
                            }
                        }
                    }
                    else // vl == 3 && vn == 3: packed 5:5:5:1
                    {
                        uint16_t packed = 0;
                        std::memcpy(&packed, src, 2u);
                        const uint32_t lanes[4] = {packed & 0x1Fu, (packed >> 5) & 0x1Fu, (packed >> 10) & 0x1Fu,
                                                   (packed >> 15) & 0x01u};
                        std::memcpy(dst, lanes, 16u);
                    }
                }
                unpackFastDone = true;
                if (s_unpackVerify)
                {
                    unpackFastResult.assign(m_vu1Data, m_vu1Data + PS2_VU1_DATA_SIZE);
                    std::memcpy(m_vu1Data, unpackBefore.data(), PS2_VU1_DATA_SIZE);
                    unpackFastDone = false; // run the generic loop too, then compare
                }
            }
            if (!unpackFastDone && m_vu1Data && totalBytes > 0 && pos + totalBytes <= sizeBytes)
            {
                const uint8_t *srcBase = data + pos;
                uint32_t srcIndex = 0u;
                uint32_t destCursor=vuAddr, writeCycle=0u;
                for (uint32_t writeIndex = 0; writeIndex < writeVectorCount; ++writeIndex)
                {
                    const uint32_t cyclePos = s_unpackNative ? writeCycle : writeIndex % wl;
                    const bool sourceAvailable = (cl >= wl) || (cyclePos < cl);

                    uint32_t destVec = 0;
                    if (s_unpackNative) destVec=destCursor;
                    else if (cl >= wl)
                    {
                        destVec = (vuAddr + (writeIndex / wl) * cl + cyclePos) & 0x3FFu;
                    }
                    else
                    {
                        destVec = (vuAddr + writeIndex) & 0x3FFu;
                    }

                    // Advance before any continue, including fill/unknown-format paths.
                    destCursor=(destCursor+1u)&1023u;
                    if(++writeCycle==wl) {
                        writeCycle=0u;
                        if(cl>=wl) destCursor=(destCursor+cl-wl)&1023u;
                    }
                    if(s_unpackVerify) {
                        const uint32_t expected=(vuAddr+(cl>=wl ? (writeIndex/wl)*cl+writeIndex%wl : writeIndex))&1023u;
                        if(destVec!=expected || cyclePos!=writeIndex%wl) {
                            std::fprintf(stderr,"[vif1:address-verify] MISMATCH\n");std::abort();
                        }
                    }
                    uint32_t destOff = destVec * 16u;
                    if (destOff + 16u > PS2_VU1_DATA_SIZE)
                    {
                        if (sourceAvailable && srcIndex < sourceVectorCount)
                            ++srcIndex;
                        continue;
                    }

                    uint32_t lanes[4] = {0u, 0u, 0u, 0u};
                    std::memcpy(lanes, m_vu1Data + destOff, sizeof(lanes));
                    uint32_t decompressed[4] = {lanes[0], lanes[1], lanes[2], lanes[3]};
                    bool decoded = false;

                    const uint8_t *srcVec = nullptr;
                    if (sourceAvailable && srcIndex < sourceVectorCount)
                    {
                        srcVec = srcBase + srcIndex * bytesPerVector;
                        ++srcIndex;
                        decoded = true;
                    }

                    auto extend16 = [&](uint16_t raw) -> uint32_t
                    {
                        if (zeroExtend)
                            return static_cast<uint32_t>(raw);
                        return static_cast<uint32_t>(static_cast<int32_t>(static_cast<int16_t>(raw)));
                    };

                    auto extend8 = [&](uint8_t raw) -> uint32_t
                    {
                        if (zeroExtend)
                            return static_cast<uint32_t>(raw);
                        return static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(raw)));
                    };

                    bool handledFormat = true;
                    if (!decoded)
                    {
                        handledFormat = false;
                    }
                    else if (vl == 0u)
                    {
                        if (components == 1)
                        {
                            uint32_t scalar = 0;
                            std::memcpy(&scalar, srcVec, sizeof(scalar));
                            decompressed[0] = scalar;
                            decompressed[1] = scalar;
                            decompressed[2] = scalar;
                            decompressed[3] = scalar;
                        }
                        else
                        {
                            const uint32_t limit = (components > 4) ? 4u : static_cast<uint32_t>(components);
                            for (uint32_t c = 0; c < limit; ++c)
                            {
                                uint32_t scalar = 0;
                                std::memcpy(&scalar, srcVec + c * 4u, sizeof(scalar));
                                decompressed[c] = scalar;
                            }
                        }
                    }
                    else if (vl == 1u)
                    {
                        if (components == 1)
                        {
                            uint16_t raw = 0;
                            std::memcpy(&raw, srcVec, sizeof(raw));
                            const uint32_t scalar = extend16(raw);
                            decompressed[0] = scalar;
                            decompressed[1] = scalar;
                            decompressed[2] = scalar;
                            decompressed[3] = scalar;
                        }
                        else
                        {
                            const uint32_t limit = (components > 4) ? 4u : static_cast<uint32_t>(components);
                            for (uint32_t c = 0; c < limit; ++c)
                            {
                                uint16_t raw = 0;
                                std::memcpy(&raw, srcVec + c * 2u, sizeof(raw));
                                decompressed[c] = extend16(raw);
                            }
                        }
                    }
                    else if (vl == 2u)
                    {
                        if (components == 1)
                        {
                            const uint32_t scalar = extend8(srcVec[0]);
                            decompressed[0] = scalar;
                            decompressed[1] = scalar;
                            decompressed[2] = scalar;
                            decompressed[3] = scalar;
                        }
                        else
                        {
                            const uint32_t limit = (components > 4) ? 4u : static_cast<uint32_t>(components);
                            for (uint32_t c = 0; c < limit; ++c)
                            {
                                decompressed[c] = extend8(srcVec[c]);
                            }
                        }
                    }
                    else if (vl == 3u && vn == 3u)
                    {
                        // V4-5: packed color-like format in a single 16-bit value.
                        uint16_t packed = 0;
                        std::memcpy(&packed, srcVec, sizeof(packed));
                        decompressed[0] = packed & 0x1Fu;
                        decompressed[1] = (packed >> 5) & 0x1Fu;
                        decompressed[2] = (packed >> 10) & 0x1Fu;
                        decompressed[3] = (packed >> 15) & 0x01u;
                    }
                    else
                    {
                        handledFormat = false;
                    }

                    // Unknown compressed format fallback: preserve legacy raw-copy behavior.
                    if (!handledFormat && decoded && !maskEnable && (vif1_regs.mode == 0u || vif1_regs.mode == 3u))
                    {
                        uint32_t copyBytes = (bytesPerVector < 16u) ? bytesPerVector : 16u;
                        std::memcpy(m_vu1Data + destOff, srcVec, copyBytes);
                        continue;
                    }

                    const bool canAdd = (vl != 3u || vn != 3u);
                    const uint32_t mode = vif1_regs.mode & 3u;
                    const uint32_t colIdx = (cyclePos > 3u) ? 3u : cyclePos;
                    const uint32_t maskCycle = (cyclePos > 3u) ? 3u : cyclePos;

                    for (uint32_t field = 0u; field < 4u; ++field)
                    {
                        uint32_t maskSpec = 0u;
                        if (maskEnable)
                        {
                            const uint32_t shift = ((maskCycle * 4u) + field) * 2u;
                            maskSpec = (vif1_regs.mask >> shift) & 0x3u;
                        }

                        // In fill-write cycles with suspended source reads, treat raw-data selections as row-fill.
                        if (!decoded && maskSpec == 0u)
                            maskSpec = 1u;

                        uint32_t writeVal = lanes[field];
                        if (maskSpec == 0u)
                        {
                            if (handledFormat)
                            {
                                writeVal = decompressed[field];
                                if (canAdd && (mode == 1u || mode == 2u))
                                {
                                    writeVal = writeVal + vif1_regs.row[field];
                                    if (mode == 2u)
                                        vif1_regs.row[field] = writeVal;
                                }
                            }
                        }
                        else if (maskSpec == 1u)
                        {
                            writeVal = vif1_regs.row[field];
                        }
                        else if (maskSpec == 2u)
                        {
                            writeVal = vif1_regs.col[colIdx];
                        }
                        else
                        {
                            continue; // write-protect
                        }

                        lanes[field] = writeVal;
                    }

                    std::memcpy(m_vu1Data + destOff, lanes, sizeof(lanes));
                }
            }
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
            // Read back VU1 q0 right after any unpack that targeted vuAddr 0, so we
            // can see whether the per-object MVP write actually lands & survives.
            if (vuAddr == 0u && m_vu1Data && (read32(0x6F7E8Cu) == 0u && read32(0x6F8464u) == 1u))
            {
                static std::atomic<uint32_t> s_rb{0u};
                if (s_rb.fetch_add(1u, std::memory_order_relaxed) < 400u)
                {
                    const float *q0 = reinterpret_cast<const float *>(m_vu1Data);
                    const float *q3 = reinterpret_cast<const float *>(m_vu1Data + 48);
                    std::fprintf(stderr, "[vif1:q0rb] wvc=%u cl=%u wl=%u q0=(%g,%g,%g,%g) q3=(%g,%g,%g,%g)\n",
                                 writeVectorCount, (unsigned)(vif1_regs.cycle & 0xFFu),
                                 (unsigned)((vif1_regs.cycle >> 8) & 0xFFu),
                                 q0[0], q0[1], q0[2], q0[3], q3[0], q3[1], q3[2], q3[3]);
                }
            }
#endif
            if (s_unpackVerify && unpackFastOk && !s_unpackFastOff)
            {
                static uint64_t s_checked = 0u, s_bad = 0u;
                ++s_checked;
                if (std::memcmp(unpackFastResult.data(), m_vu1Data, PS2_VU1_DATA_SIZE) != 0)
                {
                    if (++s_bad <= 20u)
                        std::fprintf(stderr, "[vif1:unpack-verify] MISMATCH #%llu vl=%u vn=%u num=%u cl=%u wl=%u mode=%u\n",
                                     static_cast<unsigned long long>(s_bad), vl, vn, writeVectorCount, cl, wl, unpackMode);
                }
                if ((s_checked % 50000u) == 0u)
                    std::fprintf(stderr, "[vif1:unpack-verify] checked=%llu mismatched=%llu\n",
                                 static_cast<unsigned long long>(s_checked), static_cast<unsigned long long>(s_bad));
            }
            pos += totalBytes;

            if (pos > sizeBytes)
                break;
            continue;
        }
        else
        {
            continue;
        }
    }
}
