#include "Common.h"
#include "GS.h"
#include "ps2_log.h"
#include "runtime/gs/ps2_gs_common.h"
#include "runtime/gs/ps2_gs_psmct16.h"
#include "runtime/ee_scheduler.h"

#include <atomic>
extern std::atomic<uint64_t> g_ps2xSyncPathCalls; // ps2_runtime.cpp [perf] frames

namespace ps2_stubs
{
    namespace
    {
        uint64_t makeClearPrim(bool useContext2)
        {
            return static_cast<uint64_t>(GS_PRIM_SPRITE) |
                   (static_cast<uint64_t>(useContext2 ? 1u : 0u) << 9);
        }

        uint64_t makeClearRgbaq(uint32_t rgba)
        {
            return static_cast<uint64_t>(rgba);
        }

        uint64_t makeClearXyz(int32_t x, int32_t y)
        {
            return static_cast<uint64_t>(static_cast<uint16_t>(x << 4)) |
                   (static_cast<uint64_t>(static_cast<uint16_t>(y << 4)) << 16);
        }

        void seedGsClearPacket(GsClearMem &clear,
                               int32_t width,
                               int32_t height,
                               uint32_t rgba,
                               uint32_t ztest,
                               bool useContext2)
        {
            const int32_t offX = 0x800 - (width >> 1);
            const int32_t offY = 0x800 - (height >> 1);
            const uint64_t clearTest = makeTest(0u);
            const uint64_t restoreTest = makeTest(ztest);
            const uint64_t prim = makeClearPrim(useContext2);
            const uint64_t rgbaq = makeClearRgbaq(rgba);
            const uint64_t xyz0 = makeClearXyz(offX, offY);
            const uint64_t xyz1 = makeClearXyz(offX + width, offY + height);
            const uint64_t testReg = useContext2 ? GS_REG_TEST_2 : GS_REG_TEST_1;

            clear.testa = {clearTest, testReg};
            clear.prim = {prim, GS_REG_PRIM};
            clear.rgbaq = {rgbaq, GS_REG_RGBAQ};
            clear.xyz2a = {xyz0, GS_REG_XYZ2};
            clear.xyz2b = {xyz1, GS_REG_XYZ2};
            clear.testb = {restoreTest, testReg};
        }

        bool hasSeededGsClearPacket(const GsClearMem &clear)
        {
            return clear.rgbaq.reg == GS_REG_RGBAQ &&
                   clear.xyz2a.reg == GS_REG_XYZ2 &&
                   clear.xyz2b.reg == GS_REG_XYZ2;
        }

        struct GsTrailingArgs2
        {
            uint32_t arg0 = 0u;
            uint32_t arg1 = 0u;
        };

        struct GsTrailingArgs3
        {
            uint32_t arg0 = 0u;
            uint32_t arg1 = 0u;
            uint32_t arg2 = 0u;
        };

        GsTrailingArgs2 decodeGsTrailingArgs2(uint8_t *rdram, R5900Context *ctx)
        {
            const uint32_t reg8 = getRegU32(ctx, 8);
            const uint32_t reg9 = getRegU32(ctx, 9);
            const uint32_t stack0 = readStackU32(rdram, ctx, 16);
            const uint32_t stack1 = readStackU32(rdram, ctx, 20);

            const bool hasRegArgs = (reg8 != 0u || reg9 != 0u);
            const bool hasStackArgs = (stack0 != 0u || stack1 != 0u);
            if (hasRegArgs || !hasStackArgs)
            {
                return {reg8, reg9};
            }

            return {stack0, stack1};
        }

        GsTrailingArgs3 decodeGsTrailingArgs3(uint8_t *rdram, R5900Context *ctx)
        {
            const uint32_t reg8 = getRegU32(ctx, 8);
            const uint32_t reg9 = getRegU32(ctx, 9);
            const uint32_t reg10 = getRegU32(ctx, 10);
            const uint32_t stack0 = readStackU32(rdram, ctx, 16);
            const uint32_t stack1 = readStackU32(rdram, ctx, 20);
            const uint32_t stack2 = readStackU32(rdram, ctx, 24);

            const bool hasRegArgs = (reg8 != 0u || reg9 != 0u || reg10 != 0u);
            const bool hasStackArgs = (stack0 != 0u || stack1 != 0u || stack2 != 0u);
            if (hasRegArgs || !hasStackArgs)
            {
                return {reg8, reg9, reg10};
            }

            return {stack0, stack1, stack2};
        }

        void applyGsClearPacket(PS2Runtime *runtime, const GsClearMem &clear)
        {
            if (!runtime->syncCoreSubsystems() || !hasSeededGsClearPacket(clear))
            {
                return;
            }

            runtime->gs().writeRegister(static_cast<uint8_t>(clear.testa.reg & 0xFFu), clear.testa.value);
            runtime->gs().writeRegister(static_cast<uint8_t>(clear.prim.reg & 0xFFu), clear.prim.value);
            runtime->gs().writeRegister(static_cast<uint8_t>(clear.rgbaq.reg & 0xFFu), clear.rgbaq.value);
            runtime->gs().writeRegister(static_cast<uint8_t>(clear.xyz2a.reg & 0xFFu), clear.xyz2a.value);
            runtime->gs().writeRegister(static_cast<uint8_t>(clear.xyz2b.reg & 0xFFu), clear.xyz2b.value);
            runtime->gs().writeRegister(static_cast<uint8_t>(clear.testb.reg & 0xFFu), clear.testb.value);
        }

        void refreshPacketBuilderPendingCount(uint8_t *rdram, PS2Runtime *runtime, uint32_t stateAddr);
        void writePacketBuilderCurrent(uint8_t *rdram, PS2Runtime *runtime, uint32_t stateAddr, uint32_t currentAddr);

        void initPacketBuilderState(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
        {
            const uint32_t stateAddr = getRegU32(ctx, 4);
            const uint32_t baseAddr = getRegU32(ctx, 5);
            const uint32_t words[4] = {baseAddr, baseAddr, 0u, 0u};
            writeGuestBytes(rdram,
                            runtime,
                            stateAddr,
                            reinterpret_cast<const uint8_t *>(words),
                            sizeof(words));
        }

        uint32_t terminatePacketBuilderState(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
        {
            const uint32_t stateAddr = getRegU32(ctx, 4);
            uint32_t currentAddr = 0u;
            if (!tryReadWordFromGuest(rdram, runtime, stateAddr, currentAddr))
            {
                return 0u;
            }

            const uint32_t zero = 0u;
            while ((currentAddr & 0xCu) != 0u)
            {
                writeGuestBytes(rdram,
                                runtime,
                                currentAddr,
                                reinterpret_cast<const uint8_t *>(&zero),
                                sizeof(zero));
                currentAddr += 4u;
            }

            writePacketBuilderCurrent(rdram, runtime, stateAddr, currentAddr);
            writeGuestBytes(rdram,
                            runtime,
                            stateAddr + 8u,
                            reinterpret_cast<const uint8_t *>(&zero),
                            sizeof(zero));
            return currentAddr;
        }

        void resetPacketBuilderState(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
        {
            const uint32_t stateAddr = getRegU32(ctx, 4);
            uint32_t baseAddr = 0u;
            if (!tryReadWordFromGuest(rdram, runtime, stateAddr + 4u, baseAddr))
            {
                setReturnU32(ctx, 0u);
                return;
            }

            const uint32_t words[4] = {baseAddr, baseAddr, 0u, 0u};
            writeGuestBytes(rdram,
                            runtime,
                            stateAddr,
                            reinterpret_cast<const uint8_t *>(words),
                            sizeof(words));
            setReturnU32(ctx, baseAddr);
        }

        bool tryReadQwordFromGuest(uint8_t *rdram, PS2Runtime *runtime, uint32_t addr, uint64_t &outQword)
        {
            uint32_t low = 0u;
            uint32_t high = 0u;
            if (!tryReadWordFromGuest(rdram, runtime, addr, low) ||
                !tryReadWordFromGuest(rdram, runtime, addr + 4u, high))
            {
                return false;
            }

            outQword = static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32u);
            return true;
        }

        void writeGuestU32(uint8_t *rdram, PS2Runtime *runtime, uint32_t addr, uint32_t value)
        {
            writeGuestBytes(rdram,
                            runtime,
                            addr,
                            reinterpret_cast<const uint8_t *>(&value),
                            sizeof(value));
        }

        void writeGuestU64(uint8_t *rdram, PS2Runtime *runtime, uint32_t addr, uint64_t value)
        {
            writeGuestBytes(rdram,
                            runtime,
                            addr,
                            reinterpret_cast<const uint8_t *>(&value),
                            sizeof(value));
        }

        void writeGuestVec128(uint8_t *rdram, PS2Runtime *runtime, uint32_t addr, __m128i value)
        {
            alignas(16) __m128i temp = value;
            writeGuestBytes(rdram,
                            runtime,
                            addr,
                            reinterpret_cast<const uint8_t *>(&temp),
                            sizeof(temp));
        }

        void refreshPacketBuilderPendingCount(uint8_t *rdram, PS2Runtime *runtime, uint32_t stateAddr)
        {
            uint32_t currentAddr = 0u;
            uint32_t pendingCountAddr = 0u;
            if (!tryReadWordFromGuest(rdram, runtime, stateAddr, currentAddr) ||
                !tryReadWordFromGuest(rdram, runtime, stateAddr + 8u, pendingCountAddr) ||
                pendingCountAddr == 0u ||
                currentAddr <= pendingCountAddr)
            {
                return;
            }

            uint32_t countWord = 0u;
            if (!tryReadWordFromGuest(rdram, runtime, pendingCountAddr, countWord))
            {
                return;
            }

            const uint32_t deltaBytes = currentAddr - pendingCountAddr;
            uint32_t deltaQwords = 0u;
            if (deltaBytes >= 16u)
            {
                deltaQwords = (deltaBytes >> 4u) - 1u;
            }

            countWord = (countWord & 0xFFFF0000u) | (deltaQwords & 0xFFFFu);
            writeGuestU32(rdram, runtime, pendingCountAddr, countWord);
        }

        void writePacketBuilderCurrent(uint8_t *rdram, PS2Runtime *runtime, uint32_t stateAddr, uint32_t currentAddr)
        {
            writeGuestU32(rdram, runtime, stateAddr, currentAddr);
            refreshPacketBuilderPendingCount(rdram, runtime, stateAddr);
        }

        uint32_t reservePacketBuilderWords(uint8_t *rdram, PS2Runtime *runtime, uint32_t stateAddr, uint32_t wordCount)
        {
            uint32_t currentAddr = 0u;
            if (!tryReadWordFromGuest(rdram, runtime, stateAddr, currentAddr))
            {
                return 0u;
            }

            const uint32_t reservedAddr = currentAddr;
            currentAddr += wordCount * 4u;
            writePacketBuilderCurrent(rdram, runtime, stateAddr, currentAddr);
            return reservedAddr;
        }

        void alignPacketBuilderState(uint8_t *rdram,
                                     PS2Runtime *runtime,
                                     uint32_t stateAddr,
                                     uint32_t alignMode,
                                     uint32_t reserveWords)
        {
            uint32_t currentAddr = 0u;
            if (!tryReadWordFromGuest(rdram, runtime, stateAddr, currentAddr))
            {
                return;
            }

            const uint32_t adjusted = (alignMode + 2u) & 31u;
            const uint32_t shift = (32u - adjusted) & 31u;
            const uint32_t lowMask = 0xFFFFFFFFu >> shift;
            const uint32_t alignedBase = currentAddr & ~lowMask;
            uint32_t targetAddr = alignedBase + (reserveWords << 2u);
            if (targetAddr < currentAddr)
            {
                targetAddr = (targetAddr + 1u) + lowMask;
            }

            const uint32_t zero = 0u;
            while (currentAddr < targetAddr)
            {
                writeGuestU32(rdram, runtime, currentAddr, zero);
                currentAddr += 4u;
            }
            writePacketBuilderCurrent(rdram, runtime, stateAddr, currentAddr);
        }

        void openPacketGifTag(uint8_t *rdram,
                              R5900Context *ctx,
                              PS2Runtime *runtime,
                              uint32_t stateAddr,
                              uint32_t openAddrOffset)
        {
            uint32_t currentAddr = 0u;
            if (!tryReadWordFromGuest(rdram, runtime, stateAddr, currentAddr))
            {
                return;
            }

            writeGuestVec128(rdram, runtime, currentAddr, GPR_VEC(ctx, 5));
            writePacketBuilderCurrent(rdram, runtime, stateAddr, currentAddr + 16u);
            writeGuestU32(rdram, runtime, stateAddr + openAddrOffset, currentAddr);
        }

        void closePacketGifTag(uint8_t *rdram, PS2Runtime *runtime, uint32_t stateAddr, uint32_t openAddrOffset)
        {
            uint32_t openAddr = 0u;
            uint32_t currentAddr = 0u;
            if (!tryReadWordFromGuest(rdram, runtime, stateAddr + openAddrOffset, openAddr) ||
                !tryReadWordFromGuest(rdram, runtime, stateAddr, currentAddr) ||
                openAddr == 0u)
            {
                return;
            }

            uint64_t tagValue = 0u;
            if (!tryReadQwordFromGuest(rdram, runtime, openAddr, tagValue))
            {
                return;
            }

            uint32_t packetQwords = ((currentAddr - openAddr) >> 3u) - 2u;
            const uint32_t flag = static_cast<uint32_t>((tagValue >> 58u) & 0x3u);
            if (flag != 1u)
            {
                packetQwords >>= 1u;
            }
            if (flag != 2u)
            {
                uint32_t nreg = static_cast<uint32_t>((tagValue >> 60u) & 0xFu);
                if (nreg == 0u)
                {
                    nreg = 16u;
                }
                packetQwords = (packetQwords + nreg - 1u) / nreg;
            }

            tagValue += static_cast<uint64_t>(packetQwords);
            writeGuestU32(rdram, runtime, stateAddr + openAddrOffset, 0u);
            writeGuestU64(rdram, runtime, openAddr, tagValue);

            while ((currentAddr & 0xCu) != 0u)
            {
                writeGuestU32(rdram, runtime, currentAddr, 0u);
                currentAddr += 4u;
            }
            writePacketBuilderCurrent(rdram, runtime, stateAddr, currentAddr);
        }
    }

    void sceGifPkAddGsAD(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t stateAddr = getRegU32(ctx, 4);
        uint32_t currentAddr = 0u;
        if (!tryReadWordFromGuest(rdram, runtime, stateAddr, currentAddr))
        {
            return;
        }

        const uint64_t dataValue = GPR_U64(ctx, 6);
        const uint64_t regValue = static_cast<uint64_t>(getRegU32(ctx, 5));
        writeGuestU64(rdram, runtime, currentAddr, dataValue);
        writeGuestU64(rdram, runtime, currentAddr + 8u, regValue);
        writePacketBuilderCurrent(rdram, runtime, stateAddr, currentAddr + 16u);
    }

    void sceGifPkAddGsData(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t stateAddr = getRegU32(ctx, 4);
        uint32_t currentAddr = 0u;
        if (!tryReadWordFromGuest(rdram, runtime, stateAddr, currentAddr))
        {
            return;
        }

        writeGuestU64(rdram, runtime, currentAddr, GPR_U64(ctx, 5));
        writePacketBuilderCurrent(rdram, runtime, stateAddr, currentAddr + 8u);
    }

    void sceGifPkCloseGifTag(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)ctx;
        closePacketGifTag(rdram, runtime, getRegU32(ctx, 4), 12u);
    }

    void sceGifPkCnt(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t stateAddr = getRegU32(ctx, 4);
        const uint32_t countValue = getRegU32(ctx, 5);
        const uint32_t extraValue = getRegU32(ctx, 6);
        const uint32_t tagWord = getRegU32(ctx, 7) | 0x10000000u;
        const uint32_t packetAddr = terminatePacketBuilderState(rdram, ctx, runtime);
        const uint32_t words[4] = {tagWord, 0u, countValue, extraValue};
        const uint32_t nextAddr = packetAddr + 16u;

        writeGuestU32(rdram, runtime, stateAddr + 8u, packetAddr);
        writeGuestBytes(rdram,
                        runtime,
                        packetAddr,
                        reinterpret_cast<const uint8_t *>(words),
                        sizeof(words));
        writePacketBuilderCurrent(rdram, runtime, stateAddr, nextAddr);
    }

    void sceGifPkEnd(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t stateAddr = getRegU32(ctx, 4);
        const uint32_t countValue = getRegU32(ctx, 5);
        const uint32_t extraValue = getRegU32(ctx, 6);
        const uint32_t tagWord = getRegU32(ctx, 7) | 0x70000000u;
        const uint32_t packetAddr = terminatePacketBuilderState(rdram, ctx, runtime);
        const uint32_t words[4] = {tagWord, countValue, extraValue, 0u};
        const uint32_t nextAddr = packetAddr + 16u;

        writeGuestU32(rdram, runtime, stateAddr + 8u, packetAddr);
        writeGuestBytes(rdram,
                        runtime,
                        packetAddr,
                        reinterpret_cast<const uint8_t *>(words),
                        sizeof(words));
        writePacketBuilderCurrent(rdram, runtime, stateAddr, nextAddr);
    }

    void sceGifPkInit(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        initPacketBuilderState(rdram, ctx, runtime);
    }

    void sceGifPkOpenGifTag(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        openPacketGifTag(rdram, ctx, runtime, getRegU32(ctx, 4), 12u);
    }

    void sceGifPkRef(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t stateAddr = getRegU32(ctx, 4);
        const uint32_t refAddr = getRegU32(ctx, 5) & 0x9FFFFFFFu;
        const uint32_t tagWord = getRegU32(ctx, 9) | getRegU32(ctx, 6) | 0x30000000u;
        const uint32_t extra0 = getRegU32(ctx, 7);
        const uint32_t extra1 = getRegU32(ctx, 8);
        const uint32_t packetAddr = terminatePacketBuilderState(rdram, ctx, runtime);
        const uint32_t words[4] = {tagWord, refAddr, extra0, extra1};

        writeGuestBytes(rdram,
                        runtime,
                        packetAddr,
                        reinterpret_cast<const uint8_t *>(words),
                        sizeof(words));
        writePacketBuilderCurrent(rdram, runtime, stateAddr, packetAddr + 16u);
    }

    void sceGifPkRefLoadImage(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t stateAddr = getRegU32(ctx, 4);
        const uint32_t dbp = getRegU32(ctx, 5) & 0xFFFFu;
        const uint32_t dpsm = getRegU32(ctx, 6) & 0xFFu;
        const uint32_t dbw = getRegU32(ctx, 7) & 0xFFFFu;
        uint32_t dataAddr = getRegU32(ctx, 8);
        uint32_t qwcRemaining = getRegU32(ctx, 9);
        const uint32_t dsax = getRegU32(ctx, 10);
        const uint32_t dsay = getRegU32(ctx, 11);
        const uint32_t width = readStackU32(rdram, ctx, 0);
        const uint32_t height = readStackU32(rdram, ctx, 8);

        // Open a 4-register A+D GIF tag and emit the GS load-image setup.
        {
            const uint32_t packetAddr = terminatePacketBuilderState(rdram, ctx, runtime);
            const uint32_t words[4] = {0x10000000u, 0u, 0u, 0u};
            writeGuestU32(rdram, runtime, stateAddr + 8u, packetAddr);
            writeGuestBytes(rdram,
                            runtime,
                            packetAddr,
                            reinterpret_cast<const uint8_t *>(words),
                            sizeof(words));
            writePacketBuilderCurrent(rdram, runtime, stateAddr, packetAddr + 16u);

            // Seed an open A+D tag (nloop=0, EOP clear): closePacketGifTag adds the true
            // appended qword count, so a pre-set nloop would double-count. Open variant
            // (not makeGiftagAplusD) because that always sets EOP on this chained tag.
            const uint64_t giftag[2] = {makeGiftagAplusDOpen(0u), 0xEULL};
            uint32_t currentAddr = packetAddr + 16u;
            writeGuestBytes(rdram, runtime, currentAddr, reinterpret_cast<const uint8_t *>(giftag), sizeof(giftag));
            writePacketBuilderCurrent(rdram, runtime, stateAddr, currentAddr + 16u);
            writeGuestU32(rdram, runtime, stateAddr + 12u, currentAddr);

            const uint64_t bitbltbuf =
                (static_cast<uint64_t>(dbp) << 32u) |
                (static_cast<uint64_t>(dbw & 0xFFu) << 48u) |
                (static_cast<uint64_t>(dpsm) << 56u);
            const uint64_t trxpos =
                (static_cast<uint64_t>(dsax) << 32u) |
                (static_cast<uint64_t>(dsay) << 48u);
            const uint64_t trxreg =
                static_cast<uint64_t>(width) |
                (static_cast<uint64_t>(height) << 32u);

            {
                uint32_t addr = 0u;
                if (!tryReadWordFromGuest(rdram, runtime, stateAddr, addr))
                {
                    return;
                }
                writeGuestU64(rdram, runtime, addr, bitbltbuf);
                writeGuestU64(rdram, runtime, addr + 8u, static_cast<uint64_t>(GS_REG_BITBLTBUF));
                addr += 16u;
                writeGuestU64(rdram, runtime, addr, trxpos);
                writeGuestU64(rdram, runtime, addr + 8u, static_cast<uint64_t>(GS_REG_TRXPOS));
                addr += 16u;
                writeGuestU64(rdram, runtime, addr, trxreg);
                writeGuestU64(rdram, runtime, addr + 8u, static_cast<uint64_t>(GS_REG_TRXREG));
                addr += 16u;
                writeGuestU64(rdram, runtime, addr, 0u);
                writeGuestU64(rdram, runtime, addr + 8u, static_cast<uint64_t>(GS_REG_TRXDIR));
                addr += 16u;
                writePacketBuilderCurrent(rdram, runtime, stateAddr, addr);
                closePacketGifTag(rdram, runtime, stateAddr, 12u);
            }
        }

        while (qwcRemaining != 0u)
        {
            const uint32_t chunkQwc = std::min<uint32_t>(qwcRemaining, 32767u);

            const uint32_t packetAddr = terminatePacketBuilderState(rdram, ctx, runtime);
            const uint32_t words[4] = {0x10000000u, 0u, 0u, 0u};
            writeGuestU32(rdram, runtime, stateAddr + 8u, packetAddr);
            writeGuestBytes(rdram,
                            runtime,
                            packetAddr,
                            reinterpret_cast<const uint8_t *>(words),
                            sizeof(words));
            writePacketBuilderCurrent(rdram, runtime, stateAddr, packetAddr + 16u);

            const uint32_t reservedAddr = reservePacketBuilderWords(rdram, runtime, stateAddr, 4u);
            const bool isLastChunk = (chunkQwc == qwcRemaining);
            const uint64_t gifTag =
                static_cast<uint64_t>(chunkQwc) |
                (isLastChunk ? 0x0800000000008000ULL : 0x0800000000000000ULL);
            writeGuestU64(rdram, runtime, reservedAddr, gifTag);
            writeGuestU64(rdram, runtime, reservedAddr + 8u, 0u);

            const uint32_t refPacketAddr = terminatePacketBuilderState(rdram, ctx, runtime);
            const uint32_t refWords[4] = {0x30000000u | chunkQwc, dataAddr & 0x9FFFFFFFu, 0u, 0u};
            writeGuestBytes(rdram,
                            runtime,
                            refPacketAddr,
                            reinterpret_cast<const uint8_t *>(refWords),
                            sizeof(refWords));
            writePacketBuilderCurrent(rdram, runtime, stateAddr, refPacketAddr + 16u);

            qwcRemaining -= chunkQwc;
            dataAddr += chunkQwc * 16u;
        }
    }

    void sceGifPkReset(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        resetPacketBuilderState(rdram, ctx, runtime);
    }

    void sceGifPkReserve(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnU32(ctx, reservePacketBuilderWords(rdram, runtime, getRegU32(ctx, 4), getRegU32(ctx, 5)));
    }

    void sceGifPkTerminate(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnU32(ctx, terminatePacketBuilderState(rdram, ctx, runtime));
    }

    // libgraph (graphdev.c) image structs, byte-exact with the retail SDK code
    // linked into SCUS_971.97 (sceGsSetDefLoadImage @0x2314e8, ...StoreImage @0x2316d0).
    // Games DMA these structs themselves (WotM: texmSetupTexture builds a
    // sceGsLoadImage *inside* each texture record and later sends it with
    // VIF1 DIRECT), so the guest-visible layout must be the SDK one, not a
    // private summary struct.
    //
    // sceGsLoadImage  (96 bytes, 6 qw):
    //   q0 giftag  nloop=4 eop=0 nreg=1 regs=A+D
    //   q1 BITBLTBUF | 0x50      q2 TRXPOS | 0x51
    //   q3 TRXREG   | 0x52      q4 TRXDIR=0 | 0x53
    //   q5 giftag  IMAGE nloop=w*h*bpp/128 eop=1
    // sceGsStoreImage (112 bytes, 7 qw):
    //   q0 VIF: NOP, MSKPATH3(0x8000), FLUSHA, DIRECT 6
    //   q1 giftag nloop=5 eop=1 nreg=1 A+D
    //   q2 BITBLTBUF | 0x50  q3 TRXPOS | 0x51  q4 TRXREG | 0x52
    //   q5 0 | FINISH(0x61)  q6 TRXDIR=1 | 0x53
    namespace
    {
        // IMAGE-transfer qword count for w*h texels of `psm` (SDK table).
        // Returns UINT32_MAX for formats the SDK does not handle.
        uint32_t sdkImageQwc(uint32_t psm, int32_t w, int32_t h)
        {
            const int32_t texels = w * h;
            switch (psm)
            {
            case 0x00: case 0x30:
                return static_cast<uint32_t>(texels >> 2);
            case 0x01: case 0x31:
                return static_cast<uint32_t>((texels * 3) >> 4);
            case 0x02: case 0x0a: case 0x32: case 0x3a:
                return static_cast<uint32_t>(texels >> 3);
            case 0x13: case 0x1b:
                return static_cast<uint32_t>(texels >> 4);
            case 0x14: case 0x24: case 0x2c:
                return static_cast<uint32_t>(texels >> 5);
            default:
                return 0u; // SDK leaves size=0 for unknown psm
            }
        }

        inline int16_t argS16(R5900Context *ctx, int reg)
        {
            return static_cast<int16_t>(getRegU32(ctx, reg) & 0xFFFFu);
        }

        inline uint64_t guestReadU64(const uint8_t *base, uint32_t off)
        {
            uint64_t v = 0;
            std::memcpy(&v, base + off, sizeof(v));
            return v;
        }

        // SDK sceGszbufaddr(psm, w, h): Z-buffer page address right after the
        // frame buffer(s); doubled unless interlace==1 && ffmode==0 (field mode).
        int32_t sdkZbufAddr(uint32_t psm, int32_t w, int32_t h)
        {
            const int32_t wBlocks = (w + 0x3F) >> 6;
            const int32_t hBlocks = ((psm & 2u) == 0u) ? ((h + 0x1F) >> 5) : ((h + 0x3F) >> 6);
            int32_t pages = wBlocks * hBlocks;
            const bool fieldMode = (g_gparam.interlace == 1u) && (g_gparam.ffmode == 0u);
            pages = fieldMode ? (pages * 0x10000) : (pages * 0x20000);
            return pages >> 16;
        }
    }

    void sceGsExecLoadImage(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        // int sceGsExecLoadImage(sceGsLoadImage *lp, u_long128 *srcaddr)
        // Retail: DMA 6 qw from lp (normal mode), then lp->giftag1.NLOOP qw from srcaddr.
        const uint32_t imgAddr = getRegU32(ctx, 4);
        const uint32_t srcAddr = getRegU32(ctx, 5);

        const uint8_t *hdr = getConstMemPtr(rdram, imgAddr);
        if (!runtime || !runtime->syncCoreSubsystems() || !hdr)
        {
            setReturnS32(ctx, -1);
            return;
        }

        const uint32_t headerQwc = 6u;
        const uint32_t imageQwc = static_cast<uint32_t>(guestReadU64(hdr, 0x50) & 0x7FFFu);
        const uint32_t totalQwc = headerQwc + imageQwc;
        const uint32_t totalImageBytes = imageQwc * 16u;

        uint32_t pktAddr = runtime->guestMalloc(totalQwc * 16u, 16u);
        if (pktAddr == 0)
        {
            setReturnS32(ctx, -1);
            return;
        }

        uint8_t *pkt = getMemPtr(rdram, pktAddr);
        const uint8_t *src = getConstMemPtr(rdram, srcAddr);
        if (!pkt || (!src && imageQwc != 0u))
        {
            runtime->guestFree(pktAddr);
            setReturnS32(ctx, -1);
            return;
        }

        std::memcpy(pkt, hdr, headerQwc * 16u);
        if (imageQwc != 0u)
            std::memcpy(pkt + headerQwc * 16u, src, totalImageBytes);

        constexpr uint32_t GIF_CHANNEL = 0x1000A000;
        constexpr uint32_t CHCR_STR_MODE0 = 0x101u;
        auto &mem = runtime->memory();
        mem.writeIORegister(GIF_CHANNEL + 0x10u, pktAddr);
        mem.writeIORegister(GIF_CHANNEL + 0x20u, totalQwc & 0xFFFFu);
        mem.writeIORegister(GIF_CHANNEL + 0x00u, CHCR_STR_MODE0);
        mem.processPendingTransfers();
        runtime->guestFree(pktAddr);

        setReturnS32(ctx, 0);
    }

    void sceGsExecStoreImage(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        // int sceGsExecStoreImage(sceGsStoreImage *sp, u_long128 *dstaddr)
        const uint32_t imgAddr = getRegU32(ctx, 4);
        const uint32_t dstAddr = getRegU32(ctx, 5);

        const uint8_t *sp = getConstMemPtr(rdram, imgAddr);
        if (!runtime || !runtime->syncCoreSubsystems() || !sp)
        {
            setReturnS32(ctx, -1);
            return;
        }

        // SDK sceGsStoreImage layout: bitbltbuf @+0x20, trxpos @+0x30, trxreg @+0x40.
        const uint64_t bitbltbuf = guestReadU64(sp, 0x20);
        const uint64_t trxposSrc = guestReadU64(sp, 0x30);
        const uint64_t trxreg = guestReadU64(sp, 0x40);
        const uint32_t spsm = static_cast<uint32_t>((bitbltbuf >> 24) & 0x3Fu);
        const uint32_t width = static_cast<uint32_t>(trxreg & 0xFFFu);
        const uint32_t height = static_cast<uint32_t>((trxreg >> 32) & 0xFFFu);

        const uint32_t rowBytes = bytesForPixels(spsm, width);
        if (rowBytes == 0)
        {
            setReturnS32(ctx, -1);
            return;
        }
        const uint32_t totalImageBytes = rowBytes * height;

        uint8_t *dst = getMemPtr(rdram, dstAddr);
        if (!dst)
        {
            setReturnS32(ctx, -1);
            return;
        }

        // Source rect only (the SDK struct carries ssax/ssay in the low half).
        const uint64_t trxpos = trxposSrc & 0xFFFFFFFFull;

        uint32_t pktAddr = runtime->guestMalloc(80u, 16u);
        if (pktAddr == 0)
        {
            setReturnS32(ctx, -1);
            return;
        }

        uint8_t *pkt = getMemPtr(rdram, pktAddr);
        if (!pkt)
        {
            runtime->guestFree(pktAddr);
            setReturnS32(ctx, -1);
            return;
        }

        uint64_t *q = reinterpret_cast<uint64_t *>(pkt);
        q[0] = makeGiftagAplusD(4u);
        q[1] = 0xEULL;
        q[2] = bitbltbuf;
        q[3] = 0x50ULL;
        q[4] = trxpos;
        q[5] = 0x51ULL;
        q[6] = trxreg;
        q[7] = 0x52ULL;
        q[8] = 1ULL;
        q[9] = 0x53ULL;

        constexpr uint32_t GIF_CHANNEL = 0x1000A000;
        constexpr uint32_t CHCR_STR_MODE0 = 0x101u;
        auto &mem = runtime->memory();
        mem.writeIORegister(GIF_CHANNEL + 0x10u, pktAddr);
        mem.writeIORegister(GIF_CHANNEL + 0x20u, 5u);
        mem.writeIORegister(GIF_CHANNEL + 0x00u, CHCR_STR_MODE0);
        mem.processPendingTransfers();

        ps2TraceGuestRangeWrite(rdram, dstAddr, totalImageBytes, "sceGsExecStoreImage", ctx);
        runtime->gs().consumeLocalToHostBytes(dst, totalImageBytes);
        runtime->guestFree(pktAddr);

        setReturnS32(ctx, 0);
    }

    void sceGsGetGParam(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        uint32_t addr = writeGsGParamToScratch(runtime);
        setReturnU32(ctx, addr);
    }

    void sceGsPutDispEnv(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        uint32_t envAddr = getRegU32(ctx, 4);
        GsDispEnvMem env{};
        if (!readGsDispEnv(rdram, envAddr, env))
        {
            setReturnS32(ctx, -1);
            return;
        }
        applyGsDispEnv(runtime, env);
        setReturnS32(ctx, 0);
    }

    void sceGsPutDrawEnv(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        uint32_t envAddr = getRegU32(ctx, 4);
        GsRegPairMem pairs[8]{};
        if (!readGsRegPairs(rdram, envAddr, pairs, 8u))
        {
            setReturnS32(ctx, -1);
            return;
        }
        applyGsRegPairs(runtime, pairs, 8u);
        setReturnS32(ctx, 0);
    }

    void sceGsResetGraph(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        uint32_t mode = getRegU32(ctx, 4);
        uint32_t interlace = getRegU32(ctx, 5);
        uint32_t omode = getRegU32(ctx, 6);
        uint32_t ffmode = getRegU32(ctx, 7);

        if (mode == 0)
        {
            if (runtime && !runtime->syncCoreSubsystems())
            {
                setReturnS32(ctx, -1);
                return;
            }

            g_gparam.interlace = static_cast<uint8_t>(interlace & 0x1);
            g_gparam.omode = static_cast<uint8_t>(omode & 0xFF);
            g_gparam.ffmode = static_cast<uint8_t>(ffmode & 0x1);
            writeGsGParamToScratch(runtime);
            uint64_t pmode = makePmode(1, 0, 0, 0, 0, 0x80);
            uint64_t smode2 = (interlace & 0x1) | ((ffmode & 0x1) << 1);
            uint64_t dispfb = makeDispFb(0, 10, 0, 0, 0);
            uint64_t display = makeDisplay(0, 0, 0, 0, 639, 447);
            uint64_t bgcolor = 0ULL;

            if (runtime)
            {
                uint32_t pktAddr = runtime->guestMalloc(128u, 16u);
                if (pktAddr != 0u)
                {
                    uint8_t *pkt = getMemPtr(rdram, pktAddr);
                    if (pkt)
                    {
                        uint64_t *q = reinterpret_cast<uint64_t *>(pkt);
                        q[0] = makeGiftagAplusD(7u);
                        q[1] = 0xEULL;
                        q[2] = pmode;
                        q[3] = 0x41ULL;
                        q[4] = smode2;
                        q[5] = 0x42ULL;
                        q[6] = dispfb;
                        q[7] = 0x59ULL;
                        q[8] = display;
                        q[9] = 0x5aULL;
                        q[10] = dispfb;
                        q[11] = 0x5bULL;
                        q[12] = display;
                        q[13] = 0x5cULL;
                        q[14] = bgcolor;
                        q[15] = 0x5fULL;
                        constexpr uint32_t GIF_CHANNEL = 0x1000A000;
                        constexpr uint32_t CHCR_STR_MODE0 = 0x101u;
                        auto &mem = runtime->memory();
                        mem.writeIORegister(GIF_CHANNEL + 0x10u, pktAddr);
                        mem.writeIORegister(GIF_CHANNEL + 0x20u, 8u);
                        mem.writeIORegister(GIF_CHANNEL + 0x00u, CHCR_STR_MODE0);
                        mem.processPendingTransfers();
                        runtime->guestFree(pktAddr);
                    }
                    else
                    {
                        runtime->guestFree(pktAddr);
                    }
                }
            }
        }

        setReturnS32(ctx, 0);
    }

    void sceGsResetPath(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnS32(ctx, 0);
    }

    void sceGsSetDefClear(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        // int sceGsSetDefClear(sceGsClear *cp, short ztst, short x, short y, short w, short h,
        //                      u_char r, u_char g, u_char b, u_char a, u_int z)   -> 6 (qwc)
        // Byte-exact SDK port (graphdev.c @0x231348). Six A+D pairs: TEST(ZTST=ALWAYS),
        // PRIM(sprite), RGBAQ, XYZ2, XYZ2, TEST(restore). The game DMAs this every
        // frame via its ieGsDBuffDc; the old no-op stub left it zeroed, so the frame
        // and Z buffer were never cleared (stale Z hid far geometry, e.g. the
        // drive-in movie screen on the main menu).
        (void)runtime;
        const uint32_t cpAddr = getRegU32(ctx, 4);
        const int32_t ztst = argS16(ctx, 5);
        const int32_t x = argS16(ctx, 6);
        const int32_t y = argS16(ctx, 7);
        const int32_t w = argS16(ctx, 8);
        const int32_t h = argS16(ctx, 9);
        const uint64_t r = getRegU32(ctx, 10) & 0xFFu;
        const uint64_t g = getRegU32(ctx, 11) & 0xFFu;
        const uint64_t b = readStackU32(rdram, ctx, 0) & 0xFFu;
        const uint64_t a = readStackU32(rdram, ctx, 8) & 0xFFu;
        const uint64_t z = readStackU32(rdram, ctx, 16);

        uint8_t *ptr = getMemPtr(rdram, cpAddr);
        if (!ptr)
        {
            setReturnS32(ctx, 0);
            return;
        }

        uint64_t q[12];
        q[0] = 0x30000ull; // TEST: ZTE=1, ZTST=ALWAYS
        q[1] = 0x47ull;
        q[2] = 6ull;       // PRIM: sprite
        q[3] = 0ull;
        q[4] = r | (g << 8) | (b << 16) | (a << 24) | 0x3f80000000000000ull; // RGBAQ, Q=1.0
        q[5] = 1ull;
        q[6] = static_cast<uint64_t>(static_cast<uint32_t>(x << 4)) |
               (static_cast<uint64_t>(static_cast<uint32_t>(y << 4)) << 16) | (z << 32);
        q[7] = 5ull;       // XYZ2
        q[8] = static_cast<uint64_t>(static_cast<uint32_t>((x + w) << 4)) |
               (static_cast<uint64_t>(static_cast<uint32_t>((y + h) << 4)) << 16) | (z << 32);
        q[9] = 5ull;
        q[10] = (ztst == 0) ? 0x30000ull : ((static_cast<uint64_t>(ztst & 3) << 17) | 0x10000ull);
        q[11] = 0x47ull;
        std::memcpy(ptr, q, sizeof(q));
        setReturnS32(ctx, 6);
    }

    void sceGsSetDefDBuffDc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t envAddr = getRegU32(ctx, 4);
        uint32_t psm = getRegU32(ctx, 5);
        uint32_t w = getRegU32(ctx, 6);
        uint32_t h = getRegU32(ctx, 7);
        const GsTrailingArgs3 trailing = decodeGsTrailingArgs3(rdram, ctx);
        const uint32_t ztest = trailing.arg0;
        const uint32_t zpsm = trailing.arg1;
        const uint32_t clear = trailing.arg2;

        if (w == 0u)
        {
            w = 640u;
        }
        if (h == 0u)
        {
            h = 448u;
        }

        const uint32_t fbw = std::max<uint32_t>(1u, (w + 63u) / 64u);
        const uint64_t pmode = makePmode(1u, 1u, 0u, 0u, 0u, 0x80u);
        const uint64_t smode2 =
            (static_cast<uint64_t>(g_gparam.interlace & 0x1u) << 0) |
            (static_cast<uint64_t>(g_gparam.ffmode & 0x1u) << 1);
        const uint64_t display = makeDisplay(636u, 32u, 0u, 0u, w - 1u, h - 1u);

        const int32_t drawWidth = static_cast<int32_t>(w);
        const int32_t drawHeight = static_cast<int32_t>(h);

        uint32_t zbufAddr = 0u;
        {
            R5900Context temp = *ctx;
            sceGszbufaddr(rdram, &temp, runtime);
            zbufAddr = getRegU32(&temp, 2);
        }

        const uint32_t fbp1 = zbufAddr;
        const uint64_t dispfb0 = makeDispFb(fbp1, fbw, psm, 0u, 0u);
        const uint64_t dispfb1 = makeDispFb(0u, fbw, psm, 0u, 0u);

        GsDBuffDcMem db{};
        db.disp[0].pmode = pmode;
        db.disp[0].smode2 = smode2;
        db.disp[0].dispfb = dispfb0;
        db.disp[0].display = display;
        db.disp[0].bgcolor = 0u;
        db.disp[1] = db.disp[0];
        db.disp[1].dispfb = dispfb1;

        const bool seedClear = clear != 0u;
        db.giftag0 = {makeGiftagAplusD(seedClear ? 22u : 16u), 0xEULL};
        seedGsDrawEnv1(db.draw01, drawWidth, drawHeight, 0u, fbw, psm, zbufAddr, zpsm, ztest, false);
        seedGsDrawEnv2(db.draw02, drawWidth, drawHeight, 0u, fbw, psm, zbufAddr, zpsm, ztest, false);
        db.giftag1 = db.giftag0;
        seedGsDrawEnv1(db.draw11, drawWidth, drawHeight, fbp1, fbw, psm, zbufAddr, zpsm, ztest, false);
        seedGsDrawEnv2(db.draw12, drawWidth, drawHeight, fbp1, fbw, psm, zbufAddr, zpsm, ztest, false);
        if (seedClear)
        {
            seedGsClearPacket(db.clear0, drawWidth, drawHeight, 0u, ztest, false);
            seedGsClearPacket(db.clear1, drawWidth, drawHeight, 0u, ztest, true);
        }

        if (!writeGsDBuffDc(rdram, envAddr, db))
        {
            setReturnS32(ctx, -1);
            return;
        }
        setReturnS32(ctx, 0);
    }

    void sceGsSetDefDBuff(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t envAddr = getRegU32(ctx, 4);
        uint32_t psm = getRegU32(ctx, 5);
        uint32_t w = getRegU32(ctx, 6);
        uint32_t h = getRegU32(ctx, 7);
        const uint32_t ztest = readStackU32(rdram, ctx, 16);
        const uint32_t zpsm = readStackU32(rdram, ctx, 20);
        const uint32_t clear = readStackU32(rdram, ctx, 24);
        (void)clear;

        if (w == 0u)
        {
            w = 640u;
        }
        if (h == 0u)
        {
            h = 448u;
        }

        const uint32_t fbw = std::max<uint32_t>(1u, (w + 63u) / 64u);
        const uint64_t pmode = makePmode(1u, 1u, 0u, 0u, 0u, 0x80u);
        const uint64_t smode2 =
            (static_cast<uint64_t>(g_gparam.interlace & 0x1u) << 0) |
            (static_cast<uint64_t>(g_gparam.ffmode & 0x1u) << 1);
        const uint64_t dispfb = makeDispFb(0u, fbw, psm, 0u, 0u);
        const uint64_t display = makeDisplay(636u, 32u, 0u, 0u, w - 1u, h - 1u);

        const int32_t drawWidth = static_cast<int32_t>(w);
        const int32_t drawHeight = static_cast<int32_t>(h);

        uint32_t zbufAddr = 0u;
        {
            R5900Context temp = *ctx;
            sceGszbufaddr(rdram, &temp, runtime);
            zbufAddr = getRegU32(&temp, 2);
        }

        GsDBuffMem db{};
        db.disp[0].pmode = pmode;
        db.disp[0].smode2 = smode2;
        db.disp[0].dispfb = dispfb;
        db.disp[0].display = display;
        db.disp[0].bgcolor = 0u;
        db.disp[1] = db.disp[0];

        db.giftag0 = {makeGiftagAplusD(14u), 0x0E0E0E0E0E0E0E0EULL};
        seedGsDrawEnv1(db.draw0, drawWidth, drawHeight, 0u, fbw, psm, zbufAddr, zpsm, ztest, false);
        db.giftag1 = db.giftag0;
        seedGsDrawEnv1(db.draw1, drawWidth, drawHeight, 0u, fbw, psm, zbufAddr, zpsm, ztest, false);

        if (!writeGsDBuff(rdram, envAddr, db))
        {
            setReturnS32(ctx, -1);
            return;
        }
        setReturnS32(ctx, 0);
    }

    void sceGsSetDefDispEnv(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        // u_long sceGsSetDefDispEnv(sceGsDispEnv *disp, short psm, short w, short h, short dx, short dy)
        // Byte-exact port of the SDK (graphdev.c, retail @0x230ef0). The game
        // (ieGsSetDefDBuffDc) hands this its 224-line frame buffers; with the
        // interlace=1 / frame-mode setup from sceGsResetGraph the SDK answers
        // SMODE2=3 and DH=2h-1 (each buffer line-doubled across both fields).
        // The old stub wrote SMODE2=0 / DH=h-1 = a 224-line progressive display.
        const uint32_t envAddr = getRegU32(ctx, 4);
        const int32_t psm = argS16(ctx, 5);
        const int32_t w = argS16(ctx, 6);
        const int32_t h = argS16(ctx, 7);
        const int32_t dx = argS16(ctx, 8);
        const int32_t dy = argS16(ctx, 9);

        uint8_t *ptr = getMemPtr(rdram, envAddr);
        if (!ptr || w <= 0)
        {
            setReturnS32(ctx, 0);
            return;
        }

        const bool interlaced = g_gparam.interlace == 1u;
        const bool ffmode = g_gparam.ffmode != 0u;
        const uint32_t omode = g_gparam.omode;

        uint64_t q[5] = {0x66ull, 0ull, 0ull, 0ull, 0ull}; // pmode, smode2, dispfb, display, bgcolor
        q[1] = (g_gparam.interlace == 0u) ? 2ull : (ffmode ? 3ull : 1ull);
        q[2] = (static_cast<uint64_t>(psm & 0xF) << 15) | (static_cast<uint64_t>(((w + 0x3F) >> 6) & 0x3F) << 9);

        if (omode == 2u || omode == 3u)
        {
            const int32_t magh = (w + 0x9FF) / w;
            const int32_t dxBase = (omode == 2u) ? 0x27C : 0x290;
            int32_t dyOff, dh;
            if (!interlaced)
            {
                dyOff = (omode == 2u) ? 0x19 : 0x24;
                dh = h - 1;
            }
            else
            {
                dyOff = (omode == 2u) ? 0x32 : 0x48;
                dh = ffmode ? (h * 2 - 1) : (h - 1);
            }
            q[3] = (static_cast<uint64_t>(magh - 1) << 23) |
                   (static_cast<uint64_t>(magh * w - 1) << 32) |
                   (static_cast<uint64_t>((dx * magh + dxBase) & 0xFFF)) |
                   (static_cast<uint64_t>(dh & 0x7FF) << 44) |
                   (static_cast<uint64_t>((dy + dyOff) & 0xFFF) << 12);
        }
        else
        {
            RUNTIME_LOG("sceGsDefDispEnv:Not support displaymode for " << omode << "!!" << std::endl);
        }

        std::memcpy(ptr, q, sizeof(q));
        setReturnU32(ctx, static_cast<uint32_t>(q[3]));
    }

    void sceGsSetDefDrawEnv(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        uint32_t envAddr = getRegU32(ctx, 4);
        uint32_t param_2 = getRegU32(ctx, 5);
        int32_t w = static_cast<int32_t>(static_cast<int16_t>(getRegU32(ctx, 6) & 0xFFFF));
        int32_t h = static_cast<int32_t>(static_cast<int16_t>(getRegU32(ctx, 7) & 0xFFFF));
        // SDK signature: (env, short psm, short w, short h, short ztest, short zpsm) —
        // six args, all in registers (a0-a3, t0, t1). Read t0/t1 directly: the old
        // heuristic decoder fell back to stack words whenever ztest==zpsm==0.
        uint32_t param_5 = getRegU32(ctx, 8) & 0xFFFFu;
        uint32_t param_6 = getRegU32(ctx, 9) & 0xFFFFu;

        if (w <= 0)
            w = 640;
        if (h <= 0)
            h = 448;

        uint32_t psm = param_2 & 0xFU;
        uint32_t fbw = ((static_cast<uint32_t>(w) + 63u) >> 6) & 0x3FU;
        int32_t zbuf = sdkZbufAddr(param_2, w, h);

        GsDrawEnv1Mem env{};
        seedGsDrawEnv1(env,
                       w,
                       h,
                       0u,
                       fbw,
                       psm,
                       static_cast<uint32_t>(zbuf),
                       param_6 & 0xFu,
                       param_5 & 0x3u,
                       (param_2 & 2u) != 0u);

        uint8_t *const ptr = getMemPtr(rdram, envAddr);
        if (!ptr)
        {
            setReturnS32(ctx, 8);
            return;
        }
        std::memcpy(ptr, &env, sizeof(env));

        setReturnS32(ctx, 8);
    }

    void sceGsSetDefDrawEnv2(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        uint32_t envAddr = getRegU32(ctx, 4);
        uint32_t param_2 = getRegU32(ctx, 5);
        int32_t w = static_cast<int32_t>(static_cast<int16_t>(getRegU32(ctx, 6) & 0xFFFF));
        int32_t h = static_cast<int32_t>(static_cast<int16_t>(getRegU32(ctx, 7) & 0xFFFF));
        // SDK signature: (env, short psm, short w, short h, short ztest, short zpsm) —
        // six args, all in registers (a0-a3, t0, t1). Read t0/t1 directly: the old
        // heuristic decoder fell back to stack words whenever ztest==zpsm==0.
        uint32_t param_5 = getRegU32(ctx, 8) & 0xFFFFu;
        uint32_t param_6 = getRegU32(ctx, 9) & 0xFFFFu;

        if (w <= 0)
            w = 640;
        if (h <= 0)
            h = 448;

        uint32_t psm = param_2 & 0xFU;
        uint32_t fbw = ((static_cast<uint32_t>(w) + 63u) >> 6) & 0x3FU;
        int32_t zbuf = sdkZbufAddr(param_2, w, h);

        GsDrawEnv2Mem env{};
        seedGsDrawEnv2(env,
                       w,
                       h,
                       0u,
                       fbw,
                       psm,
                       static_cast<uint32_t>(zbuf),
                       param_6 & 0xFu,
                       param_5 & 0x3u,
                       (param_2 & 2u) != 0u);

        uint8_t *const ptr = getMemPtr(rdram, envAddr);
        if (!ptr)
        {
            setReturnS32(ctx, 8);
            return;
        }

        std::memcpy(ptr, &env, sizeof(env));
        setReturnS32(ctx, 8);
    }

    void sceGsSetDefLoadImage(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        // int sceGsSetDefLoadImage(sceGsLoadImage *lp, short dbp, short dbw, short dpsm,
        //                          short x, short y, short w, short h)   -> 6 (qwc) or 0
        const uint32_t lpAddr = getRegU32(ctx, 4);
        const int16_t dbp = argS16(ctx, 5);
        const int16_t dbw = argS16(ctx, 6);
        const int16_t dpsm = argS16(ctx, 7);
        const int16_t x = argS16(ctx, 8);
        const int16_t y = argS16(ctx, 9);
        const int16_t w = argS16(ctx, 10);
        const int16_t h = argS16(ctx, 11);

        const uint32_t nloop = sdkImageQwc(static_cast<uint16_t>(dpsm), w, h);
        if (nloop >= 0x8000u)
        {
            RUNTIME_LOG("sceGsSetDefLoadImage: too big size" << std::endl);
            setReturnS32(ctx, 0);
            return;
        }

        uint8_t *ptr = getMemPtr(rdram, lpAddr);
        if (!ptr)
        {
            setReturnS32(ctx, 0);
            return;
        }

        uint64_t q[12];
        q[0] = 0x1000000000000004ull; // giftag: nloop=4 eop=0 nreg=1
        q[1] = 0xEull;                // A+D
        q[2] = (static_cast<uint64_t>(static_cast<int64_t>(dbp)) << 32) |
               (static_cast<uint64_t>(static_cast<int64_t>(dbw)) << 48) |
               (static_cast<uint64_t>(static_cast<int64_t>(dpsm)) << 56);
        q[3] = 0x50ull; // BITBLTBUF
        q[4] = (static_cast<uint64_t>(static_cast<int64_t>(x)) << 32) |
               (static_cast<uint64_t>(static_cast<int64_t>(y)) << 48);
        q[5] = 0x51ull; // TRXPOS
        q[6] = static_cast<uint64_t>(static_cast<uint16_t>(w)) |
               (static_cast<uint64_t>(static_cast<int64_t>(h)) << 32);
        q[7] = 0x52ull; // TRXREG
        q[8] = 0ull;    // TRXDIR = host -> local
        q[9] = 0x53ull;
        q[10] = 0x0800000000008000ull | (nloop & 0x7FFFu); // giftag: IMAGE, eop=1
        q[11] = 0ull;
        std::memcpy(ptr, q, sizeof(q));

        setReturnS32(ctx, 6);
    }

    void sceGsSetDefStoreImage(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        // int sceGsSetDefStoreImage(sceGsStoreImage *sp, short sbp, short sbw, short spsm,
        //                           short x, short y, short w, short h)  -> 7 (qwc)
        const uint32_t spAddr = getRegU32(ctx, 4);
        const int16_t sbp = argS16(ctx, 5);
        const int32_t sbw = static_cast<int32_t>(getRegU32(ctx, 6));
        const int16_t spsm = argS16(ctx, 7);
        const int16_t x = argS16(ctx, 8);
        const int32_t y = static_cast<int32_t>(getRegU32(ctx, 9));
        const int16_t w = argS16(ctx, 10);
        const int16_t h = argS16(ctx, 11);

        uint8_t *ptr = getMemPtr(rdram, spAddr);
        if (!ptr)
        {
            setReturnS32(ctx, 0);
            return;
        }

        uint32_t vif[4] = {0u, 0x06008000u, 0x13000000u, 0x50000006u}; // NOP, MSKPATH3, FLUSHA, DIRECT 6
        uint64_t q[12];
        q[0] = 0x1000000000008005ull; // giftag: nloop=5 eop=1 nreg=1
        q[1] = 0xEull;
        q[2] = static_cast<uint64_t>(static_cast<uint32_t>((sbw << 16) | static_cast<uint16_t>(sbp))) |
               (static_cast<uint64_t>(static_cast<int64_t>(spsm)) << 24);
        q[3] = 0x50ull;
        q[4] = static_cast<uint64_t>(static_cast<uint32_t>((y << 16) | static_cast<uint16_t>(x)));
        q[5] = 0x51ull;
        q[6] = static_cast<uint64_t>(static_cast<uint16_t>(w)) |
               (static_cast<uint64_t>(static_cast<int64_t>(h)) << 32);
        q[7] = 0x52ull;
        q[8] = 0ull;
        q[9] = 0x61ull; // FINISH
        q[10] = 1ull;   // TRXDIR = local -> host
        q[11] = 0x53ull;
        std::memcpy(ptr, vif, sizeof(vif));
        std::memcpy(ptr + 16u, q, sizeof(q));

        setReturnS32(ctx, 7);
    }

    void sceGsSwapDBuffDc(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t envAddr = getRegU32(ctx, 4);
        const uint32_t which = getRegU32(ctx, 5) & 1u;

        GsDBuffDcMem db{};
        if (!runtime || !readGsDBuffDc(rdram, envAddr, db))
        {
            setReturnS32(ctx, -1);
            return;
        }

        applyGsDispEnv(runtime, db.disp[which]);
        static uint32_t s_swapDbuffLogCount = 0u;
        if (s_swapDbuffLogCount < 32u)
        {
            const uint32_t dispFbp = static_cast<uint32_t>(db.disp[which].dispfb & 0x1FFu);
            const uint32_t clearContext = (which == 0u)
                                              ? static_cast<uint32_t>((db.clear0.prim.value >> 9) & 0x1u)
                                              : static_cast<uint32_t>((db.clear1.prim.value >> 9) & 0x1u);
            PS2_IF_AGRESSIVE_LOGS({
                RUNTIME_LOG("[gs:swapdbuff] which=" << which
                                                    << " env=0x" << std::hex << envAddr
                                                    << " dispfb=0x" << db.disp[which].dispfb
                                                    << " display=0x" << db.disp[which].display
                                                    << " pmode=0x" << db.disp[which].pmode
                                                    << " dispFbp=" << dispFbp
                                                    << " clearCtxt=" << clearContext
                                                    << std::dec << std::endl);
            });
            ++s_swapDbuffLogCount;
        }
        if (which == 0u)
        {
            applyGsRegPairs(runtime, reinterpret_cast<const GsRegPairMem *>(&db.draw01), 8u);
            applyGsRegPairs(runtime, reinterpret_cast<const GsRegPairMem *>(&db.draw02), 8u);
            if (hasSeededGsClearPacket(db.clear0))
            {
                const uint32_t clearContext = static_cast<uint32_t>((db.clear0.prim.value >> 9) & 0x1u);
                runtime->gs().clearFramebufferContext(clearContext, static_cast<uint32_t>(db.clear0.rgbaq.value));
            }
            applyGsClearPacket(runtime, db.clear0);
        }
        else
        {
            applyGsRegPairs(runtime, reinterpret_cast<const GsRegPairMem *>(&db.draw11), 8u);
            applyGsRegPairs(runtime, reinterpret_cast<const GsRegPairMem *>(&db.draw12), 8u);
            if (hasSeededGsClearPacket(db.clear1))
            {
                const uint32_t clearContext = static_cast<uint32_t>((db.clear1.prim.value >> 9) & 0x1u);
                runtime->gs().clearFramebufferContext(clearContext, static_cast<uint32_t>(db.clear1.rgbaq.value));
            }
            applyGsClearPacket(runtime, db.clear1);
        }

        setReturnS32(ctx, static_cast<int32_t>(which ^ 1u));
    }

    void sceGsSwapDBuff(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t envAddr = getRegU32(ctx, 4);
        const uint32_t which = getRegU32(ctx, 5) & 1u;

        GsDBuffMem db{};
        if (!runtime || !readGsDBuff(rdram, envAddr, db))
        {
            setReturnS32(ctx, -1);
            return;
        }

        applyGsDispEnv(runtime, db.disp[which]);
        if (which == 0u)
        {
            applyGsRegPairs(runtime, reinterpret_cast<const GsRegPairMem *>(&db.draw0), 8u);
        }
        else
        {
            applyGsRegPairs(runtime, reinterpret_cast<const GsRegPairMem *>(&db.draw1), 8u);
        }

        setReturnS32(ctx, static_cast<int32_t>(which ^ 1u));
    }

    void sceGsSyncPath(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        ::g_ps2xSyncPathCalls.fetch_add(1u, std::memory_order_relaxed);
        int32_t mode = static_cast<int32_t>(getRegU32(ctx, 4));
        auto &mem = runtime->memory();

        if (mode == 0)
        {
            mem.processPendingTransfers();
            mem.vu1SyncPathEpoch(); // threaded VU1: bound the worker's lag to one sync

            uint32_t count = 0;
            constexpr uint32_t kTimeout = 0x1000000;

            while ((mem.readIORegister(0x10009000) & 0x100) != 0)
            {
                if (++count > kTimeout)
                {
                    setReturnS32(ctx, -1);
                    return;
                }
            }

            while ((mem.readIORegister(0x1000A000) & 0x100) != 0)
            {
                if (++count > kTimeout)
                {
                    setReturnS32(ctx, -1);
                    return;
                }
            }

            while ((mem.readIORegister(0x10003C00) & 0x1F000003) != 0)
            {
                if (++count > kTimeout)
                {
                    setReturnS32(ctx, -1);
                    return;
                }
            }

            while ((mem.readIORegister(0x10003020) & 0xC00) != 0)
            {
                if (++count > kTimeout)
                {
                    setReturnS32(ctx, -1);
                    return;
                }
            }

            setReturnS32(ctx, 0);
        }
        else
        {
            uint32_t result = 0;

            if ((mem.readIORegister(0x10009000) & 0x100) != 0)
                result |= 1;
            if ((mem.readIORegister(0x1000A000) & 0x100) != 0)
                result |= 2;
            if ((mem.readIORegister(0x10003C00) & 0x1F000003) != 0)
                result |= 4;
            if ((mem.readIORegister(0x10003020) & 0xC00) != 0)
                result |= 0x10;

            setReturnS32(ctx, result);
        }
    }

    void sceGsSyncV(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        ps2_syscalls::WaitVSyncTick(rdram,
                                    ctx,
                                    runtime,
                                    g_gparam.interlace != 0u ? -1 : 1);
    }

    void sceGsSyncVCallback(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t newCallback = getRegU32(ctx, 4);
        const uint32_t callerPc = ctx ? ctx->pc : 0u;
        const uint32_t callerRa = ctx ? getRegU32(ctx, 31) : 0u;
        const uint32_t gp = getRegU32(ctx, 28);
        const uint32_t sp = getRegU32(ctx, 29);

        EeScheduler &ee = runtime->eeScheduler();
        ee.bindMainContextForSyscall(*ctx, rdram);
        const uint32_t oldCallback = ee.setGsVSyncCallback(newCallback, gp, sp);

        static uint32_t s_syncVCallbackLogCount = 0u;
        if (s_syncVCallbackLogCount < 128u)
        {
            PS2_IF_AGRESSIVE_LOGS({
                RUNTIME_LOG("[sceGsSyncVCallback:set] new=0x" << std::hex << newCallback
                                                              << " old=0x" << oldCallback
                                                              << " callerPc=0x" << callerPc
                                                              << " callerRa=0x" << callerRa
                                                              << " gp=0x" << gp
                                                              << " sp=0x" << sp
                                                              << std::dec << std::endl);
            });
            ++s_syncVCallbackLogCount;
        }

        setReturnU32(ctx, oldCallback);
    }

    void sceGszbufaddr(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)rdram;
        uint32_t param_1 = getRegU32(ctx, 4);
        int32_t w = static_cast<int32_t>(static_cast<int16_t>(getRegU32(ctx, 6) & 0xFFFF));
        int32_t h = static_cast<int32_t>(static_cast<int16_t>(getRegU32(ctx, 7) & 0xFFFF));

        int32_t width_blocks = (w + 63) >> 6;
        if (w + 63 < 0)
            width_blocks = (w + 126) >> 6;

        int32_t height_blocks;
        if ((param_1 & 2) != 0)
        {
            int32_t v = (h + 63) >> 6;
            if (h + 63 < 0)
                v = (h + 126) >> 6;
            height_blocks = v;
        }
        else
        {
            int32_t v = (h + 31) >> 5;
            if (h + 31 < 0)
                v = (h + 62) >> 5;
            height_blocks = v;
        }

        int32_t product = width_blocks * height_blocks;

        uint64_t gparam_val = 0;
        if (runtime)
        {
            uint8_t *scratch = runtime->memory().getScratchpad();
            if (scratch)
            {
                std::memcpy(&gparam_val, scratch + 0x100, sizeof(gparam_val));
            }
        }
        if ((gparam_val & 0xFFFF0000FFFFULL) == 1ULL)
            product = (product * 0x10000) >> 16;
        else
            product = (product * 0x20000) >> 16;

        setReturnS32(ctx, product);
    }

    void Ps2SwapDBuff(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        static int logCount = 0;
        if (logCount < 8)
        {
            RUNTIME_LOG("ps2_stub Ps2SwapDBuff");
            ++logCount;
        }
        setReturnS32(ctx, 0);
    }

    void sceVif1PkAddGsAD(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t stateAddr = getRegU32(ctx, 4);
        uint32_t currentAddr = 0u;
        if (!tryReadWordFromGuest(rdram, runtime, stateAddr, currentAddr))
        {
            return;
        }

        const uint64_t dataValue = GPR_U64(ctx, 6);
        const uint32_t words[4] = {
            static_cast<uint32_t>(dataValue & 0xFFFFFFFFu),
            static_cast<uint32_t>(dataValue >> 32u),
            getRegU32(ctx, 5),
            0u,
        };
        writeGuestBytes(rdram,
                        runtime,
                        currentAddr,
                        reinterpret_cast<const uint8_t *>(words),
                        sizeof(words));
        writePacketBuilderCurrent(rdram, runtime, stateAddr, currentAddr + 16u);
    }

    void sceVif1PkAlign(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        alignPacketBuilderState(rdram,
                                runtime,
                                getRegU32(ctx, 4),
                                getRegU32(ctx, 5),
                                getRegU32(ctx, 6));
    }

    void sceVif1PkCall(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t stateAddr = getRegU32(ctx, 4);
        const uint32_t refAddr = getRegU32(ctx, 5) & 0x9FFFFFFFu;
        const uint32_t tagWord = getRegU32(ctx, 6) | 0x50000000u;
        const uint32_t packetAddr = terminatePacketBuilderState(rdram, ctx, runtime);
        const uint32_t words[2] = {tagWord, refAddr};

        writeGuestU32(rdram, runtime, stateAddr + 8u, packetAddr);
        writeGuestBytes(rdram,
                        runtime,
                        packetAddr,
                        reinterpret_cast<const uint8_t *>(words),
                        sizeof(words));
        writePacketBuilderCurrent(rdram, runtime, stateAddr, packetAddr + 8u);
        writeGuestU32(rdram, runtime, stateAddr + 12u, 0u);
    }

    void sceVif1PkCloseDirectCode(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t stateAddr = getRegU32(ctx, 4);
        uint32_t currentAddr = 0u;
        uint32_t openAddr = 0u;
        if (!tryReadWordFromGuest(rdram, runtime, stateAddr, currentAddr) ||
            !tryReadWordFromGuest(rdram, runtime, stateAddr + 12u, openAddr) ||
            openAddr == 0u)
        {
            return;
        }

        const uint32_t currentMinusTag = currentAddr - 4u;
        const uint32_t wordCount = (currentMinusTag - openAddr) >> 2u;
        const uint32_t qwordCount = wordCount >> 2u;
        uint32_t tagWord = 0u;
        if (!tryReadWordFromGuest(rdram, runtime, openAddr, tagWord))
        {
            return;
        }

        tagWord += qwordCount;
        writeGuestU32(rdram, runtime, stateAddr + 12u, 0u);
        writeGuestU32(rdram, runtime, openAddr, tagWord);
    }

    void sceVif1PkCloseGifTag(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)ctx;
        closePacketGifTag(rdram, runtime, getRegU32(ctx, 4), 20u);
    }

    void sceVif1PkCnt(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t stateAddr = getRegU32(ctx, 4);
        const uint32_t tagWord = getRegU32(ctx, 5) | 0x10000000u;
        const uint32_t packetAddr = terminatePacketBuilderState(rdram, ctx, runtime);
        const uint32_t words[2] = {tagWord, 0u};

        writeGuestU32(rdram, runtime, stateAddr + 8u, packetAddr);
        writeGuestBytes(rdram,
                        runtime,
                        packetAddr,
                        reinterpret_cast<const uint8_t *>(words),
                        sizeof(words));
        writeGuestU32(rdram, runtime, stateAddr + 12u, 0u);
        writePacketBuilderCurrent(rdram, runtime, stateAddr, packetAddr + 8u);
    }

    void sceVif1PkEnd(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t stateAddr = getRegU32(ctx, 4);
        const uint32_t tagWord = getRegU32(ctx, 5) | 0x70000000u;
        const uint32_t packetAddr = terminatePacketBuilderState(rdram, ctx, runtime);
        const uint32_t words[2] = {tagWord, 0u};

        writeGuestU32(rdram, runtime, stateAddr + 8u, packetAddr);
        writeGuestBytes(rdram,
                        runtime,
                        packetAddr,
                        reinterpret_cast<const uint8_t *>(words),
                        sizeof(words));
        writeGuestU32(rdram, runtime, stateAddr + 12u, 0u);
        writePacketBuilderCurrent(rdram, runtime, stateAddr, packetAddr + 8u);
    }

    void sceVif1PkInit(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        initPacketBuilderState(rdram, ctx, runtime);
        writeGuestU32(rdram, runtime, getRegU32(ctx, 4) + 20u, 0u);
    }

    void sceVif1PkOpenDirectCode(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t stateAddr = getRegU32(ctx, 4);
        alignPacketBuilderState(rdram, runtime, stateAddr, 2u, 3u);

        uint32_t currentAddr = 0u;
        if (!tryReadWordFromGuest(rdram, runtime, stateAddr, currentAddr))
        {
            return;
        }

        const uint32_t tagWord = (getRegU32(ctx, 5) != 0u) ? 0xD0000000u : 0x50000000u;
        writeGuestU32(rdram, runtime, currentAddr, tagWord);
        writePacketBuilderCurrent(rdram, runtime, stateAddr, currentAddr + 4u);
        writeGuestU32(rdram, runtime, stateAddr + 12u, currentAddr);
    }

    void sceVif1PkOpenGifTag(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        openPacketGifTag(rdram, ctx, runtime, getRegU32(ctx, 4), 20u);
    }

    void sceVif1PkReset(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        resetPacketBuilderState(rdram, ctx, runtime);
        writeGuestU32(rdram, runtime, getRegU32(ctx, 4) + 20u, 0u);
    }

    void sceVif1PkReserve(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t stateAddr = getRegU32(ctx, 4);
        const uint32_t wordCount = getRegU32(ctx, 5);
        uint32_t currentAddr = 0u;
        tryReadWordFromGuest(rdram, runtime, stateAddr, currentAddr);
        const uint32_t reservedAddr = reservePacketBuilderWords(rdram, runtime, stateAddr, wordCount);
        setReturnU32(ctx, reservedAddr);
    }

    void sceVif1PkTerminate(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        setReturnU32(ctx, terminatePacketBuilderState(rdram, ctx, runtime));
    }
}
