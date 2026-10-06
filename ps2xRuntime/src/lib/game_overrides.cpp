#include "game_overrides.h"
#include "ps2_runtime.h"
#include "ps2_runtime_calls.h"
#include "ps2_stubs.h"
#include "ps2_syscalls.h"
#include "ps2_log.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <vector>

#include "wotm_native_traversal.inc"
#include "wotm_native_packets.inc"

// Published by the game thread; read by presentation without racing guest RAM.
std::atomic<uint64_t> g_ps2xWotmCompletedFrames{0u};
std::atomic<uint64_t> g_ps2xWotmMenuFrames{0u};
std::atomic<bool> g_ps2xModSelectSeen{false};
std::atomic<bool> g_ps2xRosterEnabled{false}, g_ps2xRosterPage{false};
std::atomic<int> g_ps2xRosterSlot{0}, g_ps2xRosterChosenType{0};
std::atomic<uint64_t> g_ps2xRosterHeartbeatNs{0};
uint32_t g_ps2xRosterIds[10]{};
uint32_t g_ps2xRosterCostumes[10]{};
char g_ps2xRosterNames[10][33]{};
std::atomic<uint32_t> g_ps2xWotmTargetHz{0u}, g_ps2xWotmPhase{0u};
std::atomic<bool> g_ps2xWotmFrameMetricsEnabled{false};
std::atomic<bool> g_ps2xWotmWideActive{false};
std::atomic<bool> g_ps2xWotmPaused{false};
#ifdef _WIN32
extern void ps2xGsSealCompletedFrame(uint64_t,const GSPresentationRequest&);
#endif
extern std::atomic<uint64_t> g_ps2xVuInstr;
extern std::atomic<uint64_t> g_ps2xVuRuns;
extern uint64_t ps2xNativeButtonTex0(int port);
extern void ps2xPacingFrameComplete(PS2Runtime *);
extern void ps2xTracePacing(const char *, uint8_t *, PS2Runtime *, uint32_t);
extern thread_local bool t_ps2xEeUnwinding;
extern thread_local uint32_t t_ps2xEeCheckpointDeferralDepth;

#include "wotm_destruction_profile.inc"
#include "wotm_motion_history.inc"

namespace
{
    std::mutex &registryMutex()
    {
        static std::mutex mutex;
        return mutex;
    }

    std::vector<ps2_game_overrides::Descriptor> &descriptorRegistry()
    {
        static std::vector<ps2_game_overrides::Descriptor> registry;
        return registry;
    }

    bool equalsIgnoreCaseAscii(std::string_view lhs, std::string_view rhs)
    {
        if (lhs.size() != rhs.size())
        {
            return false;
        }

        for (size_t i = 0; i < lhs.size(); ++i)
        {
            const auto l = static_cast<unsigned char>(lhs[i]);
            const auto r = static_cast<unsigned char>(rhs[i]);
            if (std::tolower(l) != std::tolower(r))
            {
                return false;
            }
        }

        return true;
    }

    std::string basenameFromPath(const std::string &path)
    {
        std::error_code ec;
        const std::filesystem::path fsPath(path);
        const std::filesystem::path leaf = fsPath.filename();
        if (leaf.empty())
        {
            return path;
        }
        return leaf.string();
    }

    std::optional<PS2Runtime::RecompiledFunction> resolveHandlerByName(std::string_view handlerName)
    {
        const std::string_view resolvedSyscall = ps2_runtime_calls::resolveSyscallName(handlerName);
        if (!resolvedSyscall.empty())
        {
#define PS2_RESOLVE_SYSCALL(name)                   \
    if (resolvedSyscall == std::string_view{#name}) \
    {                                               \
        return &ps2_syscalls::name;                 \
    }
            PS2_SYSCALL_LIST(PS2_RESOLVE_SYSCALL)
#undef PS2_RESOLVE_SYSCALL
        }

        const std::string_view resolvedStub = ps2_runtime_calls::resolveStubName(handlerName);
        if (!resolvedStub.empty())
        {
#define PS2_RESOLVE_STUB(name)                   \
    if (resolvedStub == std::string_view{#name}) \
    {                                            \
        return &ps2_stubs::name;                 \
    }
            PS2_STUB_LIST(PS2_RESOLVE_STUB)
#undef PS2_RESOLVE_STUB
        }

        return std::nullopt;
    }
}
namespace ps2_game_overrides
{
    AutoRegister::AutoRegister(const Descriptor &descriptor)
    {
        registerDescriptor(descriptor);
    }

    void registerDescriptor(const Descriptor &descriptor)
    {
        if (!descriptor.apply)
        {
            std::cerr << "[game_overrides] ignoring descriptor with null apply callback." << std::endl;
            return;
        }

        std::lock_guard<std::mutex> lock(registryMutex());
        descriptorRegistry().push_back(descriptor);
    }

    bool bindAddressHandler(PS2Runtime &runtime, uint32_t address, std::string_view handlerName)
    {
        const auto resolved = resolveHandlerByName(handlerName);
        if (!resolved.has_value())
        {
            std::cerr << "[game_overrides] unresolved handler '" << handlerName
                      << "' for address 0x" << std::hex << address << std::dec << std::endl;
            return false;
        }

        return runtime.replaceFunction(address, resolved.value());
    }

    void applyMatching(PS2Runtime &runtime,
                       const std::string &elfPath,
                       uint32_t entry,
                       uint32_t fileCrc32,
                       bool fileCrcValid)
    {

        std::vector<Descriptor> descriptors;
        {
            std::lock_guard<std::mutex> lock(registryMutex());
            descriptors = descriptorRegistry();
        }

        if (descriptors.empty())
        {
            return;
        }

        const std::string elfName = basenameFromPath(elfPath);
        size_t appliedCount = 0;
        for (const Descriptor &descriptor : descriptors)
        {
            if (!descriptor.apply)
            {
                continue;
            }

            if (descriptor.elfName && descriptor.elfName[0] != '\0')
            {
                if (!equalsIgnoreCaseAscii(descriptor.elfName, elfName))
                {
                    continue;
                }
            }

            if (descriptor.entry != 0u && descriptor.entry != entry)
            {
                continue;
            }

            if (descriptor.crc32 != 0u)
            {
                if (!fileCrcValid || fileCrc32 != descriptor.crc32)
                {
                    continue;
                }
            }

            const char *name = (descriptor.name && descriptor.name[0] != '\0')
                                   ? descriptor.name
                                   : "unnamed";
            RUNTIME_LOG("[game_overrides] applying '" << name << "'");
            descriptor.apply(runtime);
            ++appliedCount;
        }

        if (appliedCount > 0)
        {
            RUNTIME_LOG("[game_overrides] applied " << appliedCount << " matching override(s).");
        }
    }
}

// ---------------------------------------------------------------------------
// War of the Monsters (SCUS_971.97) — skip the intro FMV.
//
// The intro movie streams through sceMpeg + sceCdStream, a pipeline that does
// not yet run to completion under the recompiler, so the shell never advances
// past "Playing Intro movie". The shell (uiMain__Fi) runs an intro loop that
// each frame calls uiIntro (0x001DA3C8), playMovie__Fv (0x001DA4B8) and
// updateMpegMovieDecoding, then spins at 0x001D981C until the gp-relative flag
// at gp-0x7440 becomes non-zero (normally set by uiIntro once the movie has
// finished and playMovie's state at gp-0x7480 is back to 0). Replacing playMovie
// with a stub that clears its own state/arm flags (gp-0x7480, gp-0x744C) *and*
// directly raises the intro-complete flag (gp-0x7440) makes the shell leave the
// intro loop on the first frame and continue to the main menu. Remove once real
// FMV playback works.
// ---------------------------------------------------------------------------
namespace
{
    constexpr uint32_t kWotmPlayMovieAddr = 0x001DA4B8u;
    // Signed 16-bit gp displacements used by the shell intro loop.
    constexpr uint32_t kWotmMovieStateAddr = static_cast<uint32_t>(-0x7480);
    constexpr uint32_t kWotmMovieArmedAddr = static_cast<uint32_t>(-0x744C);
    constexpr uint32_t kWotmIntroDoneAddr = static_cast<uint32_t>(-0x7440);

    void wotmWriteGuestU32(uint8_t *rdram, uint32_t gp, uint32_t disp, uint32_t value)
    {
        if (uint8_t *field = getMemPtr(rdram, gp + disp))
        {
            *reinterpret_cast<uint32_t *>(field) = value;
        }
    }

    static PS2Runtime::RecompiledFunction s_wotmBootSync;
    static bool wotmBootLoading(uint8_t *ram)
    {
        uint32_t done;
        std::memcpy(&done, ram + 0x2a6c84u, 4); // MonsterMc::global boot-complete
        return done == 0;
    }
    static void wotmQuietBootSync(uint8_t *ram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t sync = getRegU32(ctx, 4);
        if (sync != 0x2a3750u || !wotmBootLoading(ram)) {
            s_wotmBootSync(ram, ctx, runtime); return;
        }
        // Only the minimum display duration is changed. Original task polling,
        // completion, settings validation and error handling still execute.
        struct Duration {
            uint8_t *field; uint32_t saved;
            explicit Duration(uint8_t *p) : field(p) {
                std::memcpy(&saved, field, 4); uint32_t zero=0;
                std::memcpy(field, &zero, 4);
            }
            ~Duration() { std::memcpy(field, &saved, 4); }
        } duration(ram + sync + 0x30u);
        static unsigned calls=0;
        if (++calls <= 8) std::fprintf(stderr,"[skip-intro] boot data task, display delay removed\n");
        s_wotmBootSync(ram, ctx, runtime);
    }
    void wotmSkipPlayMovie(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        (void)runtime;
        const uint32_t gp = getRegU32(ctx, 28); // $gp
        wotmWriteGuestU32(rdram, gp, kWotmMovieStateAddr, 0u); // playMovie state = idle
        wotmWriteGuestU32(rdram, gp, kWotmMovieArmedAddr, 0u); // movie disarmed
        wotmWriteGuestU32(rdram, gp, kWotmIntroDoneAddr, 1u);  // intro complete -> shell continues
        ctx->pc = getRegU32(ctx, 31);                          // return to caller ($ra)
    }

    // -----------------------------------------------------------------------
    // hierLoadVu1Ucode__Fv (0x2031a8) / hierLoadVu0Ucode__Fv (0x203220).
    //
    // Both functions build a VIF chain-DMA kick with the classic
    //   lui r,0x1000 ; ori r,0xNNNN ; ... ; sw x,0(r)
    // pattern. The recompiler folds the store target to the bare `lui`
    // immediate (0x10000000) instead of the ori-completed MMIO register, so
    // the TADR / D_STAT / CHCR writes all land on 0x10000000 and the VU
    // microcode DMA is never kicked. VU1 then executes all-zeros and every
    // menu/scene prim comes out untransformed (garbled). Re-issue the four
    // register writes by hand with the correct addresses. The leading
    // FlushCache / sceGsSyncPath(0,0) calls the real functions make are
    // no-ops under the recompiler (synchronous DMA, no cache model), so the
    // stub skips straight to the kick. Remove once the recompiler emits the
    // correct MMIO address for this pattern.
    // -----------------------------------------------------------------------
    void wotmHierLoadVu1Ucode(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        runtime->Store32(rdram, ctx, 0x10009020u, 0u);          // VIF1 ch1 QWC = 0
        runtime->Store32(rdram, ctx, 0x10009030u, 0x00257C10u); // VIF1 ch1 TADR
        runtime->Store32(rdram, ctx, 0x1000E010u, 2u);          // D_STAT: clear ch1
        runtime->Store32(rdram, ctx, 0x10009000u, 0x00000145u); // VIF1 ch1 CHCR: chain+TTE+STR (kick)
        ctx->pc = getRegU32(ctx, 31);
    }

    void wotmHierLoadVu0Ucode(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        runtime->Store32(rdram, ctx, 0x10008020u, 0u);          // VIF0 ch0 QWC = 0
        runtime->Store32(rdram, ctx, 0x10008030u, 0x0025B890u); // VIF0 ch0 TADR
        runtime->Store32(rdram, ctx, 0x1000E010u, 1u);          // D_STAT: clear ch0
        runtime->Store32(rdram, ctx, 0x10008000u, 0x00000145u); // VIF0 ch0 CHCR: chain+TTE+STR (kick)
        ctx->pc = getRegU32(ctx, 31);
    }

    // -----------------------------------------------------------------------
    // snd_StreamSafeCd{Read,Sync,Break,GetError} (989SND, 0x1ed000..0x1ed1b8).
    //
    // Once VAG streaming is initialised (cinema music at the end of an
    // Adventure level), 989SND routes the game's disc reads through the IOP
    // sound driver: StreamSafeCdRead sends command 0x38 and sets gStats = 1,
    // and StreamSafeCdSync spins until the IOP reads the sectors and clears
    // gStats. Our 989SND is a stub with no disc streaming, so the save /
    // next-level load after a won level hung forever. Route all four straight
    // to the libcdvd calls they use when streaming is off (the game's own
    // recompiled sceCdRead/Sync/Break/GetError, backed by the CD stubs).
    // Arguments are identical, so each is a tail call.
    // -----------------------------------------------------------------------
    void wotmTailCall(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, uint32_t target)
    {
        if (PS2Runtime::RecompiledFunction fn = runtime->lookupFunction(target))
        {
            fn(rdram, ctx, runtime); // returns to the caller's $ra itself
            return;
        }
        std::cerr << "[game_overrides] wotm: missing libcdvd function 0x" << std::hex << target << std::dec
                  << std::endl;
        setReturnS32(ctx, 0);
        ctx->pc = getRegU32(ctx, 31);
    }
    void wotmStreamSafeCdRead(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        wotmTailCall(rdram, ctx, runtime, 0x0023E360u); // sceCdRead
    }
    void wotmStreamSafeCdSync(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        wotmTailCall(rdram, ctx, runtime, 0x0023DB10u); // sceCdSync
    }
    void wotmStreamSafeCdBreak(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        wotmTailCall(rdram, ctx, runtime, 0x0023E5D8u); // sceCdBreak
    }
    void wotmStreamSafeCdGetError(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        wotmTailCall(rdram, ctx, runtime, 0x0023E540u); // sceCdGetError
    }

    PS2Runtime::RecompiledFunction s_wotmOriginalTimerEnd=nullptr;
    static void wotmTraceProjectionMatrices(uint8_t *rdram, uint64_t frame);
    void wotmPublishTarget(uint8_t *rdram)
    {
        if(auto *p=getMemPtr(rdram,0x006f8dd8u))
            g_ps2xWotmTargetHz.store(*reinterpret_cast<const uint32_t*>(p),std::memory_order_relaxed);
    }
    void wotmMenuTimerEnd(uint8_t *rdram,R5900Context *ctx,PS2Runtime *runtime)
    {
        // userintMain has incremented g_frame before this call. This timer
        // function only computes timing statistics; keep its original body.
        if(getRegU32(ctx,31)==0x001d9f64u) {
            g_ps2xWotmMenuFrames.fetch_add(1u,std::memory_order_relaxed);
            g_ps2xWotmPhase.store(1u,std::memory_order_relaxed);
            g_ps2xWotmWideActive.store(false,std::memory_order_relaxed);
            wotmPublishTarget(rdram);
        }
        s_wotmOriginalTimerEnd(rdram,ctx,runtime);
    }

    // RtLoopView::end is a retail no-op (JR ra; NOP). rtMain calls it
    // several times per loop; only RA=0x188AE0 follows the frame increment
    // at 0x188AB4 and timerWaitForMinUpdateRate. Count that boundary once.
    // This is game-loop throughput, not GPU completion or display refresh.
    void wotmEndLoopView(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t returnPc = getRegU32(ctx, 31);
        if (returnPc == 0x00188AE0u)
        {
            ps2xTracePacing("frame",rdram,runtime,returnPc);
            ps2xPacingFrameComplete(runtime);
            g_ps2xWotmPhase.store(2u,std::memory_order_relaxed);
            wotmPublishTarget(rdram);
#ifdef _WIN32
            static const bool interpolate=[] {const char* p=std::getenv("PS2X_FRAME_INTERPOLATION");return p && p[0]=='1';}();
            // Seal a complete geometry stream before advancing its history.
            // Waiting here holds no GS/record lock needed by the VU worker.
            if(interpolate)runtime->memory().drainVu1Worker();
#endif
            const uint64_t frame = g_ps2xWotmCompletedFrames.fetch_add(1u, std::memory_order_relaxed) + 1u;
            wotmTraceProjectionMatrices(rdram, frame);
            wotmDestructionFrame(frame);
            wotmMotionFrame(frame);
#ifdef _WIN32
            if(interpolate) {
            const auto& registers=runtime->memory().gs();
            GSPresentationRequest display{};display.pmode=registers.pmode;display.smode2=registers.smode2;
            display.dispfb1=registers.dispfb1;display.dispfb2=registers.dispfb2;
            display.display1=registers.display1;display.display2=registers.display2;display.bgcolor=registers.bgcolor;
            ps2xGsSealCompletedFrame(frame,display);
            }
#endif
            // Diagnostic only: flush periodically so a hang preserves evidence.
            static const char *path = std::getenv("PS2X_FRAME_CSV");
            if (path && *path)
            {
                static const uint64_t first = [] { const char *p=std::getenv("PS2X_FRAME_START"); return p?std::strtoull(p,nullptr,10):1600ull; }();
                static const uint64_t last = [] { const char *p=std::getenv("PS2X_FRAME_END"); return p?std::strtoull(p,nullptr,10):2200ull; }();
                static auto previous = std::chrono::steady_clock::now();
                static uint64_t oldInstructions=0, oldRuns=0;
                static bool written=false;
                static std::FILE *file=nullptr;
                static size_t rowCount=0;
                const auto now=std::chrono::steady_clock::now();
                const uint64_t instructions=g_ps2xVuInstr.load(std::memory_order_relaxed);
                const uint64_t runs=g_ps2xVuRuns.load(std::memory_order_relaxed);
                if(frame>=first && frame<=last && frame>1 && rowCount<100000)
                {
                    if(!file)
                    {
                        file=std::fopen(path,"w");
                        if(file) std::fprintf(file,"frame,ms,vu_instructions,vu_runs\n");
                        else std::fprintf(stderr,"[bench:frames] cannot open %s\n",path);
                    }
                    if(file)
                    {
                        std::fprintf(file,"%llu,%.6f,%llu,%llu\n",frame,
                            std::chrono::duration<double,std::milli>(now-previous).count(),
                            instructions-oldInstructions,runs-oldRuns);
                        if((++rowCount%60u)==0u) std::fflush(file);
                    }
                }
                previous=now; oldInstructions=instructions; oldRuns=runs;
                if(frame>=last && !written)
                {
                    written=true;
                    if(file)
                    {
                        std::fflush(file);
                        std::fclose(file);
                        file=nullptr;
                        std::fprintf(stderr,"[bench:frames] saved %zu rows to %s\n",rowCount,path);
                    }
                }
            }
        }
        ctx->pc = returnPc;
    }

    // -----------------------------------------------------------------------
    // War of the Monsters: the cheats, on their own screen in the pause menu.
    //
    // Retail only lets you turn these on with controller combos entered during
    // play (checkCombos__8PadFlagsR7GamePad @0x1783e8), and only once the
    // "codes" global freeThePeople has been walked up to 2. Each cheat is a
    // single byte on the player's Monster object, XOR-toggled, followed by a HUD
    // banner. We drive exactly that same state, so the game cannot tell the
    // difference -- nothing here reimplements a cheat, it only flips the bytes
    // the game already flips.
    //
    // The pause overlay is left running untouched and four of its helpers are
    // wrapped instead:
    //   getCurrentMenuSelection(count, player) takes the entry count as an
    //     argument and navigates modulo it, so handing it 6 instead of 5 is all
    //     the extra CHEATS row needs. Retail keys its actions off the RETURN
    //     value with QUIT at 4, so the two are swapped back on the way out:
    //     QUIT still tears the menu down, CHEATS returns something inert.
    //   fontSpritePrintCenteredXY() drops retail's QUIT row so we can redraw it
    //     one line lower, and drops every pause row while the cheats screen is
    //     up, which is what makes it a separate screen rather than more rows.
    //   checkInputForExit() is how X reaches the CONTINUE/RESTART/QUIT switch.
    //     On the CHEATS row X must not get there -- but START still has to
    //     unpause, and the game reports START through the out-flag, so only the
    //     X case is suppressed.
    //   pauseLevelOverlay() itself draws and drives whichever of the two
    //     screens is current, after the original has drawn its own rows.
    // -----------------------------------------------------------------------
    constexpr uint32_t kWotmPauseOverlayAddr = 0x001AE310u;
    constexpr uint32_t kWotmGetMenuSelAddr = 0x001B0070u;
    constexpr uint32_t kWotmCheckExitAddr = 0x001AFE40u;
    constexpr uint32_t kWotmSetMenuFontAddr = 0x001AFFD8u;
    constexpr uint32_t kWotmFontPrintAddr = 0x001FCD30u;
    constexpr uint32_t kWotmInputGetInputAddr = 0x0020BE08u;
    constexpr uint32_t kWotmAddMessageAddr = 0x001457D0u;
    constexpr uint32_t kWotmPlayMenuSoundAddr = 0x001B6800u;
    // The string retail draws on its last pause row.
    constexpr uint32_t kWotmQuitTextAddr = 0x006F8220u;

    // Absolute pointers the overlay itself loads through `lui 0x70`.
    constexpr uint32_t kWotmGamePtrAddr = 0x006F81F8u;
    constexpr uint32_t kWotmShellPtrAddr = 0x006F8200u;
    constexpr uint32_t kWotmPlayerThatPausedAddr = 0x006F82B8u;
    // gp-relative small-data globals.
    constexpr int32_t kWotmMenuSelDisp = -0x76E4;
    constexpr int32_t kWotmDrawOverlayDisp = -0x76D8;
    constexpr int32_t kWotmFreeThePeopleDisp = -0x6C88;

    // Per-player Monster pointer table, and the cheat bytes on that object.
    constexpr uint32_t kWotmMonsterTableOff = 0x120380u;
    constexpr uint32_t kWotmMonsterHudIdxOff = 0x6CD8u;
    constexpr uint32_t kWotmHudStride = 0x2E0u;

    // inputGetInput button bits, as the overlay itself uses them.
    constexpr int32_t kWotmPadUp = 0x10;
    constexpr int32_t kWotmPadRight = 0x20;
    constexpr int32_t kWotmPadDown = 0x40;
    constexpr int32_t kWotmPadLeft = 0x80;
    constexpr int32_t kWotmPadTriangle = 0x1000;
    constexpr int32_t kWotmPadCross = 0x4000;
    constexpr int32_t kWotmSoundMove = 0x50;
    constexpr int32_t kWotmSoundBack = 0x4F;

    // Retail draws five rows centred on x=320 from y=0x55, one every 9
    // scanlines: CONTINUE / RESTART / VIBRATION / CONTROL SCHEME / QUIT.
    constexpr int32_t kWotmMenuCentreX = 0x140;
    constexpr int32_t kWotmMenuFirstY = 0x55;
    constexpr int32_t kWotmMenuRowStep = 9;
    constexpr int32_t kWotmRetailQuitY = 0x79; // where retail puts QUIT
    constexpr int32_t kWotmCheatsRow = 4;      // CHEATS takes that slot
    constexpr int32_t kWotmQuitRow = 5;        // QUIT moves down one line
    constexpr int32_t kWotmBaseMenuRows = 6;

    // The overlay's black plate, and the switch node that selects its variant.
    constexpr uint32_t kWotmPlateCsOff = 0x610u;
    constexpr uint32_t kWotmPlateSwitchOff = 0x618u;
    constexpr uint32_t kWotmPlateMatrixXRow = 0x20u;
    constexpr uint32_t kWotmPlateMatrixYRow = 0x30u; // row1: the Y basis
    constexpr uint32_t kWotmSwitchSelectedOff = 0x8u;  // hierSetSwitch writes here
    constexpr uint32_t kWotmSwitchCountOff = 0xBu;
    constexpr uint8_t kWotmHierSwitchType = 6u;
    // Variant 0 projects to ~160 pixels wide and 49 field scanlines tall,
    // centred near y=108. Text uses the same 640x224 logical coordinates.
    constexpr float kWotmPlateCentreY = 108.0f;
    constexpr float kWotmPlateHalfHeight = 24.5f;
    // This is a divisor: lowering it widens the panel. 1.7 was narrower
    // than the user-approved 1.5; 1.3 adds room for CONTROL TYPE.
    constexpr float kWotmPlateWidthDivisor = 1.3f;

    // The cheats screen: a title, the toggles, then a back hint.
    constexpr int32_t kWotmSubTitleY = 0x55;
    constexpr int32_t kWotmSubFirstY = 0x67;

    struct WotmCheat
    {
        uint8_t flagOffset; // byte on the Monster object
        uint8_t messageOn;  // Hud::addMessage banner ids
        uint8_t messageOff;
        const char *label;
    };

    // Offsets and banner ids read out of checkCombos and cross-checked against
    // the banner string table at 0x006e9b18, which is also where the wording
    // comes from. The ids are NOT in offset order: 0xED (super homing) uses
    // 0x16/0x1E while 0xFC (rapid fire) uses 0x15/0x1D.
    constexpr WotmCheat kWotmCheats[] = {
        {0xE9u, 0x11u, 0x19u, "GOD MODE"},
        {0xEAu, 0x12u, 0x1Au, "INVINCIBLE"},
        {0xEBu, 0x13u, 0x1Bu, "ENERGY"},
        {0xECu, 0x14u, 0x1Cu, "KILLER ATTACKS"},
        {0xEDu, 0x16u, 0x1Eu, "SUPER HOMING"},
        {0xFCu, 0x15u, 0x1Du, "RAPID FIRE"},
    };
    constexpr int32_t kWotmCheatRows = static_cast<int32_t>(sizeof(kWotmCheats) / sizeof(kWotmCheats[0]));

    // Guest-visible label storage, rebuilt every frame so ON/OFF is current.
    // One slot per cheat, plus the screen title and the back hint.
    constexpr uint32_t kWotmLabelStride = 32u;
    constexpr uint32_t kWotmTitleSlot = static_cast<uint32_t>(kWotmCheatRows);
    constexpr uint32_t kWotmHintSlot = kWotmTitleSlot + 1u;
    constexpr uint32_t kWotmSelectPromptSlot = kWotmHintSlot + 1u;
    constexpr uint32_t kWotmBackPromptSlot = kWotmHintSlot + 2u;
    constexpr uint32_t kWotmScratchSlot = kWotmHintSlot + 3u;
    constexpr uint32_t kWotmLabelBytes = kWotmLabelStride * (kWotmScratchSlot + 1u);

    uint32_t s_wotmLabelBase = 0u;
    bool s_wotmInPauseOverlay = false;
    bool s_wotmCheatScreenOpen = false;
    // Set by the checkInputForExit hook, consumed by the overlay wrapper.
    bool s_wotmCrossPending = false;
    int32_t s_wotmCheatCursor = 0;
    PS2Runtime::RecompiledFunction s_wotmOriginalPauseOverlay = nullptr;
    PS2Runtime::RecompiledFunction s_wotmOriginalGetMenuSel = nullptr;
    PS2Runtime::RecompiledFunction s_wotmOriginalCheckExit = nullptr;
    PS2Runtime::RecompiledFunction s_wotmOriginalFontPrint = nullptr;

    bool wotmCheatTrace()
    {
        static const bool on = [] {
            const char *v = std::getenv("PS2X_WOTM_CHEAT_TRACE");
            return v && v[0] != '0';
        }();
        return on;
    }

    inline void wotmSetReg(R5900Context *ctx, int reg, int32_t value)
    {
        ctx->r[reg] = _mm_set_epi64x(0, static_cast<int64_t>(value));
    }

    uint32_t wotmRead32(uint8_t *rdram, uint32_t addr)
    {
        if (const uint8_t *field = getMemPtr(rdram, addr))
            return *reinterpret_cast<const uint32_t *>(field);
        return 0u;
    }

    void wotmWrite32At(uint8_t *rdram, uint32_t addr, uint32_t value)
    {
        if (uint8_t *field = getMemPtr(rdram, addr))
            *reinterpret_cast<uint32_t *>(field) = value;
    }

    // Run a guest function to completion from native code. Recompiled functions
    // return by assigning ctx->pc from $ra, so the caller's $ra and pc are put
    // back afterwards; everything else the guest ABI already treats as clobbered
    // across a call.
    int32_t wotmCallFn(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime,
                       PS2Runtime::RecompiledFunction fn, int32_t a0 = 0, int32_t a1 = 0, int32_t a2 = 0,
                       int32_t a3 = 0)
    {
        if (!fn)
            return 0;
        const uint32_t savedRa = getRegU32(ctx, 31);
        const uint32_t savedPc = ctx->pc;
        wotmSetReg(ctx, 4, a0);
        wotmSetReg(ctx, 5, a1);
        wotmSetReg(ctx, 6, a2);
        wotmSetReg(ctx, 7, a3);
        fn(rdram, ctx, runtime);
        const int32_t result = static_cast<int32_t>(getRegU32(ctx, 2));
        wotmSetReg(ctx, 31, static_cast<int32_t>(savedRa));
        ctx->pc = savedPc;
        return result;
    }

    int32_t wotmCallGuest(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, uint32_t target,
                          int32_t a0 = 0, int32_t a1 = 0, int32_t a2 = 0, int32_t a3 = 0)
    {
        return wotmCallFn(rdram, ctx, runtime, runtime->lookupFunction(target), a0, a1, a2, a3);
    }

    uint32_t wotmLabelAddr(uint32_t slot) { return s_wotmLabelBase + slot * kWotmLabelStride; }

    // PS2X_WOTM_PLATE_VARIANT=<n>: pick one of the plate shapes the game already
    // ships instead of retail's 0. Same validation hierSetSwitch does.
    void wotmSelectPlateVariant(uint8_t *rdram)
    {
        static const int32_t variant = [] {
            const char *v = std::getenv("PS2X_WOTM_PLATE_VARIANT");
            return v ? std::atoi(v) : -1;
        }();
        if (variant < 0)
            return;
        const uint32_t shell = wotmRead32(rdram, kWotmShellPtrAddr);
        const uint32_t node = shell != 0u ? wotmRead32(rdram, shell + kWotmPlateSwitchOff) : 0u;
        if (node == 0u)
            return;
        const uint8_t *type = getMemPtr(rdram, node);
        const uint8_t *count = getMemPtr(rdram, node + kWotmSwitchCountOff);
        if (!type || !count || (*type & 0x3Fu) != kWotmHierSwitchType || variant >= static_cast<int32_t>(*count))
            return;
        if (uint8_t *selected = getMemPtr(rdram, node + kWotmSwitchSelectedOff))
            *selected = static_cast<uint8_t>(variant);
    }

    // Fit the overlay plate around the text, including the final row. The
    // overlay rewrites this matrix every frame, so this has to run after it.
    void wotmScalePlate(uint8_t *rdram, int32_t lastRowY)
    {
        static const float forced = [] {
            const char *v = std::getenv("PS2X_WOTM_PLATE_SCALE");
            return v ? static_cast<float>(std::atof(v)) : 0.0f;
        }();
        // Use actual text coordinates, not a row count: CHEATS has spacer rows.
        // Scaling grows around the model's centre, so fit both ends, with room
        // for glyph descent and one row of padding below the final baseline.
        const float halfHeight = std::max(kWotmPlateCentreY - (kWotmMenuFirstY - 9.0f),
                                         lastRowY + 9.0f - kWotmPlateCentreY);
        const float scale = forced > 0.0f ? forced
                                          : halfHeight / kWotmPlateHalfHeight;

        const uint32_t shell = wotmRead32(rdram, kWotmShellPtrAddr);
        if (shell == 0u)
            return;
        const uint32_t cs = wotmRead32(rdram, shell + kWotmPlateCsOff);
        if (cs == 0u)
            return;
        for (uint32_t i = 0; i < 3u; ++i)
        {
            if (uint8_t *field = getMemPtr(rdram, cs + kWotmPlateMatrixXRow + i * 4u))
                *reinterpret_cast<float *>(field) /= kWotmPlateWidthDivisor;
            if (uint8_t *field = getMemPtr(rdram, cs + kWotmPlateMatrixYRow + i * 4u))
                *reinterpret_cast<float *>(field) /= scale;
        }
    }

    void wotmWriteLabel(uint8_t *rdram, uint32_t slot, const char *text)
    {
        if (uint8_t *field = getMemPtr(rdram, wotmLabelAddr(slot)))
            std::snprintf(reinterpret_cast<char *>(field), kWotmLabelStride, "%s", text);
    }

    // setMenuFont(i) draws in the highlight style when i == gCurrentMenuSelection
    // and the normal style otherwise, so handing it that value or anything else
    // picks a style. The cheats screen has its own cursor, unrelated to the
    // global, which is why it cannot just pass a row index.
    void wotmSetRowStyle(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, bool highlighted)
    {
        const uint32_t gp = getRegU32(ctx, 28);
        const int32_t selection =
            static_cast<int32_t>(wotmRead32(rdram, gp + static_cast<uint32_t>(kWotmMenuSelDisp)));
        wotmCallGuest(rdram, ctx, runtime, kWotmSetMenuFontAddr, highlighted ? selection : selection - 1);
    }

    void wotmPrintRow(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, int32_t y, uint32_t textAddr)
    {
        wotmCallFn(rdram, ctx, runtime, s_wotmOriginalFontPrint, 0, kWotmMenuCentreX, y,
                   static_cast<int32_t>(textAddr));
    }

    // The Monster the pausing player is controlling, or 0 when there isn't one
    // (paused over a cutscene, say).
    uint32_t wotmPausedMonster(uint8_t *rdram, uint32_t &gameBase, int32_t &player)
    {
        gameBase = wotmRead32(rdram, kWotmGamePtrAddr);
        player = static_cast<int32_t>(wotmRead32(rdram, kWotmPlayerThatPausedAddr));
        if (gameBase == 0u || player < 0 || player > 1)
            return 0u;
        return wotmRead32(rdram, gameBase + kWotmMonsterTableOff + static_cast<uint32_t>(player) * 4u);
    }

    void wotmToggleCheat(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, int32_t index)
    {
        uint32_t gameBase = 0u;
        int32_t player = 0;
        const uint32_t monster = wotmPausedMonster(rdram, gameBase, player);
        if (monster == 0u)
            return;

        const WotmCheat &cheat = kWotmCheats[index];
        uint8_t *flag = getMemPtr(rdram, monster + cheat.flagOffset);
        if (!flag)
            return;
        *flag ^= 1u;

        // Entering any code by hand leaves freeThePeople at 2; keep that true so
        // the combo path and anything else gated on it agree with the menu.
        const uint32_t gp = getRegU32(ctx, 28);
        wotmWrite32At(rdram, gp + static_cast<uint32_t>(kWotmFreeThePeopleDisp), 2u);

        if (wotmCheatTrace())
            std::fprintf(stderr, "[wotm:cheat] toggle %s -> %s (monster 0x%08x)\n", cheat.label,
                         *flag ? "ON" : "OFF", monster);

        const uint32_t hudIndex = wotmRead32(rdram, monster + kWotmMonsterHudIdxOff);
        wotmCallGuest(rdram, ctx, runtime, kWotmAddMessageAddr,
                      static_cast<int32_t>(gameBase + hudIndex * kWotmHudStride),
                      *flag ? cheat.messageOn : cheat.messageOff, 0);
    }

    void wotmPlayMenuSound(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, int32_t sound)
    {
        const uint32_t shell = wotmRead32(rdram, kWotmShellPtrAddr);
        if (shell != 0u)
            wotmCallGuest(rdram, ctx, runtime, kWotmPlayMenuSoundAddr, static_cast<int32_t>(shell + 0x2918u), sound);
    }

    // fontSpritePrintCenteredXY(font, x, y, text): drop the rows we are taking
    // over. Retail's QUIT row is redrawn a line lower; while the cheats screen
    // is up every pause row is dropped, which is what makes it its own screen.
    void wotmFontSpritePrintCentered(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (s_wotmInPauseOverlay &&
            (s_wotmCheatScreenOpen || static_cast<int32_t>(getRegU32(ctx, 6)) == kWotmRetailQuitY))
        {
            setReturnU32(ctx, 0u);
            ctx->pc = getRegU32(ctx, 31);
            return;
        }
        s_wotmOriginalFontPrint(rdram, ctx, runtime);
    }

    // getCurrentMenuSelection(count, player): one more row for the pause menu,
    // and the CHEATS/QUIT swap on the way back out. Every other caller passes
    // its own count and is untouched.
    void wotmGetCurrentMenuSelection(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        if (s_wotmInPauseOverlay && s_wotmCheatScreenOpen)
        {
            // The cheats screen owns the d-pad. Park the pause menu on CHEATS
            // and hand back a value none of its rows react to.
            setReturnU32(ctx, static_cast<uint32_t>(kWotmBaseMenuRows));
            ctx->pc = getRegU32(ctx, 31);
            return;
        }

        const bool pauseMenu = s_wotmInPauseOverlay && static_cast<int32_t>(getRegU32(ctx, 4)) == 5;
        if (pauseMenu)
            wotmSetReg(ctx, 4, kWotmBaseMenuRows);
        s_wotmOriginalGetMenuSel(rdram, ctx, runtime);
        if (t_ps2xEeUnwinding || !pauseMenu)
            return;

        const int32_t selection = static_cast<int32_t>(getRegU32(ctx, 2));
        if (selection == kWotmCheatsRow)
            setReturnU32(ctx, static_cast<uint32_t>(kWotmBaseMenuRows)); // inert
        else if (selection == kWotmQuitRow)
            setReturnU32(ctx, static_cast<uint32_t>(kWotmCheatsRow)); // retail's QUIT index
    }

    // checkInputForExit(player, outStartPressed): swallow the X press on the
    // CHEATS row so it never reaches the CONTINUE/QUIT switch. START arrives
    // with the out-flag set and is left alone, so it still unpauses.
    void wotmCheckInputForExit(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t outPtr = getRegU32(ctx, 5);
        s_wotmOriginalCheckExit(rdram, ctx, runtime);
        if (t_ps2xEeUnwinding || !s_wotmInPauseOverlay || getRegU32(ctx, 2) == 0u)
            return;
        if (outPtr != 0u && wotmRead32(rdram, outPtr) != 0u)
            return; // START: let the overlay treat it as CONTINUE
        const uint32_t gp = getRegU32(ctx, 28);
        const int32_t selection =
            static_cast<int32_t>(wotmRead32(rdram, gp + static_cast<uint32_t>(kWotmMenuSelDisp)));
        if (selection == kWotmCheatsRow)
        {
            // This is the only place the X press is visible -- inputGetInput is
            // one-shot and the call above has just consumed it.
            s_wotmCrossPending = true;
            setReturnU32(ctx, 0u);
        }
    }

    // Append screen-space primitives to retail's font packet, ahead of the
    // submenu text. This avoids scaling the camera-relative 3D plate. Addresses
    // and packet-state transitions match fontBuildPrim/fontSetColorGifTag in
    // SCUS_971.97; packetBuf spans 0x735900..0x764700 (12000 quadwords).
    bool wotmDrawCheatPanel(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime, int32_t player)
    {
        constexpr uint32_t buffer = 0x00735900u;
        constexpr uint32_t indexAddr = 0x006F8D30u;
        constexpr uint32_t stateAddr = 0x006F8D3Cu;
        constexpr uint32_t tagAddr = 0x006F8D40u;
        constexpr uint32_t countAddr = 0x006F8D2Cu;
        const uint32_t scratch = wotmLabelAddr(kWotmScratchSlot);
        const int32_t view = wotmCallGuest(rdram, ctx, runtime, 0x002287B8u);
        wotmCallGuest(rdram, ctx, runtime, 0x00229070u, view, scratch, scratch + 4u);
        wotmCallGuest(rdram, ctx, runtime, 0x002290A8u, view, scratch + 8u, scratch + 12u);
        const int32_t baseX = static_cast<int32_t>(wotmRead32(rdram, scratch)) * 16 -
                              static_cast<int32_t>(wotmRead32(rdram, scratch + 8u)) * 8;
        const int32_t baseY = static_cast<int32_t>(wotmRead32(rdram, scratch + 4u)) * 16 -
                              static_cast<int32_t>(wotmRead32(rdram, scratch + 12u)) * 8;
        struct Primitive { uint64_t prim, rgbaq, xy0, xy1; };
        std::vector<Primitive> shapes;
        const auto xy = [&](float x, float y) {
            return static_cast<uint64_t>((baseX + static_cast<int>(x * 16.0f)) & 0xffff) |
                   (static_cast<uint64_t>((baseY + static_cast<int>(y * 16.0f)) & 0xffff) << 16u) |
                   (uint64_t{0x00fffffeu} << 32u);
        };
        const auto shape = [&](uint64_t prim, float x0, float y0, float x1, float y1, uint32_t rgba) {
            shapes.push_back({prim, (uint64_t{0x3f800000u} << 32u) | rgba, xy(x0, y0), xy(x1, y1)});
        };
        const auto rect = [&](int x0, int y0, int x1, int y1, uint32_t rgba) {
            shape(0x46u, x0, y0, x1, y1, rgba); // untextured, alpha-blended GS sprite
        };
        // Rounded panel: stepped corners in the game's 640x224 coordinates.
        constexpr uint32_t black = 0x68000000u, edge = 0x80787878u;
        rect(144, 76, 496, 181, black);
        rect(140, 80, 144, 177, black);
        rect(496, 80, 500, 177, black);
        for (int i = 0; i < 4; ++i) {
            rect(143-i, 77+i, 144, 78+i, black);
            rect(496, 77+i, 497+i, 78+i, black);
            rect(143-i, 179-i, 144, 180-i, black);
            rect(496, 179-i, 497+i, 180-i, black);
            rect(143-i, 76+i, 144-i, 77+i, edge);
            rect(496+i, 76+i, 497+i, 77+i, edge);
            rect(143-i, 180-i, 144-i, 181-i, edge);
            rect(496+i, 180-i, 497+i, 181-i, edge);
        }
        rect(144, 76, 496, 77, edge);
        rect(144, 180, 496, 181, edge);
        rect(140, 80, 141, 177, edge);
        rect(499, 80, 500, 177, edge);

        // Close any pending glyph run using the game's own implementation.
        wotmCallGuest(rdram, ctx, runtime, 0x001FC740u, 0);
        const uint32_t index = wotmRead32(rdram, indexAddr);
        const uint32_t count = static_cast<uint32_t>(shapes.size());
        const uint32_t panelEnd = index + 1u + count * 2u;
        const uint32_t end = panelEnd + 11u;
        if (index >= 11000u || end >= 11000u) return false; // leave headroom for text + DMA terminator
        uint8_t *dst = getMemPtr(rdram, buffer + index * 16u);
        if (!dst || !getMemPtr(rdram, buffer + end * 16u)) return false;
        // GIF REGLIST: PRIM, RGBAQ, XYZ2, XYZ2. No persistent TEST/ALPHA/FRAME
        // changes: use the same UI state and offset as the adjacent font glyphs.
        const uint64_t tag[2] = {(uint64_t{4} << 60u) | (uint64_t{1} << 58u) | count, 0x5510u};
        std::memcpy(dst, tag, sizeof(tag));
        std::memcpy(dst + 16u, shapes.data(), shapes.size() * sizeof(Primitive));
        // Textured footer sprites use the same retail button atlas as every
        // other prompt. Its native handle selects the controller's artwork
        // without allocating/overwriting PS2 VRAM. Save retail's font TEX0
        // from its initialization packet (fontBuildPrim / fontSetTexId).
        const uint32_t texTag = wotmRead32(rdram, 0x006F8D34u);
        uint64_t fontTex0 = 0u;
        if (texTag >= 11u) return false;
        std::memcpy(&fontTex0, getMemPtr(rdram, buffer + texTag * 16u), 8u);
        uint64_t *packet = reinterpret_cast<uint64_t *>(dst + (panelEnd-index) * 16u);
        const auto qw = [&](uint64_t lo, uint64_t hi) { *packet++ = lo; *packet++ = hi; };
        qw((1ull << 60) | 1u, 0xEu); // PACKED A+D, TEX0_1
        qw(ps2xNativeButtonTex0(player), 6u);
        qw((6ull << 60) | (1ull << 58) | 2u, 0x535310u);
        const auto uv = [](int u, int v) { return static_cast<uint64_t>(u) | (static_cast<uint64_t>(v) << 16); };
        const auto button = [&](int x, int u) {
            qw(0x156u, (uint64_t{0x3f800000u} << 32) | 0x80808080u);
            qw(uv(u*16+8, 63*16+8), xy(x-12,198));
            qw(uv((u+31)*16+8, 32*16+8), xy(x+12,210));
        };
        button(194,32); // Cross / A: bottom-right cell, flipped retail V
        button(362,0);  // Triangle / Y: bottom-left cell
        qw((1ull << 60) | 1u, 0xEu);
        qw(fontTex0, 6u);
        wotmWrite32At(rdram, indexAddr, end);
        wotmWrite32At(rdram, stateAddr, 2u);
        wotmWrite32At(rdram, tagAddr, end - 2u);
        wotmWrite32At(rdram, countAddr, 1u);
        // Close the final TEX0 restore tag and resume retail's font PRIM/RGBAQ.
        wotmCallGuest(rdram, ctx, runtime, 0x001FC740u, 0);
        return true;
    }

    void wotmDrawCheatScreen(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        uint32_t gameBase = 0u;
        int32_t owner = 0;
        const uint32_t monster = wotmPausedMonster(rdram, gameBase, owner);

        wotmWriteLabel(rdram, kWotmTitleSlot, "CHEATS");
        wotmSetRowStyle(rdram, ctx, runtime, true);
        wotmPrintRow(rdram, ctx, runtime, kWotmSubTitleY, wotmLabelAddr(kWotmTitleSlot));

        for (int32_t i = 0; i < kWotmCheatRows; ++i)
        {
            const WotmCheat &cheat = kWotmCheats[i];
            bool on = false;
            if (monster != 0u)
            {
                if (const uint8_t *flag = getMemPtr(rdram, monster + cheat.flagOffset))
                    on = (*flag & 1u) != 0u;
            }
            if (uint8_t *text = getMemPtr(rdram, wotmLabelAddr(static_cast<uint32_t>(i))))
            {
                std::snprintf(reinterpret_cast<char *>(text), kWotmLabelStride, "%-17s%s", cheat.label,
                              monster == 0u ? "--" : (on ? "ON" : "OFF"));
            }
            wotmSetRowStyle(rdram, ctx, runtime, i == s_wotmCheatCursor);
            wotmPrintRow(rdram, ctx, runtime, kWotmSubFirstY + i * kWotmMenuRowStep,
                         wotmLabelAddr(static_cast<uint32_t>(i)));
        }

        wotmWriteLabel(rdram, kWotmHintSlot, "BACK");
        wotmSetRowStyle(rdram, ctx, runtime, s_wotmCheatCursor == kWotmCheatRows);
        wotmPrintRow(rdram, ctx, runtime, kWotmSubFirstY + (kWotmCheatRows + 1) * kWotmMenuRowStep,
                     wotmLabelAddr(kWotmHintSlot));
        wotmSetRowStyle(rdram, ctx, runtime, false);
        wotmWriteLabel(rdram, kWotmSelectPromptSlot, "SELECT");
        wotmWriteLabel(rdram, kWotmBackPromptSlot, "BACK");
        wotmCallFn(rdram, ctx, runtime, s_wotmOriginalFontPrint, 0, 254, 201,
                  wotmLabelAddr(kWotmSelectPromptSlot));
        wotmCallFn(rdram, ctx, runtime, s_wotmOriginalFontPrint, 0, 408, 201,
                  wotmLabelAddr(kWotmBackPromptSlot));

    }

    void wotmPauseLevelOverlay(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t callerRa = getRegU32(ctx, 31);
        g_ps2xWotmPaused.store(true,std::memory_order_release);

        // This wrapper appends native UI work after retail returns. A generated
        // checkpoint cannot unwind across it: that would discard the C++ work
        // and its callerRa while leaving a live guest frame on the stack.
        // Only this bounded menu update defers checkpoints, not gameplay or
        // blocking syscalls. RAII also clears the font/input scope if a syscall
        // transfers control back to the dispatcher.
        struct MenuCallScope
        {
            bool previous = s_wotmInPauseOverlay;
            MenuCallScope() { ++t_ps2xEeCheckpointDeferralDepth; s_wotmInPauseOverlay = true; }
            ~MenuCallScope() { s_wotmInPauseOverlay = previous; --t_ps2xEeCheckpointDeferralDepth; }
        } scope;
        s_wotmOriginalPauseOverlay(rdram, ctx, runtime);
        s_wotmInPauseOverlay = scope.previous;

        const int32_t overlayResult = static_cast<int32_t>(getRegU32(ctx, 2));
        // Non-zero means the overlay is tearing down (continue / restart / quit),
        // so the cheats screen goes with it.
        if (overlayResult != 0 || s_wotmLabelBase == 0u)
        {
            g_ps2xWotmPaused.store(false,std::memory_order_release);
            s_wotmCheatScreenOpen = false;
            s_wotmCrossPending = false; // never leak a press into the next pause
            ctx->pc = callerRa;
            return;
        }

        const uint32_t gp = getRegU32(ctx, 28);
        const int32_t selection =
            static_cast<int32_t>(wotmRead32(rdram, gp + static_cast<uint32_t>(kWotmMenuSelDisp)));
        const int32_t player = static_cast<int32_t>(wotmRead32(rdram, kWotmPlayerThatPausedAddr));
        const bool drawOverlay = wotmRead32(rdram, gp + static_cast<uint32_t>(kWotmDrawOverlayDisp)) != 0u;

        // Which screen the ORIGINAL just drew for. Opening or closing takes
        // effect next frame, so one frame never shows both.
        const bool screenWasOpen = s_wotmCheatScreenOpen;

        const auto pressed = [&](int32_t button) {
            return wotmCallGuest(rdram, ctx, runtime, kWotmInputGetInputAddr, button, player) != 0;
        };

        if (screenWasOpen)
        {
            if (pressed(kWotmPadTriangle))
            {
                s_wotmCheatScreenOpen = false;
                wotmPlayMenuSound(rdram, ctx, runtime, kWotmSoundBack);
            }
            else
            {
                if (pressed(kWotmPadDown))
                {
                    s_wotmCheatCursor = (s_wotmCheatCursor + 1) % (kWotmCheatRows + 1);
                    wotmPlayMenuSound(rdram, ctx, runtime, kWotmSoundMove);
                }
                else if (pressed(kWotmPadUp))
                {
                    s_wotmCheatCursor = (s_wotmCheatCursor + kWotmCheatRows) % (kWotmCheatRows + 1);
                    wotmPlayMenuSound(rdram, ctx, runtime, kWotmSoundMove);
                }
                if (s_wotmCheatCursor == kWotmCheatRows)
                {
                    if (s_wotmCrossPending)
                    {
                        s_wotmCheatScreenOpen = false;
                        wotmPlayMenuSound(rdram, ctx, runtime, kWotmSoundBack);
                    }
                }
                else if (s_wotmCrossPending || pressed(kWotmPadLeft) || pressed(kWotmPadRight))
                {
                    wotmToggleCheat(rdram, ctx, runtime, s_wotmCheatCursor);
                    wotmPlayMenuSound(rdram, ctx, runtime, kWotmSoundMove);
                }
            }
        }
        else if (selection == kWotmCheatsRow && s_wotmCrossPending)
        {
            s_wotmCheatScreenOpen = true;
            s_wotmCheatCursor = 0;
            wotmPlayMenuSound(rdram, ctx, runtime, kWotmSoundMove);
            if (wotmCheatTrace())
                std::fprintf(stderr, "[wotm:cheat] cheats screen opened\n");
        }

        if (drawOverlay)
        {
            // Title + toggles + back hint on the cheats screen; the six pause
            // rows otherwise.
            if (screenWasOpen)
            {
                if (wotmDrawCheatPanel(rdram, ctx, runtime, player))
                {
                    const uint32_t shell = wotmRead32(rdram, kWotmShellPtrAddr);
                    const uint32_t cs = shell ? wotmRead32(rdram, shell + kWotmPlateCsOff) : 0u;
                    // hierSetCsDrawMe stores a byte at +0xc. Retail resets it
                    // on the next overlay update, including return to pause.
                    if (cs) if (uint8_t *draw = getMemPtr(rdram, cs + 0xcu)) *draw = 0u;
                }
                wotmDrawCheatScreen(rdram, ctx, runtime);
            }
            else
            {
                wotmSelectPlateVariant(rdram);
                wotmScalePlate(rdram, kWotmMenuFirstY + kWotmQuitRow * kWotmMenuRowStep);
                // CHEATS goes where retail drew QUIT; QUIT follows a line lower,
                // its own row having been dropped by the font hook.
                wotmWriteLabel(rdram, kWotmTitleSlot, "CHEATS");
                wotmCallGuest(rdram, ctx, runtime, kWotmSetMenuFontAddr, kWotmCheatsRow);
                wotmPrintRow(rdram, ctx, runtime, kWotmMenuFirstY + kWotmCheatsRow * kWotmMenuRowStep,
                             wotmLabelAddr(kWotmTitleSlot));
                wotmCallGuest(rdram, ctx, runtime, kWotmSetMenuFontAddr, kWotmQuitRow);
                wotmPrintRow(rdram, ctx, runtime, kWotmMenuFirstY + kWotmQuitRow * kWotmMenuRowStep,
                             kWotmQuitTextAddr);
            }
        }

        s_wotmCrossPending = false;

        if (wotmCheatTrace())
        {
            // Is there a taller plate already in the hierarchy? A switch node
            // keeps its child count in the byte at +0xb, and retail always
            // picks 0. If there is more than one, that beats scaling.
            static bool switchReported = false;
            if (!switchReported)
            {
                switchReported = true;
                const uint32_t shell = wotmRead32(rdram, kWotmShellPtrAddr);
                const uint32_t node = shell != 0u ? wotmRead32(rdram, shell + kWotmPlateSwitchOff) : 0u;
                const uint8_t *type = node != 0u ? getMemPtr(rdram, node) : nullptr;
                const uint8_t *count = node != 0u ? getMemPtr(rdram, node + 0xbu) : nullptr;
                std::fprintf(stderr, "[wotm:cheat] plate switch 0x%08x type=%u children=%u cs=0x%08x\n", node,
                             type ? (*type & 0x3Fu) : 0xFFu, count ? *count : 0u,
                             shell != 0u ? wotmRead32(rdram, shell + kWotmPlateCsOff) : 0u);
            }

            static int32_t reported = -1;
            static bool reportedOpen = false;
            if (selection != reported || screenWasOpen != reportedOpen)
            {
                reported = selection;
                reportedOpen = screenWasOpen;
                uint32_t traceGame = 0u;
                int32_t traceOwner = 0;
                std::fprintf(stderr, "[wotm:cheat] sel=%d screen=%s cursor=%d monster=0x%08x draw=%d\n", selection,
                             screenWasOpen ? "cheats" : "pause", s_wotmCheatCursor,
                             wotmPausedMonster(rdram, traceGame, traceOwner), drawOverlay ? 1 : 0);
            }
        }

        setReturnU32(ctx, 0u);
        ctx->pc = callerRa;
    }

    void wotmInstallCheatMenu(PS2Runtime &runtime)
    {
        if (const char *flag = std::getenv("PS2X_WOTM_CHEAT_MENU"); flag && flag[0] == '0')
            return;

        s_wotmOriginalPauseOverlay = runtime.lookupFunction(kWotmPauseOverlayAddr);
        s_wotmOriginalGetMenuSel = runtime.lookupFunction(kWotmGetMenuSelAddr);
        s_wotmOriginalCheckExit = runtime.lookupFunction(kWotmCheckExitAddr);
        s_wotmOriginalFontPrint = runtime.lookupFunction(kWotmFontPrintAddr);
        if (!s_wotmOriginalPauseOverlay || !s_wotmOriginalGetMenuSel || !s_wotmOriginalCheckExit ||
            !s_wotmOriginalFontPrint || s_wotmOriginalPauseOverlay == &wotmPauseLevelOverlay)
        {
            std::cerr << "[game_overrides] wotm: pause-menu cheats unavailable (missing overlay functions)"
                      << std::endl;
            return;
        }

        // Label storage the guest will never allocate over. The async-callback
        // arena is carved downward out of the top of RAM and sits above the heap
        // limit the guest is told about, so a reservation here is ours for the
        // run.
        const uint32_t reserved = runtime.reserveAsyncCallbackStack(kWotmLabelBytes, 16u);
        if (reserved == 0u)
        {
            std::cerr << "[game_overrides] wotm: pause-menu cheats unavailable (no label storage)" << std::endl;
            return;
        }
        s_wotmLabelBase = reserved + 0x10u - kWotmLabelBytes;

        runtime.replaceFunction(kWotmPauseOverlayAddr, &wotmPauseLevelOverlay);
        runtime.replaceFunction(kWotmGetMenuSelAddr, &wotmGetCurrentMenuSelection);
        runtime.replaceFunction(kWotmCheckExitAddr, &wotmCheckInputForExit);
        runtime.replaceFunction(kWotmFontPrintAddr, &wotmFontSpritePrintCentered);
        RUNTIME_LOG("[game_overrides] wotm: CHEATS screen added to the pause menu ("
                    << kWotmCheatRows << " toggles)");
    }


    #include "wotm_widescreen.inc"
    #include "wotm_mods.inc"
    #include "wotm_modern_camera.inc"
    #include "wotm_random.inc"
    #include "wotm_sweet_tooth.inc"

    void applyWarOfTheMonstersOverrides(PS2Runtime &runtime)
    {
        runtime.replaceFunction(0x0023f988u, &wotmSrand);
        runtime.replaceFunction(0x0023f998u, &wotmRand);
        s_wotmSweetChoice = -2;
        const auto unlockInput = runtime.lookupFunction(0x001a58a0u);
        if (unlockInput && unlockInput != &wotmSweetInput) {
            s_wotmUnlockInput = unlockInput;
            runtime.replaceFunction(0x001a58a0u, &wotmSweetInput);
        }
        g_ps2xWotmCompletedFrames.store(0u, std::memory_order_relaxed);
        g_ps2xWotmMenuFrames.store(0u,std::memory_order_relaxed);
        g_ps2xWotmPhase.store(0u,std::memory_order_relaxed);
        g_ps2xWotmWideActive.store(false,std::memory_order_relaxed);
        g_ps2xWotmTargetHz.store(0u,std::memory_order_relaxed);
        const auto timerEnd=runtime.lookupFunction(0x002233c0u);
        if(timerEnd && timerEnd!=&wotmMenuTimerEnd) {
            s_wotmOriginalTimerEnd=timerEnd;
            runtime.replaceFunction(0x002233c0u,&wotmMenuTimerEnd);
        }

        g_ps2xWotmFrameMetricsEnabled.store(
            runtime.replaceFunction(0x00188310u, &wotmEndLoopView), std::memory_order_relaxed);
        if (!g_ps2xWotmFrameMetricsEnabled.load(std::memory_order_relaxed))
            std::cerr << "[game_overrides] wotm: frame counter hook unavailable" << std::endl;
        if (std::getenv("PS2X_WOTM_STREAMSAFE_CD") == nullptr || std::getenv("PS2X_WOTM_STREAMSAFE_CD")[0] != '0')
        {
            const bool ok = runtime.replaceFunction(0x001ED000u, &wotmStreamSafeCdRead) &&
                            runtime.replaceFunction(0x001ED0C0u, &wotmStreamSafeCdSync) &&
                            runtime.replaceFunction(0x001ED170u, &wotmStreamSafeCdBreak) &&
                            runtime.replaceFunction(0x001ED1B8u, &wotmStreamSafeCdGetError);
            if (!ok)
                std::cerr << "[game_overrides] wotm: failed to replace snd_StreamSafeCd*" << std::endl;
        }
        // Intro-FMV skip (wotmSkipPlayMovie) retired 2026-09-10: the sceMpeg +
        // sceCdStream path now plays the main-menu drive-in movie correctly, so
        // playMovie__Fv runs for real. Set PS2X_WOTM_SKIP_INTRO=1 to re-enable.
        if (const char *skip = std::getenv("PS2X_WOTM_SKIP_INTRO"); skip && skip[0] == '1')
        {
            s_wotmBootSync=runtime.lookupFunction(0x149918u);
            if(s_wotmBootSync)
                runtime.replaceFunction(0x149918u, &wotmQuietBootSync);
            if (!runtime.replaceFunction(kWotmPlayMovieAddr, &wotmSkipPlayMovie))
            {
                std::cerr << "[game_overrides] wotm-skip-intro-movie: failed to replace playMovie__Fv"
                          << std::endl;
            }
        }
        if (!runtime.replaceFunction(0x002031A8u, &wotmHierLoadVu1Ucode))
        {
            std::cerr << "[game_overrides] wotm: failed to replace hierLoadVu1Ucode__Fv"
                      << std::endl;
        }
        // hierLoadVu0Ucode (0x203220) has the same recompiler MMIO mis-fold as
        // hierLoadVu1Ucode. VU0 micro-mode IS needed for the scene-graph
        // traversal (hierTraverseAsm `vcallms` at 0x0/0x3F8/0x748/0x398 — the
        // per-node matrix math). VIF0 MPG → m_vu0Code IS wired
        // (ps2_vif1_interpreter.cpp processVIF0Data VIF_MPG). The earlier "stall"
        // when this was enabled was actually the finishXgkick stack-smash crash,
        // now fixed.
        // Replace every registered resume slot belonging to the same walker.
        // Disabling the specialization leaves the generated dispatch table intact.
        const char *nativeTraversal=std::getenv("PS2X_NATIVE_EE_TRAVERSAL");
        if (!nativeTraversal || nativeTraversal[0]!='0') {
            const bool check=std::getenv("PS2X_EE_TRAVERSAL_VERIFY")!=nullptr;
            const auto original=runtime.lookupFunction(0x207198u);
            unsigned slots=0;
            for(uint32_t pc=0x207198u;pc<0x208380u;pc+=4u)
                if(original && runtime.hasFunction(pc) && runtime.lookupFunction(pc)==original) {
                    runtime.replaceFunction(pc,check ? &wotmCheckedTraversal : &wotmNativeTraversal);
                    ++slots;
                }
            std::fprintf(stderr,"[ee:traversal] native slots=%u verify=%u\n",slots,unsigned(check));
        }
        const char *nativePackets=std::getenv("PS2X_NATIVE_EE_PACKETS");
        if(!nativePackets || nativePackets[0]!='0') {
            const bool check=std::getenv("PS2X_EE_PACKETS_VERIFY")!=nullptr;
            const auto original=runtime.lookupFunction(0x208410u);
            unsigned slots=0;
            for(uint32_t pc=0x208410u;pc<0x20895cu;pc+=4u)
                if(original && runtime.hasFunction(pc) && runtime.lookupFunction(pc)==original) {
                    runtime.replaceFunction(pc,check ? &wotmCheckedObjectPacket : &wotmNativeObjectPacket);
                    ++slots;
                }
            std::fprintf(stderr,"[ee:packets] native slots=%u verify=%u\n",slots,unsigned(check));
        }
        wotmInstallCheatMenu(runtime);
        wotmInstallWidescreen(runtime);
        wotmInstallModernCamera(runtime);
        // The launcher supplies declarative roster slots; the old replacement
        // variable remains available only for isolated diagnostic runs.
        wotmInstallMods(runtime);
        wotmInstallDestructionProfile(runtime);
        wotmInstallMotionHistory(runtime);
        if (!runtime.replaceFunction(0x00203220u, &wotmHierLoadVu0Ucode))
        {
            std::cerr << "[game_overrides] wotm: failed to replace hierLoadVu0Ucode__Fv"
                      << std::endl;
        }
    }
}

PS2_REGISTER_GAME_OVERRIDE("wotm-skip-intro-movie", "SCUS_971.97", 0u, 0u,
                           &applyWarOfTheMonstersOverrides)
