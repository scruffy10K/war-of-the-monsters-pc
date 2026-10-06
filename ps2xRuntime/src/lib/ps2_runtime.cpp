#include "installed_disc.inc"
#include <cmath>
#ifdef _WIN32
extern bool ps2xD3D12Enabled();
extern bool ps2xD3D12Present(void*);
extern void ps2xD3D12Shutdown();
#endif

#include "runtime/ps2_perf_clock.h"
#include "ps2_runtime.h"
#include "ps2_log.h"
#include "ps2_stubs.h"
#include "ps2_syscalls.h"
#include "game_overrides.h"
#include "ps2_runtime_macros.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/gs/gs_gl_backend.h"
#include "runtime/ee_scheduler.h"
#include "ThreadNaming.h"
#include "Kernel/Stubs/Audio.h"
#include "Kernel/Stubs/GS.h"
#include "Kernel/Stubs/MPEG.h"
#include "ps2_host_backend.h"
#include "rlgl.h" // blend state for presenting the GPU render target (needs raylib.h first)
#include "ps2_iop_host.h"
#include "ps2x/iop/iop_subsystem.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <limits>
#include <chrono>
#include <atomic>
#include <thread>
#include <mutex>
#include <set>
#include <unordered_map>
#include <sstream>

namespace ps2_stubs
{
    void resetSifState();
}

// Defined in gs_frontend.cpp — one-shot full-frame GS stream capture budget.
extern std::atomic<int64_t> g_gsFrameDump;
extern std::atomic<uint64_t> g_ps2xWotmCompletedFrames;
extern std::atomic<bool> g_ps2xWotmPaused;
#ifdef _WIN32
extern bool ps2xGsHasCompletedFrame();
#endif
extern std::atomic<uint64_t> g_ps2xWotmMenuFrames;
extern std::atomic<uint32_t> g_ps2xWotmTargetHz,g_ps2xWotmPhase;
extern std::atomic<bool> g_ps2xWotmWideActive;
extern std::atomic<bool> g_ps2xRosterEnabled, g_ps2xRosterPage;
extern std::atomic<int> g_ps2xRosterSlot;
extern std::atomic<uint64_t> g_ps2xRosterHeartbeatNs;
extern uint32_t g_ps2xRosterIds[10];
extern char g_ps2xRosterNames[10][33];

extern std::atomic<bool> g_ps2xWotmFrameMetricsEnabled;
extern std::atomic<uint64_t> g_ps2xEeTransferThrows; // EeScheduler.cpp
extern std::atomic<uint64_t> g_ps2xEeTransferThrowsBySite[5];
extern std::atomic<uint64_t> g_ps2xEeTransferSkips;
extern std::atomic<uint64_t> g_ps2xEeTransferSkipBlockers[4];
extern std::atomic<uint64_t> g_ps2xThrowUnwindTsc;
extern std::atomic<uint64_t> g_ps2xRotateBranch[3];
extern std::atomic<uint64_t> g_ps2xVu1NativeRuns, g_ps2xVu1InterpRuns, g_ps2xVu1NativeFallbacks; // ps2_vu1_core.cpp
extern std::atomic<uint64_t> g_ps2xVu1JobsSubmitted, g_ps2xVu1Drains, g_ps2xVu1DrainWaitNs; // ps2_memory.cpp worker
extern std::atomic<uint64_t> g_ps2xGsForeignHazards; // gs_frontend.cpp
extern std::atomic<uint64_t> g_ps2xVif1Transfers, g_ps2xVif1InsideTsc, g_ps2xVif1GapDmaReads,
    g_ps2xVif1GapsWithPoll, g_ps2xVif1GapsWithBlock, g_ps2xVif1GapsWithWaitCall,
    g_ps2xVif1GapsWithSignal; // ps2_memory.cpp parallel-VU1 probe

#define ELF_MAGIC 0x464C457F // "\x7FELF" in little endian
#define ET_EXEC 2            // Executable file
#define EM_MIPS 8            // MIPS architecture
#define PT_LOAD 1            // Loadable segment

static constexpr int FB_WIDTH = 640;
static constexpr int FB_HEIGHT = 512;
static constexpr int DEFAULT_DISPLAY_HEIGHT = 448;
static constexpr uint32_t DEFAULT_FB_SIZE = FB_WIDTH * FB_HEIGHT * 4;
static constexpr uint32_t DEFAULT_FB_ADDR = (PS2_RAM_SIZE - DEFAULT_FB_SIZE - 0x10000u);
#if defined(PLATFORM_VITA)
static constexpr int HOST_WINDOW_WIDTH = 960;
static constexpr int HOST_WINDOW_HEIGHT = 544;
#else
static constexpr int HOST_WINDOW_WIDTH = FB_WIDTH;
static constexpr int HOST_WINDOW_HEIGHT = DEFAULT_DISPLAY_HEIGHT;
#endif
struct ElfHeader
{
    uint32_t magic;
    uint8_t elf_class;
    uint8_t endianness;
    uint8_t version;
    uint8_t os_abi;
    uint8_t abi_version;
    uint8_t padding[7];
    uint16_t type;
    uint16_t machine;
    uint32_t version2;
    uint32_t entry;
    uint32_t phoff;
    uint32_t shoff;
    uint32_t flags;
    uint16_t ehsize;
    uint16_t phentsize;
    uint16_t phnum;
    uint16_t shentsize;
    uint16_t shnum;
    uint16_t shstrndx;
};

struct ProgramHeader
{
    uint32_t type;
    uint32_t offset;
    uint32_t vaddr;
    uint32_t paddr;
    uint32_t filesz;
    uint32_t memsz;
    uint32_t flags;
    uint32_t align;
};

namespace
{
    constexpr uint32_t kGuestHeapDefaultBase = 0x00100000u;
    constexpr uint32_t kGuestHeapDefaultAlignment = 16u;
    constexpr uint32_t kGuestHeapSafetyPad = 0x1000u;
    constexpr uint32_t kGuestHeapHardLimit = 0x01F00000u;

    constexpr uint32_t COP0_CAUSE_EXCCODE_MASK = 0x0000007Cu;
    constexpr uint32_t COP0_CAUSE_BD = 0x80000000u;
    constexpr uint32_t COP0_STATUS_EXL = 0x00000002u;
    constexpr uint32_t COP0_STATUS_BEV = 0x00400000u;
    constexpr uint32_t EXCEPTION_VECTOR_GENERAL = 0x80000080u;
    constexpr uint32_t EXCEPTION_VECTOR_TLB_REFILL = 0x80000000u;
    constexpr uint32_t EXCEPTION_VECTOR_BOOT = 0xBFC00200u;

    struct DispatchHistory
    {
        std::array<uint32_t, 64> pcs{};
        uint32_t next = 0u;
        bool wrapped = false;
    };

    thread_local DispatchHistory g_dispatchHistory;

    bool computeFileCrc32(const std::string &path, uint32_t &crcOut)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open())
        {
            return false;
        }

        static const std::array<uint32_t, 256> table = []
        {
            std::array<uint32_t, 256> values{};
            for (uint32_t i = 0; i < values.size(); ++i)
            {
                uint32_t value = i;
                for (uint32_t bit = 0; bit < 8; ++bit)
                {
                    value = (value & 1u) ? (0xEDB88320u ^ (value >> 1u)) : (value >> 1u);
                }
                values[i] = value;
            }
            return values;
        }();

        uint32_t crc = 0xFFFFFFFFu;
        std::array<uint8_t, 16 * 1024> buffer{};
        while (file.good())
        {
            file.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
            const std::streamsize count = file.gcount();
            for (std::streamsize i = 0; i < count; ++i)
            {
                crc = table[(crc ^ buffer[static_cast<size_t>(i)]) & 0xFFu] ^ (crc >> 8u);
            }
        }
        if (file.bad())
        {
            return false;
        }
        crcOut = ~crc;
        return true;
    }

    void pushDispatchPc(uint32_t pc)
    {
        DispatchHistory &h = g_dispatchHistory;
        h.pcs[h.next] = pc;
        h.next = (h.next + 1u) % static_cast<uint32_t>(h.pcs.size());
        if (h.next == 0u)
        {
            h.wrapped = true;
        }
    }

    std::string formatDispatchHistory()
    {
        const DispatchHistory &h = g_dispatchHistory;
        const uint32_t count = h.wrapped ? static_cast<uint32_t>(h.pcs.size()) : h.next;
        if (count == 0u)
        {
            return "(empty)";
        }

        std::ostringstream oss;
        bool first = true;
        for (uint32_t i = 0u; i < count; ++i)
        {
            const uint32_t idx = (h.next + h.pcs.size() - count + i) % static_cast<uint32_t>(h.pcs.size());
            if (!first)
            {
                oss << " -> ";
            }
            first = false;
            oss << "0x" << std::hex << h.pcs[idx];
        }
        return oss.str();
    }

    uint32_t selectExceptionVector(const R5900Context *ctx, bool tlbRefill)
    {
        if (ctx->cop0_status & COP0_STATUS_BEV)
        {
            return EXCEPTION_VECTOR_BOOT;
        }
        return tlbRefill ? EXCEPTION_VECTOR_TLB_REFILL : EXCEPTION_VECTOR_GENERAL;
    }

    void seedVu0IdleSuccess(R5900Context *ctx)
    {
        if (!ctx)
        {
            return;
        }

        ctx->vu0_clip_flags = 0;
        ctx->vu0_clip_flags2 = 0;
        ctx->vu0_mac_flags = 0;
        ctx->vu0_status = 0;
        ctx->vu0_q = 1.0f;
        ctx->vu0_r = _mm_castsi128_ps(_mm_set1_epi32(0x3F800000));
        ctx->vu0_vpu_stat = 0;
        ctx->vu0_vpu_stat2 = 0;
    }

    void copyVu0ContextToState(const R5900Context *ctx, VU1State &state)
    {
        // The sole caller has just reset the interpreter and its pipelines.
        // Avoid clearing the same state twice; constant VF0/VI0 are set below.
        for (uint32_t i = 1; i < 32u; ++i)
        {
            _mm_storeu_ps(state.vf[i], ctx->vu0_vf[i]);
        }
        for (uint32_t i = 1; i < 16u; ++i)
        {
            state.vi[i] = static_cast<int16_t>(ctx->vi[i]);
        }

        _mm_storeu_ps(state.acc, ctx->vu0_acc);
        state.q = ctx->vu0_q;
        state.p = ctx->vu0_p;
        state.i = ctx->vu0_i;
        alignas(16) uint32_t rWords[4]{};
        _mm_storeu_si128(reinterpret_cast<__m128i *>(rWords), _mm_castps_si128(ctx->vu0_r));
        state.r = 0x3F800000u | (rWords[0] & 0x007FFFFFu);
        state.pc = ctx->vu0_pc;
        state.mac = ctx->vu0_mac_flags;
        state.clip = ctx->vu0_clip_flags;
        state.status = ctx->vu0_status;
        state.itop = ctx->vu0_itop;
        state.dBitEnabled = (ctx->vu0_fbrst & (1u << 2)) != 0u;
        state.tBitEnabled = (ctx->vu0_fbrst & (1u << 3)) != 0u;

        state.vf[0][0] = 0.0f;
        state.vf[0][1] = 0.0f;
        state.vf[0][2] = 0.0f;
        state.vf[0][3] = 1.0f;
        state.vi[0] = 0;
    }

    void copyVu0StateToContext(const VU1State &state, R5900Context *ctx)
    {
        for (uint32_t i = 0; i < 32u; ++i)
        {
            ctx->vu0_vf[i] = _mm_loadu_ps(state.vf[i]);
        }
        for (uint32_t i = 0; i < 16u; ++i)
        {
            ctx->vi[i] = static_cast<uint16_t>(state.vi[i]);
        }

        ctx->vu0_acc = _mm_loadu_ps(state.acc);
        ctx->vu0_q = state.q;
        ctx->vu0_p = state.p;
        ctx->vu0_i = state.i;
        ctx->vu0_r = _mm_castsi128_ps(_mm_set1_epi32(static_cast<int32_t>(state.r)));
        ctx->vu0_mac_flags = state.mac;
        ctx->vu0_clip_flags = state.clip;
        ctx->vu0_clip_flags2 = state.clip;
        ctx->vu0_status = static_cast<uint16_t>(state.status);
        ctx->vu0_itop = state.itop;
        ctx->vu0_pc = state.pc;
        ctx->vu0_tpc = state.pc;
        ctx->vu0_vpu_stat = (ctx->vu0_vpu_stat & 0xFF00u) | (state.stoppedByD ? (1u << 1) : 0u) | (state.stoppedByT ? (1u << 2) : 0u);
        ctx->vu0_vpu_stat2 = 0;

        ctx->vu0_vf[0] = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
        ctx->vi[0] = 0;
    }

    void raiseCop0Exception(R5900Context *ctx, uint32_t exceptionCode, bool tlbRefill = false)
    {
        if (ctx->in_delay_slot)
        {
            ctx->cop0_epc = ctx->branch_pc;
            ctx->cop0_cause = (ctx->cop0_cause & ~COP0_CAUSE_EXCCODE_MASK) |
                              ((exceptionCode << 2) & COP0_CAUSE_EXCCODE_MASK) |
                              COP0_CAUSE_BD;
        }
        else
        {
            ctx->cop0_epc = ctx->pc;
            ctx->cop0_cause = (ctx->cop0_cause & ~(COP0_CAUSE_EXCCODE_MASK | COP0_CAUSE_BD)) |
                              ((exceptionCode << 2) & COP0_CAUSE_EXCCODE_MASK);
        }

        ctx->cop0_status |= COP0_STATUS_EXL;
        ctx->pc = selectExceptionVector(ctx, tlbRefill);
        ctx->in_delay_slot = false;
    }

    std::filesystem::path normalizeAbsolutePath(const std::filesystem::path &path)
    {
        if (path.empty())
        {
            return {};
        }

#if defined(PLATFORM_VITA)
        const std::string generic = path.generic_string();
        const std::size_t colon = generic.find(':');
        if (colon != std::string::npos && colon != 0u)
        {
            const std::size_t slash = generic.find_first_of("/\\");
            if (slash == std::string::npos || colon < slash)
            {
                return path.lexically_normal();
            }
        }
#endif

        std::error_code ec;
        const std::filesystem::path absolute = std::filesystem::absolute(path, ec);
        if (ec)
        {
            return path.lexically_normal();
        }
        return absolute.lexically_normal();
    }

    PS2Runtime::IoPaths &runtimeIoPaths()
    {
        static PS2Runtime::IoPaths paths = []()
        {
            PS2Runtime::IoPaths defaults;
            std::error_code ec;
            const std::filesystem::path cwd = std::filesystem::current_path(ec);
            defaults.elfDirectory = ec ? std::filesystem::path(".") : cwd.lexically_normal();
            defaults.hostRoot = defaults.elfDirectory;
            defaults.cdRoot = defaults.elfDirectory;
            defaults.mcRoot = defaults.elfDirectory / "mc0";
            return defaults;
        }();

        return paths;
    }

    std::string readGuestPrintableString(const uint8_t *rdram, uint32_t addr, size_t maxLen)
    {
        std::string out;
        if (!rdram || maxLen == 0)
        {
            return out;
        }

        out.reserve(std::min<size_t>(maxLen, 64));
        for (size_t i = 0; i < maxLen; ++i)
        {
            const char ch = static_cast<char>(rdram[(addr + static_cast<uint32_t>(i)) & PS2_RAM_MASK]);
            if (ch == '\0')
            {
                break;
            }
            if (ch >= 0x20 && ch < 0x7F)
            {
                out.push_back(ch);
            }
            else
            {
                out.push_back('.');
            }
        }
        return out;
    }
}

// Coarse hot-path accounting (always on, ~zero cost): where the EE thread's
// time goes. Printed every 5 s as [perf] from the host frame upload.
struct Ps2xPerf { std::atomic<uint64_t> rasterNs, presentNs, vu1Ns, vif1Ns, prims; };
extern std::atomic<uint32_t> g_ps2xEndimgQueued;
extern std::atomic<uint32_t> g_ps2xGifKicks;
extern std::atomic<uint32_t> g_ps2xCause2;
Ps2xPerf g_ps2xPerf{};
static double s_perfLastEeIdlePct = 0.0; // [perf] waits: EE idle share of the last window (stall detector)
std::atomic<uint64_t> g_ps2xSyncPathCalls{0u}; // Kernel/Stubs/GS.cpp sceGsSyncPath

// Latches the finished GS frame once per guest vsync. Both presentation paths
// need it: the upload path reads the latched pixels, and the GPU path relies on
// the latch to be what calls the backend's Present at all.
static bool LatchFrameIfNeeded(PS2Runtime *rt)
{
    static uint64_t s_lastPresentationTick = std::numeric_limits<uint64_t>::max();
    static bool s_hasLatched = false;
    const uint64_t currentTick = rt->eeScheduler().currentVSyncTick();
#ifdef _WIN32
    static const bool interpolate=[] {const char* p=std::getenv("PS2X_FRAME_INTERPOLATION");return p && p[0]=='1';}();
    if(interpolate && ps2xD3D12Enabled()) {
        static auto lastCompleted=std::chrono::steady_clock::now();
        const auto now=std::chrono::steady_clock::now();
        if(ps2xGsHasCompletedFrame()) {
            rt->gs().latchHostPresentationFrame();lastCompleted=now;s_lastPresentationTick=currentTick;s_hasLatched=true;return true;
        }
        if(!g_ps2xWotmPaused.load(std::memory_order_acquire) &&
           g_ps2xWotmPhase.load(std::memory_order_relaxed)==2u &&
           now-lastCompleted<std::chrono::milliseconds(80))return false;
        // Movies/loading/pause continue on the existing presentation clock.
    }
#endif
    if (s_hasLatched && currentTick == s_lastPresentationTick)
        return false;
    rt->gs().latchHostPresentationFrame();
    s_lastPresentationTick = currentTick;
    s_hasLatched = true;
    return true;
}

static void ReportRuntimeProgress(PS2Runtime *rt)
{
    {
        static auto s_perfT0 = std::chrono::steady_clock::now();
        static uint64_t s_perfTick0 = 0u;
        static Ps2xPerf s_perfLast{};
        const auto now = std::chrono::steady_clock::now();
        const double wall = std::chrono::duration<double>(now - s_perfT0).count();
        if (wall >= 5.0)
        {
            const uint64_t tick = rt->eeScheduler().currentVSyncTick();
            static uint64_t s_lastGameFrames = 0u;
            const uint64_t gameFrames = g_ps2xWotmCompletedFrames.load(std::memory_order_relaxed);
            const uint64_t gameFrameDelta = gameFrames >= s_lastGameFrames ? gameFrames - s_lastGameFrames : 0u;
            s_lastGameFrames = gameFrames;
            static uint64_t lastMenuFrames=0;
            const uint64_t menuFrames=g_ps2xWotmMenuFrames.load(std::memory_order_relaxed);
            const uint64_t menuDelta=menuFrames>=lastMenuFrames?menuFrames-lastMenuFrames:0;
            lastMenuFrames=menuFrames;
            if(g_ps2xWotmPhase.load(std::memory_order_relaxed)==1u)
                std::fprintf(stderr,"[perf:menu] %.2f FPS target=%u completed=%llu\n",menuDelta/wall,g_ps2xWotmTargetHz.load(std::memory_order_relaxed),menuFrames);

            {
                extern std::atomic<uint64_t> g_ps2xRasterIdleWaits, g_ps2xRasterIdleWaitNs, g_ps2xEeIdleWaitNs;
                static uint64_t s_lw = 0u, s_lwn = 0u, s_ein = 0u;
                const uint64_t w = g_ps2xRasterIdleWaits.load(std::memory_order_relaxed);
                const uint64_t wn = g_ps2xRasterIdleWaitNs.load(std::memory_order_relaxed);
                const uint64_t en = g_ps2xEeIdleWaitNs.load(std::memory_order_relaxed);
                std::fprintf(stderr, "[perf] waits: raster-idle %.0f/s %.1f%% of wall (%.2f ms avg) | EE idle (all guest threads blocked) %.1f%%\n",
                             static_cast<double>(w - s_lw) / wall, 100.0 * static_cast<double>(wn - s_lwn) * 1e-9 / wall,
                             (w - s_lw) ? static_cast<double>(wn - s_lwn) / static_cast<double>(w - s_lw) * 1e-6 : 0.0,
                             100.0 * static_cast<double>(en - s_ein) * 1e-9 / wall);
                s_perfLastEeIdlePct = 100.0 * static_cast<double>(en - s_ein) * 1e-9 / wall;
                s_lw = w; s_lwn = wn; s_ein = en;
            }
            {
                // Stall detector: past the boot menus, the EE was idle (every guest
                // thread blocked, no guest code running) for two 5 s windows. A
                // round-over screen keeps running guest code without counted
                // frames, so frame count alone is not a stall. Dump every guest thread,
                // its wait object, caller RA and code addresses on its stack,
                // plus all semaphores / event flags, once.
                static uint32_t s_stallWindows = 0u;
                static bool s_stallDumped = false;
                if (gameFrames > 1500u && gameFrameDelta == 0u && s_perfLastEeIdlePct >= 95.0)
                    ++s_stallWindows;
                else
                    s_stallWindows = 0u;
                if (s_stallWindows >= 2u && !s_stallDumped)
                {
                    s_stallDumped = true;
                    const EeKernelSnapshot snap = rt->eeScheduler().snapshot();
                    std::fprintf(stderr, "[stall] no game frames for 10 s at frame %llu: snapshot seq=%llu running=%d threads=%zu\n",
                                 static_cast<unsigned long long>(gameFrames), static_cast<unsigned long long>(snap.sequence),
                                 snap.runningThreadId, snap.threads.size());
                    static const char *kStatus[] = {"Running", "Ready", "Waiting", "WaitingSuspended", "Suspended", "Dormant"};
                    static const char *kReason[] = {"None", "Sleep", "Semaphore", "EventFlag", "VSync", "External", "Mpeg"};
                    uint8_t *ram = rt->memory().getRDRAM();
                    for (const auto &t : snap.threads)
                    {
                        const unsigned st = static_cast<unsigned>(t.status);
                        const unsigned wr = static_cast<unsigned>(t.waitReason);
                        std::fprintf(stderr, "[stall] thread %d %s wait=%s id=%d bits=%x mode=%x pc=0x%08x ra=0x%08x sp=0x%08x entry=0x%08x prio=%d/%d suspend=%d wakeup=%u invocations=%u\n",
                                     t.id, st < 6u ? kStatus[st] : "?", wr < 7u ? kReason[wr] : "?", t.waitId, t.waitBits, t.waitMode,
                                     t.pc, t.ra, t.sp, t.entry, t.currentPriority, t.initialPriority, t.suspendCount, t.wakeupCount,
                                     t.invocationDepth);
                        const uint32_t sp = t.sp & 0x1FFFFFFFu;
                        if (ram && sp >= 0x1000u && sp + 512u < PS2_RAM_SIZE)
                        {
                            char line[1024];
                            int len = std::snprintf(line, sizeof(line), "[stall]   stack code addrs:");
                            for (uint32_t off = 0; off < 512u && len < 900; off += 4u)
                            {
                                uint32_t word = 0;
                                std::memcpy(&word, ram + sp + off, 4u);
                                if (word >= 0x00100000u && word < 0x00300000u)
                                    len += std::snprintf(line + len, sizeof(line) - static_cast<size_t>(len), " +%x:%08x", off, word);
                            }
                            std::fprintf(stderr, "%s\n", line);
                        }
                    }
                    for (const auto &sem : snap.semaphores)
                        std::fprintf(stderr, "[stall] sema %d count=%d max=%d waiters=%u\n", sem.id, sem.count, sem.maxCount, sem.waiters);
                    for (const auto &ef : snap.eventFlags)
                        std::fprintf(stderr, "[stall] eventflag %d bits=%x init=%x attr=%x waiters=%u\n", ef.id, ef.bits, ef.initBits, ef.attr, ef.waiters);
                    std::fprintf(stderr, "[stall] INTC_STAT=%08x INTC_MASK=%08x D_STAT=%08x D_PCR=%08x\n",
                                 rt->memory().read32(0x1000F000u), rt->memory().read32(0x1000F010u),
                                 rt->memory().read32(0x1000E010u), rt->memory().read32(0x1000E020u));
                }
            }
            if (g_ps2xWotmFrameMetricsEnabled.load(std::memory_order_relaxed))
                std::fprintf(stderr, "[perf:frames] game %.2f/s | vsync %.2f/s | completed=%llu delta=%llu\n",
                             gameFrameDelta / wall, (tick - s_perfTick0) / wall,
                             static_cast<unsigned long long>(gameFrames),
                             static_cast<unsigned long long>(gameFrameDelta));
            auto take = [](std::atomic<uint64_t> &cur, std::atomic<uint64_t> &last) {
                const uint64_t c = cur.load(std::memory_order_relaxed);
                const uint64_t d = c - last.load(std::memory_order_relaxed);
                last.store(c, std::memory_order_relaxed);
                return d;
            };
            const double raster = take(g_ps2xPerf.rasterNs, s_perfLast.rasterNs) * 1e-9;
            const double present = take(g_ps2xPerf.presentNs, s_perfLast.presentNs) * 1e-9;
            const double vu1 = take(g_ps2xPerf.vu1Ns, s_perfLast.vu1Ns) * 1e-9;
            const double vif1 = take(g_ps2xPerf.vif1Ns, s_perfLast.vif1Ns) * 1e-9;
            const uint64_t prims = take(g_ps2xPerf.prims, s_perfLast.prims);
            static std::atomic<uint64_t> s_perfLastVuInstr{0u};
            static std::atomic<uint64_t> s_perfLastVuRuns{0u};
            extern std::atomic<uint64_t> g_ps2xVuInstr;
            extern std::atomic<uint64_t> g_ps2xVuRuns;
            const double vuInstr = static_cast<double>(take(g_ps2xVuInstr, s_perfLastVuInstr));
            const double vuRuns = static_cast<double>(take(g_ps2xVuRuns, s_perfLastVuRuns));
            extern std::atomic<uint64_t> g_ps2xMpegNs;
            static std::atomic<uint64_t> s_perfLastMpegNs{0u};
            const double mpeg = take(g_ps2xMpegNs, s_perfLastMpegNs) * 1e-9;
            // GS FINISH writes and the game-thread time spent waiting in their
            // raster Sync (gs_frontend.cpp GS_REG_FINISH).
            extern std::atomic<uint64_t> g_ps2xFinishCount;
            extern std::atomic<uint64_t> g_ps2xFinishWaitNs;
            static std::atomic<uint64_t> s_perfLastFinishCount{0u};
            static std::atomic<uint64_t> s_perfLastFinishWaitNs{0u};
            const double finishes = static_cast<double>(take(g_ps2xFinishCount, s_perfLastFinishCount));
            const double finishWait = take(g_ps2xFinishWaitNs, s_perfLastFinishWaitNs) * 1e-9;
            // GIF breakdown (gs_frontend.cpp / gs_cpu_backend.cpp): all game-thread
            // time inside GS::processGIFPacket, the part spent acquiring the GS
            // state lock, per-primitive build+Submit, and raster-queue-full waits.
            extern std::atomic<uint64_t> g_ps2xGifNs;
            extern std::atomic<uint64_t> g_ps2xGifLockNs;
            extern std::atomic<uint64_t> g_ps2xSubmitNs;
            extern std::atomic<uint64_t> g_ps2xRasterFullNs;
            static std::atomic<uint64_t> s_perfLastGifNs{0u};
            static std::atomic<uint64_t> s_perfLastGifLockNs{0u};
            static std::atomic<uint64_t> s_perfLastSubmitNs{0u};
            static std::atomic<uint64_t> s_perfLastRasterFullNs{0u};
            const double gifT = take(g_ps2xGifNs, s_perfLastGifNs) * 1e-9;
            const double gifLock = take(g_ps2xGifLockNs, s_perfLastGifLockNs) * 1e-9;
            const double submitT = take(g_ps2xSubmitNs, s_perfLastSubmitNs) * 1e-9;
            const double rasterFull = take(g_ps2xRasterFullNs, s_perfLastRasterFullNs) * 1e-9;
            // Guest thread switches unwind through a thrown C++ exception;
            // report the rate so its cost is measured, not assumed.
            static uint64_t s_perfLastThrows = 0u;
            const uint64_t throwsNow = g_ps2xEeTransferThrows.load(std::memory_order_relaxed);
            const uint64_t throwsDelta = throwsNow - s_perfLastThrows;
            s_perfLastThrows = throwsNow;
            {
                static uint64_t s_lastSync = 0u, s_lastVuRunsForSync = 0u;
                const uint64_t sc = g_ps2xSyncPathCalls.load(std::memory_order_relaxed);
                std::fprintf(stderr, "[perf] frames: sceGsSyncPath %.1f/s | vuInstr per sync %.0f\n",
                             static_cast<double>(sc - s_lastSync) / wall,
                             (sc - s_lastSync) ? vuInstr / static_cast<double>(sc - s_lastSync) : 0.0);
                s_lastSync = sc;
                (void)s_lastVuRunsForSync;
            }
            std::fprintf(stderr,
                         "[perf] %.1fs: vsync %.1f/s | EE-thread: raster %.0f%% vu1 %.0f%% vif1 %.0f%% | present %.0f%% (own thread) | prims/s %.0f | vuInstr/s %.0f vuRuns/s %.0f | mpeg %.1f%% | finish %.0f/s wait %.1f%% | gif %.1f%% (lockwait %.1f%% submit %.1f%% qfull %.1f%%) | threadThrows %.0f/s\n",
                         wall, (tick - s_perfTick0) / wall, raster / wall * 100.0, vu1 / wall * 100.0,
                         vif1 / wall * 100.0, present / wall * 100.0, prims / wall,
                         vuInstr / wall, vuRuns / wall, mpeg / wall * 100.0,
                         finishes / wall, finishWait / wall * 100.0,
                         gifT / wall * 100.0, gifLock / wall * 100.0, submitT / wall * 100.0,
                         rasterFull / wall * 100.0, throwsDelta / wall);
            {
                char sites[128];
                int sl = 0;
                for (int i = 0; i < 5 && sl < 100; ++i)
                    sl += std::snprintf(sites + sl, sizeof(sites) - static_cast<size_t>(sl), " s%d=%llu", i,
                                        static_cast<unsigned long long>(g_ps2xEeTransferThrowsBySite[i].load(std::memory_order_relaxed)));
                std::fprintf(stderr, "[perf] throw sites:%s skipped=%llu blockers=%llu/%llu/%llu/%llu unwindTsc=%llu rotate=%llu/%llu/skip%llu\n", sites,
                             static_cast<unsigned long long>(g_ps2xEeTransferSkips.load(std::memory_order_relaxed)),
                             static_cast<unsigned long long>(g_ps2xEeTransferSkipBlockers[0].load(std::memory_order_relaxed)),
                             static_cast<unsigned long long>(g_ps2xEeTransferSkipBlockers[1].load(std::memory_order_relaxed)),
                             static_cast<unsigned long long>(g_ps2xEeTransferSkipBlockers[2].load(std::memory_order_relaxed)),
                             static_cast<unsigned long long>(g_ps2xEeTransferSkipBlockers[3].load(std::memory_order_relaxed)),
                             static_cast<unsigned long long>(g_ps2xThrowUnwindTsc.load(std::memory_order_relaxed)),
                             static_cast<unsigned long long>(g_ps2xRotateBranch[0].load(std::memory_order_relaxed)),
                             static_cast<unsigned long long>(g_ps2xRotateBranch[1].load(std::memory_order_relaxed)),
                             static_cast<unsigned long long>(g_ps2xRotateBranch[2].load(std::memory_order_relaxed)));
            }
            {
                static uint64_t s_lastNative = 0u, s_lastInterp = 0u;
                const uint64_t nat = g_ps2xVu1NativeRuns.load(std::memory_order_relaxed);
                const uint64_t itp = g_ps2xVu1InterpRuns.load(std::memory_order_relaxed);
                {
                    extern std::atomic<bool> g_ps2xHleOn;
                    static const bool s_hleAlternate = [] { const char *p = std::getenv("PS2X_VU1_HLE_ALTERNATE"); return p && p[0] == '1'; }();
                    std::fprintf(stderr, "[perf] hle=%s\n", g_ps2xHleOn.load() ? "on" : "off");
                    void ps2xVu1EntryReport(double);
                    ps2xVu1EntryReport(wall);
                    if (s_hleAlternate)
                        g_ps2xHleOn.store(!g_ps2xHleOn.load());
                }
                std::fprintf(stderr, "[perf] vu1 runs: native %llu/s interp %llu/s fallbacks %llu\n",
                             static_cast<unsigned long long>((nat - s_lastNative) / wall),
                             static_cast<unsigned long long>((itp - s_lastInterp) / wall),
                             static_cast<unsigned long long>(g_ps2xVu1NativeFallbacks.load(std::memory_order_relaxed)));
                {
                    extern std::atomic<uint64_t> g_ps2xVu0Calls, g_ps2xVu0Ns, g_ps2xVu0SetupNs, g_ps2xVu0ExecNs;
                    static uint64_t s_lastVu0Calls = 0u, s_lastVu0Ns = 0u, s_lastSetup = 0u, s_lastExec = 0u;
                    const uint64_t vc = g_ps2xVu0Calls.load(std::memory_order_relaxed);
                    const uint64_t vn = g_ps2xVu0Ns.load(std::memory_order_relaxed);
                    const uint64_t su = g_ps2xVu0SetupNs.load(std::memory_order_relaxed);
                    const uint64_t ex = g_ps2xVu0ExecNs.load(std::memory_order_relaxed);
                    extern std::atomic<uint64_t> g_ps2xVu0Instr;
                    static uint64_t s_lastCyc = 0u;
                    const uint64_t cy = g_ps2xVu0Instr.load(std::memory_order_relaxed);
                    const double dc = static_cast<double>(vc - s_lastVu0Calls);
                    std::fprintf(stderr, "[perf] vu0 micro: %.0f calls/s | %.1f%% of wall | %.0f ns/call | setup %.0f ns | execute %.0f ns | %.1f cycles/call\n",
                                 dc / wall, 100.0 * static_cast<double>(vn - s_lastVu0Ns) * 1e-9 / wall,
                                 dc > 0 ? static_cast<double>(vn - s_lastVu0Ns) / dc : 0.0,
                                 dc > 0 ? static_cast<double>(su - s_lastSetup) / dc : 0.0,
                                 dc > 0 ? static_cast<double>(ex - s_lastExec) / dc : 0.0,
                                 dc > 0 ? static_cast<double>(cy - s_lastCyc) / dc : 0.0);
                    s_lastCyc = cy;
                    s_lastVu0Calls = vc; s_lastVu0Ns = vn; s_lastSetup = su; s_lastExec = ex;
                }
                s_lastNative = nat;
                s_lastInterp = itp;
            }
            {
                static uint64_t pJobs = 0u, pDrains = 0u, pWait = 0u, pHaz = 0u;
                const uint64_t jobs = g_ps2xVu1JobsSubmitted.load(std::memory_order_relaxed);
                const uint64_t drains = g_ps2xVu1Drains.load(std::memory_order_relaxed);
                const uint64_t waitNs = g_ps2xVu1DrainWaitNs.load(std::memory_order_relaxed);
                const uint64_t haz = g_ps2xGsForeignHazards.load(std::memory_order_relaxed);
                if (jobs != 0u)
                    std::fprintf(stderr, "[perf] vu1thread: jobs %.0f/s | drains %.0f/s | EE waiting on worker %.1f%% | GS ordering hazards %llu\n",
                                 (jobs - pJobs) / wall, (drains - pDrains) / wall,
                                 static_cast<double>(waitNs - pWait) * 1e-9 / wall * 100.0,
                                 static_cast<unsigned long long>(haz - pHaz));
                pJobs = jobs; pDrains = drains; pWait = waitNs; pHaz = haz;
            }
            {
                // Parallel-VU1 probe: of the gaps between consecutive VIF1
                // transfers, how many contain a blocking wait or DMA polling?
                static uint64_t pT = 0u, pTsc = 0u, pReads = 0u, pPoll = 0u, pBlock = 0u, pWait = 0u, pSig = 0u;
                const uint64_t wc = g_ps2xVif1GapsWithWaitCall.load(std::memory_order_relaxed);
                const uint64_t sg = g_ps2xVif1GapsWithSignal.load(std::memory_order_relaxed);
                const uint64_t t = g_ps2xVif1Transfers.load(std::memory_order_relaxed);
                const uint64_t tsc = g_ps2xVif1InsideTsc.load(std::memory_order_relaxed);
                const uint64_t rd = g_ps2xVif1GapDmaReads.load(std::memory_order_relaxed);
                const uint64_t pl = g_ps2xVif1GapsWithPoll.load(std::memory_order_relaxed);
                const uint64_t bl = g_ps2xVif1GapsWithBlock.load(std::memory_order_relaxed);
                const uint64_t dT = t - pT;
                const double gaps = dT > 0u ? static_cast<double>(dT) : 1.0;
                std::fprintf(stderr, "[perf] vif1 gap: transfers %.0f/s (%.1f/frame) | gaps with a blocking wait %.1f%% | gaps with DMA polling %.1f%% (%.1f reads/gap) | gaps with a wait CALL %.1f%% | gaps with a signal %.1f%% | insideTsc/s %.0f\n",
                             dT / wall, dT / wall / std::max(1.0, (tick - s_perfTick0) / wall),
                             100.0 * static_cast<double>(bl - pBlock) / gaps,
                             100.0 * static_cast<double>(pl - pPoll) / gaps,
                             static_cast<double>(rd - pReads) / gaps,
                             100.0 * static_cast<double>(wc - pWait) / gaps,
                             100.0 * static_cast<double>(sg - pSig) / gaps,
                             static_cast<double>(tsc - pTsc) / wall);
                pT = t; pTsc = tsc; pReads = rd; pPoll = pl; pBlock = bl; pWait = wc; pSig = sg;
            }
            s_perfT0 = now;
            s_perfTick0 = tick;
        }
    }
    // WotM shell screen changes (currScreen, gp-0x748c), printed in every build
    // so play-build test scripts can navigate by screen state instead of by
    // timer (timed key presses land mid-transition whenever game speed changes).
    // 0/16 intro, 20 press start, 1 main menu (and in-level), 2 modes, 3 char
    // select, 99 level load.
    {
        static uint32_t s_lastScr = 0xFFFFFFFFu;
        const uint32_t scr = rt->memory().read32(0x6F8464u);
        if (scr != s_lastScr)
        {
            s_lastScr = scr;
            std::fprintf(stderr, "[scr] tick=%llu scr=%d\n",
                         static_cast<unsigned long long>(rt->eeScheduler().currentVSyncTick()),
                         static_cast<int>(scr));
        }
    }
}

static void UploadFrame(Texture2D &tex, PS2Runtime *rt, uint32_t &outWidth, uint32_t &outHeight)
{
    static uint32_t s_lastDisplayFbp = std::numeric_limits<uint32_t>::max();
    static uint32_t s_lastSourceFbp = std::numeric_limits<uint32_t>::max();
    static bool s_lastPreferred = false;
    static uint32_t s_lastWidth = 0u;
    static uint32_t s_lastHeight = 0u;
    static bool s_hasUploadedFrame = false;
    static std::vector<uint8_t> s_scratch;
    static std::vector<uint8_t> s_uploadBuffer(DEFAULT_FB_SIZE, 0u);

    // Upload once per guest vsync. This has to track its OWN tick: the main
    // loop calls LatchFrameIfNeeded before deciding which present path to take,
    // so asking the latch again here answers "already done for this tick" and
    // would skip every upload after the first -- a frozen picture.
    static uint64_t s_lastUploadedTick = std::numeric_limits<uint64_t>::max();
    const uint64_t currentTick = rt->eeScheduler().currentVSyncTick();
    LatchFrameIfNeeded(rt);
    if (s_hasUploadedFrame && currentTick == s_lastUploadedTick)
    {
        outWidth = (s_lastWidth != 0u) ? s_lastWidth : FB_WIDTH;
        outHeight = (s_lastHeight != 0u) ? s_lastHeight : DEFAULT_DISPLAY_HEIGHT;
        return;
    }
    s_lastUploadedTick = currentTick;

    s_scratch.clear();
    uint32_t width = 0u;
    uint32_t height = 0u;
    uint32_t displayFbp = 0u;
    uint32_t sourceFbp = 0u;
    bool usedPreferredDisplaySource = false;
    if (!rt->gs().copyLatchedHostPresentationFrame(s_scratch,
                                                   width,
                                                   height,
                                                   &displayFbp,
                                                   &sourceFbp,
                                                   &usedPreferredDisplaySource))
    {
        Image blank = GenImageColor(FB_WIDTH, FB_HEIGHT, MAGENTA);
        UpdateTexture(tex, blank.data);
        UnloadImage(blank);
        outWidth = FB_WIDTH;
        outHeight = DEFAULT_DISPLAY_HEIGHT;
        s_lastWidth = outWidth;
        s_lastHeight = outHeight;
        s_hasUploadedFrame = true;
        return;
    }

    PS2_IF_AGRESSIVE_LOGS({
        static uint32_t s_uploadDebugCount = 0u;
        if (s_uploadDebugCount < 128u ||
            displayFbp != s_lastDisplayFbp ||
            sourceFbp != s_lastSourceFbp ||
            usedPreferredDisplaySource != s_lastPreferred ||
            width != s_lastWidth ||
            height != s_lastHeight)
        {
            std::cout << "[frame:upload] idx=" << s_uploadDebugCount
                      << " tick=" << currentTick
                      << " displayFbp=" << displayFbp
                      << " sourceFbp=" << sourceFbp
                      << " size=" << width << "x" << height
                      << " preferred=" << static_cast<uint32_t>(usedPreferredDisplaySource ? 1u : 0u)
                      << std::endl;
        }
        ++s_uploadDebugCount;
    });
    s_lastDisplayFbp = displayFbp;
    s_lastSourceFbp = sourceFbp;
    s_lastPreferred = usedPreferredDisplaySource;
    s_lastWidth = width;
    s_lastHeight = height;

    std::fill(s_uploadBuffer.begin(), s_uploadBuffer.end(), 0u);
    if (!s_scratch.empty() && width != 0u && height != 0u)
    {
        const uint32_t copyWidth = std::min<uint32_t>(width, FB_WIDTH);
        const uint32_t copyHeight = std::min<uint32_t>(height, FB_HEIGHT);
        const size_t srcRowBytes = static_cast<size_t>(width) * 4u;
        const size_t dstRowBytes = static_cast<size_t>(FB_WIDTH) * 4u;
        const size_t copyRowBytes = static_cast<size_t>(copyWidth) * 4u;
        for (uint32_t y = 0; y < copyHeight; ++y)
        {
            const size_t srcOffset = static_cast<size_t>(y) * srcRowBytes;
            const size_t dstOffset = static_cast<size_t>(y) * dstRowBytes;
            if (srcOffset + copyRowBytes > s_scratch.size() ||
                dstOffset + copyRowBytes > s_uploadBuffer.size())
            {
                break;
            }
            std::memcpy(s_uploadBuffer.data() + dstOffset, s_scratch.data() + srcOffset, copyRowBytes);
        }
    }

    UpdateTexture(tex, s_uploadBuffer.data());
    outWidth = width;
    outHeight = height;
    s_hasUploadedFrame = true;
}

PS2Runtime::PS2Runtime()
{
    m_iopHost = std::make_unique<PS2IopHostAdapter>(*this);
    m_iopSubsystem = std::make_unique<ps2x::iop::IopSubsystem>(*m_iopHost);
    m_eeScheduler = std::make_unique<EeScheduler>(*this);
#if defined(PS2X_IOP_ENABLE_PLUGINS) && PS2X_IOP_ENABLE_PLUGINS && \
    !defined(PLATFORM_VITA) && (defined(_WIN32) || defined(__linux__))
    if (const char *applicationDirectory = GetApplicationDirectory();
        applicationDirectory && applicationDirectory[0] != '\0')
    {
        m_iopSubsystem->setPluginSearchPaths({std::filesystem::path(applicationDirectory) / "iop_plugins"});
    }
#endif

    // Assign rather than memset: R5900Context's constructor zeroes itself and
    // then applies the COP0 reset values, which a memset here would discard.
    m_cpuContext = R5900Context{};

    // R0 is always zero in MIPS
    m_cpuContext.r[0] = _mm_set1_epi32(0);
    m_cpuContext.vu0_vf[0] = _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
    m_cpuContext.vu0_q = 1.0f;
    m_cpuContext.vu0_r = _mm_castsi128_ps(_mm_set1_epi32(0x3F800000));

    // Stack pointer (SP) and global pointer (GP) will be set by the loaded ELF

    m_loadedModules.clear();
    m_guestHeapBlocks.clear();
    m_guestHeapBase = kGuestHeapDefaultBase;
    m_guestHeapEnd = kGuestHeapDefaultBase;
    m_guestHeapLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    m_guestHeapSuggestedBase = kGuestHeapDefaultBase;
    m_guestHeapConfigured = false;
    m_asyncCallbackStackFloor = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    m_asyncCallbackStackTop = PS2_RAM_SIZE;
}

void PS2Runtime::setDebugUiCallbacks(DebugUiCallback initCallback,
                                     DebugUiCallback drawCallback,
                                     DebugUiCallback shutdownCallback,
                                     void *userData)
{
    if (m_debugUiInitialized && m_debugUiShutdownCallback)
    {
        m_debugUiShutdownCallback(*this, m_debugUiUserData);
        m_debugUiInitialized = false;
    }

    m_debugUiInitCallback = initCallback;
    m_debugUiDrawCallback = drawCallback;
    m_debugUiShutdownCallback = shutdownCallback;
    m_debugUiUserData = userData;
}

PS2Runtime::~PS2Runtime()
{
    try
    {
        requestStop();
        m_iopSubsystem.reset();
        m_iopHost.reset();
#if defined(PLATFORM_VITA)
        m_audioBackend.stopAll();
        m_audioBackend.setAudioReady(false);
#else
        if (IsAudioDeviceReady())
        {
            CloseAudioDevice();
            m_audioBackend.setAudioReady(false);
        }
#endif
        if (m_debugUiInitialized && m_debugUiShutdownCallback)
        {
            m_debugUiShutdownCallback(*this, m_debugUiUserData);
            m_debugUiInitialized = false;
        }

        if (IsWindowReady())
        {
            CloseWindow();
        }

        m_loadedModules.clear();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[~PS2Runtime] cleanup exception: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "[~PS2Runtime] cleanup exception: unknown" << std::endl;
    }
}

void PS2Runtime::setIopPluginSearchPaths(std::vector<std::filesystem::path> paths)
{
    m_iopSubsystem->setPluginSearchPaths(std::move(paths));
}

ps2x::iop::RpcAbi PS2Runtime::selectIopRpcAbi(const ps2x::iop::RpcAbiRequest &request) const
{
    return m_iopSubsystem->selectRpcAbi(request);
}

ps2x::iop::RpcResult PS2Runtime::handleIopRpc(uint8_t *rdram, R5900Context *ctx, ps2x::iop::RpcRequest request)
{
    auto scope = m_iopHost->enterCall(ctx, rdram);
    request.callToken = scope.token();
    return m_iopSubsystem->handleRpc(request);
}

void PS2Runtime::notifyIopSifTransfer(uint8_t *rdram, const ps2x::iop::SifTransfer &transfer)
{
    auto scope = m_iopHost->enterCall(nullptr, rdram);
    m_iopSubsystem->onSifTransfer(transfer);
}

void PS2Runtime::resetIop()
{
    m_iopSubsystem->reset();
}

ps2x::iop::DebugSnapshot PS2Runtime::iopDebugSnapshot() const
{
    return m_iopSubsystem->debugSnapshot();
}

bool PS2Runtime::syncCoreSubsystems()
{
    uint8_t *const rdram = m_memory.getRDRAM();
    uint8_t *const gsVram = m_memory.getGSVRAM();
    if (!rdram || !gsVram)
    {
        return false;
    }

    if (m_boundRdram == rdram && m_boundGSVram == gsVram)
    {
        return true;
    }

    m_gs.init(gsVram, static_cast<uint32_t>(PS2_GS_VRAM_SIZE), &m_memory.gs());
    m_gifArbiter.setProcessPacketFn([this](const uint8_t *data, uint32_t size)
                                    { m_gs.processGIFPacket(data, size); });
    m_memory.setGifArbiter(&m_gifArbiter);
    m_memory.setVu1FbrstSampler([this]() -> uint32_t
                                {
                                    R5900Context *c = m_eeScheduler ? m_eeScheduler->currentContext() : nullptr;
                                    return (c ? c : &m_cpuContext)->vu0_fbrst;
                                });
    m_memory.setVu1MscalCallback([this](uint32_t startPC, uint32_t top, uint32_t itop)
                                 {
                                     // On the VU1 worker the EE context is off limits (another
                                     // thread owns it): use the D/T bits sampled at submit time.
                                     const bool onVu1Worker = PS2Memory::isVu1WorkerThread();
                                     R5900Context *cpuContext = nullptr;
                                     if (onVu1Worker)
                                     {
                                         m_vu1.state().dBitEnabled = m_memory.vu1JobDBit();
                                         m_vu1.state().tBitEnabled = m_memory.vu1JobTBit();
                                     }
                                     else
                                     {
                                         cpuContext = m_eeScheduler ? m_eeScheduler->currentContext() : nullptr;
                                         if (!cpuContext)
                                         {
                                             cpuContext = &m_cpuContext;
                                         }
                                         m_vu1.state().dBitEnabled =
                                             (cpuContext->vu0_fbrst & (1u << 10)) != 0u;
                                         m_vu1.state().tBitEnabled =
                                             (cpuContext->vu0_fbrst & (1u << 11)) != 0u;
                                     }
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
                                     // What is in VU1 DMEM q0/q3 at the moment MSCAL fires?
                                     // Compare to [vif1:q0rb] (post-unpack). If it differs,
                                     // something clobbers q0-q3 between unpack and MSCAL.
                                     if (m_memory.read32(0x6F7E8Cu) == 0u && m_memory.read32(0x6F8464u) == 1u)
                                     {
                                         static std::atomic<uint32_t> s_ms{0u};
                                         if (s_ms.fetch_add(1u, std::memory_order_relaxed) < 200u)
                                         {
                                             const float *d = reinterpret_cast<const float *>(m_memory.getVU1Data());
                                             std::fprintf(stderr,
                                                 "[vu1:mscalq] startPC=0x%x top=%u q0=(%g,%g,%g,%g) q3=(%g,%g,%g,%g)\n",
                                                 startPC, top, d[0], d[1], d[2], d[3], d[12], d[13], d[14], d[15]);
                                         }
                                     }
#endif
                                     m_vu1.execute(m_memory.getVU1Code(), PS2_VU1_CODE_SIZE,
                                                   m_memory.getVU1Data(), PS2_VU1_DATA_SIZE,
                                                   m_gs, &m_memory, startPC, top, itop, 16u * 1024u * 1024u);
                                     if (cpuContext)
                                         cpuContext->vu0_vpu_stat =
                                             (cpuContext->vu0_vpu_stat & ~0x0600u) |
                                             (m_vu1.state().stoppedByD ? 0x0200u : 0u) |
                                             (m_vu1.state().stoppedByT ? 0x0400u : 0u); });
    m_memory.setVu1MscntCallback([this](uint32_t top, uint32_t itop)
                                 {
                                     // On the VU1 worker the EE context is off limits (another
                                     // thread owns it): use the D/T bits sampled at submit time.
                                     const bool onVu1Worker = PS2Memory::isVu1WorkerThread();
                                     R5900Context *cpuContext = nullptr;
                                     if (onVu1Worker)
                                     {
                                         m_vu1.state().dBitEnabled = m_memory.vu1JobDBit();
                                         m_vu1.state().tBitEnabled = m_memory.vu1JobTBit();
                                     }
                                     else
                                     {
                                         cpuContext = m_eeScheduler ? m_eeScheduler->currentContext() : nullptr;
                                         if (!cpuContext)
                                         {
                                             cpuContext = &m_cpuContext;
                                         }
                                         m_vu1.state().dBitEnabled =
                                             (cpuContext->vu0_fbrst & (1u << 10)) != 0u;
                                         m_vu1.state().tBitEnabled =
                                             (cpuContext->vu0_fbrst & (1u << 11)) != 0u;
                                     }
                                     m_vu1.resume(m_memory.getVU1Code(), PS2_VU1_CODE_SIZE,
                                                  m_memory.getVU1Data(), PS2_VU1_DATA_SIZE,
                                                  m_gs, &m_memory, top, itop, 16u * 1024u * 1024u);
                                     if (cpuContext)
                                         cpuContext->vu0_vpu_stat =
                                             (cpuContext->vu0_vpu_stat & ~0x0600u) |
                                             (m_vu1.state().stoppedByD ? 0x0200u : 0u) |
                                             (m_vu1.state().stoppedByT ? 0x0400u : 0u); });
    resetIop();
    m_vu0.reset();
    m_vu1.reset();

    m_boundRdram = rdram;
    m_boundGSVram = gsVram;
    return true;
}

bool PS2Runtime::initialize(const char *title)
{
    title = "War of the Monsters - SCUS-97197";
    try
    {
        if (!m_memory.initialize())
        {
            std::cerr << "Failed to initialize PS2 memory" << std::endl;
            return false;
        }

        if (!syncCoreSubsystems())
        {
            std::cerr << "Failed to bind runtime core subsystems" << std::endl;
            return false;
        }
#if defined(PS2X_IOP_ENABLE_PLUGINS) && PS2X_IOP_ENABLE_PLUGINS && \
    !defined(PLATFORM_VITA) && (defined(_WIN32) || defined(__linux__))
        std::string pluginError;
        if (!m_iopSubsystem->loadPlugins(&pluginError))
        {
            std::cerr << "Failed to load IOP plugins: " << pluginError << std::endl;
            return false;
        }
#endif
#if defined(PLATFORM_VITA)
        InitWindow(HOST_WINDOW_WIDTH, HOST_WINDOW_HEIGHT, title); // raylib vita does not support audio
#else
        SetConfigFlags(FLAG_WINDOW_RESIZABLE);
        // PS2X_WINDOW=<w>x<h> sizes the window; otherwise PS2X_GS_SCALE (the GPU
        // renderer's internal resolution multiplier) also scales it, so asking
        // for a higher render resolution gives a window worth seeing it in.
        int windowWidth = HOST_WINDOW_WIDTH;
        int windowHeight = HOST_WINDOW_HEIGHT;
        if (const char *spec = std::getenv("PS2X_WINDOW"))
        {
            int w = 0, h = 0;
            if (std::sscanf(spec, "%dx%d", &w, &h) == 2 && w >= 256 && h >= 192)
            {
                windowWidth = w;
                windowHeight = h;
            }
        }
        else if (const char *scaleSpec = std::getenv("PS2X_GS_SCALE"))
        {
            const long scale = std::strtol(scaleSpec, nullptr, 10);
            if (scale > 1 && scale <= 8)
            {
                windowWidth = static_cast<int>(HOST_WINDOW_WIDTH * scale);
                windowHeight = static_cast<int>(HOST_WINDOW_HEIGHT * scale);
            }
        }
        InitWindow(windowWidth, windowHeight, title);
        // Shrink to fit the monitor if that asked for more than the screen has.
        {
            const int monitor = GetCurrentMonitor();
            const int maxWidth = GetMonitorWidth(monitor);
            const int maxHeight = GetMonitorHeight(monitor) - 64; // leave room for the title bar
            if (maxWidth > 0 && maxHeight > 0 && (windowWidth > maxWidth || windowHeight > maxHeight))
            {
                const float fit = std::min(static_cast<float>(maxWidth) / static_cast<float>(windowWidth),
                                           static_cast<float>(maxHeight) / static_cast<float>(windowHeight));
                SetWindowSize(static_cast<int>(windowWidth * fit), static_cast<int>(windowHeight * fit));
            }
        }
        const char* borderless = std::getenv("PS2X_BORDERLESS");
        if (borderless && borderless[0] == '1')
            ToggleBorderlessWindowed();
        else if (const char *full = std::getenv("PS2X_FULLSCREEN"))
            if (full[0] == '1')
            {
                SetWindowSize(windowWidth, windowHeight);
                ToggleFullscreen();
            }
        InitAudioDevice();
        // Automated visual tests can exercise the full audio path silently.
        // This is per process and does not change saved game/launcher volume.
        if (const char *mute = std::getenv("PS2X_MUTE"))
            if (mute[0] == '1' && IsAudioDeviceReady())
                SetMasterVolume(0.0f);
        m_audioBackend.setAudioReady(IsAudioDeviceReady());
#endif
        // raylib closes the window on ESC by default, so ESC quit the game
        // outright -- and ESC is also mapped to Circle (back), so one press did
        // both. Close via the window button / Alt+F4 instead.
        SetExitKey(KEY_NULL);
        // Presentation cadence only; EE scheduler/simulation timing is unchanged.
        int presentLimit = 60;
        if (const char* spec = std::getenv("PS2X_PRESENT_FPS"))
        {
            char* end = nullptr;
            const long value = std::strtol(spec, &end, 10);
            if (end != spec && *end == '\0' && (value == 0 || (value >= 30 && value <= 360)))
                presentLimit = static_cast<int>(value);
        }
        if (const char* vsync = std::getenv("PS2X_DISPLAY_VSYNC"))
            if (vsync[0] == '1') SetWindowState(FLAG_VSYNC_HINT);
        SetTargetFPS(presentLimit);
        std::fprintf(stderr, "[display] presentation cap=%d (0=unlimited); simulation clock unchanged\n", presentLimit);
        if (m_debugUiInitCallback)
        {
            m_debugUiInitCallback(*this, m_debugUiUserData);
            m_debugUiInitialized = true;
        }

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to initialize PS2 runtime: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Failed to initialize PS2 runtime: unknown exception" << std::endl;
    }

    return false;
}

bool PS2Runtime::loadELF(const std::string &elfPath)
{
    std::cerr << "[TRACE] loadELF called with path: " << elfPath << std::endl;
    configureIoPathsFromElf(elfPath);

    std::ifstream file(elfPath, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open ELF file: " << elfPath << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    const std::streamoff fileSize = file.tellg();
    if (fileSize < static_cast<std::streamoff>(sizeof(ElfHeader)))
    {
        std::cerr << "ELF file is too small: " << elfPath << std::endl;
        return false;
    }
    file.seekg(0, std::ios::beg);

    ElfHeader header{};
    if (!file.read(reinterpret_cast<char *>(&header), sizeof(header)))
    {
        std::cerr << "Failed to read ELF header from: " << elfPath << std::endl;
        return false;
    }

    if (header.magic != ELF_MAGIC)
    {
        std::cerr << "Invalid ELF magic number" << std::endl;
        return false;
    }

    if (header.elf_class != 1u || header.endianness != 1u)
    {
        std::cerr << "Unsupported ELF format (expected 32-bit little-endian)." << std::endl;
        return false;
    }

    if (header.machine != EM_MIPS || header.type != ET_EXEC)
    {
        std::cerr << "Not a MIPS executable ELF file" << std::endl;
        return false;
    }

    if (header.phnum != 0u && header.phentsize < sizeof(ProgramHeader))
    {
        std::cerr << "Unsupported ELF program-header entry size: " << header.phentsize << std::endl;
        return false;
    }

    const uint64_t programHeaderTableEnd =
        static_cast<uint64_t>(header.phoff) +
        static_cast<uint64_t>(header.phnum) * static_cast<uint64_t>(header.phentsize);
    if (programHeaderTableEnd > static_cast<uint64_t>(fileSize))
    {
        std::cerr << "ELF program-header table is out of range." << std::endl;
        return false;
    }

    m_cpuContext.pc = header.entry;
    std::cerr << "[TRACE] header.entry = 0x" << std::hex << header.entry << std::dec << std::endl;
    m_debugPc.store(m_cpuContext.pc, std::memory_order_relaxed);

    uint32_t maxLoadedRdramEnd = kGuestHeapDefaultBase;
    uint32_t moduleBase = std::numeric_limits<uint32_t>::max();
    uint32_t moduleEnd = 0u;
    bool loadedAnySegment = false;

    for (uint16_t i = 0; i < header.phnum; i++)
    {
        const uint64_t phOffset =
            static_cast<uint64_t>(header.phoff) +
            static_cast<uint64_t>(i) * static_cast<uint64_t>(header.phentsize);
        if (phOffset + sizeof(ProgramHeader) > static_cast<uint64_t>(fileSize))
        {
            std::cerr << "ELF program header " << i << " is out of range." << std::endl;
            return false;
        }

        ProgramHeader ph{};
        file.seekg(static_cast<std::streamoff>(phOffset), std::ios::beg);
        if (!file.read(reinterpret_cast<char *>(&ph), sizeof(ph)))
        {
            std::cerr << "Failed to read ELF program header " << i << std::endl;
            return false;
        }

        if (ph.type != PT_LOAD || ph.memsz == 0u)
        {
            continue;
        }

        if (ph.filesz > ph.memsz)
        {
            std::cerr << "ELF segment " << i << " has filesz > memsz." << std::endl;
            return false;
        }

        const uint64_t segmentFileEnd = static_cast<uint64_t>(ph.offset) + static_cast<uint64_t>(ph.filesz);
        if (segmentFileEnd > static_cast<uint64_t>(fileSize))
        {
            std::cerr << "ELF segment " << i << " exceeds file bounds." << std::endl;
            return false;
        }

        const bool scratch =
            ph.vaddr >= PS2_SCRATCHPAD_BASE &&
            ph.vaddr < (PS2_SCRATCHPAD_BASE + PS2_SCRATCHPAD_SIZE);

        uint32_t physAddr = 0u;
        try
        {
            physAddr = m_memory.translateAddress(ph.vaddr);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Failed to translate ELF segment " << i
                      << " virtual address 0x" << std::hex << ph.vaddr
                      << std::dec << ": " << e.what() << std::endl;
            return false;
        }
        const uint64_t regionSize = scratch ? static_cast<uint64_t>(PS2_SCRATCHPAD_SIZE)
                                            : static_cast<uint64_t>(PS2_RAM_SIZE);
        const uint64_t segmentMemEnd = static_cast<uint64_t>(physAddr) + static_cast<uint64_t>(ph.memsz);
        if (segmentMemEnd > regionSize)
        {
            std::cerr << "ELF segment " << i << " exceeds "
                      << (scratch ? "scratchpad" : "RDRAM")
                      << " bounds (vaddr=0x" << std::hex << ph.vaddr
                      << " memsz=0x" << ph.memsz << std::dec << ")." << std::endl;
            return false;
        }

        uint8_t *destBase = scratch ? m_memory.getScratchpad() : m_memory.getRDRAM();
        if (!destBase)
        {
            std::cerr << "ELF segment " << i << " has no destination memory backing." << std::endl;
            return false;
        }

        uint8_t *dest = destBase + physAddr;
        if (ph.filesz > 0u)
        {
            file.seekg(static_cast<std::streamoff>(ph.offset), std::ios::beg);
            if (!file.read(reinterpret_cast<char *>(dest), ph.filesz))
            {
                std::cerr << "Failed to read ELF segment " << i << " payload." << std::endl;
                return false;
            }
        }

        if (ph.memsz > ph.filesz)
        {
            std::memset(dest + ph.filesz, 0, ph.memsz - ph.filesz);
        }

        RUNTIME_LOG("Loading segment: 0x" << std::hex << ph.vaddr
                                          << " - 0x" << (static_cast<uint64_t>(ph.vaddr) + static_cast<uint64_t>(ph.memsz))
                                          << " (filesz: 0x" << ph.filesz
                                          << ", memsz: 0x" << ph.memsz << ")"
                                          << std::dec << std::endl);

        if (!scratch)
        {
            maxLoadedRdramEnd = std::max(maxLoadedRdramEnd, static_cast<uint32_t>(segmentMemEnd));
        }

        if (ph.flags & 0x1u) // PF_X
        {
            const uint64_t execEnd = static_cast<uint64_t>(ph.vaddr) + static_cast<uint64_t>(ph.filesz);
            if (execEnd <= std::numeric_limits<uint32_t>::max())
            {
                m_memory.registerCodeRegion(ph.vaddr, static_cast<uint32_t>(execEnd));
            }
        }

        loadedAnySegment = true;
        moduleBase = std::min(moduleBase, ph.vaddr);
        const uint64_t segmentVirtualEnd = static_cast<uint64_t>(ph.vaddr) + static_cast<uint64_t>(ph.memsz);
        const uint32_t clampedVirtualEnd =
            (segmentVirtualEnd > std::numeric_limits<uint32_t>::max())
                ? std::numeric_limits<uint32_t>::max()
                : static_cast<uint32_t>(segmentVirtualEnd);
        moduleEnd = std::max(moduleEnd, clampedVirtualEnd);
    }

    if (!loadedAnySegment)
    {
        std::cerr << "ELF contains no loadable PT_LOAD segments." << std::endl;
        return false;
    }

    if (maxLoadedRdramEnd > PS2_RAM_SIZE)
    {
        maxLoadedRdramEnd = PS2_RAM_SIZE;
    }

    const uint32_t paddedEnd = (maxLoadedRdramEnd > (PS2_RAM_SIZE - kGuestHeapSafetyPad))
                                   ? PS2_RAM_SIZE
                                   : (maxLoadedRdramEnd + kGuestHeapSafetyPad);
    const uint32_t suggestedHeapBase = alignGuestHeapValue(paddedEnd, kGuestHeapDefaultAlignment);
    {
        std::lock_guard<std::mutex> lock(m_guestHeapMutex);
        if (!m_guestHeapConfigured)
        {
            const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
            m_guestHeapSuggestedBase = std::min(suggestedHeapBase, hardLimit);
            m_guestHeapBase = m_guestHeapSuggestedBase;
            m_guestHeapEnd = m_guestHeapSuggestedBase;
            m_guestHeapLimit = hardLimit;
        }
    }
    {
        std::lock_guard<std::mutex> lock(m_asyncCallbackStackMutex);
        const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
        m_asyncCallbackStackFloor = std::min(std::max(hardLimit, suggestedHeapBase), PS2_RAM_SIZE);
        m_asyncCallbackStackTop = PS2_RAM_SIZE;
    }

    LoadedModule module;
    module.name = elfPath.substr(elfPath.find_last_of("/\\") + 1);
    module.baseAddress = (moduleBase == std::numeric_limits<uint32_t>::max()) ? 0x00100000u : moduleBase;
    module.size = (moduleEnd > module.baseAddress) ? static_cast<size_t>(moduleEnd - module.baseAddress) : 0u;
    module.active = true;

    m_loadedModules.push_back(module);

    uint32_t elfCrc32 = 0u;
    const bool elfCrc32Valid = computeFileCrc32(elfPath, elfCrc32);
    if (!elfCrc32Valid)
    {
        std::cerr << "[ps2xIOP] failed to compute ELF CRC32 for '" << elfPath << "'" << std::endl;
    }
    ps2x::iop::GameIdentity identity;
    identity.elfName = module.name;
    identity.entryPoint = m_cpuContext.pc;
    identity.crc32 = elfCrc32;
    std::string iopError;
    if (!m_iopSubsystem->configure(identity, &iopError))
    {
        std::cerr << "[ps2xIOP] failed to configure profile: " << iopError << std::endl;
        return false;
    }

    ps2_game_overrides::applyMatching(*this,
                                      elfPath,
                                      m_cpuContext.pc,
                                      elfCrc32,
                                      elfCrc32Valid);

    RUNTIME_LOG("ELF file loaded successfully. Entry point: 0x" << std::hex << m_cpuContext.pc << std::dec);
    return true;
}

const PS2Runtime::IoPaths &PS2Runtime::getIoPaths()
{
    return runtimeIoPaths();
}

void PS2Runtime::setIoPaths(const IoPaths &paths)
{
    IoPaths normalized = paths;
    normalized.elfPath = normalizeAbsolutePath(normalized.elfPath);
    normalized.elfDirectory = normalizeAbsolutePath(normalized.elfDirectory);
    normalized.hostRoot = normalizeAbsolutePath(normalized.hostRoot);
    normalized.cdRoot = normalizeAbsolutePath(normalized.cdRoot);
    normalized.mcRoot = normalizeAbsolutePath(normalized.mcRoot);
    normalized.cdImage = normalizeAbsolutePath(normalized.cdImage);

    if (normalized.elfDirectory.empty() && !normalized.elfPath.empty())
    {
        normalized.elfDirectory = normalized.elfPath.parent_path();
    }

    if (normalized.hostRoot.empty())
    {
        normalized.hostRoot = normalized.elfDirectory;
    }
    if (normalized.cdRoot.empty())
    {
        normalized.cdRoot = normalized.elfDirectory;
    }
    if (normalized.mcRoot.empty())
    {
        normalized.mcRoot = normalized.elfDirectory / "mc0";
    }

    if (wotm_disc::mount(normalized.cdImage))
        normalized.cdRoot = normalized.cdImage.parent_path() / "files";
    runtimeIoPaths() = normalized;
}

void PS2Runtime::configureIoPathsFromElf(const std::string &elfPath)
{
    IoPaths paths = runtimeIoPaths();
    paths.elfPath = normalizeAbsolutePath(std::filesystem::path(elfPath));
    if (!paths.elfPath.empty())
    {
        paths.elfDirectory = paths.elfPath.parent_path();
    }

    if (!paths.elfDirectory.empty())
    {
        paths.hostRoot = paths.elfDirectory;
        paths.cdRoot = paths.elfDirectory;
        paths.mcRoot = paths.elfDirectory / "mc0";
    }

    setIoPaths(paths);
}

namespace
{
    bool generatedFunctionTableSlot(uint32_t address, uint32_t &slot)
    {
        if ((address & 3u) != 0u || g_ps2RecompiledFunctionTableSlotCount == 0u)
        {
            return false;
        }

        if (address < g_ps2RecompiledFunctionTableBase || address >= g_ps2RecompiledFunctionTableEnd)
        {
            return false;
        }

        const uint32_t offset = address - g_ps2RecompiledFunctionTableBase;
        slot = offset >> 2;
        return slot < g_ps2RecompiledFunctionTableSlotCount;
    }
}

bool PS2Runtime::replaceFunction(uint32_t address, RecompiledFunction func)
{
    uint32_t slot = 0u;
    if (!generatedFunctionTableSlot(address, slot))
    {
        std::cerr << "[function-table] cannot replace guest PC 0x" << std::hex << address
                  << ": outside generated dense table [0x" << g_ps2RecompiledFunctionTableBase
                  << ", 0x" << g_ps2RecompiledFunctionTableEnd << ")"
                  << std::dec << std::endl;
        return false;
    }

    g_ps2RecompiledFunctionTable[slot] = func;
    return true;
}

bool PS2Runtime::registerFunction(uint32_t address, RecompiledFunction func)
{
    return replaceFunction(address, func);
}

bool PS2Runtime::hasFunction(uint32_t address) const
{
    uint32_t slot = 0u;
    return generatedFunctionTableSlot(address, slot) && g_ps2RecompiledFunctionTable[slot] != nullptr;
}

const char *describeGuestBranchKind(PS2Runtime::GuestBranchKind kind)
{
    switch (kind)
    {
    case PS2Runtime::GuestBranchKind::DirectJump:
        return "DirectJump";
    case PS2Runtime::GuestBranchKind::DirectCall:
        return "DirectCall";
    case PS2Runtime::GuestBranchKind::IndirectJump:
        return "IndirectJump";
    case PS2Runtime::GuestBranchKind::IndirectCall:
        return "IndirectCall";
    case PS2Runtime::GuestBranchKind::Return:
        return "Return";
    default:
        return "Unknown";
    }
}

PS2Runtime::RecompiledFunction PS2Runtime::lookupFunction(uint32_t address)
{
    pushDispatchPc(address);

    uint32_t slot = 0u;
    if (generatedFunctionTableSlot(address, slot))
    {
        RecompiledFunction fn = g_ps2RecompiledFunctionTable[slot];
        if (fn != nullptr)
        {
            return fn;
        }
    }

    std::cerr << "Error: No exact recompiled function for guest PC 0x" << std::hex << address
              << " tableBase=0x" << g_ps2RecompiledFunctionTableBase
              << " tableEnd=0x" << g_ps2RecompiledFunctionTableEnd
              << " codeRegion=" << (m_memory.isCodeAddress(address) ? "yes" : "no")
              << " trace=" << formatDispatchHistory()
              << std::dec << std::endl;

    static RecompiledFunction missingFunction = [](uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
    {
        const uint32_t badPc = ctx->pc;
        runtime->reportMissingFunction(rdram,
                                       ctx,
                                       badPc,
                                       0u,
                                       PS2Runtime::GuestBranchKind::IndirectJump,
                                       "dispatch");
    };

    return missingFunction;
}

void PS2Runtime::setMissingFunctionPolicy(MissingFunctionPolicy policy)
{
    m_missingFunctionPolicy.store(static_cast<uint32_t>(policy), std::memory_order_release);
}

PS2Runtime::MissingFunctionPolicy PS2Runtime::missingFunctionPolicy() const
{
    return static_cast<MissingFunctionPolicy>(m_missingFunctionPolicy.load(std::memory_order_acquire));
}

void PS2Runtime::resetMissingFunctionReportOnce()
{
    m_missingFunctionReported.store(false, std::memory_order_release);
}

void PS2Runtime::reportMissingFunction(uint8_t *rdram,
                                       R5900Context *ctx,
                                       uint32_t targetPc,
                                       uint32_t sourcePc,
                                       GuestBranchKind kind,
                                       const char *debugName)
{
    const MissingFunctionPolicy policy = missingFunctionPolicy();
    const bool firstReport = !m_missingFunctionReported.exchange(true, std::memory_order_acq_rel);

    const uint32_t pc = ctx->pc;
    const uint32_t ra = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[31], 0));
    const uint32_t sp = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[29], 0));
    const uint32_t gp = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[28], 0));
    const uint32_t a0 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[4], 0));
    const uint32_t a1 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[5], 0));
    const uint32_t a2 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[6], 0));
    const uint32_t a3 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[7], 0));
    const uint32_t s0 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[16], 0));
    const uint32_t s1 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[17], 0));
    const uint32_t v0 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[2], 0));
    const uint32_t v1 = static_cast<uint32_t>(_mm_extract_epi32(ctx->r[3], 0));

    auto readGuestU32At = [rdram](uint32_t addr, uint32_t &out) -> bool
    {
        // TODO this !rdram exist only because of test fix those test later
        if (!rdram || addr > PS2_RAM_SIZE - sizeof(uint32_t))
        {
            out = 0u;
            return false;
        }

        std::memcpy(&out, rdram + addr, sizeof(uint32_t));
        return true;
    };

    auto readGuestU32Offset = [&readGuestU32At](uint32_t base, uint32_t offset, uint32_t &out) -> bool
    {
        if (base > PS2_RAM_SIZE - sizeof(uint32_t) || offset > PS2_RAM_SIZE - sizeof(uint32_t) - base)
        {
            out = 0u;
            return false;
        }

        return readGuestU32At(base + offset, out);
    };

    uint32_t a0Word0 = 0u;
    uint32_t a0Word4 = 0u;
    uint32_t a0Word8 = 0u;
    uint32_t a0WordC = 0u;
    const bool a0Readable =
        readGuestU32Offset(a0, 0x00u, a0Word0) &&
        readGuestU32Offset(a0, 0x04u, a0Word4) &&
        readGuestU32Offset(a0, 0x08u, a0Word8) &&
        readGuestU32Offset(a0, 0x0cu, a0WordC);

    uint32_t s0Word0 = 0u;
    uint32_t s0Word4 = 0u;
    uint32_t s0Word8 = 0u;
    uint32_t s0WordC = 0u;
    const bool s0Readable =
        readGuestU32Offset(s0, 0x00u, s0Word0) &&
        readGuestU32Offset(s0, 0x04u, s0Word4) &&
        readGuestU32Offset(s0, 0x08u, s0Word8) &&
        readGuestU32Offset(s0, 0x0cu, s0WordC);

    uint32_t recordWord0 = 0u;
    uint32_t recordWord4 = 0u;
    uint32_t recordWord8 = 0u;
    uint32_t recordWordC = 0u;
    const bool recordReadable =
        s0Readable && s0Word4 != 0u &&
        readGuestU32Offset(s0Word4, 0x00u, recordWord0) &&
        readGuestU32Offset(s0Word4, 0x04u, recordWord4) &&
        readGuestU32Offset(s0Word4, 0x08u, recordWord8) &&
        readGuestU32Offset(s0Word4, 0x0cu, recordWordC);

    uint32_t vtableSlot0 = 0u;
    uint32_t vtableSlot4 = 0u;
    uint32_t vtableSlot8 = 0u;
    uint32_t vtableSlotC = 0u;
    const bool vtableReadable =
        a0Readable && a0Word0 != 0u &&
        readGuestU32Offset(a0Word0, 0x00u, vtableSlot0) &&
        readGuestU32Offset(a0Word0, 0x04u, vtableSlot4) &&
        readGuestU32Offset(a0Word0, 0x08u, vtableSlot8) &&
        readGuestU32Offset(a0Word0, 0x0cu, vtableSlotC);

    if (firstReport)
    {
        std::ostringstream oss;
        oss << "[guest-branch:missing-target] kind=" << describeGuestBranchKind(kind)
            << " op=" << (debugName ? debugName : "<unknown>")
            << " source=0x" << std::hex << sourcePc
            << " target=0x" << targetPc
            << " pc=0x" << pc
            << " ra=0x" << ra
            << " sp=0x" << sp
            << " gp=0x" << gp
            << " a0=0x" << a0
            << " a1=0x" << a1
            << " a2=0x" << a2
            << " a3=0x" << a3
            << " s0=0x" << s0
            << " s1=0x" << s1
            << " v0=0x" << v0
            << " v1=0x" << v1
            << " a0Readable=" << (a0Readable ? "yes" : "no")
            << " a0[0]=0x" << a0Word0
            << " a0[4]=0x" << a0Word4
            << " a0[8]=0x" << a0Word8
            << " a0[c]=0x" << a0WordC
            << " s0Readable=" << (s0Readable ? "yes" : "no")
            << " s0[0]=0x" << s0Word0
            << " s0[4]=0x" << s0Word4
            << " s0[8]=0x" << s0Word8
            << " s0[c]=0x" << s0WordC
            << " recordReadable=" << (recordReadable ? "yes" : "no")
            << " record[0]=0x" << recordWord0
            << " record[4]=0x" << recordWord4
            << " record[8]=0x" << recordWord8
            << " record[c]=0x" << recordWordC
            << " vtableReadable=" << (vtableReadable ? "yes" : "no")
            << " vtbl[0]=0x" << vtableSlot0
            << " vtbl[4]=0x" << vtableSlot4
            << " vtbl[8]=0x" << vtableSlot8
            << " vtbl[c]=0x" << vtableSlotC
            << " codeRegion=" << (m_memory.isCodeAddress(targetPc) ? "yes" : "no")
            << " policy=" << static_cast<uint32_t>(policy)
            << " trace=" << formatDispatchHistory()
            << std::dec;

        static std::mutex s_missingFunctionLogMutex;
        {
            std::lock_guard<std::mutex> lock(s_missingFunctionLogMutex);
            std::cerr << oss.str() << std::endl;
        }
    }

    if (firstReport && policy == MissingFunctionPolicy::BreakOnce)
    {
#if defined(_MSC_VER)
        __debugbreak();
#endif // TODO others breakpoints
    }

    if (ctx)
    {
        ctx->pc = targetPc;
    }

    if (policy == MissingFunctionPolicy::Stop)
    {
        requestStop();
    }
}

// These are the original generated leaf routines; slow frames reuse their
// force equations and integrator rather than substituting new physics.
extern void applyDampeningForce__9PointMassf_0x21da98(uint8_t*, R5900Context*, PS2Runtime*);
extern void applyForces__6Spring_0x21dbf0(uint8_t*, R5900Context*, PS2Runtime*);
extern void resolveForcesEuler__9PointMassf_0x21db70(uint8_t*, R5900Context*, PS2Runtime*);

static bool ps2xStepSlowBoneSpring(uint8_t* rdram, R5900Context* ctx, PS2Runtime* runtime)
{
    const float dt = ctx->f[12];
    // Keep the original 30/60 Hz path bit-for-bit. WotM caps missed ticks at
    // 15 vblanks, so its largest legitimate interval is a quarter second.
    if (!(dt > (1.0f / 30.0f) && dt <= 0.25f)) return false;
    static const bool enabled = [] { const char* v=std::getenv("PS2X_BONE_SUBSTEPS"); return !v || v[0]!='0'; }();
    if (!enabled) return false;
    const uint32_t object = GPR_U32(ctx,17), point = GPR_U32(ctx,4);
    if (object > 0x02000000u-196 || point != object+64) return false;
    uint32_t first, second, game, position;
    std::memcpy(&first,rdram+object+48,4);std::memcpy(&second,rdram+object+52,4);
    std::memcpy(&position,rdram+point,4);std::memcpy(&game,rdram+0x6f81f8,4);
    if (first!=point || second!=object+144 || position>0x02000000u-16 || game>0x02000000u-0x1203c8) return false;
    uint32_t anchor;
    std::memcpy(&anchor,rdram+second,4);
    if(anchor>0x02000000u-16) return false;
    float damping, mass, gravity;
    std::memcpy(&damping,rdram+object+112,4);std::memcpy(&mass,rdram+object+32,4);
    std::memcpy(&gravity,rdram+game+0x1203c4,4);
    const float externalZ=FPU_ADD_S(0.0f,FPU_MUL_S(gravity,mass));
    const unsigned steps=static_cast<unsigned>(std::ceil(double(dt)*30.0));
    const float h=dt/static_cast<float>(steps);
    for(unsigned step=0;step<steps;++step)
    {
        if(step)
        {
            const float force[3]={0.0f,0.0f,externalZ};
            std::memcpy(rdram+point+32,force,12);
            SET_GPR_U32(ctx,4,point);ctx->f[12]=damping;
            applyDampeningForce__9PointMassf_0x21da98(rdram,ctx,runtime);
            SET_GPR_U32(ctx,4,object+36);
            applyForces__6Spring_0x21dbf0(rdram,ctx,runtime);
        }
        SET_GPR_U32(ctx,4,point);ctx->f[12]=h;
        resolveForcesEuler__9PointMassf_0x21db70(rdram,ctx,runtime);
    }
    ctx->f[12]=dt;
    static const bool trace=[] {const char* v=std::getenv("PS2X_BONE_CHECK");return v && v[0]=='1';}();
    if(trace){static uint64_t calls=0;if(++calls<=4 || calls%10000==0)std::fprintf(stderr,"[bone:substeps] calls=%llu dt=%g steps=%u\n",calls,dt,steps);}
    return true;
}

bool PS2Runtime::dispatchGuestBranch(uint8_t *rdram,
                                     R5900Context *ctx,
                                     uint32_t targetPc,
                                     uint32_t sourcePc,
                                     uint32_t fallthroughPc,
                                     GuestBranchKind kind,
                                     const char *debugName)
{
    ctx->pc = targetPc;
    const bool isCall = (kind == GuestBranchKind::DirectCall || kind == GuestBranchKind::IndirectCall);
    // Inspect the existing spring clamp's output before its matrix transform.
    if (sourcePc == 0x11b748u && targetPc == 0x20c6c0u)
    {
        static const bool enabled = [] { const char* v=std::getenv("PS2X_BONE_CHECK"); return v && v[0]=='1'; }();
        if (enabled)
        {
            const uint32_t obj=GPR_U32(ctx,17)&0x1fffffffu;
            const uint32_t matrix=GPR_U32(ctx,4)&0x1fffffffu;
            const uint32_t vector=GPR_U32(ctx,5)&0x1fffffffu;
            if (obj<=0x02000000u-196 && matrix<=0x02000000u-64 && vector<=0x02000000u-16)
            {
                float input[4],mat[16],object[49];
                std::memcpy(input,rdram+vector,16);std::memcpy(mat,rdram+matrix,64);std::memcpy(object,rdram+obj,196);
                const double len=std::sqrt(double(input[0])*input[0]+double(input[1])*input[1]+double(input[2])*input[2]);
                double maxMat=0;for(unsigned i=0;i<12;++i)maxMat=std::max(maxMat,std::fabs(double(mat[i])));
                static uint64_t calls=0,large=0,logged=0; ++calls;
                const bool bad=!std::isfinite(len)||len>1000||maxMat>1000;
                large+=bad;
                if (calls<=4 || (bad && logged++<16))
                {
                    std::fprintf(stderr,"[bone:spring] call=%llu obj=%x dest=%x matrix=%x len=%g limit=%g maxMat=%g f0=%g f1=%g f2=%g fcr=%x\n",
                        calls,obj,GPR_U32(ctx,6),matrix,len,object[14],maxMat,ctx->f[0],ctx->f[1],ctx->f[2],ctx->fcr31);
                    std::fprintf(stderr,"[bone:spring] delta=(%g,%g,%g,%g) base=(%g,%g,%g,%g)\n",input[0],input[1],input[2],input[3],object[4],object[5],object[6],object[7]);
                    uint32_t rate;std::memcpy(&rate,rdram+0x6f8db8,4);
                    std::fprintf(stderr,"[bone:spring-physics] dt=%g rate=%u invMass=%g damping=%g rest=%g stiffness=%g springDamping=%g velocity=(%g,%g,%g) force=(%g,%g,%g)\n",ctx->f[12],rate,object[27],object[28],object[9],object[10],object[11],object[20],object[21],object[22],object[24],object[25],object[26]);
                    for(int row=0;row<4;++row)std::fprintf(stderr,"[bone:spring] matrix%d=(%g,%g,%g,%g)\n",row,mat[row*4],mat[row*4+1],mat[row*4+2],mat[row*4+3]);
                }
                if(calls%100000==0)std::fprintf(stderr,"[bone:spring-summary] calls=%llu large=%llu\n",calls,large);
            }
        }
    }


#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    // Remember which hierTrace* launched the current hierTraverseAsm so the
    // vcallms hook in executeVU0Microprogram can attribute each transform node
    // to its traversal (0x202fbc=world, 0x203088=csList, 0x203170=HP csList).
    extern std::atomic<uint32_t> g_travOrigin;
    if (targetPc == 0x00207198u)
        g_travOrigin.store(sourcePc, std::memory_order_relaxed);
#endif


    // Every inter-function transfer is also a deterministic EE safe point.
    // Backward edges inside generated functions use eeCheckpointDue(), while
    // this charge bounds straight-line call chains that have no local loop.
    if (m_eeScheduler && m_eeScheduler->checkpointDue(EeScheduler::kGuestDispatchCycles))
    {
        return false;
    }

    if (isCall && sourcePc==0x11b6a4u && targetPc==0x21db70u && ps2xStepSlowBoneSpring(rdram,ctx,this))
        return true;

    if (!isCall)
    {
        if (!hasFunction(targetPc))
        {
            reportMissingFunction(rdram, ctx, targetPc, sourcePc, kind, debugName);
        }

        ctx->pc = targetPc;
        return false;
    }

    if (!hasFunction(targetPc))
    {
        reportMissingFunction(rdram, ctx, targetPc, sourcePc, kind, debugName);

        const MissingFunctionPolicy policy = missingFunctionPolicy();

        if (policy == MissingFunctionPolicy::SkipCallDebug && isCall)
        {
            ctx->pc = fallthroughPc;
            return true;
        }

        if (policy == MissingFunctionPolicy::ContinueToTarget)
        {
            ctx->pc = targetPc;
            return true;
        }

        return false;
    }

    RecompiledFunction targetFn = lookupFunction(targetPc);
    const uint32_t entryPc = ctx->pc;
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    // [travjal] — jal issued from inside hierTraverseAsm's node-type handlers
    // (0x207198..0x2083ff). Lets us see the non-vcallms node handlers (type 25
    // @0x2080b4, type 37 @0x208208 = the camera node calling hierSetCamera
    // @0x204F50, etc.) + the recursion counter s2 (GPR18) at that point.
    {
        extern std::atomic<uint32_t> g_travOrigin;
        if (sourcePc >= 0x00207198u && sourcePc <= 0x002083FFu &&
            g_travOrigin.load(std::memory_order_relaxed) == 0x00203088u &&
            GPR_U32(ctx, 21) != 0x00A028C0u)
        {
            static std::atomic<uint32_t> s_tj{0u};
            if (s_tj.fetch_add(1u, std::memory_order_relaxed) < 400u)
                std::fprintf(stderr,
                    "[travjal] srcPc=0x%x target=0x%x node=0x%x s2=0x%x s1=0x%x s3=0x%x depthCtr=%u\n",
                    sourcePc, targetPc, GPR_U32(ctx, 21), GPR_U32(ctx, 18),
                    GPR_U32(ctx, 17), GPR_U32(ctx, 19), m_memory.read32(0x70000050u));
        }
    }
    // Only trace hierTrace* calls once the shell has SETTLED on screenMain
    // (betweenScreens==0) — otherwise the caps fill during the slow transition
    // when cs 0x800bd0 is still disabled.
    const bool shellSettled = (m_memory.read32(0x6F7E8Cu) == 0u &&
                               m_memory.read32(0x6F8464u) == 1u);
    if (shellSettled &&
        (targetPc == 0x002029B0u || targetPc == 0x00202F40u || targetPc == 0x00207198u ||
        targetPc == 0x0020F310u || targetPc == 0x00204F50u || targetPc == 0x00202FD8u ||
        targetPc == 0x002030C0u || targetPc == 0x00202EE0u))
    {
        // Separate counter for hierTraverseAsm (0x207198) so its call-sites
        // (0x202fbc = hierTraceWorld, 0x203170 = hierTraceHPCsList, inside
        // hierTraceCsList) aren't crowded out of the shared budget.
        // Is hierTraceHPCsList / hierTraceCsList being RESTARTED (resume pc ==
        // its entry) or resumed mid-loop? entryPc here = ctx->pc set by the
        // caller before dispatch == the resume point.
        if (targetPc == 0x002030C0u || targetPc == 0x00202FD8u)
        {
            static std::atomic<uint32_t> s_re{0u};
            if (s_re.fetch_add(1u, std::memory_order_relaxed) < 40u)
                std::fprintf(stderr, "[csresume] tracer=0x%x  ctx->pc(resume)=0x%x  from=0x%x  s0=0x%x\n",
                             targetPc, ctx->pc, sourcePc, GPR_U32(ctx, 16));
        }
        static std::atomic<uint32_t> s_um{0u};
        static std::atomic<uint32_t> s_umTrav{0u};
        const bool isTrav = (targetPc == 0x00207198u);
        const uint32_t n = (isTrav ? s_umTrav : s_um).fetch_add(1u, std::memory_order_relaxed);
        if (n < (isTrav ? 300u : 60u))
        {
            // For hierTraverseAsm from a cs-list tracer (0x203170 = HP, 0x203088 =
            // normal), a0 is the cs ptr — dump the "active" flag word cs[+0xc]
            // + link chain the tracer walked so we can see why it isn't skipped.
            const uint32_t a0 = GPR_U32(ctx, 4);
            const uint32_t csFlag = (isTrav ? m_memory.read32(a0 + 0x0Cu) : 0u);
            std::fprintf(stderr, "[hiercall] 0x%x  a0=0x%x a1=0x%x from 0x%x  cs[+0xc]=0x%x s0=0x%x\n",
                         targetPc, a0, GPR_U32(ctx, 5), sourcePc, csFlag, GPR_U32(ctx, 16));
        }
    }
    // pktAddVu1ObjAsm (0x208410): dump the per-object MVP it is about to write to
    // the VIF1 packet (COP2 vf1/vf2/vf3 = matStack rows, vf6 = prog-B translation)
    // + the scene node ptr (s5 = GPR 21) + caller. Match by node ptr for a
    // synchronized A/B vs PCSX2 (bp 0x208458 reads the same COP2 regs).
    // Menu-3D subtree only (cs 0x800bd0): root nodes 0xA01xxx + instanced 0xE5xxxx.
    // Gate on shellSettled so the cap isn't eaten by world/HP/cs-0x800c80 packets
    // emitted before the menu traversal each frame.
    {
      const uint32_t pkNd = GPR_U32(ctx, 21);
      const bool pkMenu = (pkNd >= 0x00E00000u && pkNd < 0x01000000u) ||
                          (pkNd >= 0x00A01000u && pkNd < 0x00A02000u);
    if (targetPc == 0x00208410u && pkNd != 0x00A028C0u && pkMenu && shellSettled)
    {
        static std::atomic<uint32_t> s_pkt{0u};
        if (s_pkt.fetch_add(1u, std::memory_order_relaxed) < 400u)
        {
            alignas(16) float v1[4], v2[4], v3[4], v6[4];
            _mm_store_ps(v1, ctx->vu0_vf[1]);
            _mm_store_ps(v2, ctx->vu0_vf[2]);
            _mm_store_ps(v3, ctx->vu0_vf[3]);
            _mm_store_ps(v6, ctx->vu0_vf[6]);
            // s3 (GPR 19) = packed stack idx; low16 = matStack depth. Read the
            // matStack row this node's depth points at + the node header word.
            const uint32_t s3 = GPR_U32(ctx, 19);
            const uint32_t depth = s3 & 0xFFFFu;
            const uint32_t msAddr = 0x70000400u + (depth & 0x3Fu) * 0x40u;
            float ms[12];
            for (int k = 0; k < 12; ++k)
            {
                uint32_t b = m_memory.read32(msAddr + (uint32_t)k * 4u);
                std::memcpy(&ms[k], &b, 4);
            }
            const uint32_t nodeHdr = m_memory.read32(GPR_U32(ctx, 21));
            std::fprintf(stderr,
                "[pktobj] node=0x%x hdr=0x%x s3=0x%x depth=%u from=0x%x "
                "vf1=(%g,%g,%g,%g) vf2=(%g,%g,%g,%g) vf3=(%g,%g,%g,%g) vf6=(%g,%g,%g,%g) "
                "matStack[%u]=[(%g,%g,%g,%g)(%g,%g,%g,%g)(%g,%g,%g,%g)]\n",
                GPR_U32(ctx, 21), nodeHdr, s3, depth, sourcePc,
                v1[0], v1[1], v1[2], v1[3], v2[0], v2[1], v2[2], v2[3],
                v3[0], v3[1], v3[2], v3[3], v6[0], v6[1], v6[2], v6[3],
                depth, ms[0], ms[1], ms[2], ms[3], ms[4], ms[5], ms[6], ms[7],
                ms[8], ms[9], ms[10], ms[11]);
        }
    }
    }
    // Before/after s0 (GPR 16) + pc across a hierTraverseAsm call from a cs-list
    // tracer: the tracer's loop advances s0 = *(s0+4) after the call; if s0 is
    // unchanged and pc != the fallthrough, the loop is stuck.
    const bool traceS0 = (targetPc == 0x00207198u && sourcePc == 0x00203088u);
    const uint32_t s0Before = traceS0 ? GPR_U32(ctx, 16) : 0u;
    const uint32_t s3Before = traceS0 ? GPR_U32(ctx, 19) : 0u;
    const uint32_t raBefore = traceS0 ? GPR_U32(ctx, 31) : 0u;
    // func_203580 (hierRotateSkel) is jal'd from the hierTraverseAsm type-4
    // handler (guest 0x207a5c) just before `ori v0,s0,0x400`. If it fails to
    // preserve callee-saved s0/s3, the matStack write target is garbage.
    const bool traceSkel = (targetPc == 0x00203580u && sourcePc == 0x00207A5Cu);
    const uint32_t skelS0Before = traceSkel ? GPR_U32(ctx, 16) : 0u;
    const uint32_t skelS3Before = traceSkel ? GPR_U32(ctx, 19) : 0u;
    const uint32_t skelS5Before = traceSkel ? GPR_U32(ctx, 21) : 0u;
#endif
    targetFn(rdram, ctx, this);

    extern thread_local bool t_ps2xEeUnwinding; // EeScheduler.cpp
    if (t_ps2xEeUnwinding && ctx->pc == entryPc)
    {
        // The case the old "pc == entry means returned" rule got wrong.
        static uint64_t s_unwindAtEntry = 0u;
        ++s_unwindAtEntry;
        if ((s_unwindAtEntry & (s_unwindAtEntry - 1u)) == 0u)
            std::fprintf(stderr, "[ee:unwind-at-entry] count=%llu target=0x%08x from=0x%08x\n",
                         static_cast<unsigned long long>(s_unwindAtEntry), targetPc, sourcePc);
    }
    // PS2X_EE_OLD_ENTRY_RULE=1 (diagnostic A/B): the previous behaviour.
    static const bool s_oldEntryRule = []
    {
        const char *value = std::getenv("PS2X_EE_OLD_ENTRY_RULE");
        return value != nullptr && value[0] == '1';
    }();
    if (isStopRequested() || ctx->pc == 0u || (t_ps2xEeUnwinding && !s_oldEntryRule))
    {
        return false;
    }

    if (ctx->pc == entryPc)
    {
        ctx->pc = fallthroughPc;
    }

    const bool ret = (ctx->pc == fallthroughPc);
    if (!ret && !hasFunction(ctx->pc))
    {
        // A callee returned to a PC with no recompiled code (the freeze).
        static std::atomic<uint32_t> s_badReturns{0u};
        if (s_badReturns.fetch_add(1u, std::memory_order_relaxed) < 8u)
            std::fprintf(stderr, "[bad-return] call 0x%08x -> 0x%08x expected 0x%08x got pc=0x%08x ra=0x%08x sp=0x%08x a0=0x%08x s0=0x%08x s1=0x%08x\n",
                         sourcePc, targetPc, fallthroughPc, ctx->pc, GPR_U32(ctx, 31), GPR_U32(ctx, 29),
                         GPR_U32(ctx, 4), GPR_U32(ctx, 16), GPR_U32(ctx, 17));
    }
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    if (traceS0 && m_memory.read32(0x6F7E8Cu) == 0u && m_memory.read32(0x6F8464u) == 1u)
    {
        static std::atomic<uint32_t> s_s0{0u};
        if (s_s0.fetch_add(1u, std::memory_order_relaxed) < 120u)
        {
            const uint32_t s0now = GPR_U32(ctx, 16);
            std::fprintf(stderr,
                "[travret] from=0x%x s0:0x%x->0x%x s3:0x%x->0x%x ra:0x%x->0x%x retPc=0x%x ret=%d *(s0)=0x%x *(s0+4)=0x%x "
                "| mActiveNext=0x%x cs800bd0[+0]=0x%x cs800bd0[+0xC]=0x%x cs800c80[+0xC]=0x%x link808e80cs=0x%x\n",
                sourcePc, s0Before, s0now, s3Before, GPR_U32(ctx, 19),
                raBefore, GPR_U32(ctx, 31), ctx->pc, (int)ret,
                m_memory.read32(s0now), m_memory.read32(s0now + 4u),
                m_memory.read32(0x0043A2C4u), m_memory.read32(0x00800BD0u),
                m_memory.read32(0x00800BDCu), m_memory.read32(0x00800C8Cu),
                m_memory.read32(0x00808E80u));
        }
    }
    if (traceSkel)
    {
        static std::atomic<uint32_t> s_skel{0u};
        if (s_skel.fetch_add(1u, std::memory_order_relaxed) < 60u)
        {
            std::fprintf(stderr,
                "[skelret] node=0x%x s0:0x%x->0x%x s3:0x%x->0x%x s5:0x%x->0x%x retPc=0x%x\n",
                skelS5Before, skelS0Before, GPR_U32(ctx, 16),
                skelS3Before, GPR_U32(ctx, 19),
                skelS5Before, GPR_U32(ctx, 21), ctx->pc);
        }
    }
#endif
    return ret;
}

void PS2Runtime::SignalException(R5900Context *ctx, PS2Exception exception)
{
    if (exception == EXCEPTION_INTEGER_OVERFLOW)
    {
        HandleIntegerOverflow(ctx);
        return;
    }

    raiseCop0Exception(ctx, static_cast<uint32_t>(exception),
                       exception == EXCEPTION_TLB_REFILL);
}


// ---------------------------------------------------------------------------
// PS2X_BONE_CHECK=1: skeleton matrix checker.
// animTNodeSkel (hierTraverseAsm, 0x207de4) builds each bone's world matrix
// with VU0 micro-routine 0xA00 (vu0MulMatrixSkel1): local rows vf1,vf2,vf3,vf5
// x parent rows vf14..vf17 -> vf28..vf31; 0xA80 (vu0MulMatrixSkel2) turns
// vf20..vf23 into the skinning matrix. A rigid (rotation x uniform scale) 3x3
// has equal-length, mutually perpendicular rows, so a stretched bone shows up
// as unequal row lengths / skewed rows. For every 0xA00 run we classify the
// inputs and output and recompute the product in C++ to tell a VU0 emulation
// error (good inputs, wrong output) from bad inputs.
// ---------------------------------------------------------------------------
namespace
{
    struct BoneCheckRow3 { double x, y, z; };
    inline BoneCheckRow3 bcRow(const float *v) { return {v[0], v[1], v[2]}; }
    inline double bcLen(const BoneCheckRow3 &r) { return std::sqrt(r.x * r.x + r.y * r.y + r.z * r.z); }
    inline double bcDot(const BoneCheckRow3 &a, const BoneCheckRow3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
    // 0 = rigid; 1 = stretched/skewed; 2 = NaN/huge; 3 = degenerate (zero row)
    int bcClassify(const float *r0, const float *r1, const float *r2, double *ratioOut, double *skewOut)
    {
        const BoneCheckRow3 a = bcRow(r0), b = bcRow(r1), c = bcRow(r2);
        const double la = bcLen(a), lb = bcLen(b), lc = bcLen(c);
        if (!std::isfinite(la) || !std::isfinite(lb) || !std::isfinite(lc) || la > 1e5 || lb > 1e5 || lc > 1e5)
            return 2;
        const double mn = std::min(la, std::min(lb, lc)), mx = std::max(la, std::max(lb, lc));
        if (mn < 1e-6)
            return 3;
        const double ratio = mx / mn;
        const double skew = std::max(std::fabs(bcDot(a, b)) / (la * lb),
                                     std::max(std::fabs(bcDot(a, c)) / (la * lc), std::fabs(bcDot(b, c)) / (lb * lc)));
        if (ratioOut) *ratioOut = ratio;
        if (skewOut) *skewOut = skew;
        return (ratio > 1.25 || skew > 0.15) ? 1 : 0;
    }
    struct BoneCheckStats
    {
        std::atomic<uint64_t> calls{0}, outBad{0}, localBad{0}, parentBad{0}, emuErr{0};
    };
    BoneCheckStats g_boneCheck;
    bool boneCheckEnabled()
    {
        static const bool on = []
        {
            const char *v = std::getenv("PS2X_BONE_CHECK");
            return v != nullptr && v[0] == '1';
        }();
        return on;
    }
}

extern thread_local bool t_vu0ForceInterpreter; // ps2_vu1_core.cpp
std::atomic<uint64_t> g_ps2xVu0Calls{0u};
std::atomic<uint64_t> g_ps2xVu0Ns{0u};
std::atomic<uint64_t> g_ps2xVu0Instr{0u};
std::atomic<uint64_t> g_ps2xVu0SetupNs{0u}, g_ps2xVu0ExecNs{0u};

namespace
{
    // Adds the call's wall time to g_ps2xVu0Ns on scope exit ([perf] vu0 line).
    struct Vu0CallTimer
    {
        bool enabled;
        Ps2xTscClock::time_point t0;
        explicit Vu0CallTimer(bool profile):enabled(profile),t0(profile?Ps2xTscClock::now():Ps2xTscClock::time_point{}){}
        ~Vu0CallTimer()
        {
            if(!enabled)return;
            g_ps2xVu0Calls.fetch_add(1u, std::memory_order_relaxed);
            g_ps2xVu0Ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      Ps2xTscClock::now() - t0).count()),
                                  std::memory_order_relaxed);
        }
    };
}

void PS2Runtime::executeVU0Microprogram(uint8_t *rdram, R5900Context *ctx, uint32_t address)
{
    (void)rdram;
    // Shipping execution does no per-call clock reads or diagnostic atomics.
    // PERF still enables the existing timing report; VU0_PROFILE overrides it.
    static const bool profile=[] {const char* p=std::getenv("PS2X_VU0_PROFILE");
        if(p)return p[0]=='1';p=std::getenv("PS2X_PERF");return p && p[0]!='0';}();
    Vu0CallTimer vu0Timer(profile);

    uint8_t *const vu0Code = m_memory.getVU0Code();
    uint8_t *const vu0Data = m_memory.getVU0Data();
    const uint32_t startPC = address & ~0x7u;

    if (!vu0Code || !vu0Data || startPC + 8u > PS2_VU0_CODE_SIZE)
    {
        seedVu0IdleSuccess(ctx);
        return;
    }

    const auto vu0T0 = profile?Ps2xTscClock::now():Ps2xTscClock::time_point{};
    m_vu0.reset();
    copyVu0ContextToState(ctx, m_vu0.state());
    const auto vu0T1 = profile?Ps2xTscClock::now():Ps2xTscClock::time_point{};
    const bool boneCheckHere = startPC == 0xA00u && boneCheckEnabled();
    // Extended checker: the skinning matrix (0xA80, vu0MulMatrixSkel2: vf20..vf23
    // -> skin matrix vf20..vf23 and bone vf28..vf31) and the matrix-stack push
    // composer (0x1A8: stores rows vf21-23, vf25-27, vf9-11, vf28-30).
    const bool boneCheckExt = startPC == 0xA80u && boneCheckEnabled(); // 0x1A8 outputs are not rigid rotations (false alarms)
    float bcInExt[32][4];
    if (boneCheckExt)
        std::memcpy(bcInExt, m_vu0.state().vf, sizeof(bcInExt));
    float bcIn[32][4];
    if (boneCheckHere)
        std::memcpy(bcIn, m_vu0.state().vf, sizeof(bcIn));
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    {
        // One line per distinct startPC: dump the first 4 instruction words so we
        // can see whether real microcode is present at that entry point.
        static std::mutex s_v0mtx;
        static std::set<uint32_t> s_v0seen;
        bool fresh = false;
        {
            std::lock_guard<std::mutex> lk(s_v0mtx);
            fresh = s_v0seen.insert(startPC).second;
        }
        if (fresh)
        {
            uint32_t w[8] = {0};
            for (int i = 0; i < 8 && startPC + (i + 1) * 4u <= PS2_VU0_CODE_SIZE; ++i)
                std::memcpy(&w[i], vu0Code + startPC + i * 4u, 4);
            std::fprintf(stderr,
                "[vu0:exec] startPC=0x%x code=%08x:%08x %08x:%08x %08x:%08x %08x:%08x itop=%u\n",
                startPC, w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], ctx->vu0_itop);
        }
    }
#endif
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    static std::atomic<uint32_t> s_vu0baA{0u};   // 0x0 / 0x3F8 / 0x398
    static std::atomic<uint32_t> s_vu0baB{0u};   // 0x1A8 / 0x748 (transform-node composers)
    static std::atomic<uint32_t> s_vu0baC{0u};   // 0xB00 / 0xB50 (hierCsUpdateAsm light/norm)
    bool traceBA = false;
    if (startPC == 0x1A8u || startPC == 0x748u)
        traceBA = s_vu0baB.fetch_add(1u, std::memory_order_relaxed) < 40u;
    else if (startPC == 0x0u || startPC == 0x3F8u || startPC == 0x398u)
        traceBA = s_vu0baA.fetch_add(1u, std::memory_order_relaxed) < 20u;
    else if (startPC == 0xB00u || startPC == 0xB50u)
        traceBA = s_vu0baC.fetch_add(1u, std::memory_order_relaxed) < 24u;
    auto dumpVf = [](const char *tag, const VU1State &st)
    {
        std::fprintf(stderr, "[vu0:ba] %s vf1=(%g,%g,%g,%g) vf2=(%g,%g,%g,%g) vf3=(%g,%g,%g,%g)"
                             " vf4=(%g,%g,%g,%g) vf5=(%g,%g,%g,%g) vf6=(%g,%g,%g,%g)"
                             " vf7=(%g,%g,%g,%g) vf8=(%g,%g,%g,%g) vf9=(%g,%g,%g,%g)\n",
            tag,
            st.vf[1][0], st.vf[1][1], st.vf[1][2], st.vf[1][3],
            st.vf[2][0], st.vf[2][1], st.vf[2][2], st.vf[2][3],
            st.vf[3][0], st.vf[3][1], st.vf[3][2], st.vf[3][3],
            st.vf[4][0], st.vf[4][1], st.vf[4][2], st.vf[4][3],
            st.vf[5][0], st.vf[5][1], st.vf[5][2], st.vf[5][3],
            st.vf[6][0], st.vf[6][1], st.vf[6][2], st.vf[6][3],
            st.vf[7][0], st.vf[7][1], st.vf[7][2], st.vf[7][3],
            st.vf[8][0], st.vf[8][1], st.vf[8][2], st.vf[8][3],
            st.vf[9][0], st.vf[9][1], st.vf[9][2], st.vf[9][3]);
        std::fprintf(stderr, "[vu0:ba] %s   vf10=(%g,%g,%g,%g) vf11=(%g,%g,%g,%g)"
                             " vf12=(%g,%g,%g,%g) vf16=(%g,%g,%g,%g) vf20=(%g,%g,%g,%g)"
                             " vf21=(%g,%g,%g,%g) vf22=(%g,%g,%g,%g) vf23=(%g,%g,%g,%g)\n",
            tag,
            st.vf[10][0], st.vf[10][1], st.vf[10][2], st.vf[10][3],
            st.vf[11][0], st.vf[11][1], st.vf[11][2], st.vf[11][3],
            st.vf[12][0], st.vf[12][1], st.vf[12][2], st.vf[12][3],
            st.vf[16][0], st.vf[16][1], st.vf[16][2], st.vf[16][3],
            st.vf[20][0], st.vf[20][1], st.vf[20][2], st.vf[20][3],
            st.vf[21][0], st.vf[21][1], st.vf[21][2], st.vf[21][3],
            st.vf[22][0], st.vf[22][1], st.vf[22][2], st.vf[22][3],
            st.vf[23][0], st.vf[23][1], st.vf[23][2], st.vf[23][3]);
        // hierCsUpdateAsm (vcallms 0xB00/0xB50): vf13=csMat/paraL input, vf14-16=csNode mat,
        // vf17-19=lightDir/fovNorms output, vf24-31 scratch.
        std::fprintf(stderr, "[vu0:ba] %s   vf13=(%g,%g,%g,%g) vf14=(%g,%g,%g,%g)"
                             " vf15=(%g,%g,%g,%g) vf17=(%g,%g,%g,%g) vf18=(%g,%g,%g,%g)"
                             " vf19=(%g,%g,%g,%g) vf27=(%g,%g,%g,%g)\n",
            tag,
            st.vf[13][0], st.vf[13][1], st.vf[13][2], st.vf[13][3],
            st.vf[14][0], st.vf[14][1], st.vf[14][2], st.vf[14][3],
            st.vf[15][0], st.vf[15][1], st.vf[15][2], st.vf[15][3],
            st.vf[17][0], st.vf[17][1], st.vf[17][2], st.vf[17][3],
            st.vf[18][0], st.vf[18][1], st.vf[18][2], st.vf[18][3],
            st.vf[19][0], st.vf[19][1], st.vf[19][2], st.vf[19][3],
            st.vf[27][0], st.vf[27][1], st.vf[27][2], st.vf[27][3]);
    };
    if (traceBA)
    {
        char b[16];
        std::snprintf(b, sizeof(b), "0x%x PRE ", startPC);
        dumpVf(b, m_vu0.state());
    }
#endif
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    extern std::atomic<bool> g_vu0InstrTrace;
    extern std::atomic<uint32_t> g_vu0InstrTraceStartPc;
    g_vu0InstrTrace.store(traceBA, std::memory_order_relaxed);
    g_vu0InstrTraceStartPc.store(startPC, std::memory_order_relaxed);
#endif
    // PS2X_VU0_NATIVE_VERIFY=1: run the interpreter on a copy of the same
    // inputs and compare everything observable with the native VU0 program
    // (first 20000 calls, then every 16th). The native result stays live.
    static const bool s_vu0Verify = []
    {
        const char *value = std::getenv("PS2X_VU0_NATIVE_VERIFY");
        return value != nullptr && value[0] == '1';
    }();
    static uint64_t s_vu0VerifyCalls = 0u;
    const bool vu0VerifyHere = s_vu0Verify && (++s_vu0VerifyCalls <= 20000u || (s_vu0VerifyCalls % 16u) == 0u);
    std::unique_ptr<VU1Interpreter> vu0Ref;
    std::vector<uint8_t> vu0DataBefore;
    if (vu0VerifyHere)
    {
        vu0Ref = std::make_unique<VU1Interpreter>(m_vu0);
        vu0DataBefore.assign(vu0Data, vu0Data + PS2_VU0_DATA_SIZE);
    }
    m_vu0.execute(vu0Code, PS2_VU0_CODE_SIZE,
                  vu0Data, PS2_VU0_DATA_SIZE,
                  m_gs, &m_memory,
                  startPC, 0u, ctx->vu0_itop, 4096);
    if (vu0VerifyHere)
    {
        std::vector<uint8_t> vu0DataNative(vu0Data, vu0Data + PS2_VU0_DATA_SIZE);
        std::memcpy(vu0Data, vu0DataBefore.data(), PS2_VU0_DATA_SIZE);
        t_vu0ForceInterpreter = true;
        vu0Ref->execute(vu0Code, PS2_VU0_CODE_SIZE, vu0Data, PS2_VU0_DATA_SIZE, m_gs, &m_memory,
                        startPC, 0u, ctx->vu0_itop, 4096);
        t_vu0ForceInterpreter = false;
        const auto &a = vu0Ref->state();
        const auto &b = m_vu0.state();
        char what[160] = {0};
        for (int r = 1; r < 32 && !what[0]; ++r)
            if (std::memcmp(a.vf[r], b.vf[r], sizeof(a.vf[r])) != 0)
                std::snprintf(what, sizeof(what), "vf%d interp=(%g %g %g %g) native=(%g %g %g %g)", r,
                              a.vf[r][0], a.vf[r][1], a.vf[r][2], a.vf[r][3], b.vf[r][0], b.vf[r][1], b.vf[r][2], b.vf[r][3]);
        for (int r = 1; r < 16 && !what[0]; ++r)
            if (a.vi[r] != b.vi[r])
                std::snprintf(what, sizeof(what), "vi%d interp=%d native=%d", r, a.vi[r], b.vi[r]);
        if (!what[0] && std::memcmp(a.acc, b.acc, sizeof(a.acc)) != 0) std::snprintf(what, sizeof(what), "ACC");
        if (!what[0] && std::memcmp(&a.q, &b.q, sizeof(a.q)) != 0) std::snprintf(what, sizeof(what), "Q");
        if (!what[0] && std::memcmp(&a.i, &b.i, sizeof(a.i)) != 0) std::snprintf(what, sizeof(what), "I");
        if (!what[0] && a.mac != b.mac) std::snprintf(what, sizeof(what), "MAC interp=%x native=%x", a.mac, b.mac);
        if (!what[0] && a.status != b.status) std::snprintf(what, sizeof(what), "STATUS interp=%x native=%x", a.status, b.status);
        if (!what[0] && a.clip != b.clip) std::snprintf(what, sizeof(what), "CLIP interp=%x native=%x", a.clip, b.clip);
        if (!what[0] && a.cycles != b.cycles) std::snprintf(what, sizeof(what), "cycles interp=%llu native=%llu",
                                                              static_cast<unsigned long long>(a.cycles), static_cast<unsigned long long>(b.cycles));
        if (!what[0] && std::memcmp(vu0Data, vu0DataNative.data(), PS2_VU0_DATA_SIZE) != 0) std::snprintf(what, sizeof(what), "VU0 data memory");
        std::memcpy(vu0Data, vu0DataNative.data(), PS2_VU0_DATA_SIZE); // keep the native result live
        static uint64_t s_checked = 0u, s_bad = 0u;
        ++s_checked;
        if (what[0])
        {
            ++s_bad;
            if (s_bad <= 20u)
                std::fprintf(stderr, "[vu0:verify] MISMATCH #%llu startPC=0x%x %s\n",
                             static_cast<unsigned long long>(s_bad), startPC, what);
        }
        if ((s_checked % 20000u) == 0u)
            std::fprintf(stderr, "[vu0:verify] checked=%llu mismatched=%llu\n",
                         static_cast<unsigned long long>(s_checked), static_cast<unsigned long long>(s_bad));
    }
    if(profile)
    {
        const auto vu0T2 = Ps2xTscClock::now();
        g_ps2xVu0SetupNs.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(vu0T1 - vu0T0).count()), std::memory_order_relaxed);
        g_ps2xVu0Instr.fetch_add(static_cast<uint64_t>(m_vu0.state().cycles), std::memory_order_relaxed);
        g_ps2xVu0ExecNs.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(vu0T2 - vu0T1).count()), std::memory_order_relaxed);
    }
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    g_vu0InstrTrace.store(false, std::memory_order_relaxed);
    if (traceBA)
    {
        char b[16];
        std::snprintf(b, sizeof(b), "0x%x POST", startPC);
        dumpVf(b, m_vu0.state());
        std::fprintf(stderr, "[vu0:ba] 0x%x endpc=0x%x ebit=%d cyc=%llu stoppedD=%d stoppedT=%d\n",
                     startPC, m_vu0.state().pc, (int)m_vu0.state().ebit,
                     (unsigned long long)m_vu0.state().cycles,
                     (int)m_vu0.state().stoppedByD, (int)m_vu0.state().stoppedByT);
    }
    // [travnode] — every vcallms issued from inside hierTraverseAsm
    // (0x207198..0x2083ff). The issue PC identifies the node-type handler:
    //   0x2073c4  type-1 (vcallms 0)      0x207478/0x2074e8 type-0 (0 / 0x3F8)
    //   0x207850  type-2 (vcallms 0x398)  0x207ad8 type-4 (vcallms 0x1A8, matStack push)
    //   0x207e70  (vcallms 0xA80)         0x207f70 (2nd matStack push, ~type>=3 handler)
    // Logs node (s5=GPR21), s3 (packed depth), depthCtr *(0x70000050), and which
    // hierTrace* launched this traversal (g_travOrigin). Compare the node-visit
    // sequence + per-node handler vs retail (bp 0x207280, log s5 + t0&0x3F).
    if (ctx->pc >= 0x00207198u && ctx->pc <= 0x002083FFu &&
        GPR_U32(ctx, 21) != 0x00A028C0u)
    {
        extern std::atomic<uint32_t> g_travOrigin;
        const uint32_t tnOrigin = g_travOrigin.load(std::memory_order_relaxed);
        static std::atomic<uint32_t> s_tn{0u};
        // Only the CsList traversal (cs 0x800c80) — the one retail uses for the
        // node-0xa1df80 subtree. World (0x202fbc) + HP (0x203170) flood it out.
        if (tnOrigin == 0x00203088u &&
            s_tn.fetch_add(1u, std::memory_order_relaxed) < 1200u)
        {
            const VU1State &st = m_vu0.state();
            const uint32_t s3 = GPR_U32(ctx, 19);
            const uint32_t nd = GPR_U32(ctx, 21);
            // sibling array: type-17 handler writes frame[+0xC] = s5+0xD0 (guest
            // 0x207f78/0x208038). The pop does s5 = *(*(frame+0xC)). Dump node[+0xD0],
            // *node[+0xD0], and the s1=0x70001830 frame's +0xC/+0x14 slots.
            const uint32_t nD0 = m_memory.read32(nd + 0xD0u);
            std::fprintf(stderr,
                "[travnode] from=0x%x issuePc=0x%x vcallms=0x%x node=0x%x n0=0x%x n4=0x%x nCC=0x%x "
                "nD0=0x%x *nD0=0x%x n820_28=0x%x n820_2C=0x%x "
                "s2=0x%x s3=0x%x idx=%u depthCtr=%u s1=0x%x s7=0x%x s4=0x%x vf21POST=(%g,%g,%g,%g)\n",
                tnOrigin, ctx->pc, startPC, nd,
                m_memory.read32(nd), m_memory.read32(nd + 4u), m_memory.read32(nd + 0xCCu),
                nD0, (nD0 >= 0x100000u && nD0 < 0x2000000u) ? m_memory.read32(nD0) : 0u,
                m_memory.read32(0x00A00848u), m_memory.read32(0x00A0084Cu),
                GPR_U32(ctx, 18), s3, s3 & 0xFFFFu, m_memory.read32(0x70000050u),
                GPR_U32(ctx, 17), GPR_U32(ctx, 23), GPR_U32(ctx, 20),
                st.vf[21][0], st.vf[21][1], st.vf[21][2], st.vf[21][3]);
        }
    }
#endif
    if (boneCheckHere)
    {
        const auto &vf = m_vu0.state().vf;
        const uint64_t n = g_boneCheck.calls.fetch_add(1u, std::memory_order_relaxed) + 1u;
        double ro = 0, so = 0, rl = 0, sl = 0, rp = 0, sp = 0;
        const int outC = bcClassify(vf[28], vf[29], vf[30], &ro, &so);
        const int locC = bcClassify(bcIn[1], bcIn[2], bcIn[3], &rl, &sl);
        const int parC = bcClassify(bcIn[14], bcIn[15], bcIn[16], &rp, &sp);
        // Recompute local x parent (row vectors) for the 3x3 part.
        double expected[3][3]{}, worstErr = 0.0, scale = 1e-9;
        for (int i = 0; i < 3; ++i)
        {
            const float *li = bcIn[i + 1];
            for (int j = 0; j < 3; ++j)
            {
                double acc = 0.0;
                for (int k = 0; k < 3; ++k)
                    acc += static_cast<double>(li[k]) * static_cast<double>(bcIn[14 + k][j]);
                expected[i][j] = acc;
                worstErr = std::max(worstErr, std::fabs(acc - static_cast<double>(vf[28 + i][j])));
                scale = std::max(scale, std::fabs(acc));
            }
        }
        const bool emuErr = locC == 0 && parC == 0 && worstErr > 1e-3 * scale;
        if (outC != 0) g_boneCheck.outBad.fetch_add(1u, std::memory_order_relaxed);
        if (locC != 0) g_boneCheck.localBad.fetch_add(1u, std::memory_order_relaxed);
        if (parC != 0) g_boneCheck.parentBad.fetch_add(1u, std::memory_order_relaxed);
        if (emuErr) g_boneCheck.emuErr.fetch_add(1u, std::memory_order_relaxed);
        static std::atomic<uint32_t> s_logged{0u};
        if ((outC != 0 || emuErr) && s_logged.fetch_add(1u, std::memory_order_relaxed) < 40u)
        {
            std::fprintf(stderr,
                "[bone] call#%llu tick=%llu out=%d(ratio %.3f skew %.3f) local=%d(ratio %.3f skew %.3f) parent=%d(ratio %.3f skew %.3f) "
                "recomputeErr=%.4g%s\n"
                "[bone]   local  r0=(%g %g %g %g) r1=(%g %g %g %g) r2=(%g %g %g %g) t=(%g %g %g %g)\n"
                "[bone]   parent r0=(%g %g %g %g) r1=(%g %g %g %g) r2=(%g %g %g %g) t=(%g %g %g %g)\n"
                "[bone]   out    r0=(%g %g %g %g) r1=(%g %g %g %g) r2=(%g %g %g %g) t=(%g %g %g %g)\n"
                "[bone]   expect r0=(%g %g %g) r1=(%g %g %g) r2=(%g %g %g)\n",
                static_cast<unsigned long long>(n), static_cast<unsigned long long>(eeScheduler().currentVSyncTick()),
                outC, ro, so, locC, rl, sl, parC, rp, sp, worstErr, emuErr ? "  <== VU0 EMULATION ERROR" : "",
                bcIn[1][0], bcIn[1][1], bcIn[1][2], bcIn[1][3], bcIn[2][0], bcIn[2][1], bcIn[2][2], bcIn[2][3],
                bcIn[3][0], bcIn[3][1], bcIn[3][2], bcIn[3][3], bcIn[5][0], bcIn[5][1], bcIn[5][2], bcIn[5][3],
                bcIn[14][0], bcIn[14][1], bcIn[14][2], bcIn[14][3], bcIn[15][0], bcIn[15][1], bcIn[15][2], bcIn[15][3],
                bcIn[16][0], bcIn[16][1], bcIn[16][2], bcIn[16][3], bcIn[17][0], bcIn[17][1], bcIn[17][2], bcIn[17][3],
                vf[28][0], vf[28][1], vf[28][2], vf[28][3], vf[29][0], vf[29][1], vf[29][2], vf[29][3],
                vf[30][0], vf[30][1], vf[30][2], vf[30][3], vf[31][0], vf[31][1], vf[31][2], vf[31][3],
                expected[0][0], expected[0][1], expected[0][2], expected[1][0], expected[1][1], expected[1][2],
                expected[2][0], expected[2][1], expected[2][2]);
        }
        if ((n % 200000u) == 0u)
            std::fprintf(stderr, "[bone] calls=%llu outBad=%llu localBad=%llu parentBad=%llu vu0EmuErr=%llu\n",
                         static_cast<unsigned long long>(n),
                         static_cast<unsigned long long>(g_boneCheck.outBad.load()),
                         static_cast<unsigned long long>(g_boneCheck.localBad.load()),
                         static_cast<unsigned long long>(g_boneCheck.parentBad.load()),
                         static_cast<unsigned long long>(g_boneCheck.emuErr.load()));
    }
    if (boneCheckExt)
    {
        const auto &vf = m_vu0.state().vf;
        struct Triple { const char *name; int first; };
        static constexpr Triple kA80[] = {{"skin(vf20-22)", 20}, {"bone(vf28-30)", 28}};
        static constexpr Triple k1A8[] = {{"m0(vf21-23)", 21}, {"m1(vf25-27)", 25}, {"m2(vf9-11)", 9}, {"m3(vf28-30)", 28}};
        const Triple *triples = startPC == 0xA80u ? kA80 : k1A8;
        const size_t count = startPC == 0xA80u ? 2u : 4u;
        static std::atomic<uint64_t> s_callsA80{0u}, s_calls1A8{0u}, s_badA80{0u}, s_bad1A8{0u};
        std::atomic<uint64_t> &calls = startPC == 0xA80u ? s_callsA80 : s_calls1A8;
        std::atomic<uint64_t> &bad = startPC == 0xA80u ? s_badA80 : s_bad1A8;
        const uint64_t n = calls.fetch_add(1u, std::memory_order_relaxed) + 1u;
        for (size_t t = 0; t < count; ++t)
        {
            const int f = triples[t].first;
            double ro = 0, so = 0, ri = 0, si = 0;
            const int outC = bcClassify(vf[f], vf[f + 1], vf[f + 2], &ro, &so);
            if (outC == 0)
                continue;
            const int inC = bcClassify(bcInExt[f], bcInExt[f + 1], bcInExt[f + 2], &ri, &si);
            bad.fetch_add(1u, std::memory_order_relaxed);
            static std::atomic<uint32_t> s_logged{0u};
            if (s_logged.fetch_add(1u, std::memory_order_relaxed) < 60u)
            {
                std::fprintf(stderr,
                    "[bone2] vcallms=0x%x %s call#%llu tick=%llu out=%d(ratio %.3f skew %.3f) sameRegsBefore=%d(ratio %.3f skew %.3f)\n"
                    "[bone2]   out r0=(%g %g %g %g) r1=(%g %g %g %g) r2=(%g %g %g %g)\n"
                    "[bone2]   in  r0=(%g %g %g %g) r1=(%g %g %g %g) r2=(%g %g %g %g)\n",
                    startPC, triples[t].name, static_cast<unsigned long long>(n),
                    static_cast<unsigned long long>(eeScheduler().currentVSyncTick()), outC, ro, so, inC, ri, si,
                    vf[f][0], vf[f][1], vf[f][2], vf[f][3], vf[f + 1][0], vf[f + 1][1], vf[f + 1][2], vf[f + 1][3],
                    vf[f + 2][0], vf[f + 2][1], vf[f + 2][2], vf[f + 2][3],
                    bcInExt[f][0], bcInExt[f][1], bcInExt[f][2], bcInExt[f][3],
                    bcInExt[f + 1][0], bcInExt[f + 1][1], bcInExt[f + 1][2], bcInExt[f + 1][3],
                    bcInExt[f + 2][0], bcInExt[f + 2][1], bcInExt[f + 2][2], bcInExt[f + 2][3]);
            }
        }
        if ((n % 200000u) == 0u)
            std::fprintf(stderr, "[bone2] vcallms=0x%x calls=%llu badTriples=%llu\n",
                         startPC, static_cast<unsigned long long>(n), static_cast<unsigned long long>(bad.load()));
    }
    // Diagnostic only: the rotation checker cannot detect a bad translation.
    if (boneCheckHere || boneCheckExt)
    {
        const float (*before)[4] = boneCheckHere ? bcIn : bcInExt;
        const auto &after = m_vu0.state().vf;
        static uint64_t calls[2]{}, bad[2]{}, printed[2]{};
        static double peak[2]{};
        const unsigned kind = boneCheckExt ? 1u : 0u;
        ++calls[kind];
        double magnitude = 0.0;
        bool invalid = false;
        const int rows[] = {5, 17, 23, 31};
        for (int row : rows) for (int lane = 0; lane < 3; ++lane)
        {
            // Only the output translation registers count as a trigger;
            // other registers can contain unrelated intermediate values.
            if (row != 31 && !(boneCheckExt && row == 23)) continue;
            const double v = after[row][lane];
            invalid |= !std::isfinite(v);
            magnitude = std::max(magnitude, std::fabs(v));
        }
        peak[kind] = std::max(peak[kind], magnitude);
        if (invalid || magnitude > 100000.0)
        {
            ++bad[kind];
            if (printed[kind]++ < 8)
            {
                std::fprintf(stderr, "[bone:translation] micro=%x guest=%x call=%llu tick=%llu magnitude=%g t8=%x t9=%x s0=%x\n",
                    startPC, ctx->pc, calls[kind], static_cast<unsigned long long>(eeScheduler().currentVSyncTick()), magnitude,
                    GPR_U32(ctx,24), GPR_U32(ctx,25), GPR_U32(ctx,16));
                for (int reg=0; reg<32; ++reg)
                    std::fprintf(stderr,"[bone:translation] vf%d before=(%g,%g,%g,%g) after=(%g,%g,%g,%g)\n",reg,
                        before[reg][0],before[reg][1],before[reg][2],before[reg][3],
                        after[reg][0],after[reg][1],after[reg][2],after[reg][3]);
            }
        }
        if (calls[kind] == 1 || calls[kind] % 100000 == 0)
            std::fprintf(stderr,"[bone:translation-summary] micro=%x calls=%llu large=%llu peak=%g\n",startPC,calls[kind],bad[kind],peak[kind]);
    }
    copyVu0StateToContext(m_vu0.state(), ctx);
}

void PS2Runtime::vu0StartMicroProgram(uint8_t *rdram, R5900Context *ctx, uint32_t address)
{
    // VCALLMS and VCALLMSR both route here.
    executeVU0Microprogram(rdram, ctx, address);
}

void PS2Runtime::handleSyscall(uint8_t *rdram, R5900Context *ctx)
{
    handleSyscall(rdram, ctx, 0);
}

void PS2Runtime::handleSyscall(uint8_t *rdram, R5900Context *ctx, uint32_t encodedSyscallId)
{
    if (ctx->in_delay_slot)
    {
        throw std::runtime_error("Attempted to execute a syscall inside a branch delay slot! "
                                 "This breaks the atomic basic block model and is structurally unsupported by the emulator.");
    }

    const uint32_t syscallId = (encodedSyscallId != 0u)
                                   ? encodedSyscallId
                                   : getRegU32(ctx, 3); // $v1 / $3 is the EE kernel syscall number

    if (ps2_syscalls::dispatchNumericSyscall(syscallId, rdram, ctx, this))
    {
        return;
    }

    // God help you
    ps2_syscalls::TODO(rdram, ctx, this, encodedSyscallId);
}

void PS2Runtime::handleBreak(uint8_t *rdram, R5900Context *ctx)
{
    raiseCop0Exception(ctx, EXCEPTION_BREAKPOINT);
}

void PS2Runtime::drainCompletedDmacHandlers(uint8_t *rdram)
{
    for (uint32_t cause : m_memory.consumeCompletedDmacCauses())
    {
        ps2_syscalls::dispatchDmacHandlersForCause(rdram, this, cause);
    }
}

void PS2Runtime::handleTrap(uint8_t *rdram, R5900Context *ctx)
{
    raiseCop0Exception(ctx, EXCEPTION_TRAP);
}

void PS2Runtime::handleTLBR(uint8_t *rdram, R5900Context *ctx)
{
    uint32_t vpn = 0;
    uint32_t pfn = 0;
    uint32_t mask = 0;
    bool valid = false;

    const uint32_t index = ctx->cop0_index & 0x3Fu;
    if (!m_memory.tlbRead(index, vpn, pfn, mask, valid))
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
        return;
    }

    // Preserve low ASID bits in EntryHi.
    ctx->cop0_entryhi = (ctx->cop0_entryhi & 0x00000FFFu) | (vpn & 0xFFFFF000u);
    ctx->cop0_entrylo0 = (ctx->cop0_entrylo0 & ~0x03FFFFC2u) |
                         ((pfn & 0x000FFFFFu) << 6) |
                         (valid ? 0x2u : 0u);
    ctx->cop0_pagemask = mask & 0x01FFE000u;
}

void PS2Runtime::handleTLBWI(uint8_t *rdram, R5900Context *ctx)
{
    const uint32_t index = ctx->cop0_index & 0x3Fu;
    const uint32_t vpn = ctx->cop0_entryhi & 0xFFFFF000u;
    const uint32_t pfn = (ctx->cop0_entrylo0 >> 6) & 0x000FFFFFu;
    const uint32_t mask = ctx->cop0_pagemask & 0x01FFE000u;
    const bool valid = (ctx->cop0_entrylo0 & 0x2u) != 0u;

    if (!m_memory.tlbWrite(index, vpn, pfn, mask, valid))
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
    }
}

void PS2Runtime::handleTLBWR(uint8_t *rdram, R5900Context *ctx)
{
    const uint32_t entryCount = static_cast<uint32_t>(m_memory.tlbEntryCount());
    if (entryCount == 0)
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
        return;
    }

    const uint32_t wired = std::min(ctx->cop0_wired, entryCount - 1);
    uint32_t random = ctx->cop0_random % entryCount;
    if (random < wired)
    {
        random = wired;
    }

    const uint32_t vpn = ctx->cop0_entryhi & 0xFFFFF000u;
    const uint32_t pfn = (ctx->cop0_entrylo0 >> 6) & 0x000FFFFFu;
    const uint32_t mask = ctx->cop0_pagemask & 0x01FFE000u;
    const bool valid = (ctx->cop0_entrylo0 & 0x2u) != 0u;

    if (!m_memory.tlbWrite(random, vpn, pfn, mask, valid))
    {
        raiseCop0Exception(ctx, EXCEPTION_RESERVED_INSTRUCTION);
        return;
    }

    // Keep COP0 bookkeeping in sync with the selected slot.
    ctx->cop0_index = (ctx->cop0_index & ~0x3Fu) | (random & 0x3Fu);
    ctx->cop0_random = (random <= wired) ? (entryCount - 1) : (random - 1);
}

void PS2Runtime::handleTLBP(uint8_t *rdram, R5900Context *ctx)
{
    const int32_t index = m_memory.tlbProbe(ctx->cop0_entryhi & 0xFFFFF000u);
    if (index >= 0)
    {
        ctx->cop0_index = (ctx->cop0_index & ~0x8000003Fu) |
                          (static_cast<uint32_t>(index) & 0x3Fu);
    }
    else
    {
        // MIPS sets probe failure bit (P) in Index[31].
        ctx->cop0_index |= 0x80000000u;
    }
}

void PS2Runtime::clearLLBit(R5900Context *ctx)
{
    // LL/SC reservation is tracked separately from COP0 Status.
    ctx->llbit = 0;
    ctx->lladdr = 0;
}

uint32_t PS2Runtime::alignGuestHeapValue(uint32_t value, uint32_t alignment)
{
    if (alignment == 0)
    {
        return value;
    }

    const uint32_t mask = alignment - 1u;
    if (value > (std::numeric_limits<uint32_t>::max() - mask))
    {
        return std::numeric_limits<uint32_t>::max();
    }
    return (value + mask) & ~mask;
}

bool PS2Runtime::isGuestHeapAlignmentValid(uint32_t alignment)
{
    return alignment != 0u && (alignment & (alignment - 1u)) == 0u;
}

uint32_t PS2Runtime::normalizeGuestHeapAlignment(uint32_t alignment)
{
    if (!isGuestHeapAlignmentValid(alignment))
    {
        return kGuestHeapDefaultAlignment;
    }
    return std::max(alignment, kGuestHeapDefaultAlignment);
}

uint32_t PS2Runtime::clampGuestHeapBase(uint32_t guestBase) const
{
    uint32_t normalized = guestBase;
    if (normalized >= PS2_RAM_SIZE)
    {
        normalized &= PS2_RAM_MASK;
    }
    const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    return std::min(normalized, hardLimit);
}

uint32_t PS2Runtime::clampGuestHeapLimit(uint32_t guestLimit) const
{
    const uint32_t hardLimit = std::min(kGuestHeapHardLimit, PS2_RAM_SIZE);
    if (guestLimit == 0u || guestLimit > hardLimit)
    {
        return hardLimit;
    }
    return guestLimit;
}

void PS2Runtime::resetGuestHeapLocked(uint32_t guestBase, uint32_t guestLimit)
{
    uint32_t base = alignGuestHeapValue(clampGuestHeapBase(guestBase), kGuestHeapDefaultAlignment);
    uint32_t limit = clampGuestHeapLimit(guestLimit);
    if (base == 0u)
    {
        const uint32_t fallbackBase = (m_guestHeapSuggestedBase != 0u) ? m_guestHeapSuggestedBase : kGuestHeapDefaultBase;
        base = alignGuestHeapValue(clampGuestHeapBase(fallbackBase), kGuestHeapDefaultAlignment);
    }

    if (limit <= base)
    {
        base = alignGuestHeapValue(clampGuestHeapBase(m_guestHeapSuggestedBase), kGuestHeapDefaultAlignment);
        limit = clampGuestHeapLimit(0u);
    }

    if (limit <= base)
    {
        base = 0u;
        limit = 0u;
    }

    m_guestHeapBlocks.clear();
    if (limit > base)
    {
        m_guestHeapBlocks.push_back({base, limit - base, true});
    }

    m_guestHeapBase = base;
    m_guestHeapEnd = base;
    m_guestHeapLimit = limit;
    m_guestHeapConfigured = true;
}

void PS2Runtime::ensureGuestHeapInitializedLocked()
{
    if (m_guestHeapConfigured)
    {
        return;
    }

    const uint32_t suggested = (m_guestHeapSuggestedBase == 0u) ? kGuestHeapDefaultBase : m_guestHeapSuggestedBase;
    resetGuestHeapLocked(suggested, clampGuestHeapLimit(0u));
}

int32_t PS2Runtime::findGuestHeapBlockIndexLocked(uint32_t guestAddr) const
{
    const uint32_t normalizedAddr = guestAddr & PS2_RAM_MASK;
    for (size_t i = 0; i < m_guestHeapBlocks.size(); ++i)
    {
        const GuestHeapBlock &block = m_guestHeapBlocks[i];
        if (!block.free && block.addr == normalizedAddr)
        {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

uint32_t PS2Runtime::allocateGuestBlockLocked(uint32_t size, uint32_t alignment)
{
    if (size == 0u)
    {
        return 0u;
    }

    const uint32_t normalizedAlignment = normalizeGuestHeapAlignment(alignment);
    if (size > (std::numeric_limits<uint32_t>::max() - (kGuestHeapDefaultAlignment - 1u)))
    {
        return 0u;
    }

    const uint32_t allocSize = alignGuestHeapValue(size, kGuestHeapDefaultAlignment);
    if (allocSize == 0u)
    {
        return 0u;
    }

    for (size_t i = 0; i < m_guestHeapBlocks.size(); ++i)
    {
        const GuestHeapBlock block = m_guestHeapBlocks[i];
        if (!block.free)
        {
            continue;
        }

        const uint64_t blockStart = block.addr;
        const uint64_t blockEnd = blockStart + static_cast<uint64_t>(block.size);
        const uint32_t alignedAddr = alignGuestHeapValue(block.addr, normalizedAlignment);
        if (alignedAddr < block.addr)
        {
            continue;
        }

        const uint64_t alignedStart = alignedAddr;
        if (alignedStart > blockEnd)
        {
            continue;
        }

        const uint64_t allocEnd = alignedStart + static_cast<uint64_t>(allocSize);
        if (allocEnd > blockEnd)
        {
            continue;
        }

        const uint32_t prefixSize = static_cast<uint32_t>(alignedStart - blockStart);
        const uint32_t suffixSize = static_cast<uint32_t>(blockEnd - allocEnd);

        std::vector<GuestHeapBlock> replacement;
        replacement.reserve(3);
        if (prefixSize > 0u)
        {
            replacement.push_back({block.addr, prefixSize, true});
        }
        replacement.push_back({alignedAddr, allocSize, false});
        if (suffixSize > 0u)
        {
            replacement.push_back({static_cast<uint32_t>(allocEnd), suffixSize, true});
        }

        m_guestHeapBlocks.erase(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(i));
        m_guestHeapBlocks.insert(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(i),
                                 replacement.begin(),
                                 replacement.end());

        m_guestHeapEnd = std::max(m_guestHeapEnd, static_cast<uint32_t>(allocEnd));
        return alignedAddr;
    }

    return 0u;
}

void PS2Runtime::coalesceGuestHeapLocked()
{
    if (m_guestHeapBlocks.empty())
    {
        return;
    }

    size_t i = 1;
    while (i < m_guestHeapBlocks.size())
    {
        GuestHeapBlock &prev = m_guestHeapBlocks[i - 1];
        GuestHeapBlock &curr = m_guestHeapBlocks[i];
        const uint64_t prevEnd = static_cast<uint64_t>(prev.addr) + static_cast<uint64_t>(prev.size);
        if (prev.free && curr.free && prevEnd == curr.addr)
        {
            prev.size += curr.size;
            m_guestHeapBlocks.erase(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(i));
            continue;
        }
        ++i;
    }
}

void PS2Runtime::freeGuestBlockLocked(uint32_t guestAddr)
{
    const int32_t index = findGuestHeapBlockIndexLocked(guestAddr);
    if (index < 0)
    {
        return;
    }

    m_guestHeapBlocks[static_cast<size_t>(index)].free = true;
    coalesceGuestHeapLocked();
}

void PS2Runtime::configureGuestHeap(uint32_t guestBase, uint32_t guestLimit)
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    uint32_t normalizedBase = alignGuestHeapValue(clampGuestHeapBase(guestBase), kGuestHeapDefaultAlignment);
    if (normalizedBase == 0u)
    {
        normalizedBase = (m_guestHeapSuggestedBase != 0u) ? m_guestHeapSuggestedBase : kGuestHeapDefaultBase;
    }
    m_guestHeapSuggestedBase = normalizedBase;
    resetGuestHeapLocked(normalizedBase, guestLimit);
}

uint32_t PS2Runtime::guestMalloc(uint32_t size, uint32_t alignment)
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    ensureGuestHeapInitializedLocked();
    return allocateGuestBlockLocked(size, alignment);
}

uint32_t PS2Runtime::guestCalloc(uint32_t count, uint32_t size, uint32_t alignment)
{
    if (count == 0u || size == 0u)
    {
        return 0u;
    }
    if (count > (std::numeric_limits<uint32_t>::max() / size))
    {
        return 0u;
    }

    const uint32_t totalSize = count * size;
    const uint32_t guestAddr = guestMalloc(totalSize, alignment);
    if (guestAddr != 0u)
    {
        uint8_t *rdram = m_memory.getRDRAM();
        if (rdram)
        {
            uint32_t physAddr = guestAddr & PS2_RAM_MASK;
            if (physAddr + totalSize <= PS2_RAM_SIZE)
                std::memset(rdram + physAddr, 0, totalSize);
        }
    }

    return guestAddr;
}

uint32_t PS2Runtime::guestRealloc(uint32_t guestAddr, uint32_t newSize, uint32_t alignment)
{
    if (guestAddr == 0u)
    {
        return guestMalloc(newSize, alignment);
    }
    if (newSize == 0u)
    {
        guestFree(guestAddr);
        return 0u;
    }

    if (newSize > (std::numeric_limits<uint32_t>::max() - (kGuestHeapDefaultAlignment - 1u)))
    {
        return 0u;
    }

    const uint32_t normalizedAlignment = normalizeGuestHeapAlignment(alignment);
    const uint32_t requestedSize = alignGuestHeapValue(newSize, kGuestHeapDefaultAlignment);

    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    ensureGuestHeapInitializedLocked();

    const int32_t index = findGuestHeapBlockIndexLocked(guestAddr);
    if (index < 0)
    {
        return 0u;
    }

    const size_t blockIndex = static_cast<size_t>(index);
    const uint32_t oldAddr = m_guestHeapBlocks[blockIndex].addr;
    const uint32_t oldSize = m_guestHeapBlocks[blockIndex].size;

    if (requestedSize <= oldSize)
    {
        if (requestedSize < oldSize)
        {
            const uint32_t tailAddr = oldAddr + requestedSize;
            const uint32_t tailSize = oldSize - requestedSize;
            m_guestHeapBlocks[blockIndex].size = requestedSize;
            m_guestHeapBlocks.insert(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(blockIndex + 1u),
                                     GuestHeapBlock{tailAddr, tailSize, true});
            coalesceGuestHeapLocked();
        }
        return oldAddr;
    }

    if (blockIndex + 1u < m_guestHeapBlocks.size())
    {
        GuestHeapBlock &next = m_guestHeapBlocks[blockIndex + 1u];
        const uint64_t blockEnd = static_cast<uint64_t>(m_guestHeapBlocks[blockIndex].addr) +
                                  static_cast<uint64_t>(m_guestHeapBlocks[blockIndex].size);
        if (next.free && blockEnd == next.addr)
        {
            const uint64_t combined = static_cast<uint64_t>(m_guestHeapBlocks[blockIndex].size) +
                                      static_cast<uint64_t>(next.size);
            if (combined >= requestedSize)
            {
                const uint32_t extraNeeded = requestedSize - m_guestHeapBlocks[blockIndex].size;
                m_guestHeapBlocks[blockIndex].size = requestedSize;
                if (next.size == extraNeeded)
                {
                    m_guestHeapBlocks.erase(m_guestHeapBlocks.begin() + static_cast<std::ptrdiff_t>(blockIndex + 1u));
                }
                else
                {
                    next.addr += extraNeeded;
                    next.size -= extraNeeded;
                }
                m_guestHeapEnd = std::max(m_guestHeapEnd, oldAddr + requestedSize);
                return oldAddr;
            }
        }
    }

    const uint32_t newAddr = allocateGuestBlockLocked(newSize, normalizedAlignment);
    if (newAddr == 0u)
    {
        return 0u;
    }

    uint8_t *rdram = m_memory.getRDRAM();
    if (rdram)
    {
        const uint32_t copyBytes = std::min(oldSize, newSize);
        uint32_t dstPhys = newAddr & PS2_RAM_MASK;
        uint32_t srcPhys = oldAddr & PS2_RAM_MASK;
        if (dstPhys + copyBytes <= PS2_RAM_SIZE && srcPhys + copyBytes <= PS2_RAM_SIZE)
            std::memmove(rdram + dstPhys, rdram + srcPhys, copyBytes);
    }

    freeGuestBlockLocked(oldAddr);
    return newAddr;
}

void PS2Runtime::guestFree(uint32_t guestAddr)
{
    if (guestAddr == 0u)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    ensureGuestHeapInitializedLocked();
    freeGuestBlockLocked(guestAddr);
}

uint32_t PS2Runtime::guestHeapBase() const
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    return m_guestHeapConfigured ? m_guestHeapBase : m_guestHeapSuggestedBase;
}

uint32_t PS2Runtime::guestHeapEnd() const
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    return m_guestHeapConfigured ? m_guestHeapEnd : m_guestHeapSuggestedBase;
}

uint32_t PS2Runtime::guestHeapLimit() const
{
    std::lock_guard<std::mutex> lock(m_guestHeapMutex);
    return m_guestHeapConfigured ? m_guestHeapLimit : m_guestHeapSuggestedBase;
}

uint32_t PS2Runtime::reserveAsyncCallbackStack(uint32_t size, uint32_t alignment)
{
    if (size == 0u)
    {
        return 0u;
    }

    const uint32_t normalizedAlignment = normalizeGuestHeapAlignment(alignment);
    const uint32_t allocSize = alignGuestHeapValue(size, kGuestHeapDefaultAlignment);
    if (allocSize == 0u)
    {
        return 0u;
    }

    std::lock_guard<std::mutex> lock(m_asyncCallbackStackMutex);
    uint32_t top = m_asyncCallbackStackTop;
    if (top > PS2_RAM_SIZE)
    {
        top = PS2_RAM_SIZE;
    }
    top &= ~(kGuestHeapDefaultAlignment - 1u);

    if (top <= allocSize)
    {
        return 0u;
    }

    uint32_t base = top - allocSize;
    base &= ~(normalizedAlignment - 1u);
    if (base < m_asyncCallbackStackFloor || base >= top)
    {
        return 0u;
    }

    m_asyncCallbackStackTop = base;
    return top - 0x10u;
}

uint8_t PS2Runtime::Load8(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read8(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

uint16_t PS2Runtime::Load16(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read16(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

uint32_t PS2Runtime::Load32(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read32(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

uint64_t PS2Runtime::Load64(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read64(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return 0;
    }
}

__m128i PS2Runtime::Load128(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr)
{
    try
    {
        return m_memory.read128(vaddr);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_LOAD);
        return _mm_setzero_si128();
    }
}

void PS2Runtime::Store8(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint8_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 1u, value, 0u, "WRITE8", ctx);
    try
    {
        m_memory.write8(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store16(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint16_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 2u, value, 0u, "WRITE16", ctx);
    try
    {
        m_memory.write16(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store32(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint32_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 4u, value, 0u, "WRITE32", ctx);
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    // Who kicks VIF1 (0x10009000) chain DMA, and how often? (settled menu only)
    if ((vaddr & 0x1FFFFFFFu) == 0x10009000u && (value & 0x100u) != 0u &&
        m_memory.read32(0x6F7E8Cu) == 0u && m_memory.read32(0x6F8464u) == 1u)
    {
        static std::atomic<uint32_t> s_k{0u};
        if (s_k.fetch_add(1u, std::memory_order_relaxed) < 300u)
            std::fprintf(stderr, "[vif1:kick] chcr=0x%x pc=0x%x ra=0x%x\n",
                         value, ctx->pc, getRegU32(ctx, 31));
    }
#endif
    try
    {
        m_memory.write32(vaddr, value);
        drainCompletedDmacHandlers(rdram);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store64(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, uint64_t value)
{
    ps2TraceGuestWrite(rdram, vaddr, 8u, value, 0u, "WRITE64", ctx);
    try
    {
        m_memory.write64(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::Store128(uint8_t *rdram, R5900Context *ctx, uint32_t vaddr, __m128i value)
{
    alignas(16) uint64_t _parts[2];
    _mm_storeu_si128(reinterpret_cast<__m128i *>(_parts), value);
    ps2TraceGuestWrite(rdram, vaddr, 16u, _parts[0], _parts[1], "WRITE128", ctx);
    try
    {
        m_memory.write128(vaddr, value);
    }
    catch (const std::exception &)
    {
        SignalException(ctx, EXCEPTION_ADDRESS_ERROR_STORE);
    }
}

void PS2Runtime::kickGifDmaChainFromMMIO(uint8_t *rdram,
                                         R5900Context *ctx,
                                         uint32_t dPcrValue,
                                         uint32_t dStatValue,
                                         uint32_t tadr,
                                         uint32_t chcr)
{
    constexpr uint32_t D_PCR = 0x1000E020u;
    constexpr uint32_t D_STAT = 0x1000E010u;
    constexpr uint32_t GIF_TADR = 0x1000A030u;
    constexpr uint32_t GIF_CHCR = 0x1000A000u;

    ps2TraceGuestWrite(rdram, D_PCR, 4u, dPcrValue, 0u, "WRITE32", ctx);
    m_memory.writeIORegister(D_PCR, dPcrValue);
    ps2TraceGuestWrite(rdram, D_STAT, 4u, dStatValue, 0u, "WRITE32", ctx);
    m_memory.writeIORegister(D_STAT, dStatValue);
    ps2TraceGuestWrite(rdram, GIF_TADR, 4u, tadr, 0u, "WRITE32", ctx);
    m_memory.writeIORegister(GIF_TADR, tadr);
    ps2TraceGuestWrite(rdram, GIF_CHCR, 4u, chcr, 0u, "WRITE32", ctx);
    if (m_memory.tryProcessNativeGifImageUploadChain(m_gs, tadr, chcr))
    {
        drainCompletedDmacHandlers(rdram);
        return;
    }
    if (m_memory.tryProcessNativeGifPackedChain(m_gs, tadr, chcr))
    {
        drainCompletedDmacHandlers(rdram);
        return;
    }
    m_memory.writeIORegister(GIF_CHCR, chcr);
    m_memory.processPendingTransfers();
    drainCompletedDmacHandlers(rdram);
}

void PS2Runtime::requestStop()
{
    m_stopRequested.store(true, std::memory_order_relaxed);
    if (m_eeScheduler)
    {
        m_eeScheduler->requestStop();
    }
}

bool PS2Runtime::isStopRequested() const
{
    return m_stopRequested.load(std::memory_order_relaxed);
}

EeScheduler &PS2Runtime::eeScheduler()
{
    return *m_eeScheduler;
}

const EeScheduler &PS2Runtime::eeScheduler() const
{
    return *m_eeScheduler;
}

void PS2Runtime::postEeEvent(EeEvent event)
{
    m_eeScheduler->postEvent(event);
}

bool PS2Runtime::eeCheckpointDue(uint32_t cycles) noexcept
{
    return m_eeScheduler->checkpointDue(cycles);
}

[[noreturn]] void PS2Runtime::eeWaitVSyncTicks(uint32_t ticks, uint32_t resumePc)
{
    const uint64_t currentTick = m_eeScheduler->currentVSyncTick();
    const uint64_t waitTicks = std::max<uint64_t>(1u, ticks);
    m_eeScheduler->waitVSync(currentTick + waitTicks - 1u,
                             0,
                             [resumePc](R5900Context &context)
                             {
                                 context.pc = resumePc;
                             });
}

void PS2Runtime::addEeExitHandler(int threadId, uint32_t function, uint32_t argument)
{
    std::lock_guard lock(m_eeKernelStateMutex);
    m_eeExitHandlers[threadId].push_back({function, argument});
}

std::vector<PS2Runtime::EeExitHandlerRegistration> PS2Runtime::takeEeExitHandlers(int threadId)
{
    std::lock_guard lock(m_eeKernelStateMutex);
    auto it = m_eeExitHandlers.find(threadId);
    if (it == m_eeExitHandlers.end())
    {
        return {};
    }
    auto handlers = std::move(it->second);
    m_eeExitHandlers.erase(it);
    return handlers;
}

void PS2Runtime::removeEeExitHandlers(int threadId)
{
    std::lock_guard lock(m_eeKernelStateMutex);
    m_eeExitHandlers.erase(threadId);
}

bool PS2Runtime::findEeSyscallOverride(uint32_t syscallNumber, uint32_t &handler) const
{
    std::lock_guard lock(m_eeKernelStateMutex);
    const auto it = m_eeSyscallOverrides.find(syscallNumber);
    if (it == m_eeSyscallOverrides.end())
    {
        return false;
    }
    handler = it->second;
    return true;
}

void PS2Runtime::setEeSyscallOverride(uint8_t *rdram, uint32_t syscallNumber, uint32_t handler)
{
    constexpr uint32_t kTableBase = 0x80011F80u & 0x1FFFFFFFu;
    constexpr uint32_t kMirrorLimit = 0x00080000u;
    const int64_t offset = static_cast<int64_t>(static_cast<int32_t>(syscallNumber)) * 4;
    const int64_t address = static_cast<int64_t>(kTableBase) + offset;

    std::lock_guard lock(m_eeKernelStateMutex);
    if (handler == 0u)
    {
        m_eeSyscallOverrides.erase(syscallNumber);
    }
    else
    {
        m_eeSyscallOverrides[syscallNumber] = handler;
    }
    if (!rdram || address < 0 || address + 4 > kMirrorLimit)
    {
        return;
    }
    const uint32_t guestAddress = static_cast<uint32_t>(address);
    std::memcpy(rdram + guestAddress, &handler, sizeof(handler));
    if (handler == 0u)
    {
        m_eeSyscallMirrorAddresses.erase(guestAddress);
    }
    else
    {
        m_eeSyscallMirrorAddresses.insert(guestAddress);
    }
}

void PS2Runtime::initializeEeKernelState(uint8_t *rdram)
{
    if (!rdram)
    {
        return;
    }
    constexpr uint32_t kTableGuestBase = 0x80011F80u;
    constexpr uint32_t kTableBase = kTableGuestBase & 0x1FFFFFFFu;
    constexpr uint32_t kMirrorLimit = 0x00080000u;
    constexpr uint32_t kProbeBase = 0x000002F0u;

    std::lock_guard lock(m_eeKernelStateMutex);
    for (const uint32_t address : m_eeSyscallMirrorAddresses)
    {
        const uint32_t zero = 0u;
        std::memcpy(rdram + address, &zero, sizeof(zero));
    }
    m_eeSyscallMirrorAddresses.clear();
    const uint32_t high = kTableGuestBase >> 16;
    const uint32_t low = kTableGuestBase & 0xFFFFu;
    std::memcpy(rdram + kProbeBase, &high, sizeof(high));
    std::memcpy(rdram + kProbeBase + 8u, &low, sizeof(low));
    m_eeSyscallMirrorAddresses.insert(kProbeBase);
    m_eeSyscallMirrorAddresses.insert(kProbeBase + 8u);

    for (const auto &[syscallNumber, handler] : m_eeSyscallOverrides)
    {
        const int64_t offset = static_cast<int64_t>(static_cast<int32_t>(syscallNumber)) * 4;
        const int64_t address = static_cast<int64_t>(kTableBase) + offset;
        if (address < 0 || address + 4 > kMirrorLimit)
        {
            continue;
        }
        const uint32_t guestAddress = static_cast<uint32_t>(address);
        std::memcpy(rdram + guestAddress, &handler, sizeof(handler));
        m_eeSyscallMirrorAddresses.insert(guestAddress);
    }
}

void PS2Runtime::HandleIntegerOverflow(R5900Context *ctx)
{
    raiseCop0Exception(ctx, EXCEPTION_INTEGER_OVERFLOW);
}

void PS2Runtime::run()
{
    m_stopRequested.store(false, std::memory_order_relaxed);
    ps2_stubs::resetSifState();
    resetIop();
    ps2_stubs::resetAudioStubState();
    ps2_stubs::resetMpegStubState();
    initializeEeKernelState(m_memory.getRDRAM());
    m_cpuContext.r[4] = _mm_setzero_si128();
    m_cpuContext.r[5] = _mm_setzero_si128();
    m_cpuContext.r[29] = _mm_set_epi64x(0, static_cast<int64_t>(PS2_RAM_SIZE - 0x10u));
    m_debugPc.store(m_cpuContext.pc, std::memory_order_relaxed);
    m_debugRa.store(static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[31], 0)), std::memory_order_relaxed);
    m_debugSp.store(static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[29], 0)), std::memory_order_relaxed);
    m_debugGp.store(static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[28], 0)), std::memory_order_relaxed);

    RUNTIME_LOG("Starting execution at address 0x" << std::hex << m_cpuContext.pc << std::dec);

    // A blank image to use as a framebuffer
    Image blank = GenImageColor(FB_WIDTH, FB_HEIGHT, BLANK);
    Texture2D frameTex = LoadTextureFromImage(blank);
    UnloadImage(blank);
    // Filter only presentation: leave GS alpha, depth and render targets intact.
    Shader aaShader{};
    if (const char* aa = std::getenv("PS2X_AA"))
    {
        if (std::strcmp(aa, "fxaa") == 0)
        {
            aaShader = LoadShader(nullptr, "launcher/fxaa.fs");
            if (aaShader.id == rlGetShaderIdDefault()) aaShader = Shader{};
            std::fprintf(stderr, "[display] FXAA %s\n", aaShader.id ? "enabled" : "unavailable; using unfiltered output");
        }
    }
    const int aaStep = aaShader.id ? GetShaderLocation(aaShader, "pixelStep") : -1;
    const int aaBounds = aaShader.id ? GetShaderLocation(aaShader, "sampleBounds") : -1;


    std::atomic<bool> gameThreadFinished{false};
    extern void ps2xSamplerRegisterGameThread(); // ps2_sampler.cpp (PS2X_SAMPLE=1)

    std::thread gameThread([&]()
                           {
        ThreadNaming::SetCurrentThreadName("GameThread");
        ps2xSamplerRegisterGameThread();
        try
        {
            m_eeScheduler->reset(m_memory.getRDRAM(), m_cpuContext);
            m_eeScheduler->run();
            uint32_t pc = m_debugPc.load(std::memory_order_relaxed);
            RUNTIME_LOG("Game thread returned. PC=0x" << std::hex << pc
                      << " RA=0x" << static_cast<uint32_t>(_mm_extract_epi32(m_cpuContext.r[31], 0)) << std::dec << std::endl);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error during program execution: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "Error during program execution: unknown exception" << std::endl;
        }
        gameThreadFinished.store(true, std::memory_order_release); });

    uint64_t tick = 0;
    while (!isStopRequested() && !gameThreadFinished.load(std::memory_order_acquire))
    {
        PS2_IF_AGRESSIVE_LOGS({
            tick++;
            if ((tick % 120) == 0)
            {
                uint64_t curDma = m_memory.dmaStartCount();
                uint64_t curGif = m_memory.gifCopyCount();
                uint64_t curGs = m_memory.gsWriteCount();
                uint64_t curVif = m_memory.vifWriteCount();
                const GSRegisters &gs = m_memory.gs();
                const uint32_t dbgPc = m_debugPc.load(std::memory_order_relaxed);
                const uint32_t dbgRa = m_debugRa.load(std::memory_order_relaxed);
                const uint32_t dbgSp = m_debugSp.load(std::memory_order_relaxed);
                const uint32_t dbgGp = m_debugGp.load(std::memory_order_relaxed);
                const auto eeSnapshot = m_eeScheduler->snapshot();

                // WotM shell-menu state probe (gp=0x6ff8f0). See userintMain__Fv.
                const uint32_t wotmCurScreen   = m_memory.read32(0x6F8464u); // currScreen (gp-0x748c)
                const uint32_t wotmBetween     = m_memory.read32(0x6F7E8Cu); // betweenScreens
                const uint32_t wotmUiRet       = m_memory.read32(0x0042FF10u); // gUserintReturn (s2-0xf0)
                const uint32_t wotmInitAnim    = m_memory.read32(0x6F84A0u); // initialAnimation (gp-0x7450)
                const uint32_t wotmContDec     = m_memory.read32(0x6F8458u); // continueDecoding (gp-0x7498)
                const uint32_t wotmFadeOff     = m_memory.read32(0x6F84D4u); // g_fadeFrameOffset (gp-0x741c)
                const uint32_t wotmGFrame      = m_memory.read32(0x6F7E38u); // g_frame (gp-0x7ab8)
                // screenTransition__Fb (0x19e698) stuck-transition probe.
                const uint32_t wotmLastScreen  = m_memory.read32(0x6F8468u); // lastScreen (0x700000-0x7b98)
                const uint32_t wotmCurAnimIdx  = m_memory.read32(0x6F7E88u); // currAnimationIndex (gp-0x7a68)
                const uint32_t wotmWaitAnim    = m_memory.read32(0x6F8074u); // waitingForAnimation_2977 (gp-0x787c)
                const uint32_t wotmWaitFade    = m_memory.read32(0x6F8078u); // waitingForAlphaFade_2978 (gp-0x7878)
                const uint32_t wotmTgtAlphaB   = m_memory.read32(0x6F7E74u); // targetAlpha (gp-0x7a7c), float bits
                const uint32_t wotmFadingOut   = m_memory.read32(0x6F8080u); // fadingOut_2980 (gp-0x7870)  [approx]
                const uint32_t wotmCs800bd0C   = m_memory.read32(0x00800BDCu) & 0xFFu; // cs 0x800bd0 [+0xC] drawMe byte
                float wotmTgtAlpha; std::memcpy(&wotmTgtAlpha, &wotmTgtAlphaB, 4);

                // [viewmat] A/B vs retail: the resident screen/viewProj matrices
                // that feed VU1 q9-q12. Retail's dmaScreenMat carries camera+proj;
                // ours appears to be a bare viewport. Dump the 4 matrices that
                // viewUpdate__Fi -> viewSetVUPacketMat__FPA3_fN30i produce.
                if (wotmCurScreen == 1u && wotmBetween == 0u)
                {
                    static std::atomic<uint32_t> s_vm{0u};
                    if (s_vm.fetch_add(1u, std::memory_order_relaxed) < 6u)
                    {
                        auto dumpMat = [&](const char *name, uint32_t addr) {
                            float m[16];
                            for (int k = 0; k < 16; ++k) {
                                uint32_t b = m_memory.read32(addr + (uint32_t)k * 4u);
                                std::memcpy(&m[k], &b, 4);
                            }
                            std::fprintf(stderr,
                                "[viewmat] %s @0x%x = "
                                "[(%g,%g,%g,%g)(%g,%g,%g,%g)(%g,%g,%g,%g)(%g,%g,%g,%g)]\n",
                                name, addr,
                                m[0],m[1],m[2],m[3], m[4],m[5],m[6],m[7],
                                m[8],m[9],m[10],m[11], m[12],m[13],m[14],m[15]);
                        };
                        dumpMat("dmaScreenMat",   0x0025B1A8u);
                        dumpMat("worldToScreen",  0x006E1FD0u);
                        dumpMat("viewScreenMats", 0x006E1990u);
                        dumpMat("viewClipMats",   0x006E1AD0u);
                        dumpMat("cameraViewMat",  0x00445D00u);
                    }
                }

                // [ngpreloc] one-shot: dump NGP load bases + scan RRAM for the
                // menu GIF-primitive TEMPLATE qword. Retail: templates stay
                // `04 80 00 00 00 40 3e 30 12 04 00 00` (NGP loads @0xa00000 so
                // dbsRelocateViaPtrListFile delta==0). If ours are already
                // `01 80 00 00 00 40 3b 60 ...` here -> the reloc pass corrupted
                // them at load. Fires every 60 ticks up to 6x so we catch it
                // both early (post-load) and settled.
                {
                    static std::atomic<uint32_t> s_ngpN{0u};
                    const uint32_t nn = s_ngpN.load(std::memory_order_relaxed);
                    if ((wotmGFrame % 30u) == 0u && nn < 6u)
                    {
                        s_ngpN.store(nn + 1u, std::memory_order_relaxed);
                        const uint32_t ngpCnt  = m_memory.read32(0x00445404u);
                        std::fprintf(stderr, "[ngpreloc] #%u gFrame=%u ngpCnt=%u base[0..3]=%08x %08x %08x %08x\n",
                                     nn, wotmGFrame, ngpCnt,
                                     m_memory.read32(0x004453CCu), m_memory.read32(0x004453D0u),
                                     m_memory.read32(0x004453D4u), m_memory.read32(0x004453D8u));
                        uint32_t hits = 0u;
                        for (uint32_t a = 0x00a00000u; a < 0x00a26000u && hits < 60u; a += 4u)
                        {
                            // GIFtag word0 = NLOOP/EOP/NREG (`0x00008004` correct / `0x00008001` ours),
                            // word1 = `0x303e4000` correct / `0x303b6000` ours.
                            const uint32_t w1 = m_memory.read32(a);
                            if ((w1 & 0xFF00F000u) == 0x30004000u)
                            {
                                const uint32_t w0 = m_memory.read32(a - 4u);
                                if ((w0 & 0xFFFF0000u) == 0u && (w0 & 0x8000u) != 0u)
                                {
                                    std::fprintf(stderr, "[ngpreloc]   @0x%x  %08x %08x %08x %08x  %s\n",
                                                 a - 4u, w0, w1, m_memory.read32(a + 4u), m_memory.read32(a + 8u),
                                                 (w0 == 0x8004u && w1 == 0x303E4000u) ? "OK"
                                                   : (w1 == 0x303E4000u ? "ok?" : "CORRUPT"));
                                    ++hits;
                                }
                            }
                        }
                        // Explicit peeks at the addresses our [dmatag] chain references.
                        for (uint32_t base : {0x00a1e080u, 0x00a1f780u, 0x00a1ff00u, 0x00a20300u, 0x00a20500u})
                        {
                            std::fprintf(stderr, "[ngpreloc]   peek 0x%x:", base);
                            for (uint32_t k = 0u; k < 12u; ++k)
                                std::fprintf(stderr, " %08x", m_memory.read32(base + k * 4u));
                            std::fprintf(stderr, "\n");
                        }
                    }
                }

                // One-shot: when WotM settles on screenMain (currScreen==1, not
                // mid-transition), arm a full GS register+prim stream capture.
                // Opt-in: set PS2X_GS_FRAMEDUMP=<budget> in the environment.
                static bool s_gsDumpChecked = false;
                static int64_t s_gsDumpBudget = 0;
                if (!s_gsDumpChecked)
                {
                    s_gsDumpChecked = true;
                    if (const char *e = std::getenv("PS2X_GS_FRAMEDUMP"))
                        s_gsDumpBudget = std::atoll(e);
                }
                static bool s_gsDumpFired = false;
                if (s_gsDumpBudget > 0 && !s_gsDumpFired && wotmCurScreen == 1u && wotmBetween == 0u)
                {
                    s_gsDumpFired = true;
                    g_gsFrameDump.store(s_gsDumpBudget, std::memory_order_relaxed);
                    RUNTIME_LOG("[gs:dump] ARMED budget=" << s_gsDumpBudget << " gFrame=" << wotmGFrame << std::endl);
                }

                // One-shot scene-graph matrix dump once screenMain is live.
                static int s_matDumpCountdown = -1;
                if (s_matDumpCountdown < 0 && (wotmCurScreen == 1u || wotmCurScreen == 0x14u) &&
                    wotmBetween == 0u && wotmGFrame > 4u)
                    s_matDumpCountdown = 3; // dump on the next 3 ticks (screenMain OR screenWaitForStart)
                if (s_matDumpCountdown > 0)
                {
                    s_matDumpCountdown--;
                    auto dumpMat = [&](const char *tag, uint32_t addr, int rows)
                    {
                        std::ostringstream os;
                        os << "[mat] " << tag << " @0x" << std::hex << addr << std::dec;
                        for (int r = 0; r < rows; ++r)
                        {
                            os << " |";
                            for (int c = 0; c < 4; ++c)
                            {
                                uint32_t bits = m_memory.read32(addr + static_cast<uint32_t>((r * 4 + c) * 4));
                                float f;
                                std::memcpy(&f, &bits, 4);
                                os << " " << f;
                            }
                        }
                        RUNTIME_LOG(os.str() << std::endl);
                    };
                    {
                        const uint32_t wptr = m_memory.read32(0x006F87C4u);
                        const uint32_t sroot = wptr ? m_memory.read32(wptr) : 0u;
                        RUNTIME_LOG("[mat] worldGlobal[0x6F87C4]=0x" << std::hex << wptr
                            << " *world(sceneRoot)=0x" << sroot << std::dec << std::endl);
                    }
                    RUNTIME_LOG("[mat] ---- scene-graph matrix dump gFrame=" << wotmGFrame << " ----" << std::endl);
                    dumpMat("matStack[0]", 0x70000400u, 4);
                    dumpMat("matStack[1]", 0x70000440u, 4);
                    dumpMat("matStack[2]", 0x70000480u, 4);
                    // hierCsUpdateAsm (0x207068) reads sp[0x70000390] as the CS-world
                    // translation base — retail = world[2] = (125,-1572,-0.883).
                    dumpMat("sp_0x70000380", 0x70000380u, 2);
                    dumpMat("sp_0x70000390", 0x70000390u, 1);
                    dumpMat("sp_0x700003a0", 0x700003A0u, 2);
                    dumpMat("sp_0x70002d00", 0x70002D00u, 4);
                    dumpMat("cammat_0x777f80", 0x00777F80u, 4);
                    dumpMat("camtrans_0x777fc0", 0x00777FC0u, 1);
                    dumpMat("zyfix_0x445c80", 0x00445C80u, 4);
                    dumpMat("world_0x445cd0", 0x00445CD0u, 4);   // worldCtx: [0]=root ptr etc.
                    dumpMat("viewRot_0x800d70", 0x00800D70u, 4); // heap 0x800d50 + 0x20
                    dumpMat("fovNorms_0x70000900", 0x70000900u, 4);
                    dumpMat("lightDir_0x70001300", 0x70001300u, 4);
                    // paraL global (plightGetParaLight returns &paraL = 0x006DFC00).
                    // hierCsUpdateAsm's vcallms 0xB00 should write lightDir = csMat3x3 * paraL.
                    dumpMat("paraL_0x6dfc00", 0x006DFC00u, 5);
                    dumpMat("csMat_0x70002d00", 0x70002D00u, 4);
                    dumpMat("dmaVu1arr_0x446680", 0x00446680u, 2);
                    // The assembled VU1 DMA packet hierTraverseAsm writes; on real
                    // HW its head is a CALL DMAtag whose TTE carries UNPACK-4@0
                    // (the object MVP w/ camera rotation).
                    auto dumpRam = [&](const char *tag, uint32_t base, uint32_t bytes)
                    {
                        for (uint32_t off = 0; off < bytes; off += 0x10u)
                        {
                            uint32_t w0 = m_memory.read32(base + off);
                            uint32_t w1 = m_memory.read32(base + off + 4u);
                            uint32_t w2 = m_memory.read32(base + off + 8u);
                            uint32_t w3 = m_memory.read32(base + off + 12u);
                            RUNTIME_LOG("[mat] " << tag << "+" << std::hex << off << " = "
                                << w0 << " " << w1 << " " << w2 << " " << w3 << std::dec << std::endl);
                        }
                    };
                    dumpRam("pkt_0x446700", 0x00446700u, 0x90u);
                    dumpRam("call_0xa00280", 0x00A00280u, 0x60u);   // our CALL-tag target
                    dumpRam("call_0xa02000", 0x00A02000u, 0x60u);   // PCSX2's CALL-tag target
                    // CsPool::m_HPActiveList sentinel 0x0043A2E0 (.next @ +4);
                    // 14 cs nodes @ 0x800230.. (0xB0 stride). cs[+0xc] = "active"
                    // flag hierTraceHPCsList tests. Node 0xa028c0 = a 0xffffffff
                    // terminator on retail but our traversal emits it repeatedly.
                    dumpRam("csHPlist_0x43a2e0", 0x0043A2E0u, 0x20u);
                    dumpRam("csList_0x43a2c0", 0x0043A2C0u, 0x20u);
                    dumpRam("hplinks_0x808da0", 0x00808DA0u, 0xF0u);   // 15 link nodes {payload,next,prev,0}
                    dumpRam("cs_0x800230", 0x00800230u, 0x10u);
                    dumpRam("cs_0x8002e0", 0x008002E0u, 0x10u);
                    dumpRam("cs_0x800390", 0x00800390u, 0x50u);
                    dumpRam("node_0xa028c0", 0x00A028C0u, 0x40u);
                    dumpRam("node_0xa1df80", 0x00A1DF80u, 0x40u);
                    // VU0 microcode: dump the programs invoked by hierTraverseAsm's
                    // vcallms (0x0/0x1A8/0x398/0x3F8/0x748) so we can decode 0x748
                    // even though it never executes in the recomp.
                    if (const uint8_t *vc = m_memory.getVU0Code())
                    {
                        for (uint32_t base : {0x0u, 0x1A8u, 0x398u, 0x3F8u, 0x430u, 0x748u, 0xB00u, 0xB50u})
                        {
                            const uint32_t span = (base == 0x1A8u) ? 0x1A0u : 0x80u;
                            for (uint32_t off = 0; off < span; off += 0x10u)
                            {
                                uint32_t w[4];
                                std::memcpy(w, vc + base + off, 16);
                                RUNTIME_LOG("[mat] vu0code+0x" << std::hex << (base + off) << " = "
                                    << w[0] << " " << w[1] << " " << w[2] << " " << w[3] << std::dec << std::endl);
                            }
                        }
                    }
                }

                PS2_IF_AGRESSIVE_LOGS({
                    // [run:threads] — every guest thread's status/wait/pc, to see who is
                    // parked when only the idle thread runs (e.g. the intro FMV stall).
                    static uint32_t s_thrDump = 0u;
                    if ((s_thrDump++ % 4u) == 0u)
                    {
                        std::ostringstream ts;
                        for (const auto &t : eeSnapshot.threads)
                            ts << " [" << t.id << " st=" << static_cast<int>(t.status)
                               << " wait=" << static_cast<int>(t.waitReason) << "/" << t.waitId
                               << " pri=" << t.currentPriority
                               << " pc=0x" << std::hex << t.pc << std::dec << "]";
                        RUNTIME_LOG("[run:threads] tick=" << tick << " running=" << eeSnapshot.runningThreadId
                                                          << ts.str() << std::endl);
                    }
                });
                RUNTIME_LOG("[run:tick] tick=" << tick
                                               << " pc=0x" << std::hex << dbgPc
                                               << " ra=0x" << dbgRa
                                               << " sp=0x" << dbgSp
                                               << " gp=0x" << dbgGp
                                               << " dispfb1=0x" << gs.dispfb1
                                               << " display1=0x" << gs.display1
                                               << std::dec
                                               << " voCnt=" << m_memory.read32(0x54d974u)
                                               << " vdState=" << m_memory.read32(0x54da28u)
                                               << " vbhEn=" << m_memory.read32(0x6F8654u)
                                               << " cntVb=" << m_memory.read32(0x6F8638u)
                                               << " isUp=" << m_memory.read32(0x6F8640u)
                                               << " hErr=" << static_cast<int32_t>(m_memory.read32(0x6F8644u))
                                               << " frmEnd=" << m_memory.read32(0x6F863Cu)
                                               << " csr=0x" << std::hex << m_memory.readIORegister(0x12001000u) << std::dec
                                               << " gifKick=" << g_ps2xGifKicks.load(std::memory_order_relaxed)
                                               << " c2=" << g_ps2xCause2.load(std::memory_order_relaxed)
                                               << " eiQ=" << g_ps2xEndimgQueued.load(std::memory_order_relaxed)
                                               << " d2chcr=0x" << std::hex << m_memory.readIORegister(0x1000A000u) << std::dec
                                               << " activeThreads=" << eeSnapshot.threads.size()
                                               << " dma=" << curDma
                                               << " gif=" << curGif
                                               << " gsw=" << curGs
                                               << " vif=" << curVif
                                               << " | scr=" << wotmCurScreen
                                               << " btwn=" << wotmBetween
                                               << " uiRet=" << wotmUiRet
                                               << " initAnim=" << wotmInitAnim
                                               << " contDec=" << wotmContDec
                                               << " fade=" << wotmFadeOff
                                               << " gFrame=" << wotmGFrame
                                               << " | lastScr=" << wotmLastScreen
                                               << " animIdx=" << wotmCurAnimIdx
                                               << " waitAnim=" << wotmWaitAnim
                                               << " waitFade=" << wotmWaitFade
                                               << " tgtAlpha=" << wotmTgtAlpha
                                               << " cs800bd0C=" << wotmCs800bd0C
                                               << std::endl);
                (void)wotmFadingOut;
            }
        });
        uint32_t presentWidth = FB_WIDTH;
        uint32_t presentHeight = DEFAULT_DISPLAY_HEIGHT;
        // With the GPU rasterizer the finished frame is already a texture in a
        // context shared with this one, so it is drawn straight from there: no
        // read back, no per-frame upload.
        Texture2D presentTex = frameTex;
        Ps2xGpuFrame gpuFrame{};
        ReportRuntimeProgress(this); // independent of CPU/GPU presentation
        LatchFrameIfNeeded(this); // drives the backend's Present for both paths
#ifdef _WIN32
        const bool d3dLive=ps2xD3D12Enabled();
#else
        const bool d3dLive=false;
#endif
        const bool gpuDirect = !d3dLive && ps2xGsGpuPresentTexture(&gpuFrame);
        if (gpuDirect)
        {
            presentTex.id = gpuFrame.texture;
            presentTex.width = static_cast<int>(gpuFrame.textureWidth);
            presentTex.height = static_cast<int>(gpuFrame.textureHeight);
            presentTex.mipmaps = 1;
            presentTex.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
            // The aspect ratio is the PS2's; the source rectangle is in the
            // render target's own (possibly larger) pixels.
            presentWidth = gpuFrame.displayWidth;
            presentHeight = gpuFrame.displayHeight;
        }
        else if(!d3dLive)
        {
            UploadFrame(frameTex, this, presentWidth, presentHeight);
        }

        extern void ps2xMovieAudioUpdate();
        ps2xMovieAudioUpdate();
#ifdef _WIN32
        if(d3dLive) {
            const double beforePresent=GetTime();
            if(!ps2xD3D12Present(GetWindowHandle())) {requestStop();break;}
            // EndDrawing swaps OpenGL buffers. D3D12 owns presentation here.
            // Retain the host input pump and independently paced display loop.
            static double lastPresent=GetTime();
            static const int cap=[] {const char* p=std::getenv("PS2X_PRESENT_FPS");int v=p?std::atoi(p):60;
                // Zero means unlimited presentation, including interpolation.
                // The independent gameplay clock remains at its selected rate.
                return v==0?0:std::clamp(v,30,360);}();
            const double afterPresent=GetTime();
            const double remaining=cap?1.0/cap-(afterPresent-lastPresent):0;
            if(remaining>0)WaitTime(remaining);
            lastPresent=GetTime();PollInputEvents();
            static const bool trace=std::getenv("PS2X_PRESENT_TIMING")!=nullptr;
            if(trace) {
                static double start=beforePresent,render=0,wait=0,poll=0;static unsigned count=0;
                const double afterPoll=GetTime();render+=afterPresent-beforePresent;wait+=lastPresent-afterPresent;poll+=afterPoll-lastPresent;++count;
                if(afterPoll-start>=5.0) {
                    std::fprintf(stderr,"[present:timing] cap=%d render=%.3fms wait=%.3fms poll=%.3fms count=%u\n",cap,render*1000/count,wait*1000/count,poll*1000/count,count);
                    start=afterPoll;render=wait=poll=0;count=0;
                }
            }
        } else
#endif
        {
        BeginDrawing();
        ClearBackground(BLACK);
        const float srcWidth = static_cast<float>(std::max<uint32_t>(1u, presentWidth));
        const float srcHeight = static_cast<float>(std::max<uint32_t>(1u, presentHeight));
        const float screenWidth = static_cast<float>(GetScreenWidth());
        const float screenHeight = static_cast<float>(GetScreenHeight());
        const bool wide = gpuDirect && g_ps2xWotmWideActive.load(std::memory_order_relaxed) &&
            presentWidth == 640u && presentHeight == 448u;
        const float displayWidth = wide ? srcHeight * (16.0f / 9.0f) : srcWidth;
        const float scale = std::min(screenWidth / displayWidth, screenHeight / srcHeight);
        const float dstWidth = displayWidth * scale;
        const float dstHeight = srcHeight * scale;
        // A GL render target's first row is the bottom of the image, so the
        // source rectangle is read upwards (raylib flips on a negative height).
        // A GL render target's first row is the bottom of the image, so the
        // source rectangle is read upwards: a negative height flips it. raylib
        // then does "source.y -= source.height" before building the texture
        // coordinates, so y has to start one rectangle height short of the top,
        // or the V coordinates land outside 0..1 and clamp to the edge row --
        // which is the letterbox bar, i.e. a black window.
        const Rectangle srcRect =
            gpuDirect ? Rectangle{0.0f, static_cast<float>(gpuFrame.textureHeight - gpuFrame.sourceHeight),
                                  static_cast<float>(gpuFrame.sourceWidth), -static_cast<float>(gpuFrame.sourceHeight)}
                      : Rectangle{0.0f, 0.0f, srcWidth, srcHeight};
        const Rectangle dstRect{
            (screenWidth - dstWidth) * 0.5f,
            (screenHeight - dstHeight) * 0.5f,
            dstWidth,
            dstHeight};
        if (aaShader.id)
        {
            // One output pixel in source UV space, independent of internal scale.
            const float sw = std::abs(srcRect.width), sh = std::abs(srcRect.height);
            const float tw = static_cast<float>(presentTex.width), th = static_cast<float>(presentTex.height);
            const float step[2] = {sw / (std::max(1.0f, dstWidth) * tw), sh / (std::max(1.0f, dstHeight) * th)};
            // DrawTexturePro flips negative source heights by adjusting source.y.
            // The occupied image can sit at the TOP of a larger GL target.
            const float left = std::min(srcRect.x, srcRect.x + srcRect.width);
            const float bottom = srcRect.height < 0 ? srcRect.y : std::min(srcRect.y, srcRect.y + srcRect.height);
            const float bounds[4] = {(left + 0.5f) / tw, (bottom + 0.5f) / th,
                                     (left + sw - 0.5f) / tw, (bottom + sh - 0.5f) / th};
            SetShaderValue(aaShader, aaStep, step, SHADER_UNIFORM_VEC2);
            SetShaderValue(aaShader, aaBounds, bounds, SHADER_UNIFORM_VEC4);
            BeginShaderMode(aaShader);
        }
        if (gpuDirect)
        {
            // The render target's alpha channel is a GS blend operand, not
            // coverage (and this game's FRAME is PSMCT24, which has no alpha at
            // all). Drawing it with the default alpha blend multiplied the whole
            // picture by that channel, leaving only the opaque pixels -- a
            // nearly black window. Present it with no blending.
            rlSetBlendFactors(RL_SRC_ALPHA, RL_ONE_MINUS_SRC_ALPHA, RL_FUNC_ADD);
            rlDrawRenderBatchActive();
            rlDisableColorBlend();
            DrawTexturePro(presentTex, srcRect, dstRect, Vector2{0.0f, 0.0f}, 0.0f, WHITE);
            rlDrawRenderBatchActive();
            rlEnableColorBlend();
        }
        else
        {
            DrawTexturePro(presentTex, srcRect, dstRect, Vector2{0.0f, 0.0f}, 0.0f, WHITE);
        }
        if (aaShader.id) EndShaderMode();
        if (m_debugUiInitialized && m_debugUiDrawCallback)
        {
            m_debugUiDrawCallback(*this, m_debugUiUserData);
        }
        EndDrawing();
        // Optional one-shot capture of the final post-AA display for diagnostics.
        static unsigned displayShotFrame = 0;
        static const unsigned displayShotTarget = [] {
            const char* value = std::getenv("PS2X_PRESENT_SHOT_FRAME");
            return value ? static_cast<unsigned>(std::max(1, std::atoi(value))) : 180u;
        }();
        if (++displayShotFrame == displayShotTarget)
            if (const char* shot = std::getenv("PS2X_PRESENT_SHOT"))
                if (*shot) {
                    Image displayImage = LoadImageFromScreen();
                    const bool saved = ExportImage(displayImage, shot);
                    UnloadImage(displayImage);
                    std::fprintf(stderr, "[display] post-AA screenshot %s\n", saved ? "saved" : "failed");
                }


        } // OpenGL presentation

        // PS2X_EXIT_AFTER=<seconds>: end the run by itself, for unattended
        // benchmark runs.
        {
            static const double kExitAfter = []
            {
                const char *value = std::getenv("PS2X_EXIT_AFTER");
                return value != nullptr ? std::atof(value) : 0.0;
            }();
            static const double kRunStart = GetTime();
            if (kExitAfter > 0.0 && GetTime() - kRunStart > kExitAfter)
            {
                std::fprintf(stderr, "[bench] PS2X_EXIT_AFTER=%.0fs reached, stopping\n", kExitAfter);
                break;
            }
        }
        if (WindowShouldClose())
        {
            RUNTIME_LOG("[run] window close requested, breaking out of loop");
            requestStop();
            break;
        }
    }

    requestStop();
    if (gameThread.joinable())
    {
        gameThread.join();
    }
    // The VU1 worker uses m_gs and m_vu1, which are destroyed before m_memory.
    m_memory.stopVu1Worker();

    if (m_debugUiInitialized && m_debugUiShutdownCallback)
    {
        m_debugUiShutdownCallback(*this, m_debugUiUserData);
        m_debugUiInitialized = false;
    }
    extern void ps2xMovieAudioShutdown();
    ps2xMovieAudioShutdown();
    if (aaShader.id) UnloadShader(aaShader);
    UnloadTexture(frameTex);
#ifdef _WIN32
    ps2xD3D12Shutdown();
#endif
    CloseWindow();

    RUNTIME_LOG("[run] exiting loop");
}


// VU0 macro stores address the VU data bank, not EE RDRAM. Keep this out of
// shared headers so instruction fixes do not rebuild every generated function.
void ps2xVu0StoreQuad(PS2Runtime* runtime, R5900Context* ctx,
                     uint32_t vf, uint32_t vi, uint32_t mask, bool predecrement)
{
    vf &= 31u;
    vi &= 15u;
    uint16_t index = vi ? ctx->vi[vi] : 0u;
    if (predecrement && vi)
        index = static_cast<uint16_t>(index - 1u);

    // Bit 10 selects the VU1 register window, not a mirror of VU0 RAM.
    // The macro runtime has no mapped-register implementation yet.
    if ((mask & 15u) && (index & 0x400u))
        throw std::runtime_error("VU0 macro store to mapped VU1 registers is not implemented");

    uint8_t* const dst = runtime->memory().getVU0Data() + ((index & 0xffu) << 4);
    uint32_t source[4];
    const __m128 value = vf ? ctx->vu0_vf[vf] : _mm_set_ps(1.0f, 0.0f, 0.0f, 0.0f);
    std::memcpy(source, &value, sizeof(source));
    for (uint32_t lane = 0; lane < 4u; ++lane)
        if (mask & (8u >> lane))
            std::memcpy(dst + lane * 4u, &source[lane], sizeof(uint32_t));

    if (vi)
        ctx->vi[vi] = predecrement ? index : static_cast<uint16_t>(index + 1u);

    static const bool trace = [] { const char* v = std::getenv("PS2X_VU0_STORE_TRACE"); return v && v[0] == '1'; }();
    if (trace)
    {
        static std::atomic<uint64_t> calls{0};
        const uint64_t n = calls.fetch_add(1, std::memory_order_relaxed) + 1;
        if (n <= 12 || n % 1000 == 0)
            std::fprintf(stderr, "[vu0:macro-store] calls=%llu pc=%08x vf=%u vi=%u qword=%u mask=%x\n",
                         static_cast<unsigned long long>(n), ctx->pc, vf, vi, index, mask);
    }

}
