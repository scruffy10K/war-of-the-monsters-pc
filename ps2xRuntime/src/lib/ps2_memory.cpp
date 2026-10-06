#include <intrin.h>
#include "runtime/ps2_memory.h"
#include "runtime/ps2_address.h"
#include "runtime/gs/gs_frontend.h"
#include "ps2_log.h"
#include <atomic>
#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <vector>
#include "motion_provenance_impl.inc"
#include "vif_motion_signature.inc"

// Parallel-VU1 feasibility probe (see ps2_runtime.cpp [perf] "vif1 gap").
std::atomic<uint64_t> g_ps2xVif1Transfers{0u};
std::atomic<uint64_t> g_ps2xVif1InsideTsc{0u};
std::atomic<uint64_t> g_ps2xVif1GapDmaReads{0u};     // DMA register reads between transfers
std::atomic<uint64_t> g_ps2xVif1GapsWithPoll{0u};    // gaps containing at least one DMA read
std::atomic<uint64_t> g_ps2xVif1GapsWithBlock{0u};   // gaps containing a guest blocking wait
std::atomic<bool> g_ps2xVif1GapOpen{false};
std::atomic<bool> g_ps2xVif1GapPolled{false};
std::atomic<bool> g_ps2xVif1GapBlocked{false};
std::atomic<bool> g_ps2xVif1GapWaitCall{false};       // WaitSema/PollSema/WaitEventFlag/PollEventFlag called
std::atomic<bool> g_ps2xVif1GapSignal{false};         // SignalSema/SetEventFlag called
std::atomic<uint64_t> g_ps2xVif1GapsWithWaitCall{0u};
std::atomic<uint64_t> g_ps2xVif1GapsWithSignal{0u};

std::atomic<uint32_t> g_ps2xGifKicks{0u};
std::atomic<uint32_t> g_ps2xCause2{0u};

namespace
{
    inline void inRange(uint32_t offset, size_t bytes, size_t regionSize, const char *op, uint32_t address)
    {
        if (static_cast<uint64_t>(offset) + static_cast<uint64_t>(bytes) > static_cast<uint64_t>(regionSize))
        {
            throw std::runtime_error(std::string(op) + " out-of-bounds at address: 0x" + std::to_string(address));
        }
    }

    template <typename T>
    inline T loadScalar(const uint8_t *base, uint32_t offset, size_t regionSize, const char *op, uint32_t address)
    {
        inRange(offset, sizeof(T), regionSize, op, address);
        T value{};
        std::memcpy(&value, base + offset, sizeof(T));
        return value;
    }

    template <typename T>
    inline void storeScalar(uint8_t *base, uint32_t offset, size_t regionSize, T value, const char *op, uint32_t address)
    {
        inRange(offset, sizeof(T), regionSize, op, address);
        std::memcpy(base + offset, &value, sizeof(T));
    }

    inline bool isGsPrivReg(uint32_t addr)
    {
        return Ps2AddressInRange(addr, PS2_GS_PRIV_REG_BASE, PS2_GS_PRIV_REG_SIZE);
    }

    inline bool isIoRegister(uint32_t addr)
    {
        return Ps2AddressInRange(addr, PS2_IO_BASE, PS2_IO_SIZE);
    }

    inline uint64_t *gsRegPtr(GSRegisters &gs, uint32_t addr)
    {
        // Support both 64-bit base offsets and +4 dword aliases.
        uint32_t off = (addr - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        switch (off)
        {
        case 0x0000:
            return &gs.pmode;
        case 0x0010:
            return &gs.smode1;
        case 0x0020:
            return &gs.smode2;
        case 0x0030:
            return &gs.srfsh;
        case 0x0040:
            return &gs.synch1;
        case 0x0050:
            return &gs.synch2;
        case 0x0060:
            return &gs.syncv;
        case 0x0070:
            return &gs.dispfb1;
        case 0x0080:
            return &gs.display1;
        case 0x0090:
            return &gs.dispfb2;
        case 0x00A0:
            return &gs.display2;
        case 0x00B0:
            return &gs.extbuf;
        case 0x00C0:
            return &gs.extdata;
        case 0x00D0:
            return &gs.extwrite;
        case 0x00E0:
            return &gs.bgcolor;
        // CSR (offset 0x1000) is intentionally not handled here: it is
        // std::atomic<uint64_t> and no longer converts to uint64_t*. Callers must
        // check for offset 0x1000 themselves and go through writeCsrHalf/
        // writeCsrFull/gs.csr.load() instead of gsRegPtr().
        case 0x1010:
            return &gs.imr;
        case 0x1040:
            return &gs.busdir;
        case 0x1080:
            return &gs.siglblid;
        default:
            return nullptr;
        }
    }

    constexpr uint32_t kGsCsrRegOffset = 0x1000u;

    // Atomically apply a 32-bit write to one half (off=0 low dword, off=4 high
    // dword) of the GS CSR register. Bits 0..1 of the low dword (SIGNAL/FINISH) are
    // write-one-to-clear; everything else is a plain merge. Uses compare_exchange
    // so the whole read-modify-write is a single atomic step -- this register is
    // also touched by the vsync worker (FIELD bit) and the GIF (SIGNAL/FINISH) on
    // other threads, so a load-then-store here would race with them.
    inline void writeCsrHalf(std::atomic<uint64_t> &csr, uint32_t off, uint32_t value)
    {
        constexpr uint32_t kW1cMask = 0x3u;
        uint64_t expected = csr.load();
        uint64_t desired;
        do
        {
            if (off == 0u)
            {
                uint32_t oldLow = static_cast<uint32_t>(expected & 0xFFFFFFFFull);
                uint32_t mergedLow = (oldLow & kW1cMask) | (value & ~kW1cMask);
                desired = (expected & 0xFFFFFFFF00000000ull) | static_cast<uint64_t>(mergedLow);
                desired &= ~static_cast<uint64_t>(value & kW1cMask);
            }
            else
            {
                uint64_t mask = 0xFFFFFFFFull << (off * 8u);
                desired = (expected & ~mask) | (static_cast<uint64_t>(value) << (off * 8u));
            }
        } while (!csr.compare_exchange_weak(expected, desired));
    }

    // Same as writeCsrHalf but for a full 64-bit CSR write (bits 0..1 are still
    // write-one-to-clear against the current value).
    inline void writeCsrFull(std::atomic<uint64_t> &csr, uint64_t value)
    {
        constexpr uint64_t kW1cMask = 0x3ull;
        uint64_t expected = csr.load();
        uint64_t desired;
        do
        {
            desired = (expected & kW1cMask) | (value & ~kW1cMask);
            desired &= ~(value & kW1cMask);
        } while (!csr.compare_exchange_weak(expected, desired));
    }

    constexpr std::array<uint32_t, 4> kEeTimerBases = {
        0x10000000u,
        0x10000800u,
        0x10001000u,
        0x10001800u,
    };
    constexpr uint32_t kEeTimerCountOffset = 0x00u;
    constexpr uint32_t kEeTimerModeOffset = 0x10u;
    constexpr uint32_t kEeTimerCompareOffset = 0x20u;
    constexpr uint32_t kEeTimerHoldOffset = 0x30u;
    constexpr uint32_t kEeTimerModeClksMask = 0x3u;
    constexpr uint32_t kEeTimerModeConfigMask = 0x3FFu;
    constexpr uint32_t kEeTimerModeStatusMask = 0xC00u;
    constexpr uint32_t kEeTimerModeZret = 1u << 6;
    constexpr uint32_t kEeTimerModeCue = 1u << 7;
    constexpr uint32_t kEeTimerModeCmpe = 1u << 8;
    constexpr uint32_t kEeTimerModeOvfe = 1u << 9;
    constexpr uint32_t kEeTimerModeEquf = 1u << 10;
    constexpr uint32_t kEeTimerModeOvff = 1u << 11;
    constexpr uint64_t kEeClockHz = 294912000ull;
    constexpr std::array<uint64_t, 4> kEeTimerClockHz = {
        147456000ull,
        9216000ull,
        576000ull,
        15734ull,
    };

    inline bool decodeEeTimerRegister(uint32_t address, size_t &timerIndex, uint32_t &offset)
    {
        for (size_t index = 0; index < kEeTimerBases.size(); ++index)
        {
            const uint32_t candidateOffset = address - kEeTimerBases[index];
            if (candidateOffset == kEeTimerCountOffset ||
                candidateOffset == kEeTimerModeOffset ||
                candidateOffset == kEeTimerCompareOffset ||
                (index < 2u && candidateOffset == kEeTimerHoldOffset))
            {
                timerIndex = index;
                offset = candidateOffset;
                return true;
            }
        }
        return false;
    }

    constexpr uint64_t ticksUntilMatch(uint32_t count, uint32_t target)
    {
        const uint32_t distance = (target - count) & 0xFFFFu;
        return distance == 0u ? 0x10000ull : static_cast<uint64_t>(distance);
    }

    struct DmaTagView
    {
        uint16_t qwc = 0;
        uint8_t id = 0;
        bool irq = false;
        uint32_t addr = 0;
        uint32_t upper = 0;
    };

    inline DmaTagView decodeDmaTag(uint64_t tag)
    {
        DmaTagView out{};
        out.qwc = static_cast<uint16_t>(tag & 0xFFFFu);
        out.id = static_cast<uint8_t>((tag >> 28u) & 0x7u);
        out.irq = ((tag >> 31u) & 0x1ull) != 0ull;
        out.addr = static_cast<uint32_t>((tag >> 32u) & 0x7FFFFFFFu);
        out.upper = static_cast<uint32_t>((tag >> 16u) & 0xFFFFu);
        return out;
    }

    inline uint32_t gifTagNloop(uint64_t tagLo)
    {
        return static_cast<uint32_t>(tagLo & 0x7FFFu);
    }

    inline uint8_t gifTagFlg(uint64_t tagLo)
    {
        return static_cast<uint8_t>((tagLo >> 58u) & 0x3u);
    }

    inline uint32_t gifTagNreg(uint64_t tagLo)
    {
        uint32_t nreg = static_cast<uint32_t>((tagLo >> 60u) & 0xFu);
        return nreg == 0u ? 16u : nreg;
    }

}

// Helpers for GS VRAM addressing (PSMCT32 path).
static inline uint32_t gs_vram_offset(uint32_t basePage, uint32_t x, uint32_t y, uint32_t fbw)
{
    // basePage is in 2048-byte units; fbw is in blocks of 64 pixels.
    uint32_t strideBytes = fbw * 64 * 4;
    return basePage * 2048 + y * strideBytes + x * 4;
}

PS2Memory::PS2Memory()
    : m_rdram(nullptr), m_scratchpad(nullptr), iop_ram(nullptr), m_seenGifCopy(false), m_gsVRAM(nullptr)
{
    ps2SetScratchpadHostPtr(nullptr);
}

// ---- VU1 worker thread ------------------------------------------------------
namespace
{
    const bool s_vu1Threaded = []
    {
        const char *value = std::getenv("PS2X_VU1_THREAD");
        if (value) return value[0] == '1';
        // The verified fast GPU path can overlap VIF/VU work with EE gameplay.
        // Keep serial execution for small CPUs and exact/software diagnostics;
        // PS2X_VU1_THREAD=0 remains the explicit reference/compatibility path.
        const char *gpu = std::getenv("PS2X_GS_GPU");
        const char *fast = std::getenv("PS2X_VU1_FAST");
        return gpu && gpu[0] == '1' && fast && fast[0] == '1' &&
               std::thread::hardware_concurrency() >= 4u;
    }();
    // 0 retains the former per-job signaling for measured comparisons.
    const bool s_vu1SelectiveNotify = [] {
        const char* value=std::getenv("PS2X_VU1_SELECTIVE_NOTIFY");
        return !value || value[0]!='0';
    }();
    const bool s_vu1NotifyAudit=std::getenv("PS2X_VU1_NOTIFY_AUDIT")!=nullptr;
    struct Vu1NotifyAudit {
        std::atomic<uint64_t> completions{0},completionSignals{0},submissionSignals{0};
        std::atomic<uint64_t> drainChecks{0},drainFalse{0},epochChecks{0},epochFalse{0};
    } s_vu1NotifyAuditCounts;
    #include "vu1_worker_notify.inc"
    thread_local bool t_ps2xIsVu1Worker = false;
}
std::atomic<bool> g_ps2xVu1WorkerBusy{false};
std::atomic<uint64_t> g_ps2xVu1JobsSubmitted{0u};
std::atomic<uint64_t> g_ps2xVu1Drains{0u};
std::atomic<uint64_t> g_ps2xVu1DrainWaitNs{0u};

bool PS2Memory::isVu1WorkerThread()
{
    return t_ps2xIsVu1Worker;
}

// For the GS frontend's ordering-hazard detector: true when a thread other than
// the worker is touching the GS while the worker still has drawing in flight.
bool ps2xVu1ForeignAccessWhileBusy()
{
    return s_vu1Threaded && !t_ps2xIsVu1Worker && g_ps2xVu1WorkerBusy.load(std::memory_order_relaxed);
}

void PS2Memory::startVu1WorkerIfNeeded()
{
    if (!s_vu1Threaded || m_vu1ThreadStarted)
        return;
    m_vu1ThreadStarted = true;
    m_vu1Stop = false;
    m_vu1Thread = std::thread([this] { vu1WorkerMain(); });
    std::fprintf(stderr, "[vu1thread] VU1 worker thread started\n");
}

void PS2Memory::vu1WorkerMain()
{
    t_ps2xIsVu1Worker = true;
    extern void ps2xSamplerRegisterVu1Thread();
    ps2xSamplerRegisterVu1Thread();
    for (;;)
    {
        Vu1Job job;
        {
            std::unique_lock<std::mutex> lock(m_vu1JobMutex);
            m_vu1JobCv.wait(lock, [this] { return m_vu1Stop || !m_vu1Jobs.empty(); });
            if (m_vu1Jobs.empty())
                return; // stopping, and nothing left to do
            job = std::move(m_vu1Jobs.front());
            m_vu1Jobs.pop_front();
            m_vu1Busy = true;
        }
        m_vu1JobDBit = job.dBit;
        m_vu1JobTBit = job.tBit;
        if (job.path3)
        {
            for (const std::vector<uint8_t> &chunk : job.chunks)
            {
                m_seenGifCopy = true;
                m_gifCopyCount.fetch_add(1, std::memory_order_relaxed);
                submitGifPacket(GifPathId::Path3, chunk.data(), static_cast<uint32_t>(chunk.size()), false);
            }
        }
        else
        {
            for (const std::vector<uint8_t> &chunk : job.chunks)
                processVIF1Data(chunk.data(), static_cast<uint32_t>(chunk.size()));
        }
        // The synchronous path drained the arbiter at the end of each transfer;
        // the PATH1/PATH2 packets this transfer produced are the worker's own.
        if (m_gifArbiter)
            m_gifArbiter->drain();
        {
            std::lock_guard<std::mutex> lock(m_vu1JobMutex);
            m_vu1Busy = false;
            ++m_vu1DoneSeq;
            if (m_vu1Jobs.empty())
                g_ps2xVu1WorkerBusy.store(false, std::memory_order_relaxed);
            const bool wake=!s_vu1SelectiveNotify || vu1CompletionNeedsWake(m_vu1Jobs.empty(),m_vu1DoneSeq,m_vu1SyncTarget);
            if(s_vu1NotifyAudit) {
                s_vu1NotifyAuditCounts.completions.fetch_add(1,std::memory_order_relaxed);
                if(wake)s_vu1NotifyAuditCounts.completionSignals.fetch_add(1,std::memory_order_relaxed);
            }
            if(wake)m_vu1IdleCv.notify_all();
        }
    }
}

void PS2Memory::submitVu1Job(Vu1Job &&job)
{
    bool wake;
    {
        std::lock_guard<std::mutex> lock(m_vu1JobMutex);
        wake=!s_vu1SelectiveNotify || vu1SubmissionNeedsWake(m_vu1Jobs.empty(),m_vu1Busy);
        m_vu1Jobs.push_back(std::move(job));
        ++m_vu1SubmitSeq;
        g_ps2xVu1WorkerBusy.store(true, std::memory_order_relaxed);
    }
    g_ps2xVu1JobsSubmitted.fetch_add(1u, std::memory_order_relaxed);
    if(s_vu1NotifyAudit && wake)s_vu1NotifyAuditCounts.submissionSignals.fetch_add(1,std::memory_order_relaxed);
    if(wake)m_vu1JobCv.notify_one();
}

static thread_local int g_ps2xDrainSite = 0;
static std::atomic<uint64_t> g_ps2xDrainSiteNs[4096];
static std::atomic<uint64_t> g_ps2xDrainSiteCnt[4096];
void PS2Memory::drainVu1Worker()
{
    // Never wait from the worker itself (it would wait for its own job).
    if (!s_vu1Threaded || !m_vu1ThreadStarted || t_ps2xIsVu1Worker)
        return;
    std::unique_lock<std::mutex> lock(m_vu1JobMutex);
    if (m_vu1Jobs.empty() && !m_vu1Busy)
        return;
    const auto t0 = std::chrono::steady_clock::now();
    // A wait that never ends means the caller is holding something the worker
    // needs (e.g. the GS state mutex): report it rather than freeze silently.
    while (!m_vu1IdleCv.wait_for(lock, std::chrono::seconds(2),
                                 [this] {
                                     const bool ready=m_vu1Jobs.empty() && !m_vu1Busy;
                                     if(s_vu1NotifyAudit) {
                                         s_vu1NotifyAuditCounts.drainChecks.fetch_add(1,std::memory_order_relaxed);
                                         if(!ready)s_vu1NotifyAuditCounts.drainFalse.fetch_add(1,std::memory_order_relaxed);
                                     }
                                     return ready;
                                 }))
    {
        std::fprintf(stderr, "[vu1thread] WARNING: drain still waiting after 2 s (%zu jobs queued, busy=%d)"
                             " -- possible deadlock: the draining thread may hold a lock the worker needs\n",
                     m_vu1Jobs.size(), m_vu1Busy ? 1 : 0);
    }
    {
        const uint64_t ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
        const int site = g_ps2xDrainSite & 4095;
        g_ps2xDrainSiteNs[site].fetch_add(ns, std::memory_order_relaxed);
        g_ps2xDrainSiteCnt[site].fetch_add(1u, std::memory_order_relaxed);
    }
    g_ps2xVu1Drains.fetch_add(1u, std::memory_order_relaxed);
    g_ps2xVu1DrainWaitNs.fetch_add(static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()),
        std::memory_order_relaxed);
}

void PS2Memory::vu1SyncPathEpoch()
{
    if (!s_vu1Threaded || !m_vu1ThreadStarted || t_ps2xIsVu1Worker)
        return;
    std::unique_lock<std::mutex> lock(m_vu1JobMutex);
    const uint64_t target = m_vu1SyncTarget;
    if (m_vu1DoneSeq < target)
    {
        const auto t0 = std::chrono::steady_clock::now();
        while (!m_vu1IdleCv.wait_for(lock, std::chrono::seconds(2), [&] {
            const bool ready=m_vu1DoneSeq >= target;
            if(s_vu1NotifyAudit) {
                s_vu1NotifyAuditCounts.epochChecks.fetch_add(1,std::memory_order_relaxed);
                if(!ready)s_vu1NotifyAuditCounts.epochFalse.fetch_add(1,std::memory_order_relaxed);
            }
            return ready;
        }))
            std::fprintf(stderr, "[vu1thread] WARNING: sync-path wait still waiting after 2 s (done %llu target %llu)\n",
                         static_cast<unsigned long long>(m_vu1DoneSeq), static_cast<unsigned long long>(target));
        g_ps2xVu1Drains.fetch_add(1u, std::memory_order_relaxed);
        g_ps2xVu1DrainWaitNs.fetch_add(static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()),
            std::memory_order_relaxed);
    }
    m_vu1SyncTarget = m_vu1SubmitSeq;
}

void PS2Memory::stopVu1Worker()
{
    if (!m_vu1ThreadStarted)
        return;
    for (int i = 0; i < 4096; ++i)
        if (g_ps2xDrainSiteCnt[i].load())
            std::fprintf(stderr, "[vu1thread] drain site line %d: %llu drains, %.2f s waiting\n", i,
                         (unsigned long long)g_ps2xDrainSiteCnt[i].load(), g_ps2xDrainSiteNs[i].load() / 1e9);
    {
        std::lock_guard<std::mutex> lock(m_vu1JobMutex);
        m_vu1Stop = true;
    }
    m_vu1JobCv.notify_all();
    if (m_vu1Thread.joinable())
        m_vu1Thread.join();
    m_vu1ThreadStarted = false;
    if(s_vu1NotifyAudit)std::fprintf(stderr,
        "[vu1thread:notify] selective=%u jobs=%llu completion-signals=%llu submission-signals=%llu drain-checks=%llu drain-false=%llu epoch-checks=%llu epoch-false=%llu\n",
        unsigned(s_vu1SelectiveNotify),s_vu1NotifyAuditCounts.completions.load(),
        s_vu1NotifyAuditCounts.completionSignals.load(),s_vu1NotifyAuditCounts.submissionSignals.load(),
        s_vu1NotifyAuditCounts.drainChecks.load(),s_vu1NotifyAuditCounts.drainFalse.load(),
        s_vu1NotifyAuditCounts.epochChecks.load(),s_vu1NotifyAuditCounts.epochFalse.load());
}

PS2Memory::~PS2Memory()
{
    stopVu1Worker();
    if (m_rdram)
    {
        delete[] m_rdram;
        m_rdram = nullptr;
    }

    if (m_scratchpad)
    {
        ps2SetScratchpadHostPtr(nullptr);
        delete[] m_scratchpad;
        m_scratchpad = nullptr;
    }

    if (m_gsVRAM)
    {
        delete[] m_gsVRAM;
        m_gsVRAM = nullptr;
    }

    if (m_vu1Code)
    {
        delete[] m_vu1Code;
        m_vu1Code = nullptr;
    }
    if (m_vu1Data)
    {
        delete[] m_vu1Data;
        m_vu1Data = nullptr;
    }
    if (m_vu0Code)
    {
        delete[] m_vu0Code;
        m_vu0Code = nullptr;
    }
    if (m_vu0Data)
    {
        delete[] m_vu0Data;
        m_vu0Data = nullptr;
    }

    if (iop_ram)
    {
        delete[] iop_ram;
        iop_ram = nullptr;
    }
}

bool PS2Memory::initialize(size_t ramSize)
{
    auto cleanup = [this]()
    {
        delete[] m_rdram;
        delete[] m_scratchpad;
        delete[] iop_ram;
        delete[] m_gsVRAM;
        delete[] m_vu0Code;
        delete[] m_vu0Data;
        delete[] m_vu1Code;
        delete[] m_vu1Data;
        m_rdram = nullptr;
        m_scratchpad = nullptr;
        ps2SetScratchpadHostPtr(nullptr);
        iop_ram = nullptr;
        m_gsVRAM = nullptr;
        m_vu0Code = nullptr;
        m_vu0Data = nullptr;
        m_vu1Code = nullptr;
        m_vu1Data = nullptr;
    };

    cleanup();
    m_seenGifCopy = false;
    m_dmaStartCount.store(0, std::memory_order_relaxed);
    m_gifCopyCount.store(0, std::memory_order_relaxed);
    m_gsWriteCount.store(0, std::memory_order_relaxed);
    m_vifWriteCount.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(m_completedDmacMutex);
        m_completedDmacCauses.clear();
    }
    m_codeRegions.clear();
    m_path3Masked = false;
    m_path3MaskedFifo.clear();
    m_vif1PendingPath2ImageQwc = 0u;
    m_vif1PendingPath2DirectHl = false;
    resetEeTimers();

    try
    {
        // Allocate main RAM
        m_rdram = new uint8_t[ramSize];
        std::memset(m_rdram, 0, ramSize);

        // Allocate scratchpad
        m_scratchpad = new uint8_t[PS2_SCRATCHPAD_SIZE];
        std::memset(m_scratchpad, 0, PS2_SCRATCHPAD_SIZE);
        ps2SetScratchpadHostPtr(m_scratchpad);

        // Initialize EE TLB entries (R5900 has 48 entries).
        m_tlbEntries.assign(48, TLBEntry{0, 0, 0, false});

        // Allocate IOP RAM
        iop_ram = new uint8_t[2 * 1024 * 1024]; // 2MB

        // Initialize IOP RAM with zeros
        std::memset(iop_ram, 0, 2 * 1024 * 1024);

        // Initialize I/O registers
        m_ioRegisters.clear();

        // Initialize GS registers
        memset(&gs_regs, 0, sizeof(gs_regs));
        // memset zero-fills std::atomic<uint64_t>::csr's bytes, which is not itself
        // a guaranteed-valid atomic store; make the zero-initialization explicit.
        gs_regs.csr.store(0);
        gs_regs.dispfb1 = (0ULL << 0) | (10ULL << 9) | (0ULL << 15) | (0ULL << 32) | (0ULL << 43);
        gs_regs.display1 = (0ULL << 0) | (0ULL << 12) | (0ULL << 23) | (0ULL << 27) | (639ULL << 32) | (447ULL << 44);
        gs_regs.dispfb2 = gs_regs.dispfb1;
        gs_regs.display2 = gs_regs.display1;

        // Allocate GS VRAM (4MB)
        m_gsVRAM = new uint8_t[PS2_GS_VRAM_SIZE];
        std::memset(m_gsVRAM, 0, PS2_GS_VRAM_SIZE);

        m_vu0Code = new uint8_t[PS2_VU0_CODE_SIZE];
        m_vu0Data = new uint8_t[PS2_VU0_DATA_SIZE];
        std::memset(m_vu0Code, 0, PS2_VU0_CODE_SIZE);
        std::memset(m_vu0Data, 0, PS2_VU0_DATA_SIZE);

        m_vu1Code = new uint8_t[PS2_VU1_CODE_SIZE];
        m_vu1Data = new uint8_t[PS2_VU1_DATA_SIZE];
        std::memset(m_vu1Code, 0, PS2_VU1_CODE_SIZE);
        std::memset(m_vu1Data, 0, PS2_VU1_DATA_SIZE);
        markVU0CodeModified();
        markVU1CodeModified();

        // Initialize VIF registers
        memset(&vif0_regs, 0, sizeof(vif0_regs));
        memset(&vif1_regs, 0, sizeof(vif1_regs));

        // Initialize DMA registers
        memset(dma_regs, 0, sizeof(dma_regs));

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error initializing PS2 memory: " << e.what() << std::endl;
        cleanup();
        return false;
    }
}

// EE timer batching. EeScheduler::checkpointDue() -> accountCycles() calls
// advanceEeTimers() at every generated-code checkpoint (millions of times a
// second, ~4.5% of the game thread in gameplay), almost always to learn that
// nothing happened. Cycles are now accumulated and only applied when
//  - the total reaches the next enabled timer event (cyclesUntilNextEeTimer-
//    Interrupt(), the minimum cycles for any compare/overflow flag to be set), or
//  - a timer register is read or written, or the next event is queried.
// Exact: the tick maths carries its remainder, so applying N cycles at once
// gives the same count/remainder as applying them in pieces, and the call that
// crosses an event is the same call as before, so interrupts fire at the same
// checkpoint. There is one PS2Memory per process, hence file-static state (no
// header change).
namespace
{
    uint64_t s_eeTimerPendingCycles = 0u;
    uint64_t s_eeTimerFlushAtCycles = 0u; // 0 = apply on the next call (state changed)
}

void PS2Memory::resetEeTimers() noexcept
{
    m_eeTimers = {};
    s_eeTimerPendingCycles = 0u;
    s_eeTimerFlushAtCycles = 0u;
}

uint32_t PS2Memory::advanceEeTimers(uint64_t eeCycles) noexcept
{
    if (eeCycles == 0u)
    {
        return 0u;
    }

    s_eeTimerPendingCycles += eeCycles;
    if (s_eeTimerPendingCycles < s_eeTimerFlushAtCycles)
    {
        return 0u;
    }
    eeCycles = s_eeTimerPendingCycles;
    s_eeTimerPendingCycles = 0u;

    uint32_t interruptMask = 0u;
    for (size_t index = 0; index < m_eeTimers.size(); ++index)
    {
        EeTimer &timer = m_eeTimers[index];
        if ((timer.mode & kEeTimerModeCue) == 0u)
        {
            continue;
        }

        const uint64_t clockHz = kEeTimerClockHz[timer.mode & kEeTimerModeClksMask];
        const uint64_t wholeSeconds = eeCycles / kEeClockHz;
        const uint64_t remainingCycles = eeCycles % kEeClockHz;
        const uint64_t scaled = remainingCycles * clockHz + timer.clockRemainder;
        const uint64_t ticks = wholeSeconds * clockHz + scaled / kEeClockHz;
        timer.clockRemainder = scaled % kEeClockHz;
        if (ticks == 0u)
        {
            continue;
        }

        const uint32_t oldCount = timer.count & 0xFFFFu;
        const uint32_t compare = timer.compare & 0xFFFFu;
        const uint64_t compareDistance = ticksUntilMatch(oldCount, compare);
        const uint64_t overflowDistance = 0x10000ull - oldCount;
        const bool zeroReturn = (timer.mode & kEeTimerModeZret) != 0u;
        const bool compareReached = ticks >= compareDistance;
        bool overflowReached = false;

        if (zeroReturn)
        {
            overflowReached = ticks >= overflowDistance && overflowDistance <= compareDistance;
            if (compareReached)
            {
                const uint64_t remaining = ticks - compareDistance;
                timer.count = compare == 0u
                                  ? static_cast<uint32_t>(remaining & 0xFFFFu)
                                  : static_cast<uint32_t>(remaining % compare);
            }
            else
            {
                timer.count = static_cast<uint32_t>((oldCount + ticks) & 0xFFFFu);
            }
        }
        else
        {
            overflowReached = ticks >= overflowDistance;
            timer.count = static_cast<uint32_t>((oldCount + ticks) & 0xFFFFu);
        }

        if (compareReached && (timer.mode & kEeTimerModeCmpe) != 0u && (timer.mode & kEeTimerModeEquf) == 0u)
        {
            timer.mode |= kEeTimerModeEquf;
            interruptMask |= 1u << index;
        }
        if (overflowReached && (timer.mode & kEeTimerModeOvfe) != 0u && (timer.mode & kEeTimerModeOvff) == 0u)
        {
            timer.mode |= kEeTimerModeOvff;
            interruptMask |= 1u << index;
        }
    }
    // Next batch boundary (s_eeTimerPendingCycles is 0 here, so this does not
    // re-enter the flush below).
    s_eeTimerFlushAtCycles = cyclesUntilNextEeTimerInterrupt();
    return interruptMask;
}

uint64_t PS2Memory::cyclesUntilNextEeTimerInterrupt() const noexcept
{
    if (s_eeTimerPendingCycles != 0u)
    {
        // Below the event threshold by construction, so no interrupt is lost.
        const uint64_t pending = s_eeTimerPendingCycles;
        s_eeTimerFlushAtCycles = 0u;
        s_eeTimerPendingCycles = 0u;
        (void)const_cast<PS2Memory *>(this)->advanceEeTimers(pending);
    }
    uint64_t nearest = std::numeric_limits<uint64_t>::max();
    for (const EeTimer &timer : m_eeTimers)
    {
        if ((timer.mode & kEeTimerModeCue) == 0u)
        {
            continue;
        }

        const uint32_t count = timer.count & 0xFFFFu;
        const uint32_t compare = timer.compare & 0xFFFFu;
        const uint64_t compareDistance = ticksUntilMatch(count, compare);
        const uint64_t overflowDistance = 0x10000ull - count;
        uint64_t eventTicks = std::numeric_limits<uint64_t>::max();

        if ((timer.mode & kEeTimerModeCmpe) != 0u &&
            (timer.mode & kEeTimerModeEquf) == 0u)
        {
            eventTicks = compareDistance;
        }
        const bool overflowCanOccur = (timer.mode & kEeTimerModeZret) == 0u ||
                                      overflowDistance <= compareDistance;
        if (overflowCanOccur &&
            (timer.mode & kEeTimerModeOvfe) != 0u &&
            (timer.mode & kEeTimerModeOvff) == 0u)
        {
            eventTicks = std::min(eventTicks, overflowDistance);
        }
        if (eventTicks == std::numeric_limits<uint64_t>::max())
        {
            continue;
        }

        const uint64_t clockHz = kEeTimerClockHz[timer.mode & kEeTimerModeClksMask];
        const uint64_t numerator = eventTicks * kEeClockHz - timer.clockRemainder;
        const uint64_t cycles = (numerator + clockHz - 1u) / clockHz;
        nearest = std::min(nearest, std::max<uint64_t>(1u, cycles));
    }
    return nearest;
}

bool PS2Memory::isScratchpad(uint32_t address) const
{
    return ps2IsScratchpadAddress(address);
}

uint8_t *PS2Memory::mapVuMemory(uint32_t physAddr, uint32_t size, uint32_t &offset, uint32_t &limit)
{
    return const_cast<uint8_t *>(static_cast<const PS2Memory *>(this)->mapVuMemory(physAddr, size, offset, limit));
}

const uint8_t *PS2Memory::mapVuMemory(uint32_t physAddr, uint32_t size, uint32_t &offset, uint32_t &limit) const
{
    auto mapRange = [&](uint32_t base, uint32_t rangeSize, const uint8_t *ptr) -> const uint8_t *
    {
        if (!ptr || physAddr < base)
        {
            return nullptr;
        }
        const uint32_t local = physAddr - base;
        if (local >= rangeSize || size > (rangeSize - local))
        {
            return nullptr;
        }
        offset = local;
        limit = rangeSize;
        return ptr;
    };

    if (const uint8_t *ptr = mapRange(PS2_VU0_CODE_BASE, PS2_VU0_CODE_SIZE, m_vu0Code))
    {
        return ptr;
    }
    if (const uint8_t *ptr = mapRange(PS2_VU0_DATA_BASE, PS2_VU0_DATA_SIZE, m_vu0Data))
    {
        return ptr;
    }
    if ((physAddr >= PS2_VU1_CODE_BASE && physAddr < PS2_VU1_CODE_BASE + PS2_VU1_CODE_SIZE) ||
        (physAddr >= PS2_VU1_DATA_BASE && physAddr < PS2_VU1_DATA_BASE + PS2_VU1_DATA_SIZE))
        { g_ps2xDrainSite = 739; const_cast<PS2Memory *>(this)->drainVu1Worker(); g_ps2xDrainSite = 0; } // VU1 memory belongs to the worker
    if (const uint8_t *ptr = mapRange(PS2_VU1_CODE_BASE, PS2_VU1_CODE_SIZE, m_vu1Code))
    {
        return ptr;
    }
    return mapRange(PS2_VU1_DATA_BASE, PS2_VU1_DATA_SIZE, m_vu1Data);
}

uint32_t PS2Memory::translateAddress(uint32_t virtualAddress)
{
    if (isScratchpad(virtualAddress))
    {
        return ps2ScratchpadOffset(virtualAddress);
    }

    // EE uncached aliases of main RAM (per PS2 memory map):
    //   0x20000000-0x3FFFFFFF -> 32MB mirror of RDRAM
    // This includes the accelerated window rooted at 0x30100000.
    if (Ps2IsUncachedRamMirrorAddress(virtualAddress))
    {
        return virtualAddress & PS2_RAM_MASK;
    }

    // KSEG0/KSEG1 direct-mapped window.
    if (Ps2IsKseg01Address(virtualAddress))
    {
        return Ps2DirectMappedPhysicalAddress(virtualAddress);
    }

    // In this runtime, low segments are treated as physical-style addresses already.
    if (virtualAddress < 0x80000000)
    {
        return virtualAddress;
    }

    // KSEG2/KSEG3 are TLB mapped.
    if (Ps2IsKseg23Address(virtualAddress))
    {
        for (const auto &entry : m_tlbEntries)
        {
            if (entry.valid)
            {
                // PageMask uses bits [24:13]. Build an address-level mask (plus 4KB base page bits).
                const uint32_t mask = entry.mask & 0x01FFE000u;
                const uint32_t compareMask = ~(mask | 0xFFFu);
                if ((virtualAddress & compareMask) == (entry.vpn & compareMask))
                {
                    // TLB hit
                    const uint32_t pageOffsetMask = mask | 0xFFFu;
                    const uint32_t physBase = entry.pfn << 12;
                    return physBase | (virtualAddress & pageOffsetMask);
                }
            }
        }
        throw std::runtime_error("TLB miss for address: 0x" + std::to_string(virtualAddress));
    }

    return virtualAddress;
}

bool PS2Memory::tlbRead(uint32_t index, uint32_t &vpn, uint32_t &pfn, uint32_t &mask, bool &valid) const
{
    if (index >= m_tlbEntries.size())
    {
        return false;
    }

    const TLBEntry &entry = m_tlbEntries[index];
    vpn = entry.vpn;
    pfn = entry.pfn;
    mask = entry.mask;
    valid = entry.valid;
    return true;
}

bool PS2Memory::tlbWrite(uint32_t index, uint32_t vpn, uint32_t pfn, uint32_t mask, bool valid)
{
    if (index >= m_tlbEntries.size())
    {
        return false;
    }

    TLBEntry &entry = m_tlbEntries[index];
    entry.vpn = vpn & 0xFFFFF000u;
    entry.pfn = pfn & 0x000FFFFFu;
    entry.mask = mask & 0x01FFE000u;
    entry.valid = valid;
    return true;
}

int32_t PS2Memory::tlbProbe(uint32_t vpn) const
{
    const uint32_t normalizedVpn = vpn & 0xFFFFF000u;
    for (uint32_t i = 0; i < static_cast<uint32_t>(m_tlbEntries.size()); ++i)
    {
        const TLBEntry &entry = m_tlbEntries[i];
        if (!entry.valid)
        {
            continue;
        }

        const uint32_t mask = entry.mask & 0x01FFE000u;
        const uint32_t compareMask = ~(mask | 0xFFFu);
        if ((normalizedVpn & compareMask) == (entry.vpn & compareMask))
        {
            return static_cast<int32_t>(i);
        }
    }

    return -1;
}

uint8_t PS2Memory::read8(uint32_t address)
{
    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        return m_scratchpad[physAddr];
    }
    if (physAddr < PS2_RAM_SIZE)
    {
        return m_rdram[physAddr];
    }
    uint32_t vuOffset = 0;
    uint32_t vuLimit = 0;
    if (const uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint8_t), vuOffset, vuLimit))
    {
        (void)vuLimit;
        return vuMem[vuOffset];
    }
    else if (isIoRegister(physAddr))
    {
        uint32_t regAddr = physAddr & ~0x3;
        uint32_t value = readIORegister(regAddr);
        uint32_t shift = (physAddr & 3) * 8;
        return static_cast<uint8_t>((value >> shift) & 0xFF);
    }

    return 0;
}

uint16_t PS2Memory::read16(uint32_t address)
{
    if (address & 1)
    {
        throw std::runtime_error("Unaligned 16-bit read at address: 0x" + std::to_string(address));
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        return loadScalar<uint16_t>(m_scratchpad, physAddr, PS2_SCRATCHPAD_SIZE, "read16 scratchpad", address);
    }
    if (physAddr < PS2_RAM_SIZE)
    {
        return loadScalar<uint16_t>(m_rdram, physAddr, PS2_RAM_SIZE, "read16 rdram", address);
    }
    uint32_t vuOffset = 0;
    uint32_t vuLimit = 0;
    if (const uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint16_t), vuOffset, vuLimit))
    {
        return loadScalar<uint16_t>(vuMem, vuOffset, vuLimit, "read16 vu", address);
    }
    else if (isIoRegister(physAddr))
    {
        uint32_t regAddr = physAddr & ~0x3;
        uint32_t value = readIORegister(regAddr);
        uint32_t shift = (physAddr & 2) * 8;
        return static_cast<uint16_t>((value >> shift) & 0xFFFF);
    }

    return 0;
}

uint32_t PS2Memory::read32(uint32_t address)
{
    if (address & 3)
    {
        throw std::runtime_error("Unaligned 32-bit read at address: 0x" + std::to_string(address));
    }

    if (isGsPrivReg(address))
    {
        { g_ps2xDrainSite = 926; const_cast<PS2Memory *>(this)->drainVu1Worker(); g_ps2xDrainSite = 0; } // may observe VU1/GS side effects
        uint32_t off = address & 7;
        const uint32_t regOff = (address - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        if (regOff == kGsCsrRegOffset)
        {
            uint64_t val = gs_regs.csr.load();
            return (uint32_t)(val >> (off * 8));
        }
        uint64_t *reg = gsRegPtr(gs_regs, address);
        if (!reg)
            return 0;
        uint64_t val = *reg;
        return (uint32_t)(val >> (off * 8));
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        return loadScalar<uint32_t>(m_scratchpad, physAddr, PS2_SCRATCHPAD_SIZE, "read32 scratchpad", address);
    }
    if (physAddr < PS2_RAM_SIZE)
    {
        return loadScalar<uint32_t>(m_rdram, physAddr, PS2_RAM_SIZE, "read32 rdram", address);
    }
    uint32_t vuOffset = 0;
    uint32_t vuLimit = 0;
    if (const uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint32_t), vuOffset, vuLimit))
    {
        return loadScalar<uint32_t>(vuMem, vuOffset, vuLimit, "read32 vu", address);
    }
    else if (isIoRegister(physAddr))
    {
        return readIORegister(physAddr);
    }

    return 0;
}

uint64_t PS2Memory::read64(uint32_t address)
{
    if (address & 7)
    {
        throw std::runtime_error("Unaligned 64-bit read at address: 0x" + std::to_string(address));
    }

    if (isGsPrivReg(address))
    {
        { g_ps2xDrainSite = 975; const_cast<PS2Memory *>(this)->drainVu1Worker(); g_ps2xDrainSite = 0; } // may observe VU1/GS side effects
        const uint32_t regOff = (address - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        if (regOff == kGsCsrRegOffset)
        {
            return gs_regs.csr.load();
        }
        uint64_t *reg = gsRegPtr(gs_regs, address);
        return reg ? *reg : 0;
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        return loadScalar<uint64_t>(m_scratchpad, physAddr, PS2_SCRATCHPAD_SIZE, "read64 scratchpad", address);
    }
    if (physAddr < PS2_RAM_SIZE)
    {
        return loadScalar<uint64_t>(m_rdram, physAddr, PS2_RAM_SIZE, "read64 rdram", address);
    }
    uint32_t vuOffset = 0;
    uint32_t vuLimit = 0;
    if (const uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint64_t), vuOffset, vuLimit))
    {
        return loadScalar<uint64_t>(vuMem, vuOffset, vuLimit, "read64 vu", address);
    }

    // 64-bit IO read: compose from the two adjacent 32-bit IO register slots
    // to avoid any side-effects from read32 handlers.
    if (isIoRegister(address))
    {
        uint32_t lo = m_ioRegisters.count(address) ? m_ioRegisters[address] : 0u;
        uint32_t hi = m_ioRegisters.count(address + 4) ? m_ioRegisters[address + 4] : 0u;
        return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
    }
    return (uint64_t)read32(address) | ((uint64_t)read32(address + 4) << 32);
}

__m128i PS2Memory::read128(uint32_t address)
{
    if (address & 15)
    {
        throw std::runtime_error("Unaligned 128-bit read at address: 0x" + std::to_string(address));
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        inRange(physAddr, sizeof(__m128i), PS2_SCRATCHPAD_SIZE, "read128 scratchpad", address);
        return _mm_loadu_si128(reinterpret_cast<__m128i *>(&m_scratchpad[physAddr]));
    }
    if (physAddr < PS2_RAM_SIZE)
    {
        inRange(physAddr, sizeof(__m128i), PS2_RAM_SIZE, "read128 rdram", address);
        return _mm_loadu_si128(reinterpret_cast<__m128i *>(&m_rdram[physAddr]));
    }
    uint32_t vuOffset = 0;
    uint32_t vuLimit = 0;
    if (const uint8_t *vuMem = mapVuMemory(physAddr, sizeof(__m128i), vuOffset, vuLimit))
    {
        inRange(vuOffset, sizeof(__m128i), vuLimit, "read128 vu", address);
        return _mm_loadu_si128(reinterpret_cast<const __m128i *>(vuMem + vuOffset));
    }

    // 128-bit reads are primarily for quad-word loads in the EE, which are only valid for RAM areas
    // Return zeroes for unsupported areas
    return _mm_setzero_si128();
}

void PS2Memory::write8(uint32_t address, uint8_t value)
{
    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        m_scratchpad[physAddr] = value;
    }
    else if (physAddr < PS2_RAM_SIZE)
    {
        m_rdram[physAddr] = value;
    }
    else
    {
        uint32_t vuOffset = 0;
        uint32_t vuLimit = 0;
        if (uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint8_t), vuOffset, vuLimit))
        {
            (void)vuLimit;
            vuMem[vuOffset] = value;
            if (vuMem == m_vu0Code)
                markVU0CodeModified();
            else if (vuMem == m_vu1Code)
                markVU1CodeModified();
            return;
        }
    }
    if (isIoRegister(physAddr))
    {
        // IO registers - handle byte writes by modifying the appropriate byte in the word
        uint32_t regAddr = physAddr & ~0x3;
        uint32_t shift = (physAddr & 3) * 8;
        uint32_t mask = ~(0xFF << shift);
        uint32_t newValue = (m_ioRegisters[regAddr] & mask) | ((uint32_t)value << shift);
        writeIORegister(regAddr, newValue);
    }
}

void PS2Memory::write16(uint32_t address, uint16_t value)
{
    if (address & 1)
    {
        throw std::runtime_error("Unaligned 16-bit write at address: 0x" + std::to_string(address));
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        storeScalar<uint16_t>(m_scratchpad, physAddr, PS2_SCRATCHPAD_SIZE, value, "write16 scratchpad", address);
    }
    else if (physAddr < PS2_RAM_SIZE)
    {
        storeScalar<uint16_t>(m_rdram, physAddr, PS2_RAM_SIZE, value, "write16 rdram", address);
    }
    else
    {
        uint32_t vuOffset = 0;
        uint32_t vuLimit = 0;
        if (uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint16_t), vuOffset, vuLimit))
        {
            storeScalar<uint16_t>(vuMem, vuOffset, vuLimit, value, "write16 vu", address);
            if (vuMem == m_vu0Code)
                markVU0CodeModified();
            else if (vuMem == m_vu1Code)
                markVU1CodeModified();
            return;
        }
    }
    if (isIoRegister(physAddr))
    {
        uint32_t regAddr = physAddr & ~0x3;
        uint32_t shift = (physAddr & 2) * 8;
        uint32_t mask = ~(0xFFFF << shift);
        uint32_t newValue = (m_ioRegisters[regAddr] & mask) | ((uint32_t)value << shift);
        writeIORegister(regAddr, newValue);
    }
}

void PS2Memory::write32(uint32_t address, uint32_t value)
{
    if (address & 3)
    {
        throw std::runtime_error("Unaligned 32-bit write at address: 0x" + std::to_string(address));
    }

    if (isGsPrivReg(address))
    {
        { g_ps2xDrainSite = 1137; const_cast<PS2Memory *>(this)->drainVu1Worker(); g_ps2xDrainSite = 0; } // may observe VU1/GS side effects
        uint32_t off = address & 7;
        const uint32_t regOff = (address - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        if (regOff == kGsCsrRegOffset)
        {
            // CSR: bits 0..1 of the low dword are write-one-to-clear status bits.
            // Done as a single atomic RMW -- see writeCsrHalf's comment.
            writeCsrHalf(gs_regs.csr, off, value);
        }
        else if (uint64_t *reg = gsRegPtr(gs_regs, address))
        {
            uint64_t mask = 0xFFFFFFFFULL << (off * 8);
            uint64_t newVal = (*reg & ~mask) | ((uint64_t)value << (off * 8));
            *reg = newVal;
        }
        return;
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        storeScalar<uint32_t>(m_scratchpad, physAddr, PS2_SCRATCHPAD_SIZE, value, "write32 scratchpad", address);
    }
    else if (physAddr < PS2_RAM_SIZE)
    {
        // Check if this might be code modification
        markModified(address, 4);

        storeScalar<uint32_t>(m_rdram, physAddr, PS2_RAM_SIZE, value, "write32 rdram", address);
    }
    else
    {
        uint32_t vuOffset = 0;
        uint32_t vuLimit = 0;
        if (uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint32_t), vuOffset, vuLimit))
        {
            storeScalar<uint32_t>(vuMem, vuOffset, vuLimit, value, "write32 vu", address);
            if (vuMem == m_vu0Code)
                markVU0CodeModified();
            else if (vuMem == m_vu1Code)
                markVU1CodeModified();
            return;
        }
    }
    if (isIoRegister(physAddr))
    {
        writeIORegister(physAddr, value);
    }
}

void PS2Memory::write64(uint32_t address, uint64_t value)
{
    if (address & 7)
    {
        throw std::runtime_error("Unaligned 64-bit write at address: 0x" + std::to_string(address));
    }

    if (isGsPrivReg(address))
    {
        { g_ps2xDrainSite = 1198; const_cast<PS2Memory *>(this)->drainVu1Worker(); g_ps2xDrainSite = 0; } // may observe VU1/GS side effects
        const uint32_t regOff = (address - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        if (regOff == kGsCsrRegOffset)
        {
            // CSR: bits 0..1 are write-one-to-clear status bits. Done as a single
            // atomic RMW -- see writeCsrFull's comment.
            writeCsrFull(gs_regs.csr, value);
        }
        else if (uint64_t *reg = gsRegPtr(gs_regs, address))
        {
            *reg = value;
        }
        return;
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        storeScalar<uint64_t>(m_scratchpad, physAddr, PS2_SCRATCHPAD_SIZE, value, "write64 scratchpad", address);
    }
    else if (physAddr < PS2_RAM_SIZE)
    {
        markModified(address, 8);
        storeScalar<uint64_t>(m_rdram, physAddr, PS2_RAM_SIZE, value, "write64 rdram", address);
    }
    else
    {
        uint32_t vuOffset = 0;
        uint32_t vuLimit = 0;
        if (uint8_t *vuMem = mapVuMemory(physAddr, sizeof(uint64_t), vuOffset, vuLimit))
        {
            storeScalar<uint64_t>(vuMem, vuOffset, vuLimit, value, "write64 vu", address);
            if (vuMem == m_vu0Code)
                markVU0CodeModified();
            else if (vuMem == m_vu1Code)
                markVU1CodeModified();
            return;
        }
    }
    if (isIoRegister(physAddr))
    {
        write32(address, (uint32_t)value);
        write32(address + 4, (uint32_t)(value >> 32));
    }
}

void PS2Memory::write128(uint32_t address, __m128i value)
{
    if (address & 15)
    {
        throw std::runtime_error("Unaligned 128-bit write at address: 0x" + std::to_string(address));
    }

    const bool scratch = isScratchpad(address);
    uint32_t physAddr = translateAddress(address);

    if (scratch)
    {
        inRange(physAddr, sizeof(__m128i), PS2_SCRATCHPAD_SIZE, "write128 scratchpad", address);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(&m_scratchpad[physAddr]), value);
    }
    else if (physAddr < PS2_RAM_SIZE)
    {
        markModified(address, 16);
        inRange(physAddr, sizeof(__m128i), PS2_RAM_SIZE, "write128 rdram", address);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(&m_rdram[physAddr]), value);
    }
    else
    {
        uint32_t vuOffset = 0;
        uint32_t vuLimit = 0;
        if (uint8_t *vuMem = mapVuMemory(physAddr, sizeof(__m128i), vuOffset, vuLimit))
        {
            inRange(vuOffset, sizeof(__m128i), vuLimit, "write128 vu", address);
            _mm_storeu_si128(reinterpret_cast<__m128i *>(vuMem + vuOffset), value);
            if (vuMem == m_vu0Code)
                markVU0CodeModified();
            else if (vuMem == m_vu1Code)
                markVU1CodeModified();
            return;
        }
    }
    if (isIoRegister(physAddr))
    {
        // Non-RAM 128-bit stores are modeled as two 64-bit stores.
        uint64_t lo = _mm_extract_epi64(value, 0);
        uint64_t hi = _mm_extract_epi64(value, 1);

        write64(address, lo);
        write64(address + 8, hi);
    }
}

bool PS2Memory::writeIORegister(uint32_t address, uint32_t value)
{
    if ((address >= 0x10003C00u && address < 0x10003E00u) || (address >= 0x10005000u && address < 0x10006000u))
        { g_ps2xDrainSite = 1296; drainVu1Worker(); g_ps2xDrainSite = 0; } // VIF1 registers / FIFO reflect the worker's progress
    size_t timerIndex = 0u;
    uint32_t timerOffset = 0u;
    if (decodeEeTimerRegister(address, timerIndex, timerOffset))
    {
        if (s_eeTimerPendingCycles != 0u)
        {
            const uint64_t pending = s_eeTimerPendingCycles;
            s_eeTimerFlushAtCycles = 0u;
            s_eeTimerPendingCycles = 0u;
            (void)advanceEeTimers(pending);
        }
        s_eeTimerFlushAtCycles = 0u; // mode/count/compare may change the next event
        EeTimer &timer = m_eeTimers[timerIndex];
        switch (timerOffset)
        {
        case kEeTimerCountOffset:
            timer.count = value & 0xFFFFu;
            timer.clockRemainder = 0u;
            break;
        case kEeTimerModeOffset:
        {
            const uint32_t previousMode = timer.mode;
            const uint32_t status = (previousMode & kEeTimerModeStatusMask) &~(value & kEeTimerModeStatusMask);
            timer.mode = (value & kEeTimerModeConfigMask) | status;
            if (((previousMode ^ timer.mode) & (kEeTimerModeClksMask | kEeTimerModeCue)) != 0u)
            {
                timer.clockRemainder = 0u;
            }
            break;
        }
        case kEeTimerCompareOffset:
            timer.compare = value & 0xFFFFu;
            break;
        case kEeTimerHoldOffset:
            timer.hold = value & 0xFFFFu;
            break;
        default:
            return false;
        }
        return true;
    }

    if (isGsPrivReg(address))
    {
        { g_ps2xDrainSite = 1341; const_cast<PS2Memory *>(this)->drainVu1Worker(); g_ps2xDrainSite = 0; } // may observe VU1/GS side effects
        // NB: unreachable from write8/16/32/64 today since those all funnel IO
        // register writes through addresses in PS2_IO_BASE's range, which is
        // disjoint from PS2_GS_PRIV_REG_BASE; kept correct for direct callers.
        m_ioRegisters[address] = value;
        const uint32_t off = address & 7u;
        const uint32_t regOff = (address - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        if (regOff == kGsCsrRegOffset)
        {
            writeCsrHalf(gs_regs.csr, off, value);
        }
        else if (uint64_t *reg = gsRegPtr(gs_regs, address))
        {
            const uint64_t mask = 0xFFFFFFFFull << (off * 8u);
            *reg = (*reg & ~mask) | (static_cast<uint64_t>(value) << (off * 8u));
        }
        m_gsWriteCount.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    if (address >= 0x10002000 && address <= 0x10002030)
    {
        if (address == 0x10002010)
        {
            m_ioRegisters[address] = value & ~(1u << 31);
            if (value & (1u << 30))
            {
                m_ioRegisters[0x10002000] = 0;
                m_ioRegisters[0x10002020] = 0;
                m_ioRegisters[0x10002030] = 0;
            }
        }
        else
        {
            m_ioRegisters[address] = value;
        }
        return true;
    }

    if (address == 0x1000E010u)
    {
        const uint32_t current = m_ioRegisters.count(address) ? m_ioRegisters[address] : 0u;
        uint32_t status = current & 0x3FFu;
        uint32_t mask = (current >> 16) & 0x3FFu;

        // D_STAT low bits are W1C status, high bits [16..25] toggle masks on write-one.
        status &= ~(value & 0x3FFu);
        mask ^= ((value >> 16) & 0x3FFu);

        uint32_t next = (current & ~((0x3FFu) | (0x3FFu << 16) | (1u << 31)));
        next |= status | (mask << 16);
        if ((status & mask) != 0u)
            next |= (1u << 31);
        m_ioRegisters[address] = next;
        return true;
    }

    m_ioRegisters[address] = value;

    if (address >= 0x10003C00u && address < 0x10003E00u)
    {
        m_vifWriteCount.fetch_add(1, std::memory_order_relaxed);

        switch (address)
        {
        case 0x10003C10u:     // VIF1_FBRST
            if (value & 0x1u) // RST
            {
                std::memset(&vif1_regs, 0, sizeof(vif1_regs));
                m_vif1PendingPath2ImageQwc = 0u;
                m_vif1PendingPath2DirectHl = false;
            }
            if (value & 0x8u) // STC
            {
                vif1_regs.stat &= ~((1u << 8) | (1u << 9) | (1u << 10) | (1u << 11) | (1u << 12) | (1u << 13));
            }
            break;
        case 0x10003C30u:
            vif1_regs.mark = value & 0xFFFFu;
            vif1_regs.stat &= ~(1u << 6); // clear MRK flag on CPU write
            break;
        case 0x10003C40u:
            vif1_regs.cycle = value & 0xFFFFu;
            break;
        case 0x10003C50u:
            vif1_regs.mode = value & 0x3u;
            break;
        case 0x10003C60u:
            vif1_regs.num = value & 0xFFu;
            break;
        case 0x10003C70u:
            vif1_regs.mask = value;
            break;
        case 0x10003C80u:
            vif1_regs.code = value;
            break;
        case 0x10003C90u:
            vif1_regs.itops = value & 0x3FFu;
            break;
        case 0x10003CA0u:
            vif1_regs.base = value & 0x3FFu;
            break;
        case 0x10003CB0u:
            vif1_regs.ofst = value & 0x3FFu;
            break;
        case 0x10003CC0u:
            vif1_regs.tops = value & 0x3FFu;
            break;
        case 0x10003CD0u:
            vif1_regs.itop = value & 0x3FFu;
            break;
        case 0x10003CE0u:
            vif1_regs.top = value & 0x3FFu;
            break;
        default:
            break;
        }

        return true;
    }

    if (address >= 0x10003800u && address < 0x10003A00u)
    {
        m_vifWriteCount.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    if (address >= 0x10008000 && address < 0x1000F000)
    {
        if ((address & 0xFF) == 0x00 && (value & 0x100))
        {
            const auto dctrlIt = m_ioRegisters.find(0x1000E000u);
            const bool dmacEnabled = (dctrlIt == m_ioRegisters.end()) || ((dctrlIt->second & 0x1u) != 0u);
            if (!dmacEnabled)
            {
                return true;
            }

            const uint32_t channelBase = address & 0xFFFFFF00;
            const uint32_t madr = m_ioRegisters[channelBase + 0x10];
            const uint32_t qwc = m_ioRegisters[channelBase + 0x20];
            m_dmaStartCount.fetch_add(1, std::memory_order_relaxed);

            // Channel 4 (toIPU, 0x1000B000): the MPEG elementary stream the game
            // hands to the IPU. Our sceMpeg stub decodes straight from the PSS, so
            // nothing consumed these transfers and the game's ES ring (WotM viBuf,
            // which derives its fill level from D4_MADR) stayed full forever —
            // every flush/abort wait then spun. Complete the transfer instantly:
            // walk the chain, leave MADR past the last data, clear STR, raise the
            // channel interrupt.
            if (channelBase == 0x1000B000u)
            {
                uint32_t chcr = value;
                const uint32_t mode = (chcr >> 2) & 0x3u;
                uint32_t endMadr = madr + qwc * 16u;
                uint32_t tadr = m_ioRegisters[channelBase + 0x30];
                if (mode == 1u)
                {
                    for (int guard = 0; guard < 4096; ++guard)
                    {
                        uint32_t physTag = 0u;
                        try { physTag = translateAddress(tadr); } catch (...) { break; }
                        if (physTag + 16u > PS2_RAM_SIZE)
                            break;
                        uint64_t tag = 0u;
                        std::memcpy(&tag, m_rdram + physTag, sizeof(tag));
                        const uint32_t tqwc = static_cast<uint32_t>(tag & 0xFFFFu);
                        const uint32_t id = static_cast<uint32_t>((tag >> 28) & 0x7u);
                        const uint32_t addr = static_cast<uint32_t>((tag >> 32) & 0x7FFFFFFFu);
                        bool done = false;
                        switch (id)
                        {
                        case 0: endMadr = addr + tqwc * 16u; tadr += 16u; done = true; break;          // refe
                        case 1: endMadr = tadr + 16u + tqwc * 16u; tadr = endMadr; break;              // cnt
                        case 2: endMadr = tadr + 16u + tqwc * 16u; tadr = addr; break;                 // next
                        case 3: case 4: endMadr = addr + tqwc * 16u; tadr += 16u; break;               // ref / refs
                        case 7: endMadr = tadr + 16u + tqwc * 16u; tadr = endMadr; done = true; break; // end
                        default: done = true; break;
                        }
                        if (done || ((tag >> 31) & 1u) != 0u && (chcr & 0x80u) != 0u)
                            break;
                    }
                    m_ioRegisters[channelBase + 0x30] = tadr;
                }
                m_ioRegisters[channelBase + 0x10] = endMadr;
                m_ioRegisters[channelBase + 0x20] = 0u;
                m_ioRegisters[channelBase + 0x00] = chcr & ~0x100u;
                uint32_t dstat = m_ioRegisters.count(0x1000E010u) ? m_ioRegisters[0x1000E010u] : 0u;
                dstat |= (1u << 4u);
                if (((dstat & 0x3FFu) & ((dstat >> 16u) & 0x3FFu)) != 0u)
                    dstat |= (1u << 31u);
                m_ioRegisters[0x1000E010u] = dstat;
                queueCompletedDmacCause(4u);
                return true;
            }
            if (channelBase == 0x1000A000u)
                g_ps2xGifKicks.fetch_add(1u, std::memory_order_relaxed);
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
            if (channelBase == 0x1000A000u || channelBase == 0x10009000u)
            {
                static std::atomic<uint32_t> s_kick{0u};
                const uint32_t kn = s_kick.fetch_add(1u, std::memory_order_relaxed);
                if (kn < 128u || (channelBase == 0x1000A000u && value == 0x105u && (kn % 64u) == 0u))
                    std::fprintf(stderr, "[dmakick] #%u ch=%s chcr=0x%x mode=%u madr=0x%x qwc=%u tadr=0x%x path3Masked=%d\n", kn,
                                 (channelBase == 0x1000A000u) ? "GIF" : "VIF1", value, (value >> 2) & 3u,
                                 madr, qwc, m_ioRegisters[channelBase + 0x30], m_path3Masked ? 1 : 0);
            }
#endif

            if ((channelBase == 0x1000A000u || channelBase == 0x10009000u || channelBase == 0x10008000u) &&
                (m_gsVRAM || channelBase == 0x10008000u))
            {
                auto enqueueTransfer = [&](uint32_t srcAddr, uint32_t qwCount)
                {
                    if (qwCount == 0)
                        return;
                    const bool scratch = isScratchpad(srcAddr);
                    PendingTransfer pt;
                    pt.fromScratchpad = scratch;
                    pt.srcAddr = srcAddr;
                    pt.qwc = qwCount;
                    if (channelBase == 0x1000A000u)
                        m_pendingGifTransfers.push_back(pt);
                    else if (channelBase == 0x10009000u)
                        m_pendingVif1Transfers.push_back(pt);
                    else if (channelBase == 0x10008000u)
                        m_pendingVif0Transfers.push_back(pt);
                };

                uint32_t chcr = value;
                uint32_t mode = (chcr >> 2) & 0x3;

                if (mode == 0 && qwc > 0)
                {
                    enqueueTransfer(madr, qwc);
                }
                else if (mode == 1)
                {
                    uint32_t tagAddr = m_ioRegisters[channelBase + 0x30];
                    uint32_t asr0 = m_ioRegisters[channelBase + 0x40];
                    uint32_t asr1 = m_ioRegisters[channelBase + 0x50];
                    uint32_t asp = (chcr >> 4) & 0x3u;
                    const bool tieEnabled = (chcr & (1u << 7)) != 0u;
                    const int kMaxChainTags = 4096;
                    std::vector<uint8_t> chainBuf;
                    const bool traceMotion = channelBase == 0x10009000u && MotionProvenance::enabled();
                    std::vector<MotionProvenance::Span> motionSpans;
                    std::array<std::shared_ptr<const MotionProvenance::Object>,2> motionCallOwners{};
                    std::array<uint64_t,2> motionCallHash{};
                    uint32_t motionRootTag=0;

                    auto appendData = [&](uint32_t srcAddr, uint32_t qwCount)
                    {
                        const uint64_t bytes64 = static_cast<uint64_t>(qwCount) * 16ull;
                        uint32_t bytes = (bytes64 > 0xFFFFFFFFull) ? 0xFFFFFFFFu : static_cast<uint32_t>(bytes64);
                        const bool scratch = isScratchpad(srcAddr);
                        uint32_t src = 0;
                        src = translateAddress(srcAddr);
                        const uint8_t *base2;
                        uint32_t maxSz2;
                        if (scratch)
                        {
                            base2 = m_scratchpad;
                            maxSz2 = PS2_SCRATCHPAD_SIZE;
                        }
                        else
                        {
                            base2 = m_rdram;
                            maxSz2 = PS2_RAM_SIZE;
                        }

                        while (bytes > 0)
                        {
                            if (src >= maxSz2)
                                src = 0;
                            uint32_t chunk = bytes;
                            if (src + chunk > maxSz2)
                                chunk = maxSz2 - src;
                            if (chunk == 0)
                                break;
                            chainBuf.insert(chainBuf.end(), base2 + src, base2 + src + chunk);
                            bytes -= chunk;
                            src += chunk;
                        }
                    };

                    auto appendCompactVif1TagData = [&](uint32_t localTagAddr, uint32_t qwCount)
                    {
                        uint32_t tagPhys = 0u;
                        const bool tagScratch = isScratchpad(localTagAddr);
                        tagPhys = translateAddress(localTagAddr);

                        const uint8_t *localBase = tagScratch ? m_scratchpad : m_rdram;
                        const uint32_t localMax = tagScratch ? PS2_SCRATCHPAD_SIZE : PS2_RAM_SIZE;
                        if (tagPhys + 16u > localMax)
                            return;

                        // VIF packet helpers embed 8 bytes of VIF stream in the DMAtag's upper half.
                        chainBuf.insert(chainBuf.end(), localBase + tagPhys + 8u, localBase + tagPhys + 16u);
                        appendData(localTagAddr + 16u, qwCount);
                    };

                    int tagsProcessed = 0;
                    uint32_t lastTagUpper = (chcr >> 16) & 0xFFFFu;

                    while (tagsProcessed < kMaxChainTags)
                    {
                        const uint32_t currentTagAddr = tagAddr;
                        const bool tagInSPR = isScratchpad(tagAddr);
                        uint32_t physTag = 0;
                        try
                        {
                            physTag = translateAddress(tagAddr);
                        }
                        catch (...)
                        {
                            break;
                        }
                        const uint8_t *tagBase;
                        uint32_t tagMax;
                        if (tagInSPR)
                        {
                            tagBase = m_scratchpad;
                            tagMax = PS2_SCRATCHPAD_SIZE;
                        }
                        else
                        {
                            tagBase = m_rdram;
                            tagMax = PS2_RAM_SIZE;
                        }
                        if (physTag + 16 > tagMax)
                            break;

                        const uint8_t *tp = tagBase + physTag;
                        uint64_t tag = loadScalar<uint64_t>(tp, 0, 16, "dma chain tag", tagAddr);
                        uint16_t tagQwc = static_cast<uint16_t>(tag & 0xFFFF);
                        uint32_t id = static_cast<uint32_t>((tag >> 28) & 0x7);
                        const bool irq = ((tag >> 31) & 0x1ull) != 0ull;
                        uint32_t addr = static_cast<uint32_t>((tag >> 32) & 0x7FFFFFFF);
                        lastTagUpper = static_cast<uint32_t>((tag >> 16) & 0xFFFFu);
                        ++tagsProcessed;

                        uint32_t dataAddr = 0;
                        bool hasPayload = (tagQwc > 0);
                        bool endChain = false;

                        auto motionOwner = traceMotion && !tagInSPR ? MotionProvenance::lookup(physTag) : nullptr;
                        if(traceMotion && !motionOwner && asp>0 && asp<=2)motionOwner=motionCallOwners[asp-1];
                        const uint32_t motionDepth=asp;
                        const uint32_t motionRoot=asp>0 ? motionRootTag : currentTagAddr;
                        if(traceMotion && id==5 && asp==0)motionRootTag=currentTagAddr;
                        static const bool traceOwners=std::getenv("PS2X_MOTION_OWNER_TRACE")!=nullptr;
                        if(traceMotion && traceOwners && id==5 && asp==0 && g_ps2xWotmCompletedFrames.load(std::memory_order_relaxed)==3000)
                            std::fprintf(stderr,"[motion:root] tag=%x target=%x owner=%x\n",physTag,addr,motionOwner?motionOwner->node:0);
                        if(traceMotion && id==5 && asp<2){motionCallOwners[asp]=motionOwner;motionCallHash[asp]=14695981039346656037ull;}
                        switch (id)
                        {
                        case 0:
                            dataAddr = addr;
                            tagAddr = tagAddr + 16;
                            endChain = true;
                            break;
                        case 1:
                            dataAddr = tagAddr + 16;
                            tagAddr = dataAddr + static_cast<uint32_t>(tagQwc) * 16u;
                            break;
                        case 2:
                            dataAddr = tagAddr + 16;
                            tagAddr = addr;
                            break;
                        case 3:
                        case 4:
                            dataAddr = addr;
                            tagAddr = tagAddr + 16;
                            break;
                        case 5:
                            dataAddr = tagAddr + 16;
                            {
                                const uint32_t retAddr = dataAddr + static_cast<uint32_t>(tagQwc) * 16u;
                                if (asp == 0u)
                                {
                                    asr0 = retAddr;
                                    asp = 1u;
                                }
                                else if (asp == 1u)
                                {
                                    asr1 = retAddr;
                                    asp = 2u;
                                }
                            }
                            tagAddr = addr;
                            break;
                        case 6:
                            dataAddr = tagAddr + 16;
                            if (asp == 2u)
                            {
                                tagAddr = asr1;
                                asp = 1u;
                            }
                            else if (asp == 1u)
                            {
                                tagAddr = asr0;
                                asp = 0u;
                            }
                            else
                            {
                                endChain = true;
                            }
                            break;
                        case 7:
                            dataAddr = tagAddr + 16;
                            endChain = true;
                            break;
                        default:
                            hasPayload = false;
                            endChain = true;
                            break;
                        }

#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
                        // [dmatag] — every VIF1 (0x10009000) chain tag walked during a
                        // settled menu frame. Diff the id/addr/qwc/embedded-VIFcode
                        // sequence vs retail (walk from CHCR-captured TADR 0x446700).
                        if (channelBase == 0x1000A000u)
                        {
                            // [giftag] — every GIF-channel (PATH3) chain tag at boot.
                            static std::atomic<uint32_t> s_giftag{0u};
                            if (s_giftag.fetch_add(1u, std::memory_order_relaxed) < 160u)
                            {
                                uint64_t pq0 = 0u, pq0h = 0u;
                                try
                                {
                                    const uint32_t dphys = translateAddress(dataAddr);
                                    const uint8_t *dbase = isScratchpad(dataAddr) ? m_scratchpad : m_rdram;
                                    const uint32_t dmax = isScratchpad(dataAddr) ? PS2_SCRATCHPAD_SIZE : PS2_RAM_SIZE;
                                    if (hasPayload && dphys + 16u <= dmax)
                                    {
                                        std::memcpy(&pq0, dbase + dphys, 8u);
                                        std::memcpy(&pq0h, dbase + dphys + 8u, 8u);
                                    }
                                }
                                catch (...) {}
                                std::fprintf(stderr,
                                    "[giftag] #%u tagAddr=0x%x id=%u qwc=%u addr=0x%x irq=%d dataAddr=0x%x q0=%016llx.%016llx\n",
                                    tagsProcessed, currentTagAddr, id, (unsigned)tagQwc, addr, irq ? 1 : 0, dataAddr,
                                    (unsigned long long)pq0h, (unsigned long long)pq0);
                            }
                        }
                        if (channelBase == 0x10009000u)
                        {
                            // [viftag0] — boot-time VIF1 chain tags (not settled-gated): full tag
                            // qword + first 3 payload qwords, to check DIRECT data alignment.
                            static std::atomic<uint32_t> s_viftag0{0u};
                            if (s_viftag0.fetch_add(1u, std::memory_order_relaxed) < 120u)
                            {
                                uint64_t tw[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                                const uint32_t avail = (physTag + 64u <= tagMax) ? 64u : 16u;
                                std::memcpy(tw, tp, avail);
                                std::fprintf(stderr,
                                    "[viftag0] #%u tagAddr=0x%x id=%u qwc=%u addr=0x%x tag=%016llx.%016llx "
                                    "p0=%016llx.%016llx p1=%016llx.%016llx p2=%016llx.%016llx\n",
                                    tagsProcessed, currentTagAddr, id, (unsigned)tagQwc, addr,
                                    (unsigned long long)tw[1], (unsigned long long)tw[0],
                                    (unsigned long long)tw[3], (unsigned long long)tw[2],
                                    (unsigned long long)tw[5], (unsigned long long)tw[4],
                                    (unsigned long long)tw[7], (unsigned long long)tw[6]);
                            }
                            static std::atomic<uint32_t> s_dmatag{0u};
                            const bool settled = (read32(0x6F7E8Cu) == 0u && read32(0x6F8464u) == 1u);
                            if (settled && s_dmatag.fetch_add(1u, std::memory_order_relaxed) < 1500u)
                            {
                                const uint32_t vc1 = (physTag + 12u <= tagMax)
                                    ? *reinterpret_cast<const uint32_t *>(tp + 8) : 0xFFFFFFFFu;
                                const uint32_t vc2 = (physTag + 16u <= tagMax)
                                    ? *reinterpret_cast<const uint32_t *>(tp + 12) : 0xFFFFFFFFu;
                                uint32_t pq0 = 0u, pq1 = 0u, pq2 = 0u, pq3 = 0u;
                                if (physTag + 32u <= tagMax)
                                {
                                    pq0 = *reinterpret_cast<const uint32_t *>(tp + 16);
                                    pq1 = *reinterpret_cast<const uint32_t *>(tp + 20);
                                    pq2 = *reinterpret_cast<const uint32_t *>(tp + 24);
                                    pq3 = *reinterpret_cast<const uint32_t *>(tp + 28);
                                }
                                std::fprintf(stderr,
                                    "[dmatag] #%u tagAddr=0x%x id=%u qwc=%u addr=0x%x "
                                    "vc=[%08x %08x] pay0=[%08x %08x %08x %08x]\n",
                                    tagsProcessed, currentTagAddr, id, (unsigned)tagQwc, addr,
                                    vc1, vc2, pq0, pq1, pq2, pq3);
                            }
                        }
#endif

                        const uint32_t motionBegin = static_cast<uint32_t>(chainBuf.size());
                        const bool compactVifLocalTag =
                            (channelBase == 0x10009000u || channelBase == 0x10008000u) &&
                            (id == 1u || id == 2u || id == 5u || id == 6u || id == 7u);
                        if (compactVifLocalTag)
                            appendCompactVif1TagData(currentTagAddr, 0u);

                        if (hasPayload)
                        {
                            if (compactVifLocalTag)
                                appendData(currentTagAddr + 16u, tagQwc);
                            else
                                appendData(dataAddr, tagQwc);
                        }
                        if (traceMotion && chainBuf.size() > motionBegin) {
                            uint64_t signature=0;
                            if(MotionProvenance::vectorsEnabled() && motionDepth==1) {
                                auto& hash=motionCallHash[0];
                                const auto* payload=chainBuf.data()+motionBegin;
                                const size_t payloadBytes=chainBuf.size()-motionBegin;
                                static const bool cached=[] {const char* p=std::getenv("PS2X_MOTION_SIGNATURE_CACHE");return !p || p[0]!='0';}();
                                static const bool verify=[] {const char* p=std::getenv("PS2X_MOTION_SIGNATURE_VERIFY");return p && p[0]=='1';}();
                                static const bool stats=std::getenv("PS2X_MOTION_SIGNATURE_STATS")!=nullptr;
                                static thread_local WotmMotionSignatureCache cache;
                                // The continuation seed is part of the key. Full byte
                                // equality is required, even for the same guest address.
                                signature=cached && payloadBytes>=WotmMotionSignatureCache::minBytes
                                    ? cache.hash(currentTagAddr,hash,payload,payloadBytes)
                                    : WotmMotionSignatureCache::reference(hash,payload,payloadBytes);
                                static thread_local uint64_t calls=0,checks=0;
                                if(verify || stats) {
                                    ++calls;
                                    if(verify && (calls&255u)==0u) {
                                        if(signature!=WotmMotionSignatureCache::reference(hash,payload,payloadBytes)) {
                                            std::fprintf(stderr,"[motion:signature] MISMATCH tag=%x bytes=%zu\n",currentTagAddr,payloadBytes);std::abort();
                                        }
                                        ++checks;
                                    }
                                    if((calls&65535u)==0u)
                                        std::fprintf(stderr,"[motion:signature] enabled=%d calls=%llu hits=%llu misses=%llu compared-bytes=%llu hashed-bytes=%llu checks=%llu mismatches=0\n",
                                            cached,(unsigned long long)calls,(unsigned long long)cache.hits,
                                            (unsigned long long)cache.misses,(unsigned long long)cache.comparedBytes,
                                            (unsigned long long)cache.hashedBytes,(unsigned long long)checks);
                                }
                                hash=signature;
                            }
                            motionSpans.push_back({motionBegin,static_cast<uint32_t>(chainBuf.size()),currentTagAddr,motionOwner,signature,motionRoot});
                        }
                        if (irq && tieEnabled)
                            endChain = true;
                        if (endChain)
                            break;
                    }

                    m_ioRegisters[channelBase + 0x30] = tagAddr;
                    m_ioRegisters[channelBase + 0x40] = asr0;
                    m_ioRegisters[channelBase + 0x50] = asr1;
                    chcr = (chcr & ~(0x3u << 4)) | ((asp & 0x3u) << 4);
                    chcr = (chcr & 0x0000FFFFu) | (lastTagUpper << 16);
                    m_ioRegisters[channelBase + 0x00] = chcr;

                    if (!chainBuf.empty())
                    {
                        PendingTransfer pt;
                        pt.fromScratchpad = false;
                        pt.srcAddr = 0;
                        pt.qwc = 0;
                        if (traceMotion)MotionProvenance::publish(chainBuf.data(),std::move(motionSpans));
                        pt.chainData = std::move(chainBuf);
                        if (channelBase == 0x1000A000)
                        {
                            m_pendingGifTransfers.push_back(std::move(pt));
                        }
                        else if (channelBase == 0x10009000u)
                        {
                            m_pendingVif1Transfers.push_back(std::move(pt));
                        }
                        else if (channelBase == 0x10008000u)
                        {
                            m_pendingVif0Transfers.push_back(std::move(pt));
                        }
                    }
                    // else if (channelBase == 0x10009000u)
                    // {

                    // }
                }
                else if (qwc > 0)
                {
                    enqueueTransfer(madr, qwc);
                }

                const bool autoProcessTransfers =
                    (channelBase == 0x1000A000u) ? (m_gifPacketCallback || m_gifArbiter != nullptr) : true;
                if (autoProcessTransfers)
                {
                    processPendingTransfers();
                }
            }
        }
        return true;
    }

    if (address >= 0x10000000 && address < 0x10010000)
    {
        if (address >= 0x10000200 && address < 0x10000300)
        {
            return true;
        }
        if (address >= 0x10000000 && address < 0x10000100)
        {
            return true;
        }
    }

    return false;
}

void PS2Memory::processPendingTransfers()
{
    // Threaded VU1: VIF1 goes to the worker. A call that also carries a GIF
    // (PATH3) transfer runs the whole thing synchronously instead, after the
    // worker has caught up -- the arbiter orders PATH1 before PATH3 within one
    // call, and splitting the call across threads would change that order.
    // PATH3 GIF transfers are queued on the worker too (in call order, before
    // this call's VIF1 work, as the synchronous path runs them), so draw order
    // holds without the game thread waiting.
    const bool vu1Async = s_vu1Threaded;
    Vu1Job path3Job;
    path3Job.path3 = true;
    if (s_vu1Threaded)
        startVu1WorkerIfNeeded();
    const auto gifOut = [&](const uint8_t *data, uint32_t size)
    {
        if (vu1Async)
        {
            path3Job.chunks.emplace_back(data, data + size);
            return;
        }
        m_seenGifCopy = true;
        m_gifCopyCount.fetch_add(1, std::memory_order_relaxed);
        submitGifPacket(GifPathId::Path3, data, size, false);
    };
    const bool hadGif = !m_pendingGifTransfers.empty();
    for (size_t idx = 0; idx < m_pendingGifTransfers.size(); ++idx)
    {
        auto &p = m_pendingGifTransfers[idx];
        if (!p.chainData.empty())
        {
            gifOut(p.chainData.data(), static_cast<uint32_t>(p.chainData.size()));
        }
        else if (p.qwc > 0)
        {
            const uint64_t bytes64 = static_cast<uint64_t>(p.qwc) * 16ull;
            uint32_t sizeBytes = (bytes64 > 0xFFFFFFFFull) ? 0xFFFFFFFFu : static_cast<uint32_t>(bytes64);
            uint32_t srcPhys = 0;
            try
            {
                srcPhys = translateAddress(p.srcAddr);
            }
            catch (const std::exception &)
            {
                continue;
            }
            if (p.fromScratchpad)
            {
                uint32_t bytesLeft = sizeBytes;
                while (bytesLeft >= 16)
                {
                    if (srcPhys >= PS2_SCRATCHPAD_SIZE)
                        srcPhys = 0;
                    uint32_t chunk = bytesLeft;
                    if (srcPhys + chunk > PS2_SCRATCHPAD_SIZE)
                        chunk = PS2_SCRATCHPAD_SIZE - srcPhys;
                    if (chunk == 0)
                        break;
                    gifOut(m_scratchpad + srcPhys, chunk);
                    bytesLeft -= chunk;
                    srcPhys += chunk;
                }
            }
            else
            {
                uint32_t bytesLeft = sizeBytes;
                while (bytesLeft >= 16)
                {
                    if (srcPhys >= PS2_RAM_SIZE)
                        srcPhys = 0;
                    uint32_t chunk = bytesLeft;
                    if (srcPhys + chunk > PS2_RAM_SIZE)
                        chunk = PS2_RAM_SIZE - srcPhys;
                    if (chunk == 0)
                        break;
                    gifOut(m_rdram + srcPhys, chunk);
                    bytesLeft -= chunk;
                    srcPhys += chunk;
                }
            }
        }
    }
    m_pendingGifTransfers.clear();
    if (vu1Async && !path3Job.chunks.empty())
        submitVu1Job(std::move(path3Job));

    const bool hadVif0 = !m_pendingVif0Transfers.empty();
    for (auto &p : m_pendingVif0Transfers)
    {
        if (!p.chainData.empty())
        {
            processVIF0Data(p.chainData.data(), static_cast<uint32_t>(p.chainData.size()));
        }
        else if (p.qwc > 0)
        {
            uint32_t srcPhys = 0;
            const uint64_t bytes64 = static_cast<uint64_t>(p.qwc) * 16ull;
            uint32_t sizeBytes = (bytes64 > 0xFFFFFFFFull) ? 0xFFFFFFFFu : static_cast<uint32_t>(bytes64);
            try
            {
                srcPhys = translateAddress(p.srcAddr);
            }
            catch (const std::exception &)
            {
                continue;
            }
            if (p.fromScratchpad)
            {
                uint32_t bytesLeft = sizeBytes;
                while (bytesLeft > 0)
                {
                    if (srcPhys >= PS2_SCRATCHPAD_SIZE)
                        srcPhys = 0;
                    uint32_t chunk = bytesLeft;
                    if (srcPhys + chunk > PS2_SCRATCHPAD_SIZE)
                        chunk = PS2_SCRATCHPAD_SIZE - srcPhys;
                    if (chunk == 0)
                        break;
                    processVIF0Data(m_scratchpad + srcPhys, chunk);
                    bytesLeft -= chunk;
                    srcPhys += chunk;
                }
            }
            else
            {
                uint32_t bytesLeft = sizeBytes;
                while (bytesLeft > 0)
                {
                    if (srcPhys >= PS2_RAM_SIZE)
                        srcPhys = 0;
                    uint32_t chunk = bytesLeft;
                    if (srcPhys + chunk > PS2_RAM_SIZE)
                        chunk = PS2_RAM_SIZE - srcPhys;
                    if (chunk == 0)
                        break;
                    processVIF0Data(srcPhys, chunk);
                    bytesLeft -= chunk;
                    srcPhys += chunk;
                }
            }
        }
    }
    m_pendingVif0Transfers.clear();

    const bool hadVif1 = !m_pendingVif1Transfers.empty();
    const uint64_t vif1ProbeT0 = hadVif1 ? __rdtsc() : 0u;
    if (hadVif1)
    {
        // Close the gap that started when the previous transfer ended.
        if (g_ps2xVif1GapOpen.load(std::memory_order_relaxed))
        {
            if (g_ps2xVif1GapPolled.load(std::memory_order_relaxed))
                g_ps2xVif1GapsWithPoll.fetch_add(1u, std::memory_order_relaxed);
            if (g_ps2xVif1GapBlocked.load(std::memory_order_relaxed))
                g_ps2xVif1GapsWithBlock.fetch_add(1u, std::memory_order_relaxed);
            if (g_ps2xVif1GapWaitCall.load(std::memory_order_relaxed))
                g_ps2xVif1GapsWithWaitCall.fetch_add(1u, std::memory_order_relaxed);
            if (g_ps2xVif1GapSignal.load(std::memory_order_relaxed))
                g_ps2xVif1GapsWithSignal.fetch_add(1u, std::memory_order_relaxed);
        }
        g_ps2xVif1GapOpen.store(false, std::memory_order_relaxed);
    }
    if (vu1Async && hadVif1)
    {
        // Snapshot everything the worker needs NOW, on this thread: the game
        // may reuse the source buffer as soon as this call returns, and the
        // VU0 FBRST D/T bits live in the EE context.
        const uint32_t fbrst = m_vu1FbrstSampler ? m_vu1FbrstSampler() : 0u;
        for (auto &p : m_pendingVif1Transfers)
        {
            Vu1Job job;
            job.dBit = (fbrst & (1u << 10)) != 0u;
            job.tBit = (fbrst & (1u << 11)) != 0u;
            if (!p.chainData.empty())
            {
                job.chunks.push_back(std::move(p.chainData));
            }
            else if (p.qwc > 0)
            {
                uint32_t srcPhys = 0;
                const uint64_t bytes64 = static_cast<uint64_t>(p.qwc) * 16ull;
                uint32_t sizeBytes = (bytes64 > 0xFFFFFFFFull) ? 0xFFFFFFFFu : static_cast<uint32_t>(bytes64);
                try
                {
                    srcPhys = translateAddress(p.srcAddr);
                }
                catch (const std::exception &)
                {
                    continue;
                }
                const uint8_t *base = p.fromScratchpad ? m_scratchpad : m_rdram;
                const uint32_t limit = p.fromScratchpad ? PS2_SCRATCHPAD_SIZE : PS2_RAM_SIZE;
                uint32_t bytesLeft = sizeBytes;
                while (bytesLeft > 0)
                {
                    if (srcPhys >= limit)
                        srcPhys = 0;
                    uint32_t chunk = bytesLeft;
                    if (srcPhys + chunk > limit)
                        chunk = limit - srcPhys;
                    if (chunk == 0)
                        break;
                    job.chunks.emplace_back(base + srcPhys, base + srcPhys + chunk);
                    if(MotionProvenance::enabled())MotionProvenance::publish(job.chunks.back().data(),{});
                    bytesLeft -= chunk;
                    srcPhys += chunk;
                }
            }
            if (!job.chunks.empty())
                submitVu1Job(std::move(job));
        }
    }
    else
    for (auto &p : m_pendingVif1Transfers)
    {
        if (!p.chainData.empty())
        {
            processVIF1Data(p.chainData.data(), static_cast<uint32_t>(p.chainData.size()));
        }
        else if (p.qwc > 0)
        {
            uint32_t srcPhys = 0;
            const uint64_t bytes64 = static_cast<uint64_t>(p.qwc) * 16ull;
            uint32_t sizeBytes = (bytes64 > 0xFFFFFFFFull) ? 0xFFFFFFFFu : static_cast<uint32_t>(bytes64);
            try
            {
                srcPhys = translateAddress(p.srcAddr);
            }
            catch (const std::exception &)
            {
                continue;
            }
            if (p.fromScratchpad)
            {
                uint32_t bytesLeft = sizeBytes;
                while (bytesLeft > 0)
                {
                    if (srcPhys >= PS2_SCRATCHPAD_SIZE)
                        srcPhys = 0;
                    uint32_t chunk = bytesLeft;
                    if (srcPhys + chunk > PS2_SCRATCHPAD_SIZE)
                        chunk = PS2_SCRATCHPAD_SIZE - srcPhys;
                    if (chunk == 0)
                        break;
                    processVIF1Data(m_scratchpad + srcPhys, chunk);
                    bytesLeft -= chunk;
                    srcPhys += chunk;
                }
            }
            else
            {
                uint32_t bytesLeft = sizeBytes;
                while (bytesLeft > 0)
                {
                    if (srcPhys >= PS2_RAM_SIZE)
                        srcPhys = 0;
                    uint32_t chunk = bytesLeft;
                    if (srcPhys + chunk > PS2_RAM_SIZE)
                        chunk = PS2_RAM_SIZE - srcPhys;
                    if (chunk == 0)
                        break;
                    processVIF1Data(srcPhys, chunk);
                    bytesLeft -= chunk;
                    srcPhys += chunk;
                }
            }
        }
    }
    m_pendingVif1Transfers.clear();
    if (hadVif1)
    {
        g_ps2xVif1Transfers.fetch_add(1u, std::memory_order_relaxed);
        g_ps2xVif1InsideTsc.fetch_add(__rdtsc() - vif1ProbeT0, std::memory_order_relaxed);
        // Open a new gap: watch what the EE does until the next transfer.
        g_ps2xVif1GapPolled.store(false, std::memory_order_relaxed);
        g_ps2xVif1GapBlocked.store(false, std::memory_order_relaxed);
        g_ps2xVif1GapWaitCall.store(false, std::memory_order_relaxed);
        g_ps2xVif1GapSignal.store(false, std::memory_order_relaxed);
        g_ps2xVif1GapOpen.store(true, std::memory_order_relaxed);
    }

    // With VIF1 on the worker the arbiter belongs to it; this thread only
    // drains it on the synchronous path (where the worker is idle).
    if (m_gifArbiter && !vu1Async)
        m_gifArbiter->drain();

    static constexpr uint32_t GIF_CHANNEL = 0x1000A000;
    static constexpr uint32_t VIF0_CHANNEL = 0x10008000;
    static constexpr uint32_t VIF1_CHANNEL = 0x10009000;
    static constexpr uint32_t D_STAT = 0x1000E010u;

    auto raiseDStatChannel = [&](uint32_t channelBit)
    {
        uint32_t dstat = m_ioRegisters.count(D_STAT) ? m_ioRegisters[D_STAT] : 0u;
        dstat |= (1u << channelBit);

        const uint32_t status = dstat & 0x3FFu;
        const uint32_t mask = (dstat >> 16) & 0x3FFu;
        if ((status & mask) != 0u)
            dstat |= (1u << 31);
        else
            dstat &= ~(1u << 31);

        m_ioRegisters[D_STAT] = dstat;
    };

    if (hadGif)
    {
        raiseDStatChannel(2u); // GIF channel
        queueCompletedDmacCause(2u);
        m_ioRegisters[GIF_CHANNEL + 0x00] &= ~0x100u;
        m_ioRegisters[GIF_CHANNEL + 0x20] = 0;
    }
    if (hadVif0)
    {
        raiseDStatChannel(0u); // VIF0 channel
        queueCompletedDmacCause(0u);
        m_ioRegisters[VIF0_CHANNEL + 0x00] &= ~0x100u;
        m_ioRegisters[VIF0_CHANNEL + 0x20] = 0;
    }
    if (hadVif1)
    {
        raiseDStatChannel(1u); // VIF1 channel
        queueCompletedDmacCause(1u);
        m_ioRegisters[VIF1_CHANNEL + 0x00] &= ~0x100u;
        m_ioRegisters[VIF1_CHANNEL + 0x20] = 0;
    }
}

void PS2Memory::queueCompletedDmacCause(uint32_t cause)
{
    if (cause == 2u)
        g_ps2xCause2.fetch_add(1u, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(m_completedDmacMutex);
    m_completedDmacCauses.push_back(cause);
}

std::vector<uint32_t> PS2Memory::consumeCompletedDmacCauses()
{
    std::lock_guard<std::mutex> lock(m_completedDmacMutex);
    std::vector<uint32_t> causes;
    causes.swap(m_completedDmacCauses);
    return causes;
}

void PS2Memory::flushMaskedPath3Packets(bool drainImmediately)
{
    if (m_path3Masked || m_path3MaskedFifo.empty())
        return;

    auto emit = [&](const uint8_t *packetData, uint32_t packetSize)
    {
        if (m_gifArbiter)
            m_gifArbiter->submit(GifPathId::Path3, packetData, packetSize, false);
        else if (m_gifPacketCallback)
            m_gifPacketCallback(packetData, packetSize);
    };

    for (const auto &packet : m_path3MaskedFifo)
    {
        if (packet.size() >= 16u)
            emit(packet.data(), static_cast<uint32_t>(packet.size()));
    }
    m_path3MaskedFifo.clear();

    if (m_gifArbiter && drainImmediately)
        m_gifArbiter->drain();
}

void PS2Memory::submitGifPacket(GifPathId pathId, const uint8_t *data, uint32_t sizeBytes, bool drainImmediately, bool path2DirectHl)
{
    if (!data || sizeBytes < 16)
        return;

    if (pathId == GifPathId::Path3)
    {
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
        {
            static std::atomic<uint32_t> s_p3{0u};
            if (s_p3.fetch_add(1u, std::memory_order_relaxed) < 96u)
            {
                uint64_t q0 = 0u, q0h = 0u;
                std::memcpy(&q0, data, 8u);
                std::memcpy(&q0h, data + 8u, 8u);
                std::fprintf(stderr, "[gifsub] PATH3 bytes=%u masked=%d fifo=%zu q0=%016llx.%016llx\n",
                             sizeBytes, m_path3Masked ? 1 : 0, m_path3MaskedFifo.size(),
                             (unsigned long long)q0h, (unsigned long long)q0);
            }
        }
#endif
        if (m_path3Masked)
        {
            m_path3MaskedFifo.emplace_back(data, data + sizeBytes);
            return;
        }
        flushMaskedPath3Packets(false);
    }

    if (m_gifArbiter)
        m_gifArbiter->submit(pathId, data, sizeBytes, path2DirectHl);
    else if (m_gifPacketCallback)
        m_gifPacketCallback(data, sizeBytes);

    if (m_gifArbiter && drainImmediately)
        m_gifArbiter->drain();
}

void PS2Memory::processGIFPacket(uint32_t srcPhysAddr, uint32_t qwCount)
{
    if (!m_rdram || qwCount == 0)
        return;
    const uint64_t bytes64 = static_cast<uint64_t>(qwCount) * 16ull;
    uint32_t sizeBytes = (bytes64 > 0xFFFFFFFFull) ? 0xFFFFFFFFu : static_cast<uint32_t>(bytes64);
    uint32_t bytesLeft = sizeBytes;
    while (bytesLeft >= 16)
    {
        if (srcPhysAddr >= PS2_RAM_SIZE)
            srcPhysAddr = 0;
        uint32_t chunk = bytesLeft;
        if (srcPhysAddr + chunk > PS2_RAM_SIZE)
            chunk = PS2_RAM_SIZE - srcPhysAddr;
        if (chunk == 0)
            break;

        m_seenGifCopy = true;
        m_gifCopyCount.fetch_add(1, std::memory_order_relaxed);
        submitGifPacket(GifPathId::Path3, m_rdram + srcPhysAddr, chunk);

        bytesLeft -= chunk;
        srcPhysAddr += chunk;
    }
}

void PS2Memory::processGIFPacket(const uint8_t *data, uint32_t sizeBytes)
{
    if (m_gifArbiter)
        submitGifPacket(GifPathId::Path3, data, sizeBytes);
    else if (m_gifPacketCallback && data && sizeBytes >= 16)
        m_gifPacketCallback(data, sizeBytes);
}

bool PS2Memory::tryProcessNativeGifImageUploadChain(GS &gs, uint32_t tadr, uint32_t chcr)
{
    { g_ps2xDrainSite = 2335; drainVu1Worker(); g_ps2xDrainSite = 0; } // PATH3 into the GS: let queued PATH1 land first
    static constexpr uint32_t GIF_CHANNEL = 0x1000A000u;
    static constexpr uint32_t D_STAT = 0x1000E010u;
    static constexpr uint32_t D_CTRL = 0x1000E000u;

    if (!m_rdram || !m_gsVRAM || m_path3Masked)
        return false;
    if (m_gifArbiter && !m_gifArbiter->empty())
        return false;
    if ((chcr & 0x100u) == 0u || ((chcr >> 2u) & 0x3u) != 1u)
        return false;
    if ((chcr & (1u << 7u)) != 0u || ((chcr >> 4u) & 0x3u) != 0u)
        return false;

    const auto dctrlIt = m_ioRegisters.find(D_CTRL);
    if (dctrlIt != m_ioRegisters.end() && ((dctrlIt->second & 0x1u) == 0u))
        return false;

    auto resolveContiguous = [&](uint32_t guestAddr, uint32_t bytes, const uint8_t *&out) -> bool
    {
        try
        {
            const bool scratch = isScratchpad(guestAddr);
            const uint32_t phys = translateAddress(guestAddr);
            const uint8_t *base = scratch ? m_scratchpad : m_rdram;
            const uint32_t limit = scratch ? PS2_SCRATCHPAD_SIZE : PS2_RAM_SIZE;
            if (!base || phys > limit || bytes > limit - phys)
                return false;
            out = base + phys;
            return true;
        }
        catch (const std::exception &)
        {
            return false;
        }
    };

    auto loadDmaTagAt = [&](uint32_t guestAddr, DmaTagView &out) -> bool
    {
        const uint8_t *ptr = nullptr;
        if (!resolveContiguous(guestAddr, 16u, ptr))
            return false;
        out = decodeDmaTag(loadScalar<uint64_t>(ptr, 0u, 16u, "native gif dma tag", guestAddr));
        return true;
    };

    auto decodeSetupPayload = [&](const uint8_t *payload, uint64_t (&regs)[4]) -> bool
    {
        const uint64_t tagLo = loadScalar<uint64_t>(payload, 0u, 80u, "native gif setup tag", 0u);
        const uint64_t tagHi = loadScalar<uint64_t>(payload, 8u, 80u, "native gif setup regs", 0u);
        if (gifTagNloop(tagLo) != 4u ||
            gifTagFlg(tagLo) != GIF_FMT_PACKED ||
            gifTagNreg(tagLo) != 1u ||
            (tagHi & 0xFull) != 0x0Eull)
        {
            return false;
        }

        static constexpr uint8_t kExpectedRegs[4] = {
            GS_REG_BITBLTBUF,
            GS_REG_TRXPOS,
            GS_REG_TRXREG,
            GS_REG_TRXDIR,
        };

        uint32_t offset = 16u;
        for (uint32_t i = 0; i < 4u; ++i)
        {
            regs[i] = loadScalar<uint64_t>(payload, offset, 80u, "native gif setup value", 0u);
            const uint64_t reg = loadScalar<uint64_t>(payload, offset + 8u, 80u, "native gif setup register", 0u);
            if ((reg & 0xFFu) != kExpectedRegs[i])
                return false;
            offset += 16u;
        }

        const uint32_t trxdirMode = static_cast<uint32_t>(regs[3] & 0x3ull);
        const uint32_t rrw = static_cast<uint32_t>(regs[2] & 0xFFFull);
        const uint32_t rrh = static_cast<uint32_t>((regs[2] >> 32u) & 0xFFFull);
        return trxdirMode == 0u && rrw != 0u && rrh != 0u;
    };

    DmaTagView setupTag{};
    if (!loadDmaTagAt(tadr, setupTag) ||
        setupTag.id != 1u ||
        setupTag.qwc != 5u ||
        setupTag.irq)
    {
        return false;
    }

    const uint8_t *setupPayload = nullptr;
    const uint32_t setupPayloadAddr = tadr + 16u;
    if (!resolveContiguous(setupPayloadAddr, 5u * 16u, setupPayload))
        return false;

    uint64_t setupRegs[4] = {};
    if (!decodeSetupPayload(setupPayload, setupRegs))
        return false;

    uint32_t imageTagDmaAddr = setupPayloadAddr + 5u * 16u;
    DmaTagView imageTagDma{};
    if (!loadDmaTagAt(imageTagDmaAddr, imageTagDma) ||
        imageTagDma.id != 1u ||
        imageTagDma.qwc != 1u ||
        imageTagDma.irq)
    {
        return false;
    }

    const uint8_t *imageGifTag = nullptr;
    if (!resolveContiguous(imageTagDmaAddr + 16u, 16u, imageGifTag))
        return false;

    const uint64_t imageTagLo = loadScalar<uint64_t>(imageGifTag, 0u, 16u, "native gif image tag", imageTagDmaAddr + 16u);
    if (gifTagFlg(imageTagLo) != GIF_FMT_IMAGE)
        return false;

    const uint32_t imageQwc = gifTagNloop(imageTagLo);
    if (imageQwc == 0u)
        return false;

    const uint64_t imageBytes64 = static_cast<uint64_t>(imageQwc) * 16ull;
    if (imageBytes64 > 0xFFFFFFFFull)
        return false;
    const uint32_t imageBytes = static_cast<uint32_t>(imageBytes64);

    const uint32_t payloadTagAddr = imageTagDmaAddr + 32u;
    DmaTagView payloadTag{};
    if (!loadDmaTagAt(payloadTagAddr, payloadTag) ||
        payloadTag.qwc != imageQwc ||
        payloadTag.irq)
    {
        return false;
    }

    uint32_t imageDataAddr = 0u;
    uint32_t finalTadr = payloadTagAddr;
    uint32_t lastTagUpper = payloadTag.upper;
    if (payloadTag.id == 3u || payloadTag.id == 4u)
    {
        imageDataAddr = payloadTag.addr;
        const uint32_t terminalTagAddr = payloadTagAddr + 16u;
        DmaTagView terminalTag{};
        if (!loadDmaTagAt(terminalTagAddr, terminalTag) ||
            terminalTag.qwc != 0u ||
            terminalTag.irq ||
            (terminalTag.id != 0u && terminalTag.id != 7u))
        {
            return false;
        }
        finalTadr = (terminalTag.id == 0u) ? (terminalTagAddr + 16u) : terminalTagAddr;
        lastTagUpper = terminalTag.upper;
    }
    else if (payloadTag.id == 7u)
    {
        imageDataAddr = payloadTagAddr + 16u;
        finalTadr = payloadTagAddr;
    }
    else
    {
        return false;
    }

    const uint8_t *imageData = nullptr;
    if (!resolveContiguous(imageDataAddr, imageBytes, imageData))
        return false;

    m_dmaStartCount.fetch_add(1, std::memory_order_relaxed);
    m_seenGifCopy = true;
    m_gifCopyCount.fetch_add(1, std::memory_order_relaxed);
    gs.uploadImageNative(setupRegs[0], setupRegs[1], setupRegs[2], setupRegs[3], imageData, imageBytes);

    m_ioRegisters[GIF_CHANNEL + 0x30u] = finalTadr;
    m_ioRegisters[GIF_CHANNEL + 0x40u] = 0u;
    m_ioRegisters[GIF_CHANNEL + 0x50u] = 0u;
    m_ioRegisters[GIF_CHANNEL + 0x00u] = ((chcr & 0x0000FFFFu) | (lastTagUpper << 16u)) & ~0x100u;
    m_ioRegisters[GIF_CHANNEL + 0x20u] = 0u;

    uint32_t dstat = m_ioRegisters.count(D_STAT) ? m_ioRegisters[D_STAT] : 0u;
    dstat |= (1u << 2u);
    const uint32_t status = dstat & 0x3FFu;
    const uint32_t mask = (dstat >> 16u) & 0x3FFu;
    if ((status & mask) != 0u)
        dstat |= (1u << 31u);
    else
        dstat &= ~(1u << 31u);
    m_ioRegisters[D_STAT] = dstat;
    queueCompletedDmacCause(2u);
    return true;
}

bool PS2Memory::tryProcessNativeGifPackedChain(GS &gs, uint32_t tadr, uint32_t chcr)
{
    { g_ps2xDrainSite = 2528; drainVu1Worker(); g_ps2xDrainSite = 0; } // PATH3 into the GS: let queued PATH1 land first
    static constexpr uint32_t GIF_CHANNEL = 0x1000A000u;
    static constexpr uint32_t D_STAT = 0x1000E010u;
    static constexpr uint32_t D_CTRL = 0x1000E000u;

    if (!m_rdram || !m_gsVRAM || m_path3Masked)
        return false;
    if (m_gifArbiter && !m_gifArbiter->empty())
        return false;
    if ((chcr & 0x100u) == 0u || ((chcr >> 2u) & 0x3u) != 1u)
        return false;
    if ((chcr & (1u << 7u)) != 0u || ((chcr >> 4u) & 0x3u) != 0u)
        return false;

    const auto dctrlIt = m_ioRegisters.find(D_CTRL);
    if (dctrlIt != m_ioRegisters.end() && ((dctrlIt->second & 0x1u) == 0u))
        return false;

    auto resolveContiguous = [&](uint32_t guestAddr, uint32_t bytes, const uint8_t *&out) -> bool
    {
        try
        {
            const bool scratch = isScratchpad(guestAddr);
            const uint32_t phys = translateAddress(guestAddr);
            const uint8_t *base = scratch ? m_scratchpad : m_rdram;
            const uint32_t limit = scratch ? PS2_SCRATCHPAD_SIZE : PS2_RAM_SIZE;
            if (!base || phys > limit || bytes > limit - phys)
                return false;
            out = base + phys;
            return true;
        }
        catch (const std::exception &)
        {
            return false;
        }
    };

    const uint8_t *tagPtr = nullptr;
    if (!resolveContiguous(tadr, 16u, tagPtr))
        return false;

    const DmaTagView tag = decodeDmaTag(loadScalar<uint64_t>(tagPtr, 0u, 16u, "native packed gif dma tag", tadr));
    if (tag.id != 7u || tag.qwc == 0u || tag.irq)
        return false;

    const uint64_t payloadBytes64 = static_cast<uint64_t>(tag.qwc) * 16ull;
    if (payloadBytes64 > 0xFFFFFFFFull)
        return false;
    const uint32_t payloadBytes = static_cast<uint32_t>(payloadBytes64);

    const uint8_t *payload = nullptr;
    if (!resolveContiguous(tadr + 16u, payloadBytes, payload))
        return false;
    if (!gs.processNativePackedGIFPacket(payload, payloadBytes))
        return false;

    m_dmaStartCount.fetch_add(1, std::memory_order_relaxed);
    m_seenGifCopy = true;
    m_gifCopyCount.fetch_add(1, std::memory_order_relaxed);

    m_ioRegisters[GIF_CHANNEL + 0x30u] = tadr;
    m_ioRegisters[GIF_CHANNEL + 0x40u] = 0u;
    m_ioRegisters[GIF_CHANNEL + 0x50u] = 0u;
    m_ioRegisters[GIF_CHANNEL + 0x00u] = ((chcr & 0x0000FFFFu) | (tag.upper << 16u)) & ~0x100u;
    m_ioRegisters[GIF_CHANNEL + 0x20u] = 0u;

    uint32_t dstat = m_ioRegisters.count(D_STAT) ? m_ioRegisters[D_STAT] : 0u;
    dstat |= (1u << 2u);
    const uint32_t status = dstat & 0x3FFu;
    const uint32_t mask = (dstat >> 16u) & 0x3FFu;
    if ((status & mask) != 0u)
        dstat |= (1u << 31u);
    else
        dstat &= ~(1u << 31u);
    m_ioRegisters[D_STAT] = dstat;
    queueCompletedDmacCause(2u);
    return true;
}

int PS2Memory::pollDmaRegisters()
{
    return 0;
}

uint32_t PS2Memory::readIORegister(uint32_t address)
{
    // VIF1_STAT (0x10003C00) is polled by the sceGsSyncPath stub every frame;
    // transfers complete instantly here, so its value never depends on the
    // worker (verified: unchanged across 42k drained reads). The stub bounds
    // the lag with vu1SyncPathEpoch() instead.
    if (address != 0x10003C00u &&
        ((address >= 0x10003C00u && address < 0x10003E00u) || (address >= 0x10005000u && address < 0x10006000u)))
        { g_ps2xDrainSite = 2615; drainVu1Worker(); g_ps2xDrainSite = 0; } // VIF1 registers / FIFO reflect the worker's progress
    // DMA controller (0x1000E000..) and channel registers (0x10008000..0x1000D000)
    if (g_ps2xVif1GapOpen.load(std::memory_order_relaxed) &&
        ((address >= 0x10008000u && address < 0x1000E000u) || (address >= 0x1000E000u && address < 0x1000E060u)))
    {
        g_ps2xVif1GapDmaReads.fetch_add(1u, std::memory_order_relaxed);
        g_ps2xVif1GapPolled.store(true, std::memory_order_relaxed);
    }
    size_t timerIndex = 0u;
    uint32_t timerOffset = 0u;
    if (decodeEeTimerRegister(address, timerIndex, timerOffset))
    {
        if (s_eeTimerPendingCycles != 0u)
        {
            const uint64_t pending = s_eeTimerPendingCycles;
            s_eeTimerFlushAtCycles = 0u;
            s_eeTimerPendingCycles = 0u;
            (void)advanceEeTimers(pending);
        }
        const EeTimer &timer = m_eeTimers[timerIndex];
        switch (timerOffset)
        {
        case kEeTimerCountOffset:
            return timer.count & 0xFFFFu;
        case kEeTimerModeOffset:
            return timer.mode & (kEeTimerModeConfigMask | kEeTimerModeStatusMask);
        case kEeTimerCompareOffset:
            return timer.compare & 0xFFFFu;
        case kEeTimerHoldOffset:
            return timer.hold & 0xFFFFu;
        default:
            return 0u;
        }
    }

    if (isGsPrivReg(address))
    {
        { g_ps2xDrainSite = 2652; const_cast<PS2Memory *>(this)->drainVu1Worker(); g_ps2xDrainSite = 0; } // may observe VU1/GS side effects
        // NB: unreachable from read8/16/32/64 today, same reasoning as the write
        // path above; kept correct for direct callers.
        const uint32_t off = address & 7u;
        const uint32_t regOff = (address - PS2_GS_PRIV_REG_BASE) & ~0x7u;
        if (regOff == kGsCsrRegOffset)
        {
            return static_cast<uint32_t>((gs_regs.csr.load() >> (off * 8u)) & 0xFFFFFFFFull);
        }
        if (uint64_t *reg = gsRegPtr(gs_regs, address))
        {
            return static_cast<uint32_t>((*reg >> (off * 8u)) & 0xFFFFFFFFull);
        }
        return 0u;
    }

    if (address >= 0x10002000 && address <= 0x10002030)
    {
        uint32_t val = 0;
        switch (address)
        {
        case 0x10002000:
            val = m_ioRegisters[address];
            break;
        case 0x10002010:
            val = m_ioRegisters[address] & ~(1u << 31);
            break;
        case 0x10002020:
        case 0x10002030:
            val = m_ioRegisters[address];
            break;
        default:
            val = 0;
            break;
        }
        return val;
    }
    if (address >= 0x10000000 && address < 0x10010000)
    {
        if (address >= 0x10008000 && address < 0x1000F000)
        {
            if ((address & 0xFF) == 0x00)
            {
                uint32_t channelStatus = m_ioRegisters[address] & ~0x100u;
                m_ioRegisters[address] = channelStatus;
                return channelStatus;
            }
        }

        if (address >= 0x10000200 && address < 0x10000300)
        {
            return 0;
        }

        if (address >= 0x1000F200 && address <= 0x1000F260)
        {
            if (address == 0x1000F230)
            {
                return 0x60000;
            }
            if (address == 0x1000F240)
            {
                return 0xF0000002;
            }
            return 0;
        }
    }

    auto it = m_ioRegisters.find(address);
    if (it != m_ioRegisters.end())
    {
        return it->second;
    }

    return 0;
}

void PS2Memory::registerCodeRegion(uint32_t start, uint32_t end)
{
    if (end <= start)
    {
        std::cerr << "Ignoring invalid code region: start=0x" << std::hex << start
                  << " end=0x" << end << std::dec << std::endl;
        return;
    }

    if ((end - start) > PS2_RAM_SIZE)
    {
        std::cerr << "Ignoring oversized code region: start=0x" << std::hex << start
                  << " end=0x" << end << std::dec << std::endl;
        return;
    }

    for (const auto &existing : m_codeRegions)
    {
        if (existing.start == start && existing.end == end)
        {
            return;
        }
    }

    CodeRegion region;
    region.start = start;
    region.end = end;

    size_t sizeInWords = (end - start + 3u) / 4u;
    region.modified.resize(sizeInWords, false);

    m_codeRegions.push_back(region);
    RUNTIME_LOG("Registered code region: " << std::hex << start << " - " << end << std::dec);
}

bool PS2Memory::isAddressInRegion(uint32_t address, const CodeRegion &region)
{
    return (address >= region.start && address < region.end);
}

bool PS2Memory::isCodeAddress(uint32_t address) const
{
    for (const auto &region : m_codeRegions)
    {
        if (address >= region.start && address < region.end)
        {
            return true;
        }
    }
    return false;
}

void PS2Memory::markModified(uint32_t address, uint32_t size)
{
    if (size == 0)
    {
        return;
    }

    const uint64_t writeEnd = static_cast<uint64_t>(address) + static_cast<uint64_t>(size);
    for (auto &region : m_codeRegions)
    {
        const uint64_t regionStart = region.start;
        const uint64_t regionEnd = region.end;
        if (writeEnd <= regionStart || static_cast<uint64_t>(address) >= regionEnd)
        {
            continue;
        }

        uint32_t overlapStart = static_cast<uint32_t>(std::max<uint64_t>(address, regionStart));
        uint32_t overlapEnd = static_cast<uint32_t>(std::min<uint64_t>(writeEnd, regionEnd));

        for (uint32_t addr = overlapStart; addr < overlapEnd; addr += 4)
        {
            size_t bitIndex = (addr - region.start) / 4;
            if (bitIndex < region.modified.size())
            {
                region.modified[bitIndex] = true;
                RUNTIME_LOG("Marked code at " << std::hex << addr << std::dec << " as modified");
            }
        }
    }
}

bool PS2Memory::isCodeModified(uint32_t address, uint32_t size)
{
    if (size == 0)
    {
        return false;
    }

    const uint64_t writeEnd = static_cast<uint64_t>(address) + static_cast<uint64_t>(size);
    for (const auto &region : m_codeRegions)
    {
        const uint64_t regionStart = region.start;
        const uint64_t regionEnd = region.end;
        if (writeEnd <= regionStart || static_cast<uint64_t>(address) >= regionEnd)
        {
            continue;
        }

        uint32_t overlapStart = static_cast<uint32_t>(std::max<uint64_t>(address, regionStart));
        uint32_t overlapEnd = static_cast<uint32_t>(std::min<uint64_t>(writeEnd, regionEnd));

        for (uint32_t addr = overlapStart; addr < overlapEnd; addr += 4)
        {
            size_t bitIndex = (addr - region.start) / 4;
            if (bitIndex < region.modified.size() && region.modified[bitIndex])
            {
                return true; // Found modified code
            }
        }
    }

    return false; // No modifications found
}

void PS2Memory::clearModifiedFlag(uint32_t address, uint32_t size)
{
    if (size == 0)
    {
        return;
    }

    const uint64_t writeEnd = static_cast<uint64_t>(address) + static_cast<uint64_t>(size);
    for (auto &region : m_codeRegions)
    {
        const uint64_t regionStart = region.start;
        const uint64_t regionEnd = region.end;
        if (writeEnd <= regionStart || static_cast<uint64_t>(address) >= regionEnd)
        {
            continue;
        }

        uint32_t overlapStart = static_cast<uint32_t>(std::max<uint64_t>(address, regionStart));
        uint32_t overlapEnd = static_cast<uint32_t>(std::min<uint64_t>(writeEnd, regionEnd));

        for (uint32_t addr = overlapStart; addr < overlapEnd; addr += 4)
        {
            size_t bitIndex = (addr - region.start) / 4;
            if (bitIndex < region.modified.size())
            {
                region.modified[bitIndex] = false;
            }
        }
    }
}
