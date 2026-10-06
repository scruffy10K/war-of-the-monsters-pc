#include "../motion_provenance.inc"
#define WOTM_PACKED_DECODE_ONLY
#include "gs_packed_world.inc"
#undef WOTM_PACKED_DECODE_ONLY
#if defined(_WIN32)
#include <immintrin.h>
#endif
#include "runtime/gs/gs_gl_backend.h"
#include "runtime/gs/gs_frontend.h"
#include "runtime/ps2_perf_clock.h"
#include "runtime/gs/gs_cpu_backend.h"
#if !defined(PLATFORM_VITA)
#include "raylib.h"
#endif
#include "ps2_log.h"
#include "runtime/ps2_memory.h"
#include "runtime/gs/ps2_gif_arbiter.h"
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <fstream>
#include <filesystem>

// Header-free bridge: a homogeneous stream of already assembled triangles.
void ps2xGlSubmitGeometry(GSGlBackend &backend, const GSPrimitiveBatch &prototype,
                         const GSVertex *vertices, size_t count);
void ps2xGlSubmitNearCullReference(GSGlBackend &backend, const GSPrimitiveBatch &batch);
void ps2xGlSubmitIndexedGeometry(GSGlBackend &backend, const GSPrimitiveBatch &prototype,
                               const GSVertex *vertices, size_t uniqueCount,
                               const uint16_t *indices, size_t count);

void ps2xGlSubmitPackedWorldGeometry(GSGlBackend&, const GSPrimitiveBatch&,
    const uint8_t*, size_t, const uint16_t*, size_t, uint16_t, uint16_t);

namespace
{
    // Launch-only settings: no local-static initialization guards per primitive.
    const bool s_bigPrimCensus = []
    {
        const char *value = std::getenv("PS2X_BIGPRIM_CENSUS");
        return value != nullptr && value[0] == '1';
    }();
    const bool s_sliverCull = []
    {
        const char *value = std::getenv("PS2X_SLIVER_CULL");
        return value != nullptr && value[0] == '1';
    }();
    const bool s_nearCull = []
    {
        const char *value = std::getenv("PS2X_NEAR_CULL");
        return value == nullptr || value[0] != '0';
    }();
    const bool s_nearCullReference = std::getenv("PS2X_NEAR_CULL_REFERENCE") != nullptr;
    const bool s_sliverCullReference = std::getenv("PS2X_SLIVER_CULL_REFERENCE") != nullptr;
    const bool captureStretch=std::getenv("PS2X_STRETCH_CHECKPOINT")!=nullptr;
    const uint64_t traceFrame=[] {const char *p=std::getenv("PS2X_GEOMETRY_TRACE_FRAME");return p?std::strtoull(p,nullptr,10):~0ull;}();
    const uint64_t traceEnd=[] {const char *p=std::getenv("PS2X_GEOMETRY_TRACE_END");return p?std::strtoull(p,nullptr,10):traceFrame;}();
    const float traceSpan=[] {const char *p=std::getenv("PS2X_GEOMETRY_TRACE_MIN_SPAN");return p?std::strtof(p,nullptr):0.0f;}();
    const bool s_bigTri = []
    {
        const char *value = std::getenv("PS2X_BIGTRI");
        return value != nullptr && value[0] == '1';
    }();
    const uint64_t s_geometryStartFrame = [] {
        const char *v = std::getenv("PS2X_GEOMETRY_START_FRAME");
        return v ? std::strtoull(v, nullptr, 10) : 0ull;
    }();

    struct NativeGeometryBatch;
    thread_local NativeGeometryBatch *s_nativeGeometry = nullptr;
    thread_local std::vector<GSVertex> s_nativeGeometryVertices;
    const bool s_nativeGeometryEnabled = [] {
        const char *p = std::getenv("PS2X_NATIVE_GEOMETRY"); return !p || p[0] != '0';
    }();
    const bool s_nativeGeometryVerify = std::getenv("PS2X_NATIVE_GEOMETRY_VERIFY") != nullptr;
    const bool s_nativeIndexedEnabled=[] {
        const char *p=std::getenv("PS2X_NATIVE_INDEXED");return !p || p[0]!='0';
    }();
    const bool s_nativeIndexedVerify=std::getenv("PS2X_NATIVE_INDEXED_VERIFY")!=nullptr;
    // Whole reset dense STQ strips only; switch 0 retains the decoded route.
    const bool s_packedWorld=[] {
        const char *p=std::getenv("PS2X_GS_PACKED_WORLD");return !p || p[0]!='0';
    }();

    // Only vertex-only PACKED tags qualify. In particular A+D, PRIM and texture
    // writes must never be deferred past the geometry that precedes them.
    bool homogeneousGeometryTag(const uint8_t *regs, uint32_t count)
    {
        bool vertex = false;
        for (uint32_t i = 0; i < count; ++i) {
            switch (regs[i]) {
            case 4: case 5: case 12: case 13: vertex = true; break;
            case 1: case 2: case 3: case 10: case 15: break;
            default: return false;
            }
        }
        return vertex;
    }

    struct NativeGeometryBatch
    {
        GSGlBackend *backend;
        const GS *owner;
        std::vector<GSVertex> &vertices;
        GSPrimitiveBatch prototype{};
        bool valid = false;
        NativeGeometryBatch(GSGlBackend *b, const GS *g)
            : backend(b), owner(g), vertices(s_nativeGeometryVertices)
        {
            if (backend) {
                s_nativeGeometryVertices.clear();
                s_nativeGeometry = this;
            }
        }
        ~NativeGeometryBatch() { if (backend) s_nativeGeometry = nullptr; }
        void flush()
        {
            if (s_nativeGeometryVertices.empty()) return;
            ps2xGlSubmitGeometry(*backend, prototype, s_nativeGeometryVertices.data(),
                                s_nativeGeometryVertices.size());
            noteSubmission(s_nativeGeometryVertices.size());
            s_nativeGeometryVertices.clear();
        }
        void flushIndexed(const GSVertex *input, size_t uniqueCount,
                          const uint16_t *indices, size_t count)
        {
            if (!count) return;
            ps2xGlSubmitIndexedGeometry(*backend, prototype, input, uniqueCount, indices, count);
            noteSubmission(count);
        }
        void noteSubmission(size_t count)
        {
            static thread_local uint64_t streams = 0, triangles = 0;
            ++streams;
            const uint64_t before = triangles;
            triangles += count / 3;
            if (s_nativeGeometryVerify && before / 1000000 != triangles / 1000000)
                std::fprintf(stderr, "[gs:native-geometry] streams=%llu triangles=%llu state-mismatches=0\n",
                    (unsigned long long)streams, (unsigned long long)triangles);
        }
    };
}

namespace {
    struct NativeVuMeshMessage {
        GS *gs; uint64_t tag; const GSVertex *vertices; const uint8_t *adc;
        uint32_t count; const uint8_t *raw; uint32_t bytes;
    };
    thread_local const NativeVuMeshMessage *s_vuMeshMessage = nullptr;
    thread_local const GSVertex *s_vuExpectedVertex = nullptr;
    thread_local bool s_vuExpectedDrawing = false;
    constexpr uint64_t kNativeVuToken = 0x4853454d55563250ull;
}

void ps2xGsSubmitVuMesh(GS &gs, PS2Memory *memory, uint64_t tag, const GSVertex *vertices,
                       const uint8_t *adc, uint32_t count, const uint8_t *raw, uint32_t bytes)
{
    const NativeVuMeshMessage message{&gs, tag, vertices, adc, count, raw, bytes};
    const uint64_t token[2] = {kNativeVuToken, reinterpret_cast<uintptr_t>(&message)};
    struct Scope {
        const NativeVuMeshMessage *previous = s_vuMeshMessage;
        ~Scope() { s_vuMeshMessage = previous; }
    } scope;
    s_vuMeshMessage = &message;
    // A small synchronous token retains PATH1 arbitration with queued PATH2/3
    // traffic. Vertex data never gets copied through the GIF packet queue.
    if (memory) memory->submitGifPacket(GifPathId::Path1, reinterpret_cast<const uint8_t *>(token), sizeof(token));
    else gs.processGIFPacket(reinterpret_cast<const uint8_t *>(token), sizeof(token));
}

extern std::atomic<uint64_t> g_ps2xWotmCompletedFrames;
void ps2xSaveStretchCheckpoint();

namespace
{
    static constexpr uint32_t kHostFrameWidth = 640u;

    GSPrimReg decodePrimRegister(uint64_t value)
    {
        GSPrimReg prim{};
        prim.type = static_cast<GSPrimType>(value & 0x7u);
        prim.iip = ((value >> 3) & 1u) != 0u;
        prim.tme = ((value >> 4) & 1u) != 0u;
        prim.fge = ((value >> 5) & 1u) != 0u;
        prim.abe = ((value >> 6) & 1u) != 0u;
        prim.aa1 = ((value >> 7) & 1u) != 0u;
        prim.fst = ((value >> 8) & 1u) != 0u;
        prim.ctxt = ((value >> 9) & 1u) != 0u;
        prim.fix = ((value >> 10) & 1u) != 0u;
        return prim;
    }

    static inline uint64_t loadLE64(const uint8_t *p)
    {
        uint64_t v;
        std::memcpy(&v, p, 8);
        return v;
    }

    struct PackedGifPacketTag
    {
        uint64_t lo = 0u;
        uint64_t hi = 0u;
        uint32_t payloadOffset = 0u;
        uint32_t nloop = 0u;
        uint32_t nreg = 0u;
        uint8_t regs[16]{};
    };

    template <typename Visitor>
    bool visitPackedGifPacket(const uint8_t *data, uint32_t sizeBytes, Visitor &&visitor)
    {
        uint32_t offset = 0u;
        while (offset + 16u <= sizeBytes)
        {
            PackedGifPacketTag tag{};
            tag.lo = loadLE64(data + offset);
            tag.hi = loadLE64(data + offset + 8u);

            const uint8_t flg = static_cast<uint8_t>((tag.lo >> 58u) & 0x3u);
            if (flg != GIF_FMT_PACKED)
                return false;

            tag.nloop = static_cast<uint32_t>(tag.lo & 0x7FFFu);
            tag.nreg = static_cast<uint32_t>((tag.lo >> 60u) & 0xFu);
            if (tag.nreg == 0u)
                tag.nreg = 16u;

            const uint64_t payloadBytes64 =
                static_cast<uint64_t>(tag.nloop) * static_cast<uint64_t>(tag.nreg) * 16ull;
            if (payloadBytes64 > 0xFFFFFFFFull)
                return false;

            offset += 16u;
            const uint32_t payloadBytes = static_cast<uint32_t>(payloadBytes64);
            if (payloadBytes > sizeBytes - offset)
                return false;

            tag.payloadOffset = offset;
            for (uint32_t i = 0u; i < tag.nreg; ++i)
                tag.regs[i] = static_cast<uint8_t>((tag.hi >> (i * 4u)) & 0xFu);

            if (!visitor(tag))
                return false;

            offset += payloadBytes;
        }

        return offset == sizeBytes;
    }

    bool validatePackedGifPacket(const uint8_t *data, uint32_t sizeBytes)
    {
        return visitPackedGifPacket(data, sizeBytes, [](const PackedGifPacketTag &)
                                    { return true; });
    }

    std::atomic<uint32_t> s_debugGifPacketCount{0};

    // Bytes of GIF PATH3 IMAGE-mode data still owed to the GS from an IMAGE-mode
    // GIFtag whose NLOOP payload was split across multiple DMA packets. The GIF
    // FIFO consumes exactly NLOOP qwords regardless of DMA packet boundaries, so
    // GS::processGIFPacket must carry this across calls or the trailing pixels of
    // every large texture / video-frame upload get mis-parsed as GIFtags.
    // Guarded by GS::m_stateMutex (held by every reader/writer below).
    uint32_t s_pendingImageBytes = 0u;
    std::atomic<uint32_t> s_debugGsRegisterCount{0};
    std::atomic<uint32_t> s_debugGsPackedVertexCount{0};
    std::atomic<uint32_t> s_debugGsVertexKickCount{0};
    std::atomic<uint32_t> s_debugCopyRegCount{0};
    std::atomic<uint32_t> s_debugTexaWriteCount{0};
    std::atomic<uint32_t> s_debugCvFontUploadCount{0};
    std::atomic<uint32_t> s_debugLocalCopyCount{0};
}

// One-shot full-frame GS stream capture. When > 0, every GS register write,
// vertex kick and rasterized primitive is logged verbatim (bypassing the normal
// per-tag debug caps) and the budget is decremented; it self-disarms at 0.
// Armed from the [run:tick] shell-state probe in ps2_runtime.cpp once WotM
// reaches screenMain (currScreen==1, betweenScreens==0).
std::atomic<int64_t> g_gsFrameDump{0};
// GS FINISH register writes, and game-thread time spent in their raster Sync
// (see GS_REG_FINISH in writeRegisterUnlocked); printed by [perf].
std::atomic<uint64_t> g_ps2xFinishCount{0u};
// Defined in ps2_vu1_core.cpp: the VU1 XGKICK packet being submitted right now.
struct Ps2xVu1KickContext
{
    const uint8_t *packet;
    uint32_t bytes, startPc, pc, src, top, itop;
    uint64_t serial;
};
extern Ps2xVu1KickContext g_ps2xVu1Kick;
void ps2xVu1FlightDump(const char *reason); // ps2_vu1_core.cpp (PS2X_VU1_FLIGHT=1)
std::atomic<uint64_t> g_ps2xFinishWaitNs{0u};
// [perf] gif breakdown (game thread): all time inside GS::processGIFPacket, the
// part spent waiting to acquire m_stateMutex, and the per-primitive
// buildDrawBatch + backend Submit (incl. raster queue backpressure).
std::atomic<uint64_t> g_ps2xGifNs{0u};
std::atomic<uint64_t> g_ps2xGifLockNs{0u};
std::atomic<uint64_t> g_ps2xSubmitNs{0u};


// PS2X_GS_GPU=1 selects the OpenGL backend (gs_gl_backend.cpp); it falls back
// to the CPU rasterizer by itself when no OpenGL 4.3 context is available.
static std::unique_ptr<GSRasterBackend> ps2xMakeGsBackend()
{
    static const bool useGpu = [] { const char *p = std::getenv("PS2X_GS_GPU"); return p && p[0] == '1'; }();
    if (useGpu)
        return std::make_unique<GSGlBackend>();
    return std::make_unique<GSCpuBackend>();
}

GS::GS()
    : m_backend(ps2xMakeGsBackend())
{
    reset();
}

void GS::init(uint8_t *vram, uint32_t vramSize, GSRegisters *privRegs)
{
    m_localMemoryStorage = vram;
    m_localMemorySize = vramSize;
    m_privRegs = privRegs;
    if (!m_backend)
        m_backend = ps2xMakeGsBackend();
    m_backend->Initialize(vram, vramSize);
    reset();
}

void GS::reset()
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    s_pendingImageBytes = 0u;
    std::memset(m_ctx, 0, sizeof(m_ctx));
    m_prim = {};
    m_primRegister = {};
    m_prmodeRegister = {};
    m_curR = 0x80;
    m_curG = 0x80;
    m_curB = 0x80;
    m_curA = 0x80;
    m_curQ = 1.0f;
    m_curS = 0.0f;
    m_curT = 0.0f;
    m_curU = 0;
    m_curV = 0;
    m_curFog = 0;
    m_fogR = 0;
    m_fogG = 0;
    m_fogB = 0;
    m_prmodecont = true;
    m_pabe = false;
    m_scanmsk = 0u;
    m_dimx = 0u;
    m_dthe = 0u;
    m_colclamp = 0u;
    m_texa = {0u, false, 0u};
    m_texclut = {0u, 0u, 0u};
    m_bitbltbuf = {};
    m_trxpos = {};
    m_trxreg = {};
    m_trxdir = 3;
    m_vtxCount = 0;
    m_vtxIndex = 0;
    m_preferredDisplaySourceFrame = {};
    m_preferredDisplayDestFbp = 0;
    m_hasPreferredDisplaySource = false;
    if (m_backend)
    {
        m_backend->Flush();
        m_backend->Sync(GSSyncReason::Reset);
        m_backend->Reset();
    }
    {
        std::lock_guard<std::mutex> presentationLock(m_presentationMutex);
        m_hostPresentationFrame.clear();
        m_hostPresentationWidth = 0u;
        m_hostPresentationHeight = 0u;
        m_hostPresentationDisplayFbp = 0u;
        m_hostPresentationSourceFbp = 0u;
        m_hostPresentationUsedPreferred = false;
        m_hasHostPresentationFrame = false;
    }

    m_debugHistoryWrite = 0;
    m_debugHistoryCount = 0;
    m_debugNextSeq = 1;
    m_debugFrameIndex = 0;
    m_debugLastVsyncTick = UINT64_MAX;

    for (int i = 0; i < 2; ++i)
    {
        m_ctx[i].frame.fbw = 10;
        m_ctx[i].scissor = {0, 639, 0, 447};
        m_ctx[i].xyoffset = {0, 0};
    }
}

GSContext &GS::activeContext()
{
    return m_ctx[m_prim.ctxt ? 1 : 0];
}

void GS::snapshotVRAM()
{
    // Presentation/debug snapshots run outside m_stateMutex so the EE can keep
    // feeding the GS while a backend performs host-side conversion. Keep the
    // selected backend alive and unswappable for the duration of the call.
    std::lock_guard<std::mutex> backendLock(m_backendLifetimeMutex);
    if (!m_backend)
        return;
    std::vector<uint8_t> snapshot;
    m_backend->Sync(GSSyncReason::DebugReadback);
    m_backend->SnapshotVram(snapshot);
    std::lock_guard<std::mutex> lock(m_snapshotMutex);
    m_displaySnapshot.swap(snapshot);
}

const uint8_t *GS::lockDisplaySnapshot(uint32_t &outSize)
{
    m_snapshotMutex.lock();
    if (m_displaySnapshot.empty())
    {
        outSize = 0;
        return nullptr;
    }

    outSize = static_cast<uint32_t>(m_displaySnapshot.size());
    return m_displaySnapshot.data();
}

GSDebugSnapshot GS::getDebugSnapshot() const
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);

    GSDebugSnapshot snapshot{};
    snapshot.ctx[0] = m_ctx[0];
    snapshot.ctx[1] = m_ctx[1];
    snapshot.prim = m_prim;
    snapshot.texa = m_texa;
    snapshot.texclut = m_texclut;
    snapshot.scanmsk = m_scanmsk;
    snapshot.dimx = m_dimx;
    snapshot.dthe = m_dthe;
    snapshot.colclamp = m_colclamp;
    snapshot.bitbltbuf = m_bitbltbuf;
    snapshot.trxpos = m_trxpos;
    snapshot.trxreg = m_trxreg;
    const GSTransferSnapshot transfer = m_backend ? m_backend->GetTransferSnapshot() : GSTransferSnapshot{};
    snapshot.trxdir = transfer.direction;
    snapshot.transferX = transfer.x;
    snapshot.transferY = transfer.y;
    snapshot.transferTotalPixels = transfer.totalPixels;
    snapshot.transferCopiedPixels = transfer.copiedPixels;
    snapshot.lastDisplayBaseBytes = m_lastDisplayBaseBytes;
    snapshot.preferredDisplaySourceFrame = m_preferredDisplaySourceFrame;
    snapshot.preferredDisplayDestFbp = m_preferredDisplayDestFbp;
    snapshot.hasPreferredDisplaySource = m_hasPreferredDisplaySource;
    {
        std::lock_guard<std::mutex> presentationLock(m_presentationMutex);
        snapshot.hostPresentationWidth = m_hostPresentationWidth;
        snapshot.hostPresentationHeight = m_hostPresentationHeight;
        snapshot.hostPresentationDisplayFbp = m_hostPresentationDisplayFbp;
        snapshot.hostPresentationSourceFbp = m_hostPresentationSourceFbp;
        snapshot.hostPresentationUsedPreferred = m_hostPresentationUsedPreferred;
        snapshot.hasHostPresentationFrame = m_hasHostPresentationFrame;
    }
    snapshot.localToHostPendingBytes = transfer.localToHostPendingBytes;
    return snapshot;
}

std::vector<GSDebugHistoryEntry> GS::getDebugHistory() const
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);

    std::vector<GSDebugHistoryEntry> out;
    out.reserve(m_debugHistoryCount);
    const size_t first = (m_debugHistoryWrite + kDebugHistoryCapacity - m_debugHistoryCount) % kDebugHistoryCapacity;
    for (size_t i = 0; i < m_debugHistoryCount; ++i)
    {
        out.push_back(m_debugHistory[(first + i) % kDebugHistoryCapacity]);
    }
    return out;
}

void GS::clearDebugHistory()
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    m_debugHistoryWrite = 0;
    m_debugHistoryCount = 0;
    m_debugNextSeq = 1;
    m_debugFrameIndex = 0;
    m_debugLastVsyncTick = UINT64_MAX;
}

bool GS::isDebugHistoryPaused() const
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    return m_debugHistoryPaused;
}

void GS::setDebugHistoryPaused(bool paused)
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    m_debugHistoryPaused = paused;
}

GSDebugHistoryEntry GS::makeDebugEventUnlocked(GSDebugEventKind kind) const
{
    GSDebugHistoryEntry entry{};
    entry.kind = kind;
    entry.prim = m_prim;
    const uint32_t ci = m_prim.ctxt ? 1u : 0u;
    entry.frame = m_ctx[ci].frame;
    entry.zbuf = m_ctx[ci].zbuf;
    entry.tex0 = m_ctx[ci].tex0;
    entry.scissor = m_ctx[ci].scissor;
    entry.test = m_ctx[ci].test;
    entry.alpha = m_ctx[ci].alpha;
    entry.bitbltbuf = m_bitbltbuf;
    entry.trxpos = m_trxpos;
    entry.trxreg = m_trxreg;
    const GSTransferSnapshot transfer = m_backend ? m_backend->GetTransferSnapshot() : GSTransferSnapshot{};
    entry.trxdir = transfer.direction;
    entry.transferPixels = transfer.totalPixels;
    return entry;
}

void GS::recordDebugEventUnlocked(GSDebugHistoryEntry entry)
{
    if (m_debugHistoryPaused)
    {
        return;
    }

    const uint64_t tick = m_privRegs ? m_privRegs->vsyncTick.load(std::memory_order_acquire) : 0u;
    if (m_debugLastVsyncTick == UINT64_MAX)
    {
        m_debugLastVsyncTick = tick;
    }
    else if (tick != m_debugLastVsyncTick)
    {
        ++m_debugFrameIndex;
        m_debugLastVsyncTick = tick;
    }

    entry.seq = m_debugNextSeq++;
    entry.vsyncTick = tick;
    entry.frameIndex = m_debugFrameIndex;

    m_debugHistory[m_debugHistoryWrite] = entry;
    m_debugHistoryWrite = (m_debugHistoryWrite + 1u) % kDebugHistoryCapacity;
    if (m_debugHistoryCount < kDebugHistoryCapacity)
    {
        ++m_debugHistoryCount;
    }
}

void GS::recordGifTagDebugEventUnlocked(uint32_t sizeBytes, uint32_t nloop, uint8_t flg, uint32_t nreg)
{
    if (m_debugHistoryPaused)
    {
        return;
    }

    GSDebugHistoryEntry entry = makeDebugEventUnlocked(GSDebugEventKind::GifTag);
    entry.gifSizeBytes = sizeBytes;
    entry.gifNloop = nloop;
    entry.gifFlg = flg;
    entry.gifNreg = static_cast<uint8_t>(std::min<uint32_t>(nreg, 16u));
    recordDebugEventUnlocked(entry);
}

void GS::recordRegisterDebugEventUnlocked(uint8_t regAddr, uint64_t value)
{
    if (m_debugHistoryPaused)
    {
        return;
    }

    switch (regAddr)
    {
    case GS_REG_PRIM:
    case GS_REG_TEX0_1:
    case GS_REG_TEX0_2:
    case GS_REG_TEX2_1:
    case GS_REG_TEX2_2:
    case GS_REG_TEXA:
    case GS_REG_TEXCLUT:
    case GS_REG_FRAME_1:
    case GS_REG_FRAME_2:
    case GS_REG_ZBUF_1:
    case GS_REG_ZBUF_2:
    case GS_REG_ALPHA_1:
    case GS_REG_ALPHA_2:
    case GS_REG_TEST_1:
    case GS_REG_TEST_2:
    case GS_REG_SCISSOR_1:
    case GS_REG_SCISSOR_2:
    case GS_REG_XYOFFSET_1:
    case GS_REG_XYOFFSET_2:
    case GS_REG_BITBLTBUF:
    case GS_REG_TRXPOS:
    case GS_REG_TRXREG:
    case GS_REG_TRXDIR:
        break;
    default:
        return;
    }

    GSDebugHistoryEntry entry = makeDebugEventUnlocked(GSDebugEventKind::Register);
    entry.reg = regAddr;
    entry.regValue = value;
    recordDebugEventUnlocked(entry);
}

void GS::recordDrawDebugEventUnlocked(int vertexCount)
{
    if (m_debugHistoryPaused)
    {
        return;
    }

    if (vertexCount <= 0)
    {
        return;
    }

    GSDebugHistoryEntry entry = makeDebugEventUnlocked(GSDebugEventKind::Draw);
    entry.vertexCount = static_cast<uint32_t>(vertexCount);

    const int count = std::min(vertexCount, kMaxVerts);
    entry.xMin = entry.xMax = m_vtxQueue[0].x;
    entry.yMin = entry.yMax = m_vtxQueue[0].y;
    entry.zMin = entry.zMax = m_vtxQueue[0].z;
    entry.aMin = entry.aMax = m_vtxQueue[0].a;

    for (int i = 1; i < count; ++i)
    {
        const GSVertex &v = m_vtxQueue[i];
        entry.xMin = std::min(entry.xMin, v.x);
        entry.xMax = std::max(entry.xMax, v.x);
        entry.yMin = std::min(entry.yMin, v.y);
        entry.yMax = std::max(entry.yMax, v.y);
        entry.zMin = std::min(entry.zMin, v.z);
        entry.zMax = std::max(entry.zMax, v.z);
        entry.aMin = std::min(entry.aMin, v.a);
        entry.aMax = std::max(entry.aMax, v.a);
    }

    recordDebugEventUnlocked(entry);
}

void GS::recordTransferDebugEventUnlocked()
{
    if (m_debugHistoryPaused)
    {
        return;
    }

    GSDebugHistoryEntry entry = makeDebugEventUnlocked(GSDebugEventKind::Transfer);
    entry.transferPixels = m_backend ? m_backend->GetTransferSnapshot().totalPixels : 0u;
    recordDebugEventUnlocked(entry);
}

void GS::recordPresentDebugEventUnlocked(uint32_t displayFbp, uint32_t sourceFbp, uint32_t width, uint32_t height, bool usedPreferred)
{
    if (m_debugHistoryPaused)
    {
        return;
    }

    GSDebugHistoryEntry entry = makeDebugEventUnlocked(GSDebugEventKind::Present);
    entry.displayFbp = displayFbp;
    entry.sourceFbp = sourceFbp;
    entry.width = width;
    entry.height = height;
    entry.usedPreferred = usedPreferred;
    recordDebugEventUnlocked(entry);
}

bool GS::getPreferredDisplaySource(GSFrameReg &outSource, uint32_t &outDestFbp) const
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    if (!m_hasPreferredDisplaySource)
    {
        outSource = {};
        outDestFbp = 0u;
        return false;
    }

    outSource = m_preferredDisplaySourceFrame;
    outDestFbp = m_preferredDisplayDestFbp;
    return true;
}

void GS::unlockDisplaySnapshot()
{
    m_snapshotMutex.unlock();
}

uint32_t GS::getLastDisplayBaseBytes() const
{
    return m_lastDisplayBaseBytes;
}

void GS::refreshDisplaySnapshot()
{
    snapshotVRAM();
}

GSPresentationRequest GS::buildPresentationRequestUnlocked() const
{
    GSPresentationRequest request{};
    if (!m_privRegs)
        return request;
    request.pmode = m_privRegs->pmode;
    request.smode2 = m_privRegs->smode2;
    request.dispfb1 = m_privRegs->dispfb1;
    request.display1 = m_privRegs->display1;
    request.dispfb2 = m_privRegs->dispfb2;
    request.display2 = m_privRegs->display2;
    request.bgcolor = m_privRegs->bgcolor;
    request.vsyncTick = m_privRegs->vsyncTick.load(std::memory_order_acquire);
    request.contextFrames[0] = m_ctx[0].frame;
    request.contextFrames[1] = m_ctx[1].frame;
    request.preferredSource = m_preferredDisplaySourceFrame;
    request.preferredDestFbp = m_preferredDisplayDestFbp;
    request.hasPreferredSource = m_hasPreferredDisplaySource;
    return request;
}

struct Ps2xPerf { std::atomic<uint64_t> rasterNs, presentNs, vu1Ns, vif1Ns, prims; };
extern Ps2xPerf g_ps2xPerf;

namespace
{
    // Pages recently used as a Z buffer by a depth-writing draw. The presenter
    // refuses to show them as a picture (see latchHostPresentationFrame). A
    // transition can re-point ZBUF elsewhere a frame before the display is
    // pointed at the old depth page, so "current ZBUF" alone missed some of
    // WotM's white flashes (tick 1047/1049 in flash2). Guarded by m_stateMutex.
    constexpr uint32_t kRecentDepthPages = 4u;
    uint32_t g_recentDepthPages[kRecentDepthPages] = {~0u, ~0u, ~0u, ~0u};
    uint32_t g_recentDepthPagesNext = 0u;

    void noteDepthPage(uint32_t zbp)
    {
        for (uint32_t page : g_recentDepthPages)
            if (page == zbp)
                return;
        g_recentDepthPages[g_recentDepthPagesNext % kRecentDepthPages] = zbp;
        ++g_recentDepthPagesNext;
    }

    // [gs:flash] frame dumps (PS2X_FLASH_DUMP_DIR): when the tracer flags a
    // WHITE frame or a GEOMETRY DROP, the presented frame and the one before it
    // are written as BMPs so "buildings vanished" can be told apart from "camera
    // cut to a close-up". Capped; presenter thread only.
    bool g_flashDumpPending = false;
    char g_flashDumpReason[32] = {0};
    uint32_t g_flashDumpsWritten = 0u;

    void writeBmp(const std::string &path, const uint8_t *pixels, uint32_t width, uint32_t height, size_t bpp)
    {
        if (!pixels || width == 0u || height == 0u || bpp < 3u)
            return;
        std::FILE *f = std::fopen(path.c_str(), "wb");
        if (!f)
            return;
        const uint32_t rowBytes = (width * 3u + 3u) & ~3u;
        const uint32_t dataBytes = rowBytes * height;
        uint8_t header[54] = {'B', 'M'};
        auto put32 = [&](int off, uint32_t v)
        {
            header[off] = static_cast<uint8_t>(v);
            header[off + 1] = static_cast<uint8_t>(v >> 8);
            header[off + 2] = static_cast<uint8_t>(v >> 16);
            header[off + 3] = static_cast<uint8_t>(v >> 24);
        };
        put32(2, 54u + dataBytes);
        put32(10, 54u);
        put32(14, 40u);
        put32(18, width);
        put32(22, height);
        header[26] = 1;
        header[28] = 24;
        put32(34, dataBytes);
        std::fwrite(header, 1, sizeof(header), f);
        std::vector<uint8_t> row(rowBytes, 0u);
        for (uint32_t y = 0; y < height; ++y)
        {
            const uint8_t *src = pixels + static_cast<size_t>(height - 1u - y) * width * bpp;
            for (uint32_t x = 0; x < width; ++x)
            {
                row[x * 3u + 0u] = src[x * bpp + 2u];
                row[x * 3u + 1u] = src[x * bpp + 1u];
                row[x * 3u + 2u] = src[x * bpp + 0u];
            }
            std::fwrite(row.data(), 1, rowBytes, f);
        }
        std::fclose(f);
    }

    // Primitives submitted since the last presented frame (m_stateMutex).
    uint64_t g_primsSinceLastPresent = 0u;

    bool isRecentDepthPage(uint32_t page)
    {
        for (uint32_t recorded : g_recentDepthPages)
            if (recorded == page)
                return true;
        return false;
    }
}

// PS2X_GS_SHOT=<file.png>[,<frame>]: write one presented frame to a PNG, so a
// GPU frame can be compared with the CPU rasterizer's.
// Writes any RGBA8 buffer to a PNG (texture dumps, debug captures).
// Microsoft's DirectXTK controller sheet, kept unmodified in runtime assets.
// Assemble the retail face-button layout: B/X above Y/A. Retail's mesh UVs
// invert each button vertically, so this atlas follows the original convention.
bool ps2xLoadXboxButtonAtlas(std::vector<uint8_t> &rgba)
{
#if !defined(PLATFORM_VITA)
    Image sheet = LoadImage("ps2xRuntime/assets/controller/xboxControllerSpriteFont.png");
    if (!sheet.data || sheet.width != 1729 || sheet.height != 188) {
        if (sheet.data) UnloadImage(sheet);
        std::fprintf(stderr, "[pad:glyphs] Xbox artwork unavailable; retaining retail buttons\n");
        return false;
    }
    constexpr int starts[4] = {1038, 798, 958, 878}; // Circle, Square, Triangle, Cross
    constexpr int cell = 128, width = cell * 2;
    rgba.assign(width * width * 4, 0);
    for (int button = 0; button < 4; ++button) {
        Image glyph = ImageFromImage(sheet, Rectangle{static_cast<float>(starts[button]), 54, 78, 80});
        ImageFormat(&glyph, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
        ImageResize(&glyph, cell, cell);
        const auto *src = static_cast<const uint8_t *>(glyph.data);
        for (int y = 0; y < cell; ++y) for (int x = 0; x < cell; ++x) {
            const auto *pixel = src + ((cell - 1 - y) * cell + x) * 4;
            auto *dst = rgba.data() + (((button / 2) * cell + y) * width + (button % 2) * cell + x) * 4;
            std::memcpy(dst, pixel, 3);
            dst[3] = static_cast<uint8_t>((static_cast<unsigned>(pixel[3]) * 128u + 127u) / 255u);
        }
        UnloadImage(glyph);
    }
    UnloadImage(sheet);
    return true;
#else
    return false;
#endif
}

void ps2xGsSaveImage(const char *path, const uint8_t *rgba, uint32_t width, uint32_t height)
{
    if (!path || !rgba || width == 0u || height == 0u)
        return;
    Image image{};
    image.data = const_cast<uint8_t *>(rgba);
    image.width = static_cast<int>(width);
    image.height = static_cast<int>(height);
    image.mipmaps = 1;
    image.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
    ExportImage(image, path);
}

void ps2xGsSaveFrame(const uint8_t *rgba, uint32_t width, uint32_t height, uint64_t frameIndex)
{
    static const std::string spec = [] { const char *p = std::getenv("PS2X_GS_SHOT"); return std::string(p ? p : ""); }();
    if (spec.empty() || !rgba || width == 0u || height == 0u)
        return;
    const size_t comma = spec.find(',');
    const uint64_t wanted = comma == std::string::npos ? 600ull : std::strtoull(spec.c_str() + comma + 1, nullptr, 10);
    // PS2X_GS_SHOT=<file>,<frame>[,<count>]: a run of consecutive presented
    // frames, so flicker can be measured instead of argued about.
    const size_t comma2 = comma == std::string::npos ? std::string::npos : spec.find(',', comma + 1);
    const uint64_t count =
        comma2 == std::string::npos ? 1ull : std::max<uint64_t>(std::strtoull(spec.c_str() + comma2 + 1, nullptr, 10), 1ull);
    if (frameIndex < wanted || frameIndex >= wanted + count)
        return;
    std::string path = spec.substr(0, comma);
    if (count > 1ull)
    {
        char suffix[32];
        std::snprintf(suffix, sizeof(suffix), ".%03llu.png", (unsigned long long)(frameIndex - wanted));
        path += suffix;
    }
    Image image{};
    image.data = const_cast<uint8_t *>(rgba);
    image.width = static_cast<int>(width);
    image.height = static_cast<int>(height);
    image.mipmaps = 1;
    image.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
    ExportImage(image, path.c_str());
    std::fprintf(stderr, "[gs:shot] wrote %s (%ux%u, frame %llu)\n", path.c_str(), width, height,
                 (unsigned long long)frameIndex);
}

void GS::latchHostPresentationFrame()
{
    struct PerfScope
    {
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~PerfScope()
        {
            g_ps2xPerf.presentNs.fetch_add(
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()),
                std::memory_order_relaxed);
        }
    } perfScope;
    GSPresentationRequest request{};
    // Presentation copy ordered in the raster FIFO (GSCpuBackend::
    // EnqueuePresentSnapshot), queued below while m_stateMutex is held.
    // PS2X_GS_PRESENT_UNORDERED=1 restores the old unordered Sync+Present (A/B).
    static const bool s_presentUnordered = []
    {
        const char *value = std::getenv("PS2X_GS_PRESENT_UNORDERED");
        return value != nullptr && value[0] == '1';
    }();
    GSCpuBackend *presentBackend = nullptr;
    uint64_t presentToken = 0u;
    bool displaysDepthBuffer = false;
    // A reset-default DISPLAY window is never visible on real hardware, so
    // holding through it is always right -- no cap (WotM keeps it up for 30+
    // vsyncs at one transition, which the depth-page cap released early).
    bool displaysResetWindow = false;
    {
        std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
        if (!m_backend || !m_privRegs)
        {
            std::lock_guard<std::mutex> presentationLock(m_presentationMutex);
            m_hostPresentationFrame.clear();
            m_hasHostPresentationFrame = false;
            m_hostPresentationWidth = m_hostPresentationHeight = 0u;
            return;
        }
        request = buildPresentationRequestUnlocked();
        // Everything submitted up to here belongs to the frame being presented.
        m_backend->MarkPresentBoundary();
        if (!s_presentUnordered)
            if (GSCpuBackend *cpuBackend = m_backend->PresentSnapshotBackend())
            {
                // Queue it here, while the state lock is held, so the copy lands
                // in raster order right after the display registers were read.
                presentBackend = cpuBackend;
                if (m_backend->NeedsPresentSnapshot(request))
                {
                    presentToken = cpuBackend->EnqueuePresentSnapshot();
                    request.presentToken = presentToken;
                }
            }

        // Is an enabled display circuit pointed at a page that holds a Z buffer?
        // WotM does this for a few vsyncs during screen transitions (PMODE 0x8066,
        // DISPFB2 -> page 0 with a reset-default DISPLAY), and page 0 is its Z
        // buffer, so the presenter was showing depth data as a picture: the
        // whole-screen WHITE flashes the user saw ([gs:flash] tracer: mean 253.5 of
        // 255, every white frame on display page 0; draws log zbuf=(0,...)).
        // DISPFB.FBP and ZBUF.ZBP are both in 2048-word pages.
        const uint32_t zbp0 = m_ctx[0].zbuf.zbp;
        const uint32_t zbp1 = m_ctx[1].zbuf.zbp;
        // Also hold while the enabled circuit's DISPLAY window is still at the
        // reset default (DX=0, DY=0). On a real TV that window sits in the
        // blanking area and is never visible; our presenter ignores the window
        // position, so it showed whatever was in the buffer full-screen. WotM hits
        // this at transitions with page 0 holding either depth data (white) or a
        // solid olive scratch buffer (flash_00_white_t1049). Normal setups never
        // match: sceGsSetDefDispEnv places the window at DX>=0x27C, DY>=0x32.
        auto showsDepth = [&](bool enabled, uint64_t dispfb, uint64_t display)
        {
            if (!enabled)
                return false;
            const uint32_t dx = static_cast<uint32_t>(display & 0xFFFu);
            const uint32_t dy = static_cast<uint32_t>((display >> 12) & 0x7FFu);
            if (dx == 0u && dy == 0u)
            {
                displaysResetWindow = true;
                return true;
            }
            const uint32_t fbp = static_cast<uint32_t>(dispfb & 0x1FFu);
            return fbp == zbp0 || fbp == zbp1 || isRecentDepthPage(fbp);
        };
        displaysDepthBuffer = showsDepth((request.pmode & 0x1u) != 0u, request.dispfb1, request.display1) ||
                              showsDepth((request.pmode & 0x2u) != 0u, request.dispfb2, request.display2);

        // Only frames the game actually flipped to carry a full frame of draws,
        // so compare per flip (display source change), not per vsync.
        static const bool s_geomTrace = []
        {
            const char *value = std::getenv("PS2X_FLASH_TRACE");
            return value != nullptr && value[0] == '1';
        }();
        static uint64_t s_lastFlipDispfb = ~0ull;
        const uint64_t activeDispfb = ((request.pmode & 0x2u) != 0u) ? request.dispfb2 : request.dispfb1;
        if (s_geomTrace && activeDispfb != s_lastFlipDispfb)
        {
            s_lastFlipDispfb = activeDispfb;
            static double s_recentPrims[8] = {};
            static uint32_t s_recentPrimsCount = 0u;
            static uint64_t s_geomDrops = 0u;
            const double prims = static_cast<double>(g_primsSinceLastPresent);
            const uint32_t n = std::min<uint32_t>(s_recentPrimsCount, 8u);
            double avg = 0.0;
            for (uint32_t k = 0; k < n; ++k)
                avg += s_recentPrims[k];
            avg = n ? avg / n : prims;
            if (n >= 4u && avg >= 200.0 && prims < avg * 0.6)
            {
                ++s_geomDrops;
                // Dump only level-scale drops (the intro movie and menu transitions
                // average 1k-13k prims and used up the whole dump quota first).
                static uint32_t s_geomDumpsArmed = 0u;
                if (!g_flashDumpPending && avg >= 20000.0 && s_geomDumpsArmed < 9u)
                {
                    ++s_geomDumpsArmed;
                    g_flashDumpPending = true;
                    std::snprintf(g_flashDumpReason, sizeof(g_flashDumpReason), "geomdrop");
                }
                if (s_geomDrops <= 300u)
                    std::fprintf(stderr, "[gs:flash] GEOMETRY DROP tick=%llu prims=%.0f recentAvg=%.0f (%.0f%%) dispfb=%llx\n",
                                 static_cast<unsigned long long>(request.vsyncTick), prims, avg,
                                 avg > 0.0 ? 100.0 * prims / avg : 0.0,
                                 static_cast<unsigned long long>(activeDispfb));
            }
            s_recentPrims[s_recentPrimsCount % 8u] = prims;
            ++s_recentPrimsCount;
            g_primsSinceLastPresent = 0u;
        }
    }

    // Hold the last good frame instead -- but only if there is one (never at
    // boot), and for at most 30 consecutive vsyncs, so a screen that genuinely
    // displays that page (e.g. before the game ever set up a Z buffer) costs at
    // worst a half-second freeze rather than a stuck screen.
    {
        static uint32_t s_heldDepthFrames = 0u;
        bool havePrevious = false;
        {
            std::lock_guard<std::mutex> presentationLock(m_presentationMutex);
            havePrevious = m_hasHostPresentationFrame && !m_hostPresentationFrame.empty();
        }
        if (displaysDepthBuffer && havePrevious && (displaysResetWindow || s_heldDepthFrames < 180u))
        {
            ++s_heldDepthFrames;
            static const bool s_traceHold = []
            {
                const char *value = std::getenv("PS2X_FLASH_TRACE");
                return value != nullptr && value[0] == '1';
            }();
            if (s_traceHold)
                std::fprintf(stderr, "[gs:flash] HELD last frame tick=%llu (display points at a Z buffer page; pmode=%llx dispfb1=%llx dispfb2=%llx)\n",
                             static_cast<unsigned long long>(request.vsyncTick),
                             static_cast<unsigned long long>(request.pmode),
                             static_cast<unsigned long long>(request.dispfb1),
                             static_cast<unsigned long long>(request.dispfb2));
            return;
        }
        if (!displaysDepthBuffer)
            s_heldDepthFrames = 0u;
    }

#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    {
        // [gs:crt] — privileged display state whenever it changes (cap 48).
        static uint64_t s_last[6] = {~0ull, ~0ull, ~0ull, ~0ull, ~0ull, ~0ull};
        static std::atomic<uint32_t> s_crt{0u};
        const uint64_t cur[6] = {request.pmode, request.smode2, request.dispfb1, request.display1, request.dispfb2, request.display2};
        bool changed = false;
        for (int i = 0; i < 6; ++i)
            changed |= (cur[i] != s_last[i]);
        if (changed && s_crt.fetch_add(1u, std::memory_order_relaxed) < 48u)
        {
            std::memcpy(s_last, cur, sizeof(cur));
            std::fprintf(stderr, "[gs:crt] tick=%llu pmode=%llx smode2=%llx dispfb1=%llx display1=%llx dispfb2=%llx display2=%llx ctx0fbp=%u ctx1fbp=%u\n",
                         (unsigned long long)request.vsyncTick, (unsigned long long)cur[0], (unsigned long long)cur[1],
                         (unsigned long long)cur[2], (unsigned long long)cur[3], (unsigned long long)cur[4], (unsigned long long)cur[5],
                         request.contextFrames[0].fbp, request.contextFrames[1].fbp);
        }
    }
#endif

    PresentationFrame frame{};
    {
        std::lock_guard<std::mutex> backendLock(m_backendLifetimeMutex);
        if (m_backend)
        {
            m_backend->Flush();
            if (presentBackend && presentBackend == m_backend.get())
                frame = presentBackend->PresentSnapshot(presentToken, request);
            else
            {
                m_backend->Sync(GSSyncReason::Presentation);
                frame = m_backend->Present(request);
            }
        }
    }

    const bool hasFrame = static_cast<bool>(frame);
    const uint32_t displayFbp = frame.displayFbp;
    const uint32_t sourceFbp = frame.sourceFbp;
    const uint32_t width = frame.width;
    const uint32_t height = frame.height;
    const bool usedPreferred = frame.usedPreferred;

    // Opt-in sparse RGB captures of the actual presented pixels, paired with
    // gameplay-loop numbers. Diagnostic only; never changes the rendered image.
    static const char *geometryFrameDir = std::getenv("PS2X_GEOMETRY_FRAME_DIR");
    if (geometryFrameDir && *geometryFrameDir && hasFrame && width && height)
    {
        static const uint32_t captureLimit = [] { const char *v=std::getenv("PS2X_GEOMETRY_FRAME_COUNT"); return v?std::clamp(static_cast<uint32_t>(std::strtoul(v,nullptr,10)),1u,64u):10u; }();
        static const uint32_t captureStep = [] { const char *v=std::getenv("PS2X_GEOMETRY_FRAME_STEP"); return v?std::max(1u,static_cast<uint32_t>(std::strtoul(v,nullptr,10))):100u; }();
        static uint32_t captured = 0;
        static uint64_t lastLoop = 0;
        static const uint64_t firstLoop = [] { const char *v=std::getenv("PS2X_GEOMETRY_START_FRAME"); return v?std::strtoull(v,nullptr,10):1200ull; }();
        const uint64_t loop=g_ps2xWotmCompletedFrames.load(std::memory_order_relaxed);
        const size_t bpp=frame.pixels.size()/(size_t(width)*height);
        if(captured<captureLimit && loop>=firstLoop && (!captured || loop>=lastLoop+captureStep) && bpp>=3)
        {
            std::filesystem::path path=std::filesystem::path(geometryFrameDir)/("frame-"+std::to_string(loop)+".ppm");
            if(!std::filesystem::exists(path))
            {
                std::ofstream out(path,std::ios::binary);
                out << "P6\n" << width << " " << height << "\n255\n";
                for(size_t i=0;i<size_t(width)*height;++i)out.write(reinterpret_cast<const char*>(frame.pixels.data()+i*bpp),3);
                out.flush();
                std::fprintf(stderr,"[gs:geometry-frame] loop=%llu %s %s\n",static_cast<unsigned long long>(loop),path.string().c_str(),out?"saved":"failed");
            }
            ++captured;lastLoop=loop;
        }
    }

    // [gs:flash] PS2X_FLASH_TRACE=1 -- per-presented-frame flash detector. The
    // user sees whole-screen white flashes and objects popping in and out that
    // sparse window screenshots cannot catch (they sample ~1 in 5 frames and
    // only see the last composited frame). This looks at EVERY frame the
    // presenter hands out: mean luminance of every 64th pixel vs the average of
    // the previous 8 frames, plus which framebuffer was shown. A white frame that
    // coincides with a display-source change means wrong-buffer selection; one
    // with a stable source means the content itself (draw ordering) is wrong.
    static const bool s_flashTrace = []
    {
        const char *value = std::getenv("PS2X_FLASH_TRACE");
        return value != nullptr && value[0] == '1';
    }();
    if (s_flashTrace && hasFrame)
    {
        static double s_recent[8] = {};
        static uint32_t s_recentCount = 0u;
        static uint64_t s_frames = 0u, s_flashes = 0u, s_sourceChanges = 0u;
        static uint32_t s_lastDisplay = ~0u, s_lastSource = ~0u, s_lastW = 0u, s_lastH = 0u;
        static bool s_lastPreferred = false;

        const size_t pixelCount = static_cast<size_t>(width) * height;
        const size_t bpp = pixelCount ? frame.pixels.size() / pixelCount : 0u;
        double lum = 0.0;
        size_t samples = 0u;
        if (bpp >= 3u)
        {
            for (size_t px = 0; px < pixelCount; px += 64u)
            {
                const uint8_t *c = frame.pixels.data() + px * bpp;
                lum += 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
                ++samples;
            }
        }
        const double mean = samples ? lum / static_cast<double>(samples) : 0.0;
        double avg = 0.0;
        const uint32_t n = std::min<uint32_t>(s_recentCount, 8u);
        for (uint32_t k = 0; k < n; ++k)
            avg += s_recent[k];
        avg = n ? avg / n : mean;

        const bool sourceChanged = s_frames > 0u &&
                                   (displayFbp != s_lastDisplay || sourceFbp != s_lastSource ||
                                    usedPreferred != s_lastPreferred || width != s_lastW || height != s_lastH);
        if (sourceChanged)
        {
            ++s_sourceChanges;
            if (s_sourceChanges <= 200u)
                std::fprintf(stderr,
                             "[gs:flash] source change tick=%llu display %u->%u source %u->%u preferred %d->%d size %ux%u->%ux%u\n",
                             static_cast<unsigned long long>(request.vsyncTick), s_lastDisplay, displayFbp, s_lastSource,
                             sourceFbp, s_lastPreferred ? 1 : 0, usedPreferred ? 1 : 0, s_lastW, s_lastH, width, height);
        }
        // PS2X_FLASH_REGION=1: every presented frame's mean luminance of the screen
        // centre (x 25-75%, y 22-67%: where WotM's drive-in billboard sits) next to
        // the whole-frame mean, to tell a smooth camera pan (gradual change) from
        // the billboard popping in and out (single-frame jumps).
        static const bool s_regionTrace = []
        {
            const char *value = std::getenv("PS2X_FLASH_REGION");
            return value != nullptr && value[0] == '1';
        }();
        if (s_regionTrace && bpp >= 3u && width >= 8u && height >= 8u)
        {
            double centre = 0.0;
            size_t centreSamples = 0u;
            for (uint32_t y = height * 22u / 100u; y < height * 67u / 100u; y += 4u)
                for (uint32_t x = width / 4u; x < width * 3u / 4u; x += 4u)
                {
                    const uint8_t *c = frame.pixels.data() + (static_cast<size_t>(y) * width + x) * bpp;
                    centre += 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
                    ++centreSamples;
                }
            // The dark drive-in makes mean luminance useless (sky-only 32 vs
            // billboard 34). logo = dim-orange "MONSTERS" logo pixels (x 22-78%,
            // y 13-42%; logo ~R90 G36-52 B30, sky is grey R~G~B) -> 0 when the
            // billboard is not drawn. diff = mean |luma - previous frame's| on a
            // coarse grid -> spikes when things pop in or out.
            uint32_t logo = 0u;
            for (uint32_t y = height * 13u / 100u; y < height * 42u / 100u; y += 2u)
                for (uint32_t x = width * 22u / 100u; x < width * 78u / 100u; x += 2u)
                {
                    const uint8_t *c = frame.pixels.data() + (static_cast<size_t>(y) * width + x) * bpp;
                    if (c[0] > 60u && c[0] > c[1] + 25u && c[0] > c[2] + 35u)
                        ++logo;
                }
            static std::vector<uint8_t> s_prevGrid;
            std::vector<uint8_t> grid;
            grid.reserve((width / 8u + 1u) * (height / 8u + 1u));
            for (uint32_t y = 0; y < height; y += 8u)
                for (uint32_t x = 0; x < width; x += 8u)
                {
                    const uint8_t *c = frame.pixels.data() + (static_cast<size_t>(y) * width + x) * bpp;
                    grid.push_back(static_cast<uint8_t>((77u * c[0] + 150u * c[1] + 29u * c[2]) >> 8));
                }
            double diff = 0.0;
            if (s_prevGrid.size() == grid.size() && !grid.empty())
            {
                for (size_t k = 0; k < grid.size(); ++k)
                    diff += std::abs(static_cast<int>(grid[k]) - static_cast<int>(s_prevGrid[k]));
                diff /= static_cast<double>(grid.size());
            }
            s_prevGrid.swap(grid);
            std::fprintf(stderr, "[gs:region] tick=%llu centre=%.1f full=%.1f logo=%u diff=%.2f display=%u\n",
                         static_cast<unsigned long long>(request.vsyncTick),
                         centreSamples ? centre / static_cast<double>(centreSamples) : 0.0, mean, logo, diff,
                         displayFbp);
        }
        const bool white = n >= 4u && mean > avg + 40.0 && mean > avg * 1.6;
        static uint32_t s_whiteDumpsArmed = 0u;
        if (white && !g_flashDumpPending && s_whiteDumpsArmed < 3u)
        {
            ++s_whiteDumpsArmed;
            g_flashDumpPending = true;
            std::snprintf(g_flashDumpReason, sizeof(g_flashDumpReason), "white");
        }
        static std::vector<uint8_t> s_prevPixels;
        static uint32_t s_prevW = 0u, s_prevH = 0u;
        static const char *s_dumpDir = std::getenv("PS2X_FLASH_DUMP_DIR");
        if (g_flashDumpPending && s_dumpDir && g_flashDumpsWritten < 12u)
        {
            char name[160];
            std::snprintf(name, sizeof(name), "/flash_%02u_%s_t%llu_now.bmp", g_flashDumpsWritten, g_flashDumpReason,
                          static_cast<unsigned long long>(request.vsyncTick));
            writeBmp(std::string(s_dumpDir) + name, frame.pixels.data(), width, height, bpp);
            if (!s_prevPixels.empty() && s_prevW == width && s_prevH == height)
            {
                std::snprintf(name, sizeof(name), "/flash_%02u_%s_t%llu_before.bmp", g_flashDumpsWritten, g_flashDumpReason,
                              static_cast<unsigned long long>(request.vsyncTick));
                writeBmp(std::string(s_dumpDir) + name, s_prevPixels.data(), width, height, bpp);
            }
            ++g_flashDumpsWritten;
        }
        g_flashDumpPending = false;
        s_prevPixels = frame.pixels;
        s_prevW = width;
        s_prevH = height;
        const bool dark = n >= 4u && avg > 20.0 && mean < avg * 0.4;
        if (white || dark)
        {
            ++s_flashes;
            if (s_flashes <= 200u)
                std::fprintf(stderr,
                             "[gs:flash] %s FRAME tick=%llu mean=%.1f recentAvg=%.1f display=%u source=%u preferred=%d size=%ux%u%s\n",
                             white ? "WHITE" : "DARK", static_cast<unsigned long long>(request.vsyncTick), mean, avg,
                             displayFbp, sourceFbp, usedPreferred ? 1 : 0, width, height,
                             sourceChanged ? " (source changed this frame)" : "");
        }
        s_recent[s_recentCount % 8u] = mean;
        ++s_recentCount;
        ++s_frames;
        s_lastDisplay = displayFbp;
        s_lastSource = sourceFbp;
        s_lastPreferred = usedPreferred;
        s_lastW = width;
        s_lastH = height;
        if ((s_frames % 300u) == 0u)
            std::fprintf(stderr, "[gs:flash] summary frames=%llu flashes=%llu sourceChanges=%llu lastMean=%.1f\n",
                         static_cast<unsigned long long>(s_frames), static_cast<unsigned long long>(s_flashes),
                         static_cast<unsigned long long>(s_sourceChanges), mean);
    }

    {
        std::lock_guard<std::mutex> presentationLock(m_presentationMutex);
        {
            static uint64_t s_presented = 0u;
            ps2xGsSaveFrame(frame.pixels.data(), frame.width, frame.height, ++s_presented);
        }
        m_hostPresentationFrame = std::move(frame.pixels);
        m_hostPresentationWidth = width;
        m_hostPresentationHeight = height;
        m_hostPresentationDisplayFbp = displayFbp;
        m_hostPresentationSourceFbp = sourceFbp;
        m_hostPresentationUsedPreferred = usedPreferred;
        m_hasHostPresentationFrame = hasFrame;
    }

    if (hasFrame)
    {
        std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
        recordPresentDebugEventUnlocked(displayFbp, sourceFbp, width, height, usedPreferred);
    }
}

bool GS::copyLatchedHostPresentationFrame(std::vector<uint8_t> &outPixels,
                                          uint32_t &outWidth,
                                          uint32_t &outHeight,
                                          uint32_t *outDisplayFbp,
                                          uint32_t *outSourceFbp,
                                          bool *outUsedPreferred) const
{
    std::lock_guard<std::mutex> lock(m_presentationMutex);
    if (!m_hasHostPresentationFrame || m_hostPresentationFrame.empty())
    {
        outPixels.clear();
        outWidth = 0u;
        outHeight = 0u;
        if (outDisplayFbp)
            *outDisplayFbp = 0u;
        if (outSourceFbp)
            *outSourceFbp = 0u;
        if (outUsedPreferred)
            *outUsedPreferred = false;
        return false;
    }

    outWidth = m_hostPresentationWidth;
    outHeight = m_hostPresentationHeight;
    if (outDisplayFbp)
        *outDisplayFbp = m_hostPresentationDisplayFbp;
    if (outSourceFbp)
        *outSourceFbp = m_hostPresentationSourceFbp;
    if (outUsedPreferred)
        *outUsedPreferred = m_hostPresentationUsedPreferred;

    const size_t packedRowBytes = static_cast<size_t>(outWidth) * 4u;
    outPixels.resize(packedRowBytes * static_cast<size_t>(outHeight));
    if (outWidth != 0u && outHeight != 0u)
    {
        const size_t sourceRowBytes = static_cast<size_t>(kHostFrameWidth) * 4u;
        for (uint32_t y = 0; y < outHeight; ++y)
        {
            const size_t srcOffset = static_cast<size_t>(y) * sourceRowBytes;
            const size_t dstOffset = static_cast<size_t>(y) * packedRowBytes;
            if (srcOffset + packedRowBytes > m_hostPresentationFrame.size() ||
                dstOffset + packedRowBytes > outPixels.size())
            {
                outPixels.clear();
                outWidth = 0u;
                outHeight = 0u;
                if (outDisplayFbp)
                    *outDisplayFbp = 0u;
                if (outSourceFbp)
                    *outSourceFbp = 0u;
                if (outUsedPreferred)
                    *outUsedPreferred = false;
                return false;
            }

            std::memcpy(outPixels.data() + dstOffset,
                        m_hostPresentationFrame.data() + srcOffset,
                        packedRowBytes);
        }
    }
    return true;
}

bool ps2xVu1ForeignAccessWhileBusy(); // ps2_memory.cpp
std::atomic<uint64_t> g_ps2xGsForeignHazards{0u};

void GS::processGIFPacket(const uint8_t *data, uint32_t sizeBytes)
{
    const auto gifT0 = Ps2xTscClock::now();
    // PS2X_VU1_THREAD: a thread other than the VU1 worker feeding the GS while
    // the worker still has drawing queued can reorder draws. Every such path
    // should drain first; count (and name a few of) any that do not.
    if (ps2xVu1ForeignAccessWhileBusy())
    {
        const uint64_t n = g_ps2xGsForeignHazards.fetch_add(1u, std::memory_order_relaxed) + 1u;
        if (n <= 5u)
            std::fprintf(stderr, "[vu1thread] GS ORDERING HAZARD #%llu: non-worker thread sent a GIF packet while the worker was busy\n",
                         static_cast<unsigned long long>(n));
    }
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    struct GifPerfScope
    {
        Ps2xTscClock::time_point t0;
        ~GifPerfScope()
        {
            g_ps2xGifNs.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                            Ps2xTscClock::now() - t0)
                                                            .count()),
                                  std::memory_order_relaxed);
        }
    } gifPerfScope{gifT0};
    g_ps2xGifLockNs.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                        Ps2xTscClock::now() - gifT0)
                                                        .count()),
                              std::memory_order_relaxed);
    if (!data || sizeBytes < 16 || !m_backend)
        return;

#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    // [gs:pkt] — every packet handed to the GIF parser during boot, with the
    // path it came from and the image bytes still owed *before* this packet.
    // A packet whose q0 is not a plausible GIFtag while pendImg==0 is the
    // moment raw pixels get parsed as tags (=> garbage register kicks).
    extern int g_gifArbiterCurrentPath;
    {
        static std::atomic<uint32_t> s_pkt{0u};
        const uint32_t n = s_pkt.fetch_add(1u, std::memory_order_relaxed);
        if (n < 96u)
        {
            const uint64_t q0 = loadLE64(data), q0h = loadLE64(data + 8);
            const uint64_t qL = loadLE64(data + sizeBytes - 16), qLh = loadLE64(data + sizeBytes - 8);
            std::fprintf(stderr, "[gs:pkt] #%u path=%d bytes=%u pendImg=%u q0=%016llx.%016llx qLast=%016llx.%016llx\n",
                         n, g_gifArbiterCurrentPath, sizeBytes, s_pendingImageBytes,
                         (unsigned long long)q0h, (unsigned long long)q0,
                         (unsigned long long)qLh, (unsigned long long)qL);
        }
    }
#endif

    // Whole strips have immutable material and a reset vertex queue. Retain
    // unique vertices and ADC decisions through packet decoding and submission.
    // Diagnostic or continuing tags retain the established per-vertex path.
    auto indexedStrip = [&](const GSVertex *input, const uint8_t *adc, uint32_t count,
                            const uint8_t *raw, NativeGeometryBatch &geometry, bool verify) {
        if (!s_nativeIndexedEnabled || !geometry.backend || m_prim.type!=GS_PRIM_TRISTRIP ||
            count<3u || count>256u || m_vtxCount!=0 || m_vtxIndex!=0 || !m_debugHistoryPaused ||
            s_nearCullReference || s_sliverCull || s_bigPrimCensus || s_bigTri ||
            captureStretch || traceFrame!=~0ull) return false;
        const bool packed=input==nullptr;
        if (packed && (!s_packedWorld || !raw || m_prim.fst || count<32u)) return false;
#if defined(_WIN32)
        // Preserve legacy exceptional Q values and unmasked trap order.
        if(packed && !wotm_vertex::packedWorldQAdmitted(raw,count,_mm_getcsr()))return false;
#else
        if(packed)return false;
#endif
        verify=verify || s_nativeIndexedVerify;
        static thread_local std::array<GSVertex,256> vertices;
        if (!packed || verify) for (uint32_t i=0;i<count;++i) {
            vertices[i]=packed ? wotm_vertex::decodePackedGSVertex<GSVertex>(raw+i*48u,m_curU,m_curV) : input[i];
            vertices[i].u=m_curU;vertices[i].v=m_curV;
        }
        std::array<uint16_t,762> indices;
        size_t indexCount=0;
        const bool nearTest=s_nearCull && m_prim.tme && !m_prim.fst;
        std::array<uint8_t,256> positive;
        if (packed && nearTest) for(uint32_t i=0;i<count;++i) {
            float q;std::memcpy(&q,raw+i*48u+8u,4);
            if(q==0.0f)q=1.0f;
            positive[i]=q>0.0f;
        }
        for (uint32_t i=2;i<count;++i) {
            const bool disabled=packed ? ((loadLE64(raw+i*48u+40u)>>47u)&1u)!=0 : adc[i]!=0;
            if (disabled || (nearTest && (packed ?
                (!positive[i-2] || !positive[i-1] || !positive[i]) :
                (!(vertices[i-2].q>0.0f) || !(vertices[i-1].q>0.0f) || !(vertices[i].q>0.0f))))) continue;
            indices[indexCount++]=uint16_t(i-2);
            indices[indexCount++]=uint16_t(i-1);
            indices[indexCount++]=uint16_t(i);
        }
        // Sparse/empty strips fall through without publishing any GS state.
        if(packed && indexCount<count)return false;
        if(packed && !verify) {
            for(unsigned j=0;j<3;++j) {
                const auto i=indices[j];
                vertices[i]=wotm_vertex::decodePackedGSVertex<GSVertex>(raw+i*48u,m_curU,m_curV);
            }
            vertices[count-2]=wotm_vertex::decodePackedGSVertex<GSVertex>(raw+(count-2)*48u,m_curU,m_curV);
            vertices[count-1]=wotm_vertex::decodePackedGSVertex<GSVertex>(raw+(count-1)*48u,m_curU,m_curV);
        }
        GSPrimitiveBatch prototype=buildDrawBatch(3);
        if (indexCount)
            for (unsigned j=0;j<3;++j) prototype.vertices[j]=vertices[indices[j]];
        const auto &last=vertices[count-1];
        if (verify) {
            for (uint32_t i=0;i<count;++i) {
                const auto *p=raw+i*48u;
                s_vuExpectedVertex=&vertices[i];
                s_vuExpectedDrawing=packed ? ((loadLE64(p+40u)>>47u)&1u)==0 : adc[i]==0;
                writeRegisterPacked(2,loadLE64(p),loadLE64(p+8));
                writeRegisterPacked(1,loadLE64(p+16),loadLE64(p+24));
                writeRegisterPacked(4,loadLE64(p+32),loadLE64(p+40));
                s_vuExpectedVertex=nullptr;
            }
            constexpr size_t bytes=offsetof(GSVertex,fog)+sizeof(uint8_t);
            bool same=geometry.vertices.size()==indexCount && m_vtxCount==2 && m_vtxIndex==count;
            for (size_t i=0;same && i<indexCount;++i)
                same=std::memcmp(&geometry.vertices[i],&vertices[indices[i]],bytes)==0;
            for (unsigned i=0;same && i<3;++i)
                same=std::memcmp(&m_vtxQueue[i],&vertices[count-(i==0?2u:1u)],bytes)==0;
            const auto &a=geometry.prototype.state,&b=prototype.state;
            // Compare all fields, excluding only uninitialized aggregate padding.
            const bool stateSame=std::memcmp(&a.context,&b.context,sizeof(a.context))==0 &&
                std::memcmp(&a.prim,&b.prim,sizeof(a.prim))==0 &&
                std::memcmp(&a.texa,&b.texa,sizeof(a.texa))==0 &&
                std::memcmp(&a.texclut,&b.texclut,sizeof(a.texclut))==0 &&
                a.pabe==b.pabe && a.scanmsk==b.scanmsk && a.dimx==b.dimx && a.dthe==b.dthe &&
                a.colclamp==b.colclamp && a.fogR==b.fogR && a.fogG==b.fogG && a.fogB==b.fogB &&
                a.textureWidth==b.textureWidth && a.textureHeight==b.textureHeight && a.linearFilter==b.linearFilter;
            same=same && m_curR==last.r && m_curG==last.g && m_curB==last.b && m_curA==last.a &&
                std::memcmp(&m_curS,&last.s,4)==0 && std::memcmp(&m_curT,&last.t,4)==0 &&
                std::memcmp(&m_curQ,&last.q,4)==0 &&
                (!indexCount || stateSame);
            if (!same) {
                std::fprintf(stderr,"[gs:indexed-strip] MISMATCH count=%u output=%zu expected=%zu queue=%u index=%u near=%u\n",
                    count,geometry.vertices.size(),indexCount,m_vtxCount,m_vtxIndex,nearTest);
                for (size_t i=0;i<std::min(indexCount,geometry.vertices.size());++i)
                    if (std::memcmp(&geometry.vertices[i],&vertices[indices[i]],bytes)) {
                        const auto &a=geometry.vertices[i],&b=vertices[indices[i]];
                        std::fprintf(stderr,"[gs:indexed-strip] vertex=%zu index=%u xy=%g,%g/%g,%g z=%g/%g q=%g/%g uv=%u,%u/%u,%u\n",
                            i,indices[i],a.x,a.y,b.x,b.y,a.z,b.z,a.q,b.q,a.u,a.v,b.u,b.v);break;
                    }
                for (unsigned i=0;i<3;++i)
                    std::fprintf(stderr,"[gs:indexed-strip] queue%u same=%u\n",i,
                        std::memcmp(&m_vtxQueue[i],&vertices[count-(i==0?2u:1u)],bytes)==0);
                std::fprintf(stderr,"[gs:indexed-strip] material=%u color=%u stq=%u\n",
                    std::memcmp(&geometry.prototype.state,&prototype.state,sizeof(GSDrawState))==0,
                    m_curR==last.r && m_curG==last.g && m_curB==last.b && m_curA==last.a,
                    std::memcmp(&m_curS,&last.s,4)==0 && std::memcmp(&m_curT,&last.t,4)==0 && std::memcmp(&m_curQ,&last.q,4)==0);
                std::abort();
            }
            geometry.vertices.clear();
            static thread_local uint64_t checked=0;
            if (++checked==1 || checked%100000==0)
                std::fprintf(stderr,"[gs:indexed-strip] verified=%llu mismatches=0\n",checked);
        } else {
            m_vtxQueue[0]=vertices[count-2];m_vtxQueue[1]=m_vtxQueue[2]=last;
            m_vtxCount=2;m_vtxIndex=count;
            m_curR=last.r;m_curG=last.g;m_curB=last.b;m_curA=last.a;
            m_curS=last.s;m_curT=last.t;m_curQ=last.q;
            if (indexCount) {
                updatePreferredDisplaySourceForDraw(prototype);
                if (!prototype.state.context.zbuf.zmask) noteDepthPage(prototype.state.context.zbuf.zbp);
                g_primsSinceLastPresent+=indexCount/3u;
            }
        }
        geometry.prototype=prototype;geometry.valid=indexCount!=0;
        if(packed) {
            ps2xGlSubmitPackedWorldGeometry(*geometry.backend,prototype,raw,count,
                indices.data(),indexCount,m_curU,m_curV);
            geometry.noteSubmission(indexCount);
        } else geometry.flushIndexed(vertices.data(),count,indices.data(),indexCount);
        return true;
    };

    if (s_vuMeshMessage && s_vuMeshMessage->gs == this && sizeBytes == 16u &&
        loadLE64(data) == kNativeVuToken && loadLE64(data + 8u) == reinterpret_cast<uintptr_t>(s_vuMeshMessage)) {
        const auto &mesh = *s_vuMeshMessage;
        if (s_pendingImageBytes != 0u) {
            // Preserve the existing continuation semantics in unusual streams.
            data = mesh.raw; sizeBytes = mesh.bytes;
        } else {
            m_curQ = 1.0f;
            recordGifTagDebugEventUnlocked(mesh.bytes, mesh.count, GIF_FMT_PACKED, 3u);
            writeRegisterUnlocked(GS_REG_PRIM, (mesh.tag >> 47u) & 0x7ffu);
            std::shared_ptr<MotionProvenance::Mesh> motion;
            if(MotionProvenance::vectorsEnabled()) {
                const auto& context=m_ctx[m_prim.ctxt];
                const float ox=float(context.xyoffset.ofx)/16.f;
                float oy=float(context.xyoffset.ofy)/16.f;
                static const bool snap=[] {const char* p=std::getenv("PS2X_GS_OFY_SNAP");return !(p && p[0]=='0');}();
                if(snap)oy=std::floor(oy);
                std::vector<MotionProvenance::Point> points;points.reserve(mesh.count);
                for(uint32_t i=0;i<mesh.count;++i){const auto& v=mesh.vertices[i];points.push_back({v.x-ox,v.y-oy,float(v.z/16777215.0),v.q});}
                motion=MotionProvenance::prepareMesh(points,mesh.adc,mesh.tag);
            }
            MotionProvenance::MeshScope motionScope(std::move(motion));
            NativeGeometryBatch geometry(!s_nativeGeometryEnabled || s_nativeGeometry ? nullptr : dynamic_cast<GSGlBackend *>(m_backend.get()), this);
            static const bool verify = std::getenv("PS2X_VU1_DIRECT_VERIFY") != nullptr;
            static const bool stripsEnabled = [] {
                const char *p=std::getenv("PS2X_NATIVE_STRIPS"); return !p || p[0]!='0';
            }();
            static const bool stripsVerify=std::getenv("PS2X_NATIVE_STRIPS_VERIFY")!=nullptr;
            const uint32_t start=g_ps2xVu1Kick.packet?g_ps2xVu1Kick.startPc:~0u;
            const bool skinned=start==0x3060u || start==0x19e8u || start==0x1868u || start==0xbf8u;
            // PRIM above resets the assembly queue. A strip can therefore
            // be assembled directly from adjacent triples, including ADC skips.
            // Keep the diagnostic/skin-specific handlers on the generic path.
            const bool nativeStrip=stripsEnabled && !s_nearCullReference && geometry.backend && m_prim.type==GS_PRIM_TRISTRIP &&
                mesh.count>=3u && mesh.count<=256u && m_debugHistoryPaused &&
                !(s_sliverCull && skinned) && !s_bigPrimCensus && !s_bigTri &&
                !captureStretch && traceFrame==~0ull && (!verify || stripsVerify);
            if (stripsEnabled && indexedStrip(mesh.vertices,mesh.adc,mesh.count,mesh.raw+16u,
                                              geometry,verify || stripsVerify)) return;
            static thread_local std::vector<GSVertex> stripReference;
            if(nativeStrip) {
                auto &out=stripsVerify?stripReference:geometry.vertices;
                out.clear();out.reserve((mesh.count-2u)*3u);
                const bool nearTest=s_nearCull && m_prim.tme && !m_prim.fst;
                for(uint32_t i=2;i<mesh.count;++i) {
                    if(mesh.adc[i] || (nearTest && (!(mesh.vertices[i-2].q>0.0f) ||
                        !(mesh.vertices[i-1].q>0.0f) || !(mesh.vertices[i].q>0.0f)))) continue;
                    for(uint32_t j=i-2;j<=i;++j) {
                        GSVertex vertex=mesh.vertices[j];vertex.u=m_curU;vertex.v=m_curV;
                        out.push_back(vertex);
                    }
                }
                if(!stripsVerify) {
                    if(!out.empty()) {
                        geometry.prototype=buildDrawBatch(3);
                        std::copy_n(out.data(),3,geometry.prototype.vertices.begin());
                        geometry.valid=true;
                        updatePreferredDisplaySourceForDraw(geometry.prototype);
                        if(!geometry.prototype.state.context.zbuf.zmask)
                            noteDepthPage(geometry.prototype.state.context.zbuf.zbp);
                        g_primsSinceLastPresent+=out.size()/3u;
                    }
                    // Preserve the complete continuation state of vertexKick,
                    // including the duplicate final queue slot and current color.
                    GSVertex last=mesh.vertices[mesh.count-1u];last.u=m_curU;last.v=m_curV;
                    m_vtxQueue[0]=mesh.vertices[mesh.count-2u];
                    m_vtxQueue[0].u=m_curU;m_vtxQueue[0].v=m_curV;
                    m_vtxQueue[1]=m_vtxQueue[2]=last;
                    m_vtxCount=2;m_vtxIndex=mesh.count;
                    m_curR=last.r;m_curG=last.g;m_curB=last.b;m_curA=last.a;
                    m_curS=last.s;m_curT=last.t;m_curQ=last.q;
                    geometry.flush();
                    return;
                }
            }
            for (uint32_t i = 0; i < mesh.count; ++i) {
                GSVertex vertex = mesh.vertices[i];
                vertex.u = m_curU; vertex.v = m_curV;
                if (verify) {
                    // The unchanged packed decoder remains authoritative in
                    // verification mode; vertexKick checks the independent result.
                    s_vuExpectedVertex = &vertex;
                    s_vuExpectedDrawing = mesh.adc[i] == 0;
                    const uint8_t *p = mesh.raw + 16u + i * 48u;
                    writeRegisterPacked(2, loadLE64(p), loadLE64(p + 8));
                    writeRegisterPacked(1, loadLE64(p + 16), loadLE64(p + 24));
                    writeRegisterPacked(4, loadLE64(p + 32), loadLE64(p + 40));
                    s_vuExpectedVertex = nullptr;
                } else {
                    m_curR = vertex.r; m_curG = vertex.g; m_curB = vertex.b; m_curA = vertex.a;
                    m_curS = vertex.s; m_curT = vertex.t; m_curQ = vertex.q;
                    m_vtxQueue[m_vtxCount % kMaxVerts] = vertex;
                    vertexKick(mesh.adc[i] == 0);
                }
            }
            if(nativeStrip && stripsVerify) {
                constexpr size_t vertexBytes=offsetof(GSVertex,fog)+sizeof(uint8_t);
                bool same=stripReference.size()==geometry.vertices.size() &&
                    m_vtxCount==2 && m_vtxIndex==mesh.count;
                for(size_t i=0;same && i<stripReference.size();++i)
                    same=std::memcmp(&stripReference[i],&geometry.vertices[i],vertexBytes)==0;
                for(unsigned i=0;same && i<3;++i) {
                    auto expected=mesh.vertices[mesh.count-(i==0?2u:1u)];
                    expected.u=m_curU;expected.v=m_curV;
                    same=std::memcmp(&expected,&m_vtxQueue[i],vertexBytes)==0;
                }
                const auto &last=mesh.vertices[mesh.count-1];
                same=same && m_curR==last.r && m_curG==last.g && m_curB==last.b && m_curA==last.a &&
                    std::memcmp(&m_curS,&last.s,4)==0 && std::memcmp(&m_curT,&last.t,4)==0 &&
                    std::memcmp(&m_curQ,&last.q,4)==0;
                if(!same) {std::fprintf(stderr,"[gs:native-strip] MISMATCH\n");std::abort();}
                static uint64_t checked=0;
                if(++checked==1 || checked%100000==0)
                    std::fprintf(stderr,"[gs:native-strip] verified=%llu mismatches=0\n",checked);
            }
            if (geometry.backend) geometry.flush();
            return;
        }
    }

    if (tryProcessNativeImageUploadPacket(data, sizeBytes))
    {
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
        static std::atomic<uint32_t> s_nat{0u};
        if (s_nat.fetch_add(1u, std::memory_order_relaxed) < 64u)
            std::fprintf(stderr, "[gs:pkt]   -> native image upload (%u bytes)\n", sizeBytes);
#endif
        return;
    }

    // Continue an IMAGE-mode transfer whose payload spilled past the previous
    // DMA packet: these bytes are raw pixels for the GS, not a GIFtag stream.
    if (s_pendingImageBytes > 0u)
    {
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
        static std::atomic<uint32_t> s_cont{0u};
        if (s_cont.fetch_add(1u, std::memory_order_relaxed) < 64u)
            std::fprintf(stderr, "[gs:pkt]   -> image continuation: take=%u of pend=%u (pkt %u)\n",
                         std::min(s_pendingImageBytes, sizeBytes), s_pendingImageBytes, sizeBytes);
#endif
        const uint32_t take = std::min(s_pendingImageBytes, sizeBytes);
        processImageData(data, take);
        s_pendingImageBytes -= take;
        if (take >= sizeBytes)
            return;
        data += take;
        sizeBytes -= take;
    }

    PS2_IF_AGRESSIVE_LOGS({
        const uint32_t packetIndex = s_debugGifPacketCount.fetch_add(1, std::memory_order_relaxed);
        if (packetIndex < 48u)
        {
            const uint64_t tagLo = loadLE64(data);
            const uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFFu);
            const uint8_t flg = static_cast<uint8_t>((tagLo >> 58) & 0x3u);
            uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xFu);
            if (nreg == 0u)
                nreg = 16u;
            RUNTIME_LOG("[gs:gif] idx=" << packetIndex
                                        << " size=" << sizeBytes
                                        << " nloop=" << nloop
                                        << " flg=" << static_cast<uint32_t>(flg)
                                        << " nreg=" << nreg
                                        << " ctx0fbp=" << m_ctx[0].frame.fbp
                                        << " ctx1fbp=" << m_ctx[1].frame.fbp
                                        << std::endl);
        }
    });

    uint32_t offset = 0;
    while (offset + 16 <= sizeBytes)
    {
        uint64_t tagLo = loadLE64(data + offset);
        uint64_t tagHi = loadLE64(data + offset + 8);
        offset += 16;

        m_curQ = 1.0f;

        uint32_t nloop = static_cast<uint32_t>(tagLo & 0x7FFF);
        uint8_t flg = static_cast<uint8_t>((tagLo >> 58) & 0x3);
        uint32_t nreg = static_cast<uint32_t>((tagLo >> 60) & 0xF);
        if (nreg == 0)
            nreg = 16;

        recordGifTagDebugEventUnlocked(sizeBytes, nloop, flg, nreg);

        bool pre = ((tagLo >> 46) & 1) != 0;
        if (pre)
        {
            writeRegisterUnlocked(GS_REG_PRIM, (tagLo >> 47) & 0x7FF);
        }

        uint8_t regs[16];
        for (uint32_t i = 0; i < nreg; ++i)
            regs[i] = static_cast<uint8_t>((tagHi >> (i * 4)) & 0xF);

        if (flg == GIF_FMT_PACKED)
        {
            const bool triangles = m_prim.type == GS_PRIM_TRIANGLE ||
                m_prim.type == GS_PRIM_TRISTRIP || m_prim.type == GS_PRIM_TRIFAN;
            // Validate the whole payload before activating the deferred stream.
            // Truncated tags retain the old incremental handling.
            GSGlBackend *nativeBackend = nullptr;
            if (s_nativeGeometryEnabled && !s_nativeGeometry && triangles &&
                uint64_t(nloop) * nreg * 16u <= sizeBytes - offset &&
                homogeneousGeometryTag(regs, nreg))
                nativeBackend = dynamic_cast<GSGlBackend *>(m_backend.get());
            NativeGeometryBatch geometry(nativeBackend, this);
            static const bool meshDecode = [] { const char *p = std::getenv("PS2X_GS_MESH_DECODE"); return !p || p[0] != '0'; }();
            static const bool meshVerify = std::getenv("PS2X_GS_MESH_VERIFY") != nullptr;
            const bool single = nreg == 3u && (tagHi & 0xfffu) == 0x412u;
            const bool paired = nreg == 6u && (tagHi & 0xffffffu) == 0x4f2412u;
            if (meshDecode && ((triangles && single) || (paired && m_prim.type == GS_PRIM_SPRITE)) &&
                uint64_t(nloop) * nreg * 16u <= sizeBytes - offset) {
                // These layouts contain only STQ, optional RGBA, XYZF2 and NOP.
                // State-setting tags before/after this tag still use the normal
                // decoder. Preserve color carry between the paired vertices,
                // UV carry, ADC, and the existing strip/fan assembly and culling.
                if (single && nloop>=3u && nloop<=256u && nativeBackend &&
                    m_prim.type==GS_PRIM_TRISTRIP && m_vtxCount==0 && m_vtxIndex==0 &&
                    s_nativeIndexedEnabled) {
                    if(s_packedWorld && indexedStrip(nullptr,nullptr,nloop,data+offset,geometry,meshVerify)) {
                        offset+=nloop*48u;continue;
                    }
                    static thread_local std::array<GSVertex,256> decoded;
                    std::array<uint8_t,256> adc;
                    for (uint32_t i=0;i<nloop;++i) {
                        const auto *p=data+offset+i*48u;
                        auto &v=decoded[i];
                        std::memcpy(&v.s,p,4);std::memcpy(&v.t,p+4,4);std::memcpy(&v.q,p+8,4);
                        if (v.q==0.0f) v.q=1.0f;
                        v.r=p[16];v.g=p[20];v.b=p[24];v.a=p[28];
                        v.u=m_curU;v.v=m_curV;
                        const auto xy=loadLE64(p+32),zf=loadLE64(p+40);
                        v.x=float(xy&0xffffu)/16.0f;v.y=float((xy>>32)&0xffffu)/16.0f;
                        v.z=float((zf>>4)&0xffffffu);v.fog=uint8_t(zf>>36);
                        adc[i]=uint8_t((zf>>47)&1u);
                    }
                    if (indexedStrip(decoded.data(),adc.data(),nloop,data+offset,geometry,meshVerify)) {
                        offset+=nloop*48u;continue;
                    }
                }
                auto emit = [&](const uint8_t *st, const uint8_t *rgba, const uint8_t *xyz) {
                    GSVertex vertex{};
                    std::memcpy(&vertex.s, st, 4); std::memcpy(&vertex.t, st + 4, 4);
                    std::memcpy(&vertex.q, st + 8, 4);
                    if (vertex.q == 0.0f) vertex.q = 1.0f;
                    vertex.r = rgba ? rgba[0] : m_curR; vertex.g = rgba ? rgba[4] : m_curG;
                    vertex.b = rgba ? rgba[8] : m_curB; vertex.a = rgba ? rgba[12] : m_curA;
                    vertex.u = m_curU; vertex.v = m_curV;
                    const uint64_t xy = loadLE64(xyz), zf = loadLE64(xyz + 8);
                    vertex.x = float(xy & 0xffffu) / 16.0f;
                    vertex.y = float((xy >> 32) & 0xffffu) / 16.0f;
                    vertex.z = float((zf >> 4) & 0xffffffu);
                    vertex.fog = uint8_t(zf >> 36);
                    const bool drawing = ((zf >> 47) & 1u) == 0;
                    if (meshVerify) {
                        s_vuExpectedVertex = &vertex; s_vuExpectedDrawing = drawing;
                        writeRegisterPacked(2, loadLE64(st), loadLE64(st + 8));
                        if (rgba) writeRegisterPacked(1, loadLE64(rgba), loadLE64(rgba + 8));
                        writeRegisterPacked(4, xy, zf);
                        s_vuExpectedVertex = nullptr;
                    } else {
                        m_curR = vertex.r; m_curG = vertex.g; m_curB = vertex.b; m_curA = vertex.a;
                        m_curS = vertex.s; m_curT = vertex.t; m_curQ = vertex.q;
                        m_vtxQueue[m_vtxCount % kMaxVerts] = vertex;
                        vertexKick(drawing);
                    }
                };
                for (uint32_t loop = 0; loop < nloop; ++loop) {
                    const uint8_t *p = data + offset;
                    emit(p, p + 16, p + 32);
                    if (paired) emit(p + 48, nullptr, p + 80);
                    offset += nreg * 16u;
                }
                if (meshVerify) {
                    static thread_local uint64_t vertices = 0, pairedVertices = 0, tags = 0;
                    vertices += nloop * (paired ? 2u : 1u);
                    if (paired) pairedVertices += nloop * 2u;
                    if (++tags == 1 || tags % 50000u == 0)
                        std::fprintf(stderr, "[gs:mesh-decode] tags=%llu vertices=%llu paired=%llu mismatches=0\n",
                            (unsigned long long)tags, (unsigned long long)vertices, (unsigned long long)pairedVertices);
                }
                if (nativeBackend) geometry.flush();
                continue;
            }
            for (uint32_t loop = 0; loop < nloop; ++loop)
            {
                for (uint32_t r = 0; r < nreg; ++r)
                {
                    if (offset + 16 > sizeBytes)
                        return;
                    uint64_t lo = loadLE64(data + offset);
                    uint64_t hi = loadLE64(data + offset + 8);
                    offset += 16;
                    writeRegisterPacked(regs[r], lo, hi);
                }
            }
            if (nativeBackend) geometry.flush();
        }
        else if (flg == GIF_FMT_REGLIST)
        {
            for (uint32_t loop = 0; loop < nloop; ++loop)
            {
                for (uint32_t r = 0; r < nreg; ++r)
                {
                    if (offset + 8 > sizeBytes)
                        return;
                    writeRegisterUnlocked(regs[r], loadLE64(data + offset));
                    offset += 8;
                }
            }
            if ((nloop * nreg) & 1)
                offset += 8;
        }
        else if (flg == GIF_FMT_IMAGE)
        {
            const uint64_t wantBytes = static_cast<uint64_t>(nloop) * 16ull;
            const uint32_t availHere = sizeBytes - offset;
            const uint32_t take = static_cast<uint32_t>(std::min<uint64_t>(wantBytes, availHere));
#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
            {
                static std::atomic<uint32_t> s_img{0u};
                if (s_img.fetch_add(1u, std::memory_order_relaxed) < 64u)
                    std::fprintf(stderr, "[gs:pkt]   -> IMAGE tag nloop=%u eop=%u want=%llu inline=%u owed=%llu\n",
                                 nloop, (unsigned)((tagLo >> 15) & 1u), (unsigned long long)wantBytes, take,
                                 (unsigned long long)(wantBytes - take));
            }
#endif
            processImageData(data + offset, take);
            offset += take;
            if (wantBytes > take)
            {
                // Payload continues in the following DMA packet(s).
                s_pendingImageBytes = static_cast<uint32_t>(
                    std::min<uint64_t>(wantBytes - take, 0xFFFFFFFFull));
                break;
            }
        }
    }
}

bool GS::processNativePackedGIFPacket(const uint8_t *data, uint32_t sizeBytes)
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    if (!data || sizeBytes < 16u || !m_backend)
        return false;

    if (!validatePackedGifPacket(data, sizeBytes))
        return false;

    const bool processed = visitPackedGifPacket(data, sizeBytes, [&](const PackedGifPacketTag &tag)
                                                {
        m_curQ = 1.0f;

        recordGifTagDebugEventUnlocked(sizeBytes, tag.nloop, GIF_FMT_PACKED, tag.nreg);

        const bool pre = ((tag.lo >> 46u) & 1u) != 0u;
        if (pre)
            writeRegisterUnlocked(GS_REG_PRIM, (tag.lo >> 47u) & 0x7FFu);

        uint32_t offset = tag.payloadOffset;
        for (uint32_t loop = 0u; loop < tag.nloop; ++loop)
        {
            for (uint32_t r = 0u; r < tag.nreg; ++r)
            {
                const uint64_t lo = loadLE64(data + offset);
                const uint64_t hi = loadLE64(data + offset + 8u);
                offset += 16u;
                writeRegisterPacked(tag.regs[r], lo, hi);
            }
        }

        return true; });

    if (!processed)
        return false;

    ++m_nativePackedGIFPacketCount;
    return true;
}

// Procedural packets exercise complete strips, ADC gaps, near clipping and
// subsequent non-PRE continuation tags. No game assets or GPU context required.
int ps2xBenchmarkNativeStripPackets()
{
    GS gs;
    gs.setRasterBackend(std::make_unique<GSGlBackend>());
    std::vector<uint8_t> vram(4u*1024u*1024u);
    gs.init(vram.data(),uint32_t(vram.size()));gs.setDebugHistoryPaused(true);
    struct Packet {
        std::vector<uint8_t> bytes; uint32_t count;
        std::vector<GSVertex> vertices;std::vector<uint8_t> adc;
    };
    const bool tokens=std::getenv("PS2X_GS_STRIP_TOKEN_BENCH")!=nullptr;
    std::vector<Packet> packets;
    auto append=[&](std::vector<uint8_t>& out,uint64_t lo,uint64_t hi) {
        const size_t n=out.size();out.resize(n+16);std::memcpy(out.data()+n,&lo,8);std::memcpy(out.data()+n+8,&hi,8);
    };
    const uint32_t counts[]={3,7,24,36,64,128,256};
    size_t vertices=0;
    for (unsigned sample=0;sample<56;++sample) {
        Packet packet;packet.count=counts[sample%7];vertices+=packet.count+2u;
        const uint64_t prim=GS_PRIM_TRISTRIP | (1ull<<3) | (1ull<<4);
        append(packet.bytes,packet.count | (1ull<<46) | (prim<<47) | (3ull<<60),0x412ull);
        auto vertex=[&](unsigned i,bool continuation) {
            const float st[3]={float(i%7)*0.125f,float(i%11)*0.0625f,
                !continuation && sample%4==1 && i%13==7 ? -0.25f:0.5f};
            uint64_t lo=0,hi=0;std::memcpy(&lo,st,8);std::memcpy(&hi,st+2,4);append(packet.bytes,lo,hi);
            append(packet.bytes,(uint64_t(20u+i%200u)<<32)|(80u+sample),
                (uint64_t(128u)<<32)|(120u+i%100u));
            const uint64_t xy=uint64_t(16u*(i%32u+64u)) | (uint64_t(16u*(i/32u+48u))<<32);
            const bool adc=!continuation && (i<2u || (sample%4==2 && i%7==5));
            append(packet.bytes,xy,(uint64_t(100000u+i*17u)<<4) | (uint64_t(64u+i%128u)<<36) | (uint64_t(adc)<<47));
            if (!continuation) {
                GSVertex v{};v.x=float(i%32u+64u);v.y=float(i/32u+48u);v.z=100000u+i*17u;
                v.r=uint8_t(80u+sample);v.g=uint8_t(20u+i%200u);v.b=uint8_t(120u+i%100u);v.a=128;
                v.s=st[0];v.t=st[1];v.q=st[2];v.fog=uint8_t(64u+i%128u);
                packet.vertices.push_back(v);packet.adc.push_back(uint8_t(adc));
            }
        };
        for (unsigned i=0;i<packet.count;++i) vertex(i,false);
        append(packet.bytes,2ull|(1ull<<15)|(3ull<<60),0x412ull);
        vertex(packet.count,true);vertex(packet.count+1u,true);
        packets.push_back(std::move(packet));
    }
    extern uint64_t g_ps2xGeometryDigest;
    auto replay=[&] {
        gs.reset();gs.writeRegister(GS_REG_PRMODECONT,1);
        gs.writeRegister(GS_REG_TEX0_1, (1ull<<14)|(6ull<<26)|(6ull<<30));
        gs.writeRegister(GS_REG_SCISSOR_1,(639ull<<16)|(447ull<<48));
        gs.writeRegister(GS_REG_TEST_1,0);
        g_ps2xGeometryDigest=1469598103934665603ull;
        for (const auto &p:packets) {
            if (tokens) {
                const uint32_t bytes=16u+p.count*48u;
                ps2xGsSubmitVuMesh(gs,nullptr,loadLE64(p.bytes.data()),p.vertices.data(),p.adc.data(),
                                  p.count,p.bytes.data(),bytes);
                gs.processGIFPacket(p.bytes.data()+bytes,uint32_t(p.bytes.size())-bytes);
            } else gs.processGIFPacket(p.bytes.data(),uint32_t(p.bytes.size()));
        }
    };
    for (unsigned i=0;i<8;++i) replay();
    std::vector<double> samples;
    for (unsigned i=0;i<21;++i) {
        const auto begin=std::chrono::steady_clock::now();
        for (unsigned repeat=0;repeat<128;++repeat) replay();
        const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();
        samples.push_back(double(ns)/(128.0*vertices));
    }
    std::sort(samples.begin(),samples.end());
    std::fprintf(stderr,"[gs:%s-bench] packets=%zu vertices=%zu median_ns_per_vertex=%.2f min=%.2f max=%.2f\n",
        tokens?"token":"packet",packets.size(),vertices,samples[10],samples.front(),samples.back());
    if (std::getenv("PS2X_GS_GEOMETRY_DIGEST"))
        std::fprintf(stderr,"[gs:geometry-digest] %016llx\n",(unsigned long long)g_ps2xGeometryDigest);
    return 0;
}

void GS::uploadImageNative(uint64_t bitbltbuf,
                           uint64_t trxpos,
                           uint64_t trxreg,
                           uint64_t trxdir,
                           const uint8_t *data,
                           uint32_t sizeBytes)
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    uploadImageNativeUnlocked(bitbltbuf, trxpos, trxreg, trxdir, data, sizeBytes);
}

void GS::uploadImageNativeUnlocked(uint64_t bitbltbuf,
                                   uint64_t trxpos,
                                   uint64_t trxreg,
                                   uint64_t trxdir,
                                   const uint8_t *data,
                                   uint32_t sizeBytes)
{
    if (!data || sizeBytes == 0 || !m_backend)
        return;

#if defined(AGRESSIVE_LOGS) && AGRESSIVE_LOGS
    {
        static std::atomic<uint32_t> s_up{0u};
        if (s_up.fetch_add(1u, std::memory_order_relaxed) < 400u)
            std::fprintf(stderr, "[gs:upload] dbp=0x%x dbw=%u dpsm=0x%x dst=(%u,%u) size=%ux%u bytes=%u\n",
                         (unsigned)((bitbltbuf >> 32) & 0x3FFFu), (unsigned)((bitbltbuf >> 48) & 0x3Fu),
                         (unsigned)((bitbltbuf >> 56) & 0x3Fu), (unsigned)((trxpos >> 32) & 0x7FFu),
                         (unsigned)((trxpos >> 48) & 0x7FFu), (unsigned)(trxreg & 0xFFFu),
                         (unsigned)((trxreg >> 32) & 0xFFFu), sizeBytes);
    }
#endif
    writeRegisterUnlocked(GS_REG_BITBLTBUF, bitbltbuf);
    writeRegisterUnlocked(GS_REG_TRXPOS, trxpos);
    writeRegisterUnlocked(GS_REG_TRXREG, trxreg);
    writeRegisterUnlocked(GS_REG_TRXDIR, trxdir);
    processImageData(data, sizeBytes);
    ++m_nativeImageUploadCount;
}

bool GS::tryProcessNativeImageUploadPacket(const uint8_t *data, uint32_t sizeBytes)
{
    constexpr uint32_t kSetupRegisters = 4u;
    constexpr uint32_t kPackedAdPayloadBytes = kSetupRegisters * 16u;
    constexpr uint64_t kPackedAdDescriptor = 0x0Eull;

    if (!data || sizeBytes < 16u + kPackedAdPayloadBytes + 16u)
        return false;

    const uint64_t setupTagLo = loadLE64(data);
    const uint64_t setupTagHi = loadLE64(data + 8u);
    const uint32_t setupNloop = static_cast<uint32_t>(setupTagLo & 0x7FFFu);
    const uint8_t setupFlg = static_cast<uint8_t>((setupTagLo >> 58u) & 0x3u);
    uint32_t setupNreg = static_cast<uint32_t>((setupTagLo >> 60u) & 0xFu);
    if (setupNreg == 0u)
        setupNreg = 16u;

    if (setupNloop != kSetupRegisters ||
        setupFlg != GIF_FMT_PACKED ||
        setupNreg != 1u ||
        (setupTagHi & 0xFull) != kPackedAdDescriptor)
    {
        return false;
    }

    uint64_t regs[kSetupRegisters] = {};
    uint32_t offset = 16u;
    constexpr uint8_t expectedRegs[kSetupRegisters] = {
        GS_REG_BITBLTBUF,
        GS_REG_TRXPOS,
        GS_REG_TRXREG,
        GS_REG_TRXDIR,
    };

    for (uint32_t i = 0; i < kSetupRegisters; ++i)
    {
        regs[i] = loadLE64(data + offset);
        const uint64_t reg = loadLE64(data + offset + 8u);
        if ((reg & 0xFFu) != expectedRegs[i])
            return false;
        offset += 16u;
    }

    const uint32_t trxdirMode = static_cast<uint32_t>(regs[3] & 0x3ull);
    const uint32_t rrw = static_cast<uint32_t>(regs[2] & 0xFFFull);
    const uint32_t rrh = static_cast<uint32_t>((regs[2] >> 32u) & 0xFFFull);
    if (trxdirMode != 0u || rrw == 0u || rrh == 0u)
        return false;

    if (offset + 16u > sizeBytes)
        return false;

    const uint64_t imageTagLo = loadLE64(data + offset);
    const uint8_t imageFlg = static_cast<uint8_t>((imageTagLo >> 58u) & 0x3u);
    const uint32_t imageNloop = static_cast<uint32_t>(imageTagLo & 0x7FFFu);
    if (imageFlg != GIF_FMT_IMAGE || imageNloop == 0u)
        return false;

    offset += 16u;
    const uint64_t imageBytes64 = static_cast<uint64_t>(imageNloop) * 16ull;
    if (imageBytes64 > 0xFFFFFFFFull)
        return false;
    const uint32_t imageBytes = static_cast<uint32_t>(imageBytes64);
    if (offset + imageBytes != sizeBytes)
        return false;

    uploadImageNativeUnlocked(regs[0], regs[1], regs[2], regs[3], data + offset, imageBytes);
    return true;
}

void GS::writeRegisterPacked(uint8_t regDesc, uint64_t lo, uint64_t hi)
{
    switch (regDesc)
    {
    case 0x00:
        writeRegisterUnlocked(GS_REG_PRIM, lo & 0x7FF);
        break;
    case 0x01:
        m_curR = static_cast<uint8_t>(lo & 0xFF);
        m_curG = static_cast<uint8_t>((lo >> 32) & 0xFF);
        m_curB = static_cast<uint8_t>(hi & 0xFF);
        m_curA = static_cast<uint8_t>((hi >> 32) & 0xFF);
        break;
    case 0x02:
    {
        uint32_t sBits = static_cast<uint32_t>(lo & 0xFFFFFFFF);
        uint32_t tBits = static_cast<uint32_t>((lo >> 32) & 0xFFFFFFFF);
        uint32_t qBits = static_cast<uint32_t>(hi & 0xFFFFFFFF);
        std::memcpy(&m_curS, &sBits, 4);
        std::memcpy(&m_curT, &tBits, 4);
        std::memcpy(&m_curQ, &qBits, 4);
        if (m_curQ == 0.0f)
            m_curQ = 1.0f;
        break;
    }
    case 0x03:
        m_curU = static_cast<uint16_t>(lo & 0x3FFFu);
        m_curV = static_cast<uint16_t>((lo >> 32) & 0x3FFFu);
        break;
    case 0x04:
    {
        uint16_t x = static_cast<uint16_t>(lo & 0xFFFF);
        uint16_t y = static_cast<uint16_t>((lo >> 32) & 0xFFFF);
        uint32_t z = static_cast<uint32_t>((hi >> 4) & 0xFFFFFF);
        uint8_t f = static_cast<uint8_t>((hi >> 36) & 0xFF);
        bool adk = ((hi >> 47) & 1) != 0;
        PS2_IF_AGRESSIVE_LOGS({
            const uint32_t debugIndex = s_debugGsPackedVertexCount.fetch_add(1, std::memory_order_relaxed);
            if (debugIndex < 64u)
            {
                RUNTIME_LOG("[gs:packed-xyzf] idx=" << debugIndex
                                                    << " x=" << x
                                                    << " y=" << y
                                                    << " z=0x" << std::hex << z
                                                    << std::dec
                                                    << " fog=" << static_cast<uint32_t>(f)
                                                    << " kick=" << static_cast<uint32_t>(!adk ? 1u : 0u)
                                                    << " prim=" << static_cast<uint32_t>(m_prim.type)
                                                    << std::endl);
            }
        });
        GSVertex &vtx = m_vtxQueue[m_vtxCount % kMaxVerts];
        vtx.x = static_cast<float>(x) / 16.0f;
        vtx.y = static_cast<float>(y) / 16.0f;
        vtx.z = static_cast<float>(z);
        vtx.r = m_curR;
        vtx.g = m_curG;
        vtx.b = m_curB;
        vtx.a = m_curA;
        vtx.q = m_curQ;
        vtx.s = m_curS;
        vtx.t = m_curT;
        vtx.u = m_curU;
        vtx.v = m_curV;
        vtx.fog = f;
        vertexKick(!adk);
        break;
    }
    case 0x05:
    {
        uint16_t x = static_cast<uint16_t>(lo & 0xFFFF);
        uint16_t y = static_cast<uint16_t>((lo >> 32) & 0xFFFF);
        uint32_t z = static_cast<uint32_t>(hi & 0xFFFFFFFF);
        bool adk = ((hi >> 47) & 1) != 0;
        PS2_IF_AGRESSIVE_LOGS({
            const uint32_t debugIndex = s_debugGsPackedVertexCount.fetch_add(1, std::memory_order_relaxed);
            if (debugIndex < 64u)
            {
                RUNTIME_LOG("[gs:packed-xyz] idx=" << debugIndex
                                                   << " x=" << x
                                                   << " y=" << y
                                                   << " z=0x" << std::hex << z
                                                   << std::dec
                                                   << " kick=" << static_cast<uint32_t>(!adk ? 1u : 0u)
                                                   << " prim=" << static_cast<uint32_t>(m_prim.type)
                                                   << std::endl);
            }
        });
        GSVertex &vtx = m_vtxQueue[m_vtxCount % kMaxVerts];
        vtx.x = static_cast<float>(x) / 16.0f;
        vtx.y = static_cast<float>(y) / 16.0f;
        vtx.z = static_cast<float>(z);
        vtx.r = m_curR;
        vtx.g = m_curG;
        vtx.b = m_curB;
        vtx.a = m_curA;
        vtx.q = m_curQ;
        vtx.s = m_curS;
        vtx.t = m_curT;
        vtx.u = m_curU;
        vtx.v = m_curV;
        vtx.fog = m_curFog;
        vertexKick(!adk);
        break;
    }
    case 0x0A:
        m_curFog = static_cast<uint8_t>((hi >> 36) & 0xFF);
        break;
    case 0x0C:
    {
        PS2_IF_AGRESSIVE_LOGS({
            const uint32_t debugIndex = s_debugGsPackedVertexCount.fetch_add(1, std::memory_order_relaxed);
            if (debugIndex < 64u)
            {
                RUNTIME_LOG("[gs:packed-xyzf3] idx=" << debugIndex
                                                     << " x=" << static_cast<uint32_t>(lo & 0xFFFFu)
                                                     << " y=" << static_cast<uint32_t>((lo >> 32) & 0xFFFFu)
                                                     << " kick=0"
                                                     << " prim=" << static_cast<uint32_t>(m_prim.type)
                                                     << std::endl);
            }
        });
        GSVertex &vtx = m_vtxQueue[m_vtxCount % kMaxVerts];
        vtx.x = static_cast<float>(lo & 0xFFFF) / 16.0f;
        vtx.y = static_cast<float>((lo >> 32) & 0xFFFF) / 16.0f;
        vtx.z = static_cast<float>((hi >> 4) & 0xFFFFFF);
        vtx.r = m_curR;
        vtx.g = m_curG;
        vtx.b = m_curB;
        vtx.a = m_curA;
        vtx.q = m_curQ;
        vtx.s = m_curS;
        vtx.t = m_curT;
        vtx.u = m_curU;
        vtx.v = m_curV;
        vtx.fog = static_cast<uint8_t>((hi >> 36) & 0xFF);
        vertexKick(false);
        break;
    }
    case 0x0D:
    {
        PS2_IF_AGRESSIVE_LOGS({
            const uint32_t debugIndex = s_debugGsPackedVertexCount.fetch_add(1, std::memory_order_relaxed);
            if (debugIndex < 64u)
            {
                RUNTIME_LOG("[gs:packed-xyz3] idx=" << debugIndex
                                                    << " x=" << static_cast<uint32_t>(lo & 0xFFFFu)
                                                    << " y=" << static_cast<uint32_t>((lo >> 32) & 0xFFFFu)
                                                    << " kick=0"
                                                    << " prim=" << static_cast<uint32_t>(m_prim.type)
                                                    << std::endl);
            }
        });
        GSVertex &vtx = m_vtxQueue[m_vtxCount % kMaxVerts];
        vtx.x = static_cast<float>(lo & 0xFFFF) / 16.0f;
        vtx.y = static_cast<float>((lo >> 32) & 0xFFFF) / 16.0f;
        vtx.z = static_cast<float>(hi & 0xFFFFFFFF);
        vtx.r = m_curR;
        vtx.g = m_curG;
        vtx.b = m_curB;
        vtx.a = m_curA;
        vtx.q = m_curQ;
        vtx.s = m_curS;
        vtx.t = m_curT;
        vtx.u = m_curU;
        vtx.v = m_curV;
        vtx.fog = m_curFog;
        vertexKick(false);
        break;
    }
    case 0x0E:
    {
        uint8_t addr = static_cast<uint8_t>(hi & 0xFF);
        writeRegisterUnlocked(addr, lo);
        break;
    }
    case 0x0F:
        break;
    default:
        writeRegisterUnlocked(regDesc, lo);
        break;
    }
}

void GS::writeRegister(uint8_t regAddr, uint64_t value)
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    writeRegisterUnlocked(regAddr, value);
}

void GS::writeRegisterUnlocked(uint8_t regAddr, uint64_t value)
{
    PS2_IF_AGRESSIVE_LOGS({
        if (g_gsFrameDump.load(std::memory_order_relaxed) > 0)
        {
            g_gsFrameDump.fetch_sub(1, std::memory_order_relaxed);
            RUNTIME_LOG("[gs:dump-reg] reg=0x" << std::hex << static_cast<uint32_t>(regAddr)
                                               << " value=0x" << value << std::dec
                                               << " primCtxt=" << static_cast<uint32_t>(m_prim.ctxt)
                                               << std::endl);
        }
    });

    const bool interestingReg =
        regAddr == GS_REG_PRIM ||
        regAddr == GS_REG_RGBAQ ||
        regAddr == GS_REG_ST ||
        regAddr == GS_REG_UV ||
        regAddr == GS_REG_XYZ2 ||
        regAddr == GS_REG_XYZ3 ||
        regAddr == GS_REG_XYZF2 ||
        regAddr == GS_REG_XYZF3 ||
        regAddr == GS_REG_TEX0_1 ||
        regAddr == GS_REG_TEX0_2 ||
        regAddr == GS_REG_TEX2_1 ||
        regAddr == GS_REG_TEX2_2 ||
        regAddr == GS_REG_TEXCLUT ||
        regAddr == GS_REG_TEXA ||
        regAddr == GS_REG_XYOFFSET_1 ||
        regAddr == GS_REG_XYOFFSET_2 ||
        regAddr == GS_REG_SCISSOR_1 ||
        regAddr == GS_REG_SCISSOR_2 ||
        regAddr == GS_REG_FRAME_1 ||
        regAddr == GS_REG_FRAME_2 ||
        regAddr == GS_REG_ALPHA_1 ||
        regAddr == GS_REG_ALPHA_2 ||
        regAddr == GS_REG_TEST_1 ||
        regAddr == GS_REG_TEST_2 ||
        regAddr == GS_REG_BITBLTBUF ||
        regAddr == GS_REG_TRXPOS ||
        regAddr == GS_REG_TRXREG ||
        regAddr == GS_REG_TRXDIR;

    PS2_IF_AGRESSIVE_LOGS({
        if (interestingReg)
        {
            const uint32_t debugIndex = s_debugGsRegisterCount.fetch_add(1, std::memory_order_relaxed);
            if (debugIndex < 128u)
            {
                RUNTIME_LOG("[gs:reg] idx=" << debugIndex
                                            << " reg=0x" << std::hex << static_cast<uint32_t>(regAddr)
                                            << " value=0x" << value
                                            << std::dec
                                            << std::endl);
            }
        }
    });

    const bool isCopyRelevantReg =
        regAddr == GS_REG_PRIM ||
        regAddr == GS_REG_TEX0_2 ||
        regAddr == GS_REG_TEX1_2 ||
        regAddr == GS_REG_ALPHA_2 ||
        regAddr == GS_REG_TEST_2 ||
        regAddr == GS_REG_PABE ||
        regAddr == GS_REG_FRAME_2 ||
        regAddr == GS_REG_XYOFFSET_2 ||
        regAddr == GS_REG_SCISSOR_2;
    PS2_IF_AGRESSIVE_LOGS({
        if (isCopyRelevantReg &&
            s_debugCopyRegCount.fetch_add(1u, std::memory_order_relaxed) < 64u)
        {
            RUNTIME_LOG("[gs:copy-reg] reg=0x"
                        << std::hex << static_cast<uint32_t>(regAddr)
                        << " value=0x" << value
                        << std::dec
                        << " primCtxt=" << static_cast<uint32_t>(m_prim.ctxt)
                        << " ctx0fbp=" << m_ctx[0].frame.fbp
                        << " ctx1fbp=" << m_ctx[1].frame.fbp
                        << std::endl);
        }
    });

    switch (regAddr)
    {
    case GS_REG_PRIM:
    {
        m_primRegister = decodePrimRegister(value);
        if (m_prmodecont)
        {
            m_prim = m_primRegister;
        }
        else
        {
            // PRIM always selects the primitive topology. With AC=0, all
            // rendering attributes remain sourced from PRMODE.
            m_prim.type = m_primRegister.type;
        }
        m_vtxCount = 0;
        m_vtxIndex = 0;
        break;
    }
    case GS_REG_RGBAQ:
    {
        m_curR = static_cast<uint8_t>(value & 0xFF);
        m_curG = static_cast<uint8_t>((value >> 8) & 0xFF);
        m_curB = static_cast<uint8_t>((value >> 16) & 0xFF);
        m_curA = static_cast<uint8_t>((value >> 24) & 0xFF);
        uint32_t qBits = static_cast<uint32_t>((value >> 32) & 0xFFFFFFFF);
        std::memcpy(&m_curQ, &qBits, 4);
        if (m_curQ == 0.0f)
            m_curQ = 1.0f;
        break;
    }
    case GS_REG_ST:
    {
        uint32_t sBits = static_cast<uint32_t>(value & 0xFFFFFFFF);
        uint32_t tBits = static_cast<uint32_t>((value >> 32) & 0xFFFFFFFF);
        std::memcpy(&m_curS, &sBits, 4);
        std::memcpy(&m_curT, &tBits, 4);
        break;
    }
    case GS_REG_UV:
    {
        m_curU = static_cast<uint16_t>(value & 0x3FFFu);
        m_curV = static_cast<uint16_t>((value >> 16) & 0x3FFFu);
        break;
    }
    case GS_REG_XYZF2:
    case GS_REG_XYZF3:
    {
        GSVertex &vtx = m_vtxQueue[m_vtxCount % kMaxVerts];
        vtx.x = static_cast<float>(value & 0xFFFF) / 16.0f;
        vtx.y = static_cast<float>((value >> 16) & 0xFFFF) / 16.0f;
        vtx.z = static_cast<double>((value >> 32) & 0xFFFFFF);
        vtx.fog = static_cast<uint8_t>((value >> 56) & 0xFF);
        vtx.r = m_curR;
        vtx.g = m_curG;
        vtx.b = m_curB;
        vtx.a = m_curA;
        vtx.q = m_curQ;
        vtx.s = m_curS;
        vtx.t = m_curT;
        vtx.u = m_curU;
        vtx.v = m_curV;
        vertexKick(regAddr == GS_REG_XYZF2);
        break;
    }
    case GS_REG_XYZ2:
    case GS_REG_XYZ3:
    {
        GSVertex &vtx = m_vtxQueue[m_vtxCount % kMaxVerts];
        vtx.x = static_cast<float>(value & 0xFFFF) / 16.0f;
        vtx.y = static_cast<float>((value >> 16) & 0xFFFF) / 16.0f;
        vtx.z = static_cast<double>((value >> 32) & 0xFFFFFFFF);
        vtx.r = m_curR;
        vtx.g = m_curG;
        vtx.b = m_curB;
        vtx.a = m_curA;
        vtx.q = m_curQ;
        vtx.s = m_curS;
        vtx.t = m_curT;
        vtx.u = m_curU;
        vtx.v = m_curV;
        vtx.fog = m_curFog;
        vertexKick(regAddr == GS_REG_XYZ2);
        break;
    }
    case GS_REG_TEX0_1:
    case GS_REG_TEX0_2:
    {
        int ci = (regAddr == GS_REG_TEX0_2) ? 1 : 0;
        auto &t = m_ctx[ci].tex0;
        t.tbp0 = static_cast<uint32_t>(value & 0x3FFF);
        t.tbw = static_cast<uint8_t>((value >> 14) & 0x3F);
        t.psm = static_cast<uint8_t>((value >> 20) & 0x3F);
        t.tw = static_cast<uint8_t>((value >> 26) & 0xF);
        t.th = static_cast<uint8_t>((value >> 30) & 0xF);
        t.tcc = static_cast<uint8_t>((value >> 34) & 0x1);
        t.tfx = static_cast<uint8_t>((value >> 35) & 0x3);
        t.cbp = static_cast<uint32_t>((value >> 37) & 0x3FFF);
        t.cpsm = static_cast<uint8_t>((value >> 51) & 0xF);
        t.csm = static_cast<uint8_t>((value >> 55) & 0x1);
        t.csa = static_cast<uint8_t>((value >> 56) & 0x1F);
        t.cld = static_cast<uint8_t>((value >> 61) & 0x7);
        break;
    }
    case GS_REG_CLAMP_1:
    case GS_REG_CLAMP_2:
    {
        int ci = (regAddr == GS_REG_CLAMP_2) ? 1 : 0;
        m_ctx[ci].clamp = value;
        break;
    }
    case GS_REG_FOG:
        m_curFog = static_cast<uint8_t>((value >> 56) & 0xFF);
        break;
    case GS_REG_TEX1_1:
    case GS_REG_TEX1_2:
    {
        int ci = (regAddr == GS_REG_TEX1_2) ? 1 : 0;
        m_ctx[ci].tex1 = value;
        break;
    }
    case GS_REG_TEX2_1:
    case GS_REG_TEX2_2:
    {
        int ci = (regAddr == GS_REG_TEX2_2) ? 1 : 0;
        auto &t = m_ctx[ci].tex0;
        t.psm = static_cast<uint8_t>((value >> 20) & 0x3F);
        t.cbp = static_cast<uint32_t>((value >> 37) & 0x3FFF);
        t.cpsm = static_cast<uint8_t>((value >> 51) & 0xF);
        t.csm = static_cast<uint8_t>((value >> 55) & 0x1);
        t.csa = static_cast<uint8_t>((value >> 56) & 0x1F);
        t.cld = static_cast<uint8_t>((value >> 61) & 0x7);
        break;
    }
    case GS_REG_XYOFFSET_1:
    case GS_REG_XYOFFSET_2:
    {
        int ci = (regAddr == GS_REG_XYOFFSET_2) ? 1 : 0;
        m_ctx[ci].xyoffset.ofx = static_cast<uint16_t>(value & 0xFFFF);
        m_ctx[ci].xyoffset.ofy = static_cast<uint16_t>((value >> 32) & 0xFFFF);
        break;
    }
    case GS_REG_PRMODECONT:
    {
        m_prmodecont = (value & 1) != 0;
        const GSPrimType type = m_primRegister.type;
        m_prim = m_prmodecont ? m_primRegister : m_prmodeRegister;
        m_prim.type = type;
        break;
    }
    case GS_REG_PRMODE:
    {
        m_prmodeRegister = decodePrimRegister(value);
        if (!m_prmodecont)
        {
            const GSPrimType type = m_primRegister.type;
            m_prim = m_prmodeRegister;
            m_prim.type = type;
        }
        break;
    }
    case GS_REG_TEXCLUT:
        m_texclut.cbw = static_cast<uint8_t>(value & 0x3Fu);
        m_texclut.cou = static_cast<uint8_t>((value >> 6) & 0x3Fu);
        m_texclut.cov = static_cast<uint16_t>((value >> 12) & 0x3FFu);
        break;
    case GS_REG_SCISSOR_1:
    case GS_REG_SCISSOR_2:
    {
        int ci = (regAddr == GS_REG_SCISSOR_2) ? 1 : 0;
        m_ctx[ci].scissor.x0 = static_cast<uint16_t>(value & 0x7FF);
        m_ctx[ci].scissor.x1 = static_cast<uint16_t>((value >> 16) & 0x7FF);
        m_ctx[ci].scissor.y0 = static_cast<uint16_t>((value >> 32) & 0x7FF);
        m_ctx[ci].scissor.y1 = static_cast<uint16_t>((value >> 48) & 0x7FF);
        break;
    }
    case GS_REG_ALPHA_1:
    case GS_REG_ALPHA_2:
    {
        int ci = (regAddr == GS_REG_ALPHA_2) ? 1 : 0;
        m_ctx[ci].alpha = value;
        break;
    }
    case GS_REG_TEST_1:
    case GS_REG_TEST_2:
    {
        int ci = (regAddr == GS_REG_TEST_2) ? 1 : 0;
        m_ctx[ci].test = value;
        break;
    }
    case GS_REG_FRAME_1:
    case GS_REG_FRAME_2:
    {
        int ci = (regAddr == GS_REG_FRAME_2) ? 1 : 0;
        m_ctx[ci].frame.fbp = static_cast<uint32_t>(value & 0x1FF);
        m_ctx[ci].frame.fbw = static_cast<uint32_t>((value >> 16) & 0x3F);
        m_ctx[ci].frame.psm = static_cast<uint8_t>((value >> 24) & 0x3F);
        m_ctx[ci].frame.fbmsk = static_cast<uint32_t>((value >> 32) & 0xFFFFFFFF);
        break;
    }
    case GS_REG_ZBUF_1:
    case GS_REG_ZBUF_2:
    {
        int ci = (regAddr == GS_REG_ZBUF_2) ? 1 : 0;
        m_ctx[ci].zbuf.zbp = value & 0x1FF;
        m_ctx[ci].zbuf.psm = ((value >> 24) & 0xF) | 0x30;
        m_ctx[ci].zbuf.zmask = (value >> 32) & 1;
        break;
    }
    case GS_REG_FBA_1:
    case GS_REG_FBA_2:
    {
        int ci = (regAddr == GS_REG_FBA_2) ? 1 : 0;
        m_ctx[ci].fba = value;
        break;
    }
    case GS_REG_BITBLTBUF:
    {
        m_bitbltbuf.sbp = static_cast<uint32_t>(value & 0x3FFF);
        m_bitbltbuf.sbw = static_cast<uint8_t>((value >> 16) & 0x3F);
        m_bitbltbuf.spsm = static_cast<uint8_t>((value >> 24) & 0x3F);
        m_bitbltbuf.dbp = static_cast<uint32_t>((value >> 32) & 0x3FFF);
        m_bitbltbuf.dbw = static_cast<uint8_t>((value >> 48) & 0x3F);
        m_bitbltbuf.dpsm = static_cast<uint8_t>((value >> 56) & 0x3F);
        break;
    }
    case GS_REG_TRXPOS:
    {
        m_trxpos.ssax = static_cast<uint16_t>(value & 0x7FF);
        m_trxpos.ssay = static_cast<uint16_t>((value >> 16) & 0x7FF);
        m_trxpos.dsax = static_cast<uint16_t>((value >> 32) & 0x7FF);
        m_trxpos.dsay = static_cast<uint16_t>((value >> 48) & 0x7FF);
        m_trxpos.dir = static_cast<uint8_t>((value >> 59) & 0x3);
        break;
    }
    case GS_REG_TRXREG:
    {
        m_trxreg.rrw = static_cast<uint16_t>(value & 0xFFF);
        m_trxreg.rrh = static_cast<uint16_t>((value >> 32) & 0xFFF);
        break;
    }
    case GS_REG_TRXDIR:
    {
        m_trxdir = static_cast<uint32_t>(value & 0x3);

        if (m_backend)
        {
            GSTransferCommand command{};
            command.bitbltbuf = m_bitbltbuf;
            command.trxpos = m_trxpos;
            command.trxreg = m_trxreg;
            command.direction = m_trxdir;
            m_backend->BeginTransfer(command);
        }
        recordTransferDebugEventUnlocked();
        break;
    }
    case GS_REG_HWREG:
    {
        uint8_t buf[8];
        std::memcpy(buf, &value, 8);
        processImageData(buf, 8);
        break;
    }
    case GS_REG_PABE:
        m_pabe = (value & 1u) != 0u;
        break;
    case GS_REG_FOGCOL:
        m_fogR = static_cast<uint8_t>(value & 0xFFu);
        m_fogG = static_cast<uint8_t>((value >> 8) & 0xFFu);
        m_fogB = static_cast<uint8_t>((value >> 16) & 0xFFu);
        break;
    case GS_REG_TEXFLUSH:
        if (m_backend)
            m_backend->TextureFlush();
        break;
    case GS_REG_SCANMSK:
        m_scanmsk = value;
        break;
    case GS_REG_DIMX:
        m_dimx = value;
        break;
    case GS_REG_DTHE:
        m_dthe = value;
        break;
    case GS_REG_COLCLAMP:
        m_colclamp = value;
        break;
    case GS_REG_MIPTBP1_1:
    case GS_REG_MIPTBP1_2:
    {
        const int ci = (regAddr == GS_REG_MIPTBP1_2) ? 1 : 0;
        m_ctx[ci].miptbp1 = value;
        break;
    }
    case GS_REG_MIPTBP2_1:
    case GS_REG_MIPTBP2_2:
    {
        const int ci = (regAddr == GS_REG_MIPTBP2_2) ? 1 : 0;
        m_ctx[ci].miptbp2 = value;
        break;
    }
    case GS_REG_TEXA:
    {
        m_texa.ta0 = static_cast<uint8_t>(value & 0xFFu);
        m_texa.aem = ((value >> 15) & 0x1u) != 0u;
        m_texa.ta1 = static_cast<uint8_t>((value >> 32) & 0xFFu);
        PS2_IF_AGRESSIVE_LOGS({
            const uint32_t texaIndex = s_debugTexaWriteCount.fetch_add(1u, std::memory_order_relaxed);
            if (texaIndex < 24u)
            {
                RUNTIME_LOG("[gs:texa] idx=" << texaIndex
                                             << " value=0x" << std::hex << value
                                             << " ta0=0x" << ((value >> 0) & 0xFFu)
                                             << " aem=" << ((value >> 15) & 0x1u)
                                             << " ta1=0x" << ((value >> 32) & 0xFFu)
                                             << std::dec
                                             << std::endl);
            }
        });
        break;
    }
    case GS_REG_SIGNAL:
    {
        if (m_privRegs)
        {
            uint32_t id = static_cast<uint32_t>(value & 0xFFFFFFFF);
            uint32_t mask = static_cast<uint32_t>(value >> 32);
            uint32_t lo = static_cast<uint32_t>(m_privRegs->siglblid & 0xFFFFFFFF);
            lo = (lo & ~mask) | (id & mask);
            m_privRegs->siglblid = (m_privRegs->siglblid & 0xFFFFFFFF00000000ULL) | lo;
            m_privRegs->csr.fetch_or(0x1);
        }
        break;
    }
    case GS_REG_FINISH:
    {
        // Counted and timed for [perf] (finish=N/s wait=X%): this Sync blocks the
        // game thread until the raster worker has drawn everything queued, which
        // serialises game and raster if the game writes FINISH every frame.
        // PS2X_GS_FINISH_NOSYNC=1 skips the wait (A/B): VRAM readers (readback,
        // local->host transfers, presentation) already wait on their own ticket.
        static const bool s_finishNoSync = []
        {
            const char *value = std::getenv("PS2X_GS_FINISH_NOSYNC");
            return value != nullptr && value[0] == '1';
        }();
        extern std::atomic<uint64_t> g_ps2xFinishCount;
        extern std::atomic<uint64_t> g_ps2xFinishWaitNs;
        g_ps2xFinishCount.fetch_add(1u, std::memory_order_relaxed);
        if (m_backend && !s_finishNoSync)
        {
            const auto finishT0 = std::chrono::steady_clock::now();
            m_backend->Flush();
            m_backend->Sync(GSSyncReason::Finish);
            g_ps2xFinishWaitNs.fetch_add(
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                          std::chrono::steady_clock::now() - finishT0)
                                          .count()),
                std::memory_order_relaxed);
        }
        if (m_privRegs)
            m_privRegs->csr.fetch_or(0x2);
        break;
    }
    case GS_REG_LABEL:
    {
        if (m_privRegs)
        {
            uint32_t id = static_cast<uint32_t>(value & 0xFFFFFFFF);
            uint32_t mask = static_cast<uint32_t>(value >> 32);
            uint32_t hi = static_cast<uint32_t>(m_privRegs->siglblid >> 32);
            hi = (hi & ~mask) | (id & mask);
            m_privRegs->siglblid = (static_cast<uint64_t>(hi) << 32) | (m_privRegs->siglblid & 0xFFFFFFFF);
        }
        break;
    }
    case 0x59:
        if (m_privRegs)
            m_privRegs->dispfb1 = value;
        break;
    case 0x5a:
        if (m_privRegs)
            m_privRegs->display1 = value;
        break;
    case 0x5b:
        if (m_privRegs)
            m_privRegs->dispfb2 = value;
        break;
    case 0x5c:
        if (m_privRegs)
            m_privRegs->display2 = value;
        break;
    case 0x5f:
        if (m_privRegs)
            m_privRegs->bgcolor = value;
        break;
    default:
        break;
    }

    recordRegisterDebugEventUnlocked(regAddr, value);
}

void GS::vertexKick(bool drawing)
{
    if (s_vuExpectedVertex) {
        const GSVertex &a = m_vtxQueue[m_vtxCount % kMaxVerts], &b = *s_vuExpectedVertex;
        // Compare values bitwise (including NaN payloads and signed zeros),
        // but never compare structure padding.
        auto sameFloat = [](float x, float y) { return std::memcmp(&x, &y, sizeof(float)) == 0; };
        if (!sameFloat(a.x,b.x) || !sameFloat(a.y,b.y) || !sameFloat(a.z,b.z) ||
            !sameFloat(a.s,b.s) || !sameFloat(a.t,b.t) || !sameFloat(a.q,b.q) ||
            a.r!=b.r || a.g!=b.g || a.b!=b.b || a.a!=b.a || a.u!=b.u || a.v!=b.v || a.fog!=b.fog ||
            drawing != s_vuExpectedDrawing) {
            std::fprintf(stderr, "[gs:direct-mesh] decoded vertex mismatch\n"); std::abort();
        }
    }
    ++m_vtxCount;
    ++m_vtxIndex;

    PS2_IF_AGRESSIVE_LOGS({
        const uint32_t debugIndex = s_debugGsVertexKickCount.fetch_add(1, std::memory_order_relaxed);
        if (debugIndex < 96u || g_gsFrameDump.load(std::memory_order_relaxed) > 0)
        {
            RUNTIME_LOG("[gs:kick] idx=" << debugIndex
                                         << " drawing=" << static_cast<uint32_t>(drawing ? 1u : 0u)
                                         << " prim=" << static_cast<uint32_t>(m_prim.type)
                                         << " vtxCount=" << m_vtxCount
                                         << std::endl);
        }
    });

    int needed = 0;
    switch (m_prim.type)
    {
    case GS_PRIM_POINT:
        needed = 1;
        break;
    case GS_PRIM_LINE:
        needed = 2;
        break;
    case GS_PRIM_LINESTRIP:
        needed = 2;
        break;
    case GS_PRIM_TRIANGLE:
        needed = 3;
        break;
    case GS_PRIM_TRISTRIP:
        needed = 3;
        break;
    case GS_PRIM_TRIFAN:
        needed = 3;
        break;
    case GS_PRIM_SPRITE:
        needed = 2;
        break;
    default:
        return;
    }

    if (m_vtxCount < needed)
        return;

    // Diagnostic only: attribute oversized primitives to the VU1 program that
    // kicked them. This used to run for every triangle in normal play, including
    // four min/max reductions, aspect checks, wall-clock reads and periodic
    // string formatting. Keep it available without charging the shipping path.

    if (s_bigPrimCensus && drawing && needed == 3)
    {
        float bx0 = 1e30f, bx1 = -1e30f, by0 = 1e30f, by1 = -1e30f;
        for (int k = 0; k < 3; ++k)
        {
            const GSVertex &bv = m_vtxQueue[k];
            bx0 = std::min(bx0, bv.x);
            bx1 = std::max(bx1, bv.x);
            by0 = std::min(by0, bv.y);
            by1 = std::max(by1, bv.y);
        }
        const float bw = bx1 - bx0, bh = by1 - by0;
        if (bw > 256.0f || bh > 256.0f)
        {
            struct BigByPc { uint32_t startPc; uint64_t count; uint64_t area; uint64_t sliverCount; uint64_t sliverArea; };
            static BigByPc s_big[8]{};
            static auto s_bigT = std::chrono::steady_clock::now();
            const uint32_t pc = g_ps2xVu1Kick.packet ? g_ps2xVu1Kick.startPc : 0xFFFFFFFFu;
            const uint64_t area = static_cast<uint64_t>(bw) * static_cast<uint64_t>(bh);
            for (int i = 0; i < 8; ++i)
            {
                if (s_big[i].count == 0u || s_big[i].startPc == pc)
                {
                    s_big[i].startPc = pc;
                    ++s_big[i].count;
                    s_big[i].area += area;
                    // Aspect ratio separates genuine close-up geometry from the
                    // stretched spikes: a real character triangle is roughly
                    // compact, a smeared one is many times longer than it is wide.
                    const float longSide2 = std::max(bw, bh);
                    const float shortSide2 = std::min(bw, bh);
                    if (shortSide2 * 8.0f < longSide2)
                    {
                        ++s_big[i].sliverCount;
                        s_big[i].sliverArea += area;
                    }
                    break;
                }
            }
            const auto nowBig = std::chrono::steady_clock::now();
            if (nowBig - s_bigT >= std::chrono::seconds(5))
            {
                s_bigT = nowBig;
                std::string line = "[bigprim] 5s by VU1 program:";
                char part[128];
                for (int i = 0; i < 8; ++i)
                {
                    if (s_big[i].count == 0u)
                        continue;
                    std::snprintf(part, sizeof(part), " start=0x%x n=%llu Mpx=%.0f sliverN=%llu sliverMpx=%.0f;", s_big[i].startPc,
                                  static_cast<unsigned long long>(s_big[i].count),
                                  static_cast<double>(s_big[i].area) / 1.0e6,
                                  static_cast<unsigned long long>(s_big[i].sliverCount),
                                  static_cast<double>(s_big[i].sliverArea) / 1.0e6);
                    line += part;
                }
                std::fprintf(stderr, "%s\n", line.c_str());
                for (int i = 0; i < 8; ++i)
                    s_big[i] = {};
            }
        }
    }

    // Legacy stretch workaround, diagnostic opt-in only (PS2X_SLIVER_CULL=1).
    // Neither an entry PC nor a screen-space aspect ratio identifies corrupt
    // character geometry: these paths also emit valid roofs and sky polygons.
    // Same-frame CPU/GPU comparisons confirmed this filter cut holes in both.
    // Normal rendering preserves the game's ADC/drawing-kick decisions.
    const uint32_t kickStart = g_ps2xVu1Kick.packet ? g_ps2xVu1Kick.startPc : 0xFFFFFFFFu;
    const bool skinnedMeshKick = kickStart == 0x3060u || kickStart == 0x19E8u ||
                                 kickStart == 0x1868u || kickStart == 0xBF8u;
    if (s_sliverCull && drawing && needed == 3 && skinnedMeshKick)
    {
        float lo0 = 1e30f, hi0 = -1e30f, lo1 = 1e30f, hi1 = -1e30f;
        for (int k = 0; k < 3; ++k)
        {
            const GSVertex &v = m_vtxQueue[k];
            lo0 = std::min(lo0, v.x);
            hi0 = std::max(hi0, v.x);
            lo1 = std::min(lo1, v.y);
            hi1 = std::max(hi1, v.y);
        }
        const float w = hi0 - lo0, h = hi1 - lo1;
        const float longSide = std::max(w, h), shortSide = std::min(w, h);
        if (longSide > 300.0f && shortSide * 8.0f < longSide)
        {
            if (s_sliverCullReference) {
                if (auto *gl = dynamic_cast<GSGlBackend *>(m_backend.get())) {
                    if (s_nativeGeometry && s_nativeGeometry->owner == this) s_nativeGeometry->flush();
                    ps2xGlSubmitNearCullReference(*gl, buildDrawBatch(3));
                }
            }
            static std::atomic<uint64_t> s_culled{0u};
            const uint64_t n = s_culled.fetch_add(1u, std::memory_order_relaxed) + 1u;
            if (n == 1u || (n % 50000u) == 0u)
                std::fprintf(stderr, "[sliver] culled %llu (long=%.0f short=%.0f)\n",
                             static_cast<unsigned long long>(n), longSide, shortSide);
            drawing = false;
        }
    }

    // Near-plane guard: a perspective-textured triangle whose vertices straddle
    // the camera plane (some Q > 0, some Q <= 0) is a vertex behind the camera
    // projected through a non-clipping VU1 mesh program; drawn, it smears across
    // the screen ('limbs stretching to the camera'). Drop it. PS2X_NEAR_CULL=0
    // turns the guard off.

    if (s_nearCull && drawing && needed == 3 && m_prim.tme && !m_prim.fst)
    {
        int positive = 0;
        for (int k = 0; k < 3; ++k)
            positive += m_vtxQueue[k].q > 0.0f ? 1 : 0;
        if (positive != 3)
        {
            if (s_nearCullReference) {
                if (auto *gl = dynamic_cast<GSGlBackend *>(m_backend.get())) {
                    if (s_nativeGeometry && s_nativeGeometry->owner == this) s_nativeGeometry->flush();
                    ps2xGlSubmitNearCullReference(*gl, buildDrawBatch(3));
                    static uint64_t restored = 0;
                    if (++restored == 1u || restored % 100000u == 0u)
                        std::fprintf(stderr,"[gs:near-reference] restored=%llu start=%x positive=%d\n",restored,kickStart,positive);
                }
            }
            drawing = false;
        }
    }

    // Diagnostic trigger for the captured Congar fur corruption: save the
    // complete input of the VU run that emitted this saturated-depth vertex.

    if(captureStretch && drawing && needed==3 && g_ps2xVu1Kick.startPc==0x20 && m_ctx[m_prim.ctxt?1:0].tex0.tbp0==0x2ffe) {
        for(unsigned i=0;i<3;++i)if(m_vtxQueue[i].z>=16776192.0f && m_vtxQueue[i].q>0 && m_vtxQueue[i].q<0.000001f) {
            ps2xSaveStretchCheckpoint();break;
        }
    }

    // One requested frame: attribute visible triangles to their source VU
    // packet. Diagnostic only; records surviving geometry without changing it.



    const uint64_t traceLoop=traceFrame==~0ull?0:g_ps2xWotmCompletedFrames.load(std::memory_order_relaxed);
    bool traceTriangle=false;
    if(traceFrame!=~0ull && drawing && needed==3 && traceLoop>=traceFrame && traceLoop<=traceEnd) {
        float x0=m_vtxQueue[0].x,x1=x0,y0=m_vtxQueue[0].y,y1=y0;
        for(unsigned i=1;i<3;++i){x0=std::min(x0,m_vtxQueue[i].x);x1=std::max(x1,m_vtxQueue[i].x);y0=std::min(y0,m_vtxQueue[i].y);y1=std::max(y1,m_vtxQueue[i].y);}
        traceTriangle=std::max(x1-x0,y1-y0)>=traceSpan;
    }
    if(traceTriangle) {
        const auto &ctx=m_ctx[m_prim.ctxt?1:0];
        char line[768];
        int n=std::snprintf(line,sizeof(line),"[geometry:triangle] loop=%llu kick=%llu start=%x pc=%x src=%x tex=%x clut=%x prim=%u",traceLoop,g_ps2xVu1Kick.serial,g_ps2xVu1Kick.startPc,g_ps2xVu1Kick.pc,g_ps2xVu1Kick.src,ctx.tex0.tbp0,ctx.tex0.cbp,unsigned(m_prim.type));
        for(unsigned k=0;k<3 && n>0 && n<int(sizeof(line))-100;++k) {
            const auto &v=m_vtxQueue[k];
            n+=std::snprintf(line+n,sizeof(line)-size_t(n)," v%u=%.5g,%.5g,%.8g,%.8g",k,v.x-float(ctx.xyoffset.ofx)/16.0f,v.y-float(ctx.xyoffset.ofy)/16.0f,v.z,v.q);
        }
        std::fprintf(stderr,"%s\n",line);
    }

    // PS2X_BIGTRI=1: log primitives whose screen bounding box is absurd (the
    // user's 'monster stretches when close to the camera' spikes). Tells apart
    // garbage vertices from VU1 (near-plane / clip-flag / ADC problem) from a
    // rasterizer mis-handling of in-range coordinates.

    // Do not consume the finite diagnostic samples on menu borders when the
    // requested observation is a late-fight intermittent geometry artifact.

    bool diagnosticVisible = false;
    if (s_bigTri && drawing && needed >= 2 && g_ps2xWotmCompletedFrames.load(std::memory_order_relaxed) >= s_geometryStartFrame)
    {
        const auto &ctx = m_ctx[m_prim.ctxt ? 1 : 0];
        float x0=1e30f,x1=-1e30f,y0=1e30f,y1=-1e30f;
        for(int k=0;k<needed;++k){x0=std::min(x0,m_vtxQueue[k].x);x1=std::max(x1,m_vtxQueue[k].x);y0=std::min(y0,m_vtxQueue[k].y);y1=std::max(y1,m_vtxQueue[k].y);}
        const float ox=float(ctx.xyoffset.ofx>>4),oy=float(ctx.xyoffset.ofy>>4);
        diagnosticVisible=x1-ox>=ctx.scissor.x0 && x0-ox<=ctx.scissor.x1 && y1-oy>=ctx.scissor.y0 && y0-oy<=ctx.scissor.y1;
    }
    if (diagnosticVisible)
    {
        const GSContext &bctx = m_ctx[m_prim.ctxt ? 1 : 0];
        float minX = 1e30f, maxX = -1e30f, minY = 1e30f, maxY = -1e30f;
        for (int k = 0; k < needed; ++k)
        {
            const GSVertex &v = m_vtxQueue[k];
            minX = std::min(minX, v.x);
            maxX = std::max(maxX, v.x);
            minY = std::min(minY, v.y);
            maxY = std::max(maxY, v.y);
        }
        // Huge primitives are mostly legit (ground/sky extending off-screen), so
        // they are only counted. A drawn primitive with a vertex whose Q (1/w) is
        // <= 0 has a vertex behind the camera: the VU1 clip test should have set
        // its ADC (no-draw) bit. Those are logged, one write per primitive (so
        // PowerShell's per-write line splitting cannot break the line).
        static uint64_t s_bigCount = 0u, s_negCount = 0u, s_negLogged = 0u, s_primCount = 0u;
        static auto s_bigT0 = std::chrono::steady_clock::now();
        ++s_primCount;
        bool negQ = false;
        for (int k = 0; k < needed; ++k)
            negQ |= !(m_vtxQueue[k].q > 0.0f);
        if (drawing && (maxX - minX > 1024.0f || maxY - minY > 1024.0f))
        {
            ++s_bigCount;
            // Attribute huge prims to the VU1 kick PC (256-byte buckets) and log
            // a sample, so 'limbs stretching to the camera' can be tied to the
            // near-plane clipper (kicks ~0x2500-0x2800) or the plain mesh path.
            static std::map<uint32_t, uint64_t> s_bigByPc;
            static uint32_t s_bigLogged = 0u;
            const uint32_t bucket = g_ps2xVu1Kick.packet ? (g_ps2xVu1Kick.pc & ~0xFFu) : 0xFFFFFFFFu;
            ++s_bigByPc[bucket];
            if (s_bigLogged < 200u && g_ps2xVu1Kick.packet && bucket != 0xF00u)
            {
                ++s_bigLogged;
                char big[640];
                int blen = std::snprintf(big, sizeof(big), "[bigtri] HUGE pc=0x%x start=0x%x src=0x%x prim=%u tme=%u fst=%u negq=%u w=%.0f h=%.0f",
                                         g_ps2xVu1Kick.pc, g_ps2xVu1Kick.startPc, g_ps2xVu1Kick.src,
                                         static_cast<unsigned>(m_prim.type), m_prim.tme ? 1u : 0u, m_prim.fst ? 1u : 0u,
                                         negQ ? 1u : 0u, maxX - minX, maxY - minY);
                for (int k = 0; k < needed && blen > 0 && blen < static_cast<int>(sizeof(big)) - 70; ++k)
                {
                    const GSVertex &v = m_vtxQueue[k];
                    blen += std::snprintf(big + blen, sizeof(big) - static_cast<size_t>(blen),
                                          " v%d=(%.1f,%.1f z=%.0f q=%.4g)", k, v.x, v.y, v.z, v.q);
                }
                std::fprintf(stderr, "%s\n", big);
                // Clamped-vertex spikes from the 0xbf8 mesh program (CLIP +
                // FCAND clipping): dump the VU1 flight recorder on the first few.
                static uint32_t s_hugeFlightDumps = 0u;
                if (g_ps2xVu1Kick.startPc == 0xBF8u && s_hugeFlightDumps < 3u)
                {
                    ++s_hugeFlightDumps;
                    ps2xVu1FlightDump(big);
                }
            }
            static auto s_bucketT0 = std::chrono::steady_clock::now();
            if (std::chrono::steady_clock::now() - s_bucketT0 >= std::chrono::seconds(5))
            {
                s_bucketT0 = std::chrono::steady_clock::now();
                std::string line = "[bigtri] huge by kick pc:";
                char part[48];
                for (const auto &[pcBucket, n] : s_bigByPc)
                {
                    std::snprintf(part, sizeof(part), " %x:%llu", pcBucket, static_cast<unsigned long long>(n));
                    line += part;
                }
                std::fprintf(stderr, "%s\n", line.c_str());
                s_bigByPc.clear();
            }
        }
        // The actual artefact: long needle-thin slivers ("limbs stretching to
        // the camera"). Every earlier test keyed on size, which legitimate
        // ground/sky geometry also trips; shape does not. Attribute them to the
        // VU1 program that kicked them.
        if (drawing && needed == 3)
        {
            const float w = maxX - minX, h = maxY - minY;
            const float longSide = std::max(w, h), shortSide = std::min(w, h);
            if (longSide > 300.0f && shortSide < 40.0f)
            {
                static uint64_t s_slivers = 0u;
                static std::map<uint32_t, uint64_t> s_sliverByPc;
                static auto s_slivT0 = std::chrono::steady_clock::now();
                ++s_slivers;
                ++s_sliverByPc[g_ps2xVu1Kick.packet ? g_ps2xVu1Kick.startPc : 0xFFFFFFFFu];
                static uint32_t s_sliverLogged = 0u;
                if (s_sliverLogged < 40u)
                {
                    ++s_sliverLogged;
                    char sv[640];
                    int slen = std::snprintf(sv, sizeof(sv),
                                             "[sliver] #%llu start=0x%x pc=0x%x src=0x%x tme=%u fst=%u negq=%u long=%.0f short=%.0f",
                                             static_cast<unsigned long long>(s_slivers), g_ps2xVu1Kick.startPc,
                                             g_ps2xVu1Kick.pc, g_ps2xVu1Kick.src, m_prim.tme ? 1u : 0u,
                                             m_prim.fst ? 1u : 0u, negQ ? 1u : 0u, longSide, shortSide);
                    for (int k = 0; k < 3 && slen > 0 && slen < static_cast<int>(sizeof(sv)) - 80; ++k)
                    {
                        const GSVertex &v = m_vtxQueue[k];
                        slen += std::snprintf(sv + slen, sizeof(sv) - static_cast<size_t>(slen),
                                              " v%d=(%.1f,%.1f z=%.0f q=%.4g)", k, v.x, v.y, v.z, v.q);
                    }
                    std::fprintf(stderr, "%s\n", sv);
                }
                if (std::chrono::steady_clock::now() - s_slivT0 >= std::chrono::seconds(5))
                {
                    s_slivT0 = std::chrono::steady_clock::now();
                    std::string line = "[sliver] last 5 s by start pc:";
                    char part[48];
                    for (const auto &[startPc, n] : s_sliverByPc)
                    {
                        std::snprintf(part, sizeof(part), " %x:%llu", startPc, static_cast<unsigned long long>(n));
                        line += part;
                    }
                    std::fprintf(stderr, "%s (total %llu)\n", line.c_str(),
                                 static_cast<unsigned long long>(s_slivers));
                    s_sliverByPc.clear();
                }
            }
        }

        if (drawing && negQ)
        {
            ++s_negCount;
            if (s_negLogged < 120u)
            {
                ++s_negLogged;
                char line[768];
                int len = std::snprintf(line, sizeof(line),
                                        "[bigtri] NEGQ #%llu prim=%u tme=%u fst=%u ctxt=%u ofs=(%u,%u) sc=(%u,%u)-(%u,%u) fbp=%u w=%.0f h=%.0f",
                                        static_cast<unsigned long long>(s_negCount), static_cast<unsigned>(m_prim.type),
                                        m_prim.tme ? 1u : 0u, m_prim.fst ? 1u : 0u, m_prim.ctxt ? 1u : 0u,
                                        bctx.xyoffset.ofx >> 4, bctx.xyoffset.ofy >> 4, bctx.scissor.x0, bctx.scissor.y0,
                                        bctx.scissor.x1, bctx.scissor.y1, bctx.frame.fbp, maxX - minX, maxY - minY);
                for (int k = 0; k < needed && len > 0 && len < static_cast<int>(sizeof(line)) - 80; ++k)
                {
                    const GSVertex &v = m_vtxQueue[k];
                    len += std::snprintf(line + len, sizeof(line) - static_cast<size_t>(len),
                                         " v%d=(%.1f,%.1f z=%.0f q=%.4g s=%.4g t=%.4g)", k, v.x, v.y, v.z, v.q, v.s, v.t);
                }
                if (g_ps2xVu1Kick.packet && len > 0 && len < static_cast<int>(sizeof(line)) - 120)
                    len += std::snprintf(line + len, sizeof(line) - static_cast<size_t>(len),
                                         " vu1[start=0x%x pc=0x%x src=0x%x top=0x%x itop=0x%x kick=%llu bytes=%u]",
                                         g_ps2xVu1Kick.startPc, g_ps2xVu1Kick.pc, g_ps2xVu1Kick.src,
                                         g_ps2xVu1Kick.top, g_ps2xVu1Kick.itop,
                                         static_cast<unsigned long long>(g_ps2xVu1Kick.serial), g_ps2xVu1Kick.bytes);
                std::fprintf(stderr, "%s\n", line);
                static uint32_t s_flightDumps = 0u;
                if (s_flightDumps < 3u)
                {
                    ++s_flightDumps;
                    ps2xVu1FlightDump(line);
                }
                // Dump the first few offending VU1 packets (hex, 16 B per line).
                static uint64_t s_lastDumpedKick = 0u;
                static uint32_t s_packetsDumped = 0u;
                if (g_ps2xVu1Kick.packet && s_packetsDumped < 6u && g_ps2xVu1Kick.serial != s_lastDumpedKick)
                {
                    s_lastDumpedKick = g_ps2xVu1Kick.serial;
                    ++s_packetsDumped;
                    if (std::FILE *pf = std::fopen("bigtri_packets.txt", s_packetsDumped == 1u ? "w" : "a"))
                    {
                        std::fprintf(pf, "=== kick %llu start=0x%x pc=0x%x src=0x%x top=0x%x itop=0x%x bytes=%u\n",
                                     static_cast<unsigned long long>(g_ps2xVu1Kick.serial), g_ps2xVu1Kick.startPc,
                                     g_ps2xVu1Kick.pc, g_ps2xVu1Kick.src, g_ps2xVu1Kick.top, g_ps2xVu1Kick.itop,
                                     g_ps2xVu1Kick.bytes);
                        for (uint32_t off = 0; off + 16u <= g_ps2xVu1Kick.bytes; off += 16u)
                        {
                            const uint8_t *qw = g_ps2xVu1Kick.packet + off;
                            uint64_t lo = 0, hi = 0;
                            std::memcpy(&lo, qw, 8);
                            std::memcpy(&hi, qw + 8, 8);
                            std::fprintf(pf, "%04x %016llx %016llx\n", off / 16u, static_cast<unsigned long long>(hi),
                                         static_cast<unsigned long long>(lo));
                        }
                        std::fclose(pf);
                    }
                }
            }
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - s_bigT0 >= std::chrono::seconds(5))
        {
            s_bigT0 = now;
            std::fprintf(stderr, "[bigtri] last 5 s: %llu huge, %llu drawn with Q<=0, of %llu primitives\n",
                         static_cast<unsigned long long>(s_bigCount), static_cast<unsigned long long>(s_negCount),
                         static_cast<unsigned long long>(s_primCount));
            s_primCount = 0u;
            s_bigCount = 0u;
            s_negCount = 0u;
        }
    }

    if (drawing && m_backend)
    {
        NativeGeometryBatch *geometry = s_nativeGeometry;
        if (geometry && geometry->owner == this && needed == 3) {
            if (!geometry->valid) {
                geometry->prototype = buildDrawBatch(needed);
                geometry->valid = true;
                updatePreferredDisplaySourceForDraw(geometry->prototype);
                if (!geometry->prototype.state.context.zbuf.zmask)
                    noteDepthPage(geometry->prototype.state.context.zbuf.zbp);
            }
            if (s_nativeGeometryVerify) {
                const GSPrimitiveBatch expected = buildDrawBatch(needed);
                if (std::memcmp(&expected.state, &geometry->prototype.state, sizeof(GSDrawState)) != 0) {
                    std::fprintf(stderr, "[gs:native-geometry] state mismatch\n");
                    std::abort();
                }
            }
            // The tag already owns this thread's buffer; avoid repeating TLS lookup.
            geometry->vertices.insert(geometry->vertices.end(), m_vtxQueue, m_vtxQueue + 3);
            // Bound temporary storage even for unusually large tags.
            if (geometry->vertices.size() >= 768u) geometry->flush();
        } else {
            GSPrimitiveBatch batch = buildDrawBatch(needed);
            updatePreferredDisplaySourceForDraw(batch);
            if (!batch.state.context.zbuf.zmask)
                noteDepthPage(batch.state.context.zbuf.zbp);
            m_backend->Submit(batch);
        }
        ++g_primsSinceLastPresent;
        recordDrawDebugEventUnlocked(needed);
    }

    switch (m_prim.type)
    {
    case GS_PRIM_LINE:
    case GS_PRIM_TRIANGLE:
    case GS_PRIM_SPRITE:
    case GS_PRIM_POINT:
        m_vtxCount = 0;
        break;
    case GS_PRIM_LINESTRIP:
        m_vtxQueue[0] = m_vtxQueue[1];
        m_vtxCount = 1;
        break;
    case GS_PRIM_TRISTRIP:
        m_vtxQueue[0] = m_vtxQueue[1];
        m_vtxQueue[1] = m_vtxQueue[2];
        m_vtxCount = 2;
        break;
    case GS_PRIM_TRIFAN:
        m_vtxQueue[1] = m_vtxQueue[2];
        m_vtxCount = 2;
        break;
    default:
        m_vtxCount = 0;
        break;
    }
}

void GS::processImageData(const uint8_t *data, uint32_t sizeBytes)
{
    if (m_backend)
        m_backend->UploadImage(data, sizeBytes);
}


bool GS::clearFramebufferContext(uint32_t contextIndex, uint32_t rgba)
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    return m_backend && m_backend->ClearFramebuffer(m_ctx[(contextIndex != 0u) ? 1 : 0], rgba);
}

bool GS::clearActiveFramebuffer(uint32_t rgba)
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    return m_backend && m_backend->ClearFramebuffer(activeContext(), rgba);
}

uint32_t GS::consumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes)
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    return m_backend ? m_backend->ConsumeLocalToHostBytes(dst, maxBytes) : 0u;
}

void GS::setRasterBackend(std::unique_ptr<GSRasterBackend> backend)
{
    if (!backend)
        backend = std::make_unique<GSCpuBackend>();

    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    std::lock_guard<std::mutex> backendLock(m_backendLifetimeMutex);
    if (m_backend)
    {
        m_backend->Flush();
        m_backend->Sync(GSSyncReason::Reset);

        // The external 4 MiB GS allocation is the backend hand-off format.
        // This keeps hot backend replacement deterministic even when a future
        // GPU backend keeps a private/mirrored local-memory representation.
        if (m_localMemoryStorage && m_localMemorySize != 0u)
        {
            std::vector<uint8_t> localMemory;
            m_backend->SnapshotVram(localMemory);
            const size_t bytes = std::min<size_t>(localMemory.size(), m_localMemorySize);
            if (bytes != 0u)
                std::memcpy(m_localMemoryStorage, localMemory.data(), bytes);
        }
    }

    m_backend = std::move(backend);
    m_backend->Initialize(m_localMemoryStorage, m_localMemorySize);
}

uint32_t GS::ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    return m_backend ? m_backend->ReadVram(psm, base, bw, x, y) : 0u;
}

void GS::WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value)
{
    std::lock_guard<std::recursive_mutex> lock(m_stateMutex);
    if (m_backend)
        m_backend->WriteVram(psm, base, bw, x, y, value);
}

GSPrimitiveBatch GS::buildDrawBatch(int vertexCount) const
{
    GSPrimitiveBatch batch{};
    // Raw state keys include padding between fields. Give those bytes a stable
    // value before filling every field, so identical materials remain reusable.
    std::memset(&batch.state,0,sizeof(batch.state));
    batch.vertexCount = static_cast<uint8_t>(std::min(vertexCount, 3));
    for (int i = 0; i < batch.vertexCount; ++i)
        batch.vertices[static_cast<size_t>(i)] = m_vtxQueue[i];
    batch.state.context = m_ctx[m_prim.ctxt ? 1 : 0];
    batch.state.prim = m_prim;
    batch.state.texa = m_texa;
    batch.state.texclut = m_texclut;
    batch.state.pabe = m_pabe;
    batch.state.scanmsk = m_scanmsk;
    batch.state.dimx = m_dimx;
    batch.state.dthe = m_dthe;
    batch.state.colclamp = m_colclamp;
    batch.state.fogR = m_fogR;
    batch.state.fogG = m_fogG;
    batch.state.fogB = m_fogB;
    batch.state.textureWidth = static_cast<uint16_t>(1u << std::min<uint32_t>(batch.state.context.tex0.tw, 10u));
    batch.state.textureHeight = static_cast<uint16_t>(1u << std::min<uint32_t>(batch.state.context.tex0.th, 10u));
    const uint64_t tex1 = batch.state.context.tex1;
    const uint8_t mmag = static_cast<uint8_t>((tex1 >> 5u) & 0x1u);
    const uint8_t mmin = static_cast<uint8_t>((tex1 >> 6u) & 0x7u);
    batch.state.linearFilter = mmag != 0u || mmin == 1u || (mmin & 0x4u) != 0u;
    return batch;
}

void GS::updatePreferredDisplaySourceForDraw(const GSPrimitiveBatch &batch)
{
    const GSDrawState &state = batch.state;
    const GSContext &ctx = state.context;
    if (m_hasPreferredDisplaySource && ctx.frame.fbp == m_preferredDisplayDestFbp)
        m_hasPreferredDisplaySource = false;
    if (state.prim.type != GS_PRIM_SPRITE || batch.vertexCount < 2u)
        return;

    const GSVertex &v0 = batch.vertices[0];
    const GSVertex &v1 = batch.vertices[1];
    int x0 = static_cast<int>(v0.x) - (ctx.xyoffset.ofx >> 4);
    int y0 = static_cast<int>(v0.y) - (ctx.xyoffset.ofy >> 4);
    int x1 = static_cast<int>(v1.x) - (ctx.xyoffset.ofx >> 4);
    int y1 = static_cast<int>(v1.y) - (ctx.xyoffset.ofy >> 4);
    if (x0 > x1) std::swap(x0, x1);
    if (y0 > y1) std::swap(y0, y1);
    const int xEnd = x0 + std::max(1, x1 - x0) - 1;
    const int yEnd = y0 + std::max(1, y1 - y0) - 1;
    const uint8_t alphaMode = static_cast<uint8_t>(ctx.alpha & 0xFFu);
    const uint8_t alphaFix = static_cast<uint8_t>((ctx.alpha >> 32u) & 0xFFu);
    const bool displayCopy = state.prim.tme && state.prim.abe && state.prim.fst && state.prim.ctxt &&
                             ctx.frame.fbp != ctx.tex0.tbp0 && alphaMode == 0x64u &&
                             (alphaFix == 0x60u || alphaFix == 0x80u) &&
                             x0 <= 0 && y0 <= 0 && xEnd >= 639 && yEnd >= 447;
    if (displayCopy)
    {
        m_preferredDisplaySourceFrame = {ctx.tex0.tbp0, ctx.tex0.tbw, ctx.tex0.psm, 0u};
        m_preferredDisplayDestFbp = ctx.frame.fbp;
        m_hasPreferredDisplaySource = true;
    }
}
