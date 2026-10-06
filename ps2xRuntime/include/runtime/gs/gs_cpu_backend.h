#pragma once

#include "runtime/gs/gs_backend.h"

#include <array>
#include <condition_variable>
#include <thread>
#include <functional>
#include <mutex>
#include <vector>

class GSCpuBackend final : public GSRasterBackend
{
public:
    GSCpuBackend();
    ~GSCpuBackend() override;

    void Initialize(uint8_t *vram, uint32_t vramSize) override;
    void Reset() override;

    void Submit(const GSPrimitiveBatch &batch) override;
    void BeginTransfer(const GSTransferCommand &command) override;
    void UploadImage(const uint8_t *data, uint32_t sizeBytes) override;

    void Flush() override;
    void TextureFlush() override;
    void Sync(GSSyncReason reason) override;
    PresentationFrame Present(const GSPresentationRequest &request) override;

    bool ClearFramebuffer(const GSContext &context, uint32_t rgba) override;
    uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) override;

    uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const override;
    void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) override;
    void SnapshotVram(std::vector<uint8_t> &out) const override;
    GSTransferSnapshot GetTransferSnapshot() const override;

    // Presentation snapshot ordered in the raster FIFO. The frontend enqueues it
    // while holding the GS state lock, right after reading DISPFB/DISPLAY, so the
    // copy of local memory contains every draw submitted before the display
    // registers were read and none submitted after. (Present() copied VRAM after
    // an unordered Sync, so a frame the game had already started redrawing -- it
    // flipped away and cleared/drew that buffer again -- could be shown half
    // drawn: WotM's drive-in billboard vanished for single frames.)
    // Returns the token PresentSnapshot waits for.
    GSCpuBackend *PresentSnapshotBackend() override { return this; }

    uint64_t EnqueuePresentSnapshot();
    PresentationFrame PresentSnapshot(uint64_t token, const GSPresentationRequest &request);

    // Decodes the texture TEX0 currently selects into RGBA8 (one byte per
    // channel, ascending, PS2 alpha where 128 = 1.0), using the same swizzle,
    // palette and TEXA rules as texel sampling. GPU backends use it to turn a
    // PS2 texture into something the graphics driver can hold; filtering and
    // wrapping are the caller's job, so it reads texel centres with neither.
    void DecodeTextureRgba(const GSDrawState &state, uint32_t width, uint32_t height, uint8_t *outRgba);

private:
    // Raster worker. Primitives and every VRAM-mutating operation (transfers,
    // image uploads, clears, pokes) are appended to one ordered FIFO and executed
    // on a dedicated thread, so rasterization no longer runs inline inside the
    // game thread's VU1 XGKICK. Anything that reads VRAM or transfer state first
    // waits for the work submitted before it (a ticket, not "queue empty", so a
    // reader on another thread can't be starved by a producer that keeps
    // submitting). The thread starts lazily on the first command, so the
    // thread_local snapshot instance Present() uses never spawns one.
    // PS2X_GS_SYNC=1 runs every command inline instead (old behaviour) for A/B.
    struct RasterCommand
    {
        enum class Kind : uint8_t
        {
            Draw,
            BeginTransfer,
            UploadImage,
            Clear,
            WriteVram,
            PresentSnapshot,
        };
        Kind kind = Kind::Draw;
        GSPrimitiveBatch batch{};
        GSTransferCommand transfer{};
        std::vector<uint8_t> bytes;
        GSContext clearContext{};
        uint32_t clearRgba = 0;
        uint32_t psm = 0, base = 0, bw = 0, x = 0, y = 0, value = 0;
    };
    static constexpr size_t kMaxPendingRasterCommands = 16384u;

    void Enqueue(RasterCommand &&command);
    void ExecuteCommand(RasterCommand &command);
    void WorkerMain();
    void WaitForRasterIdle() const;
    void BeginTransferUnlocked(const GSTransferCommand &command);
    void UploadImageUnlocked(const uint8_t *data, uint32_t sizeBytes);
    bool ClearFramebufferUnlocked(const GSContext &context, uint32_t rgba);
    void SnapshotVramUnsynced(std::vector<uint8_t> &out) const;

    // Banded parallel raster. The raster worker acts as band 0 and dispatcher;
    // m_bandCount-1 extra threads each rasterize the same run of draws but only
    // write rows whose 8-row group (y >> 3) % m_bandCount equals their band, so
    // no two threads ever write the same framebuffer/Z bytes. Anything that is
    // not a plain draw (transfers, uploads, clears, pokes), draws into 8/4-bit
    // or unknown targets, and render-to-texture dependencies inside a run are
    // executed single-threaded between runs (see ExecuteWorkParallel).
    // PS2X_GS_THREADS=N overrides the band count (1 = single raster thread).
    void ExecuteWorkParallel(std::vector<RasterCommand> &work);
    void RunSegmentOnAllBands(const RasterCommand *commands, size_t count);
    void BandThreadMain(uint32_t band);
    void StartBandThreads();

    void ResetUnlocked();
    uint32_t ReadVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const;
    void WriteVramUnlocked(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value);

    void DrawPrimitive(const GSPrimitiveBatch &batch);
    void DrawSprite(const GSPrimitiveBatch &batch);
    void DrawTriangle(const GSPrimitiveBatch &batch);
    void DrawLine(const GSPrimitiveBatch &batch);
    void WritePixel(const GSDrawState &state, int x, int y, int z, uint8_t r, uint8_t g, uint8_t b, uint8_t a, uint8_t fog);
    uint32_t SampleTexture(const GSDrawState &state, float s, float t, float q, uint16_t u, uint16_t v);
    uint32_t LookupCLUT(const GSDrawState &state, uint8_t index, uint32_t cbp, uint8_t cpsm, uint8_t csm, uint8_t csa, uint8_t sourcePsm);

    void PerformLocalToLocalTransfer();
    void PerformLocalToHostTransfer();
    PresentationFrame PresentFromLocalMemory(const GSPresentationRequest &request);
    bool CopyFrameToHostRgba(const GSFrameReg &frame,
                             uint32_t width,
                             uint32_t height,
                             std::vector<uint8_t> &outPixels,
                             bool preserveAlpha,
                             bool useLocalMemoryLayout,
                             bool frameBaseIsPages,
                             uint32_t sourceOriginX,
                             uint32_t sourceOriginY) const;

    // Raw function pointers, not std::function: these are called for every
    // texel fetch (four per bilinear sample) and every framebuffer / Z
    // read-modify-write, and every slot holds a plain free function (unused PSMs
    // get ReadNull/WriteNull, so no slot is ever empty). std::function added a
    // type-erased indirect call the compiler could not see through, on the
    // hottest path in the rasterizer, for no benefit.
    using WriteVramFunc = void (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
    using ReadVramFunc = uint32_t (*)(uint8_t *, uint32_t, uint32_t, uint32_t, uint32_t);

    static constexpr size_t kPsmHandlerCount = 1u << 6u;
    mutable std::mutex m_mutex;
    uint8_t *m_vram = nullptr;
    uint32_t m_vramSize = 0;
    std::array<ReadVramFunc, kPsmHandlerCount> m_readVramFuncs{};
    std::array<WriteVramFunc, kPsmHandlerCount> m_writeVramFuncs{};

    GSTransferCommand m_transfer{};
    GSTransferSnapshot m_transferState{};
    std::vector<uint8_t> m_localToHostBuffer;
    size_t m_localToHostReadPos = 0;

    // Filled by the raster worker when it reaches a PresentSnapshot command.
    std::mutex m_presentSnapshotMutex;
    std::vector<uint8_t> m_presentSnapshot;

    mutable std::mutex m_queueMutex;
    mutable std::condition_variable m_doneCv;
    std::condition_variable m_queueCv;
    std::condition_variable m_spaceCv;
    std::vector<RasterCommand> m_pending;
    uint64_t m_submittedSeq = 0;
    uint64_t m_completedSeq = 0;
    bool m_stopWorker = false;
    std::thread m_worker;

    std::vector<std::thread> m_bandThreads;
    uint32_t m_bandCount = 1;
    std::mutex m_bandMutex;
    std::condition_variable m_bandStartCv;
    std::condition_variable m_bandDoneCv;
    const RasterCommand *m_bandSegment = nullptr;
    size_t m_bandSegmentCount = 0;
    uint64_t m_bandGeneration = 0;
    uint32_t m_bandsRemaining = 0;
    bool m_bandStop = false;
};
