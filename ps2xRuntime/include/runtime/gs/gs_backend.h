#pragma once

#include "runtime/gs/gs_types.h"

#include <cstdint>
#include <vector>

class GSCpuBackend;

class GSRasterBackend
{
public:
    virtual ~GSRasterBackend() = default;

    // The backend that owns local memory and can take an ordered snapshot of
    // it, if any. A GPU backend returns the CPU backend it delegates to, so the
    // frontend can queue the snapshot in raster order even though the GPU
    // backend is the one presenting.
    virtual GSCpuBackend *PresentSnapshotBackend() { return nullptr; }

    // Will this frame be presented from local memory? The frontend only pays
    // for a snapshot when the answer is yes.
    virtual bool NeedsPresentSnapshot(const GSPresentationRequest &) { return true; }

    // Called while the GS state lock is held, at the instant the display
    // registers were read: everything submitted before this belongs to the
    // frame about to be presented, everything after it to the next one.
    virtual void MarkPresentBoundary() {}

    virtual void Initialize(uint8_t *vram, uint32_t vramSize) = 0;
    virtual void Reset() = 0;

    virtual void Submit(const GSPrimitiveBatch &batch) = 0;

    virtual void BeginTransfer(const GSTransferCommand &command) = 0;
    virtual void UploadImage(const uint8_t *data, uint32_t sizeBytes) = 0;

    virtual void Flush() = 0;
    virtual void TextureFlush() = 0;
    virtual void Sync(GSSyncReason reason) = 0;
    virtual PresentationFrame Present(const GSPresentationRequest &request) = 0;

    virtual bool ClearFramebuffer(const GSContext &context, uint32_t rgba) = 0;
    virtual uint32_t ConsumeLocalToHostBytes(uint8_t *dst, uint32_t maxBytes) = 0;

    virtual uint32_t ReadVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y) const = 0;
    virtual void WriteVram(uint32_t psm, uint32_t base, uint32_t bw, uint32_t x, uint32_t y, uint32_t value) = 0;
    virtual void SnapshotVram(std::vector<uint8_t> &out) const = 0;
    virtual GSTransferSnapshot GetTransferSnapshot() const = 0;
};
