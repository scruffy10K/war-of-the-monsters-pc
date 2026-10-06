// OpenGL GS backend (PS2X_GS_GPU=1). Skeleton: draws the game's primitives into
// a GPU render target instead of the CPU rasterizer, then hands the result to
// the existing presentation path. Everything that is not drawing (VRAM
// transfers, reads, presentation decode) is delegated to the CPU backend.
#pragma once

#include "gs_backend.h"
#include "gs_cpu_backend.h"

#include <memory>

class GSGlBackend final : public GSRasterBackend
{
public:
    GSGlBackend();
    ~GSGlBackend() override;

    // Returns false when no OpenGL 4.3 context could be created; the caller
    // should keep using the CPU backend in that case.
    bool Available() const;

    void Initialize(uint8_t *vram, uint32_t vramSize) override;
    void Reset() override;

    void Submit(const GSPrimitiveBatch &batch) override;
    void BeginTransfer(const GSTransferCommand &command) override;
    void UploadImage(const uint8_t *data, uint32_t sizeBytes) override;

    // Presentation still comes out of local memory for anything the game puts
    // on screen by transfer rather than by drawing (the intro movies).
    GSCpuBackend *PresentSnapshotBackend() override { return &m_cpu; }
    bool NeedsPresentSnapshot(const GSPresentationRequest &request) override;
    void MarkPresentBoundary() override;

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

    // Published so the host renderer's present path can reach the live
    // instance's state (see ps2xGsGpuPresentTexture).
    struct Impl;

private:
    std::unique_ptr<Impl> m_impl;
    GSCpuBackend m_cpu; // VRAM ownership, transfers, reads, presentation decode
};

// A finished GPU frame the host can draw straight from the render target, with
// no read back. The texture belongs to a context shared with the host
// renderer's, so its id is valid on the host thread.
struct Ps2xGpuFrame
{
    unsigned int texture = 0u;
    uint32_t textureWidth = 0u, textureHeight = 0u;   // the whole render target
    uint32_t sourceWidth = 0u, sourceHeight = 0u;     // the visible part, in target pixels
    uint32_t displayWidth = 0u, displayHeight = 0u;   // the same, in PS2 pixels (aspect ratio)
};

bool ps2xGsGpuPresentTexture(Ps2xGpuFrame *out);
