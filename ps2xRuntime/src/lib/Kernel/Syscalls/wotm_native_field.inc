// Native GPU frames do not reconfigure an interlaced scanout buffer.
// rtMain's MotionBlurAA exit waits until sceGsSyncV returns the parity
// already saved in its frame-local slot (SLTIU result, not the raw field).
// Complete that configuration handshake without consuming another game tick.
inline bool wotmNativeConfigField(bool nativeGpu, uint32_t ra, int fixedResult,
                                  uint32_t target, bool recentFrame,
                                  uint32_t savedParity, int &result)
{
    if (!nativeGpu || ra != 0x001884b0u || fixedResult >= 0 ||
        target != 60u || !recentFrame || savedParity > 1u)
        return false;
    result = static_cast<int>(savedParity ^ 1u);
    return true;
}
