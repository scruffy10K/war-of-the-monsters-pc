// Retail viewUpdate builds camera-relative clipping planes and the three
// projection matrices consumed by EE/VU geometry. Adjust its INPUTS only for
// a full-width gameplay view, then restore them before another view uses them.
// The composed outputs remain wide. This also survives later viewSetFov calls.
static PS2Runtime::RecompiledFunction s_wotmViewUpdate = nullptr;
static PS2Runtime::RecompiledFunction s_wotmViewSetFov = nullptr;
static bool wotmWidescreenRequested()
{
    static const bool requested = [] {
        const char *value = std::getenv("PS2X_ASPECT");
        return value && std::strcmp(value, "16:9") == 0;
    }();
    return requested;
}
static void wotmWideViewUpdate(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
{
    const uint32_t view = getRegU32(ctx, 4);
    const uint32_t gp = getRegU32(ctx, 28);
    const uint32_t views = gp >= 0x6e10u && gp < PS2_RAM_SIZE ? wotmRead32(rdram, gp - 0x6e10u) : 0u;
    const uint32_t definition = view < 5u ? wotmRead32(rdram, 0x006e1904u + view * 20u) : 0u;
    const bool valid = definition && definition <= PS2_RAM_SIZE - 0xb4u;
    const uint32_t dimensions = valid ? wotmRead32(rdram, definition + 4u) : 0u;
    const bool wide = wotmWidescreenRequested() && view == 0u && views == 1u &&
        g_ps2xWotmPhase.load(std::memory_order_relaxed) == 2u &&
        (dimensions & 0xffffu) == 640u && (dimensions >> 16u) == 224u;
    if (view == 0u) g_ps2xWotmWideActive.store(wide, std::memory_order_relaxed);
    static const bool trace = std::getenv("PS2X_WIDESCREEN_TRACE") != nullptr;
    if (trace) {
        static uint32_t reported = 0;
        const uint32_t key = (wide ? 8u : 0u) + std::min(view, 7u);
        if (!(reported & (1u << key))) {
            reported |= 1u << key;
            std::fprintf(stderr,"[ws:view] view=%u count=%u size=%ux%u phase=%u wide=%u definition=%08x\n",
                view,views,dimensions&0xffffu,dimensions>>16u,
                g_ps2xWotmPhase.load(std::memory_order_relaxed),wide,definition);
        }
    }
    if (!wide) { s_wotmViewUpdate(rdram, ctx, runtime); return; }
    // Existing presentation uses 640x448 square host pixels (field line doubling).
    // Keep its vertical framing/proportions while expanding the output to 16:9.
    constexpr float horizontalScale = (640.0f / 448.0f) / (16.0f / 9.0f);
    struct ProjectionScope {
        uint8_t *ram;
        uint32_t definition;
        float x[3];
        float planes[32];
        ProjectionScope(uint8_t *r, uint32_t d) : ram(r), definition(d) {
            ++t_ps2xEeCheckpointDeferralDepth;
            for (unsigned i=0;i<3;++i) std::memcpy(&x[i],ram+0x006e1990u+i*0x140u,4);
            std::memcpy(planes,ram+definition+0x30u,sizeof(planes));
        }
        ~ProjectionScope() {
            for (unsigned i=0;i<3;++i) std::memcpy(ram+0x006e1990u+i*0x140u,&x[i],4);
            std::memcpy(ram+definition+0x30u,planes,sizeof(planes));
            --t_ps2xEeCheckpointDeferralDepth;
        }
    } saved(rdram, definition);
    for (unsigned i=0;i<3;++i) {
        const float x = saved.x[i] * horizontalScale;
        std::memcpy(rdram+0x006e1990u+i*0x140u,&x,4);
    }
    float planes[32];
    std::memcpy(planes,saved.planes,sizeof(planes));
    // viewSetFov transposes each four-plane block. A plane is a COLUMN here.
    // Transform/renormalize its normal; retaining unit length preserves sphere
    // and box culling distances. Top/bottom and near/far normals stay unchanged.
    for (unsigned block=0;block<32;block+=16) for (unsigned column=0;column<4;++column) {
        const unsigned p=block+column;
        if (planes[p] == 0.0f) continue;
        planes[p] *= horizontalScale;
        const float length = std::sqrt(planes[p]*planes[p]+planes[p+4]*planes[p+4]+planes[p+8]*planes[p+8]);
        if (length > 0.0f && std::isfinite(length))
            for (unsigned row=0;row<4;++row) planes[p+row*4] /= length;
    }
    std::memcpy(rdram+definition+0x30u,planes,sizeof(planes));
    s_wotmViewUpdate(rdram, ctx, runtime);
}
static void wotmTraceProjectionMatrices(uint8_t *rdram, uint64_t frame)
{
    static const bool trace = std::getenv("PS2X_WIDESCREEN_TRACE") != nullptr;
    if (!trace ||
        (frame != 1u && frame != 30u && frame != 300u && frame != 600u)) return;
    constexpr uint32_t bases[] = {0x006E1990u, 0x006E1AD0u, 0x006E1C10u, 0x006E1FD0u};
    for (uint32_t base : bases) {
        std::fprintf(stderr, "[ws:matrix] frame=%llu base=%08x",
            static_cast<unsigned long long>(frame), base);
        for (unsigned index = 0; index < 16; ++index) {
            const uint32_t bits = wotmRead32(rdram, base + index * 4u);
            float value;
            std::memcpy(&value, &bits, sizeof(value));
            std::fprintf(stderr, " %g", static_cast<double>(value));
        }
        std::fputc('\n', stderr);
    }
}
static void wotmTraceViewSetFov(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
{
    if (ctx->pc == 0x00229328u) {
        static unsigned count = 0;
        if (count++ < 160u)
            std::fprintf(stderr,
                "[ws:fov] n=%u frame=%llu phase=%u view=%u ra=%08x f12=%g f13=%g screen=%u\n",
                count,
                static_cast<unsigned long long>(g_ps2xWotmCompletedFrames.load(std::memory_order_relaxed)),
                g_ps2xWotmPhase.load(std::memory_order_relaxed),
                getRegU32(ctx, 4), getRegU32(ctx, 31),
                static_cast<double>(ctx->f[12]), static_cast<double>(ctx->f[13]),
                wotmRead32(rdram, 0x006F8464u));
    }
    s_wotmViewSetFov(rdram, ctx, runtime);
}

// Retail HUD::update transforms these three layout anchors through the
// camera node's basis. In the game's Z-up coordinates Y is camera distance.
// Translate the anchors by the extra horizontal half-width, keeping their
// dimensions and vertical placement. Restore inputs after the bounded update;
// two-player views, later menus and repeated updates never accumulate an offset.
static PS2Runtime::RecompiledFunction s_wotmHudUpdate = nullptr;
static void wotmWideHudUpdate(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
{
    const uint32_t hud = getRegU32(ctx,4);
    if (ctx->pc != 0x143698u || !g_ps2xWotmWideActive.load(std::memory_order_relaxed) ||
        !hud || hud > PS2_RAM_SIZE-0x2e0u || wotmRead32(rdram,hud) != 0u) {
        s_wotmHudUpdate(rdram,ctx,runtime); return;
    }
    float scaleX;
    std::memcpy(&scaleX,rdram+0x6e1990u,4); // original, restored projection input
    if (!std::isfinite(scaleX) || scaleX <= 1.0f) {
        s_wotmHudUpdate(rdram,ctx,runtime); return;
    }
    float adjusted[3];
    for (unsigned i=0;i<3;++i) {
        float x, distance;
        std::memcpy(&x,rdram+hud+0x100u+i*16u,4);
        std::memcpy(&distance,rdram+hud+0x104u+i*16u,4);
        adjusted[i]=x-320.0f*(56.0f/45.0f-1.0f)*distance/scaleX;
        if (!std::isfinite(x) || !std::isfinite(distance) || distance <= 0.0f ||
            !std::isfinite(adjusted[i]) || std::fabs(adjusted[i]-x)>256.0f) {
            s_wotmHudUpdate(rdram,ctx,runtime); return;
        }
    }
    struct AnchorScope {
        uint8_t *ram; uint32_t hud; uint32_t original[3];
        AnchorScope(uint8_t *r,uint32_t h,const float *values) : ram(r),hud(h) {
            ++t_ps2xEeCheckpointDeferralDepth;
            for(unsigned i=0;i<3;++i) {
                std::memcpy(&original[i],ram+hud+0x100u+i*16u,4);
                std::memcpy(ram+hud+0x100u+i*16u,&values[i],4);
            }
        }
        ~AnchorScope() {
            for(unsigned i=0;i<3;++i) std::memcpy(ram+hud+0x100u+i*16u,&original[i],4);
            --t_ps2xEeCheckpointDeferralDepth;
        }
    } anchors(rdram,hud,adjusted);
    s_wotmHudUpdate(rdram,ctx,runtime);
}

static void wotmInstallWidescreen(PS2Runtime &runtime)
{
    if (wotmWidescreenRequested()) {
        s_wotmHudUpdate=runtime.lookupFunction(0x143698u);
        if (s_wotmHudUpdate && s_wotmHudUpdate != &wotmWideHudUpdate)
            runtime.replaceFunction(0x143698u,&wotmWideHudUpdate);
    }
    if (wotmWidescreenRequested() || std::getenv("PS2X_WIDESCREEN_TRACE")) {
        s_wotmViewUpdate = runtime.lookupFunction(0x00227e10u);
        if (s_wotmViewUpdate && s_wotmViewUpdate != &wotmWideViewUpdate)
            runtime.replaceFunction(0x00227e10u, &wotmWideViewUpdate);
    }
    if (!std::getenv("PS2X_WIDESCREEN_TRACE")) return;
    s_wotmViewSetFov = runtime.lookupFunction(0x00229328u);
    if (s_wotmViewSetFov && s_wotmViewSetFov != &wotmTraceViewSetFov)
        runtime.replaceFunction(0x00229328u, &wotmTraceViewSetFov);
}
