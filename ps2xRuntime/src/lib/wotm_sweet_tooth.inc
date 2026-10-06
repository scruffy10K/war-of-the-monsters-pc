// Retail unlocker: row 5, column 3 is Agamo Costume 4. Its unlocked word
// (2a1eb0 + 5*b0 + 3*2c + 4) is also McCheckForSweetTooth's +3f8 flag.
// Keep the existing menu/model/selection logic; replace only this activation.
static PS2Runtime::RecompiledFunction s_wotmUnlockInput = nullptr;
static int s_wotmSweetChoice = -2; // -2 unread, -1 use existing save's choice
static std::filesystem::path wotmSweetPath()
{
    if (const char *p = std::getenv("PS2X_SWEET_TOOTH_FILE")) return p;
    const auto& paths = PS2Runtime::getIoPaths();
    const auto root = paths.mcRoot.empty() ? paths.elfDirectory / "mc0" : paths.mcRoot;
    return root / ".native-sweet-tooth";
}
static FILE *wotmSweetFile(const std::filesystem::path& path, bool write)
{
#ifdef _WIN32
    return _wfopen(path.c_str(), write ? L"wb" : L"rb");
#else
    return std::fopen(path.c_str(), write ? "wb" : "rb");
#endif
}
static void wotmSweetRefresh(uint8_t *rdram)
{
    if (s_wotmSweetChoice == -2) {
        s_wotmSweetChoice = -1;
        if (FILE *f = wotmSweetFile(wotmSweetPath(), false)) {
            const int value = std::fgetc(f);
            if (value == '0' || value == '1') s_wotmSweetChoice = value - '0';
            std::fclose(f);
        }
    }
    if (s_wotmSweetChoice >= 0) wotmWrite32At(rdram, 0x002a22a8u, s_wotmSweetChoice);
    const bool on = wotmRead32(rdram, 0x002a22a8u) != 0;
    char label[25]{};
    std::snprintf(label, sizeof(label), "Sweet Tooth: %s", on ? "ON" : "OFF");
    if (auto *p = getMemPtr(rdram, 0x002a22b7u)) std::memcpy(p, label, sizeof(label));
    // Seven-byte retail price string. This option no longer costs tokens.
    wotmWrite32At(rdram, 0x002a22acu, 0u);
    if (auto *p = getMemPtr(rdram, 0x002a22b0u)) std::memcpy(p, "TOGGLE", 7);
}
static void wotmSweetToggle(uint8_t *rdram)
{
    s_wotmSweetChoice = wotmRead32(rdram, 0x002a22a8u) ? 0 : 1;
    wotmSweetRefresh(rdram);
    const auto path = wotmSweetPath();
    std::error_code ec;
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path(), ec);
    bool saved = false;
    if (!ec) if (FILE *f = wotmSweetFile(path, true)) {
        saved = std::fputc('0' + s_wotmSweetChoice, f) != EOF;
        if (std::fclose(f) != 0) saved = false;
    }
    std::fprintf(stderr, "[sweet-tooth] %s, preference %s\n", s_wotmSweetChoice ? "ON" : "OFF", saved ? "saved" : "save failed");
}
static void wotmSweetInput(uint8_t *rdram, R5900Context *ctx, PS2Runtime *runtime)
{
    wotmSweetRefresh(rdram);
    const uint32_t caller = getRegU32(ctx, 31);
    struct Scope { Scope(){++t_ps2xEeCheckpointDeferralDepth;} ~Scope(){--t_ps2xEeCheckpointDeferralDepth;} } scope;
    s_wotmUnlockInput(rdram, ctx, runtime);
    if (ctx->pc != caller || t_ps2xEeUnwinding || caller != 0x0019f804u || (getRegU32(ctx, 2) & 0xffffu) != 6u) return;
    const uint32_t gp = getRegU32(ctx, 28);
    if (wotmRead32(rdram, gp - 0x785cu) != 5u || wotmRead32(rdram, gp - 0x7868u) != 3u) return;
    wotmSweetToggle(rdram);
    wotmSetReg(ctx, 2, 0); // consume confirm; do not enter purchase/save-card flow
}
