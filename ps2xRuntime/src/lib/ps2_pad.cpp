#include "runtime/ps2_pad.h"
#include "ps2_host_backend.h"
#include <cmath>
#include <array>
#include <atomic>
#include <chrono>
#include <vector>

extern std::atomic<uint64_t> g_ps2xWotmCompletedFrames;
extern std::atomic<bool> g_ps2xModSelectSeen;
extern std::atomic<bool> g_ps2xRosterEnabled, g_ps2xRosterPage;
extern std::atomic<int> g_ps2xRosterSlot, g_ps2xRosterChosenType;
extern std::atomic<uint64_t> g_ps2xRosterHeartbeatNs;
extern uint32_t g_ps2xRosterIds[10];
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>

namespace {
    std::atomic<int> s_padGlyphFamily[2]{}; // 0: PlayStation, 1: Xbox

    int padGlyphFamilyForName(const char *name)
    {
        if (const char *overrideName = std::getenv("PS2X_BUTTON_GLYPHS")) {
            if (std::strcmp(overrideName, "xbox") == 0) return 1;
            if (std::strcmp(overrideName, "playstation") == 0) return 0;
        }
        char lower[128]{};
        if (name) for (size_t i = 0; i + 1 < sizeof(lower) && name[i]; ++i)
            lower[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(name[i])));
        if (std::strstr(lower, "playstation") || std::strstr(lower, "sony") ||
            std::strstr(lower, "dualshock") || std::strstr(lower, "dualsense") ||
            std::strstr(lower, "ps4") || std::strstr(lower, "ps5")) return 0;
        return std::strstr(lower, "xbox") || std::strstr(lower, "xinput") ||
               std::strstr(lower, "microsoft") ? 1 : 0;
    }
}

// The game thread reads only the published classification, never host input
// library state. Recomputed during pad polling so hot-plugging updates prompts.
int ps2xHostPadGlyphFamily(int port)
{
    return port >= 0 && port < 2 ? s_padGlyphFamily[port].load(std::memory_order_relaxed) : 0;
}

namespace
{
    constexpr uint8_t kPadAnalogMarker = 0x73;
    constexpr uint8_t kPadStickCenter = 0x80;

    constexpr uint16_t PAD_LEFT = 0x0080u;
    constexpr uint16_t PAD_DOWN = 0x0040u;
    constexpr uint16_t PAD_RIGHT = 0x0020u;
    constexpr uint16_t PAD_UP = 0x0010u;
    constexpr uint16_t PAD_START = 0x0008u;
    constexpr uint16_t PAD_R3 = 0x0004u;
    constexpr uint16_t PAD_L3 = 0x0002u;
    constexpr uint16_t PAD_SELECT = 0x0001u;
    constexpr uint16_t PAD_SQUARE = 0x8000u;
    constexpr uint16_t PAD_CROSS = 0x4000u;
    constexpr uint16_t PAD_CIRCLE = 0x2000u;
    constexpr uint16_t PAD_TRIANGLE = 0x1000u;
    constexpr uint16_t PAD_R1 = 0x0800u;
    constexpr uint16_t PAD_L1 = 0x0400u;
    constexpr uint16_t PAD_R2 = 0x0200u;
    constexpr uint16_t PAD_L2 = 0x0100u;
}

// Host pads map to distinct guest ports. Keyboard supplies player one
// when no first gamepad is present; it must never duplicate player two.
int ps2xHostPadIndex(int port)
{
    if(port<0 || port>1)return -1;
    int ordinal=0;
    for(int i=0;i<4;++i)if(IsGamepadAvailable(i)) {
        if(ordinal++==port)return i;
    }
    return -1;
}
bool ps2xHostPadConnected(int port,int slot)
{
    if(slot!=0 || port<0 || port>1)return false;
    // Recorded input explicitly supplies both virtual ports.
    return port==0 || std::getenv("PS2X_PAD_REPLAY")!=nullptr || ps2xHostPadIndex(port)>=0;
}

bool PSPadBackend::readState(int port, int slot, uint8_t *data, size_t size)
{
    if (!data || size < 32 || !ps2xHostPadConnected(port,slot))
        return false;

    // Input replay is indexed by completed game frame, not host time or poll
    // count. Menus use the normal automation; frame-zero records are skipped.
    struct Input { uint64_t frame; uint8_t port,slot; std::array<uint8_t,32> bytes; };
    static const char *replayPath=std::getenv("PS2X_PAD_REPLAY");
    static const std::vector<Input> replay=[] {
        std::vector<Input> inputs;
        if(!replayPath)return inputs;
        auto *f=std::fopen(replayPath,"rb");
        char header[8];
        if(!f || std::fread(header,1,8,f)!=8 || std::memcmp(header,"PS2PAD2\n",8)) {
            std::fprintf(stderr,"[pad:replay] invalid input file %s\n",replayPath);std::abort();
        }
        uint8_t record[42];size_t n;
        while((n=std::fread(record,1,42,f))==42) {
            Input x;std::memcpy(&x.frame,record,8);x.port=record[8];x.slot=record[9];std::memcpy(x.bytes.data(),record+10,32);
            if(x.port>1 || x.slot!=0 || (!inputs.empty() && x.frame<inputs.back().frame))std::abort();
            inputs.push_back(x);
        }
        std::fclose(f);
        if(n || inputs.empty())std::abort();
        std::fprintf(stderr,"[pad:replay] loaded %zu records through frame %llu\n",inputs.size(),inputs.back().frame);
        return inputs;
    }();
    const uint64_t inputFrame=g_ps2xWotmCompletedFrames.load(std::memory_order_relaxed);
    if(replayPath && inputFrame>0)
    {
        static size_t cursor=0;
        static std::array<std::array<uint8_t,32>,2> state=[] {
            std::array<std::array<uint8_t,32>,2> out{};
            for(auto &x:out){x[0]=1;x[1]=kPadAnalogMarker;x[2]=x[3]=255;x[4]=x[5]=x[6]=x[7]=kPadStickCenter;}
            return out;
        }();
        while(cursor<replay.size() && replay[cursor].frame<=inputFrame) {
            const auto &x=replay[cursor++];if(x.frame>0)state[x.port]=x.bytes;
        }
        if(port>=0 && port<2 && slot==0)std::memcpy(data,state[port].data(),32);
        else {std::memset(data,0,32);data[0]=1;data[1]=kPadAnalogMarker;data[2]=data[3]=255;data[4]=data[5]=data[6]=data[7]=kPadStickCenter;}
        if(inputFrame>replay.back().frame){data[2]=data[3]=255;data[4]=data[5]=data[6]=data[7]=kPadStickCenter;}
        return true;
    }

    std::memset(data, 0, 32);
    data[0] = 0x01;
    data[1] = kPadAnalogMarker;
    data[2] = 0xFF;
    data[3] = 0xFF;
    data[4] = data[5] = data[6] = data[7] = kPadStickCenter;

    uint16_t btns = 0xFFFFu;

    // PS2X_AUTO_START=1: synthetic START presses early on, purely so automated
    // benchmark runs can reach the main menu unattended.
    static const bool s_autoStart = std::getenv("PS2X_AUTO_START") != nullptr;
    if (s_autoStart && port==0)
    {
        static const double kStart = GetTime();
        const double elapsed = GetTime() - kStart;
        // The intro movies run for a while before the title screen accepts
        // input, so press repeatedly (0.3 s on, 2.7 s off) for the first 45 s
        // and then stop, leaving the game idling on the main menu.
        // PS2X_AUTO_START_SECONDS extends the press window. The default 28 s
        // assumes the title screen arrives on schedule; a slower or faster boot
        // can miss it entirely and sit on "press start" for the whole run,
        // which silently invalidates any menu measurement.
        static const double kPressUntil = [] {
            const char *v = std::getenv("PS2X_AUTO_START_SECONDS");
            return v ? std::atof(v) : 28.0;
        }();
        const bool pressWindow = elapsed < kPressUntil && std::fmod(elapsed, 2.0) < 0.3;
        if (pressWindow)
            btns &= static_cast<uint16_t>(~PAD_START);
        // Optional bounded UI regression: Unlocks -> Agamo Costume 4,
        // confirm three times to exercise OFF/ON/OFF/ON without other purchases.
        static const bool sweetTest = std::getenv("PS2X_AUTO_SWEET_TEST") != nullptr;
        if (sweetTest) {
            struct Step { double at; uint16_t button; };
            static const Step steps[] = {{30,PAD_DOWN},{31,PAD_DOWN},{32,PAD_DOWN},{33,PAD_DOWN},
                {34,PAD_CROSS},{40,PAD_RIGHT},{42,PAD_RIGHT},{44,PAD_RIGHT},{46,PAD_RIGHT},{48,PAD_RIGHT},
                {50,PAD_DOWN},{52,PAD_DOWN},{54,PAD_DOWN},{58,PAD_CROSS},{60,PAD_CROSS},{62,PAD_CROSS}};
            for(const auto& step:steps)if(elapsed>=step.at && elapsed<step.at+0.15)
                btns &= static_cast<uint16_t>(~step.button);
        }
        // PS2X_AUTO_FIGHT=1: then keep confirming the highlighted default
        // (CROSS every 2 s, 45-130 s). From the main menu that walks
        // 1 player -> mode -> monster -> arena into a fight, giving an
        // unattended, repeatable GAMEPLAY scene for benchmarking and for
        // chasing the stretch bug.
        static const bool s_autoFight = std::getenv("PS2X_AUTO_FIGHT") != nullptr;
        static const double s_autoFightUntil = [] {
            const char *v = std::getenv("PS2X_AUTO_FIGHT_UNTIL");
            return v ? std::atof(v) : 100.0;
        }();
        if (s_autoFight && !g_ps2xModSelectSeen.load(std::memory_order_relaxed) && elapsed >= 28.0 && elapsed < s_autoFightUntil && std::fmod(elapsed, 2.0) < 0.25)
            btns &= static_cast<uint16_t>(~PAD_CROSS);

        // Bounded regression for the optional native camera. Exercise real
        // target-lock input while the camera test continues requesting orbit.
        static const bool cameraTest = std::getenv("PS2X_CAMERA_TEST") != nullptr;
        if (cameraTest && elapsed >= 110.0 && elapsed < 115.0)
            btns &= static_cast<uint16_t>(~(PAD_L1 | PAD_R1));

        // PS2X_AUTO_PAUSE=<frame>: test scaffolding for the pause menu. At the
        // given completed frame press START, then DOWN five times, then CROSS --
        // enough to walk past the five retail rows onto the first injected cheat
        // row and toggle it. Frame-indexed rather than wall-clock so a
        // PS2X_GS_SHOT burst can be aimed at exactly these frames.
        static const uint64_t s_autoPause = [] {
            const char *v = std::getenv("PS2X_AUTO_PAUSE");
            return v ? std::strtoull(v, nullptr, 10) : 0ull;
        }();
        static const double s_autoUnpauseSeconds = [] {
            const char *v = std::getenv("PS2X_AUTO_UNPAUSE_SECONDS");
            return v ? std::strtod(v, nullptr) : 0.0;
        }();
        static const double s_autoPauseSeconds = [] {
            const char *v = std::getenv("PS2X_AUTO_PAUSE_SECONDS");
            return v ? std::strtod(v, nullptr) : 0.0;
        }();
        static const bool s_autoPauseSimple = std::getenv("PS2X_AUTO_PAUSE_SIMPLE") != nullptr;
        static const bool s_autoCheatTest = std::getenv("PS2X_AUTO_CHEAT_TEST") != nullptr;
        if ((s_autoPause != 0ull || s_autoPauseSeconds > 0.0) && port == 0)
        {
            // The frame counter is the right clock for reaching the pause, but
            // it FREEZES while the game is paused -- so everything after START
            // runs on wall time instead. Each press fires once.
            const uint64_t f = g_ps2xWotmCompletedFrames.load(std::memory_order_relaxed);
            static bool started = false;
            static double startedAt = 0.0;
            static bool fired[12] = {};
            static double firedAt[12] = {};
            if (!started && ((s_autoPause != 0ull && f >= s_autoPause) ||
                             (s_autoPauseSeconds > 0.0 && elapsed >= s_autoPauseSeconds)))
            {
                started = true;
                startedAt = GetTime();
                btns &= static_cast<uint16_t>(~PAD_START);
                std::fprintf(stderr, "[pad:autopause] START at frame %llu\n", (unsigned long long)f);
            }
            else if (started)
            {
                const double held = GetTime() - startedAt;
                // A single pad poll can fall between guest input updates.
                // Hold START briefly so the automated pause is observed.
                if (held < 0.2)
                    btns &= static_cast<uint16_t>(~PAD_START);
                const auto pulse = [&](int slot, double at, uint16_t button, const char *what) {
                    if (!fired[slot] && held >= at)
                    {
                        fired[slot] = true;
                        firedAt[slot] = held;
                        std::fprintf(stderr, "[pad:autopause] %s at +%.1fs\n", what, held);
                    }
                    if (fired[slot] && held - firedAt[slot] < 0.2)
                        btns &= static_cast<uint16_t>(~button);
                };
                // Six pause rows: UP wraps 0 -> 5 (QUIT), a second UP lands on
                // 4 (CHEATS). Then open the screen, move down one, toggle.
                if (s_autoCheatTest)
                {
                    pulse(0, 1.0, PAD_UP, "UP (QUIT)");
                    pulse(1, 1.8, PAD_UP, "UP (CHEATS)");
                    pulse(2, 2.6, PAD_CROSS, "CROSS (open)");
                    pulse(3, 3.6, PAD_UP, "UP (BACK row)");
                    pulse(4, 4.4, PAD_CROSS, "CROSS (BACK row)");
                    pulse(5, 5.4, PAD_CROSS, "CROSS (reopen)");
                    pulse(7, 6.4, PAD_TRIANGLE, "TRIANGLE (back)");
                    pulse(8, 7.4, PAD_CROSS, "CROSS (reopen for capture)");
                    pulse(9, 8.4, PAD_CIRCLE, "CIRCLE (must stay open)");
                }
                else if (!s_autoPauseSimple)
                {
                    pulse(0, 1.0, PAD_UP, "UP");
                    pulse(1, 1.8, PAD_UP, "UP");
                    pulse(2, 2.6, PAD_CROSS, "CROSS (open)");
                    pulse(3, 3.6, PAD_DOWN, "DOWN");
                    pulse(4, 4.4, PAD_CROSS, "CROSS (toggle)");
                    // Back out, so the run comes to rest on the six-row pause menu
                    // -- that is the screen whose plate is too small.
                    pulse(5, 6.0, PAD_TRIANGLE, "TRIANGLE (back)");
                }
                if (s_autoUnpauseSeconds > 6.0)
                    pulse(6, s_autoUnpauseSeconds, PAD_START, "START (resume)");
            }
        }
    }

    const int kGamepad = ps2xHostPadIndex(port);
    const bool useGamepad = kGamepad>=0;
    if (port >= 0 && port < 2)
        s_padGlyphFamily[port].store(padGlyphFamilyForName(useGamepad ? GetGamepadName(kGamepad) : nullptr),
                                    std::memory_order_relaxed);
    auto clearBit = [&btns](uint16_t mask)
    { btns &= ~mask; };

    if (useGamepad)
    {
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_FACE_UP))
            clearBit(PAD_UP);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_FACE_DOWN))
            clearBit(PAD_DOWN);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_FACE_LEFT))
            clearBit(PAD_LEFT);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_FACE_RIGHT))
            clearBit(PAD_RIGHT);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_FACE_DOWN))
            clearBit(PAD_CROSS);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_FACE_RIGHT))
            clearBit(PAD_CIRCLE);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_FACE_LEFT))
            clearBit(PAD_SQUARE);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_FACE_UP))
            clearBit(PAD_TRIANGLE);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_TRIGGER_1))
            clearBit(PAD_L1);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_TRIGGER_1))
            clearBit(PAD_R1);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_TRIGGER_2))
            clearBit(PAD_L2);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_TRIGGER_2))
            clearBit(PAD_R2);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_MIDDLE_RIGHT))
            clearBit(PAD_START);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_MIDDLE_LEFT))
            clearBit(PAD_SELECT);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_THUMB))
            clearBit(PAD_L3);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_THUMB))
            clearBit(PAD_R3);

        float lx = GetGamepadAxisMovement(kGamepad, GAMEPAD_AXIS_LEFT_X);
        float ly = GetGamepadAxisMovement(kGamepad, GAMEPAD_AXIS_LEFT_Y);
        float rx = GetGamepadAxisMovement(kGamepad, GAMEPAD_AXIS_RIGHT_X);
        float ry = GetGamepadAxisMovement(kGamepad, GAMEPAD_AXIS_RIGHT_Y);
        data[6] = static_cast<uint8_t>(128 + lx * 127);
        data[7] = static_cast<uint8_t>(128 + ly * 127);
        data[4] = static_cast<uint8_t>(128 + rx * 127);
        data[5] = static_cast<uint8_t>(128 + ry * 127);
    }
    else
    {
        // Arrows = D-pad (menus). WASD = left analog stick (WotM moves the
        // monster with it) and IJKL = right analog stick (camera): the keyboard
        // path used to leave both sticks centred, so on keyboard movement only
        // got the D-pad fallback and the camera could not be steered at all.
        if (IsKeyDown(KEY_UP))
            clearBit(PAD_UP);
        if (IsKeyDown(KEY_DOWN))
            clearBit(PAD_DOWN);
        if (IsKeyDown(KEY_LEFT))
            clearBit(PAD_LEFT);
        if (IsKeyDown(KEY_RIGHT))
            clearBit(PAD_RIGHT);
        auto axis = [](bool negative, bool positive) -> uint8_t
        {
            if (negative == positive)
                return kPadStickCenter;
            return negative ? 0x00u : 0xFFu;
        };
        data[6] = axis(IsKeyDown(KEY_A), IsKeyDown(KEY_D)); // left X
        data[7] = axis(IsKeyDown(KEY_W), IsKeyDown(KEY_S)); // left Y (up = 0)
        data[4] = axis(IsKeyDown(KEY_J), IsKeyDown(KEY_L)); // right X
        data[5] = axis(IsKeyDown(KEY_I), IsKeyDown(KEY_K)); // right Y
        if (IsKeyDown(KEY_X) || IsKeyDown(KEY_SPACE))
            clearBit(PAD_CROSS);
        if (IsKeyDown(KEY_C) || IsKeyDown(KEY_ESCAPE))
            clearBit(PAD_CIRCLE);
        if (IsKeyDown(KEY_Z) || IsKeyDown(KEY_KP_0))
            clearBit(PAD_SQUARE);
        if (IsKeyDown(KEY_V) || IsKeyDown(KEY_KP_1))
            clearBit(PAD_TRIANGLE);
        if (IsKeyDown(KEY_Q))
            clearBit(PAD_L1);
        if (IsKeyDown(KEY_E))
            clearBit(PAD_R1);
        if (IsKeyDown(KEY_LEFT_SHIFT))
            clearBit(PAD_L2);
        if (IsKeyDown(KEY_RIGHT_SHIFT))
            clearBit(PAD_R2);
        if (IsKeyDown(KEY_ENTER))
            clearBit(PAD_START);
        if (IsKeyDown(KEY_TAB))
            clearBit(PAD_SELECT);
    }

    // Opt-in control probe for the hidden Goliath Prime diagnostic. It runs
    // after menu automation has finished, and never affects normal play.
    static const bool modInputTest = std::getenv("PS2X_MOD_INPUT_TEST") != nullptr;
    if (modInputTest && port == 0) {
        static const double start = GetTime();
        const double elapsed = GetTime() - start;
        if (elapsed >= 104.0 && elapsed < 112.0) data[7] = 0;
        if (elapsed >= 114.0 && elapsed < 119.0) clearBit(PAD_SQUARE);
    }
    data[2] = static_cast<uint8_t>(btns & 0xFF);
    data[3] = static_cast<uint8_t>(btns >> 8);

    // PS2X_PAD_TRACE=1: log every change of the button word the game is handed,
    // plus which pad port/slot asked for it. Pause (START) does nothing in game
    // and this says whether the press reaches the guest at all, and on which
    // port -- the pad is polled at 60 Hz, so this only fires when a button
    // actually changes.
    static const bool s_padTrace = std::getenv("PS2X_PAD_TRACE") != nullptr;
    if (s_padTrace)
    {
        static uint16_t s_lastBtns = 0xFFFFu;
        static uint32_t s_logged = 0u;
        if (btns != s_lastBtns && s_logged < 400u)
        {
            ++s_logged;
            s_lastBtns = btns;
            std::fprintf(stderr, "[pad] port=%d slot=%d btns=%04x%s%s%s%s%s%s%s\n",
                         port, slot, btns,
                         (btns & PAD_START) == 0u ? " START" : "",
                         (btns & PAD_SELECT) == 0u ? " SELECT" : "",
                         (btns & PAD_CROSS) == 0u ? " CROSS" : "",
                         (btns & PAD_CIRCLE) == 0u ? " CIRCLE" : "",
                         (btns & PAD_TRIANGLE) == 0u ? " TRIANGLE" : "",
                         (btns & PAD_SQUARE) == 0u ? " SQUARE" : "",
                         (btns & (PAD_UP | PAD_DOWN | PAD_LEFT | PAD_RIGHT)) != (PAD_UP | PAD_DOWN | PAD_LEFT | PAD_RIGHT) ? " DPAD" : "");
        }
    }
    static std::FILE *record=[] {
        const char *p=std::getenv("PS2X_PAD_RECORD");if(!p)return static_cast<std::FILE*>(nullptr);
        auto *f=std::fopen(p,"wb");if(!f){std::fprintf(stderr,"[pad:record] cannot open %s\n",p);std::abort();}
        std::fwrite("PS2PAD2\n",1,8,f);return f;
    }();
    if(record)
    {
        uint8_t entry[42];std::memcpy(entry,&inputFrame,8);entry[8]=static_cast<uint8_t>(port);entry[9]=static_cast<uint8_t>(slot);
        std::memcpy(entry+10,data,32);std::fwrite(entry,1,42,record);std::fflush(record);
    }
    return true;
}
