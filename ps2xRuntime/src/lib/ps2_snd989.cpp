// 989SND (Sony 989 Sound System) host implementation -- music streams.
//
// The game's EE library (snd_* in the ELF) talks to 989SND.IRX on the IOP over
// a private SIF RPC server (SID 0x123456). Commands are batched by
// snd_SendCurrentBatch as RPC function 0x4D:
//     send:    u32 count, then per command { u16 id; u16 len; payload padded to 4 }
//     receive: u32 0xFFFFFFFF, u32 return value per command, u32 0xFFFFFFFF
// snd_GotReturns waits for both sentinels, then hands each value to the
// command's callback (the *_CB variants store it into a game variable).
//
// Implemented here: VAG music streams (/VAG/*.VPK). The game starts streams by
// disc location (snd_PlayVAGStreamByLoc), cross-fades ambient/battle music with
// snd_SetSoundParams_A, polls snd_SoundIsStillPlaying and restarts a stream when
// it reports 0 (the VPK data has no loop flags), and pauses/continues groups.
//
// VPK: 0x800-byte header { "VPK ", u32 bytes per channel, u32 data offset,
// u32 interleave, u32 sample rate, u32 channels, ... }; data is PS-ADPCM, 16-byte
// frames of 28 samples, interleaved in blocks: <interleave> bytes of channel 0,
// then <interleave> bytes of channel 1, repeating.
//
// Sound-effect banks (/SND/*.BNK, command 0x11 etc.) are not implemented yet:
// those commands keep the old 0xFFFFFFFF replies.
#include "ps2_runtime.h"
#include "ps2_snd989_sfx.h"
#include "installed_disc_api.inc"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if !defined(PLATFORM_VITA)
#include "raylib.h"
#endif

// Host SFX extensions, deliberately confined to CPP translation units.
void ps2x_snd989_set_reverb_type(uint32_t coreMask, uint32_t type);
void ps2x_snd989_set_reverb_depth(uint32_t coreMask, int32_t left, int32_t right);

namespace
{
    constexpr uint32_t kSnd989Sid = 0x123456u;
    constexpr uint32_t kBatchRpc = 0x4Du;
    constexpr uint32_t kOutputRate = 48000u;
    constexpr uint32_t kSectorSize = 2048u;

    enum : uint32_t
    {
        kCmdSetMasterVolume = 0x09,
        kCmdGetMasterVolume = 0x0A,
        kCmdPauseGroup = 0x16,
        kCmdContinueGroup = 0x17,
        kCmdSoundIsStillPlaying = 0x19,
        kCmdSetSoundVolPan = 0x1B,
        kCmdSetSoundParams = 0x21,
        kCmdInitVagStreaming = 0x2A,
        kCmdPlayVagStreamByLoc = 0x2C,
        kCmdPauseVagStream = 0x2D,
        kCmdContinueVagStream = 0x2E,
        kCmdStopVagStream = 0x2F,
        kCmdGetVagStreamLoc = 0x31,
        kCmdStopAllStreams = 0x34,
        kCmdIsVagStreamBuffered = 0x4F,
    };

    struct AdpcmChannel
    {
        int32_t s1 = 0;
        int32_t s2 = 0;
    };

    // PS-ADPCM frame -> 28 samples.
    void decodeAdpcmFrame(const uint8_t *frame, AdpcmChannel &st, int16_t *out)
    {
        static constexpr int32_t kF0[5] = {0, 60, 115, 98, 122};
        static constexpr int32_t kF1[5] = {0, 0, -52, -55, -60};
        const int32_t shift = frame[0] & 0x0F;
        int32_t filter = (frame[0] >> 4) & 0x0F;
        if (filter > 4)
            filter = 0;
        for (int i = 0; i < 28; ++i)
        {
            int32_t nibble = (frame[2 + i / 2] >> ((i & 1) ? 4 : 0)) & 0x0F;
            if (nibble & 0x8)
                nibble -= 16;
            int32_t sample = (nibble << 12) >> shift;
            sample += (st.s1 * kF0[filter] + st.s2 * kF1[filter] + 32) >> 6;
            sample = std::clamp(sample, -32768, 32767);
            st.s2 = st.s1;
            st.s1 = sample;
            out[i] = static_cast<int16_t>(sample);
        }
    }

    struct VagStream
    {
        uint32_t handle = 0u;
        uint32_t loc = 0u;
        uint32_t group = 0u;
        float volume = 1.0f; // 0x400 = 1.0
        bool paused = false;
        bool finished = false;
        std::FILE *file = nullptr;
        uint32_t bytesPerChannel = 0u;
        uint32_t dataOffset = 0u;
        uint32_t interleave = 0u;
        uint32_t rate = 44100u;
        uint32_t channels = 2u;
        uint32_t nextBlock = 0u;
        std::array<AdpcmChannel, 2> adpcm{};
        std::vector<int16_t> pcm; // interleaved stereo at the stream's rate
        size_t pcmFrames = 0u;
        double position = 0.0; // fractional frame position in pcm
        std::vector<uint8_t> blockBytes;

        ~VagStream()
        {
            if (file)
                wotm_disc::close(file);
        }

        bool open(const std::string &isoPath)
        {
            file = wotm_disc::open(std::filesystem::path(isoPath));
            if (!file)
                return false;
            uint8_t header[0x20]{};
            size_t got = 0;
            if (!wotm_disc::read(file, static_cast<uint64_t>(loc) * kSectorSize, header, sizeof(header), got) || got != sizeof(header))
                return false;
            if (std::memcmp(header, " KPV", 4) != 0)
                return false;
            std::memcpy(&bytesPerChannel, header + 4, 4);
            std::memcpy(&dataOffset, header + 8, 4);
            std::memcpy(&interleave, header + 12, 4);
            std::memcpy(&rate, header + 16, 4);
            std::memcpy(&channels, header + 20, 4);
            return interleave >= 16u && (interleave % 16u) == 0u && rate >= 8000u && rate <= 96000u &&
                   (channels == 1u || channels == 2u) && bytesPerChannel != 0u;
        }

        // Decode the next interleave block of every channel into pcm.
        bool decodeNextBlock()
        {
            const uint64_t blockStart = static_cast<uint64_t>(nextBlock) * interleave;
            if (blockStart >= bytesPerChannel)
                return false;
            const uint32_t bytes = static_cast<uint32_t>(std::min<uint64_t>(interleave, bytesPerChannel - blockStart));
            const uint32_t frames = bytes / 16u;
            blockBytes.resize(static_cast<size_t>(interleave) * channels);
            const int64_t filePos = static_cast<int64_t>(loc) * kSectorSize + dataOffset +
                                    static_cast<int64_t>(nextBlock) * interleave * channels;
            size_t got = 0;
            if (!wotm_disc::read(file, static_cast<uint64_t>(filePos), blockBytes.data(), blockBytes.size(), got))
                return false;
            if (got < static_cast<size_t>(interleave) * (channels - 1u) + bytes)
                return false;
            pcm.assign(static_cast<size_t>(frames) * 28u * 2u, 0);
            int16_t decoded[28];
            for (uint32_t c = 0; c < channels; ++c)
            {
                const uint8_t *base = blockBytes.data() + static_cast<size_t>(c) * interleave;
                for (uint32_t f = 0; f < frames; ++f)
                {
                    decodeAdpcmFrame(base + f * 16u, adpcm[c], decoded);
                    for (int i = 0; i < 28; ++i)
                    {
                        const size_t o = (static_cast<size_t>(f) * 28u + i) * 2u;
                        if (channels == 1u)
                        {
                            pcm[o] = decoded[i];
                            pcm[o + 1] = decoded[i];
                        }
                        else
                        {
                            pcm[o + c] = decoded[i];
                        }
                    }
                }
            }
            pcmFrames = static_cast<size_t>(frames) * 28u;
            ++nextBlock;
            return true;
        }
    };

    class Snd989Player
    {
    public:
        void handleRpc(uint32_t function, const uint8_t *send, uint32_t sendSize, uint8_t *recv, uint32_t recvSize)
        {
            if (function != kBatchRpc)
            {
                // snd_SendIOPCommandAndWait sends the command as the RPC
                // function itself, with an unwrapped argument payload. Its
                // result is between the same two sentinel words as a batch.
                // In particular GetMasterVolume must report the fade reaching
                // zero; the old all-FF placeholder kept the pause fade running.
                if (sendSize != 0u && !send)
                    return;
                uint32_t args[8]{};
                const size_t length = std::min<size_t>(sendSize, sizeof(args));
                if (length)
                    std::memcpy(args, send, length);
                uint32_t result = 0xFFFFFFFFu;
                command(function, args, static_cast<uint16_t>(length), result);
                if (recv && recvSize >= 8u)
                    std::memcpy(recv + 4u, &result, sizeof(result));
                return;
            }
            if (!send || sendSize < 4u)
                return;
            uint32_t count = 0u;
            std::memcpy(&count, send, 4u);
            std::vector<uint32_t> returns(count, 0xFFFFFFFFu);
            size_t pos = 4u;
            for (uint32_t i = 0; i < count && pos + 4u <= sendSize; ++i)
            {
                uint16_t id = 0u, len = 0u;
                std::memcpy(&id, send + pos, 2u);
                std::memcpy(&len, send + pos + 2u, 2u);
                if (pos + 4u + len > sendSize)
                    break;
                uint32_t args[8]{};
                std::memcpy(args, send + pos + 4u, std::min<size_t>(len, sizeof(args)));
                command(id, args, len, returns[i]);
                pos += 4u + ((len + 3u) & ~3u);
            }
            if (recv && recvSize >= 8u)
            {
                const uint32_t sentinel = 0xFFFFFFFFu;
                std::memcpy(recv, &sentinel, 4u);
                for (uint32_t i = 0; i < count && 4u + (i + 1u) * 4u <= recvSize; ++i)
                    std::memcpy(recv + 4u + i * 4u, &returns[i], 4u);
                if (4u + count * 4u + 4u <= recvSize)
                    std::memcpy(recv + 4u + count * 4u, &sentinel, 4u);
            }
        }

        void mix(int16_t *out, uint32_t frames)
        {
            std::fill(out, out + frames * 2u, int16_t{0});
            std::vector<int32_t> acc(frames * 2u, 0);
            Snd989Sfx::render(acc.data(), frames); // sound-effect voices (own lock)
            std::lock_guard<std::mutex> lock(m_mutex);
            for (auto &s : m_streams)
            {
                if (s->finished || s->paused)
                    continue;
                const float gain = s->volume * groupVolume(s->group);
                const double step = static_cast<double>(s->rate) / kOutputRate;
                for (uint32_t f = 0; f < frames; ++f)
                {
                    while (s->pcmFrames == 0u || s->position >= static_cast<double>(s->pcmFrames))
                    {
                        const double carry = s->pcmFrames ? s->position - static_cast<double>(s->pcmFrames) : 0.0;
                        if (!s->decodeNextBlock())
                        {
                            s->finished = true;
                            break;
                        }
                        s->position = carry;
                    }
                    if (s->finished)
                        break;
                    const size_t i0 = static_cast<size_t>(s->position);
                    const double frac = s->position - static_cast<double>(i0);
                    const size_t i1 = std::min(i0 + 1u, s->pcmFrames - 1u);
                    const double l = s->pcm[i0 * 2u] + (s->pcm[i1 * 2u] - s->pcm[i0 * 2u]) * frac;
                    const double r = s->pcm[i0 * 2u + 1u] + (s->pcm[i1 * 2u + 1u] - s->pcm[i0 * 2u + 1u]) * frac;
                    acc[f * 2u] += static_cast<int32_t>(l * gain);
                    acc[f * 2u + 1u] += static_cast<int32_t>(r * gain);
                    s->position += step;
                }
            }
            for (uint32_t i = 0; i < frames * 2u; ++i)
                out[i] = static_cast<int16_t>(std::clamp(acc[i], -32768, 32767));
            // PS2X_SND_DUMP=<file>: raw 48 kHz stereo s16 of everything mixed (diagnostic).
            static std::FILE *s_dump = [] { const char *p = std::getenv("PS2X_SND_DUMP"); return p ? std::fopen(p, "wb") : nullptr; }();
            if (s_dump)
            {
                std::fwrite(out, sizeof(int16_t), frames * 2u, s_dump);
                std::fflush(s_dump);
            }
        }

    private:
        std::mutex m_mutex;
        std::vector<std::unique_ptr<VagStream>> m_streams;
        std::array<float, 32> m_groupVolume = [] { std::array<float, 32> a; a.fill(1.0f); return a; }();
        uint32_t m_nextHandle = 0x00C00001u;
        std::atomic<bool> m_outputStarted{false};

        float groupVolume(uint32_t group) const { return group < 32u ? m_groupVolume[group] : 1.0f; }

        bool isStreamHandle(uint32_t handle) const { return handle >= 0x00C00001u && handle < m_nextHandle; }

        VagStream *find(uint32_t handle)
        {
            for (auto &s : m_streams)
                if (s->handle == handle)
                    return s.get();
            return nullptr;
        }

    public:
        void startOutput();

    private:

        void command(uint32_t id, const uint32_t *a, uint16_t len, uint32_t &ret)
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            bool ensureOutput = false;
            // PS2X_SND_TRACE=1: log stream/volume/group commands (diagnostic).
            static const bool s_trace = [] { const char *p = std::getenv("PS2X_SND_TRACE"); return p && p[0] == '1'; }();
            if (s_trace && id != kCmdIsVagStreamBuffered && id != 0x11u && id != 0x12u && id != kCmdSetSoundParams && id != kCmdSoundIsStillPlaying)
            {
                static uint32_t s_lines = 0u;
                if (++s_lines < 400u)
                    std::fprintf(stderr, "[snd989:trace] id=0x%02x len=%u %08x %08x %08x %08x %08x %08x streams=%zu\n", id, len,
                                 a[0], a[1], a[2], a[3], a[4], a[5], m_streams.size());
            }
            // Finished streams are dropped once the game has seen them end.
            switch (id)
            {
            case kCmdInitVagStreaming:
                ret = 1u;
                break;
            case 0x0E: // SetReverbType(core mask, type)
                if (len >= 8u) ps2x_snd989_set_reverb_type(a[0],a[1]);
                break;
            case 0x0F: // SetReverbDepth(core mask, left, right)
                if (len >= 12u) ps2x_snd989_set_reverb_depth(a[0],static_cast<int32_t>(a[1]),static_cast<int32_t>(a[2]));
                break;
            case kCmdPlayVagStreamByLoc:
            {
                auto s = std::make_unique<VagStream>();
                s->handle = m_nextHandle++;
                s->loc = a[0];
                s->volume = static_cast<float>((a[2] >> 16) & 0xFFFFu) / 1024.0f;
                s->group = a[4];
                const std::string iso = PS2Runtime::getIoPaths().cdImage.string();
                if (!s->open(iso))
                {
                    std::fprintf(stderr, "[snd989] cannot open VAG stream at LBA %u (%s)\n", a[0], iso.c_str());
                    s->finished = true;
                }
                else
                {
                    std::fprintf(stderr, "[snd989] stream %06x: LBA %u rate %u %u ch, %.1f s, vol %.2f group %u\n",
                                 s->handle, s->loc, s->rate, s->channels,
                                 static_cast<double>(s->bytesPerChannel) / 16.0 * 28.0 / s->rate, s->volume, s->group);
                }
                ret = s->handle;
                m_streams.push_back(std::move(s));
                ensureOutput = true;
                break;
            }
            case kCmdStopVagStream:
                m_streams.erase(std::remove_if(m_streams.begin(), m_streams.end(),
                                               [h = a[0]](const auto &s) { return s->handle == h; }),
                                m_streams.end());
                break;
            case kCmdStopAllStreams:
                m_streams.clear();
                break;
            case kCmdPauseVagStream:
                if (VagStream *s = find(a[0]))
                    s->paused = true;
                break;
            case kCmdContinueVagStream:
                if (VagStream *s = find(a[0]))
                    s->paused = false;
                ensureOutput = true;
                break;
            case 0x06: // UnloadBank(bank)
                Snd989Sfx::unloadBank(a[0]);
                break;
            case 0x11: // PlaySoundVolPanPMPB(bank, sound, vol, pan, pm, pb) -> handle
            case 0x12: // ..._A (asynchronous: no return is read)
                ret = Snd989Sfx::play(a[0], static_cast<int32_t>(a[1]), static_cast<int32_t>(a[2]), static_cast<int32_t>(a[3]),
                                      static_cast<int32_t>(a[4]), static_cast<int32_t>(a[5]));
                {
                    static uint64_t s_plays = 0u, s_failed = 0u;
                    ++s_plays;
                    s_failed += ret == 0u ? 1u : 0u;
                    static const bool s_sfxTrace = [] { const char *p = std::getenv("PS2X_SND_TRACE"); return p && p[0] == '1'; }();
                    if (s_sfxTrace && (s_plays % 50u) == 0u)
                        std::fprintf(stderr, "[snd989:sfx] plays=%llu failed=%llu active=%zu\n", (unsigned long long)s_plays,
                                     (unsigned long long)s_failed, Snd989Sfx::activeSounds());
                }
                ensureOutput = true;
                break;
            case 0x13: // PauseSound(handle)
            case 0x14: // ContinueSound(handle)
                Snd989Sfx::pause(a[0], id == 0x13);
                if (id == 0x14)
                    ensureOutput = true;
                break;
            case 0x15: // StopSound(handle)
                Snd989Sfx::stop(a[0]);
                break;
            case 0x18: // StopAllSounds
                Snd989Sfx::stopAll();
                break;
            case 0x1A: // IsSoundALooper(bank, sound)
                ret = Snd989Sfx::isLooper(a[0], static_cast<int32_t>(a[1])) ? 1u : 0u;
                break;
            case 0x1F: // SetSoundPitchBend(handle, pb)
                Snd989Sfx::setPitchBend(a[0], static_cast<int32_t>(a[1]));
                break;
            case 0x20: // SetSoundPitchModifier(handle, pm)
                Snd989Sfx::setPitchMod(a[0], static_cast<int32_t>(a[1]));
                break;
            case 0x29: // SetMIDIRegister(handle, reg, value)
                Snd989Sfx::setRegister(a[0], a[1], static_cast<int32_t>(a[2]));
                break;
            case kCmdSoundIsStillPlaying:
                if (Snd989Sfx::isHandle(a[0]))
                    ret = Snd989Sfx::stillPlaying(a[0]) ? a[0] : 0u;
                // A stream handle that was stopped (snd_StopAllStreams at the end
                // of a cinematic) or ran out reports 0: that is what makes the
                // music manager start the level music again.
                if (isStreamHandle(a[0]))
                    ret = 0u;
                if (VagStream *s = find(a[0]))
                {
                    ret = s->finished ? 0u : s->handle;
                    if (s->finished)
                        m_streams.erase(std::remove_if(m_streams.begin(), m_streams.end(),
                                                       [h = a[0]](const auto &p) { return p->handle == h; }),
                                        m_streams.end());
                }
                break;
            case kCmdIsVagStreamBuffered:
                if (isStreamHandle(a[0]))
                    ret = find(a[0]) ? 1u : 0u;
                break;
            case kCmdGetVagStreamLoc:
                if (VagStream *s = find(a[0]))
                    ret = s->loc;
                break;
            case kCmdSetSoundParams: // handle, mask (1 = volume), volume, pan, pitch mod, pitch bend
                if (Snd989Sfx::isHandle(a[0]))
                {
                    if (a[1] & 1u)
                        Snd989Sfx::setVolPan(a[0], static_cast<int32_t>(a[2]), -2); // PAN_DONT_CHANGE
                    if (a[1] & 2u)
                        Snd989Sfx::setVolPan(a[0], 0x7fffffff, static_cast<int32_t>(a[3])); // VOLUME_DONT_CHANGE
                    if (a[1] & 4u)
                        Snd989Sfx::setPitchMod(a[0], static_cast<int32_t>(a[4]));
                    if (a[1] & 8u)
                        Snd989Sfx::setPitchBend(a[0], static_cast<int32_t>(a[5]));
                }
                if (VagStream *s = find(a[0]))
                {
                    if (a[1] & 1u)
                        s->volume = static_cast<float>(a[2] & 0xFFFFu) / 1024.0f;
                }
                break;
            case kCmdSetSoundVolPan: // handle, volume, pan
                if (Snd989Sfx::isHandle(a[0]))
                    Snd989Sfx::setVolPan(a[0], static_cast<int32_t>(a[1]), static_cast<int32_t>(a[2]));
                if (VagStream *s = find(a[0]))
                    s->volume = static_cast<float>(a[1] & 0xFFFFu) / 1024.0f;
                break;
            case kCmdSetMasterVolume: // group, volume (0x400 = full)
                if (a[0] < 32u)
                    m_groupVolume[a[0]] = static_cast<float>(a[1] & 0xFFFFu) / 1024.0f;
                Snd989Sfx::setMasterVolume(a[0], static_cast<int32_t>(a[1]));
                break;
            case kCmdGetMasterVolume:
                ret = a[0] < m_groupVolume.size()
                    ? static_cast<uint32_t>(m_groupVolume[a[0]] * 1024.0f) : 0u;
                break;
            // Group pause/continue act on the sounds playing right now (a sound
            // started later in a "paused" group plays), as snd_Pause/Continue-
            // AllSoundsInGroup do in 989SND.
            case kCmdPauseGroup:
                Snd989Sfx::pauseGroups(a[0], true);
                for (auto &s : m_streams)
                    if (s->group < 32u && ((a[0] >> s->group) & 1u))
                        s->paused = true;
                break;
            case kCmdContinueGroup:
                Snd989Sfx::pauseGroups(a[0], false);
                for (auto &s : m_streams)
                    if (s->group < 32u && ((a[0] >> s->group) & 1u))
                        s->paused = false;
                ensureOutput = true;
                break;
            default:
                break;
            }
            // The callback holds raylib's audio lock before entering mix().
            // Never acquire that lock while holding m_mutex: opposite ordering
            // deadlocks sound RPCs against the callback during active fights.
            lock.unlock();
            if (ensureOutput)
                startOutput();
        }
    };

    Snd989Player &player()
    {
        static Snd989Player s_player;
        return s_player;
    }

#if !defined(PLATFORM_VITA)
    void audioCallback(void *buffer, unsigned int frames)
    {
        player().mix(static_cast<int16_t *>(buffer), frames);
    }
#endif

    void Snd989Player::startOutput()
    {
#if !defined(PLATFORM_VITA)
        if (!IsAudioDeviceReady())
        {
            std::fprintf(stderr, "[snd989] no audio device: streams are tracked but silent\n");
            m_outputStarted.store(false, std::memory_order_relaxed);
            return;
        }
        static AudioStream stream{};
        static std::once_flag initialized;
        std::call_once(initialized, [&]
        {
            stream = LoadAudioStream(kOutputRate, 16, 2);
            SetAudioStreamCallback(stream, &audioCallback);
        });
        // A long host or game pause can leave miniaudio's stream stopped. The
        // old one-shot guard made that permanent because resume commands could
        // never reach PlayAudioStream again.
        if (!IsAudioStreamPlaying(stream))
            PlayAudioStream(stream);
        m_outputStarted.store(true, std::memory_order_relaxed);
#endif
    }
}

bool ps2xSnd989Enabled()
{
    static const bool enabled = [] { const char *p = std::getenv("PS2X_SND989"); return !p || p[0] != '0'; }();
    return enabled;
}

void ps2xSnd989HandleRpc(uint32_t sid, uint32_t function, const uint8_t *send, uint32_t sendSize, uint8_t *recv, uint32_t recvSize)
{
    if (!ps2xSnd989Enabled())
        return;
    if (sid == 0x123457u)
    {
        // snd_BankLoadByLoc: send {disc LBA, ...}, receive the 4-byte bank handle.
        if (!send || sendSize < 4u || !recv || recvSize < 4u)
            return;
        uint32_t lba = 0u;
        std::memcpy(&lba, send, 4u);
        if (const uint32_t handle = Snd989Sfx::loadBank(PS2Runtime::getIoPaths().cdImage.string(), lba))
            std::memcpy(recv, &handle, 4u);
        player().startOutput();
        return;
    }
    if (sid != kSnd989Sid)
        return;
    player().handleRpc(function, send, sendSize, recv, recvSize);
}
