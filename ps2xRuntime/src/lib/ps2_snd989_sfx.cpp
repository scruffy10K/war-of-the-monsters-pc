#include "installed_disc_api.inc"
// 989SND sound-effect banks for War of the Monsters.
//
// Banks (/SND/*.BNK) are 989 "SBlk" version 1 SFX blocks. Sound playback --
// grain interpreter, voice allocation, ADPCM voices with ADSR, pitch, LFOs --
// is the OpenGOAL 989snd engine vendored in third_party/opengoal_989snd (ISC).
// This file loads banks from the disc image, owns the engine state and exposes
// the operations the 989SND command protocol needs (ps2_snd989.cpp).
#include "ps2_snd989_sfx.h"
#include "ps2_runtime.h"

#include "game/sound/989snd/blocksound_handler.h"
#include "game/sound/989snd/sfxblock.h"
#include "game/sound/common/synth.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace snd
{
void SetReverbPreset(Synth& synth, const u16* parameters, u32 words);
void SetReverbDepth(Synth& synth, s16 left, s16 right);
namespace
{
    // Sequential little-endian reader over a byte span (OpenGOAL BinaryReader).
    struct Reader
    {
        std::span<u8> data;
        size_t pos = 0u;
        template <typename T> T read()
        {
            T value{};
            if (pos + sizeof(T) <= data.size())
                std::memcpy(&value, data.data() + pos, sizeof(T));
            pos += sizeof(T);
            return value;
        }
    };

    Tone readTone(Reader &data, u8 *samples, size_t sampleSize)
    {
        Tone tone{};
        tone.Priority = data.read<s8>();
        tone.Vol = data.read<s8>();
        tone.CenterNote = data.read<s8>();
        tone.CenterFine = data.read<s8>();
        tone.Pan = data.read<s16>();
        tone.MapLow = data.read<s8>();
        tone.MapHigh = data.read<s8>();
        tone.PBLow = data.read<s8>();
        tone.PBHigh = data.read<s8>();
        tone.ADSR1 = data.read<u16>();
        tone.ADSR2 = data.read<u16>();
        tone.Flags = data.read<u16>();
        u32 sampleOffset = data.read<u32>();
        if (sampleOffset >= sampleSize)
            sampleOffset = 0u;
        tone.Sample = samples + sampleOffset;
        data.read<u32>(); // reserved1
        return tone;
    }

    // OpenGOAL loader.cpp ReadGrainV1.
    Grain readGrainV1(Reader &data, u8 *samples, size_t sampleSize)
    {
        Grain grain{};
        const size_t start = data.pos;
        grain.Type = static_cast<GrainType>(data.read<u32>());
        grain.Delay = data.read<s32>();
        switch (grain.Type)
        {
        case GrainType::TONE:
        case GrainType::TONE2:
            grain.data = readTone(data, samples, sampleSize);
            break;
        case GrainType::LFO_SETTINGS:
            grain.data = data.read<LFOParams>();
            break;
        case GrainType::BRANCH:
        case GrainType::STARTCHILDSOUND:
        case GrainType::STOPCHILDSOUND:
            grain.data = data.read<PlaySoundParams>();
            break;
        case GrainType::PLUGIN_MESSAGE:
            grain.data = data.read<PluginParams>();
            break;
        case GrainType::RAND_DELAY:
            grain.data = data.read<RandDelayParams>();
            break;
        default:
            grain.data = data.read<ControlParams>();
            break;
        }
        data.pos = start + 0x28;
        return grain;
    }
}

// PLUGIN_MESSAGE grains are Jak-specific (OpenGOAL plugin.cpp); no War of the
// Monsters bank contains one.
s32 HandlePluginMessage(u32, u32, u32, const u8 *, const s8 *, s32)
{
    return 0;
}

// OpenGOAL loader.cpp SFXBlock::ReadBlock, version 1 blocks.
SFXBlock *SFXBlock::ReadBlock(std::span<u8> bank_data, std::span<u8> samples)
{
    Reader data{bank_data};
    auto block = std::make_unique<SFXBlock>();
    block->DataID = data.read<u32>();
    if (std::memcmp(bank_data.data(), "SBlk", 4) != 0)
        return nullptr;
    block->SampleData = std::make_unique<u8[]>(samples.size() + 16u);
    std::memcpy(block->SampleData.get(), samples.data(), samples.size());
    std::memset(block->SampleData.get() + samples.size(), 0, 16u);
    block->Version = data.read<u32>();
    if (block->Version != 1u)
        return nullptr;
    block->Flags.flags = data.read<u32>();
    block->BankID = data.read<u32>();
    block->BankNum = data.read<s8>();
    data.read<s8>();
    data.read<s16>();
    data.read<s16>();
    const s16 numSounds = data.read<s16>();
    data.read<s16>(); // NumGrains
    data.read<s16>(); // NumVAGs
    const u32 firstSound = data.read<u32>();
    const u32 firstGrain = data.read<u32>();
    if (numSounds < 0 || firstSound + static_cast<u64>(numSounds) * 12u > bank_data.size())
        return nullptr;
    block->Sounds.resize(static_cast<size_t>(numSounds));
    data.pos = firstSound;
    for (auto &sfx : block->Sounds)
    {
        sfx.Vol = data.read<s8>();
        sfx.VolGroup = data.read<s8>();
        sfx.Pan = data.read<s16>();
        s8 numGrains = data.read<s8>();
        if (numGrains < 0)
            numGrains = 0;
        sfx.InstanceLimit = data.read<s8>();
        sfx.Flags.flags = data.read<u16>();
        const u32 firstSfxGrain = data.read<u32>();
        sfx.Grains.resize(static_cast<size_t>(numGrains));
        Reader grains{bank_data, static_cast<size_t>(firstGrain) + firstSfxGrain};
        for (auto &grain : sfx.Grains)
            grain = readGrainV1(grains, block->SampleData.get(), samples.size());
    }
    return block.release();
}
}

namespace
{
    constexpr u32 kSectorSize = 2048u;
    constexpr u32 kFirstSfxHandle = 0x00D00001u;
    constexpr u32 kFirstBankHandle = 0x00B00100u;

    // Only the fixed sound-driver file is read, from the installed game or
    // ISO9660 directory. The game's preset data stays in memory, like textures.
    std::vector<u8> readEffectModule()
    {
        const auto paths = PS2Runtime::getIoPaths();
        std::ifstream installed(paths.cdRoot / "MOD" / "LIBSD.IRX",std::ios::binary | std::ios::ate);
        if (installed) {
            const auto bytes = installed.tellg();
            if (bytes > 0 && bytes <= 1024*1024) {
                std::vector<u8> data(static_cast<size_t>(bytes));
                installed.seekg(0);
                if (installed.read(reinterpret_cast<char*>(data.data()),bytes)) return data;
            }
        }
        std::unique_ptr<std::FILE,decltype(&std::fclose)> image(std::fopen(paths.cdImage.string().c_str(),"rb"),std::fclose);
        if (!image) return {};
        auto range = [&](u32 lba, u32 bytes) {
            std::vector<u8> data;
            if (!bytes || bytes > 1024*1024 || _fseeki64(image.get(),static_cast<s64>(lba)*2048,SEEK_SET)) return data;
            data.resize(bytes);
            if (std::fread(data.data(),1,bytes,image.get()) != bytes) data.clear();
            return data;
        };
        auto word = [](const u8* p) { u32 value; std::memcpy(&value,p,4); return value; };
        const auto descriptor = range(16,2048);
        if (descriptor.size() != 2048 || descriptor[0] != 1 || std::memcmp(descriptor.data()+1,"CD001",5)) return {};
        u32 lba = word(descriptor.data()+158), bytes = word(descriptor.data()+166);
        for (const std::string name : {"MOD","LIBSD.IRX"}) {
            if (bytes > 256*1024) return {};
            const auto directory = range(lba,bytes);
            bool found = false;
            for (size_t offset=0;offset<directory.size();) {
                const size_t length=directory[offset];
                if (!length) { offset=(offset/2048+1)*2048; continue; }
                if (length < 34 || length > directory.size()-offset) return {};
                const auto* record = directory.data()+offset;
                if (record[32] > length-33) return {};
                std::string identifier(reinterpret_cast<const char*>(record+33),record[32]);
                identifier = identifier.substr(0,identifier.find(';'));
                if (identifier == name) {
                    if (((record[25]&2) != 0) != (name == "MOD")) return {};
                    lba=word(record+2); bytes=word(record+10); found=true; break;
                }
                offset += length;
            }
            if (!found) return {};
        }
        return range(lba,bytes);
    }

    struct Engine
    {
        std::mutex mutex;
        snd::Synth synth;
        snd::VoiceManager voices{synth};
        std::map<u32, std::unique_ptr<snd::SFXBlock>> banks;
        std::map<u32, std::unique_ptr<snd::SoundHandler>> handlers;
        u32 nextHandle = kFirstSfxHandle;
        u32 nextBank = kFirstBankHandle;
        s32 tick = 0;
        int subTick = 200;
        std::array<std::array<u16,32>,10> effectPresets{};
        bool effectLoadAttempted = false, effectPresetsReady = false;
        u32 effectType = 0;
    };

    void applyEffectType(Engine& e)
    {
        // Documented work-area sizes in eight 16-bit words (SDK allocation units).
        static constexpr u16 sizes[10] = {1,0x26c,0x1f4,0x484,0x6fe,0xade,0xf6c,0x1804,0x1804,0x3c0};
        snd::SetReverbPreset(e.synth,e.effectPresets[e.effectType].data(),
                            e.effectType && e.effectPresetsReady ? sizes[e.effectType]*8u : 0u);
    }

    void loadEffectPresets(Engine& e)
    {
        if (e.effectLoadAttempted) return;
        e.effectLoadAttempted = true;
        const auto module = readEffectModule();
        if (module.size() < 680 || std::memcmp(module.data(),"\x7f" "ELF",4)) {
            std::fprintf(stderr,"[snd989:reverb] original LIBSD module unavailable; effects remain dry\n");
            return;
        }
        const u16 roomSignature[8] = {0x7d,0x5b,0x6d80,0x54b8,0xbed0,0,0,0xba80};
        for (size_t position=68;position+sizeof(roomSignature)<=module.size();position+=2) {
            if (std::memcmp(module.data()+position,roomSignature,sizeof(roomSignature))) continue;
            for (size_t stride : {68u,64u}) {
                if (position < stride || position+stride*8+64 > module.size()) continue;
                auto presets = e.effectPresets;
                bool valid = true;
                for (size_t mode=0;mode<10;++mode) {
                    std::memcpy(presets[mode].data(),module.data()+position-stride+mode*stride,64);
                    if (mode && (presets[mode][30] != 0x8000 || presets[mode][31] != 0x8000)) valid=false;
                }
                if (!valid) continue;
                e.effectPresets = presets; e.effectPresetsReady=true;
                applyEffectType(e);
                std::fprintf(stderr,"[snd989:reverb] loaded 10 presets from player LIBSD (%zu-byte records)\n",stride);
                return;
            }
        }
        std::fprintf(stderr,"[snd989:reverb] original LIBSD presets unavailable; effects remain dry\n");
    }

    Engine &engine()
    {
        static Engine s_engine;
        return s_engine;
    }
}

// Follow War of the Monsters' SFX reverb commands for core 1 (mask bit 2).
// Keep this host extension out of shared generated-code headers.
void ps2x_snd989_set_reverb_type(u32 coreMask, u32 type)
{
    if (!(coreMask&2u) || (type&~0x100u) > 9u) return;
    Engine& e=engine(); std::lock_guard<std::mutex> lock(e.mutex);
    loadEffectPresets(e);
    e.effectType=type&~0x100u; applyEffectType(e);
}
void ps2x_snd989_set_reverb_depth(u32 coreMask, s32 left, s32 right)
{
    if (!(coreMask&2u)) return;
    Engine& e=engine(); std::lock_guard<std::mutex> lock(e.mutex);
    snd::SetReverbDepth(e.synth,static_cast<s16>(left),static_cast<s16>(right));
}

u32 Snd989Sfx::loadBank(const std::string &isoPath, u32 lba)
{
    std::FILE *file = wotm_disc::open(std::filesystem::path(isoPath));
    if (!file)
        return 0u;
    u8 header[0x18]{};
    std::vector<u8> bytes;
    size_t got = 0;
    bool ok = wotm_disc::read(file, static_cast<u64>(lba) * kSectorSize, header, sizeof(header), got) && got == sizeof(header);
    u32 type = 0u, chunks = 0u, off0 = 0u, size0 = 0u, off1 = 0u, size1 = 0u;
    if (ok)
    {
        std::memcpy(&type, header, 4);
        std::memcpy(&chunks, header + 4, 4);
        std::memcpy(&off0, header + 8, 4);
        std::memcpy(&size0, header + 12, 4);
        std::memcpy(&off1, header + 16, 4);
        std::memcpy(&size1, header + 20, 4);
        ok = chunks >= 2u && off0 + static_cast<u64>(size0) <= 64u * 1024u * 1024u &&
             off1 + static_cast<u64>(size1) <= 64u * 1024u * 1024u;
    }
    if (ok)
    {
        bytes.resize(std::max(off0 + size0, off1 + size1));
        ok = wotm_disc::read(file, static_cast<u64>(lba) * kSectorSize, bytes.data(), bytes.size(), got) && got == bytes.size();
    }
    wotm_disc::close(file);
    if (!ok)
    {
        std::fprintf(stderr, "[snd989:sfx] bank at LBA %u: unreadable header\n", lba);
        return 0u;
    }
    std::unique_ptr<snd::SFXBlock> block(snd::SFXBlock::ReadBlock(std::span<u8>(bytes.data() + off0, size0),
                                                                  std::span<u8>(bytes.data() + off1, size1)));
    if (!block)
    {
        std::fprintf(stderr, "[snd989:sfx] bank at LBA %u: not an SBlk v1 block\n", lba);
        return 0u;
    }
    Engine &e = engine();
    std::lock_guard<std::mutex> lock(e.mutex);
    const u32 handle = e.nextBank;
    e.nextBank += 0x100u;
    char id[5]{};
    std::memcpy(id, &block->BankID, 4);
    std::fprintf(stderr, "[snd989:sfx] bank %06x: LBA %u '%c%c%c%c' %zu sounds, %u KB samples\n", handle, lba, id[3], id[2],
                 id[1], id[0], block->Sounds.size(), size1 / 1024u);
    e.banks[handle] = std::move(block);
    return handle;
}

void Snd989Sfx::unloadBank(u32 bank)
{
    Engine &e = engine();
    std::lock_guard<std::mutex> lock(e.mutex);
    auto it = e.banks.find(bank);
    if (it == e.banks.end())
        return;
    for (auto h = e.handlers.begin(); h != e.handlers.end();)
    {
        if (&h->second->Bank() == it->second.get())
            h = e.handlers.erase(h);
        else
            ++h;
    }
    e.banks.erase(it);
}

u32 Snd989Sfx::play(u32 bank, s32 sound, s32 vol, s32 pan, s32 pm, s32 pb)
{
    Engine &e = engine();
    std::lock_guard<std::mutex> lock(e.mutex);
    auto it = e.banks.find(bank);
    static const bool s_trace = [] { const char *p = std::getenv("PS2X_SND_TRACE"); return p && p[0] == '1'; }();
    static u32 s_failLines = 0u;
    if (it == e.banks.end() || sound < 0 || static_cast<size_t>(sound) >= it->second->Sounds.size())
    {
        if (s_trace && s_failLines++ < 30u)
            std::fprintf(stderr, "[snd989:sfx] play fail: bank %06x %s sound %d vol %d pan %d\n", bank,
                         it == e.banks.end() ? "unknown" : "known", sound, vol, pan);
        return 0u;
    }
    if (it->second->Sounds[static_cast<size_t>(sound)].Grains.empty())
    {
        if (s_trace && s_failLines++ < 30u)
            std::fprintf(stderr, "[snd989:sfx] play fail: bank %06x sound %d has no grains\n", bank, sound);
        return 0u;
    }
    const u32 handle = e.nextHandle++;
    if (e.nextHandle >= 0x00E00000u)
        e.nextHandle = kFirstSfxHandle;
    auto handler = it->second->snd::SoundBank::MakeHandler(e.voices, static_cast<u32>(sound), vol, pan, pm, pb, e.tick, handle);
    if (!handler.has_value())
        return 0u;
    if (snd::SoundHandler *toStop = handler.value()->CheckInstanceLimit(e.handlers, vol, true))
    {
        toStop->Stop();
        if (toStop == handler.value().get())
        {
            if (s_trace && s_failLines++ < 30u)
                std::fprintf(stderr, "[snd989:sfx] play fail: bank %06x sound %d instance limit\n", bank, sound);
            return 0u;
        }
    }
    handler.value()->m_sound_handle = handle;
    e.handlers.emplace(handle, std::move(handler.value()));
    return handle;
}

bool Snd989Sfx::isHandle(u32 handle) { return handle >= kFirstSfxHandle && handle < 0x00E00000u; }

bool Snd989Sfx::stillPlaying(u32 handle)
{
    Engine &e = engine();
    std::lock_guard<std::mutex> lock(e.mutex);
    return e.handlers.count(handle) != 0u;
}

bool Snd989Sfx::isLooper(u32 bank, s32 sound)
{
    Engine &e = engine();
    std::lock_guard<std::mutex> lock(e.mutex);
    auto it = e.banks.find(bank);
    return it != e.banks.end() && sound >= 0 && static_cast<size_t>(sound) < it->second->Sounds.size() &&
           it->second->Sounds[static_cast<size_t>(sound)].Flags.has_loop();
}

void Snd989Sfx::stop(u32 handle)
{
    Engine &e = engine();
    std::lock_guard<std::mutex> lock(e.mutex);
    if (auto it = e.handlers.find(handle); it != e.handlers.end())
        it->second->Stop();
}

void Snd989Sfx::stopAll()
{
    Engine &e = engine();
    std::lock_guard<std::mutex> lock(e.mutex);
    e.handlers.clear();
}

void Snd989Sfx::pause(u32 handle, bool paused)
{
    Engine &e = engine();
    std::lock_guard<std::mutex> lock(e.mutex);
    if (auto it = e.handlers.find(handle); it != e.handlers.end())
    {
        if (paused)
            it->second->Pause();
        else
            it->second->Unpause();
    }
}

void Snd989Sfx::pauseGroups(u32 mask, bool paused)
{
    Engine &e = engine();
    std::lock_guard<std::mutex> lock(e.mutex);
    for (auto &[handle, h] : e.handlers)
    {
        if (h->Group() < 32u && ((mask >> h->Group()) & 1u))
        {
            if (paused)
                h->Pause();
            else
                h->Unpause();
        }
    }
}

void Snd989Sfx::setVolPan(u32 handle, s32 vol, s32 pan)
{
    Engine &e = engine();
    std::lock_guard<std::mutex> lock(e.mutex);
    if (auto it = e.handlers.find(handle); it != e.handlers.end())
        it->second->SetVolPan(vol, pan);
}

void Snd989Sfx::setPitchMod(u32 handle, s32 pm)
{
    Engine &e = engine();
    std::lock_guard<std::mutex> lock(e.mutex);
    if (auto it = e.handlers.find(handle); it != e.handlers.end())
        it->second->SetPMod(pm);
}

void Snd989Sfx::setPitchBend(u32 handle, s32 pb)
{
    Engine &e = engine();
    std::lock_guard<std::mutex> lock(e.mutex);
    if (auto it = e.handlers.find(handle); it != e.handlers.end())
        it->second->SetPBend(pb);
}

void Snd989Sfx::setRegister(u32 handle, u32 reg, s32 value)
{
    Engine &e = engine();
    std::lock_guard<std::mutex> lock(e.mutex);
    if (auto it = e.handlers.find(handle); it != e.handlers.end() && reg < 4u)
        it->second->SetRegister(static_cast<u8>(reg), static_cast<u8>(value));
}

void Snd989Sfx::setMasterVolume(u32 group, s32 volume)
{
    Engine &e = engine();
    std::lock_guard<std::mutex> lock(e.mutex);
    volume = std::clamp(volume, 0, 0x400);
    if (group == 15u || group >= 32u)
        return;
    e.voices.SetMasterVol(static_cast<u8>(group), volume);
    if (group == 16u)
        e.synth.SetMasterVol(0x3fff * volume / 0x400);
}

size_t Snd989Sfx::activeSounds()
{
    Engine &e = engine();
    std::lock_guard<std::mutex> lock(e.mutex);
    return e.handlers.size();
}

// Mixes frames of the engine into acc (interleaved stereo); the handler tick
// runs every 200 output samples (240 Hz), as in OpenGOAL's Player::Tick.
void Snd989Sfx::render(int32_t *acc, uint32_t frames)
{
    Engine &e = engine();
    std::lock_guard<std::mutex> lock(e.mutex);
    for (uint32_t i = 0; i < frames; ++i)
    {
        if (e.subTick == 200)
        {
            ++e.tick;
            for (auto it = e.handlers.begin(); it != e.handlers.end();)
            {
                if (it->second->Tick())
                    it = e.handlers.erase(it);
                else
                    ++it;
            }
            e.subTick = 0;
        }
        ++e.subTick;
        const snd::s16Output out = e.synth.Tick();
        acc[i * 2u] += out.left;
        acc[i * 2u + 1u] += out.right;
    }
}
