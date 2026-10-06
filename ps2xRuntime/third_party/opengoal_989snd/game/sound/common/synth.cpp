// Copyright: 2021 - 2024, Ziemas
// SPDX-License-Identifier: ISC
#include "synth.h"

#include <stdexcept>

namespace snd {

namespace {
// Native implementation of the documented SPU reflection/comb/all-pass
// network and 39-tap half-band resampling. Presets come from the player's
// LIBSD module; no game effect data is compiled into the port.
// https://psx-spx.consoledev.net/ps1/spu/soundprocessingunitspu/
struct Reverb {
  Synth* owner = nullptr;
  std::array<u16, 32> parameters{};
  std::vector<s16> memory;
  std::array<std::array<s16, 64>, 2> input{}, output{};
  u32 cursor = 0, phase = 0;
  s16 depth[2]{};

  static s32 multiply(s32 value, s16 gain) {
    return static_cast<s32>((static_cast<s64>(value) * gain) >> 15);
  }
  static s16 saturate(s64 value) {
    return static_cast<s16>(std::clamp<s64>(value, INT16_MIN, INT16_MAX));
  }
  s16 read(s32 offset) const {
    s64 index = (static_cast<s64>(cursor) + offset) % static_cast<s64>(memory.size());
    if (index < 0) index += memory.size();
    return memory[static_cast<size_t>(index)];
  }
  void write(s32 offset, s32 value) {
    s64 index = (static_cast<s64>(cursor) + offset) % static_cast<s64>(memory.size());
    if (index < 0) index += memory.size();
    memory[static_cast<size_t>(index)] = saturate(value);
  }
  s32 tap(u32 index) const { return static_cast<s32>(parameters[index]) * 4; }
  s16 gain(u32 index) const { return static_cast<s16>(parameters[index]); }
  static s32 filter(const std::array<s16,64>& history, u32 position, bool upsample) {
    // The nonzero half-band coefficients, including the central 0.5 tap.
    static constexpr s16 coefficients[39] = {
      -1,0,2,0,-10,0,35,0,-103,0,266,0,-616,0,1332,0,-2960,0,10246,
      16384,10246,0,-2960,0,1332,0,-616,0,266,0,-103,0,35,0,-10,0,2,0,-1};
    s64 sum = 0;
    for (u32 i=0;i<39;++i) {
      if (coefficients[i]) sum += static_cast<s32>(history[(position-i)&63]) * coefficients[i];
    }
    return saturate((sum * (upsample ? 2 : 1)) >> 15);
  }
  s16Output tick(s64 left, s64 right) {
    if (memory.empty()) return {};
    input[0][phase] = saturate(left); input[1][phase] = saturate(right);
    const u32 channel = phase & 1;
    const s32 in = multiply(filter(input[channel],phase,false),gain(30+channel));
    const s32 samePrevious = read(tap(10+channel)-1);
    const s32 diffPrevious = read(tap(18+channel)-1);
    const s32 same = samePrevious + multiply(in + multiply(read(tap(16+channel)),gain(7)) - samePrevious,gain(2));
    const s32 diff = diffPrevious + multiply(in + multiply(read(tap(24+(channel^1))),gain(7)) - diffPrevious,gain(2));
    s32 value = 0;
    const u32 comb[4] = {12,14,20,22};
    for (u32 i=0;i<4;++i) value += multiply(read(tap(comb[i]+channel)),gain(3+i));
    const s32 apf1Previous = read(tap(26+channel)-tap(0));
    const s32 apf1 = value-multiply(apf1Previous,gain(8));
    value = apf1Previous + multiply(apf1,gain(8));
    const s32 apf2Previous = read(tap(28+channel)-tap(1));
    const s32 apf2 = value-multiply(apf2Previous,gain(9));
    value = apf2Previous + multiply(apf2,gain(9));
    write(tap(10+channel),same); write(tap(18+channel),diff);
    write(tap(26+channel),apf1); write(tap(28+channel),apf2);
    output[channel][phase] = saturate(value); output[channel^1][phase] = 0;
    s16Output result{static_cast<s16>(multiply(filter(output[0],phase,true),depth[0])),
                     static_cast<s16>(multiply(filter(output[1],phase,true),depth[1]))};
    if (channel) cursor = (cursor+1) % static_cast<u32>(memory.size());
    phase = (phase+1)&63;
    return result;
  }
};
Reverb reverb;
struct Routing { std::weak_ptr<Voice> voice; u16 flags; };
std::unordered_map<const Voice*,Routing> routing;
}

// CPP-only extension: called under the host 989 engine's mutex, like Tick.
void SetVoiceReverb(const std::shared_ptr<Voice>& voice, u16 flags) {
  routing[voice.get()] = {voice,flags};
}
void SetReverbPreset(Synth& synth, const u16* parameters, u32 words) {
  reverb.owner = &synth;
  reverb.memory.assign(words,0);
  if (parameters) std::copy_n(parameters,32,reverb.parameters.begin());
  reverb.cursor = reverb.phase = 0;
  reverb.input = {}; reverb.output = {};
}
void SetReverbDepth(Synth& synth, s16 left, s16 right) {
  reverb.owner = &synth; reverb.depth[0] = left; reverb.depth[1] = right;
}

static s16 ApplyVolume(s16 sample, s32 volume) {
  return (sample * volume) >> 15;
}

s16Output Synth::Tick() {
  // Accumulate all voices before saturating the SPU core's dry mix. Clipping
  // each addition makes opposing waveforms depend on voice allocation order.
  s64 left = 0;
  s64 right = 0;
  s64 wetLeft = 0;
  s64 wetRight = 0;

  mVoices.remove_if([](std::shared_ptr<Voice>& v) {
    if (!v->Dead()) return false;
    routing.erase(v.get()); return true;
  });
  for (auto& v : mVoices) {
    const auto voice = v->Run();
    u16 flags = 0;
    if (const auto route = routing.find(v.get()); route != routing.end() && route->second.voice.lock() == v)
      flags = route->second.flags;
    if (!(flags & 0x10)) { left += voice.left; right += voice.right; }
    if (flags & 1) { wetLeft += voice.left; wetRight += voice.right; }
  }

  left = Reverb::saturate(left); right = Reverb::saturate(right);
  if (reverb.owner == this) {
    const auto wet = reverb.tick(wetLeft,wetRight);
    left += wet.left; right += wet.right;
  }

  s16Output out{};
  // Get() is the register encoding (including sweep flags), not its gain.
  out.left = ApplyVolume(static_cast<s16>(std::clamp<s64>(left, INT16_MIN, INT16_MAX)),
                         mVolume.left.GetCurrent());
  out.right = ApplyVolume(static_cast<s16>(std::clamp<s64>(right, INT16_MIN, INT16_MAX)),
                          mVolume.right.GetCurrent());

  mVolume.Run();

  return out;
}

void Synth::AddVoice(std::shared_ptr<Voice> voice) {
  mVoices.emplace_front(voice);
}

void Synth::SetMasterVol(u32 volume) {
  mVolume.left.Set(volume);
  mVolume.right.Set(volume);
}
}  // namespace snd
