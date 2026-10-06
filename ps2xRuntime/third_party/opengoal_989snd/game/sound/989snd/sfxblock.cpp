#include "sfxblock.h"

#include "blocksound_handler.h"
#include "sfxgrain.h"



namespace snd {

std::optional<std::unique_ptr<SoundHandler>> SFXBlock::MakeHandler(VoiceManager& vm,
                                                                   u32 sound_id,
                                                                   s32 vol,
                                                                   s32 pan,
                                                                   SndPlayParams& params,
                                                                   s32 current_tick,
                                                                   u32 owner) {
  auto& SFX = Sounds[sound_id];

  if (SFX.Grains.empty()) {
    return std::nullopt;
  }

  auto handler = std::make_unique<BlockSoundHandler>(*this, SFX, vm, vol, pan, params, sound_id,
                                                     current_tick, owner);
  return handler;
}

std::optional<u32> SFXBlock::GetSoundByName(const char* name) {
  auto sound = Names.find(name);
  if (sound != Names.end()) {
    return sound->second;
  }

  return std::nullopt;
}

void SFXBlock::DebugPrintAllSounds() {}

}  // namespace snd
