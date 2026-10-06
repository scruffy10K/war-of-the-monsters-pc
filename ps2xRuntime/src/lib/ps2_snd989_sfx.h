// 989SND sound-effect banks (ps2_snd989_sfx.cpp).
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

namespace Snd989Sfx
{
    // Loads the SBlk bank stored at the given disc LBA; returns a bank handle (0 on failure).
    uint32_t loadBank(const std::string &isoPath, uint32_t lba);
    void unloadBank(uint32_t bank);
    // Starts sound `sound` of `bank`; returns a sound handle (0 if nothing plays).
    uint32_t play(uint32_t bank, int32_t sound, int32_t vol, int32_t pan, int32_t pm, int32_t pb);
    bool isHandle(uint32_t handle);
    bool stillPlaying(uint32_t handle);
    bool isLooper(uint32_t bank, int32_t sound);
    void stop(uint32_t handle);
    void stopAll();
    void pause(uint32_t handle, bool paused);
    void pauseGroups(uint32_t mask, bool paused);
    void setVolPan(uint32_t handle, int32_t vol, int32_t pan);
    void setPitchMod(uint32_t handle, int32_t pm);
    void setPitchBend(uint32_t handle, int32_t pb);
    void setRegister(uint32_t handle, uint32_t reg, int32_t value);
    void setMasterVolume(uint32_t group, int32_t volume);
    size_t activeSounds();
    void render(int32_t *acc, uint32_t frames);
}
