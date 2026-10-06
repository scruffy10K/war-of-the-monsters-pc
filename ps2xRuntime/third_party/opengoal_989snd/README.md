# OpenGOAL 989snd SFX engine (vendored)

Source: https://github.com/open-goal/jak-project, `game/sound/989snd` and
`game/sound/common`, commit a42c0719967d0869eb9ecf4cfef91a2857991428.
License: ISC (see LICENSE). Copyright (c) 2020-2026 OpenGOAL Team;
individual files: Copyright 2021-2024 Ziemas.

Used by ps2xRuntime to play War of the Monsters' 989SND sound-effect banks
(SBlk version 1). Files are unmodified except:
- `sfxblock.cpp`: DebugPrintAllSounds reduced to a no-op (fmt/magic_enum).
- `common/common_types.h`, `common/log/log.h`, `third-party/magic_enum.hpp`,
  `game/sound/989snd/loader.h`: small shims replacing OpenGOAL headers.
- `SFXBlock::ReadBlock` (OpenGOAL `loader.cpp`) is reimplemented in
  `ps2xRuntime/src/lib/ps2_snd989_sfx.cpp`.
- `common/synth.cpp`: accumulate voices before saturation and use the current
  master gain rather than the encoded volume register; native SPU reverb filter
  network with half-band resampling and in-memory player-provided LIBSD presets.
- `common/voice.cpp`: round the combined ADPCM predictor once and bound its
  filter lookup.
- `989snd/vagvoice.cpp`: forward tone dry/reverb routing flags to the mixer.

The reverb implementation follows the public hardware register equations in
https://psx-spx.consoledev.net/ps1/spu/soundprocessingunitspu/ . It does not copy
PCSX2 code or PS2SDK implementation code. Work-area/register data is read from
the player's installed or ISO-provided `MOD/LIBSD.IRX`, never bundled here.
