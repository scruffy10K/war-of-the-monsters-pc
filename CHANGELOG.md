# Launcher blank ISO default — 2026-10-06

- A fresh launcher starts with an empty ISO field instead of guessing a basegame.iso path in its own folder. Browse selects the player's ISO; subsequent launches still remember that player's saved selection.
- The download includes no settings.ini, saved ISO location or installed game data. The native game and extracted-install format are unchanged.

# Extracted installation update — 2026-10-06

- New imports no longer duplicate the full ISO. All 641 disc files are extracted; a 32,182-byte index and 1,731,712-byte metadata file retain the original disc layout.
- Installed disc data falls from 2,108,341,382 bytes to 851,289,788 bytes (about 60% less). This excludes the small executable copy at the install root, application files and saves.
- CD loading, IOP reads, music streaming and sound-effect banks can read the extracted installation while retaining original byte offsets. OpenGL and Direct3D 12 use the same file reader.
- Existing ISO installations continue to work. Failed imports do not activate the partial folder; old installations and saves are retained.

Validation: exact full-disc comparison against all 1,258,815,488 original bytes, 12,540 sequential/boundary/random reads, 4,000 concurrent reads, and 13 malformed-index rejection cases passed. Fresh installation through the actual launcher took 2.4 seconds on the development machine, with no ISO copy, tool downloads or compilation on the player path. Actual launcher Play passed muted 105-second tests on both backends with the original-ISO field pointing at an unavailable file: Direct3D 12 completed 3,879 game frames and OpenGL 3,884, with clean exits, movie PCM streams, six loaded sound banks, ten reverb presets and nonzero mixed audio. Nine installation compatibility/integrity checks passed. The original 35 launcher checks plus damaged-index/artwork startup checks are included in final UI verification. Audio was inspected as decoded samples, not listened to; full campaign and fresh-PC testing remain incomplete.

# Precompiled Windows preview — 2026-10-06

- The Windows player download now includes the compiled native game and its runtime libraries. Players no longer install or download Python, Git, Visual Studio, SDKs or compilers.
- Install Game only validates the supported ISO, imports its data and prepares local assets. The player launcher contains no compiler/download bootstrap.
- The separate Source ZIP retains developer build/generation tools. The portable-tool-download prototype is not included in either player or source ZIP.
- Added bundled app-local Microsoft C++ runtime DLLs and the required media-library subset. Added license notices, source mirrors and an About notice.
- Kept the tested native executable unchanged. OpenGL/Direct3D 12, display/controller/camera settings, native timing, optional motion smoothing, ISO theme extraction, saves and separate mod import remain available. No mods are included.

## Validation

The actual launcher's Install Game route imported a supported ISO in a new isolated folder without compiler tools or internet use. The player folder contains neither setup tool executables nor setup source scripts. Both graphics backends ran through the actual Play route using the installed disc while the original-ISO field pointed at an unavailable file.

Muted 105-second checks completed 3,889 Direct3D 12 and 3,653 OpenGL game frames, with clean timed exits, progressing gameplay counters and loaded audio reverb presets. Loaded-module checks confirmed FFmpeg and Microsoft C++ runtime libraries came from the player folder. The test PATH contained only Windows System32. This tests isolation from installed developer tools, not a fresh Windows VM or every GPU/driver.

All 35 existing launcher/resource/layout checks passed with the new About control. The final compiler output passed; valid legacy receipts resolve and invalid paths are rejected. The last binary revision changes only legacy-install recovery messages, after gameplay checks. No native game, renderer or audio code changed in this packaging update. The source-file guard passed; exact archive inventory, hashes and CRC checks are part of packaging.

The first attempt used the short input window intended for Skip Intro while mods were disabled. It stayed in movies with zero game frames; those runs are not counted as gameplay passes. The corrected normal-intro input schedule reached the fights above.

Full campaign, long-session, fresh-Windows, cross-hardware and audible audio checks remain incomplete. These are technical checks, not clearance to distribute translated original game code. See README.md and THIRD_PARTY_NOTICES.md before public publication.
