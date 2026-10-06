# War of the Monsters PC preview

An unofficial Windows port based on [PS2Recomp](https://github.com/ran-j/PS2Recomp).

## For players

Download the **Windows Player ZIP**, extract the whole folder, and open **War of the Monsters Launcher.exe**. Browse to your supported US retail ISO (SCUS-97197), select **Install Game**, choose your settings and press **Play**.

Install Game extracts the disc files without keeping a second ISO. After installation you can play without the original ISO connected.

The game is already compiled. Players do not need Python, Git, Visual Studio, a compiler, or build-tool downloads. The necessary application runtime libraries are included beside the game. Keep the whole folder together; the launcher EXE by itself is not the complete download.

The download excludes the retail ISO, extracted game data, artwork, music, movies, private saves and mod packages. Installation obtains the game data and launcher artwork from the player's ISO. The executable does contain translated game routines: a bring-your-own-ISO requirement is not itself legal clearance for distributing that executable.

## Features

- OpenGL and Direct3D 12, selectable resolution, windowed/borderless/fullscreen display and optional FXAA.
- Optional widescreen view, modern orbit camera for both players, and PlayStation/Xbox button prompts.
- Gameplay targeting 60 FPS, separate display refresh limit and optional experimental D3D12 motion smoothing.
- Movie audio, game sound/music, local saves and local two-player support.
- Mod import support; mods are separate downloads and none are bundled or enabled.

Vulkan, NVIDIA DLSS, ray tracing and online multiplayer are not implemented. The runtime debugger is excluded and the game window title is War of the Monsters - SCUS-97197.

## Source and distribution

The separate **Source ZIP** contains the port/runtime, launcher, recompiler and generation/build recipes. The C++ game translations are generated from a supported original executable during a developer build; they are not included as source files. Building from source requires developer tools; playing the Windows download does not.

Keep the Source ZIP and the matching **Third-Party Sources ZIP** available alongside a binary release. That archive mirrors the media-library sources and publisher build patches/configuration; license texts and source locations are also supplied in THIRD_PARTY_NOTICES.md. Rights to translated game code and the completeness of corresponding-source obligations still need publication review. This is a technical preview package, not a legal clearance or publisher endorsement.

Do not upload an installed folder or the development repository/history. Private game data, generated code, saves, logs and diagnostic captures belong only on the player's PC. The source-package checker catches common mistakes; it is not a legal audit.

## Preview limitations

Performance depends on hardware and scene. Motion smoothing may produce artifacts and is off by default. Full campaign, long-session, cross-hardware and audible audio testing are incomplete. When reporting a problem include renderer/settings and reproduction steps; do not attach game files or personal paths.

Read SETUP.md, CHANGELOG.md, THIRD_PARTY_NOTICES.md and LICENSE. The project is unaffiliated with the game's original developers and publisher.
