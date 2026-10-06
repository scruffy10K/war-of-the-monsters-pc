# Install and play

1. Extract the complete Windows Player ZIP into a writable folder. Do not run the launcher from inside the ZIP or move its EXE away from the other folders.
2. Open **War of the Monsters Launcher.exe**.
3. Choose **Browse...**, select your unmodified US War of the Monsters ISO (SCUS-97197), and press **Install Game**.
4. Select your display/controller options and press **Play**.

No tool downloads, compilation, administrator access or separate developer-package installation is needed. The included game is already compiled; Install Game only validates and installs your disc data. No internet connection is required for that operation or for gameplay.

## Requirements

- 64-bit Windows 10 or 11 with its desktop .NET Framework support, an AVX2-capable CPU and a current graphics driver. OpenGL 3.3 is required by the window/rendering layer; Direct3D 12 is also required when selecting that backend.
- Your supported unmodified US retail ISO. Other regions or executable revisions are rejected.
- About 1 GB of free space for the extracted installation, in addition to the player download and your original ISO. The supported disc installs approximately 851 MB of data, plus a small executable copy at the root.

Install Game extracts files directly from your ISO. It does not copy the full ISO into the installation. A small disc index and metadata file preserve streaming audio, movies and original disc locations. After installation, the original ISO location is no longer needed. Keep the installed game_data folder and SCUS_971.97 file. Saves are stored under mc0 and mc1. The game folder holds the precompiled executable and support DLLs; deleting those breaks the download.

The main download contains no mods. Use **Mods...** to import separately obtained supported mod ZIPs. Neither normal gameplay nor mods are muted by this release; muting is only used in the developer's private test processes.

If a runtime file is missing, extract the complete Windows ZIP again. Do not download individual DLLs from third-party DLL websites. An unsupported ISO message means the supplied executable does not match the supported revision. Setup preserves existing installations and saves when it fails; repeated successful imports create separate data folders.

## Updating an older installation

Existing installations containing disc.iso still work. To use the smaller format, select your original ISO (or the old installed disc.iso) and choose Install Game again. The launcher creates a separate extracted installation and activates it only after successful validation. Old data folders and saves are kept; they are not automatically removed. Keep the entire active installation, including files, disc.index, disc.meta and installation.txt together. Do not delete disc.iso from an older installation that is still active.

## Building the separate source package

These instructions are for developers only. Install Python 3.12+, Git for Windows and Visual Studio C++/Clang/CMake tools with a Windows SDK. Extract SCUS_971.97 from your own supported ISO into the source root, then run:

```
python -B release/setup_game.py --source . --install . --elf SCUS_971.97
launcher\build.bat
```

The developer script generates game-dependent C++ and VU programs locally, downloads dependencies and compiles a private game. The launcher can use its versioned game receipt. The player launcher itself never invokes this script. Keep generated files, originals, local-build, game_data and saves out of a source repository.

Supported original-executable SHA256:
67ac3e4bf656f688767ad654df7fe310317efc4909d949ad32fd762825cf659e
