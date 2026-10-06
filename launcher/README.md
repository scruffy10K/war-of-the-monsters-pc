# Windows launcher

The player launcher imports the supported ISO, saves settings and starts the precompiled game in game/War of the Monsters.exe. It never downloads developer tools or compiles the game. The complete Windows ZIP includes required runtime DLLs and shaders.

For development, build.bat compiles these C# sources with the Windows .NET Framework compiler. Developer game builds use release/setup_game.py separately; their versioned game receipts remain supported. See the root SETUP.md.

New imports extract directly into game_data and write a WOTM-INSTALL-2 receipt. disc.index maps original byte ranges to extracted files and disc.meta preserves nonzero bytes outside file extents. Verified zero gaps consume no storage. Old WOTM-INSTALL-1 ISO installations remain supported. No original assets or disc metadata are shipped in the download.
