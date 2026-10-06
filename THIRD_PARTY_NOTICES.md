# Credits and runtime-library notices

This modified unofficial port is based on PS2Recomp, upstream starting point 14b1e5c, under GNU GPL version 3. See LICENSE and https://github.com/ran-j/PS2Recomp. Port source, launcher source and generation/build recipes are supplied separately. Translated original game routines are present in the native game binary; original game data and translated source files are not included. No notice here grants rights to original game content.

The launcher uses the .NET Framework supplied with supported Windows systems. The native game is precompiled with Clang/Visual Studio tools. No compiler or build tools are included or downloaded by the player launcher.

## Included runtime components

- OpenGOAL 989snd subset, ISC, commit a42c0719967d0869eb9ecf4cfef91a2857991428: https://github.com/open-goal/jak-project. Original notice/provenance in licenses/OpenGOAL-LICENSE and licenses/OpenGOAL-README.md, and the source tree.
- raylib 5.5, zlib license with bundled component notices: https://github.com/raysan5/raylib/tree/5.5. It is linked into the game. Source is included in the matching third-party source archive; its original embedded component notices remain in that source.
- DirectXTK Xbox controller sheet, MIT: ps2xRuntime/assets/controller/LICENSE.txt includes attribution and source. PlayStation artwork and game artwork come from the player's ISO.
- Microsoft Visual C++ runtime libraries, original signed redistributable DLLs version 14.51.36231: supplied app-locally from the Visual Studio Community redistributable directory. They are Microsoft proprietary components, not covered by this project's GPL. Microsoft redistribution terms and permitted-file lists apply: https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files and https://visualstudio.microsoft.com/license-terms/.

## Movie/media libraries

This software uses libraries from the FFmpeg project under the LGPLv2.1. The original LGPL texts are in licenses/ffmpeg. The dynamic libraries remain replaceable and are unmodified from System233's n7.1-241205 lgpl-amd64-shared archive:
https://github.com/System233/ffmpeg-msvc-prebuilt/releases/tag/n7.1-241205
Archive SHA256: b985887dbca8c0a9a4c47f9d2bac7ec3ab994335b1b7fe7000bdb1dcd772ac68

The bundled subset contains avcodec-61, avutil-59, swscale-8, swresample-5 and their required JPEG XL/Brotli/WebP/zlib libraries. Unused FFmpeg programs and other media DLLs are excluded. avcodec reports LGPL version 2.1 or later, with no --enable-gpl or --enable-nonfree flag.

The matching Third-Party Sources ZIP mirrors FFmpeg commit b08d7969c550a804a59511c7b83f2dd8cc0499b8, its publisher's build scripts and patches, and the relevant dependency source revisions. SOURCE-MANIFEST.json records exact URLs, revisions, sizes and hashes. FFmpeg's actual configuration string and publisher patches are preserved alongside the archives. The publisher's upstream patch is to libavfilter/textutils.c; no additional local media-library changes were made. See https://ffmpeg.org/legal.html for licensing guidance.

Bundled library notices are under licenses/libjxl, licenses/brotli, licenses/highway, licenses/lcms, licenses/libjpeg-turbo, licenses/skcms, licenses/libvpx, licenses/libwebp and licenses/zlib. These preserve original copyrights and license/patent texts, including components statically incorporated into the media libraries. The associated source revisions come from the publisher's tagged submodule tree and libjxl's own submodule tree.

## Source-build tools

ELFIO (MIT), toml11 (MIT), fmt (MIT), libdwarf (LGPL 2.1 with per-file notices), and Rabbitizer (MIT) are used by the source/build tooling. Version and license copies are in the source package. They are not tools a player must install. Optional debug ImGui/rlImGui code is excluded from the player game.

Keep the port source and matching Third-Party Sources ZIP available with a binary download. The original game's translated-code rights and corresponding-source obligations require publication review; these notices are not a guarantee of permission to redistribute the port.
