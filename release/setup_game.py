"""Build the Windows game locally from a player's validated executable.

The source package contains no translated game programs. Generated files and
compiled games stay under the private installation and must not be uploaded.
"""
# SPDX-License-Identifier: GPL-3.0-or-later
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from extract_vu import extract_images


def announce(message):
    print('WOTM_STATUS: ' + message, flush=True)


def run(args, *, cwd, env):
    subprocess.run([str(arg) for arg in args], cwd=cwd, env=env, check=True)


def tool_environment():
    if os.name != 'nt':
        raise ValueError('This release builds for 64-bit Windows.')
    vswhere = Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)')) / 'Microsoft Visual Studio/Installer/vswhere.exe'
    if not vswhere.is_file():
        raise ValueError('Install Visual Studio Build Tools with Desktop development with C++, Windows SDK, C++ Clang tools and CMake tools. See SETUP.md.')
    root = subprocess.check_output([str(vswhere), '-latest', '-products', '*', '-property', 'installationPath'], text=True).strip()
    if not root or '\n' in root or any(c in root for c in '"\r'):
        raise ValueError('A supported Visual Studio C++ installation was not found. See SETUP.md.')
    vs = Path(root)
    vcvars = vs/'VC/Auxiliary/Build/vcvars64.bat'
    compiler = vs/'VC/Tools/Llvm/x64/bin/clang-cl.exe'
    cmake = vs/'Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe'
    ninja = vs/'Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe'
    if not cmake.is_file():
        cmake = Path(shutil.which('cmake') or '')
    if not all(p.is_file() for p in (vcvars, compiler, cmake, ninja)):
        raise ValueError('C++ build components are missing. Install the Windows SDK, x64 C++ tools, Clang and CMake tools; see SETUP.md.')
    # Only the discovered developer-tool path enters cmd. ISO/output paths are
    # always subprocess arguments, never interpolated shell commands.
    # cmd has its own quote rules; subprocess's argv quoting would turn the
    # inner quotes into literal backslash-quotes and break paths with spaces.
    command = '"' + os.environ.get('COMSPEC', 'cmd.exe') + '" /d /s /c "call "' + str(vcvars) + '" >nul && set"'
    output = subprocess.check_output(command, text=True, errors='replace')
    env = dict(os.environ)
    for line in output.splitlines():
        name, sep, value = line.partition('=')
        if sep and name and not name.startswith('='):
            env[name.upper()] = value
    if not env.get('WINDOWSSDKDIR') or not env.get('INCLUDE'):
        raise ValueError('The Windows SDK environment could not be initialized. See SETUP.md.')
    git = shutil.which('git.exe', path=env.get('PATH')) or shutil.which('git.exe')
    if not git:
        raise ValueError('Install Git for Windows so setup can obtain the open-source dependencies. See SETUP.md.')
    env['PATH'] = str(Path(git).parent) + os.pathsep + env.get('PATH', '')
    return cmake, compiler, ninja, env


def source_id(source):
    digest = hashlib.sha256()
    paths = [source/'CMakeLists.txt', source/'release/game.toml']
    for folder in ('ps2xRuntime', 'ps2xIOP', 'ps2xRecomp', 'release'):
        for directory, dirs, files in os.walk(source/folder):
            dirs[:] = sorted(name for name in dirs if name not in {'__pycache__', 'runner', 'local-generated', 'staging', 'reports'})
            for name in sorted(files):
                path = Path(directory)/name
                if path.suffix.lower() in {'.cpp','.c','.h','.hpp','.inc','.cmake','.py','.json','.toml'} or name == 'CMakeLists.txt':
                    paths.append(path)
    for path in sorted(set(paths)):
        digest.update(path.relative_to(source).as_posix().encode('utf-8'))
        digest.update(b'\0')
        digest.update(path.read_bytes())
    return digest.hexdigest()[:24]


def write_once(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Existing private build input differs: ' + path.name)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def dependency_options(cache_file):
    if cache_file is None:
        return {}
    cache = json.loads(cache_file.read_text(encoding='utf-8'))
    admitted = {'elfio','toml11','fmt','libdwarf','rabbitizer','raylib','ffmpeg'}
    if not isinstance(cache, dict) or set(cache)-admitted:
        raise ValueError('Invalid dependency cache configuration')
    for key, value in cache.items():
        if not isinstance(value, str) or not Path(value).exists():
            raise ValueError('Missing dependency cache: ' + key)
    return cache


def stage_runtime_resources(source, installation):
    # These are our shader and licensed controller sheet, not extracted game
    # assets. A separate install directory needs them beside its launcher.
    for relative in ('launcher/fxaa.fs', 'ps2xRuntime/src/lib/gs/gs_d3d12.hlsl',
                     'ps2xRuntime/assets/controller/xboxControllerSpriteFont.png',
                     'ps2xRuntime/assets/controller/README.md',
                     'ps2xRuntime/assets/controller/LICENSE.txt'):
        write_once(installation/relative, (source/relative).read_bytes())


def build(source, installation, elf, jobs, cache_file=None):
    recipe = json.loads((source/'release/vu-extraction.json').read_text(encoding='utf-8'))
    extract_images(elf.read_bytes(), recipe)  # Full revision/hash checks before building.
    cmake, compiler, ninja, env = tool_environment()
    key = source_id(source)
    private = installation/'local-build'/key
    private.mkdir(parents=True, exist_ok=True)
    write_once(private/'SCUS_971.97', elf.read_bytes())
    cache = dependency_options(cache_file)
    common = ['-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
              '-DCMAKE_C_COMPILER='+str(compiler), '-DCMAKE_CXX_COMPILER='+str(compiler),
              '-DCMAKE_MAKE_PROGRAM='+str(ninja), '-DPS2X_ENABLE_LTO=OFF',
              '-DPS2X_BUILD_ANALYZER=OFF', '-DPS2X_BUILD_TEST=OFF', '-DPS2X_BUILD_STUDIO=OFF']
    if cache:
        common.append('-DFETCHCONTENT_FULLY_DISCONNECTED=ON')
        for name, path in cache.items():
            if name != 'ffmpeg':
                common.append('-DFETCHCONTENT_SOURCE_DIR_'+name.upper()+'='+path)
    tools_build = private/'tools'
    announce('Building the code translator. First setup can take several minutes.')
    run([cmake, '-S', source, '-B', tools_build, *common,
         '-DPS2X_BUILD_RECOMP=ON', '-DPS2X_BUILD_RUNTIME=OFF'], cwd=private, env=env)
    run([cmake, '--build', tools_build, '--target', 'ps2_recomp', '--parallel', jobs], cwd=private, env=env)
    ee = private/'generated-ee'
    vu = private/'generated-vu'
    config = (source/'release/game.toml').read_text(encoding='utf-8-sig')
    if config.count('input = "SCUS_971.97"') != 1 or config.count('output = "./output/"') != 1:
        raise ValueError('Unexpected game generation configuration')
    # Relative paths avoid TOML quoting hazards from arbitrary install locations.
    config = config.replace('output = "./output/"', 'output = "./generated-ee/"')
    write_once(private/'generate.toml', config.encode('utf-8'))
    marker = private/'generation.complete'
    if not marker.exists():
        announce('Translating your game to native C++ and generating optimized geometry code.')
        run([tools_build/'ps2xRecomp/ps2_recomp.exe', private/'generate.toml'], cwd=private, env=env)
        if not (ee/'register_functions.cpp').is_file():
            raise ValueError('The translator did not produce its function registry.')
        for script in ('generate_vu.py','generate_blocks.py','generate_fused.py'):
            run([sys.executable, source/'release'/script, private/'SCUS_971.97', vu], cwd=private, env=env)
        write_once(marker, (key+'\n').encode('ascii'))
    game_build = private/'runtime'
    runtime_options = ['-DPS2X_BUILD_RECOMP=OFF', '-DPS2X_BUILD_RUNTIME=ON',
                       '-DPS2X_ENABLE_DEBUG_UI=OFF', '-DPS2X_SHOW_WINDOWS_CONSOLE=OFF',
                       '-DPS2X_GUEST_SOURCE_DIR='+str(ee), '-DPS2X_VU_GENERATED_DIR='+str(vu),
                       '-DPS2X_VU_KERNEL_DIR='+str(vu)]
    if 'ffmpeg' in cache:
        runtime_options.append('-DFFMPEG_PREBUILT_URL='+Path(cache['ffmpeg']).resolve().as_posix())
    announce('Compiling your PC game. Leave the launcher open until setup finishes.')
    run([cmake, '-S', source, '-B', game_build, *common, *runtime_options], cwd=private, env=env)
    run([cmake, '--build', game_build, '--target', 'ps2EntryRunner', '--parallel', jobs], cwd=private, env=env)
    executable = game_build/'ps2xRuntime/ps2EntryRunner.exe'
    if not executable.is_file():
        raise ValueError('The game build did not produce an executable.')
    version = installation/'game/versions'/key
    version.mkdir(parents=True, exist_ok=True)
    write_once(version/'War of the Monsters.exe', executable.read_bytes())
    for dll in executable.parent.glob('*.dll'):
        write_once(version/dll.name, dll.read_bytes())
    if not (version/'avcodec-61.dll').is_file():
        raise ValueError('The movie/audio runtime files are missing.')
    stage_runtime_resources(source, installation)
    # Activate only a completely built version; preserve previous working files.
    active = installation/'game/active.txt'
    pending = active.with_suffix('.pending')
    data = 'WOTM-GAME-1\n'+key+'\n'+hashlib.sha256(executable.read_bytes()).hexdigest()+'\n'
    pending.write_text(data, encoding='ascii')
    os.replace(pending, active)
    announce('Game setup complete. Ready to play.')


def main():
    if sys.version_info < (3, 12):
        raise ValueError('Python 3.12 or newer is required. See SETUP.md.')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--install', type=Path)
    parser.add_argument('--elf', type=Path)
    parser.add_argument('--jobs', type=int, default=max(1, min(4, (os.cpu_count() or 2)//2)))
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--dependency-cache', type=Path, help='Optional developer cache of dependency sources/archive; not part of distribution.')
    args = parser.parse_args()
    if not 1 <= args.jobs <= 32:
        parser.error('--jobs must be between 1 and 32')
    if args.check:
        tool_environment()
        announce('Build prerequisites are available.')
        return
    if args.install is None or args.elf is None:
        parser.error('--install and --elf are required')
    build(args.source.resolve(strict=True), args.install.resolve(), args.elf.resolve(strict=True), args.jobs, args.dependency_cache)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print('WOTM_ERROR: '+str(error), file=sys.stderr, flush=True)
        sys.exit(1)
