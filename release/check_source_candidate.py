"""Inspect a source-only staging folder before creating a public repository.

This catches common packaging mistakes, not all copyrighted content or secrets.
It never exports files, changes Git, uploads anything, or enters runner folders.
Project/dependency licenses and game-specific source still require review.
"""
import argparse
import os
from pathlib import Path
import re
import sys

BLOCK_DIRS = {'.git', '.vs', '.vscode', '.idea', 'runner', 'output', 'game_data',
              'mc0', 'mc1', 'scratch_mc_backup', 'work', 'staging', 'reports',
              '__pycache__', 'local-build', 'generated-ee', 'generated-vu', 'mods'}
BLOCK_FILES = {'ps2_project_state.md', '.mcp.json', 'imgui.ini', 'settings.ini',
               'native_parameters.inc', 'fused_parameters.inc', 'heavyblockimpl.inc',
               'menushaderblock.inc', 'fightshaderblock.inc', 'objectshadercases.inc',
               'ps2_vu1_hle1028.inc', 'ps2_vu1_hle1028_values.inc', 'ps2_vu1_hle40.inc',
               'ps2_vu1_block_prototype.cpp', 'ps2_vu1_native_registry.cpp'}
SOURCE_SUFFIXES = {'.cpp', '.c', '.h', '.hpp', '.inc', '.cs', '.hlsl', '.fs',
                   '.cmake', '.toml', '.md', '.txt', '.py', '.bat', '.yml', '.yaml',
                   '.java', '.json', '.gitignore', '.gitattributes'}

def inspect(root):
    issues = []
    for required in ('README.md', 'LICENSE'):
        if not (root / required).is_file():
            issues.append((required, 'missing required release document'))
    for directory, dirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in list(dirs):
            child = base / name
            lower = name.lower()
            if (child.is_symlink() or child.is_junction() or lower in BLOCK_DIRS or
                (base == root and lower == 'game') or
                lower.startswith(('build', 'output_prev', 'ghidra', 'pcsx2', 'ps2-recomp-agent-skill'))):
                issues.append((child.relative_to(root).as_posix(), 'local/generated/tool directory excluded; contents not scanned'))
                dirs.remove(name)
        for name in files:
            file = base / name
            relative = file.relative_to(root).as_posix()
            lower = name.lower()
            if file.is_symlink():
                issues.append((relative, 'symbolic link requires explicit review')); continue
            if (lower in BLOCK_FILES or lower.startswith(('scus_', 'slus_', 'sles_', 'sces_', 'vu0_image_', 'vu1_image_')) or
                'ps2_recompiled' in lower or re.match(r'vu[01]_native_[0-9a-f]+\.cpp$', lower) or
                lower.endswith(('.bak', '.tmp', '.ini'))):
                issues.append((relative, 'game-generated or private file')); continue
            if file.suffix.lower() == '.zip':
                issues.append((relative, 'archive excluded from main source package; mods are separate downloads'))
                continue
            if relative == 'ps2xRuntime/assets/controller/xboxControllerSpriteFont.png':
                if not (file.parent / 'LICENSE.txt').is_file():
                    issues.append((relative, 'controller artwork notice missing'))
                continue
            if (file.suffix.lower() not in SOURCE_SUFFIXES and
                name not in {'LICENSE', 'COPYING', 'CMakeLists.txt', '.gitignore', '.gitattributes'}):
                issues.append((relative, 'binary/data or unrecognized file requires review')); continue
            if file.stat().st_size > 8 * 1024 * 1024:
                issues.append((relative, 'oversized source/document requires review')); continue
            try:
                text = file.read_text(encoding='utf-8-sig')
            except (OSError, UnicodeError):
                issues.append((relative, 'unreadable/non-UTF8 source requires review')); continue
            # Exclude this checker from matching its own diagnostic patterns.
            if file.resolve() == Path(__file__).resolve(): continue
            if re.search(r'(?:stepPair|execLowerN|operator\(\))\s*<\s*0x[0-9a-f]+', text, re.I):
                issues.append((relative, 'literal VU instruction pairs require game-code provenance review'))
            if re.search(r'(?:[A-Z]:[/\\]+Users[/\\]|/home/|/Users/)', text):
                issues.append((relative, 'personal filesystem path'))
            if re.search(r'\b(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{24,})\b', text):
                issues.append((relative, 'possible credential; value intentionally omitted'))
    return issues

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    root = parser.parse_args().candidate.resolve(strict=True)
    if not root.is_dir(): parser.error('candidate must be a staging directory')
    issues = inspect(root)
    for path, reason in issues[:30]: print(path + ': ' + reason)
    if issues:
        print('BLOCKED: %d packaging findings. No files changed or uploaded.' % len(issues))
        return 1
    print('Basic source-package checks passed. Manual provenance, license and history review still required.')
    return 0

if __name__ == '__main__':
    sys.exit(main())
