#!/usr/bin/env python3
"""Generate compact native VU control flow from a player's validated code image.

Profiles describe our optimization boundaries and host-kernel integration points.
Instruction words and branch destinations are decoded from the local image.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path

from extract_vu import extract_images
from generate_vu import BRANCH_OPS, Pair, plan_pairs


@dataclass(frozen=True)
class Hook:
    call: str
    condition: str = ''
    targets: tuple[int, ...] = ()
    replaces: bool = False
    optional_pointer: bool = False
    declaration: str = ''
    boolean_result: bool = False
    poll: bool = False


@dataclass(frozen=True)
class Profile:
    name: str
    prefix: str
    regions: tuple[tuple[int, int], ...]
    dead_flags: tuple[tuple[int, int, str], ...] = ()
    hooks: dict[int, Hook] = field(default_factory=dict)
    entry: str = ''
    template: str = ''
    has_reference: bool = True
    exit_hints: dict[int, tuple[int, ...]] = field(default_factory=dict)


MATRIX_OPTION = ('static const bool matrixKernels=[] {const char *p='
                 'std::getenv("PS2X_VU1_MATRIX_KERNELS");return !p || p[0]!=\'0\';}();')
SECOND_OPTION = ('static const bool second=[] {const char *p='
                 'std::getenv("PS2X_VU1_SECOND_TRANSFORM");return !p || p[0]!=\'0\';}();')
POLL = ('if(Vu1NativeAccess::stopRequested(v) || Vu1NativeAccess::cycle(v)>=c.budgetEnd)'
        'return Vu1NativeStep::Stop;')


def matrix_hooks(condition: str) -> dict[int, Hook]:
    return {pc: Hook(f'g_ps2xVu1GameMatrix[{index}]', condition, (target,))
            for index, (pc, target) in enumerate(((0x1ba8, 0x1bc0), (0x1bc0, 0x1bd8), (0x1be8, 0x1c00)))}


def profiles() -> tuple[Profile, ...]:
    # These are integration annotations, not serialized VU instructions.
    common = ((0x0f80, 0x1010),)
    tail = ((0x1b38, 0x1cc8), (0x1ea0, 0x2918))
    heavy_hooks = matrix_hooks('if constexpr(GameMath)if(matrixKernels)')
    heavy_hooks.update({0x1978: Hook('g_ps2xVu1ConvertPair[0]', replaces=True),
                        0x19a8: Hook('g_ps2xVu1ConvertPair[1]', targets=(0x19b8,), replaces=True)})
    menu_hooks = matrix_hooks('if(matrixKernels)')
    menu_hooks.update({
        0x1a48: Hook('g_ps2xVu1MenuVertex', targets=(0x1ab0,), boolean_result=True),
        0x1a60: Hook('g_ps2xVu1ShaderTransform', 'if(matrixKernels)', (0x1a78,)),
        0x1ab8: Hook('g_ps2xVu1MenuLoop', 'nativeMenuLoop && g_ps2xVu1MenuLoop',
                     (0x1ab8, 0x1b28), boolean_result=True, poll=True,
                     declaration='static const bool nativeMenuLoop=[] {const char *p=std::getenv("PS2X_VU1_MENU_LOOP");return !p || p[0]!=\'0\';}();')})
    return (
        Profile('heavyBlockImpl', 'L', common+((0x1878, 0x19e0),)+tail,
                ((0x1878, 0x1930, '(GameMath && LightingFlags)'),
                 (0x19a0, 0x19c8, '(GameMath && LightingFlags)'),
                 (0x1938, 0x1998, 'GameMath'), (0x1b78, 0x1c18, 'GameMath')),
                heavy_hooks, MATRIX_OPTION, 'template<bool GameMath,bool LightingFlags=false>'),
        Profile('menuShaderBlock', 'M', ((0x0c08, 0x1010), (0x19f8, 0x1cc8), (0x1ea0, 0x2918)),
                ((0x0cc0, 0x0d88, 'true'), (0x1a48, 0x1ae0, 'true'), (0x1b78, 0x1c18, 'true')),
                menu_hooks, MATRIX_OPTION, exit_hints={0x1ea8: ()}),
        Profile('fightShaderBlock', 'M', common+((0x1028, 0x1210),)+tail+((0x3080, 0x3258),),
                ((0x10a8, 0x1198, 'true'), (0x30f0, 0x31e8, 'true'), (0x3210, 0x3238, 'true')),
                {0x10f8: Hook('hleFight1028Try', targets=(0x10f8, 0x1208), boolean_result=True, poll=True)},
                exit_hints={0x1ea8: ()}),
        Profile('ps2xVu1BlockPrototype', 'L', ((0x0020, 0x0028), (0x0360, 0x04a8)),
                hooks={0x03e8: Hook('g_ps2xVu1FusedTransform', targets=(0x0408,), replaces=True, optional_pointer=True),
                       0x0410: Hook('g_ps2xVu1FusedTransformSecond', 'if(second && g_ps2xVu1FusedTransformSecond)', (0x0430,))},
                entry=SECOND_OPTION, has_reference=False,
                # Profile-guided dispatch candidates are only guarded fast
                # paths. The actual PC remains determined by stepPair.
                exit_hints={0x0028: (0x03c8,), 0x0490: (0x03c8,), 0x04a8: (0x03c8,)}),
    )


def admitted_pcs(profile: Profile) -> set[int]:
    pcs: set[int] = set()
    for first, last in profile.regions:
        if first < 0 or last >= 16384 or first % 8 or last % 8 or first > last:
            raise ValueError('Invalid compact block region')
        region = set(range(first, last+1, 8))
        if pcs & region:
            raise ValueError('Overlapping compact block regions')
        pcs |= region
    if not pcs:
        raise ValueError('Empty compact block')
    for pc, hook in profile.hooks.items():
        if pc not in pcs or any(target not in pcs for target in hook.targets):
            raise ValueError('Host hook points outside the compact block')
    for pc, targets in profile.exit_hints.items():
        if pc not in pcs or any(target not in pcs for target in targets):
            raise ValueError('Dispatch hint points outside the compact block')
    return pcs


def direct_candidates(plans: list[Pair], index: int, admitted: set[int]) -> list[int]:
    """Fast exits after a delay slot; indirect/unknown destinations redispatch."""
    result = []
    previous = plans[(index-1) % len(plans)]
    op = -1 if previous.upper & 0x80000000 else previous.lower >> 25
    if op in BRANCH_OPS and op not in (0x24, 0x25):
        offset = previous.lower & 2047
        if offset & 1024:
            offset -= 2048
        target = (previous.pc+8+offset*8) % (len(plans)*8)
        if target in admitted:
            result.append(target)
    sequential = (plans[index].pc+8) % (len(plans)*8)
    if sequential in admitted and sequential not in result:
        result.append(sequential)
    return result


def hook_lines(hook: Hook, prefix: str) -> list[str]:
    out = [hook.declaration] if hook.declaration else []
    args = 'v, c, nativeProgram' if hook.call == 'hleFight1028Try' else 'v,c'
    call = f'{hook.call}({args})'
    if hook.optional_pointer:
        out.append(f'if(!{hook.call})return Vu1NativeStep::Fallback;')
    if hook.boolean_result:
        condition = f'{hook.condition} && ' if hook.condition else ''
        out.append(f'if({condition}{call}) {{')
    elif hook.condition:
        out.append(hook.condition+' {')
    if not hook.boolean_result:
        out.append(f'if((r={call})!=Vu1NativeStep::Continue)return r;')
    if hook.poll:
        out.append(POLL)
        for target in hook.targets:
            out.append(f'if(Vu1NativeAccess::pc(v)==0x{target:x}u)goto {prefix}{target:04x};')
        out.append('goto dispatch;')
    else:
        if len(hook.targets) > 1:
            raise ValueError('Non-polling hook needs at most one continuation')
        for target in hook.targets:
            out.append(f'goto {prefix}{target:04x};')
    if hook.boolean_result or hook.condition:
        out.append('}')
    return out


def block_source(image: bytes, profile: Profile) -> str:
    plans = plan_pairs(image, 1)
    admitted = admitted_pcs(profile)
    out = ['// Generated locally from the supplied executable. Do not redistribute.']
    if profile.template:
        out.append(profile.template)
    reference_arg = ', Vu1NativeProgram nativeProgram' if profile.has_reference else ''
    linkage = 'static ' if profile.has_reference else ''
    out.append(f'{linkage}Vu1NativeStep {profile.name}(VU1Interpreter &v, Vu1NativeCtx &c{reference_arg}) {{')
    if profile.entry:
        out.append(profile.entry)
    out += ['Vu1NativeStep r;', 'goto dispatch;']
    for pc in sorted(admitted):
        pair = plans[pc//8]
        out.append(f'{profile.prefix}{pc:04x}:')
        hook = profile.hooks.get(pc)
        if hook:
            out.extend(hook_lines(hook, profile.prefix))
            if hook.replaces:
                continue
        if pair.set_pc:
            out.append(f'Vu1NativeAccess::setPc(v, 0x{pc:04x}u);')
        dead = next((expr for first, last, expr in profile.dead_flags if first <= pc <= last), 'false')
        flags = ['true' if pair.immediate_flags else 'false', 'true', 'true', dead, 'true' if pair.mid else 'false']
        out.append(f'if ((r = Vu1NativeAccess::stepPair<0x{pair.upper:08x}u, 0x{pair.lower:08x}u, '+', '.join(flags)+'> (v, c)) != Vu1NativeStep::Continue) return r;')
        if not pair.fallthrough:
            out.append(POLL)
            for target in profile.exit_hints.get(pc, direct_candidates(plans, pc//8, admitted)):
                out.append(f'if(Vu1NativeAccess::pc(v)==0x{target:x}u)goto {profile.prefix}{target:04x};')
            out.append('goto dispatch;')
        elif pc+8 not in admitted:
            # Region metadata must not accidentally fall into a distant block
            # with an unpublished relaxed PC. Reject instead of guessing.
            raise ValueError(f'Compact region ends in a fallthrough at {pc:x}')
    out += ['dispatch:', POLL, 'switch(Vu1NativeAccess::pc(v)) {']
    for pc in sorted(admitted):
        out.append(f'case 0x{pc:x}:goto {profile.prefix}{pc:04x};')
    out.append('default:return nativeProgram(v,c);' if profile.has_reference else 'default:return Vu1NativeStep::Fallback;')
    out += ['}', '}', '']
    return '\n'.join(out)


def generated_sources(image: bytes) -> dict[str, str]:
    sources = {f'{profile.name}.inc': block_source(image, profile) for profile in profiles()}
    prototype = sources.pop('ps2xVu1BlockPrototype.inc')
    sources['ps2_vu1_block_prototype.cpp'] = ('#include "ps2_vu1_native.h"\n'
        'extern Vu1NativeStep (*g_ps2xVu1FusedTransform)(VU1Interpreter &, Vu1NativeCtx &);\n'
        'extern Vu1NativeStep (*g_ps2xVu1FusedTransformSecond)(VU1Interpreter &, Vu1NativeCtx &);\n'+prototype)
    sources['objectShaderCases.inc'] = object_cases(image)
    return sources


def object_cases(image: bytes) -> str:
    """Original switch boundaries with our native matrix/clip integration."""
    profile = Profile('objectShaderBlock', 'O',
                      ((0x0f80, 0x1010), (0x15e8, 0x1870), (0x1ea0, 0x1ea8)))
    admitted = admitted_pcs(profile)
    plans = plan_pairs(image, 1)
    dead_regions = ((0x1610, 0x1628), (0x1668, 0x1680), (0x1698, 0x1698),
                    (0x16d8, 0x16e0), (0x1708, 0x1730), (0x1748, 0x1758),
                    (0x1770, 0x1770), (0x1788, 0x17a8), (0x1810, 0x1810),
                    (0x1828, 0x1830))
    kernels = {0x1698: (0, 0x16b0), 0x1750: (1, 0x1768),
               0x1790: (2, 0x17a8), 0x17d0: (3, None)}
    continuations = {target for _, target in kernels.values() if target is not None}
    out = ['// Generated locally from the supplied executable. Do not redistribute.']
    for pc in sorted(admitted):
        pair = plans[pc//8]
        out.append(f'case 0x{pc:04x}u:')
        if pc in continuations:
            out.append(f'O{pc:04x}:')
        if pc in kernels:
            kernel, target = kernels[pc]
            out.append(f'if ((r=g_ps2xVu1ObjectKernel[{kernel}](v,c))!=Vu1NativeStep::Continue) return r;')
            if target is None:
                out.append('[[fallthrough]];')
                continue
            out.append(f'goto O{target:04x};')
            # Retain alternate direct-entry cases, including unreachable pair
            # code after this unconditional host hook, to preserve comparison.
        if pair.set_pc:
            out.append(f'Vu1NativeAccess::setPc(v, 0x{pc:04x}u);')
        dead = 'DeadFlags' if any(first <= pc <= last for first, last in dead_regions) else 'false'
        flags = ['true' if pair.immediate_flags else 'false', 'true', 'true', dead, 'true' if pair.mid else 'false']
        out.append(f'if ((r = Vu1NativeAccess::stepPair<0x{pair.upper:08x}u, 0x{pair.lower:08x}u, '+', '.join(flags)+'> (v, c)) != Vu1NativeStep::Continue) return r;')
        if pair.fallthrough and pc+8 not in admitted:
            raise ValueError(f'Object region ends in a fallthrough at {pc:x}')
        out.append('[[fallthrough]];' if pair.fallthrough else 'break;')
    return '\n'.join(out)+'\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('elf', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    recipe = json.loads(Path(__file__).with_name('vu-extraction.json').read_text(encoding='utf-8'))
    images = extract_images(args.elf.read_bytes(), recipe)
    outputs = generated_sources(images['vu1_image_7d7edbc78086ac63.bin'])
    args.destination.mkdir(parents=True, exist_ok=True)
    for name, source in outputs.items():
        target = args.destination/name
        if target.is_symlink() or (target.exists() and target.read_text(encoding='utf-8') != source):
            raise ValueError(f'Existing generated output differs: {name}')
    for name, source in outputs.items():
        target = args.destination/name
        if not target.exists():
            with target.open('x', encoding='utf-8', newline='\n') as stream:
                stream.write(source)
    print(f'Generated {len(outputs)} private compact-block sources.')


if __name__ == '__main__':
    main()
