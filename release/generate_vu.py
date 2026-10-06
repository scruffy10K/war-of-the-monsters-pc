#!/usr/bin/env python3
"""Compile locally recovered VU microcode into native C++ specializations.

No game instructions are stored in this generator. Output is private game data.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import struct

from extract_vu import extract_images

BRANCH_OPS = frozenset((0x20, 0x21, 0x24, 0x25, 0x28, 0x29, 0x2c, 0x2d, 0x2e, 0x2f))


@dataclass(frozen=True)
class Pair:
    pc: int
    upper: int
    lower: int
    immediate_flags: bool
    mid: bool
    set_pc: bool
    fallthrough: bool


def plan_pairs(image: bytes, unit: int) -> list[Pair]:
    if unit not in (0, 1) or len(image) != (4096 if unit == 0 else 16384):
        raise ValueError("Expected a complete VU0 or VU1 code image")
    words = tuple(struct.iter_unpack("<II", image))
    count = len(words)

    def op(index: int) -> int:
        lower, upper = words[index % count]
        return -1 if upper & 0x80000000 else lower >> 25

    def control(index: int) -> bool:
        return op(index) in BRANCH_OPS or bool(words[index % count][1] & 0x58000000)

    @lru_cache(maxsize=None)
    def flag_barrier(index: int, remaining: int, pending: tuple[int, ...] = ()) -> bool:
        index %= count
        kind = op(index)
        # Flag observations/writes, indirect destinations and exit points must
        # retain delayed flags. I-bit pairs have no lower instruction.
        if words[index][1] & 0x58000000 or kind in (0x24, 0x25) or 0x10 <= kind <= 0x1c:
            return True
        if remaining == 1:
            return False
        if pending:
            return any(flag_barrier(target, remaining - 1) for target in pending)
        if kind in BRANCH_OPS:
            offset = words[index][0] & 2047
            if offset & 1024:
                offset -= 2048
            target = (index + 1 + offset) % count
            targets = (target,) if kind in (0x20, 0x21) else (target, (index + 2) % count)
            return flag_barrier((index + 1) % count, remaining - 1, targets)
        return flag_barrier((index + 1) % count, remaining - 1)

    plans = []
    run_length = 0
    previous_mid = False
    for index, (lower, upper) in enumerate(words):
        run_length += 1
        previous_control = index > 0 and control(index - 1)
        # The relaxed compiler polls at most every 48 pairs. Branches and
        # their delay slots always publish PC and process branch/halt state.
        cut = index == count - 1 or (run_length >= 48 and not control(index))
        falls = not (previous_control or cut)
        mid = unit == 1 and not (control(index) or previous_control or cut)
        plans.append(Pair(index * 8, upper, lower,
                          unit == 1 and not flag_barrier(index, 6),
                          mid, not mid and previous_mid, falls))
        previous_mid = mid
        if not falls:
            run_length = 0
    return plans


def native_source(image: bytes, unit: int, key: str, hooks: dict[int, str] | None = None) -> str:
    hooks = hooks or {}
    name = f"vu1NativeProgram_{key}"
    lines = ['// Generated locally from the supplied game executable. Do not redistribute.',
             '#include "ps2_vu1_native.h"']
    for hook in sorted(set(hooks.values())):
        lines.append(f'bool {hook}(VU1Interpreter &, Vu1NativeCtx &, Vu1NativeProgram);')
    lines += ['namespace {', f'Vu1NativeStep {name}(VU1Interpreter &v, Vu1NativeCtx &c);',
              f'Vu1NativeStep {name}_c0(VU1Interpreter &v, Vu1NativeCtx &c) {{',
              '    Vu1NativeStep r;', '    switch (Vu1NativeAccess::pc(v)) {']
    for pair in plan_pairs(image, unit):
        lines.append(f'    case 0x{pair.pc:04x}u:')
        if pair.pc in hooks:
            lines += [f'        if ({hooks[pair.pc]}(v, c, &{name}))',
                      '            return Vu1NativeStep::Continue;']
        if pair.set_pc:
            lines.append(f'        Vu1NativeAccess::setPc(v, 0x{pair.pc:04x}u);')
        flags = [pair.immediate_flags, unit == 1, unit == 1, False, pair.mid]
        if unit == 0:
            flags.append(True)
        flag_text = ', '.join('true' if flag else 'false' for flag in flags)
        lines += [f'        if ((r = Vu1NativeAccess::stepPair<0x{pair.upper:08x}u, 0x{pair.lower:08x}u, {flag_text}>(v, c)) != Vu1NativeStep::Continue)',
                  '            return r;',
                  '        [[fallthrough]];' if pair.fallthrough else '        break;']
    lines += ['    default: return Vu1NativeStep::Fallback;', '    }',
              '    return Vu1NativeStep::Continue;', '}',
              f'Vu1NativeStep {name}(VU1Interpreter &v, Vu1NativeCtx &c) {{',
              '    for (;;) {',
              '        if (Vu1NativeAccess::stopRequested(v) || Vu1NativeAccess::cycle(v) >= c.budgetEnd)',
              '            return Vu1NativeStep::Stop;',
              '        if ((Vu1NativeAccess::pc(v) >> 14) != 0)',
              '            return Vu1NativeStep::Fallback;',
              f'        const Vu1NativeStep r = {name}_c0(v, c);',
              '        if (r != Vu1NativeStep::Continue) return r;',
              '    }', '}', '}',
              f'Vu1NativeProgram ps2xVu1NativeProgram_{key}() {{ return &{name}; }}', '']
    return '\n'.join(lines)


def registry_source(items: list[dict]) -> str:
    lines = ['// Generated locally. Image identities only; no instructions.',
             '#include "ps2_vu1_native.h"']
    for item in items:
        lines.append(f'Vu1NativeProgram ps2xVu1NativeProgram_{item["id"]}();')
    lines += ['void ps2xVu1NativeRegister(uint64_t, Vu1NativeProgram) {}',
              'Vu1NativeProgram ps2xVu1NativeLookup(uint64_t imageHash) {', '    switch (imageHash) {']
    for item in items:
        key = item['id']
        lines.append(f'    case 0x{key}ull: return ps2xVu1NativeProgram_{key}();')
    lines += ['    default: return nullptr;', '    }', '}', '']
    return '\n'.join(lines)


def parameter_source(image: bytes) -> str:
    """Compile-time arguments for our generic native kernels and diagnostics."""
    if len(image) != 16384:
        raise ValueError('Expected a full VU1 image')
    lines = ['// Generated locally from the supplied executable. Do not redistribute.',
             'namespace locallyGeneratedVu {',
             'struct PairWords { uint32_t lower, upper; };',
             'inline constexpr PairWords referenceCode[] = {']
    for lower, upper in struct.iter_unpack('<II', image):
        lines.append(f'    {{0x{lower:08x}u, 0x{upper:08x}u}},')
    lines += ['};',
              'template<uint32_t PC> inline constexpr uint32_t upper = referenceCode[PC / 8u].upper;',
              'template<uint32_t PC> inline constexpr uint32_t lower = referenceCode[PC / 8u].lower;',
              '}', '']
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('elf', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    recipe = json.loads(Path(__file__).with_name('vu-extraction.json').read_text(encoding='utf-8'))
    images = extract_images(args.elf.read_bytes(), recipe)
    outputs = {}
    for item in recipe['images']:
        unit, key = item['unit'], item['id']
        # Integration point for the independently optimized geometry kernel.
        hooks = {0x128: 'hleShader40Try'} if key == '7d7edbc78086ac63' else {}
        outputs[f'vu{unit}_native_{key}.cpp'] = native_source(images[f'vu{unit}_image_{key}.bin'], unit, key, hooks)
    outputs['ps2_vu1_native_registry.cpp'] = registry_source(recipe['images'])
    outputs['native_parameters.inc'] = parameter_source(images['vu1_image_7d7edbc78086ac63.bin'])
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
    print(f'Generated {len(outputs)} private native source files from the validated executable.')


if __name__ == '__main__':
    main()
