#!/usr/bin/env python3
"""Emit guarded SIMD VU kernels from locally recovered microcode.

This compiler contains instruction semantics, not game instruction sequences.
Unsupported operations fail generation rather than silently changing behavior.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from extract_vu import extract_images
from generate_vu import plan_pairs


def signed11(word: int) -> int:
    value = word & 2047
    return value-2048 if value & 1024 else value


def mask(word: int) -> int:
    return int(f'{(word >> 21) & 15:04b}'[::-1], 2)


def upper_kind(word: int) -> tuple[int, bool]:
    op = word & 63
    return (((word & 3) | ((word >> 4) & 124)), True) if op >= 60 else (op, False)


@dataclass(frozen=True)
class Kernel:
    first: int
    last: int
    dead_until: int = 0
    events: bool = False
    prefix_gate: bool = False
    value_math: bool = False


class Emitter:
    def __init__(self, image: bytes, kernel: Kernel):
        self.plans = plan_pairs(image, 1)
        self.kernel = kernel
        if kernel.first < 0 or kernel.last >= len(image) or kernel.first % 8 or kernel.last % 8 or kernel.first > kernel.last:
            raise ValueError('Invalid fused kernel extent')
        self.lines: list[str] = []
        self.known = [15]+[0]*31
        self.acc_known = 0
        self.pending: list[tuple[int, int, str]] = []
        self.event_count = 0
        self.q_ready = -1
        self.clip_ready = -1
        self.skip_regions: dict[int, tuple[int, str]] = {}
        self.skip_conditions: list[tuple[str, int]] = []
        self.fifo_pushes = 0
        self.last_flag_index = -1
        self.last_upper = 0

    def emit(self, *lines: str) -> None:
        self.lines.extend(lines)

    def vector(self, reg: int) -> str:
        return f'R[{reg}]' if self.known[reg] == 15 else f'hn(R[{reg}])'

    def broadcast(self, reg: int, component: int) -> str:
        return f'_mm_set1_ps(hlane(R[{reg}], {component}))' if self.known[reg] & (1 << component) else f'hbc(R[{reg}], {component})'

    def write_vector(self, reg: int, lanes: int, known: int) -> None:
        if reg == 0:
            raise ValueError('Fused path cannot write VF0')
        self.known[reg] = (self.known[reg] & ~lanes) | (known & lanes)

    @staticmethod
    def commit_event(index: int, kind: str) -> str:
        if kind == 'clip':
            return f'CLIPR = fe{index}_c;'
        return (f'MAC = fe{index}_m; {{ const uint32_t cur = fe{index}_s & 0xFu; '
                f'ST = (ST & 0xFF0u) | cur | ((cur | fe{index}_x) << 6); }}')

    def flush(self, kind: str, index: int | None) -> None:
        retained = []
        for event, ready, event_kind in self.pending:
            if event_kind == kind and (index is None or ready <= index):
                self.emit(self.commit_event(event, kind))
            else:
                retained.append((event, ready, event_kind))
        self.pending = retained

    def flag_event(self, pair, index: int, clip: bool = False) -> int:
        event = self.event_count
        self.event_count += 1
        kind = 'clip' if clip else 'mac'
        if pair.immediate_flags:
            self.flush(kind, None)
            self.emit(self.commit_event(event, kind))
        else:
            self.pending.append((event, index+4, kind))
            self.fifo_pushes += 1
        self.last_flag_index = index
        self.last_upper = pair.upper
        return event

    def upper(self, pair, index: int) -> None:
        word = pair.upper
        op, special = upper_kind(word)
        fs, ft, fd, lanes = (word >> 11) & 31, (word >> 16) & 31, (word >> 6) & 31, mask(word)
        if special and op == 0x2f:
            return
        if not lanes:
            raise ValueError('Zero-mask fused operation is unsupported')
        dest = 'ACC' if special else f'R[{fd}]'
        dead = pair.pc < self.kernel.dead_until
        if special and 0x10 <= op <= 0x17:
            scale = (1, 16, 4096, 32768)[op & 3]
            if op < 0x14:
                expr = f'_mm_cvtepi32_ps(_mm_castps_si128(R[{fs}]))'
                if scale != 1:
                    expr = f'_mm_div_ps({expr}, _mm_set1_ps({scale}.0f))'
                known = 15
            else:
                expr = f'hftoi(R[{fs}], {scale}.0f)'
                known = 0
            self.emit(f'R[{ft}] = _mm_blend_ps(R[{ft}], {expr}, {lanes});')
            self.write_vector(ft, lanes, known)
            return
        if special and op == 0x1f:
            self.emit(f'WCLIP = ((WCLIP << 6) | (hclip(R[{fs}], R[{ft}]) & 0x3Fu)) & 0xFFFFFFu;'+
                      (f' fe{self.event_count}_c = WCLIP;' if self.kernel.events else ' const uint32_t clipEntry = WCLIP;'))
            if self.kernel.events:
                self.flag_event(pair, index, True)
            else:
                if self.clip_ready >= 0:
                    raise ValueError('Multiple clip writes require event mode')
                self.clip_ready = index+4
                self.fifo_pushes += int(not pair.immediate_flags)
            return
        if op == 0x2e:
            if not self.kernel.events:
                raise ValueError('Cross-product requires event mode')
            subtract = not special
            self.emit('{', f'const __m128 vsN = hn(R[{fs}]), vtN = hn(R[{ft}]);')
            if subtract:
                self.emit('const __m128 accN = hn(ACC);')
            self.emit('float res[4]; uint8_t lf[4] = {0u, 0u, 0u, 0u}; uint32_t sticky = 0u;',
                      f'hcross(vsN, vtN, '+('accN' if subtract else '_mm_setzero_ps()')+f', {lanes}u, '+('true' if subtract else 'false')+', res, lf, sticky);',
                      f'hflaglanes({lanes}u, lf, fe{self.event_count}_m, fe{self.event_count}_s); fe{self.event_count}_x = sticky;')
            self.flag_event(pair, index)
            self.emit(f'{dest} = _mm_blend_ps({dest}, _mm_loadu_ps(res), {lanes});', '}')
            if special:
                self.acc_known &= ~lanes
            else:
                self.write_vector(fd, lanes, 0)
            return
        if not special and (0x10 <= op <= 0x17 or op in (0x1d, 0x1f, 0x2b, 0x2f)):
            is_min = (0x14 <= op <= 0x17) or op in (0x1f, 0x2f)
            rhs = self.broadcast(ft, op & 3) if op <= 0x17 else ('_mm_set1_ps(IR)' if op < 0x20 else self.vector(ft))
            self.emit(f'{dest} = _mm_blend_ps({dest}, _mm_'+('min' if is_min else 'max')+f'_ps(R[{fs}], {rhs}), {lanes});')
            # Only inputs known ordinary are used by the verified profiles.
            self.write_vector(fd, lanes, self.known[fs])
            return
        if op <= 0x0f:
            operation = ('add', 'sub', 'madd', 'msub')[op//4]
            rhs = self.broadcast(ft, op & 3)
        elif 0x18 <= op <= 0x1b:
            operation, rhs = 'mul', self.broadcast(ft, op & 3)
        elif op in (0x1c, 0x1e):
            operation, rhs = 'mul', '_mm_set1_ps('+('Q' if op == 0x1c else 'IR')+')'
        elif 0x20 <= op <= 0x27:
            operation = ('add', 'madd', 'add', 'madd', 'sub', 'msub', 'sub', 'msub')[op-0x20]
            rhs = '_mm_set1_ps('+('Q' if (op & 2) == 0 else 'IR')+')'
        elif op in (0x28, 0x29, 0x2a, 0x2c, 0x2d):
            operation = {0x28:'add', 0x29:'madd', 0x2a:'mul', 0x2c:'sub', 0x2d:'msub'}[op]
            rhs = self.vector(ft)
        else:
            raise ValueError(f'Unsupported fused upper opcode {op:x}')
        if special and op not in tuple(range(0x10))+tuple(range(0x18,0x1d))+(0x1e,)+tuple(range(0x20,0x2e)):
            raise ValueError('Unsupported accumulator operation')
        if operation == 'mul':
            result = 'prod'
        elif operation in ('add', 'sub'):
            result = f'_mm_{operation}_ps(vs, r)'
        else:
            acc = 'ACC' if self.acc_known == 15 else 'hn(ACC)'
            result = f'_mm_'+('add' if operation=='madd' else 'sub')+f'_ps({acc}, prod)'
        value_only = self.kernel.value_math and dead and operation in ('mul', 'add', 'sub')
        self.emit('{', f'const __m128 vs = {self.vector(fs)};', f'const __m128 r = {rhs};',
                  'const __m128 prod = _mm_mul_ps(vs, r);', f'const __m128 res = '+(f'hn({result})' if value_only else result)+';')
        if not value_only:
            self.emit('const uint32_t resZero = vu1n::vuZeroLanes(res);')
            if operation in ('add', 'sub'):
                ordinary = 'vu1n::vuOrdinaryLanes(res) | resZero'
            else:
                self.emit('const uint32_t opZero = vu1n::vuZeroLanes(vs) | vu1n::vuZeroLanes(r);')
                ordinary = 'vu1n::vuOrdinaryLanes(res) | (opZero & resZero)'
                if operation in ('madd', 'msub'):
                    ordinary = '(vu1n::vuOrdinaryLanes(prod) | opZero) & ('+ordinary+')'
            self.emit(f'const uint32_t ordinary = {ordinary};', f'if (({lanes}u & ~ordinary) != 0u) return false;')
        if not dead:
            if not self.kernel.events:
                self.emit('const uint32_t negMask = static_cast<uint32_t>(_mm_movemask_ps(res));')
            self.emit('uint32_t sticky = 0u;')
            if operation in ('madd', 'msub'):
                self.emit(f'if ((static_cast<uint32_t>(_mm_movemask_ps(prod)) & {lanes}u) != 0u) sticky |= 0x2u;',
                          f'if ((vu1n::vuZeroLanes(prod) & {lanes}u) != 0u) sticky |= 0x1u;')
            if self.kernel.events:
                event = self.event_count
                self.emit(f'hflagbits({lanes}u, static_cast<uint32_t>(_mm_movemask_ps(res)), resZero, fe{event}_m, fe{event}_s); fe{event}_x = sticky;')
                self.flag_event(pair, index)
            else:
                self.emit(f'hflags({lanes}u, negMask, resZero, sticky, MAC, ST);')
                self.fifo_pushes += int(not pair.immediate_flags)
                self.last_flag_index = index
                self.last_upper = word
        self.emit(f'{dest} = _mm_blend_ps({dest}, res, {lanes});', '}')
        if special:
            self.acc_known |= lanes
        else:
            self.write_vector(fd, lanes, 15)

    def lower(self, pair, index: int) -> None:
        word = pair.lower
        if pair.upper & (1 << 31):
            self.emit(f'{{ float imm; const uint32_t bits = 0x{word:08x}u; std::memcpy(&imm, &bits, 4); IR = Vu1NativeAccess::normOp(imm); }}')
            return
        op = word >> 25
        fs, ft = (word >> 11) & 31, (word >> 16) & 31
        vs, vt, vd = fs & 15, ft & 15, (word >> 6) & 15
        lanes, raw_lanes, offset = mask(word), (word >> 21) & 15, signed11(word)
        if op == 0:
            if word == 0:
                return
            self.emit(f'{{ const uint32_t addr = hlqaddr(VI[{vs}], {offset}); loadAddr[nLoad] = addr; loadIdx[nLoad++] = {index};',
                      f'float t[4]; std::memcpy(t, mem + addr, 16); R[{ft}] = _mm_blend_ps(R[{ft}], _mm_loadu_ps(t), {lanes}); }}')
            self.write_vector(ft, lanes, 0)
        elif op == 1:
            self.emit(f'{{ const uint32_t addr = hlqaddr(VI[{vt}], {offset}); float w[4]; _mm_storeu_ps(w, R[{fs}]);',
                      f'storeAddr[nStore] = addr; storeLanes[nStore] = {raw_lanes}u; storeIdx[nStore] = {index}; std::memcpy(storeWords[nStore++], w, 16); }}')
        elif op == 4:
            if not raw_lanes or raw_lanes & (raw_lanes-1):
                raise ValueError('Fused ILW needs one lane')
            component = 3-(raw_lanes.bit_length()-1)
            self.emit(f'{{ const uint32_t addr = hlqaddr(VI[{vs}], {offset}); loadAddr[nLoad] = addr; loadIdx[nLoad++] = {index}; uint32_t w; std::memcpy(&w, mem + addr + {component*4}, 4);',
                      f'VI[{vt}] = static_cast<int32_t>(static_cast<int16_t>(w & 0xFFFFu)); }}')
        elif op == 5:
            self.emit(f'{{ const uint32_t addr = hlqaddr(VI[{vs}], {offset}); const uint32_t val = static_cast<uint32_t>(static_cast<uint16_t>(VI[{vt}] & 0xFFFF)); const uint32_t w[4] = {{val, val, val, val}};',
                      f'storeAddr[nStore] = addr; storeLanes[nStore] = {raw_lanes}u; storeIdx[nStore] = {index}; std::memcpy(storeWords[nStore++], w, 16); }}')
        elif op == 8:
            immediate = (word & 2047) | ((word >> 10) & 0x7800)
            self.emit(f'oldBranchVi = VI[{vt}]; branchViReg = {vt}; VI[{vt}] = static_cast<int16_t>(VI[{vs}] + {immediate});')
        elif op == 0x12:
            if self.kernel.events:
                self.flush('clip', index)
            else:
                if index < self.clip_ready or self.clip_ready < 0:
                    raise ValueError('Clip read precedes its modeled write')
                self.emit('CLIPR = clipEntry;')
            self.emit(f'VI[1] = ((CLIPR & 0x{word & 0xffffff:06x}u) != 0u) ? 1 : 0;')
        elif op == 0x1a:
            if not self.kernel.events:
                raise ValueError('MAC read needs event mode')
            self.flush('mac', index)
            self.emit(f'VI[{vt}] = static_cast<int32_t>(MAC & static_cast<uint32_t>(static_cast<uint16_t>(VI[{vs}])));')
        elif op in (0x28, 0x29):
            target = (pair.pc+8+offset*8) & 0x3fff
            condition = f'static_cast<int16_t>(VI[{vs}]) '+('==' if op==0x28 else '!=')+f' static_cast<int16_t>(VI[{vt}])'
            if target == self.kernel.first:
                self.emit(f'const bool loopTaken = {condition};')
            elif self.kernel.first <= target <= self.kernel.last:
                if target <= pair.pc+16:
                    raise ValueError('Unsupported fused internal branch')
                skip_name = f'skip{index}'
                count = (target-pair.pc-16)//8
                self.skip_regions[pair.pc+16] = (target, skip_name)
                self.skip_conditions.append((skip_name, count))
                self.emit(f'const bool {skip_name} = {condition};')
            else:
                self.emit(f'if ({condition}) return false;')
        elif op == 0x40:
            direct = word & 63
            special = (word & 3) | ((word >> 4) & 124)
            if direct == 0x35:
                self.emit(f'oldBranchVi = VI[{vd}]; branchViReg = {vd}; VI[{vd}] = VI[{vs}] | VI[{vt}];')
            elif direct < 60:
                raise ValueError(f'Unsupported fused lower direct opcode {direct:x}')
            elif special == 0x30:
                if ft != 0 and lanes:
                    self.emit(f'R[{ft}] = _mm_blend_ps(R[{ft}], R[{fs}], {lanes});')
                    self.write_vector(ft, lanes, self.known[fs])
            elif special == 0x38:
                if self.q_ready >= 0:
                    raise ValueError('Multiple division writes unsupported')
                self.q_ready = index+7
                self.emit(f'{{ float num = hlane(R[{fs}], {(word >> 21) & 3}), den = hlane(R[{ft}], {(word >> 23) & 3}); num = Vu1NativeAccess::normOp(num); den = Vu1NativeAccess::normOp(den);',
                          'uint32_t di = 0u; float result;',
                          'if (den == 0.0f) { di = num == 0.0f ? 0x10u : 0x20u; result = std::signbit(num) != std::signbit(den) ? -std::numeric_limits<float>::max() : std::numeric_limits<float>::max(); }',
                          'else result = num / den;',
                          'result = hnormres(result); qPending = hnormres(result); diPending = di & 0x30u; }')
            elif special == 0x3c:
                self.emit(f'oldBranchVi = VI[{vt}]; branchViReg = {vt}; {{ uint32_t fval; float t = hlane(R[{fs}], {(word >> 21) & 3}); std::memcpy(&fval, &t, 4); VI[{vt}] = static_cast<int32_t>(static_cast<int16_t>(fval & 0xFFFFu)); }}')
            else:
                raise ValueError(f'Unsupported fused lower special opcode {special:x}')
        else:
            raise ValueError(f'Unsupported fused lower opcode {op:x}')

    def generate(self) -> str:
        close_at = -1
        for index, pc in enumerate(range(self.kernel.first, self.kernel.last+1, 8)):
            pair = self.plans[pc//8]
            if pair.upper & 0x58000000:
                raise ValueError('Fused kernel cannot contain an end or debug halt')
            if pc == close_at:
                self.emit('}')
                close_at = -1
            if pc in self.skip_regions:
                if close_at >= 0:
                    raise ValueError('Nested fused branch unsupported')
                close_at, condition = self.skip_regions[pc]
                self.emit(f'if (!{condition}) {{')
            if self.kernel.prefix_gate and index == 0:
                self.emit('if (!prefixExecuted || iterations != 0u) {')
            self.emit(f'// 0x{pc:04x} locally decoded'+(' noflags' if pc < self.kernel.dead_until else ''))
            if index == self.q_ready:
                self.emit('Q = qPending; { const uint32_t cur = diPending; ST = (ST & 0xFCFu) | cur | (cur << 6); }')
            self.upper(pair, index)
            self.lower(pair, index)
            if self.kernel.prefix_gate and index == 0:
                self.emit('}')
        if close_at >= 0 or self.pending or self.q_ready > index:
            raise ValueError('Unresolved fused branch or flag events at exit')
        if self.kernel.events:
            events = [f'fe{event}_{kind}' for event in range(self.event_count) for kind in 'msxc']
            self.lines[:0] = ['uint32_t '+', '.join(f'{name} = 0u' for name in events)+';',
                             ' '.join(f'(void){name};' for name in events)]
            skipped = ' + '.join(f'({condition} ? {count}u : 0u)' for condition,count in self.skip_conditions) or '0u'
            self.emit(f'const uint32_t hleSkipped = {skipped};',
                      f'const uint32_t hlePairs = {(self.kernel.last-self.kernel.first)//8+1}u - hleSkipped;',
                      f'const uint64_t hleLastFlagCycle = c0 + {self.last_flag_index}u - ({skipped});',
                      f'const uint32_t hleFifoPushes = {self.fifo_pushes}u;',
                      f'const uint32_t hleLastUpper = 0x{self.last_upper:08x}u;')
        return '// Generated locally from the supplied executable. Do not redistribute.\n'+'\n'.join(self.lines)+'\n'


def generated_sources(image: bytes) -> dict[str,str]:
    plans = plan_pairs(image, 1)
    parameters = ['// Generated locally. Instruction metadata must remain private.',
                  'namespace locallyGeneratedVu {']
    for name, pc, member in (('hle1028LastUpper', 0x1200, 'upper'),
                             ('hle40FirstUpper', 0x128, 'upper'),
                             ('hle40FirstLower', 0x128, 'lower')):
        parameters.append(f'constexpr uint32_t {name} = 0x{getattr(plans[pc//8], member):08x}u;')
    parameters += ['}', '']
    return {'fused_parameters.inc': '\n'.join(parameters),
            'ps2_vu1_hle1028.inc': Emitter(image, Kernel(0x10f8, 0x1200, 0x11a0)).generate(),
            'ps2_vu1_hle1028_values.inc': Emitter(image, Kernel(0x10f8, 0x1200, 0x11a0, value_math=True)).generate(),
            'ps2_vu1_hle40.inc': Emitter(image, Kernel(0x128, 0x200, events=True, prefix_gate=True)).generate()}


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
    print(f'Generated {len(outputs)} private fused-kernel sources.')


if __name__ == '__main__':
    main()
