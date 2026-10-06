"""Synthetic, asset-free tests for local game-code generation."""
from copy import deepcopy
import hashlib
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extract_vu import extract_images, write_images
from generate_vu import plan_pairs, native_source
from generate_blocks import Profile, Hook, admitted_pcs, block_source, direct_candidates
from generate_fused import Emitter, Kernel, mask as fused_mask, signed11


def image_with(pairs, unit=1):
    image = bytearray(4096 if unit == 0 else 16384)
    for index, (lower, upper) in pairs.items():
        struct.pack_into('<II', image, index * 8, lower, upper)
    return bytes(image)


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.elf = b'synthetic input only; not an actual executable'
        self.expected = self.elf[:8] + bytes(4088)
        self.recipe = {'schema': 1, 'elf_size': len(self.elf),
            'elf_sha256': hashlib.sha256(self.elf).hexdigest(), 'images': [
                {'unit': 0, 'id': '0123456789abcdef', 'size': 4096,
                 'sha256': hashlib.sha256(self.expected).hexdigest(),
                 'spans': [{'destination': 0, 'offset': 0, 'length': 8}]}]}

    def test_exact_extraction_and_repeat(self):
        result = extract_images(self.elf, self.recipe)
        self.assertEqual(list(result.values()), [self.expected])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'generated'
            write_images(result, path)
            write_images(result, path)
            self.assertEqual((path/next(iter(result))).read_bytes(), self.expected)

    def test_wrong_input_rejected(self):
        for input_bytes in (self.elf[:-1], self.elf+b'x', b'X'+self.elf[1:]):
            with self.subTest(input_bytes=input_bytes), self.assertRaises(ValueError):
                extract_images(input_bytes, self.recipe)

    def test_span_and_identity_validation(self):
        edits = [('offset', -1), ('offset', len(self.elf)), ('length', 0),
                 ('length', 7), ('destination', 1), ('destination', 4096)]
        for key, value in edits:
            recipe = deepcopy(self.recipe)
            recipe['images'][0]['spans'][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                extract_images(self.elf, recipe)
        for key, value in [('id', '../../escape'), ('unit', 2), ('size', 8192), ('sha256', '0'*64)]:
            recipe = deepcopy(self.recipe)
            recipe['images'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                extract_images(self.elf, recipe)

    def test_overlaps_duplicates_and_unknown_schema(self):
        for case in ('overlap', 'duplicate', 'schema', 'empty'):
            recipe = deepcopy(self.recipe)
            if case == 'overlap':
                recipe['images'][0]['spans'] *= 2
            elif case == 'duplicate':
                recipe['images'] *= 2
            elif case == 'schema':
                recipe['schema'] = 99
            else:
                recipe['images'] = []
            with self.subTest(case=case), self.assertRaises(ValueError):
                extract_images(self.elf, recipe)

    def test_different_existing_output_preserved(self):
        result = extract_images(self.elf, self.recipe)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            file = path/next(iter(result))
            file.write_bytes(b'user file')
            with self.assertRaises(ValueError):
                write_images(result, path)
            self.assertEqual(file.read_bytes(), b'user file')


class NativeGenerationTests(unittest.TestCase):
    def test_vu0_keeps_pipeline_and_pc_tracking(self):
        plans = plan_pairs(bytes(4096), 0)
        self.assertTrue(all(not p.immediate_flags and not p.mid and not p.set_pc for p in plans))
        self.assertFalse(plans[47].fallthrough)

    def test_straight_runs_are_bounded(self):
        plans = plan_pairs(bytes(16384), 1)
        self.assertTrue(all(p.mid for p in plans[:47]))
        self.assertFalse(plans[47].mid)
        self.assertTrue(plans[47].set_pc)
        self.assertFalse(plans[47].fallthrough)
        self.assertFalse(plans[-1].fallthrough)

    def test_branch_and_delay_stay_together_at_budget_boundary(self):
        # B with a synthetic relative offset, located on the run's 48th pair.
        plans = plan_pairs(image_with({47: ((0x20 << 25) | 3, 0)}), 1)
        self.assertFalse(plans[47].mid)
        self.assertTrue(plans[47].fallthrough)
        self.assertFalse(plans[48].mid)
        self.assertFalse(plans[48].fallthrough)

    def test_immediate_lower_word_is_not_decoded_as_branch(self):
        plans = plan_pairs(image_with({0: (0x20 << 25, 1 << 31)}), 1)
        self.assertTrue(plans[0].mid)
        self.assertTrue(plans[1].mid)

    def test_flag_lookahead_follows_taken_branch(self):
        plans = plan_pairs(image_with({0: ((0x20 << 25) | 20, 0),
                                       21: (0x16 << 25, 0)}), 1)
        self.assertFalse(plans[0].immediate_flags)
        self.assertTrue(plans[2].immediate_flags)
        self.assertFalse(plans[21].immediate_flags)

    def test_conditional_branch_checks_both_paths(self):
        for destination in (2, 21):
            plans = plan_pairs(image_with({0: ((0x28 << 25) | 20, 0),
                                           destination: (0x16 << 25, 0)}), 1)
            self.assertFalse(plans[0].immediate_flags)

    def test_end_and_debug_halts_keep_boundaries(self):
        for bit in (30, 28, 27):
            plans = plan_pairs(image_with({9: (0, 1 << bit)}), 1)
            self.assertFalse(plans[9].mid)
            self.assertFalse(plans[10].fallthrough)
            self.assertFalse(plans[9].immediate_flags)

    def test_bad_size_and_unit(self):
        for data, unit in ((bytes(8), 1), (bytes(4096), 1), (bytes(4096), 5)):
            with self.assertRaises(ValueError):
                plan_pairs(data, unit)

    def test_output_is_deterministic(self):
        data = image_with({3: (0, 1 << 30)}, unit=0)
        self.assertEqual(native_source(data, 0, 'synthetic'), native_source(data, 0, 'synthetic'))
        self.assertEqual(native_source(data, 0, 'synthetic').count('case 0x'), 512)


class CompactGenerationTests(unittest.TestCase):
    def test_signed_branch_targets_and_wrap(self):
        plans = plan_pairs(image_with({0: ((0x20 << 25) | 2046, 0)}), 1)
        self.assertEqual(direct_candidates(plans, 1, {16, 16376}), [16376, 16])
        self.assertEqual(direct_candidates(plans, 1, {16}), [16])

    def test_immediate_and_indirect_words_do_not_supply_targets(self):
        for lower, upper in (((0x20 << 25) | 15, 1 << 31), ((0x24 << 25) | 15, 0)):
            plans = plan_pairs(image_with({0: (lower, upper)}), 1)
            self.assertEqual(direct_candidates(plans, 1, {16, 128}), [16])

    def test_rejects_invalid_regions_and_hooks(self):
        for regions in ((), ((1, 8),), ((8, 0),), ((0, 16384),), ((0, 8), (8, 16))):
            with self.subTest(regions=regions), self.assertRaises(ValueError):
                admitted_pcs(Profile('synthetic', 'L', regions))
        with self.assertRaises(ValueError):
            admitted_pcs(Profile('synthetic', 'L', ((0, 8),), hooks={0: Hook('test', targets=(16,))}))
        with self.assertRaises(ValueError):
            admitted_pcs(Profile('synthetic', 'L', ((0, 8),), exit_hints={0: (16,)}))

    def test_region_gap_cannot_silently_fall_through(self):
        with self.assertRaises(ValueError):
            block_source(bytes(16384), Profile('synthetic', 'L', ((0, 8),)))

    def test_dispatch_hints_are_guarded_and_fallback_remains(self):
        source = block_source(image_with({0: (0x24 << 25, 0)}),
            Profile('synthetic', 'L', ((0, 8),), exit_hints={8: (0,)}))
        self.assertIn('if(Vu1NativeAccess::pc(v)==0x0u)goto L0000;', source)
        self.assertIn('goto dispatch;', source)
        self.assertIn('default:return nativeProgram(v,c);', source)

    def test_instruction_arguments_come_from_input(self):
        profile = Profile('synthetic', 'L', ((0, 8),))
        first = block_source(image_with({0: (0, 1 << 30)}), profile)
        second = block_source(image_with({0: (7, 1 << 30)}), profile)
        self.assertNotEqual(first, second)
        self.assertEqual(first.count('stepPair<'), 2)
        self.assertEqual(second.count('stepPair<'), 2)


class FusedGenerationTests(unittest.TestCase):
    @staticmethod
    def upper(op, fs=1, ft=2, fd=3, dest=15):
        return op | (fd << 6) | (fs << 11) | (ft << 16) | (dest << 21)

    def test_lane_order_and_signed_offsets(self):
        self.assertEqual([fused_mask(bit << 21) for bit in (8,4,2,1)], [1,2,4,8])
        self.assertEqual([signed11(value) for value in (0,1023,1024,2047)], [0,1023,-1024,-1])

    def test_math_keeps_guards_and_instruction_operands(self):
        body = Emitter(image_with({0:(0,self.upper(0x2a))}), Kernel(0,0)).generate()
        for fragment in ('const __m128 vs = hn(R[1]);', 'const __m128 r = hn(R[2]);',
                         'const __m128 res = prod;', '~ordinary', 'hflags(',
                         'R[3] = _mm_blend_ps(R[3], res, 15);'):
            self.assertIn(fragment, body)

    def test_normalization_propagates_only_written_lanes(self):
        for lanes in (15,8):
            image = image_with({0:(0,self.upper(0x2a,dest=lanes)), 1:(0,self.upper(0x28,fs=3,fd=4))})
            body = Emitter(image, Kernel(0,8)).generate()
            self.assertIn('const __m128 vs = '+('R[3]' if lanes==15 else 'hn(R[3])')+';',body)

    def test_value_math_is_limited_to_dead_flags(self):
        image = image_with({0:(0,self.upper(0x2a))})
        dead = Emitter(image, Kernel(0,0,dead_until=8,value_math=True)).generate()
        live = Emitter(image, Kernel(0,0,value_math=True)).generate()
        self.assertIn('const __m128 res = hn(prod);',dead)
        self.assertNotIn('~ordinary',dead)
        self.assertIn('~ordinary',live)
        self.assertIn('hflags(',live)

    def test_unknown_halt_and_zero_register_writes_rejected(self):
        for upper in (self.upper(0x33), self.upper(0x2a,fd=0), self.upper(0x2a) | (1<<30)):
            with self.subTest(upper=upper), self.assertRaises(ValueError):
                Emitter(image_with({0:(0,upper)}), Kernel(0,0)).generate()

    def test_division_wait_is_preserved(self):
        nop_upper = 0x3f | (0x2c << 4)
        divide = (0x40 << 25) | 0x3c | (0x38 << 4) | (1 << 11) | (2 << 16)
        pairs = {i:(0,nop_upper) for i in range(8)}
        pairs[0] = (divide,nop_upper)
        image = image_with(pairs)
        body = Emitter(image, Kernel(0,56)).generate()
        self.assertEqual(body.count('Q = qPending;'),1)
        self.assertGreater(body.index('Q = qPending;'), body.index('// 0x0038'))
        with self.assertRaises(ValueError):
            Emitter(image, Kernel(0,48)).generate()

    def test_memory_load_invalidates_vector_proof(self):
        load = (3 << 16) | (15 << 21)
        body = Emitter(image_with({0:(load,self.upper(0x2a)), 1:(0,self.upper(0x28,fs=3,fd=4))}), Kernel(0,8)).generate()
        self.assertIn('loadIdx[nLoad++] = 0;', body)
        self.assertIn('const __m128 vs = hn(R[3]);', body)


if __name__ == '__main__':
    unittest.main()
