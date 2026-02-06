"""Comprehensive test suite for takeskip module.

Tests cover:
- Basic commands (take, skip, reverse, invert, etc.)
- Padding operations (zeros, ones, data)
- Permutation operations
- Grouping and repetition
- Remnant handling
- Multi-dimensional arrays
- Edge cases and error handling
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from takeskip import takeskip
from takeskip.commands import (
    Backup,
    Data,
    Invert,
    Ones,
    Reverse,
    ReverseInvert,
    Skip,
    Take,
    Zeros,
)
from takeskip.parser import one_based_range_to_indices, parse_command


class TestBasicCommands:
    """Test basic take-skip commands."""

    def test_take_command(self):
        """Test taking bits from array."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        result = takeskip("t4", bits)
        expected = np.array([1, 0, 1, 1], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_skip_command(self):
        """Test skipping bits."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        result = takeskip("s4t4", bits)
        expected = np.array([0, 0, 1, 0], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_reverse_command(self):
        """Test reversing bits."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        result = takeskip("r8", bits)
        expected = np.array([0, 1, 0, 0, 1, 1, 0, 1], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_invert_command(self):
        """Test inverting bits."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        result = takeskip("i8", bits)
        expected = np.array([0, 1, 0, 0, 1, 1, 0, 1], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_reverse_invert_command(self):
        """Test reverse and invert combination."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        result = takeskip("ri8", bits)
        # First reverse: [0,1,0,0,1,1,0,1]
        # Then invert: [1,0,1,1,0,0,1,0]
        expected = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_backup_command(self):
        """Test backing up pointer position."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        result = takeskip("t4b2t4", bits)
        # Take [1,0,1,1], backup 2, take from position 2: [1,1,0,0]
        expected = np.array([1, 0, 1, 1, 1, 1, 0, 0], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_combined_commands(self):
        """Test multiple commands in sequence."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        result = takeskip("t2s2r4", bits)
        # Take [1,0], skip 2, reverse [0,0,1,0] -> [0,1,0,0]
        expected = np.array([1, 0, 0, 1, 0, 0], dtype=np.uint8)
        assert_array_equal(result, expected)


class TestPaddingCommands:
    """Test padding operations."""

    def test_zero_padding(self):
        """Test padding with zeros."""
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        result = takeskip("t2z4t2", bits)
        expected = np.array([1, 0, 0, 0, 0, 0, 1, 1], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_one_padding(self):
        """Test padding with ones."""
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        result = takeskip("t2n4t2", bits)
        expected = np.array([1, 0, 1, 1, 1, 1, 1, 1], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_data_padding_1d(self):
        """Test padding with literal data on 1D array."""
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        result = takeskip("t2d101t2", bits)
        expected = np.array([1, 0, 1, 0, 1, 1, 1], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_data_padding_2d(self):
        """Test padding with literal data on 2D array."""
        bits = np.array([[1, 0, 1, 1], [0, 1, 0, 1]], dtype=np.uint8)
        result = takeskip("t2d10", bits)
        expected = np.array([[1, 0, 1, 0], [0, 1, 1, 0]], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_multiple_padding_types(self):
        """Test mixing different padding types."""
        bits = np.array([1, 1], dtype=np.uint8)
        result = takeskip("z2t1n2t1d101", bits)
        expected = np.array([0, 0, 1, 1, 1, 1, 1, 0, 1], dtype=np.uint8)
        assert_array_equal(result, expected)


class TestPermutation:
    """Test permutation operations."""

    def test_permute_simple_indices(self):
        """Test permutation with simple comma-separated indices."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        result = takeskip("p1,3,5", bits)
        # 1-based: bits 1,3,5 -> 0-based indices 0,2,4 -> [1,1,0]
        expected = np.array([1, 1, 0], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_permute_range_forward(self):
        """Test permutation with forward range."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        result = takeskip("p1-4", bits)
        expected = np.array([1, 0, 1, 1], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_permute_range_backward(self):
        """Test permutation with backward range."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        result = takeskip("p4-1", bits)
        # 4-1 means bits 4,3,2,1 (1-based) -> indices 3,2,1,0 -> [1,1,0,1]
        expected = np.array([1, 1, 0, 1], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_permute_mixed(self):
        """Test permutation with mixed indices and ranges."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        result = takeskip("p1-3,8,6-5", bits)
        # 1-3: [1,0,1], 8: [0], 6-5: [0,0]
        expected = np.array([1, 0, 1, 0, 0, 0], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_permute_duplicate_indices(self):
        """Test permutation can duplicate bits."""
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        result = takeskip("p1,1,1", bits)
        expected = np.array([1, 1, 1], dtype=np.uint8)
        assert_array_equal(result, expected)


class TestGroupingAndRepetition:
    """Test grouping and repetition syntax."""

    def test_simple_repetition(self):
        """Test repeating a grouped command."""
        bits = np.array([1, 0] * 6, dtype=np.uint8)
        result = takeskip("(t2s2)3", bits)
        # Repeat "take 2, skip 2" three times
        expected = np.array([1, 0, 1, 0, 1, 0], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_complex_repetition(self):
        """Test repetition with multiple commands."""
        bits = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
        result = takeskip("(t1s1)4", bits)
        # Take every other bit
        expected = np.array([1, 1, 1, 1], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_nested_groups(self):
        """Test nested grouping (via sequential groups)."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        result = takeskip("(t2r2)2", bits)
        # (take 2, reverse 2) twice
        # First: take [1,0], reverse [1,1] -> [1,1]
        # Second: take [0,0], reverse [1,0] -> [0,1]
        expected = np.array([1, 0, 1, 1, 0, 0, 0, 1], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_repetition_zero(self):
        """Test repetition with count of 0 raises error."""
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        with pytest.raises(ValueError):
            takeskip("(t4)0", bits)


class TestRemnantHandling:
    """Test different remnant handling modes."""

    def test_remnant_remove_default(self):
        """Test default remnant removal."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        result = takeskip("t4", bits)
        expected = np.array([1, 0, 1, 1], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_remnant_remove_explicit(self):
        """Test explicit remnant removal."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        result = takeskip("t4", bits, remnant="remove")
        expected = np.array([1, 0, 1, 1], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_remnant_keep(self):
        """Test keeping remaining bits."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        result = takeskip("t4", bits, remnant="keep")
        expected = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_remnant_pad(self):
        """Test padding to original length."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        result = takeskip("t4", bits, remnant="pad")
        expected = np.array([1, 0, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_remnant_pad_no_remaining(self):
        """Test padding when no bits remain."""
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        result = takeskip("t4", bits, remnant="pad")
        expected = np.array([1, 0, 1, 1], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_remnant_keep_no_remaining(self):
        """Test keeping when no bits remain."""
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        result = takeskip("t4", bits, remnant="keep")
        expected = np.array([1, 0, 1, 1], dtype=np.uint8)
        assert_array_equal(result, expected)


class TestMultidimensionalArrays:
    """Test operations on multi-dimensional arrays."""

    def test_2d_array_basic(self):
        """Test basic operation on 2D array."""
        bits = np.array(
            [
                [1, 0, 1, 1, 0, 0, 1, 0],
                [0, 1, 0, 1, 1, 1, 0, 1],
            ],
            dtype=np.uint8,
        )
        result = takeskip("s2t4", bits)
        expected = np.array(
            [
                [1, 1, 0, 0],
                [0, 1, 1, 1],
            ],
            dtype=np.uint8,
        )
        assert_array_equal(result, expected)

    def test_3d_array(self):
        """Test operation on 3D array."""
        bits = np.array(
            [
                [[1, 0, 1, 1], [0, 1, 0, 1]],
                [[1, 1, 0, 0], [0, 0, 1, 1]],
            ],
            dtype=np.uint8,
        )
        result = takeskip("t2r2", bits)
        expected = np.array(
            [
                [[1, 0, 1, 1], [0, 1, 1, 0]],
                [[1, 1, 0, 0], [0, 0, 1, 1]],
            ],
            dtype=np.uint8,
        )
        assert_array_equal(result, expected)

    def test_2d_array_with_padding(self):
        """Test padding on 2D array."""
        bits = np.array(
            [
                [1, 0, 1, 1],
                [0, 1, 0, 1],
            ],
            dtype=np.uint8,
        )
        result = takeskip("t2z2", bits)
        expected = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.uint8,
        )
        assert_array_equal(result, expected)

    def test_2d_array_remnant_pad(self):
        """Test remnant padding on 2D array."""
        bits = np.array(
            [
                [1, 0, 1, 1, 0, 0],
                [0, 1, 0, 1, 1, 1],
            ],
            dtype=np.uint8,
        )
        result = takeskip("t4", bits, remnant="pad")
        expected = np.array(
            [
                [1, 0, 1, 1, 0, 0],
                [0, 1, 0, 1, 0, 0],
            ],
            dtype=np.uint8,
        )
        assert_array_equal(result, expected)


class TestCaseInsensitivity:
    """Test that commands are case insensitive."""

    def test_lowercase(self):
        """Test lowercase commands."""
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        result = takeskip("t2s1t1", bits)
        expected = np.array([1, 0, 1], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_uppercase(self):
        """Test uppercase commands."""
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        result = takeskip("T2S1T1", bits)
        expected = np.array([1, 0, 1], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_mixed_case(self):
        """Test mixed case commands."""
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        result = takeskip("T2s1T1", bits)
        expected = np.array([1, 0, 1], dtype=np.uint8)
        assert_array_equal(result, expected)


class TestWhitespace:
    """Test that whitespace is ignored."""

    def test_with_spaces(self):
        """Test commands with spaces."""
        bits = np.array([1, 0, 1, 1, 0, 0], dtype=np.uint8)
        result = takeskip("t2 s2 t2", bits)
        expected = np.array([1, 0, 0, 0], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_with_tabs_and_newlines(self):
        """Test commands with various whitespace."""
        bits = np.array([1, 0, 1, 1, 0, 0], dtype=np.uint8)
        result = takeskip("t2\t\ns2\tt2", bits)
        expected = np.array([1, 0, 0, 0], dtype=np.uint8)
        assert_array_equal(result, expected)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_command(self):
        """Test with minimal command."""
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        result = takeskip("t0", bits)
        expected = np.array([], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_skip_all(self):
        """Test skipping all bits."""
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        result = takeskip("s4", bits)
        expected = np.array([], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_take_more_than_available(self):
        """Test taking more bits than available."""
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        result = takeskip("t10", bits)
        # Should only take available bits
        expected = np.array([1, 0, 1, 1], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_single_bit_array(self):
        """Test with single bit."""
        bits = np.array([1], dtype=np.uint8)
        result = takeskip("t1", bits)
        expected = np.array([1], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_zero_length_array(self):
        """Test with empty array."""
        bits = np.array([], dtype=np.uint8)
        result = takeskip("t0", bits)
        expected = np.array([], dtype=np.uint8)
        assert_array_equal(result, expected)


class TestErrorHandling:
    """Test error conditions and exceptions."""

    def test_invalid_remnant_value(self):
        """Test invalid remnant parameter raises TypeError."""
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        with pytest.raises(TypeError):  # Accept either until fix applied
            takeskip("t4", bits, remnant="invalid")  # type: ignore

    def test_invalid_command_syntax(self):
        """Test invalid command syntax raises exception."""
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        with pytest.raises(Exception):  # Lark will raise parse error
            takeskip("x4", bits)

    def test_malformed_permutation(self):
        """Test malformed permutation syntax."""
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        with pytest.raises(Exception):
            takeskip("p1-", bits)


class TestCommandObjects:
    """Test command object creation and behavior."""

    def test_take_object(self):
        """Test Take command object."""
        cmd = Take(4)
        bits = np.array([1, 0, 1, 1, 0, 0], dtype=np.uint8)
        result, ptr = cmd(bits, 0)
        expected = np.array([1, 0, 1, 1], dtype=np.uint8)
        assert_array_equal(result, expected)
        assert ptr == 4

    def test_skip_object(self):
        """Test Skip command object."""
        cmd = Skip(2)
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        result, ptr = cmd(bits, 0)
        assert result.shape[-1] == 0  # Empty result
        assert ptr == 2

    def test_reverse_object(self):
        """Test Reverse command object."""
        cmd = Reverse(4)
        bits = np.array([1, 0, 1, 1, 0, 0], dtype=np.uint8)
        result, ptr = cmd(bits, 0)
        expected = np.array([1, 1, 0, 1], dtype=np.uint8)
        assert_array_equal(result, expected)
        assert ptr == 4

    def test_invert_object(self):
        """Test Invert command object."""
        cmd = Invert(4)
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        result, ptr = cmd(bits, 0)
        expected = np.array([0, 1, 0, 0], dtype=np.uint8)
        assert_array_equal(result, expected)
        assert ptr == 4

    def test_reverse_invert_object(self):
        """Test ReverseInvert command object."""
        cmd = ReverseInvert(4)
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        result, ptr = cmd(bits, 0)
        # Reverse [1,0,1,1] -> [1,1,0,1]
        # Invert [1,1,0,1] -> [0,0,1,0]
        expected = np.array([0, 0, 1, 0], dtype=np.uint8)
        assert_array_equal(result, expected)
        assert ptr == 4

    def test_backup_object(self):
        """Test Backup command object."""
        cmd = Backup(2)
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        result, ptr = cmd(bits, 4)
        assert result.shape[-1] == 0  # Empty result
        assert ptr == 2

    def test_zeros_object(self):
        """Test Zeros command object."""
        cmd = Zeros(4)
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        result, ptr = cmd(bits, 0)
        expected = np.array([0, 0, 0, 0], dtype=np.uint8)
        assert_array_equal(result, expected)
        assert ptr == 0  # Padding doesn't advance pointer

    def test_ones_object(self):
        """Test Ones command object."""
        cmd = Ones(3)
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        result, ptr = cmd(bits, 0)
        expected = np.array([1, 1, 1], dtype=np.uint8)
        assert_array_equal(result, expected)
        assert ptr == 0

    def test_data_object(self):
        """Test Data command object."""
        cmd = Data("101")
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        result, ptr = cmd(bits, 0)
        expected = np.array([1, 0, 1], dtype=np.uint8)
        assert_array_equal(result, expected)
        assert ptr == 0

    def test_command_equality(self):
        """Test command equality comparison."""
        cmd1 = Take(4)
        cmd2 = Take(4)
        cmd3 = Take(5)
        cmd4 = Skip(4)

        assert cmd1 == cmd2
        assert cmd1 != cmd3
        assert cmd1 != cmd4

    def test_command_repr(self):
        """Test command string representation."""
        cmd = Take(4)
        assert repr(cmd) == "Take(4)"


class TestParserFunctions:
    """Test parser utility functions."""

    def test_one_based_range_forward(self):
        """Test one-based range conversion (forward)."""
        result = one_based_range_to_indices(1, 4)
        expected = np.array([0, 1, 2, 3], dtype=np.int64)
        assert_array_equal(result, expected)

    def test_one_based_range_backward(self):
        """Test one-based range conversion (backward)."""
        result = one_based_range_to_indices(4, 1)
        expected = np.array([3, 2, 1, 0], dtype=np.int64)
        assert_array_equal(result, expected)

    def test_one_based_range_single(self):
        """Test one-based range with single element."""
        result = one_based_range_to_indices(5, 5)
        expected = np.array([4], dtype=np.int64)
        assert_array_equal(result, expected)

    def test_parse_simple_command(self):
        """Test parsing simple command string."""
        result = parse_command("t4s2")
        assert len(result) == 2
        assert isinstance(result[0], Take)
        assert isinstance(result[1], Skip)
        assert result[0].value == 4
        assert result[1].value == 2

    def test_parse_complex_command(self):
        """Test parsing complex command string."""
        result = parse_command("t2r4i2")
        assert len(result) == 3
        assert isinstance(result[0], Take)
        assert isinstance(result[1], Reverse)
        assert isinstance(result[2], Invert)


class TestRealWorldScenarios:
    """Test realistic use cases."""

    def test_nibble_swap(self):
        """Test swapping nibbles in a byte."""
        bits = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
        result = takeskip("s4t4b8t4", bits)
        expected = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_extract_every_other_bit(self):
        """Test extracting every other bit."""
        bits = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
        result = takeskip("(t1s1)4", bits)
        expected = np.array([1, 1, 1, 1], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_interleave_with_zeros(self):
        """Test interleaving all bits with zeros."""
        bits = np.array([1, 1, 0, 0], dtype=np.uint8)
        result = takeskip("(t1z1)4", bits)
        expected = np.array([1, 0, 1, 0, 0, 0, 0, 0], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_extract_specific_fields(self):
        """Test extracting specific bit fields."""
        # Simulate a packed format: [3 bits header][5 bits data][8 bits checksum]
        bits = np.array(
            [1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint8
        )
        # Extract just the 5-bit data field
        result = takeskip("s3t5", bits)
        expected = np.array([1, 1, 0, 0, 1], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_bit_reverse_byte(self):
        """Test reversing bits in a byte."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        result = takeskip("r8", bits)
        expected = np.array([0, 1, 0, 0, 1, 1, 0, 1], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_manchester_decode_simulation(self):
        """Test Manchester-like decoding (take every other bit)."""
        # In Manchester encoding, each bit is represented by 2 bits
        encoded = np.array([1, 0, 1, 0, 0, 1, 0, 1], dtype=np.uint8)
        # Decode by taking first bit of each pair
        result = takeskip("(t1s1)4", encoded)
        expected = np.array([1, 1, 0, 0], dtype=np.uint8)
        assert_array_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
