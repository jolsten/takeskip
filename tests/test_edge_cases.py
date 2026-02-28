"""Edge case and property-based tests for takeskip."""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_array_equal

from takeskip import takeskip
from takeskip.commands import Skip, Take

# --- Strategies ---

binary_arrays = arrays(
    dtype=np.uint8,
    shape=st.integers(min_value=1, max_value=64),
    elements=st.integers(min_value=0, max_value=1),
)

binary_arrays_2d = arrays(
    dtype=np.uint8,
    shape=st.tuples(
        st.integers(min_value=1, max_value=8),
        st.integers(min_value=1, max_value=64),
    ),
    elements=st.integers(min_value=0, max_value=1),
)


# --- Edge case tests: pointer bounds ---


class TestPointerBounds:
    """Test that out-of-bounds pointer raises ValueError."""

    def test_backup_past_beginning(self):
        """Backup past position 0 should raise ValueError."""
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        with pytest.raises(ValueError, match="cannot be negative"):
            takeskip("b8t4", bits)

    def test_backup_past_beginning_after_take(self):
        """Backup more than current position should raise ValueError."""
        bits = np.array([1, 0, 1, 1, 0, 0], dtype=np.uint8)
        with pytest.raises(ValueError, match="cannot be negative"):
            takeskip("t2b4t3", bits)

    def test_skip_past_end(self):
        """Skip past array end should raise ValueError."""
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        with pytest.raises(ValueError, match="exceeds array length"):
            takeskip("s10t4", bits)

    def test_take_past_end(self):
        """Take past array end should raise ValueError."""
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        with pytest.raises(ValueError, match="exceeds array length"):
            takeskip("t10", bits)

    def test_backup_past_beginning_2d(self):
        """Out-of-bounds pointer raises on 2D arrays too."""
        bits = np.array([[1, 0, 1, 1], [0, 1, 0, 1]], dtype=np.uint8)
        with pytest.raises(ValueError, match="cannot be negative"):
            takeskip("b8t4", bits)

    def test_backup_exact_is_valid(self):
        """Backup exactly to 0 should work."""
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        result = takeskip("t4b4t4", bits)
        expected = np.array([1, 0, 1, 1, 1, 0, 1, 1], dtype=np.uint8)
        assert_array_equal(result, expected)

    def test_skip_exact_end_is_valid(self):
        """Skipping exactly to the end should work."""
        bits = np.array([1, 0, 1, 1], dtype=np.uint8)
        result = takeskip("s4", bits)
        expected = np.array([], dtype=np.uint8)
        assert_array_equal(result, expected)


# --- Edge case tests: __eq__ ---


class TestCommandEquality:
    """Test __eq__ with non-Command types returns NotImplemented."""

    def test_eq_with_non_command(self):
        """Comparing Command with non-Command should not raise."""
        cmd = Take(4)
        assert cmd != "not a command"
        assert cmd != 4
        assert cmd != None  # noqa: E711

    def test_eq_in_list_membership(self):
        """Command should work with 'in' operator on mixed lists."""
        cmd = Take(4)
        assert cmd not in [1, 2, "hello"]
        assert cmd in [Skip(2), Take(4)]

    def test_eq_different_class_same_value(self):
        """Different command classes with same value should not be equal."""
        assert Take(4) != Skip(4)


# --- Property-based tests ---


class TestPropertyBased:
    """Property-based tests using hypothesis."""

    @given(bits=binary_arrays)
    def test_take_all_is_identity(self, bits):
        """Taking all bits should return the original array."""
        n = len(bits)
        result = takeskip(f"t{n}", bits)
        assert_array_equal(result, bits)

    @given(bits=binary_arrays)
    def test_reverse_twice_is_identity(self, bits):
        """Reversing twice should return the original array."""
        n = len(bits)
        intermediate = takeskip(f"r{n}", bits)
        result = takeskip(f"r{n}", intermediate)
        assert_array_equal(result, bits)

    @given(bits=binary_arrays)
    def test_invert_twice_is_identity(self, bits):
        """Inverting twice should return the original array."""
        n = len(bits)
        intermediate = takeskip(f"i{n}", bits)
        result = takeskip(f"i{n}", intermediate)
        assert_array_equal(result, bits)

    @given(bits=binary_arrays)
    def test_remnant_keep_with_take_all(self, bits):
        """Take all + keep remnant should equal take all."""
        n = len(bits)
        result = takeskip(f"t{n}", bits, remnant="keep")
        assert_array_equal(result, bits)

    @given(bits=binary_arrays)
    def test_remnant_pad_preserves_length(self, bits):
        """Pad remnant should preserve original array length."""
        n = len(bits)
        take_n = n // 2
        result = takeskip(f"t{take_n}", bits, remnant="pad")
        assert result.shape[-1] == n

    @given(bits=binary_arrays)
    def test_skip_n_take_rest_equals_slice(self, bits):
        """Skip n then take rest should equal array[n:]."""
        n = len(bits)
        skip_n = n // 3
        take_n = n - skip_n
        result = takeskip(f"s{skip_n}t{take_n}", bits)
        assert_array_equal(result, bits[skip_n:])

    @given(bits=binary_arrays)
    def test_take_skip_partition(self, bits):
        """Take n + skip rest with keep should equal original."""
        n = len(bits)
        take_n = n // 2
        skip_n = n - take_n
        result = takeskip(f"t{take_n}s{skip_n}", bits, remnant="keep")
        # take_n bits taken + 0 remaining (skip consumed rest)
        assert_array_equal(result, bits[:take_n])

    @given(bits=binary_arrays)
    def test_output_dtype_is_uint8(self, bits):
        """Output should always be uint8."""
        result = takeskip("t1", bits)
        assert result.dtype == np.uint8

    @given(bits=binary_arrays)
    def test_reverse_invert_equals_separate_operations(self, bits):
        """ri<n> should equal reversing then inverting the result."""
        n = len(bits)
        result_ri = takeskip(f"ri{n}", bits)
        intermediate = takeskip(f"r{n}", bits)
        result_separate = takeskip(f"i{n}", intermediate)
        assert_array_equal(result_ri, result_separate)

    @given(bits=binary_arrays_2d)
    def test_take_all_2d_is_identity(self, bits):
        """Taking all bits on 2D array should return the original."""
        n = bits.shape[-1]
        result = takeskip(f"t{n}", bits)
        assert_array_equal(result, bits)

    @given(bits=binary_arrays_2d)
    def test_reverse_twice_2d_is_identity(self, bits):
        """Reversing twice on 2D array should return the original."""
        n = bits.shape[-1]
        intermediate = takeskip(f"r{n}", bits)
        result = takeskip(f"r{n}", intermediate)
        assert_array_equal(result, bits)

    @given(
        bits=binary_arrays,
        data=st.data(),
    )
    def test_zero_padding_inserts_zeros(self, bits, data):
        """Zero padding should insert actual zeros."""
        pad_n = data.draw(st.integers(min_value=1, max_value=16))
        result = takeskip(f"z{pad_n}", bits)
        assert_array_equal(result, np.zeros(pad_n, dtype=np.uint8))

    @given(
        bits=binary_arrays,
        data=st.data(),
    )
    def test_one_padding_inserts_ones(self, bits, data):
        """One padding should insert actual ones."""
        pad_n = data.draw(st.integers(min_value=1, max_value=16))
        result = takeskip(f"n{pad_n}", bits)
        assert_array_equal(result, np.ones(pad_n, dtype=np.uint8))

    @given(bits=binary_arrays)
    def test_skip_all_gives_empty(self, bits):
        """Skipping all bits should give empty array."""
        n = len(bits)
        result = takeskip(f"s{n}", bits)
        assert result.shape[-1] == 0

    @given(bits=binary_arrays)
    def test_backup_past_beginning_raises(self, bits):
        """Backup past position 0 should raise ValueError."""
        n = len(bits)
        with pytest.raises(ValueError, match="cannot be negative"):
            takeskip(f"b{n + 10}t{n}", bits)

    @given(bits=binary_arrays)
    @settings(max_examples=50)
    def test_permute_identity(self, bits):
        """Permuting with 1..n should be identity."""
        n = len(bits)
        assume(n > 0)
        result = takeskip(f"p1-{n}", bits)
        assert_array_equal(result, bits)

    @given(bits=binary_arrays)
    @settings(max_examples=50)
    def test_permute_reverse_equals_reverse(self, bits):
        """Permuting with n..1 should equal reverse."""
        n = len(bits)
        assume(n > 0)
        result_permute = takeskip(f"p{n}-1", bits)
        result_reverse = takeskip(f"r{n}", bits)
        assert_array_equal(result_permute, result_reverse)
