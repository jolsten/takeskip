"""Command classes for take-skip operations.

This module defines the command classes that implement individual take-skip operations.
Each command takes an array and a pointer position, performs its operation, and returns
the result along with an updated pointer position.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


def empty_array_like(array: np.ndarray) -> np.ndarray:
    """Create an empty array matching input shape except with 0-length last dimension.

    Args:
        array: Input array to match shape.

    Returns:
        Array with same shape except last dimension is 0 (effectively null for np.hstack).
    """
    return np.zeros((*array.shape[0:-1], 0), dtype="u1")


class Command(ABC):
    """Base class for all take-skip commands.

    All commands implement a __call__ method that processes an array starting at
    a given pointer position and returns results plus updated pointer.

    Attributes:
        value: The parameter value for this command (e.g., number of bits).
    """

    def __init__(self, value: Any) -> None:
        """Initialize command with a value parameter.

        Args:
            value: The parameter for this command (type varies by subclass).
        """
        self.value = value

    def __repr__(self) -> str:
        """Return string representation of the command.

        Returns:
            String in format "ClassName(value)".
        """
        return f"{self.__class__.__name__}({self.value})"

    def __eq__(self, other) -> bool:
        """Check equality with another Command.

        Args:
            other: Another object to compare.

        Returns:
            True if both are same class with equal values.

        Raises:
            TypeError: If other is not a Command instance.
        """
        if not isinstance(other, Command):
            return NotImplemented

        if isinstance(self.value, np.ndarray):
            a = self.value.tolist()
            b = other.value.tolist()
            if a != b:
                return False
        elif self.value != other.value:
            return False

        if self.__class__ != other.__class__:
            return False

        return True

    @abstractmethod
    def __call__(
        self,
        array: np.ndarray,
        in_ptr: int,
    ) -> tuple[np.ndarray, int]:
        """Execute the command on an array.

        Args:
            array: Input array (uint8 binary data).
            in_ptr: Current pointer position in the array.

        Returns:
            Tuple of (result_array, updated_pointer).
        """
        ...


class Take(Command):
    """Take n bits from current position without modification.

    Attributes:
        value: Number of bits to take.
    """

    def __init__(self, value: int) -> None:
        """Initialize Take command.

        Args:
            value: Number of bits to take.
        """
        self.value = int(value)

    def __call__(self, array: np.ndarray, ptr: int) -> tuple[np.ndarray, int]:
        """Take bits from array and advance pointer.

        Args:
            array: Input binary array.
            ptr: Current pointer position.

        Returns:
            Tuple of (selected_bits, new_pointer_position).
        """
        result = array[..., ptr : ptr + self.value]
        ptr += self.value
        return result, ptr


class Skip(Command):
    """Skip n bits (advance pointer without outputting anything).

    Attributes:
        value: Number of bits to skip.
    """

    def __init__(self, value: int) -> None:
        """Initialize Skip command.

        Args:
            value: Number of bits to skip.
        """
        self.value = int(value)

    def __call__(self, array: np.ndarray, ptr: int) -> tuple[np.ndarray, int]:
        """Skip bits by advancing pointer without output.

        Args:
            array: Input binary array.
            ptr: Current pointer position.

        Returns:
            Tuple of (empty_array, new_pointer_position).
        """
        result = empty_array_like(array)
        ptr += self.value
        return result, ptr


class Invert(Take):
    """Take n bits and invert them (0->1, 1->0).

    Attributes:
        value: Number of bits to invert.
    """

    def __call__(self, array: np.ndarray, ptr: int) -> tuple[np.ndarray, int]:
        """Take and invert bits using XOR with 1.

        Args:
            array: Input binary array.
            ptr: Current pointer position.

        Returns:
            Tuple of (inverted_bits, new_pointer_position).
        """
        result = array[..., ptr : ptr + self.value]
        result = np.bitwise_xor(result, np.uint8(1))
        ptr += self.value
        return result, ptr


class Reverse(Take):
    """Take n bits and reverse their order.

    Attributes:
        value: Number of bits to reverse.
    """

    def __call__(self, array: np.ndarray, ptr: int) -> tuple[np.ndarray, int]:
        """Take bits and reverse along last axis.

        Args:
            array: Input binary array.
            ptr: Current pointer position.

        Returns:
            Tuple of (reversed_bits, new_pointer_position).
        """
        result = array[..., ptr : ptr + self.value]
        result = result[..., ::-1]
        ptr += self.value
        return result, ptr


class ReverseInvert(Take):
    """Take n bits, reverse their order, and invert them.

    Attributes:
        value: Number of bits to reverse and invert.
    """

    def __call__(self, array: np.ndarray, ptr: int) -> tuple[np.ndarray, int]:
        """Take bits, reverse, and invert them.

        Args:
            array: Input binary array.
            ptr: Current pointer position.

        Returns:
            Tuple of (reversed_and_inverted_bits, new_pointer_position).
        """
        result = array[..., ptr : ptr + self.value]
        result = result[..., ::-1]
        result = np.bitwise_xor(result, np.uint8(1))
        ptr += self.value
        return result, ptr


class Backup(Command):
    """Move pointer backward by n positions (doesn't output anything).

    Attributes:
        value: Number of positions to move backward.
    """

    def __init__(self, value: Any) -> None:
        """Initialize Backup command.

        Args:
            value: Number of positions to backup.
        """
        self.value = int(value)

    def __call__(self, array: np.ndarray, ptr: int) -> tuple[np.ndarray, int]:
        """Move pointer backward without output.

        Args:
            array: Input binary array.
            ptr: Current pointer position.

        Returns:
            Tuple of (empty_array, new_pointer_position).
        """
        result = empty_array_like(array)
        ptr += -self.value
        return result, ptr


class Pad(Command):
    """Base class for padding commands that insert bits without advancing pointer."""

    pass


class Zeros(Pad):
    """Insert n zero bits without advancing pointer.

    Attributes:
        value: Number of zeros to insert.
    """

    def __init__(self, value: int) -> None:
        self.value = int(value)

    def __call__(self, array: np.ndarray, ptr: int) -> tuple[np.ndarray, int]:
        """Insert zeros without advancing pointer.

        Args:
            array: Input binary array.
            ptr: Current pointer position.

        Returns:
            Tuple of (zero_array, unchanged_pointer).
        """
        result = np.zeros((*array.shape[0:-1], self.value), dtype="u1")
        return result, ptr


class Ones(Pad):
    """Insert n one bits without advancing pointer.

    Attributes:
        value: Number of ones to insert.
    """

    def __init__(self, value: int) -> None:
        self.value = int(value)

    def __call__(self, array: np.ndarray, ptr: int) -> tuple[np.ndarray, int]:
        """Insert ones without advancing pointer.

        Args:
            array: Input binary array.
            ptr: Current pointer position.

        Returns:
            Tuple of (ones_array, unchanged_pointer).
        """
        result = np.ones((*array.shape[0:-1], self.value), dtype="u1")
        return result, ptr


class Data(Pad):
    """Insert literal binary data without advancing pointer.

    Attributes:
        value: Numpy array of binary values to insert.
    """

    def __init__(self, value: str) -> None:
        """Initialize Data command from binary string.

        Args:
            value: Binary string (e.g., "101010").
        """
        self.value = np.array([int(c) for c in value], dtype="u1")

    def __call__(self, array: np.ndarray, ptr: int) -> tuple[np.ndarray, int]:
        """Insert literal data without advancing pointer.

        Args:
            array: Input binary array.
            ptr: Current pointer position.

        Returns:
            Tuple of (data_array, unchanged_pointer).
        """
        # Broadcast the data pattern to match input array's non-last dimensions
        if array.ndim > 1:
            # Repeat for each element in the non-last dimensions
            n_repeats = np.prod(array.shape[:-1])
            result = np.tile(self.value, (int(n_repeats), 1))
            result = result.reshape(*array.shape[:-1], len(self.value))
        else:
            result = self.value.copy()
        return result, ptr


class Permute(Command):
    """Reorder bits according to specified indices (1-based in input, 0-based internally).

    Attributes:
        value: Numpy array of zero-based indices for permutation.
    """

    def __init__(self, args: list[int | np.ndarray]) -> None:
        """Initialize Permute command from list of indices/ranges.

        Args:
            args: List of integers or numpy arrays representing indices.

        Raises:
            TypeError: If args contain unsupported types.
        """
        results = []
        for item in args:
            if isinstance(item, np.ndarray):
                results.append(item)
            elif isinstance(item, int):
                idx = np.array([item], dtype="int64")
                results.append(idx)
            else:
                raise TypeError(f"Unsupported type in permute args: {type(item)}")
        self.value = np.concatenate(results)

    def __call__(self, array: np.ndarray, ptr: int) -> tuple[np.ndarray, int]:
        """Permute bits according to stored indices.

        Args:
            array: Input binary array.
            ptr: Current pointer position (not used, permute is absolute).

        Returns:
            Tuple of (permuted_bits, new_pointer_position).

        Note:
            Pointer advances by (max_index + 1) to move past all referenced bits.
        """
        result = array[..., self.value]
        # Advance by the maximum bit number seen in the list
        ptr += int(np.max(self.value)) + 1
        return result, ptr
