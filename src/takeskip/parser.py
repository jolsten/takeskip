"""Parser for take-skip command strings.

This module uses Lark to parse command strings into command objects that can be
executed on binary arrays.
"""

import pathlib

import numpy as np
import numpy.typing as npt
from lark import Lark, Transformer, v_args

from takeskip.commands import (
    Backup,
    Command,
    Data,
    Invert,
    Ones,
    Permute,
    Reverse,
    ReverseInvert,
    Skip,
    Take,
    Zeros,
)

grammar = (pathlib.Path(__file__).parent / "takeskip.lark").read_text()
command_parser = Lark(grammar, parser="earley")


def one_based_range_to_indices(start: int, end: int) -> npt.NDArray[np.int64]:
    """Convert one-based range to zero-based indices.

    Ranges are inclusive on both ends. Direction (forward/backward) is preserved.

    Args:
        start: One-based start position (integer).
        end: One-based end position (integer).

    Returns:
        Numpy array of zero-based indices.

    Examples:
        >>> one_based_range_to_indices(1, 4)
        array([0, 1, 2, 3])
        >>> one_based_range_to_indices(4, 1)
        array([3, 2, 1, 0])
        >>> one_based_range_to_indices(5, 5)
        array([4])
    """
    # Convert to zero-based
    start_idx = start - 1
    end_idx = end - 1

    # Determine step direction
    step = 1 if start <= end else -1
    return np.arange(start_idx, end_idx + step, step, dtype=np.int64)


class CommandParser(Transformer):
    """Lark transformer that converts parse tree into Command objects.

    This class defines methods matching the grammar rules in takeskip.lark.
    Each method transforms a parse tree node into the appropriate Command object.
    """

    @v_args(inline=True)
    def integer(self, s: str) -> int:
        """Convert string token to integer.

        Args:
            s: String representation of an integer.

        Returns:
            Integer value.
        """
        return int(s)

    def flatten(self, args: list) -> list:
        """Flatten nested lists into a single list.

        Used to handle grouped commands and nested structures.

        Args:
            args: List that may contain nested lists.

        Returns:
            Flattened list of commands.
        """
        result = []
        for item in args:
            if isinstance(item, list):
                result.extend(item)
            else:
                result.append(item)
        return result

    def repeat(self, args: list) -> list:
        """Repeat a sequence of commands n times.

        Args:
            args: List where all but last element are commands, last is repeat count.

        Returns:
            List of commands repeated n times.
        """
        *commands, n = args
        return commands * n

    @v_args(inline=True)
    def take(self, n: int) -> Take:
        """Create Take command.

        Args:
            n: Number of bits to take.

        Returns:
            Take command instance.
        """
        return Take(n)

    @v_args(inline=True)
    def skip(self, n: int) -> Skip:
        """Create Skip command.

        Args:
            n: Number of bits to skip.

        Returns:
            Skip command instance.
        """
        return Skip(n)

    @v_args(inline=True)
    def invert(self, n: int) -> Invert:
        """Create Invert command.

        Args:
            n: Number of bits to invert.

        Returns:
            Invert command instance.
        """
        return Invert(n)

    @v_args(inline=True)
    def reverse(self, n: int) -> Reverse:
        """Create Reverse command.

        Args:
            n: Number of bits to reverse.

        Returns:
            Reverse command instance.
        """
        return Reverse(n)

    @v_args(inline=True)
    def reverse_invert(self, n: int) -> ReverseInvert:
        """Create ReverseInvert command.

        Args:
            n: Number of bits to reverse and invert.

        Returns:
            ReverseInvert command instance.
        """
        return ReverseInvert(n)

    @v_args(inline=True)
    def backup(self, n: int) -> Backup:
        """Create Backup command.

        Args:
            n: Number of positions to move pointer backward.

        Returns:
            Backup command instance.
        """
        return Backup(n)

    @v_args(inline=True)
    def zero_pad(self, n: int) -> Zeros:
        """Create Zeros padding command.

        Args:
            n: Number of zero bits to insert.

        Returns:
            Zeros command instance.
        """
        return Zeros(n)

    @v_args(inline=True)
    def one_pad(self, n: int) -> Ones:
        """Create Ones padding command.

        Args:
            n: Number of one bits to insert.

        Returns:
            Ones command instance.
        """
        return Ones(n)

    @v_args(inline=True)
    def data_pad(self, s: str) -> Data:
        """Create Data padding command from binary string.

        Args:
            s: Binary string (e.g., "101010").

        Returns:
            Data command instance.
        """
        return Data(s)

    @v_args(inline=True)
    def range(self, a: int, b: int) -> npt.NDArray[np.int64]:
        """Convert range notation to array of indices.

        Args:
            a: Start position (1-based).
            b: End position (1-based).

        Returns:
            Array of zero-based indices.
        """
        return one_based_range_to_indices(a, b)

    def csv(self, args: list) -> list[npt.NDArray[np.int64]]:
        """Process comma-separated values (indices and ranges).

        Converts 1-based single indices to 0-based arrays for consistency.

        Args:
            args: List of integers and/or numpy arrays (ranges).

        Returns:
            List of numpy arrays containing zero-based indices.
        """
        result = []
        for x in args:
            if isinstance(x, int):
                # Convert to zero-based, like the ranges already are
                idx = np.array([x - 1], dtype="int64")
                result.append(idx)
            else:
                result.append(x)
        return result

    @v_args(inline=True)
    def permute(self, args: list) -> Permute:
        """Create Permute command from list of indices/ranges.

        Args:
            args: List of numpy arrays representing indices.

        Returns:
            Permute command instance.
        """
        return Permute(args)


def parse_command(s: str) -> list[Command]:
    """Parse a command string into a list of Command objects.

    Args:
        s: Command string (e.g., "s4t8r4" or "(t8s8)3").

    Returns:
        List of Command objects ready to execute.

    Raises:
        Various Lark exceptions if the command string has invalid syntax.

    Examples:
        >>> parse_command("t4s2")
        [Take(4), Skip(2)]
        >>> parse_command("(t8s8)2")
        [Take(8), Skip(8), Take(8), Skip(8)]
    """
    tree = command_parser.parse(s)
    return CommandParser().transform(tree)
