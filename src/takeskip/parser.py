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
    Skip,
    Take,
    Zeros,
)

grammar = (pathlib.Path(__file__).parent / "takeskip.lark").read_text()
command_parser = Lark(grammar, parser="earley")


def one_based_range_to_indices(start, end) -> npt.NDArray[np.int64]:
    """
    Convert one-based range to zero-based indices.

    Args:
        start: One-based start position (integer)
        end: One-based end position (integer)

    Returns:
        List of zero-based indices

    Examples:
        1-4 -> [0, 1, 2, 3]
        4-1 -> [3, 2, 1, 0]
        5-5 -> [4]
    """
    # Convert to zero-based
    start_idx = start - 1
    end_idx = end - 1

    # Determine step direction
    step = 1 if start <= end else -1
    return np.arange(start_idx, end_idx + step, step)


class CommandParser(Transformer):
    @v_args(inline=True)
    def integer(self, s: str):
        return int(s)

    def flatten(self, args):
        result = []
        for item in args:
            if isinstance(item, list):
                result.extend(item)
            else:
                result.append(item)
        return result

    def repeat(self, args):
        print(repr(args))
        *commands, n = args
        return commands * n

    @v_args(inline=True)
    def take(self, n: int) -> Take:
        return Take(n)

    @v_args(inline=True)
    def skip(self, n: int) -> Skip:
        return Skip(n)

    @v_args(inline=True)
    def invert(self, n: int) -> Invert:
        return Invert(n)

    @v_args(inline=True)
    def reverse(self, n: int) -> Reverse:
        return Reverse(n)

    @v_args(inline=True)
    def backup(self, n: int) -> Backup:
        return Backup(n)

    @v_args(inline=True)
    def zero_pad(self, n: int) -> Zeros:
        return Zeros(n)

    @v_args(inline=True)
    def one_pad(self, n: int) -> Ones:
        return Ones(n)

    @v_args(inline=True)
    def data_pad(self, s: str) -> Data:
        return Data(s)

    @v_args(inline=True)
    def range(self, a, b) -> npt.NDArray[np.int64]:
        return one_based_range_to_indices(a, b)

    def csv(self, args) -> list[npt.NDArray[np.int64]]:
        result = []
        for x in args:
            if isinstance(x, int):
                # convert to zero-based, like the ranges already are
                idx = np.array([x - 1], dtype="int64")
                result.append(idx)
            else:
                result.append(x)
        return result

    @v_args(inline=True)
    def permute(self, args) -> Permute:
        print("args =", args)
        return Permute(args)


def parse_command(s: str) -> list[Command]:
    tree = command_parser.parse(s)
    return CommandParser().transform(tree)
