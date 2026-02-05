import pathlib
from abc import abstractmethod
from typing import Any, Optional, Union

import numpy as np
from lark import Lark, Transformer, v_args

grammar = (pathlib.Path(__file__).parent / "takeskip.lark").read_text()
command_parser = Lark(grammar, parser="earley")


class Command:
    def __init__(self, value: Any) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Command):
            raise TypeError

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

    @property
    def input_size(self) -> int:
        """The consumed number of input bits."""
        return self.value

    @property
    @abstractmethod
    def result_size(self) -> int:
        """The output number of bits."""
        ...

    @abstractmethod
    def __call__(self, array: np.ndarray) -> tuple[np.ndarray, int]: ...


class Take(Command):
    def __init__(self, value: int) -> None:
        self.value = int(value)

    @property
    def result_size(self) -> int:
        return self.value

    def __call__(self, array: np.ndarray, ptr: int) -> tuple[np.ndarray, int]:
        return array[..., 0 : self.value]


class Skip(Command):
    def __init__(self, value: int) -> None:
        self.value = int(value)

    @property
    def result_size(self) -> int:
        return 0

    def __call__(self, array: np.ndarray) -> Optional[np.ndarray]:
        return None


class Invert(Take):
    def __call__(self, array: np.ndarray) -> Optional[np.ndarray]:
        return np.bitwise_xor(array, np.uint8(1))


class Reverse(Take):
    def __call__(self, array: np.ndarray) -> Optional[np.ndarray]:
        return array[..., ::-1]


class Backup(Command):
    def __init__(self, value: Any) -> None:
        self.value = int(value)

    @property
    def input_size(self) -> int:
        return -self.value

    @property
    def result_size(self) -> int:
        return 0


class Pad(Command):
    @property
    def input_size(self) -> int:
        return 0

    @property
    def result_size(self) -> int:
        return self.value


class Ones(Pad):
    def __call__(self, array: np.ndarray) -> Optional[np.ndarray]:
        return np.ones((*array.shape[0:-1], self.value), dtype="u1")


class Zeros(Pad):
    def __call__(self, array: np.ndarray) -> Optional[np.ndarray]:
        return np.zeros((*array.shape[0:-1], self.value), dtype="u1")


class Data(Pad):
    def __init__(self, value: str) -> None:
        self.value = str(value)

    def __call__(self, array: np.ndarray) -> Optional[np.ndarray]:
        raise NotImplementedError


class Permute(Command):
    def __init__(self, args: list[Union[int, tuple[int, int]]]) -> None:
        results = []
        for item in args:
            if isinstance(item, np.ndarray):
                results.append(item)
            elif isinstance(item, int):
                idx = np.array([item], dtype="int64")
                results.append(idx)
            else:
                raise TypeError
        self.value = np.concatenate(results)

    @property
    def input_size(self) -> int:
        return max(self.value) + 1

    @property
    def result_size(self) -> int:
        return len(self.value)

    def __call__(self, array: np.ndarray) -> Optional[np.ndarray]:
        return array[..., self.value]


def one_based_range_to_indices(start, end):
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
    def range(self, a, b) -> tuple[int, int]:
        return one_based_range_to_indices(a, b)

    @v_args(inline=True)
    def csv(self, first, *rest) -> list[int]:
        result = [first]
        for x in rest:
            if isinstance(x, int):
                # convert to zero-based
                idx = np.array([x - 1], dtype="int64")
                result.append(idx)
            else:
                result.append(x)
        return result

    @v_args(inline=True)
    def permute(self, args) -> Permute:
        return Permute(args)


def parse_command(s: str) -> list[Command]:
    tree = command_parser.parse(s)
    return CommandParser().transform(tree)
