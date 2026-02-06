from abc import abstractmethod
from typing import Any, Union

import numpy as np


def empty_array_like(array: np.ndarray) -> np.ndarray:
    # An array with the same shape as the input, except the final dimension which is size 0
    # Effectively a null array when used with np.hstack
    return np.zeros((*array.shape[0:-1], 0), dtype="u1")


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

    @abstractmethod
    def __call__(
        self,
        array: np.ndarray,
        in_ptr: int,
    ) -> tuple[np.ndarray, int]: ...


class Take(Command):
    def __init__(self, value: int) -> None:
        self.value = int(value)

    def __call__(self, array: np.ndarray, ptr: int) -> tuple[np.ndarray, int]:
        result = array[..., ptr : ptr + self.value]
        ptr += self.value
        return result, ptr


class Skip(Command):
    def __init__(self, value: int) -> None:
        self.value = int(value)

    def __call__(self, array: np.ndarray, ptr: int) -> tuple[np.ndarray, int]:
        result = empty_array_like(array)
        ptr += self.value
        return result, ptr


class Invert(Take):
    def __call__(self, array: np.ndarray, ptr: int) -> tuple[np.ndarray, int]:
        result = array[..., ptr : ptr + self.value]
        result = np.bitwise_xor(result, np.uint8(1))
        ptr += self.value
        return result, ptr


class Reverse(Take):
    def __call__(self, array: np.ndarray, ptr: int) -> tuple[np.ndarray, int]:
        result = array[..., ptr : ptr + self.value]
        result = result[..., ::-1]
        ptr += self.value
        return result, ptr


class Backup(Command):
    def __init__(self, value: Any) -> None:
        self.value = int(value)

    def __call__(self, array: np.ndarray, ptr: int) -> tuple[np.ndarray, int]:
        result = empty_array_like(array)
        ptr += -self.value
        return result, ptr


class Pad(Command):
    pass


class Zeros(Pad):
    def __call__(self, array: np.ndarray, ptr: int) -> tuple[np.ndarray, int]:
        result = np.zeros((*array.shape[0:-1], self.value), dtype="u1")
        return result, ptr


class Ones(Pad):
    def __call__(self, array: np.ndarray, ptr: int) -> tuple[np.ndarray, int]:
        result = np.ones((*array.shape[0:-1], self.value), dtype="u1")
        return result, ptr


class Data(Pad):
    def __init__(self, value: str) -> None:
        self.value = np.array([int(c) for c in value], dtype="u1")

    def __call__(self, array: np.ndarray, ptr: int) -> tuple[np.ndarray, int]:
        result = np.repeat(self.value, repeats=(*array.shape[-1:], 1))
        return result, ptr


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
        print(self.value)

    def __call__(self, array: np.ndarray, ptr: int) -> tuple[np.ndarray, int]:
        result = array[..., self.value]
        # advance by the maximum bit number seen in the list
        ptr += np.max(self.value) + 1
        return result, ptr
