import pytest

from takeskip.commands import (
    Backup,
    Command,
    Invert,
    Ones,
    Permute,
    Reverse,
    Skip,
    Take,
    Zeros,
)
from takeskip.parser import one_based_range_to_indices, parse_command


@pytest.mark.parametrize(
    "a, b, values",
    [
        (1, 4, [0, 1, 2, 3]),
        (1, 1, [0]),
        (8, 8, [7]),
    ],
)
def test_one_based_ranges(a: int, b: int, values: list[int]):
    assert one_based_range_to_indices(a, b).tolist() == values


@pytest.mark.parametrize(
    "s, expected",
    [
        ("s8", (Skip(8),)),
        ("i8", (Invert(8),)),
        ("t4", (Take(4),)),
        ("r8", (Reverse(8),)),
        ("n8", (Ones(8),)),
        ("z8", (Zeros(8),)),
        ("s1 t8 s1", (Skip(1), Take(8), Skip(1))),
        ("p1-4", (Permute([0, 1, 2, 3]),)),
        ("p4-1", (Permute([3, 2, 1, 0]),)),
        ("t4 b4 i4", (Take(4), Backup(4), Invert(4))),
    ],
)
def test_parser(s: str, expected: tuple[Command, ...]):
    instructions = parse_command(s)
    assert instructions == expected
