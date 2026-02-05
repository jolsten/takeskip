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
    parse_command,
)


@pytest.mark.parametrize(
    "s, expected",
    [
        ("s8", [Skip(8)]),
        ("i8", [Invert(8)]),
        ("t4", [Take(4)]),
        ("r8", [Reverse(8)]),
        ("n8", [Ones(8)]),
        ("z8", [Zeros(8)]),
        ("s1 t8 s1", [Skip(1), Take(8), Skip(1)]),
        ("p1-4", [Permute([0, 1, 2, 3])]),
        ("p4-1", [Permute([3, 2, 1, 0])]),
        ("t4 b4 i4", [Take(4), Backup(4), Invert(4)]),
    ],
)
def test_parser(s: str, expected: list[Command]):
    instructions = parse_command(s)
    assert instructions == expected
