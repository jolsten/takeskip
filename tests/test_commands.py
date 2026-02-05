import numpy as np
import pytest

from takeskip.commands import (
    Command,
    Take,
)


@pytest.mark.parametrize(
    "in_, command, out",
    [
        ([0, 1, 0, 1, 0, 0, 0, 0], Take(4), [0, 1, 0, 1]),
    ],
)
def test_command(in_: list[int], command: Command, out: list[int]):
    array = np.array(in_, dtype="u1")
    result = command(array)
    print(result)
    if result is not None:
        assert result.tolist() == out
