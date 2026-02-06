import numpy as np
import pytest

from takeskip import takeskip


@pytest.mark.parametrize(
    "command, out",
    [
        ("t4", [0, 1, 0, 1]),
        ("s4", []),
        ("s4t4", [0, 0, 0, 0]),
        ("t4 b4 t4", [0, 1, 0, 1, 0, 1, 0, 1]),
        ("p1,3,5-8", [0, 0, 0, 0, 0, 0]),
    ],
)
class TestCommands:
    def setup_class(self):
        self.NUM_ROWS = 5
        self.array = np.array([0, 1, 0, 1, 0, 0, 0, 0], dtype="u1")

    def test_1d(self, command: str, out: list[int]):
        result = takeskip(command, self.array)
        assert result.tolist() == out

    def test_2d(self, command: str, out: list[int]):
        array = np.tile(self.array, (self.NUM_ROWS, 1))
        result = takeskip(command, array)
        for idx in range(self.NUM_ROWS):
            assert result[idx].tolist() == out


# import numpy as np
# import pytest

# from takeskip import takeskip


def test_takeskip_word_s1t8s1():
    data = np.array(range(256), dtype=">u2") << 1
    mask = np.tile(np.array([0, 512], dtype=">u2"), reps=len(data) // 2)
    data = np.bitwise_xor(data, mask)
    data = data.astype(">u2")

    array = data.view("u1").reshape(-1, 2)
    array = np.unpackbits(array, axis=-1)
    array = array[..., -10:]

    out = takeskip("s1t8s1", array)
    assert out.shape[-1] == 8

    packed = np.packbits(out, axis=-1).flatten()
    assert packed.tolist() == list(range(256))
