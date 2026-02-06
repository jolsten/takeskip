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


# example_256 = VarUIntArray(range(256), word_size=8)


# def test_s4t4():
#     out = takeskip("s4t4", example_256, mode="word")
#     assert out.word_size == 4
#     assert out.tolist() == (np.arange(256, dtype="u1") % 16).tolist()


# def test_t8():
#     out = takeskip("t8", example_256, mode="word")
#     assert out.word_size == 8
#     assert out.tolist() == example_256.tolist()


# def test_i8():
#     out = takeskip("i8", example_256, mode="word")
#     assert out.word_size == 8
#     assert out.tolist() == list(reversed(range(256)))


# @pytest.mark.parametrize(
#     "word_size, value, reverse",
#     [
#         (8, 0, 0),
#         (3, 0b001, 0b100),
#         (8, 0x55, 0xAA),
#     ],
# )
# def test_r8(word_size: int, value: int, reverse: int):
#     LEN = 10
#     array = VarUIntArray([value] * LEN, word_size=word_size)
#     out = takeskip(f"r{word_size}", array, mode="word")
#     assert out.word_size == word_size
#     assert out.tolist() == [reverse] * LEN


# class TestComplex1:
#     def setup_class(self):
#         self.unpacked = VarUIntArray([0xAF, 0x0A], word_size=8)

#     def test_permute_1(self):
#         out = takeskip("p1-4", self.unpacked, mode="row")
#         assert out.tolist() == [0xA]

#     def test_permute_2(self):
#         out = takeskip("p4-1", self.unpacked, mode="row")
#         assert out.tolist == [0xF]
