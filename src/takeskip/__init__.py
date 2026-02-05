from typing import Literal, Optional

import numpy as np
import numpy.typing as npt

from takeskip.commands import Backup, parse_command


def takeskip(
    command: str,
    array: npt.NDArray[np.uint8],
    *,
    remnant: Literal["remove", "keep", "pad"] = "remove",
) -> np.ndarray:
    """Perform a take-skip style operation.

    Take-Skip operations are a syntax of selecting (and potentially manipulating) bits from
    a sequence of bits.

    The command string is made up of individual command elements. Each element is
    a string containing a letter and a number. The letter represents the operation, and the
    number is the number of bits for the operation.

    Valid operations include:
        t: take    - take bits (no manipulation)
        s: skip    - skip bits
        r: reverse - reverse the order of bits
        i: invert  - invert
        o: ones    - pad with 1
        z: zeros   - pad with 0

    For example:
        * s4t4 - skip 4 bits, take 4 bits
        * t4r4 - take 4 bits, reverse 4 bits

    Note:
        * Commands are case insensitive and ignore whitespace.
        * In "word" mode, the command string determines the resulting array.word_size
        * In "row" mode, the command string must result in a total length that is a multiple
        of the output array.word_size. Pad bits can be used to ensure the result length is valid
        if necessary.

    Args:
        command: The string expressing the operation to perform.
        array: The target of the operation `VarUIntArray`.
        mode: Whether to execute the operation on each "word" or "row".
        word_size: For "row" mode, word_size can specify a new output word size.

    Returns:
        The resulting array.

    Raises:
        ValueError: If there is an error in the take-skip command syntax.
    """
    if remnant not in ["remove", "keep", "pad"]:
        msg = "invalid remnant argument"
        raise ValueError(msg)

    commands = parse_command(command)
    result_size = sum(i.result_size for i in commands)

    if result_size == 0:
        msg = "Command would result in output with word_size 0."
        raise ValueError(msg)

    input_size = array.shape[-1]

    # Reshape into one input length per row
    unpacked = array.reshape(-1, input_size)

    # Determine the command's resulting output length in bits
    if remnant == "remove":
        output_size = result_size
    else:
        # If "keep" or "pad", ensure the output size is the input size, plus any additional bits
        # added by backing up in the stream
        delta_size = sum([c.value for c in commands if isinstance(c, Backup)])
        output_size = input_size + delta_size
    result = np.zeros([*unpacked.shape[:-1], output_size], dtype="u1")

    in_ptr = 0
    out_ptr = 0
    for cmd in commands:
        manipulated_bits = cmd(unpacked[:, in_ptr : in_ptr + cmd.input_size])
        if manipulated_bits is not None:
            result[:, out_ptr : out_ptr + cmd.result_size] = manipulated_bits

        in_ptr += cmd.input_size
        out_ptr += cmd.result_size

    if remnant == "keep":
        result[:, out_ptr:] = unpacked[:, out_ptr:]

    return result
