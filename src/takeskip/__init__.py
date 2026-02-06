"""Take-Skip: A library for bit manipulation using a declarative command syntax.

This module provides functionality to perform take-skip style operations on numpy arrays
containing binary data (uint8 arrays of 0s and 1s representing bits).
"""

from typing import Literal

import numpy as np
import numpy.typing as npt

from takeskip.parser import parse_command


def takeskip(
    command: str,
    array: npt.NDArray[np.uint8],
    *,
    remnant: Literal["remove", "keep", "pad"] = "remove",
) -> np.ndarray:
    """Perform a take-skip style operation on a binary array.

    Take-Skip operations provide a syntax for selecting and manipulating bits from
    a sequence of bits using a declarative command string.

    The command string is composed of individual command elements. Each element consists
    of a letter (operation) followed by a number (bit count).

    Valid operations:
        t<n>: take n bits (no manipulation)
        s<n>: skip n bits
        r<n>: reverse the order of n bits
        i<n>: invert n bits (0->1, 1->0)
        ri<n>: reverse and invert n bits
        b<n>: backup pointer by n positions
        z<n>: pad with n zeros
        n<n>: pad with n ones
        d<binary>: pad with literal binary data (e.g., d101)
        p<indices>: permute bits using comma-separated indices or ranges (1-based)

    Grouping and repetition:
        (...): group commands
        (...)N: repeat grouped commands N times

    Examples:
        "s4t4" - skip 4 bits, take 4 bits
        "t4r4" - take 4 bits, reverse 4 bits
        "(t8s8)3" - repeat "take 8, skip 8" three times
        "p1,3,5" - permute bits at positions 1, 3, 5 (1-based indexing)
        "p1-4,8-5" - take bits 1-4 forward, then 8-5 backward

    Notes:
        * Commands are case insensitive and ignore whitespace
        * Permute (p) uses 1-based indexing, inclusive on both ends

    Args:
        command: The string expressing the operation to perform.
        array: The target array of dtype uint8 containing binary data (0s and 1s).
               Can be multidimensional; operations apply to the last axis.
        remnant: How to handle remaining bits after all commands execute:
                 - "remove": discard remaining bits (default)
                 - "keep": append remaining bits to result
                 - "pad": pad result with zeros to original length

    Returns:
        A numpy array with the same dtype (uint8) and shape as input, except
        the last dimension which depends on the command and remnant setting.

    Raises:
        ValueError: If the command syntax is incorrect
        TypeError: If the remnant argument is invalid

    Examples:
        >>> import numpy as np
        >>> bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        >>> takeskip("s2t4", bits)  # skip 2, take 4
        array([1, 1, 0, 0], dtype=uint8)
        >>> takeskip("t4r4", bits)  # take 4, reverse 4
        array([1, 0, 1, 1, 0, 0, 1, 1], dtype=uint8)
    """
    if remnant not in ["remove", "keep", "pad"]:
        msg = "invalid remnant argument; must be 'remove', 'keep', or 'pad'"
        raise TypeError(msg)

    commands = parse_command(command)

    components = []
    ptr = 0
    for cmd in commands:
        result, ptr = cmd(array, ptr)
        components.append(result)

    if remnant == "keep":
        result = array[..., ptr:]
        components.append(result)
    elif remnant == "pad":
        # Pad with zeros to match original length
        remaining = array.shape[-1] - ptr
        if remaining > 0:
            result = np.zeros((*array.shape[:-1], remaining), dtype="u1")
            components.append(result)

    result = np.concatenate(components, axis=-1)

    return result
