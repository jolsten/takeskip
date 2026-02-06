# TakeSkip

A Python library for declarative bit manipulation using an intuitive command syntax.

## Overview

TakeSkip provides a domain-specific language for selecting, rearranging, and manipulating bits in binary arrays. Instead of writing complex indexing logic, you can express operations using simple commands like `t8s4r8` (take 8 bits, skip 4, reverse 8).

## Installation

```bash
pip install numpy lark
```

Place the takeskip module in your Python path.

## Basic Usage

```python
import numpy as np
from takeskip import takeskip

# Create a binary array (8 bits)
bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)

# Skip first 2 bits, take next 4
result = takeskip("s2t4", bits)
print(result)  # [1, 1, 0, 0]

# Take 4 bits, reverse next 4
result = takeskip("t4r4", bits)
print(result)  # [1, 0, 1, 1, 0, 1, 0, 0]
```

## Command Reference

### Basic Operations

| Command | Description | Example |
|---------|-------------|---------|
| `t<n>` | Take n bits | `t8` - take 8 bits |
| `s<n>` | Skip n bits | `s4` - skip 4 bits |
| `r<n>` | Reverse n bits | `r8` - reverse 8 bits |
| `i<n>` | Invert n bits (0↔1) | `i4` - invert 4 bits |
| `ri<n>` | Reverse and invert n bits | `ri8` - reverse then invert 8 bits |
| `b<n>` | Backup pointer n positions | `b4` - move back 4 positions |

### Padding Operations

| Command | Description | Example |
|---------|-------------|---------|
| `z<n>` | Insert n zeros | `z4` - add 4 zeros |
| `n<n>` | Insert n ones | `n4` - add 4 ones |
| `d<binary>` | Insert literal binary data | `d101` - add bits 1,0,1 |

### Permutation

| Command | Description | Example |
|---------|-------------|---------|
| `p<indices>` | Permute using 1-based indices | `p1,3,5` - select bits 1, 3, 5 |
| `p<range>` | Permute using ranges | `p1-4` - select bits 1 through 4 |
| `p<mixed>` | Mix indices and ranges | `p1-4,8,6-5` - complex permutation |

**Note:** Permutation indices are 1-based for user convenience. Ranges are inclusive.

### Grouping and Repetition

```python
# Repeat a sequence
bits = np.array([1, 0] * 8, dtype=np.uint8)
result = takeskip("(t4s4)3", bits)  # Repeat "take 4, skip 4" three times

# Group complex operations
result = takeskip("(t2r2s1)4", bits)  # Repeat grouped operations
```

## Advanced Examples

### Extract Every Other Bit
```python
bits = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
result = takeskip("(t1s1)4", bits)  # [1, 1, 1, 1]
```

### Nibble Swap (swap 4-bit chunks)
```python
bits = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
result = takeskip("s4t4b8t4", bits)  # [0, 0, 0, 0, 1, 1, 1, 1]
```

### Complex Permutation
```python
bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
# Take bits at positions 1,2,3,4 then 8,7,6,5 (1-based)
result = takeskip("p1-4,8-5", bits)  # [1, 0, 1, 1, 0, 1, 0, 0]
```

### Interleave Pattern with Padding
```python
bits = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
result = takeskip("(t1z1)4", bits)
# [1, 0, 1, 0, 1, 0, 1, 0] - interleaved with zeros
```

### Reverse Complement (DNA-like operation)
```python
bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
result = takeskip("ri8", bits)  # Reverse and invert all
# [1, 0, 1, 1, 0, 0, 1, 0] reversed becomes [0, 1, 0, 0, 1, 1, 0, 1]
# then inverted becomes [1, 0, 1, 1, 0, 0, 1, 0]
```

## Remnant Handling

Control what happens to remaining bits after commands execute:

```python
bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)

# Default: discard remaining bits
result = takeskip("t4", bits, remnant="remove")  # [1, 0, 1, 1]

# Keep remaining bits
result = takeskip("t4", bits, remnant="keep")  # [1, 0, 1, 1, 0, 0, 1, 0]

# Pad with zeros to original length
result = takeskip("t4", bits, remnant="pad")  # [1, 0, 1, 1, 0, 0, 0, 0]
```

## Multi-dimensional Arrays

TakeSkip operates on the last axis of multi-dimensional arrays:

```python
# 2D array: 3 rows of 8 bits each
bits = np.array([
    [1, 0, 1, 1, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 0, 1],
    [1, 1, 0, 0, 1, 0, 1, 1],
], dtype=np.uint8)

result = takeskip("s2t4", bits)
# [[1, 1, 0, 0],
#  [0, 1, 1, 1],
#  [0, 0, 1, 0]]
```

## Use Cases

- **Data packing/unpacking**: Extract fields from packed binary formats
- **Protocol parsing**: Parse binary protocols with field alignment
- **Bit manipulation**: Rearrange bits in encryption/encoding operations
- **Data compression**: Pattern-based bit selection
- **Binary format conversion**: Transform between different bit layouts
- **Bioinformatics**: DNA/RNA sequence manipulation with reverse complement

## Command Chaining

Commands are executed left-to-right with a maintained pointer:

```python
bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)

# Pointer starts at 0
# t2: take bits 0-1 -> [1,0], pointer now at 2
# s2: skip bits 2-3, pointer now at 4  
# t4: take bits 4-7 -> [0,0,1,0], pointer now at 8
result = takeskip("t2s2t4", bits)  # [1, 0, 0, 0, 1, 0]
```

## Syntax Notes

- Commands are **case-insensitive**: `T4`, `t4`, and `T4` are equivalent
- **Whitespace is ignored**: `t4 s2 r8` = `t4s2r8`
- **Parentheses** create groups: `(t8s8)4` repeats the sequence 4 times
- **Permutation** uses 1-based indexing (bit 1 is the first bit)
- **Ranges** in permutation are inclusive: `1-4` includes bits 1, 2, 3, and 4

## Error Handling

```python
# Invalid remnant value
try:
    takeskip("t4", bits, remnant="invalid")
except ValueError as e:
    print(e)  # "invalid remnant argument; must be 'remove', 'keep', or 'pad'"

# Invalid command syntax
try:
    takeskip("x4", bits)  # 'x' is not a valid command
except Exception as e:
    print(e)  # Lark parse error
```

## API Documentation

### Main Function

```python
def takeskip(
    command: str,
    array: npt.NDArray[np.uint8],
    *,
    remnant: Literal["remove", "keep", "pad"] = "remove",
) -> np.ndarray
```

**Parameters:**
- `command`: Command string expressing the operation
- `array`: Target numpy array (dtype=uint8, values 0 or 1)
- `remnant`: How to handle remaining bits ("remove", "keep", or "pad")

**Returns:**
- Numpy array with same dtype, modified according to commands

**Raises:**
- `ValueError`: Invalid remnant argument or command syntax error

## Contributing

When adding new commands:
1. Define command class in `commands.py` inheriting from `Command`
2. Add grammar rule in `takeskip.lark`
3. Add transformer method in `parser.py`
4. Update documentation

## License

[Your license here]
