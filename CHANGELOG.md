# Changelog

## [0.1.0b4]

### Fixed

- `Data` command now validates input contains only `0` and `1` characters, raising `ValueError` for invalid input

### Changed

- `parse_command` results are now cached with `lru_cache` for improved performance
- `parse_command` returns `tuple[Command, ...]` instead of `list` for immutability of cached results

### Added

- Project URLs (Homepage, Repository, Issues, Changelog) in package metadata

## [0.1.0b3]

### Changed

- Pointer bounds are now validated after each command; a `ValueError` is raised if `Backup` moves before the start or `Skip`/`Take` moves past the end

## [0.1.0b2]

### Fixed

- `Command.__eq__` now returns `NotImplemented` for non-Command types instead of raising `TypeError`
- Invalid `remnant` argument now raises `ValueError` (was `TypeError`)

### Changed

- `Command` base class now inherits from `ABC`, enforcing the abstract `__call__` contract
- Switched parser from Earley to LALR for faster parsing
- `Zeros` and `Ones` now coerce their value to `int` in `__init__`, consistent with other commands

### Added

- `py.typed` marker for PEP 561 typing support
- `__all__` export list in `__init__.py`
- PyPI classifiers for Python versions, license, and typing
- Property-based tests using Hypothesis
- Edge case tests for pointer bounds clamping and `__eq__` behavior

### Removed

- Unused `pos_int` / `POS_INT` grammar rules
