# Changelog

## [0.0.3] - 2026-08-31
### Fixed
- `DataService.denormalise()` used a wrong formula and returned values far outside
  the original data range
- `DataService.normalise()` raised `ZeroDivisionError` on constant input; constant
  data now maps to zeros
- `FuzzySetBasic.get_fuzzy_set_value()` raised `ZeroDivisionError` when the kernel
  started at the base start; such a set now reports full membership at once

### Added
- `Fuzzaprox.denormalise()` converts approximations back to the original data scale,
  accepting either an `ApproxResults` instance or a plain array
- Unit tests for `DataService` and `FuzzySetBasic`
- Optional `dev` dependency group for testing and building

### Changed
- Test suite is no longer shipped inside the installed package
- Package metadata: classifiers, keywords, project URLs and SPDX license expression

## [0.0.2] - 2026-01-18
### Added
- pytest-based tests covering main transform and inverse transform

