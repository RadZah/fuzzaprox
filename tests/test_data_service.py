"""
Unit tests for DataService
"""
import pytest

import numpy as np
from fuzzaprox.services import DataService


# Rounding to 2 decimals in normalise() limits how accurately denormalise()
# can restore the original values: up to 0.005 of the normalised [0,1] scale,
# which maps back to 0.5 % of the original data range.
ROUNDING_TOLERANCE = 0.005


def tolerance_for(orig_values):
    """Absolute tolerance in original units caused by the rounding in normalise()"""
    data_range = max(orig_values) - min(orig_values)
    return ROUNDING_TOLERANCE * data_range + 1e-9


class TestCalculateRange:
    """Test calculate_range"""

    def test_calculate_range(self):
        """Test min, max and range are computed correctly"""
        result = DataService.calculate_range([10.0, 12.5, 15.0, 20.0])

        assert result["min_from_range"] == 10.0
        assert result["max_from_range"] == 20.0
        assert result["range"] == 10.0

    def test_calculate_range_with_negative_values(self):
        """Test range spanning negative values"""
        result = DataService.calculate_range([-1.4, 0.0, 1.4])

        assert result["min_from_range"] == -1.4
        assert result["max_from_range"] == 1.4
        assert result["range"] == pytest.approx(2.8)


class TestNormalise:
    """Test normalise"""

    def test_normalise_returns_ndarray(self):
        """Test normalise returns a numpy array"""
        result = DataService.normalise([10.0, 12.5, 15.0, 20.0])
        assert isinstance(result, np.ndarray)

    def test_normalise_known_values(self):
        """Test normalise maps values onto [0,1] proportionally"""
        result = DataService.normalise([10.0, 12.5, 15.0, 20.0])
        assert result == pytest.approx([0.0, 0.25, 0.5, 1.0])

    def test_normalise_with_negative_values(self):
        """Test normalise handles data spanning negative values"""
        result = DataService.normalise([-1.4, 0.0, 1.4])
        assert result == pytest.approx([0.0, 0.5, 1.0])

    def test_normalise_constant_input(self):
        """Constant data has no range, so every value maps to 0.0 instead of raising"""
        result = DataService.normalise([5.0, 5.0, 5.0])
        assert result == pytest.approx([0.0, 0.0, 0.0])

    def test_normalise_bounds(self):
        """Test the normalised result always starts at 0 and ends at 1"""
        result = DataService.normalise([3.0, -7.0, 11.0, 0.5])

        assert np.min(result) == 0.0
        assert np.max(result) == 1.0
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


class TestDenormalise:
    """Test denormalise - the inverse of normalise"""

    def test_denormalise_returns_ndarray(self):
        """Test denormalise returns a numpy array"""
        orig = [10.0, 12.5, 15.0, 20.0]
        result = DataService.denormalise(DataService.normalise(orig), orig)
        assert isinstance(result, np.ndarray)

    def test_denormalise_known_values(self):
        """Test denormalise maps [0,1] back onto the original scale"""
        orig = [10.0, 12.5, 15.0, 20.0]
        result = DataService.denormalise([0.0, 0.25, 0.5, 1.0], orig)
        assert result == pytest.approx([10.0, 12.5, 15.0, 20.0])

    def test_denormalise_roundtrip(self):
        """Test normalise followed by denormalise restores the original values"""
        orig = [10.0, 12.5, 15.0, 20.0]
        result = DataService.denormalise(DataService.normalise(orig), orig)
        assert result == pytest.approx(orig, abs=tolerance_for(orig))

    def test_denormalise_roundtrip_with_negative_values(self):
        """Test the roundtrip across a range spanning negative values"""
        orig = [-1.4, 0.0, 1.4]
        result = DataService.denormalise(DataService.normalise(orig), orig)
        assert result == pytest.approx(orig, abs=tolerance_for(orig))

    def test_denormalise_roundtrip_with_large_range(self):
        """Test the roundtrip is proportional and does not break on large ranges"""
        orig = [0.0, 2500.0, 10000.0]
        result = DataService.denormalise(DataService.normalise(orig), orig)
        assert result == pytest.approx(orig, abs=tolerance_for(orig))

    def test_denormalise_roundtrip_within_rounding_tolerance(self):
        """Test values that cannot be represented exactly after rounding to 2 decimals"""
        orig = [0.0, 1.0, 3.0]
        # normalise gives [0.0, 0.33, 1.0], so 1.0 comes back as 0.99
        result = DataService.denormalise(DataService.normalise(orig), orig)
        assert result == pytest.approx(orig, abs=tolerance_for(orig))

    def test_denormalise_stays_within_original_bounds(self):
        """Regression test: normalised input must map back inside the original data range

        The original implementation subtracted the data minimum from the normalised
        value, which pushed results far outside the input range (e.g. [10..20] came
        back as [-90..-80]).
        """
        orig = [10.0, 12.5, 15.0, 20.0]
        result = DataService.denormalise([0.0, 0.25, 0.5, 1.0], orig)

        assert np.all(result >= min(orig))
        assert np.all(result <= max(orig))

    def test_denormalise_roundtrip_constant_input(self):
        """A constant series survives the roundtrip - 0.0 maps back onto the constant"""
        orig = [5.0, 5.0, 5.0]
        result = DataService.denormalise(DataService.normalise(orig), orig)
        assert result == pytest.approx(orig)

    def test_denormalise_endpoints(self):
        """Test 0 maps to the data minimum and 1 maps to the data maximum"""
        orig = [10.0, 12.5, 15.0, 20.0]
        result = DataService.denormalise([0.0, 1.0], orig)
        assert result == pytest.approx([10.0, 20.0])
