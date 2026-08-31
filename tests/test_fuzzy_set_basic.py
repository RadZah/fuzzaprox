"""
Unit tests for FuzzySetBasic
"""
import pytest

from fuzzaprox.transformations import FuzzySetBasic


def make_set(kernel_start, kernel_end, fuzzy_set_width):
    return FuzzySetBasic({"kernel_start": kernel_start,
                          "kernel_end": kernel_end,
                          "fuzzy_set_width": fuzzy_set_width})


class TestFuzzySetShape:
    """Test the membership function of a regular trapezoidal set"""

    def setup_method(self):
        # base 0..10, kernel 4..6
        self.fs = make_set(4, 6, 10)

    def test_outside_base_is_zero(self):
        """Points outside the base have no membership"""
        assert self.fs.get_fuzzy_set_value(-1) == 0
        assert self.fs.get_fuzzy_set_value(11) == 0

    def test_rising_edge(self):
        """Membership grows linearly from the base start up to the kernel"""
        assert self.fs.get_fuzzy_set_value(0) == pytest.approx(0.0)
        assert self.fs.get_fuzzy_set_value(2) == pytest.approx(0.5)
        assert self.fs.get_fuzzy_set_value(4) == pytest.approx(1.0)

    def test_kernel_is_one(self):
        """Membership is full across the whole kernel"""
        assert self.fs.get_fuzzy_set_value(5) == pytest.approx(1.0)
        assert self.fs.get_fuzzy_set_value(6) == pytest.approx(1.0)

    def test_falling_edge(self):
        """Membership falls linearly from the kernel down to the base end"""
        assert self.fs.get_fuzzy_set_value(8) == pytest.approx(0.5)
        assert self.fs.get_fuzzy_set_value(10) == pytest.approx(0.0)

    def test_values_stay_within_zero_and_one(self):
        """Membership never leaves [0,1] anywhere on the base"""
        for x in range(-2, 13):
            value = self.fs.get_fuzzy_set_value(x)
            assert 0 <= value <= 1, f"x={x} gave {value}"


class TestFuzzySetDegenerateShapes:
    """Test sets whose kernel touches the edge of the base"""

    def test_kernel_starts_at_base_start(self):
        """A set with no rising edge starts at full membership

        base 0..10, kernel 0..5 - there is no room for a rising edge,
        so the very first point of the base already belongs fully.
        """
        fs = make_set(0, 5, 10)
        assert fs.get_fuzzy_set_value(0) == pytest.approx(1.0)
        assert fs.get_fuzzy_set_value(5) == pytest.approx(1.0)
        assert fs.get_fuzzy_set_value(10) == pytest.approx(0.0)

    def test_kernel_ends_at_base_end(self):
        """A set with no falling edge keeps full membership up to the base end"""
        fs = make_set(3, 10, 10)
        assert fs.get_fuzzy_set_value(0) == pytest.approx(0.0)
        assert fs.get_fuzzy_set_value(3) == pytest.approx(1.0)
        assert fs.get_fuzzy_set_value(10) == pytest.approx(1.0)

    def test_kernel_covers_whole_base(self):
        """A set with no edges at all is fully flat"""
        fs = make_set(0, 10, 10)
        assert fs.get_fuzzy_set_value(0) == pytest.approx(1.0)
        assert fs.get_fuzzy_set_value(5) == pytest.approx(1.0)
        assert fs.get_fuzzy_set_value(10) == pytest.approx(1.0)
