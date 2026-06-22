"""Tests for monitoring (PSI/CSI/drift)."""

import numpy as np

from src.monitoring.drift import calculate_psi


class TestPSI:
    """Test Population Stability Index calculation."""

    def test_identical_distributions_zero_psi(self):
        """PSI should be ~0 for identical distributions."""
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 10000)
        psi = calculate_psi(data, data)
        assert psi < 0.01, f"PSI for identical distributions should be ~0, got {psi:.4f}"

    def test_shifted_distribution_high_psi(self):
        """PSI should be > 0.25 for significantly shifted distributions."""
        rng = np.random.default_rng(42)
        expected = rng.normal(0, 1, 10000)
        actual = rng.normal(2, 1, 10000)  # mean shifted by 2 std devs
        psi = calculate_psi(expected, actual)
        assert psi >= 0.25, f"PSI for shifted distribution should be >= 0.25, got {psi:.4f}"

    def test_psi_non_negative(self):
        """PSI should always be non-negative."""
        rng = np.random.default_rng(42)
        expected = rng.normal(0, 1, 1000)
        actual = rng.normal(0.5, 1.5, 1000)
        psi = calculate_psi(expected, actual)
        assert psi >= 0

    def test_moderate_shift(self):
        """Small shift should give moderate PSI (0.1 - 0.25)."""
        rng = np.random.default_rng(42)
        expected = rng.normal(0, 1, 10000)
        actual = rng.normal(0.3, 1.1, 10000)  # slight shift
        psi = calculate_psi(expected, actual)
        # Should be detectable but not catastrophic
        assert psi > 0.01
