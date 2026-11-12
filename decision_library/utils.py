"""
utils.py - Small helper functions shared by all modules.

Main job: make sure probability distributions are VALID
(no negative numbers, and they add up to 1).
"""
import numpy as np

TOLERANCE = 1e-8  # how close to 1.0 the sum must be


def check_distribution(p, name="distribution"):
    """Check that p is a valid 1-D probability distribution.

    Returns p as a float NumPy array. Raises ValueError if invalid.
    """
    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {p.shape}")
    if p.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.any(np.isnan(p)):
        raise ValueError(f"{name} contains NaN")
    if np.any(p < 0):
        raise ValueError(f"{name} has negative probabilities: {p}")
    if abs(p.sum() - 1.0) > TOLERANCE:
        raise ValueError(f"{name} must sum to 1, but sums to {p.sum()}")
    return p


def check_rows_sum_to_one(matrix, name="matrix"):
    """Check that every last-axis row of `matrix` is a valid distribution."""
    m = np.asarray(matrix, dtype=float)
    if np.any(np.isnan(m)):
        raise ValueError(f"{name} contains NaN")
    if np.any(m < 0):
        raise ValueError(f"{name} has negative probabilities")
    sums = m.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=TOLERANCE):
        raise ValueError(f"every row of {name} must sum to 1, got sums {sums}")
    return m


def normalize(p):
    """Scale a non-negative vector so it sums to 1.

    Raises ValueError if everything is zero (nothing to normalize).
    """
    p = np.asarray(p, dtype=float)
    total = p.sum()
    if total <= 0:
        raise ValueError("cannot normalize: total probability is 0")
    return p / total
