"""
Submarine simulation replicating and extending the lost-submarine
thought experiment from Morey et al. (2016).

The setup: a rescue ship observes pairs of bubbles rising uniformly
at random from a submarine's hull to estimate the location of its
hatch. We construct confidence intervals using three procedures
(nonparametric, UMP, and sampling-distribution), then evaluate
coverage forecasting strategies using Brier scores.

See Section 4 of "Confidence as Forecast" (Lee, 2026) for details.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


# ---------------------------------------------------------------------------
# 1. Helper functions
# ---------------------------------------------------------------------------

def covers(ci, theta):
    """Check whether a confidence interval contains the true parameter."""
    return int(ci[0] <= theta <= ci[1])


def is_nested(inner, outer):
    """Check whether `inner` is strictly nested inside `outer`."""
    return outer[0] < inner[0] and inner[1] < outer[1]


# ---------------------------------------------------------------------------
# 2. Confidence interval constructors
#
# Each function takes a pair of bubble locations (x1, x2) and the
# half-hull-width h (= scale / 2), and returns an interval
# [lower, upper]. All three procedures have 50% nominal coverage
# under the Uniform[theta - h, theta + h] model.
# ---------------------------------------------------------------------------

def ci_nonparametric(x1, x2, h):
    """Nonparametric (NP) interval: the range spanned by the two bubbles."""
    return [min(x1, x2), max(x1, x2)]


def ci_ump(x1, x2, h):
    """
    Universally most powerful (UMP) interval: the shorter of the two
    intervals that achieve 50% coverage under the uniform model.

    When the bubbles are close together (d < h), this equals the NP
    interval. When they are far apart (d >= h), the interval flips
    to cover the region between the outer edges of the hull implied
    by each bubble, which is always shorter.
    """
    lo, hi = min(x1, x2), max(x1, x2)
    d = hi - lo
    if d < h:
        return [lo, hi]
    else:
        return [hi - h, lo + h]


def ci_sampling_distribution(x1, x2, h):
    """
    Sampling-distribution (SD) interval: x_bar +/- (h - h/sqrt(2)).

    This interval is centered on the sample mean and has constant width
    determined by the support of the uniform distribution.
    """
    x_bar = (x1 + x2) / 2
    half_width = h - h / np.sqrt(2)
    return [x_bar - half_width, x_bar + half_width]


# ---------------------------------------------------------------------------
# 3. Width-conditional coverage forecasts
#
# For each interval, we bin by its width (rounded to 1 decimal place),
# compute the empirical coverage rate within each bin, and assign that
# rate back to every interval in the bin. This is the theta-free
# conditional forecast described in Section 3 of the paper.
# ---------------------------------------------------------------------------

def conditional_coverage_forecasts(widths, covered):
    """
    Given an array of interval widths and a binary array of coverage
    outcomes, return an array of conditional coverage forecasts, one
    per interval, based on binned width.
    """
    rounded = np.round(widths, 1)
    unique_widths = np.unique(rounded)
    coverage_by_width = {
        w: covered[rounded == w].mean() for w in unique_widths
    }
    return np.array([coverage_by_width[w] for w in rounded])


# ---------------------------------------------------------------------------
# 4. Run a single simulation configuration
#
# For a given hatch location (theta) and hull width (scale), we draw
# N pairs of bubbles, construct intervals under each procedure, and
# compute coverage and Brier scores for several forecasting strategies.
# ---------------------------------------------------------------------------

def run_simulation(theta, scale, N=100_000, seed=None):
    """
    Simulate N bubble pairs and evaluate coverage forecasts.

    Parameters
    ----------
    theta : float
        True hatch location.
    scale : float
        Hull width (bubbles are Uniform[theta - scale/2, theta + scale/2]).
    N : int
        Number of bubble pairs to draw.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict with Brier scores for each forecasting strategy, plus marginal
    and joint coverage rates.
    """
    rng = np.random.default_rng(seed)
    h = scale / 2  # half-hull-width

    # Draw bubble pairs
    bubbles = rng.uniform(
        low=theta - h,
        high=theta + h,
        size=(N, 2)
    )

    # Construct intervals under each procedure
    np_intervals = [ci_nonparametric(x1, x2, h) for x1, x2 in bubbles]
    ump_intervals = [ci_ump(x1, x2, h) for x1, x2 in bubbles]
    sd_intervals = [ci_sampling_distribution(x1, x2, h) for x1, x2 in bubbles]

    # Evaluate coverage
    np_covered = np.array([covers(ci, theta) for ci in np_intervals])
    ump_covered = np.array([covers(ci, theta) for ci in ump_intervals])
    sd_covered = np.array([covers(ci, theta) for ci in sd_intervals])

    # Joint coverage: does at least one of the SD and UMP intervals cover?
    either_covered = np.maximum(sd_covered, ump_covered)

    # -----------------------------------------------------------------------
    # 4a. Marginal Brier scores for NP and UMP intervals
    # -----------------------------------------------------------------------

    # Interval widths relative to hull width (the theta-free statistic)
    np_widths = np.array([ci[1] - ci[0] for ci in np_intervals])
    ump_widths = np.array([ci[1] - ci[0] for ci in ump_intervals])

    np_rel_widths = np_widths / scale
    ump_rel_widths = ump_widths / scale

    # Conditional coverage forecasts based on relative width
    np_cond_forecasts = conditional_coverage_forecasts(np_rel_widths, np_covered)
    ump_cond_forecasts = conditional_coverage_forecasts(ump_rel_widths, ump_covered)

    # Constant forecasts
    ones = np.ones(N)
    halves = np.ones(N) * 0.5

    marginal_brier = {
        "constant_1": brier_score_loss(np_covered, ones),
        "constant_alpha": brier_score_loss(np_covered, halves),
        "np_width": brier_score_loss(np_covered, np_cond_forecasts),
        "ump_width": brier_score_loss(ump_covered, ump_cond_forecasts),
    }

    # -----------------------------------------------------------------------
    # 4b. Joint Brier scores for SD + UMP nesting analysis
    # -----------------------------------------------------------------------

    # Check nesting between SD and UMP intervals
    sd_in_ump = np.array([
        is_nested(sd_intervals[i], ump_intervals[i]) for i in range(N)
    ])
    ump_in_sd = np.array([
        is_nested(ump_intervals[i], sd_intervals[i]) for i in range(N)
    ])

    # Joint coverage rate (for the constant design-level forecast)
    p_joint = either_covered.mean()

    # Conditional coverage given nesting direction
    nesting_forecasts = np.where(
        sd_in_ump,
        either_covered[sd_in_ump].mean() if sd_in_ump.any() else p_joint,
        either_covered[ump_in_sd].mean() if ump_in_sd.any() else p_joint,
    )

    # Conditional coverage given nesting direction AND outer interval width
    outer_widths = np.where(
        sd_in_ump,
        np.array([ci[1] - ci[0] for ci in ump_intervals]),
        np.array([ci[1] - ci[0] for ci in sd_intervals]),
    )
    outer_rel_widths = outer_widths / scale
    max_width_forecasts = conditional_coverage_forecasts(
        outer_rel_widths, either_covered
    )

    joint_brier = {
        "constant_1": brier_score_loss(either_covered, ones),
        "constant_joint": brier_score_loss(either_covered, np.full(N, p_joint)),
        "nesting_cond": brier_score_loss(either_covered, nesting_forecasts),
        "max_width": brier_score_loss(either_covered, max_width_forecasts),
    }

    # -----------------------------------------------------------------------
    # 4c. Coverage rates for reporting
    # -----------------------------------------------------------------------

    coverage_rates = {
        "np": np_covered.mean(),
        "ump": ump_covered.mean(),
        "sd": sd_covered.mean(),
        "joint_sd_ump": p_joint,
        "either_covers_given_sd_in_ump": (
            either_covered[sd_in_ump].mean() if sd_in_ump.any() else np.nan
        ),
        "either_covers_given_ump_in_sd": (
            either_covered[ump_in_sd].mean() if ump_in_sd.any() else np.nan
        ),
        "sd_misses_ump_covers": (
            ((sd_covered == 0) & (ump_covered == 1)).mean()
        ),
        "ump_misses_sd_covers": (
            ((ump_covered == 0) & (sd_covered == 1)).mean()
        ),
    }

    return {
        "theta": theta,
        "scale": scale,
        "marginal_brier": marginal_brier,
        "joint_brier": joint_brier,
        "coverage_rates": coverage_rates,
    }


# ---------------------------------------------------------------------------
# 5. Run the full simulation grid
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    theta_values = np.arange(0, 11, 1)
    scale_values = np.arange(10, 111, 10)

    results = []
    for theta in theta_values:
        for scale in scale_values:
            result = run_simulation(theta, scale, N=100_000, seed=42)
            results.append(result)

    # -----------------------------------------------------------------------
    # Report: Marginal Brier scores (Table 1 in the paper)
    # -----------------------------------------------------------------------

    marginal_keys = ["constant_1", "constant_alpha", "np_width", "ump_width"]
    labels = {
        "constant_1": "Constant 1",
        "constant_alpha": "Constant 1 - alpha",
        "np_width": "NP width",
        "ump_width": "UMP width",
    }

    print("=" * 60)
    print("Table 1: Marginal Brier Scores (mean and variance)")
    print("=" * 60)
    for key in marginal_keys:
        scores = [r["marginal_brier"][key] for r in results]
        print(
            f"  {labels[key]:<20s}"
            f"  mean = {np.mean(scores):.3f}"
            f"  var = {np.var(scores):.6f}"
        )

    # -----------------------------------------------------------------------
    # Report: Joint Brier scores (Table 2 in the paper)
    # -----------------------------------------------------------------------

    joint_keys = ["constant_1", "constant_joint", "nesting_cond", "max_width"]
    joint_labels = {
        "constant_1": "Constant 1",
        "constant_joint": "Constant p_joint",
        "nesting_cond": "Nest. Cond.",
        "max_width": "Max Width",
    }

    print()
    print("=" * 60)
    print("Table 2: Joint Brier Scores (mean and variance)")
    print("=" * 60)
    for key in joint_keys:
        scores = [r["joint_brier"][key] for r in results]
        print(
            f"  {joint_labels[key]:<20s}"
            f"  mean = {np.mean(scores):.3f}"
            f"  var = {np.var(scores):.6f}"
        )

    # -----------------------------------------------------------------------
    # Report: Coverage rates
    # -----------------------------------------------------------------------

    print()
    print("=" * 60)
    print("Coverage Rates (averaged across configurations)")
    print("=" * 60)
    for key in results[0]["coverage_rates"]:
        rates = [r["coverage_rates"][key] for r in results]
        print(f"  {key:<40s}  mean = {np.nanmean(rates):.3f}")
