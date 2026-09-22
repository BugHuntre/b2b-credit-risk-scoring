import numpy as np
import pandas as pd

from creditrisk.evaluate import band_validation, bootstrap_auc_ci, cost_curve, walk_forward_eval


def test_cost_curve_extreme_thresholds():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    exposure = np.array([100, 100, 100, 100, 100])
    df = cost_curve(y, p, exposure, n_thresholds=5)

    flag_none = df.iloc[0]     # threshold 0 -> everyone flagged -> no missed bad buyers
    assert flag_none["fn_cost"] == 0
    flag_all_out = df.iloc[-1]  # threshold 1 -> no one flagged -> no wrongly-restricted good buyers
    assert flag_all_out["fp_cost"] == 0


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = np.clip(y * 0.6 + rng.normal(0, 0.3, 200), 0, 1)
    result = bootstrap_auc_ci(y, p, n_boot=200)
    assert result["ci_low"] <= result["point"] <= result["ci_high"]


def test_band_validation_handles_non_default_index():
    """y often comes in as a pandas Series with a non-contiguous index (e.g. a filtered slice);
    band_validation must align it with p by position, not by index label."""
    y = pd.Series([0, 1, 0, 1], index=[17, 42, 100, 101])
    p = [0.01, 0.99, 0.02, 0.25]   # -> Low, High, Low, Medium (cuts: medium=0.15, high=0.40)
    out = band_validation(y, p)
    assert out.loc["Low", "n"] == 2
    assert out.loc["Medium", "n"] == 1
    assert out.loc["High", "n"] == 1
    assert out.loc["High", "bad_rate"] == 1.0


def test_walk_forward_eval_runs(sample_data):
    buyers, invoices = sample_data
    df = walk_forward_eval(buyers, invoices)
    assert len(df) > 0
    assert df["roc_auc"].between(0, 1).all()
