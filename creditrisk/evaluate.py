"""Rigorous evaluation beyond a single train/test split: walk-forward CV, bootstrap confidence
intervals, calibration, and cost-based threshold selection.

`model.fit` already does one time-based hold-out for speed (used for the live model's headline
metrics). Everything here is heavier and meant for `train.py --report` / the notebook / CI, where
we want more confidence that the hold-out result wasn't a fluke of one particular split.
"""
import logging

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import SETTINGS, Settings
from .features import feature_columns
from .model import build_dataset, data_end_date, new_model, snapshot_dates

logger = logging.getLogger(__name__)


def walk_forward_eval(buyers: pd.DataFrame, invoices: pd.DataFrame,
                       settings: Settings = SETTINGS) -> pd.DataFrame:
    """Expanding-window walk-forward validation, one fold per of the last `n_wf_folds` snapshots.

    Fold i trains on every snapshot at least `horizon_days` before the test snapshot (so no
    training label's outcome window overlaps the test point) and tests on that snapshot alone.
    This mimics periodic retraining in production far better than one fixed split, and its
    fold-to-fold spread shows how much the hold-out AUC in `model.fit` could have varied by luck.
    """
    data_end = data_end_date(invoices)
    snaps = snapshot_dates(invoices, data_end, settings)
    ds = build_dataset(buyers, invoices, snaps, settings)
    cols = feature_columns(ds.drop(columns=["label", "snapshot"]))
    horizon = pd.Timedelta(days=settings.label.horizon_days)

    rows = []
    for test_snap in snaps[-settings.n_wf_folds:]:
        train = ds[ds["snapshot"] <= test_snap - horizon]
        test = ds[ds["snapshot"] == test_snap]
        if train["label"].nunique() < 2 or test["label"].nunique() < 2 or len(test) < 20:
            logger.warning("Skipping fold %s: too few rows or a single-class sample", test_snap.date())
            continue
        m = new_model(settings).fit(train[cols], train["label"])
        p = m.predict_proba(test[cols])[:, 1]
        y = test["label"].to_numpy()
        rows.append(dict(snapshot=str(test_snap.date()), n_train=len(train), n_test=len(test),
                         base_rate=float(y.mean()), roc_auc=float(roc_auc_score(y, p)),
                         pr_auc=float(average_precision_score(y, p))))
    return pd.DataFrame(rows)


def bootstrap_auc_ci(y, p, n_boot: int = 1000, seed: int = 0) -> dict:
    """Percentile bootstrap 95% CI for ROC-AUC, resampling buyers with replacement.

    Quantifies how much the reported AUC could move on a differently-sampled test set of the
    same size -- a single point estimate on a few hundred buyers is not very precise.
    """
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"point": float(roc_auc_score(y, p)), "ci_low": float(lo), "ci_high": float(hi), "n_boot_used": len(boots)}


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs. observed bad rate by probability decile. A well-calibrated model tracks the
    diagonal; if it doesn't, the raw score shouldn't be read as a literal probability even where
    it still ranks buyers correctly."""
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"predicted": mean_pred, "observed": frac_pos})


def band_validation(y, p, settings: Settings = SETTINGS) -> pd.DataFrame:
    """Out-of-sample bad rate actually observed within each Low/Medium/High cut, as a sanity
    check that the bands are monotonic and roughly match the policy's intent."""
    from .scoring import band
    y, p = np.asarray(y), np.asarray(p)   # drop any pandas index so the two columns align by position
    bands = [band(pi, settings) for pi in p]
    df = pd.DataFrame({"band": bands, "bad": y})
    return (df.groupby("band")["bad"].agg(n="count", bad_rate="mean")
            .reindex(["Low", "Medium", "High"]).fillna(0))


def cost_curve(y, p, exposure, settings: Settings = SETTINGS, n_thresholds: int = 41) -> pd.DataFrame:
    """Expected cost swept over score thresholds, using the illustrative unit costs in
    `config.CostConfig`:
      - miss a bad buyer (score below threshold, buyer actually defaults): lose `loss_given_default`
        of their exposure
      - wrongly flag a good buyer (score at/above threshold, buyer actually pays): lose
        `margin_on_revenue` of their exposure by restricting credit they didn't need restricted

    These unit costs are placeholders -- see config.py -- so treat the resulting "optimal"
    threshold as illustrative of the method, not a number to deploy as-is.
    """
    y, p, exposure = np.asarray(y, dtype=float), np.asarray(p), np.asarray(exposure, dtype=float)
    rows = []
    for t in np.linspace(0, 1, n_thresholds):
        flag = p >= t
        fn_cost = (exposure[~flag & (y == 1)] * settings.cost.loss_given_default).sum()
        fp_cost = (exposure[flag & (y == 0)] * settings.cost.margin_on_revenue).sum()
        rows.append(dict(threshold=t, fn_cost=fn_cost, fp_cost=fp_cost, total_cost=fn_cost + fp_cost))
    return pd.DataFrame(rows)


def optimal_threshold(y, p, exposure, settings: Settings = SETTINGS) -> dict:
    df = cost_curve(y, p, exposure, settings)
    best = df.loc[df["total_cost"].idxmin()]
    return {"threshold": float(best["threshold"]), "expected_cost": float(best["total_cost"]),
           "current_high_cut": settings.band.high_cut, "curve": df}
