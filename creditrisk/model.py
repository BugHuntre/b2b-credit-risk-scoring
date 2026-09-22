"""Training: monthly snapshots -> label = severe delinquency in the next `horizon` days.

Label (per buyer, per snapshot date T): among invoices falling due in (T, T+horizon],
was any of them paid more than `severe_days` late, or still unpaid `severe_days` after due?
"""
import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from .config import SETTINGS, Settings
from .features import build_features, feature_columns

logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    model: XGBClassifier
    features: list[str]
    metrics: dict
    data_end: pd.Timestamp
    settings: Settings = SETTINGS


def data_end_date(invoices: pd.DataFrame) -> pd.Timestamp:
    return max(invoices["invoice_date"].max(), invoices["paid_date"].max())


def make_labels(invoices: pd.DataFrame, t: pd.Timestamp, settings: Settings = SETTINGS) -> pd.Series:
    horizon = pd.Timedelta(days=settings.label.horizon_days)
    win = invoices[(invoices["due_date"] > t) & (invoices["due_date"] <= t + horizon)]
    late = (win["paid_date"] - win["due_date"]).dt.days
    bad = win["paid_date"].isna() | (late > settings.label.severe_days)
    return bad.groupby(win["buyer_id"]).max().astype(int).rename("label")


def snapshot_dates(invoices: pd.DataFrame, data_end: pd.Timestamp,
                    settings: Settings = SETTINGS) -> pd.DatetimeIndex:
    start = invoices["invoice_date"].min() + pd.Timedelta(days=settings.feature_warmup_days)
    resolve_by = settings.label.horizon_days + settings.label.severe_days
    end = data_end - pd.Timedelta(days=resolve_by)   # labels must be fully resolved by data_end
    return pd.date_range(start, end, freq=settings.snapshot_step)


def build_dataset(buyers: pd.DataFrame, invoices: pd.DataFrame, snapshots: pd.DatetimeIndex,
                   settings: Settings = SETTINGS) -> pd.DataFrame:
    parts = []
    for t in snapshots:
        f = build_features(buyers, invoices, t)
        f = f.merge(make_labels(invoices, t, settings), left_on="buyer_id", right_index=True, how="inner")
        f["snapshot"] = t
        parts.append(f)
    return pd.concat(parts, ignore_index=True)


def new_model(settings: Settings = SETTINGS) -> XGBClassifier:
    mc = settings.model
    return XGBClassifier(n_estimators=mc.n_estimators, max_depth=mc.max_depth,
                         learning_rate=mc.learning_rate, subsample=mc.subsample,
                         colsample_bytree=mc.colsample_bytree, min_child_weight=mc.min_child_weight,
                         eval_metric="aucpr", random_state=mc.random_state)


def _recall_at_top(y: pd.Series, p, share: float) -> float:
    k = max(1, int(len(y) * share))
    order = p.argsort()[::-1][:k]
    return float(y.iloc[order].sum() / max(y.sum(), 1))


def holdout_eval(ds: pd.DataFrame, cols: list[str], snaps: pd.DatetimeIndex,
                  settings: Settings = SETTINGS) -> dict:
    """Single time-based hold-out: train on everything before the last few snapshots, test on those.

    A gap of `horizon_days` is left between train and test so no training label's outcome window
    overlaps the test snapshots (avoids leakage across the split).
    """
    test_start = snaps[-settings.n_wf_folds]
    horizon = pd.Timedelta(days=settings.label.horizon_days)
    train = ds[ds["snapshot"] <= test_start - horizon]
    test = ds[ds["snapshot"] >= test_start]

    m = new_model(settings).fit(train[cols], train["label"])
    p = m.predict_proba(test[cols])[:, 1]
    y = test["label"].reset_index(drop=True)
    baseline = test["avg_days_late_90d"].fillna(0)
    return {
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "base_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "baseline_roc_auc_avg_days_late": float(roc_auc_score(y, baseline)),
        "baseline_pr_auc_avg_days_late": float(average_precision_score(y, baseline)),
        "recall_in_top_20pct": _recall_at_top(y, p, 0.20),
    }


def fit(buyers: pd.DataFrame, invoices: pd.DataFrame, settings: Settings = SETTINGS) -> ModelBundle:
    data_end = data_end_date(invoices)
    snaps = snapshot_dates(invoices, data_end, settings)
    if len(snaps) < settings.min_snapshots:
        raise ValueError(
            f"Not enough history: {len(snaps)} usable snapshots, need at least "
            f"{settings.min_snapshots} (~{settings.min_snapshots} months of clean invoice history)."
        )
    logger.info("Building training set from %d snapshots (%s to %s)", len(snaps), snaps[0].date(), snaps[-1].date())
    ds = build_dataset(buyers, invoices, snaps, settings)
    cols = feature_columns(ds.drop(columns=["label", "snapshot"]))

    metrics = holdout_eval(ds, cols, snaps, settings)
    logger.info("Hold-out ROC-AUC=%.3f PR-AUC=%.3f (baseline %.3f/%.3f)", metrics["roc_auc"],
               metrics["pr_auc"], metrics["baseline_roc_auc_avg_days_late"], metrics["baseline_pr_auc_avg_days_late"])

    final = new_model(settings).fit(ds[cols], ds["label"])   # refit on everything for live scoring
    return ModelBundle(final, cols, metrics, data_end, settings)
