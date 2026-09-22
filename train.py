"""Generate sample data (if missing), train, and optionally produce a full evaluation report.

    python train.py             # fit the model, save it, print hold-out metrics
    python train.py --report    # also run walk-forward CV, bootstrap CI, calibration and a
                                 # cost-based threshold sweep, saved under reports/
"""
import argparse
import json
import logging
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from creditrisk import evaluate
from creditrisk.config import SETTINGS
from creditrisk.features import feature_columns
from creditrisk.io import ensure_sample_data, load_data
from creditrisk.model import ModelBundle, build_dataset, data_end_date, fit, new_model, snapshot_dates

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def full_report(buyers: pd.DataFrame, invoices: pd.DataFrame, bundle: ModelBundle, out_dir: str = "reports") -> dict:
    os.makedirs(out_dir, exist_ok=True)

    wf = evaluate.walk_forward_eval(buyers, invoices)
    wf.to_csv(f"{out_dir}/walk_forward_folds.csv", index=False)
    logger.info("Walk-forward ROC-AUC: mean=%.3f std=%.3f across %d folds",
               wf["roc_auc"].mean(), wf["roc_auc"].std(), len(wf))

    # Rebuild the same held-out split used by model.fit to get raw predictions for
    # calibration / cost analysis (model.fit only keeps the summary metrics, not the predictions).
    data_end = data_end_date(invoices)
    snaps = snapshot_dates(invoices, data_end)
    ds = build_dataset(buyers, invoices, snaps)
    cols = feature_columns(ds.drop(columns=["label", "snapshot"]))
    test_start = snaps[-SETTINGS.n_wf_folds]
    horizon = pd.Timedelta(days=SETTINGS.label.horizon_days)
    train_ds = ds[ds["snapshot"] <= test_start - horizon]
    test_ds = ds[ds["snapshot"] >= test_start]
    m = new_model().fit(train_ds[cols], train_ds["label"])
    p = m.predict_proba(test_ds[cols])[:, 1]
    y = test_ds["label"]

    ci = evaluate.bootstrap_auc_ci(y, p)
    calib = evaluate.calibration_table(y, p)
    bands = evaluate.band_validation(y, p)
    cost = evaluate.optimal_threshold(y, p, test_ds["open_balance"])
    logger.info("Bootstrap 95%% CI for ROC-AUC: [%.3f, %.3f]", ci["ci_low"], ci["ci_high"])

    report = {
        "holdout_metrics": bundle.metrics,
        "walk_forward": {
            "mean_roc_auc": float(wf["roc_auc"].mean()), "std_roc_auc": float(wf["roc_auc"].std()),
            "folds": wf.to_dict("records"),
        },
        "bootstrap_ci_roc_auc": ci,
        "band_validation": bands.reset_index().to_dict("records"),
        "cost_optimal_threshold": {
            "threshold": cost["threshold"], "expected_cost": cost["expected_cost"],
            "current_high_cut": cost["current_high_cut"],
        },
    }
    with open(f"{out_dir}/metrics.json", "w") as fh:
        json.dump(report, fh, indent=2, default=float)

    fig, ax = plt.subplots()
    ax.plot(calib["predicted"], calib["observed"], marker="o", label="model")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed bad rate")
    ax.set_title("Calibration (held-out)")
    ax.legend()
    fig.savefig(f"{out_dir}/calibration.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    curve = cost["curve"]
    fig, ax = plt.subplots()
    ax.plot(curve["threshold"], curve["total_cost"], label="total expected cost")
    ax.axvline(cost["threshold"], color="green", linestyle="--", label=f"cost-optimal ({cost['threshold']:.2f})")
    ax.axvline(SETTINGS.band.high_cut, color="red", linestyle=":", label=f"current High cut ({SETTINGS.band.high_cut})")
    ax.set_xlabel("Score threshold")
    ax.set_ylabel("Expected cost ($, illustrative unit costs)")
    ax.set_title("Cost-based threshold sweep")
    ax.legend()
    fig.savefig(f"{out_dir}/cost_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.bar(wf["snapshot"], wf["roc_auc"], color="#4a6fa5")
    ax.axhline(wf["roc_auc"].mean(), color="black", linestyle="--", label="mean")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Walk-forward ROC-AUC by fold")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.savefig(f"{out_dir}/walk_forward_auc.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    logger.info("Report written to %s/", out_dir)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true",
                        help="also run walk-forward CV, bootstrap CI, calibration and cost analysis")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    ensure_sample_data(args.data_dir)
    buyers, invoices = load_data(f"{args.data_dir}/buyers.csv", f"{args.data_dir}/invoices.csv")

    bundle = fit(buyers, invoices)
    logger.info("Hold-out metrics:\n%s", json.dumps(bundle.metrics, indent=2))

    os.makedirs("models", exist_ok=True)
    joblib.dump(bundle, "models/bundle.joblib")

    if args.report:
        full_report(buyers, invoices, bundle)
