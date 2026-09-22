"""Buyer-level features computed strictly from what was known on `as_of`."""
import numpy as np
import pandas as pd

LATE_DAYS = 7      # "late" threshold
VERY_LATE_DAYS = 30

# feature -> (human label, formatter) used for reason codes in the UI
FEATURE_INFO = {
    "avg_days_late_90d": ("Avg days late (last 90d)", lambda v: f"{v:.0f} days"),
    "avg_days_late_12m": ("Avg days late (12m)", lambda v: f"{v:.0f} days"),
    "late_trend": ("Lateness trend vs prior 90d", lambda v: f"{v:+.0f} days"),
    "max_days_late_12m": ("Worst delay (12m)", lambda v: f"{v:.0f} days"),
    "pct_late_12m": (f"Invoices paid >{LATE_DAYS}d late (12m)", lambda v: f"{v:.0%}"),
    "pct_very_late_12m": (f"Invoices paid >{VERY_LATE_DAYS}d late (12m)", lambda v: f"{v:.0%}"),
    "open_balance": ("Open balance", lambda v: f"${v:,.0f}"),
    "overdue_balance": ("Overdue balance", lambda v: f"${v:,.0f}"),
    "overdue_pct": ("Share of open balance overdue", lambda v: f"{v:.0%}"),
    "max_overdue_days": ("Oldest overdue invoice", lambda v: f"{v:.0f} days"),
    "utilization": ("Credit limit utilisation", lambda v: f"{v:.0%}"),
    "dispute_rate_12m": ("Dispute rate (12m)", lambda v: f"{v:.0%}"),
    "amount_cv_12m": ("Invoice amount volatility", lambda v: f"{v:.2f}"),
    "days_since_last_payment": ("Days since last payment", lambda v: f"{v:.0f} days"),
    "days_since_last_invoice": ("Days since last invoice", lambda v: f"{v:.0f} days"),
    "n_inv_12m": ("Invoices (12m)", lambda v: f"{v:.0f}"),
    "billed_12m": ("Billed (12m)", lambda v: f"${v:,.0f}"),
    "tenure_days": ("Relationship length", lambda v: f"{v:.0f} days"),
    "payment_terms_days": ("Payment terms", lambda v: f"{v:.0f} days"),
    "credit_limit": ("Credit limit", lambda v: f"${v:,.0f}"),
}


def _by_buyer(df: pd.DataFrame, col: str, fn: str) -> pd.Series:
    return df.groupby("buyer_id")[col].agg(fn)


def build_features(buyers: pd.DataFrame, invoices: pd.DataFrame, as_of) -> pd.DataFrame:
    as_of = pd.Timestamp(as_of)
    inv = invoices[invoices["invoice_date"] <= as_of].copy()
    inv["paid_by"] = inv["paid_date"].where(inv["paid_date"] <= as_of)   # unknown before it happened
    inv["is_open"] = inv["paid_by"].isna()
    inv["days_late"] = (inv["paid_by"].fillna(as_of) - inv["due_date"]).dt.days
    observed = inv[(inv["due_date"] <= as_of) | ~inv["is_open"]]         # lateness is known

    f = buyers.set_index("buyer_id")[["credit_limit", "payment_terms_days", "onboarded_date"]].copy()
    f = f[f["onboarded_date"] <= as_of]
    f["tenure_days"] = (as_of - f.pop("onboarded_date")).dt.days

    # --- payment behaviour, by due date window ---
    def window(days_from, days_to):
        lo, hi = as_of - pd.Timedelta(days=days_from), as_of - pd.Timedelta(days=days_to)
        return observed[(observed["due_date"] > lo) & (observed["due_date"] <= hi)]

    w12, w90, wprev = window(365, 0), window(90, 0), window(180, 90)
    f["avg_days_late_12m"] = _by_buyer(w12, "days_late", "mean")
    f["max_days_late_12m"] = _by_buyer(w12, "days_late", "max")
    f["pct_late_12m"] = w12.assign(x=w12["days_late"] > LATE_DAYS).groupby("buyer_id")["x"].mean()
    f["pct_very_late_12m"] = w12.assign(x=w12["days_late"] > VERY_LATE_DAYS).groupby("buyer_id")["x"].mean()
    f["avg_days_late_90d"] = _by_buyer(w90, "days_late", "mean")
    f["late_trend"] = f["avg_days_late_90d"] - _by_buyer(wprev, "days_late", "mean")

    # --- open receivables ---
    open_inv = inv[inv["is_open"]]
    overdue = open_inv[open_inv["due_date"] < as_of]
    f["open_balance"] = _by_buyer(open_inv, "amount", "sum")
    f["overdue_balance"] = _by_buyer(overdue, "amount", "sum")
    f[["open_balance", "overdue_balance"]] = f[["open_balance", "overdue_balance"]].fillna(0)
    f["overdue_pct"] = np.where(f["open_balance"] > 0, f["overdue_balance"] / f["open_balance"].where(f["open_balance"] > 0), 0)
    f["max_overdue_days"] = ((as_of - overdue["due_date"]).dt.days).groupby(overdue["buyer_id"]).max()
    f["max_overdue_days"] = f["max_overdue_days"].fillna(0)
    f["utilization"] = f["open_balance"] / f["credit_limit"]

    # --- activity ---
    r12 = inv[inv["invoice_date"] > as_of - pd.Timedelta(days=365)]
    f["n_inv_12m"] = _by_buyer(r12, "amount", "count")
    f["n_inv_12m"] = f["n_inv_12m"].fillna(0)
    f["billed_12m"] = _by_buyer(r12, "amount", "sum")
    f["billed_12m"] = f["billed_12m"].fillna(0)
    f["dispute_rate_12m"] = _by_buyer(r12, "disputed", "mean")
    f["amount_cv_12m"] = _by_buyer(r12, "amount", "std") / _by_buyer(r12, "amount", "mean")
    f["days_since_last_invoice"] = (as_of - _by_buyer(inv, "invoice_date", "max")).dt.days
    f["days_since_last_payment"] = (as_of - _by_buyer(inv, "paid_by", "max")).dt.days

    seg = buyers.set_index("buyer_id")["segment"]
    seg_dummies = pd.get_dummies(seg, prefix="segment", dtype=int)
    f = f.join(seg_dummies)
    return f.reset_index()


def feature_columns(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns if c != "buyer_id"]
