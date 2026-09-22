"""Score every buyer as of a date: risk score, band, reason codes, suggested action."""
import numpy as np
import pandas as pd
import shap

from .config import SETTINGS, Settings
from .features import FEATURE_INFO, build_features
from .model import ModelBundle


def band(p: float, settings: Settings = SETTINGS) -> str:
    if p >= settings.band.high_cut:
        return "High"
    if p >= settings.band.medium_cut:
        return "Medium"
    return "Low"


def _reasons(shap_row: np.ndarray, x_row: pd.Series, cols: list[str], k: int) -> str:
    out = []
    for i in np.argsort(shap_row)[::-1][:k]:
        if shap_row[i] <= 0.05:
            break
        name = cols[i]
        if name == "late_trend" and not x_row[name] > 0:   # "improving" is not a risk reason to show
            continue
        label, fmt = FEATURE_INFO.get(name, (name, str))
        v = x_row[name]
        out.append(f"{label}: {fmt(v)}" if pd.notna(v) else f"{label}: n/a")
    return " | ".join(out) if out else "No strong risk drivers"


def score_buyers(bundle: ModelBundle, buyers: pd.DataFrame, invoices: pd.DataFrame,
                 as_of, top_k: int = 3) -> pd.DataFrame:
    settings = bundle.settings
    feats = build_features(buyers, invoices, as_of)
    X = feats[bundle.features]
    prob = bundle.model.predict_proba(X)[:, 1]
    contrib = shap.TreeExplainer(bundle.model).shap_values(X)

    out = buyers.merge(feats[["buyer_id", "open_balance", "overdue_balance", "utilization",
                              "avg_days_late_90d", "max_overdue_days"]], on="buyer_id")
    out["risk_score"] = (prob * 100).round(0).astype(int)
    out["risk_band"] = [band(p, settings) for p in prob]
    out["exposure_at_risk"] = (prob * feats["open_balance"].to_numpy()).round(0)
    out["suggested_limit"] = [round(cl * settings.band.limit_factor[b] / 1000) * 1000
                              for cl, b in zip(out["credit_limit"], out["risk_band"], strict=True)]
    out["action"] = out["risk_band"].map(settings.band.actions)
    out["reasons"] = [_reasons(contrib[i], X.iloc[i], bundle.features, top_k) for i in range(len(X))]
    return out.sort_values("exposure_at_risk", ascending=False).reset_index(drop=True)
