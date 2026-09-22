import json
import os

import pandas as pd
import streamlit as st

from creditrisk.io import SchemaError, ensure_sample_data, load_data
from creditrisk.model import data_end_date, fit
from creditrisk.scoring import score_buyers

st.set_page_config(page_title="Buyer Credit Risk", layout="wide")

BAND_COLORS = {"High": "#d64545", "Medium": "#e0a030", "Low": "#3a9a5c"}


@st.cache_data
def load(buyers_src, invoices_src):
    return load_data(buyers_src, invoices_src)


@st.cache_resource(show_spinner="Training model on your data...")
def train(buyers, invoices):
    return fit(buyers, invoices)


@st.cache_data(show_spinner="Scoring buyers...")
def score(_bundle, buyers, invoices, as_of):
    return score_buyers(_bundle, buyers, invoices, as_of)


# ---------------- sidebar: data source ----------------
ensure_sample_data("data")
st.sidebar.header("Data")
up_b = st.sidebar.file_uploader("buyers.csv", type="csv")
up_i = st.sidebar.file_uploader("invoices.csv", type="csv")
try:
    buyers, invoices = load(up_b or "data/buyers.csv", up_i or "data/invoices.csv")
except SchemaError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()
st.sidebar.caption("Using uploaded files" if (up_b and up_i) else "Using sample data in data/")

bundle = train(buyers, invoices)
as_of = st.sidebar.date_input("Score as of", value=data_end_date(invoices).date())
scored = score(bundle, buyers, invoices, pd.Timestamp(as_of))

st.title("Buyer credit risk: B2B receivables")
st.caption("Risk score = model-estimated chance (0-100) that a buyer pays an invoice more than 60 days late "
           "or not at all within the next 90 days.")

tab_port, tab_watch, tab_buyer, tab_model = st.tabs(["Portfolio", "Watchlist", "Buyer detail", "Model quality"])

# ---------------- portfolio ----------------
with tab_port:
    open_ar = scored["open_balance"].sum()
    overdue = scored["overdue_balance"].sum()
    high = scored[scored["risk_band"] == "High"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Open receivables", f"${open_ar:,.0f}")
    c2.metric("Overdue", f"${overdue:,.0f}", f"{overdue / max(open_ar, 1):.0%} of open", delta_color="off")
    c3.metric("High-risk buyers", f"{len(high)}", f"${high['open_balance'].sum():,.0f} open", delta_color="off")
    c4.metric("Exposure at risk", f"${scored['exposure_at_risk'].sum():,.0f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Open balance by risk band")
        by_band = (scored.groupby("risk_band")["open_balance"].sum()
                   .reindex(["Low", "Medium", "High"]).fillna(0))
        st.bar_chart(by_band, color="#4a6fa5")
    with right:
        st.subheader("Buyers by risk band")
        counts = scored["risk_band"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
        st.bar_chart(counts, color="#4a6fa5")

    st.subheader("Top 10 exposures at risk")
    st.dataframe(scored.head(10)[["buyer_id", "buyer_name", "risk_score", "risk_band", "open_balance",
                                  "exposure_at_risk", "reasons"]], hide_index=True, width="stretch")

# ---------------- watchlist ----------------
with tab_watch:
    f1, f2, f3 = st.columns(3)
    bands = f1.multiselect("Risk band", ["High", "Medium", "Low"], default=["High", "Medium"])
    segs = f2.multiselect("Segment", sorted(scored["segment"].unique()))
    countries = f3.multiselect("Country", sorted(scored["country"].unique()))
    view = scored[scored["risk_band"].isin(bands)]
    if segs:
        view = view[view["segment"].isin(segs)]
    if countries:
        view = view[view["country"].isin(countries)]
    st.write(f"{len(view)} buyers")
    st.dataframe(
        view[["buyer_id", "buyer_name", "country", "segment", "risk_score", "risk_band", "open_balance",
              "overdue_balance", "utilization", "credit_limit", "suggested_limit", "action", "reasons"]],
        hide_index=True, width="stretch",
        column_config={
            "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%d"),
            "utilization": st.column_config.NumberColumn("Limit used", format="percent"),
            "open_balance": st.column_config.NumberColumn(format="$%d"),
            "overdue_balance": st.column_config.NumberColumn(format="$%d"),
            "credit_limit": st.column_config.NumberColumn(format="$%d"),
            "suggested_limit": st.column_config.NumberColumn(format="$%d"),
        })
    st.download_button("Download watchlist CSV", view.to_csv(index=False), "watchlist.csv", "text/csv")

# ---------------- buyer detail ----------------
with tab_buyer:
    choice = st.selectbox("Buyer", scored["buyer_id"] + "  " + scored["buyer_name"])
    row = scored[scored["buyer_id"] == choice.split()[0]].iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("Risk score", f"{row['risk_score']}")
    b.metric("Band", row["risk_band"])
    c.metric("Open balance", f"${row['open_balance']:,.0f}")
    d.metric("Credit limit", f"${row['credit_limit']:,.0f}", f"suggested ${row['suggested_limit']:,.0f}",
             delta_color="off")
    st.markdown(f"**Why:** {row['reasons']}")
    st.markdown(f"**Recommended action:** {row['action']}")

    hist = invoices[(invoices["buyer_id"] == row["buyer_id"]) & (invoices["invoice_date"] <= pd.Timestamp(as_of))].copy()
    paid = hist["paid_date"].where(hist["paid_date"] <= pd.Timestamp(as_of))
    hist["status"] = paid.notna().map({True: "Paid", False: "Open"})
    hist["days_late"] = (paid.fillna(pd.Timestamp(as_of)) - hist["due_date"]).dt.days
    st.subheader("Days late by invoice")
    st.bar_chart(hist.set_index("due_date")["days_late"], color="#4a6fa5")
    st.dataframe(hist.sort_values("invoice_date", ascending=False)[
        ["invoice_id", "invoice_date", "due_date", "amount", "paid_date", "status", "days_late", "disputed"]],
        hide_index=True, width="stretch")

# ---------------- model quality ----------------
with tab_model:
    m = bundle.metrics
    st.caption("Evaluated on the most recent snapshots, which the model never saw during training.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{m['roc_auc']:.2f}", f"baseline {m['baseline_roc_auc_avg_days_late']:.2f}", delta_color="off")
    c2.metric("PR-AUC", f"{m['pr_auc']:.2f}", f"baseline {m['baseline_pr_auc_avg_days_late']:.2f}", delta_color="off")
    c3.metric("Bad-buyer base rate", f"{m['base_rate']:.1%}")
    c4.metric("Bad buyers caught in top 20%", f"{m['recall_in_top_20pct']:.0%}")
    st.write(f"Trained on {m['train_rows']:,} buyer-snapshots, tested on {m['test_rows']:,}. "
             "The baseline is a single rule: average days late over the last 90 days.")

    report_path = "reports/metrics.json"
    if os.path.exists(report_path):
        with open(report_path) as fh:
            report = json.load(fh)
        st.divider()
        st.subheader("Deeper evaluation")
        st.caption("From `python train.py --report`: multi-fold walk-forward validation, a bootstrap "
                   "confidence interval, and a cost-based threshold sweep. See reports/ and README for detail.")

        wf = report["walk_forward"]
        ci = report["bootstrap_ci_roc_auc"]
        c1, c2 = st.columns(2)
        c1.metric("Walk-forward ROC-AUC (mean ± std)", f"{wf['mean_roc_auc']:.2f} ± {wf['std_roc_auc']:.2f}",
                  f"{len(wf['folds'])} folds", delta_color="off")
        c2.metric("95% bootstrap CI, hold-out ROC-AUC", f"[{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]")

        st.write("**Bad rate observed per risk band (held-out; should increase Low -> Medium -> High):**")
        st.dataframe(pd.DataFrame(report["band_validation"]), hide_index=True, width="stretch")

        imgs = [p for p in ("reports/walk_forward_auc.png", "reports/calibration.png", "reports/cost_curve.png")
                if os.path.exists(p)]
        if imgs:
            for col, path in zip(st.columns(len(imgs)), imgs, strict=True):
                col.image(path, width="stretch")
    else:
        st.caption("Run `python train.py --report` to add walk-forward validation, calibration and "
                   "cost-based threshold analysis here.")
