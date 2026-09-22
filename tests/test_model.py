import pandas as pd

from creditrisk.config import SETTINGS
from creditrisk.model import fit, make_labels, snapshot_dates


def _invoices(rows):
    df = pd.DataFrame(rows)
    for c in ["invoice_date", "due_date", "paid_date"]:
        df[c] = pd.to_datetime(df[c])
    return df


def test_make_labels_severe_late_is_bad():
    t = pd.Timestamp("2024-01-01")
    invoices = _invoices([
        dict(invoice_id="I1", buyer_id="B1", invoice_date="2024-01-05", due_date="2024-02-01",
             amount=100, paid_date="2024-04-15", disputed=0),   # 73 days late > severe_days (60)
    ])
    labels = make_labels(invoices, t)
    assert labels.loc["B1"] == 1


def test_make_labels_unpaid_within_horizon_is_bad():
    t = pd.Timestamp("2024-01-01")
    invoices = _invoices([
        dict(invoice_id="I1", buyer_id="B1", invoice_date="2024-01-05", due_date="2024-02-01",
             amount=100, paid_date=None, disputed=0),
    ])
    assert make_labels(invoices, t).loc["B1"] == 1


def test_make_labels_paid_on_time_is_good():
    t = pd.Timestamp("2024-01-01")
    invoices = _invoices([
        dict(invoice_id="I1", buyer_id="B1", invoice_date="2024-01-05", due_date="2024-02-01",
             amount=100, paid_date="2024-02-03", disputed=0),   # 2 days late, not severe
    ])
    assert make_labels(invoices, t).loc["B1"] == 0


def test_make_labels_ignores_invoices_outside_horizon():
    t = pd.Timestamp("2024-01-01")
    invoices = _invoices([
        # due well after the horizon window -> should not appear in the label at all
        dict(invoice_id="I1", buyer_id="B1", invoice_date="2024-06-01", due_date="2024-07-01",
             amount=100, paid_date=None, disputed=0),
    ])
    labels = make_labels(invoices, t)
    assert "B1" not in labels.index


def test_snapshot_dates_leave_a_resolvable_gap(sample_data):
    buyers, invoices = sample_data
    data_end = invoices["paid_date"].max()
    snaps = snapshot_dates(invoices, max(invoices["invoice_date"].max(), data_end))
    resolve_by = SETTINGS.label.horizon_days + SETTINGS.label.severe_days
    assert (data_end - snaps[-1]).days >= resolve_by


def test_fit_end_to_end_produces_sane_metrics(sample_data):
    buyers, invoices = sample_data
    bundle = fit(buyers, invoices)
    assert 0.5 <= bundle.metrics["roc_auc"] <= 1.0
    assert bundle.metrics["test_rows"] > 0
    assert set(bundle.features).issuperset({"avg_days_late_90d", "overdue_pct", "utilization"})
