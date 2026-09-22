import pandas as pd

from creditrisk.features import build_features


def _invoices(rows):
    df = pd.DataFrame(rows)
    for c in ["invoice_date", "due_date", "paid_date"]:
        df[c] = pd.to_datetime(df[c])
    return df


def test_open_and_overdue_balance(tiny_buyer):
    as_of = pd.Timestamp("2024-06-01")
    invoices = _invoices([
        # paid on time, before as_of -> not open
        dict(invoice_id="I1", buyer_id="B0001", invoice_date="2024-04-01", due_date="2024-05-01",
             amount=1000, paid_date="2024-04-25", disputed=0),
        # unpaid, due before as_of -> open AND overdue
        dict(invoice_id="I2", buyer_id="B0001", invoice_date="2024-04-15", due_date="2024-05-15",
             amount=2000, paid_date=None, disputed=0),
        # unpaid, due after as_of -> open but NOT overdue
        dict(invoice_id="I3", buyer_id="B0001", invoice_date="2024-05-20", due_date="2024-06-20",
             amount=3000, paid_date=None, disputed=0),
    ])
    f = build_features(tiny_buyer, invoices, as_of).set_index("buyer_id").loc["B0001"]

    assert f["open_balance"] == 5000          # I2 + I3
    assert f["overdue_balance"] == 2000        # I2 only
    assert f["overdue_pct"] == 0.4
    assert f["utilization"] == 5000 / 100_000


def test_no_lookahead(tiny_buyer):
    """An invoice dated after `as_of` must not change any feature computed at `as_of`."""
    as_of = pd.Timestamp("2024-06-01")
    base = _invoices([
        dict(invoice_id="I1", buyer_id="B0001", invoice_date="2024-04-01", due_date="2024-05-01",
             amount=1000, paid_date="2024-04-20", disputed=0),
    ])
    future_row = _invoices([
        dict(invoice_id="I2", buyer_id="B0001", invoice_date="2024-07-01", due_date="2024-08-01",
             amount=999_999, paid_date=None, disputed=1),
    ])
    with_future = pd.concat([base, future_row], ignore_index=True)

    f1 = build_features(tiny_buyer, base, as_of).set_index("buyer_id").loc["B0001"]
    f2 = build_features(tiny_buyer, with_future, as_of).set_index("buyer_id").loc["B0001"]
    pd.testing.assert_series_equal(f1, f2, check_names=False)


def test_unresolved_future_due_date_not_counted_as_late(tiny_buyer):
    """An invoice due AFTER as_of, still unpaid, is open -- but its lateness is unknown, not bad."""
    as_of = pd.Timestamp("2024-06-01")
    invoices = _invoices([
        dict(invoice_id="I1", buyer_id="B0001", invoice_date="2024-05-25", due_date="2024-07-01",
             amount=5000, paid_date=None, disputed=0),
    ])
    f = build_features(tiny_buyer, invoices, as_of).set_index("buyer_id").loc["B0001"]
    assert f["overdue_balance"] == 0
    assert f["open_balance"] == 5000
