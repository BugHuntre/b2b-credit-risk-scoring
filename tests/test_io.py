import os

import pandas as pd
import pytest

from creditrisk.io import SchemaError, ensure_sample_data, load_data


def test_ensure_sample_data_generates_once(tmp_path):
    data_dir = str(tmp_path / "data")
    ensure_sample_data(data_dir)
    assert os.path.exists(f"{data_dir}/buyers.csv")
    assert os.path.exists(f"{data_dir}/invoices.csv")

    # a second call must not overwrite existing data
    mtime = os.path.getmtime(f"{data_dir}/invoices.csv")
    ensure_sample_data(data_dir)
    assert os.path.getmtime(f"{data_dir}/invoices.csv") == mtime


def test_load_data_round_trip(tmp_path):
    data_dir = str(tmp_path / "data")
    ensure_sample_data(data_dir)
    buyers, invoices = load_data(f"{data_dir}/buyers.csv", f"{data_dir}/invoices.csv")
    assert len(buyers) > 0
    assert pd.api.types.is_datetime64_any_dtype(invoices["invoice_date"])


def test_load_data_rejects_missing_columns(tmp_path):
    bad_buyers = tmp_path / "buyers.csv"
    bad_buyers.write_text("buyer_id,buyer_name\nB1,Test Co\n")
    good_invoices = tmp_path / "invoices.csv"
    ensure_sample_data(str(tmp_path / "data"))
    good_invoices.write_text((tmp_path / "data" / "invoices.csv").read_text())

    with pytest.raises(SchemaError):
        load_data(str(bad_buyers), str(good_invoices))
