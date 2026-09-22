"""Loading and validating buyer/invoice data, shared by train.py and app.py."""
import logging
import os

import pandas as pd

from . import generate

logger = logging.getLogger(__name__)

BUYER_COLS = {"buyer_id", "buyer_name", "country", "segment", "onboarded_date",
              "credit_limit", "payment_terms_days"}
INVOICE_COLS = {"invoice_id", "buyer_id", "invoice_date", "due_date", "amount", "paid_date", "disputed"}


class SchemaError(ValueError):
    pass


def ensure_sample_data(data_dir: str = "data") -> None:
    if not os.path.exists(f"{data_dir}/invoices.csv"):
        os.makedirs(data_dir, exist_ok=True)
        generate.write(data_dir)
        logger.info("No data found: generated synthetic sample data in %s/", data_dir)


def load_data(buyers_src="data/buyers.csv", invoices_src="data/invoices.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    # Validate columns *before* parsing dates, so a malformed upload gets one clear SchemaError
    # instead of a raw pandas ValueError from parse_dates hitting a missing column.
    buyers = pd.read_csv(buyers_src)
    invoices = pd.read_csv(invoices_src)
    missing = (BUYER_COLS - set(buyers.columns)) | (INVOICE_COLS - set(invoices.columns))
    if missing:
        raise SchemaError(f"Missing required column(s): {sorted(missing)}")

    buyers["onboarded_date"] = pd.to_datetime(buyers["onboarded_date"])
    for col in ("invoice_date", "due_date", "paid_date"):
        invoices[col] = pd.to_datetime(invoices[col])
    return buyers, invoices
