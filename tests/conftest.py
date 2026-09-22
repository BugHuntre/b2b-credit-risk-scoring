import pandas as pd
import pytest

from creditrisk import generate


@pytest.fixture
def tiny_buyer():
    """One buyer, no synthetic randomness -- hand-built invoices for exact expected values."""
    buyers = pd.DataFrame([{
        "buyer_id": "B0001", "buyer_name": "Test Co", "country": "USA", "segment": "Wholesaler",
        "onboarded_date": pd.Timestamp("2024-01-01"), "credit_limit": 100_000, "payment_terms_days": 30,
    }])
    return buyers


@pytest.fixture
def sample_data():
    """Small synthetic portfolio, fixed seed, for end-to-end / integration tests."""
    buyers, invoices = generate.generate(n_buyers=60, seed=1)
    return buyers, invoices
