import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api as api_module
from creditrisk.model import fit


@pytest.fixture
def client_with_model(sample_data):
    """A TestClient with a real, freshly-trained model bundle injected directly
    into api._state -- avoids touching disk or FastAPI's lifespan startup."""
    buyers, invoices = sample_data
    api_module._state["bundle"] = fit(buyers, invoices)
    yield TestClient(api_module.app), buyers, invoices
    api_module._state.clear()


@pytest.fixture
def client_no_model():
    api_module._state["bundle"] = None
    yield TestClient(api_module.app)
    api_module._state.clear()


def _payload(buyers: pd.DataFrame, invoices: pd.DataFrame, n_buyers: int = 5, as_of=None) -> dict:
    b = buyers.head(n_buyers).copy()
    b["onboarded_date"] = b["onboarded_date"].dt.strftime("%Y-%m-%d")
    inv = invoices[invoices["buyer_id"].isin(b["buyer_id"])].copy()
    for col in ("invoice_date", "due_date", "paid_date"):
        inv[col] = inv[col].dt.strftime("%Y-%m-%d")
    # astype(object) first: pandas 3's default string dtype doesn't accept a plain Python None
    # via .where the way a legacy object column does (it round-trips back to NaN, which json.dumps
    # then rejects) -- object dtype makes the None assignment stick.
    inv = inv.astype(object).where(pd.notna(inv), None)
    body = {"buyers": b.to_dict("records"), "invoices": inv.to_dict("records")}
    if as_of:
        body["as_of"] = as_of
    return body


def test_health_without_model(client_no_model):
    resp = client_no_model.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "no_model", "model_loaded": False, "model_roc_auc": None, "model_data_end": None}


def test_health_with_model(client_with_model):
    client, _, _ = client_with_model
    resp = client.get("/health")
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert 0 <= body["model_roc_auc"] <= 1


def test_score_without_model_returns_503(client_no_model):
    resp = client_no_model.post("/score", json={"buyers": [], "invoices": []})
    assert resp.status_code == 503


def test_score_requires_at_least_one_buyer(client_with_model):
    client, _, _ = client_with_model
    resp = client.post("/score", json={"buyers": [], "invoices": []})
    assert resp.status_code == 422


def test_score_end_to_end(client_with_model):
    client, buyers, invoices = client_with_model
    resp = client.post("/score", json=_payload(buyers, invoices))
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["scored"]) == 5
    row = body["scored"][0]
    assert 0 <= row["risk_score"] <= 100
    assert row["risk_band"] in {"Low", "Medium", "High"}
    assert row["reasons"]


def test_score_buyer_with_no_invoices_still_scores(client_with_model):
    client, buyers, invoices = client_with_model
    payload = _payload(buyers, invoices, n_buyers=1)
    payload["invoices"] = []
    resp = client.post("/score", json=payload)
    assert resp.status_code == 200
    assert len(resp.json()["scored"]) == 1


def test_score_rejects_bad_amount(client_with_model):
    client, buyers, invoices = client_with_model
    payload = _payload(buyers, invoices, n_buyers=1)
    payload["invoices"] = []
    payload["buyers"][0]["credit_limit"] = -100  # must be > 0
    resp = client.post("/score", json=payload)
    assert resp.status_code == 422
