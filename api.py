"""FastAPI service wrapping the trained buyer credit risk model.

    python train.py                    # train once, writes models/bundle.joblib
    uvicorn api:app --reload           # serve it
    open http://localhost:8000/docs    # interactive Swagger UI

POST /score accepts a buyer's own invoice history and returns a risk score,
band, reason codes and a recommended action for each buyer -- the same
computation the Streamlit dashboard uses, exposed for another system to call.
See docs/MODEL_CARD.md for what the score means and its limitations.
"""
import logging
from collections.abc import Iterable
from contextlib import asynccontextmanager
from datetime import date

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from creditrisk.model import ModelBundle, data_end_date
from creditrisk.scoring import score_buyers

logger = logging.getLogger(__name__)
MODEL_PATH = "models/bundle.joblib"

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _state["bundle"] = joblib.load(MODEL_PATH)
        logger.info("Loaded model bundle from %s", MODEL_PATH)
    except FileNotFoundError:
        _state["bundle"] = None
        logger.warning("No model at %s -- run `python train.py` first. /score will return 503 until then.", MODEL_PATH)
    yield
    _state.clear()


app = FastAPI(
    title="Buyer Credit Risk API",
    description=(
        "Scores B2B buyers on the probability of severe payment delinquency "
        "(>60 days late, or unpaid) in the next 90 days, from their own invoice "
        "history. Trained and evaluated on synthetic data -- see docs/MODEL_CARD.md "
        "in the repo before using scores from this API operationally."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------- request / response schemas ----------------
# Field names and types mirror creditrisk.io.BUYER_COLS / INVOICE_COLS exactly,
# so a request body can be built straight from the same buyers.csv / invoices.csv rows.

class Buyer(BaseModel):
    buyer_id: str
    buyer_name: str
    country: str
    segment: str
    onboarded_date: date
    credit_limit: float = Field(gt=0)
    payment_terms_days: int = Field(gt=0)


class Invoice(BaseModel):
    invoice_id: str
    buyer_id: str
    invoice_date: date
    due_date: date
    amount: float = Field(gt=0)
    paid_date: date | None = None
    disputed: int = Field(ge=0, le=1)


class ScoreRequest(BaseModel):
    buyers: list[Buyer]
    invoices: list[Invoice] = []
    as_of: date | None = Field(default=None, description="Score as of this date; defaults to the latest invoice/payment date, or today if there are none.")


class ScoredBuyer(BaseModel):
    buyer_id: str
    buyer_name: str
    risk_score: int
    risk_band: str
    exposure_at_risk: float
    open_balance: float
    overdue_balance: float
    suggested_limit: float
    action: str
    reasons: str


class ScoreResponse(BaseModel):
    as_of: date
    model_roc_auc: float
    scored: list[ScoredBuyer]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_roc_auc: float | None = None
    model_data_end: str | None = None


# ---------------- helpers ----------------

_EMPTY_INVOICE_DTYPES = {
    "invoice_id": "object", "buyer_id": "object", "invoice_date": "datetime64[ns]",
    "due_date": "datetime64[ns]", "amount": "float64", "paid_date": "datetime64[ns]", "disputed": "int64",
}


def _buyers_frame(buyers: Iterable[Buyer]) -> pd.DataFrame:
    df = pd.DataFrame([b.model_dump() for b in buyers])
    df["onboarded_date"] = pd.to_datetime(df["onboarded_date"])
    return df


def _invoices_frame(invoices: Iterable[Invoice]) -> pd.DataFrame:
    invoices = list(invoices)
    if not invoices:
        return pd.DataFrame({col: pd.array([], dtype=dt) for col, dt in _EMPTY_INVOICE_DTYPES.items()})
    df = pd.DataFrame([i.model_dump() for i in invoices])
    for col in ("invoice_date", "due_date", "paid_date"):
        df[col] = pd.to_datetime(df[col])
    return df


def _bundle() -> ModelBundle:
    bundle = _state.get("bundle")
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run `python train.py` to create models/bundle.joblib, then restart the API.")
    return bundle


# ---------------- routes ----------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    bundle = _state.get("bundle")
    if bundle is None:
        return HealthResponse(status="no_model", model_loaded=False)
    return HealthResponse(
        status="ok", model_loaded=True,
        model_roc_auc=bundle.metrics["roc_auc"], model_data_end=str(bundle.data_end.date()),
    )


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    """Score every buyer in the request. A buyer with no invoices still gets a
    (low-confidence) score, falling back on feature defaults -- see the
    "small-N buyers" limitation in docs/MODEL_CARD.md."""
    bundle = _bundle()
    if not req.buyers:
        raise HTTPException(status_code=422, detail="buyers must not be empty")

    buyers = _buyers_frame(req.buyers)
    invoices = _invoices_frame(req.invoices)
    if req.as_of:
        as_of = pd.Timestamp(req.as_of)
    elif len(invoices):
        as_of = data_end_date(invoices)
    else:
        as_of = pd.Timestamp.today().normalize()

    try:
        scored = score_buyers(bundle, buyers, invoices, as_of)
    except Exception as e:
        logger.exception("Scoring failed")
        raise HTTPException(status_code=400, detail=f"Scoring failed: {e}") from e

    cols = ["buyer_id", "buyer_name", "risk_score", "risk_band", "exposure_at_risk",
            "open_balance", "overdue_balance", "suggested_limit", "action", "reasons"]
    return ScoreResponse(
        as_of=as_of.date(),
        model_roc_auc=bundle.metrics["roc_auc"],
        scored=[ScoredBuyer(**row) for row in scored[cols].to_dict("records")],
    )
