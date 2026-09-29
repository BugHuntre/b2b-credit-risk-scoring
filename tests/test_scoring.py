from creditrisk.config import SETTINGS
from creditrisk.model import fit
from creditrisk.scoring import band, score_buyers


def test_band_thresholds():
    assert band(0.0) == "Low"
    assert band(SETTINGS.band.medium_cut) == "Medium"
    assert band(SETTINGS.band.medium_cut - 1e-9) == "Low"
    assert band(SETTINGS.band.high_cut) == "High"
    assert band(0.99) == "High"


def test_score_buyers_end_to_end(sample_data):
    buyers, invoices = sample_data
    bundle = fit(buyers, invoices)
    as_of = bundle.data_end
    scored = score_buyers(bundle, buyers, invoices, as_of)

    assert len(scored) == len(buyers)
    assert scored["risk_score"].between(0, 100).all()
    assert set(scored["risk_band"]) <= {"Low", "Medium", "High"}
    # sorted by exposure at risk, descending
    assert (scored["exposure_at_risk"].diff().dropna() <= 1e-6).all()
    # every row has a non-empty reason string
    assert (scored["reasons"].str.len() > 0).all()


def test_suggested_limit_respects_band_policy(sample_data):
    buyers, invoices = sample_data
    bundle = fit(buyers, invoices)
    scored = score_buyers(bundle, buyers, invoices, bundle.data_end)
    high = scored[scored["risk_band"] == "High"]
    if len(high):
        assert (high["suggested_limit"] == 0).all()


def test_score_buyers_handles_a_batch_missing_some_segments(sample_data):
    """A small batch (e.g. one API request, or a filtered upload) won't contain every segment
    the model saw at training time -- pd.get_dummies then produces fewer one-hot columns than
    bundle.features expects. Scoring must not crash on this; the missing dummy columns are
    just 0 (that buyer isn't in those other segments)."""
    buyers, invoices = sample_data
    bundle = fit(buyers, invoices)
    assert buyers["segment"].nunique() > 1  # sanity: the full set has more than one segment

    one_buyer = buyers.iloc[[0]]
    inv = invoices[invoices["buyer_id"] == one_buyer["buyer_id"].iloc[0]]
    scored = score_buyers(bundle, one_buyer, inv, bundle.data_end)
    assert len(scored) == 1
    assert 0 <= scored["risk_score"].iloc[0] <= 100


def test_reasons_never_leak_raw_segment_column_names(sample_data):
    """A one-hot segment_* column is a modelling artifact, not a human-meaningful reason -- it
    should never show up verbatim in the explanation text (most likely to surface for a buyer
    with little else in their feature vector, e.g. no invoice history yet)."""
    buyers, invoices = sample_data
    bundle = fit(buyers, invoices)
    no_history = buyers.iloc[[0]]
    scored = score_buyers(bundle, no_history, invoices.iloc[0:0], bundle.data_end)
    assert "segment_" not in scored["reasons"].iloc[0]
