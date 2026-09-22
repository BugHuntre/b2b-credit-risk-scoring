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
