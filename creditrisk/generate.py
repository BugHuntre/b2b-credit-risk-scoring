"""Synthetic B2B apparel receivables: buyers + invoices.

Schema mirrors what you would export from an ERP/Tally/Excel ledger, so real data
can replace these two CSVs without touching the rest of the pipeline.

buyers.csv   : buyer_id, buyer_name, country, segment, onboarded_date,
               credit_limit, payment_terms_days
invoices.csv : invoice_id, buyer_id, invoice_date, due_date, amount,
               paid_date (blank = unpaid), disputed (0/1)
"""
import numpy as np
import pandas as pd

AS_OF = pd.Timestamp("2026-09-15")
START = pd.Timestamp("2023-01-01")

# segment -> (invoices per month, typical invoice amount in USD)
SEGMENTS = {
    "Retail chain": (1.6, 60000),
    "Wholesaler": (1.2, 40000),
    "Distributor": (1.0, 30000),
    "Online brand": (0.8, 15000),
    "Boutique": (0.5, 8000),
}
COUNTRIES = ["USA", "UK", "Germany", "France", "UAE", "Australia", "Canada", "Netherlands", "Italy", "Spain"]
PREFIX = ["Nova", "Urban", "Prime", "Bright", "Metro", "Heritage", "Coastal", "Summit", "Atlas", "Willow",
          "Crown", "Maple", "Union", "Orbit", "Vista", "Ember", "Harbor", "Lotus", "Pioneer", "Zenith"]
SUFFIX = ["Apparel", "Trading", "Fashion", "Textiles", "Retail", "Wear", "Outfitters", "Garments", "Brands", "Co"]


def generate(n_buyers: int = 400, seed: int = 7):
    rng = np.random.default_rng(seed)
    buyers, invoices = [], []
    inv_id = 1
    seg_names = list(SEGMENTS)

    for i in range(n_buyers):
        bid = f"B{i + 1:04d}"
        segment = rng.choice(seg_names, p=[0.15, 0.25, 0.2, 0.2, 0.2])
        rate, mean_amt = SEGMENTS[segment]
        size = rng.lognormal(0, 0.4)
        risk = rng.beta(1.5, 6)                      # latent riskiness, hidden from the model
        terms = int(rng.choice([30, 45, 60], p=[0.4, 0.4, 0.2]))
        onboarded = START + pd.Timedelta(days=int(rng.integers(0, 900)))
        credit_limit = round(rate * mean_amt * size * terms / 30 * 2.0 / 1000) * 1000

        # some buyers deteriorate mid-life; a share of those stop paying entirely
        det_date, stop_date, hold_date, severity = None, None, AS_OF, 0.0
        if rng.random() < 0.08 + 0.25 * risk:
            lo, hi = 150, (AS_OF - onboarded).days - 30
            if hi > lo:
                det_date = onboarded + pd.Timedelta(days=int(rng.integers(lo, hi)))
                severity = rng.uniform(0.5, 1.5)
                if rng.random() < 0.4:
                    stop_date = det_date + pd.Timedelta(days=int(rng.integers(30, 90)))
                    hold_date = stop_date + pd.Timedelta(days=45)   # credit hold after a lag

        buyers.append(dict(
            buyer_id=bid,
            buyer_name=f"{rng.choice(PREFIX)} {rng.choice(SUFFIX)} {i + 1}",
            country=rng.choice(COUNTRIES), segment=segment,
            onboarded_date=onboarded, credit_limit=max(credit_limit, 5000),
            payment_terms_days=terms,
        ))

        end = min(hold_date, AS_OF)
        span = (end - onboarded).days
        if span <= 0:
            continue
        n = rng.poisson(rate * span / 30)
        offsets = np.sort(rng.integers(0, span, n))
        for off in offsets:
            inv_date = onboarded + pd.Timedelta(days=int(off))
            due = inv_date + pd.Timedelta(days=terms)
            amount = round(float(rng.lognormal(np.log(mean_amt * size), 0.5)), 2)
            disputed = int(rng.random() < 0.02 + 0.2 * risk)

            paid = pd.NaT
            if stop_date is not None and inv_date >= stop_date:
                pass                                                  # never paid
            else:
                mean_late, sd = -2 + 30 * risk, 4 + 20 * risk
                if det_date is not None and inv_date >= det_date:
                    mean_late += 25 * severity + 5 * (inv_date - det_date).days / 30
                    sd *= 1.5
                late = rng.normal(mean_late, sd) + (rng.exponential(20) if disputed else 0)
                paid = due + pd.Timedelta(days=int(round(max(late, -10))))
                if paid > AS_OF:
                    paid = pd.NaT
            invoices.append(dict(invoice_id=f"INV{inv_id:06d}", buyer_id=bid, invoice_date=inv_date,
                                 due_date=due, amount=amount, paid_date=paid, disputed=disputed))
            inv_id += 1

    return pd.DataFrame(buyers), pd.DataFrame(invoices)


def write(out_dir: str = "data"):
    buyers, invoices = generate()
    buyers.to_csv(f"{out_dir}/buyers.csv", index=False)
    invoices.to_csv(f"{out_dir}/invoices.csv", index=False)
    return buyers, invoices
