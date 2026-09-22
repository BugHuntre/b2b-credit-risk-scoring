"""All tunable assumptions in one place, so they're visible and easy to justify or override.

Every value here is a policy or modelling choice, not a derived fact. Anyone deploying this
on real data should read this file before trusting the output.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LabelConfig:
    """Defines the target: what counts as a 'bad' buyer, and how far ahead we predict."""
    horizon_days: int = 90        # predict delinquency in the next N days
    severe_days: int = 60         # "severely late" threshold that defines a bad outcome
    late_days: int = 7            # lower threshold used only for descriptive features


@dataclass(frozen=True)
class BandConfig:
    """Score -> risk band cut-offs, and the policy action attached to each band.

    Cut-offs are picked from the training data's label rate (see evaluate.suggest_cutoffs),
    not fixed arbitrarily — but review them against your own risk appetite before using them
    to gate real credit decisions.
    """
    high_cut: float = 0.40
    medium_cut: float = 0.15
    actions: dict = field(default_factory=lambda: {
        "High": "Credit hold: prepayment/LC for new orders, escalate overdue collection",
        "Medium": "Tighten terms, reduce limit, chase overdue invoices",
        "Low": "Standard terms",
    })
    limit_factor: dict = field(default_factory=lambda: {"High": 0.0, "Medium": 0.75, "Low": 1.0})


@dataclass(frozen=True)
class CostConfig:
    """Illustrative cost-benefit assumptions for threshold selection (evaluate.optimal_threshold).

    These are placeholders, not observed figures. Replace them with your own loss-given-default
    and margin numbers before using the cost-optimal threshold operationally.
    """
    loss_given_default: float = 0.70   # fraction of exposure lost when a flagged-too-late bad buyer defaults
    margin_on_revenue: float = 0.12    # gross margin lost when a good buyer is wrongly denied credit


@dataclass(frozen=True)
class ModelConfig:
    n_estimators: int = 250
    max_depth: int = 3
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    random_state: int = 0


@dataclass(frozen=True)
class Settings:
    label: LabelConfig = field(default_factory=LabelConfig)
    band: BandConfig = field(default_factory=BandConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    snapshot_step: str = "30D"
    feature_warmup_days: int = 270   # history needed before a snapshot's features are meaningful
    min_snapshots: int = 8
    n_wf_folds: int = 4              # walk-forward folds for evaluation


SETTINGS = Settings()
