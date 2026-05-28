from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class UniversalRobustnessConfig:
    """Config for the universal strategy-return robustness test.

    Inputs and outputs use decimal returns: 0.10 means +10%.
    """

    preset: str = "balanced"
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 42
    daily_block_length: int = 21
    weekly_block_length: int = 13
    monthly_block_length: int = 4
    prob_cagr_positive_min: float = 0.95
    bootstrap_cagr_p05_min: float = 0.0
    bootstrap_sharpe_p05_min: float = 0.0
    prob_target_positive_min: float = 0.80
    target_metric: str = "cagr"
    half_1_cagr_min: float = 0.0
    half_2_cagr_min: float = 0.0
    leave_one_year_out_min_cagr: float = 0.0
    max_drawdown_limit: float = 0.35
    min_profit_months_pct: float = 0.52
    min_profit_years_pct: float = 0.50
    min_oos_is_ratio: float = 0.50
    max_single_period_profit_contribution: float = 0.40
    psr_min: float = 0.95
    dsr_min: float = 0.0
    cost_stress_multiplier: float = 2.0
    cost_stress_score_retention_min: float = 0.70
    near_duplicate_corr_threshold: float = 0.98
    position_jaccard_threshold: float = 0.90
    benchmark_corr_max: float = 0.80
    beta_abs_max: float = 1.50
    slippage_bps: float = 1.0
    commission_bps: float = 0.5
    duplicate_round_decimals: int = 8
    generated_methods: tuple[str, ...] = (
        "beam",
        "genetic",
        "github_ml",
        "aurora_ml",
        "random",
        "random_broad",
        "bandit",
        "hpo",
        "machine_learning",
        "optuna_tpe_hyperband",
        "sobol_random_asha_real",
        "dehb_real",
        "bohb_real",
        "smac_mf_real",
    )
    manual_methods: tuple[str, ...] = ("manual", "handcrafted")
    min_periods_by_frequency: dict[str, int] = field(
        default_factory=lambda: {"daily": 504, "weekly": 156, "monthly": 60}
    )
    min_active_periods_by_frequency: dict[str, int] = field(
        default_factory=lambda: {"daily": 100, "weekly": 40, "monthly": 18}
    )
    periods_per_year_by_frequency: dict[str, int] = field(
        default_factory=lambda: {"daily": 252, "weekly": 52, "monthly": 12}
    )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def block_length(self, frequency: str) -> int:
        frequency = str(frequency).lower()
        if frequency == "daily":
            return int(self.daily_block_length)
        if frequency == "weekly":
            return int(self.weekly_block_length)
        if frequency == "monthly":
            return int(self.monthly_block_length)
        raise ValueError(f"unknown frequency: {frequency!r}")

    def periods_per_year(self, frequency: str) -> int:
        frequency = str(frequency).lower()
        if frequency not in self.periods_per_year_by_frequency:
            raise ValueError(f"unknown frequency: {frequency!r}")
        return int(self.periods_per_year_by_frequency[frequency])


BALANCED_CONFIG = UniversalRobustnessConfig()
