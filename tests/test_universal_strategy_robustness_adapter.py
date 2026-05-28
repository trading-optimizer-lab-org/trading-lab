from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_universal_strategy_robustness import (  # noqa: E402
    _prepare_aurora_import,
    _unsupported_row,
    _weekly_positions_to_universal_returns,
)


def test_prepare_aurora_import_loads_quantforge_package() -> None:
    _prepare_aurora_import(Path.home() / "QuantForge")
    import aurora
    from aurora.validation.universal_robustness import run_batch_universal_robustness

    assert str(Path(aurora.__file__).resolve()).endswith("QuantForge\\__init__.py")
    assert callable(run_batch_universal_robustness)


def test_weekly_positions_map_periods_to_universal_sample_roles() -> None:
    positions = pd.DataFrame(
        {
            "candidate_id": ["c1", "c1", "c1"],
            "method": ["beam", "beam", "beam"],
            "target_week_end": pd.date_range("2020-01-03", periods=3, freq="W-FRI"),
            "period": ["train", "validation", "locked"],
            "strategy_return": [0.01, -0.005, 0.002],
            "spy_return": [0.002, -0.001, 0.001],
            "exposure": [1.0, 0.5, 0.0],
            "traded_asset": ["SPY", "SPY", "SPY"],
            "target_year": [2020, 2020, 2020],
        }
    )

    returns = _weekly_positions_to_universal_returns(positions, "run_1", "beam", 10, True)

    assert list(returns["sample_role"]) == ["is", "oos", "locked"]
    assert set(
        [
            "candidate_id",
            "run_id",
            "source_method",
            "timestamp",
            "frequency",
            "strategy_return",
            "returns_are_net",
            "n_trials",
        ]
    ).issubset(returns.columns)


def test_aurora_ml_without_returns_is_marked_unsupported() -> None:
    row = _unsupported_row("aurora_candidate", "run_1", "aurora_ml")

    assert row["source_method"] == "aurora_ml"
    assert row["robust_pass"] is False
    assert row["unsupported_reason"] == "unsupported_missing_returns"
