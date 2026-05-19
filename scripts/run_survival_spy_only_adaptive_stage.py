from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_lab.config import load_optimization_config
from trading_lab.data_loader import load_market_data
from trading_lab.survival import (
    VALIDATION_END,
    VALIDATION_START,
    SurvivalCriteria,
    encode_feature_spec,
    evaluate_survival_candidate,
    public_feature_columns,
    split_train_validation,
    survival_score,
    validate_spy_only_candidate,
)
from trading_lab.survival import _run_candidate, _survival_metrics


TRAIN_RESCUE_CRITERIA = SurvivalCriteria(
    min_train_calmar=0.80,
    min_validation_calmar=0.80,
    max_train_calmar=8.0,
    max_train_validation_ratio=3.5,
    min_train_cagr=0.04,
    min_validation_cagr=0.04,
    max_train_mdd=0.35,
    max_validation_mdd=0.35,
    min_trades_per_year=6.0,
    max_trades_per_year=120.0,
    min_long_fraction=0.30,
    max_long_fraction=0.85,
)
STRICT_CRITERIA = SurvivalCriteria()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one adaptive train-first SPY-only survival stage.")
    parser.add_argument("--config", default="configs/survival_spy_only_github.yaml")
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--total-stages", type=int, default=128)
    parser.add_argument("--candidates-per-stage", type=int, default=2500)
    args = parser.parse_args()

    config = load_optimization_config(args.config)
    data = load_market_data(config.data_path)
    train, _ = split_train_validation(data)
    catalog = _build_feature_catalog(train)
    rows = []
    rng = np.random.default_rng(50_000 + args.stage)

    first_wave = min(args.candidates_per_stage, max(20, args.candidates_per_stage // 4))
    first_candidates = list(_candidate_stream(catalog, rng=rng, count=first_wave, learned_weights=None))
    rows.extend(
        _evaluate_many(
            data,
            first_candidates,
            initial_cash=config.execution.initial_cash,
            commission_bps=config.execution.commission_bps,
            slippage_bps=config.execution.slippage_bps,
        )
    )
    learned_weights = _learn_spec_weights(rows)
    remaining = max(0, args.candidates_per_stage - len(rows))
    second_candidates = list(_candidate_stream(catalog, rng=rng, count=remaining, learned_weights=learned_weights))
    rows.extend(
        _evaluate_many(
            data,
            second_candidates,
            initial_cash=config.execution.initial_cash,
            commission_bps=config.execution.commission_bps,
            slippage_bps=config.execution.slippage_bps,
        )
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"survival_spy_only_adaptive_stage_{args.stage}.csv"
    pd.DataFrame(rows).sort_values("survival_score", ascending=False).to_csv(output_path, index=False)
    print(
        json.dumps(
            {
                "stage": args.stage,
                "rows": len(rows),
                "features": len(public_feature_columns(data)),
                "catalog_specs": len(catalog),
                "soft_pass": int(sum(bool(row.get("soft_pass")) for row in rows)),
                "accepted": int(sum(bool(row.get("accepted")) for row in rows)),
                "output_path": str(output_path),
                "locked_opened": False,
            },
            indent=2,
        )
    )
    return 0


def _evaluate_many(
    data: pd.DataFrame,
    candidates: Iterable[dict[str, object]],
    *,
    initial_cash: float,
    commission_bps: float,
    slippage_bps: float,
) -> list[dict[str, object]]:
    rows = []
    for params in candidates:
        validate_spy_only_candidate(params)
        rows.append(
            _evaluate_walkforward_candidate(
                data,
                params,
                initial_cash=initial_cash,
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
            )
        )
    return rows


def _build_feature_catalog(train: pd.DataFrame) -> list[dict[str, object]]:
    catalog: list[dict[str, object]] = []
    for name in public_feature_columns(train):
        series = pd.to_numeric(train[name], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) < 500 or series.nunique(dropna=True) < 20:
            continue
        family = _feature_family(name)
        quantiles = series.quantile([0.20, 0.35, 0.50, 0.65, 0.80]).dropna().unique()
        for value in quantiles:
            if np.isfinite(value):
                for direction in (-1, 1):
                    catalog.append(
                        {
                            "spec": encode_feature_spec(name=name, kind="threshold", value=float(value), direction=direction),
                            "name": name,
                            "family": family,
                        }
                    )
        if float(series.min()) < 0.0 < float(series.max()):
            for direction in (-1, 1):
                catalog.append(
                    {
                        "spec": encode_feature_spec(name=name, kind="threshold", value=0.0, direction=direction),
                        "name": name,
                        "family": family,
                    }
                )
        for window in (20, 60, 120, 252):
            for threshold in (-1.0, -0.5, 0.0, 0.5, 1.0):
                for direction in (-1, 1):
                    catalog.append(
                        {
                            "spec": encode_feature_spec(
                                name=name,
                                kind="zscore",
                                value=threshold,
                                window=window,
                                direction=direction,
                            ),
                            "name": name,
                            "family": family,
                        }
                    )
    if len(catalog) < 20:
        raise ValueError("adaptive SPY-only search needs at least twenty usable feature specs")
    return catalog


def _candidate_stream(
    catalog: list[dict[str, object]],
    *,
    rng: np.random.Generator,
    count: int,
    learned_weights: dict[str, float] | None,
) -> Iterable[dict[str, object]]:
    seen: set[str] = set()
    families = sorted({str(item["family"]) for item in catalog})
    family_weights = _family_weights(families, learned_weights)
    by_family = defaultdict(list)
    for item in catalog:
        by_family[str(item["family"])].append(item)
    attempts = 0
    while len(seen) < count and attempts < count * 40:
        attempts += 1
        family_count = int(rng.choice([1, 2], p=[0.55, 0.45]))
        chosen_families = list(rng.choice(families, size=family_count, replace=False, p=family_weights))
        pool = [item for family in chosen_families for item in by_family[family]]
        combo_size = int(rng.choice([2, 3, 4, 5], p=[0.20, 0.30, 0.35, 0.15]))
        selected = _weighted_pick(pool, rng=rng, size=combo_size, learned_weights=learned_weights)
        if len(selected) < 2:
            continue
        specs = [str(item["spec"]) for item in selected]
        rule = str(rng.choice(["spy_long_short_always", "spy_long_short_score"], p=[0.72, 0.28]))
        confirm_days = int(rng.choice([1, 2, 3, 5, 8], p=[0.15, 0.25, 0.25, 0.25, 0.10]))
        min_hold_days = int(rng.choice([1, 3, 5, 10, 15, 20], p=[0.10, 0.20, 0.25, 0.25, 0.10, 0.10]))
        if rule == "spy_long_short_score":
            params: dict[str, object] = {
                "rule": rule,
                "feature_specs": ";".join(sorted(specs)),
                "score_threshold": float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0])),
                "confirm_days": confirm_days,
                "min_hold_days": min_hold_days,
                "combo_size": combo_size,
            }
        else:
            rng.shuffle(specs)
            long_count = max(1, combo_size // 2)
            long_specs = sorted(specs[:long_count])
            short_specs = sorted(specs[long_count:])
            if not short_specs:
                continue
            params = {
                "rule": rule,
                "long_specs": ";".join(long_specs),
                "short_specs": ";".join(short_specs),
                "long_min_votes": int(rng.choice([1, min(2, len(long_specs))])),
                "short_min_votes": int(rng.choice([1, min(2, len(short_specs))])),
                "confirm_days": confirm_days,
                "min_hold_days": min_hold_days,
                "combo_size": combo_size,
            }
        key = json.dumps(params, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        yield params


def _family_weights(families: list[str], learned_weights: dict[str, float] | None) -> np.ndarray:
    base = {
        "spy_price": 1.15,
        "rolling": 1.10,
        "momentum": 1.10,
        "volatility": 1.05,
        "credit": 1.00,
        "rates": 0.95,
        "defensive_cyclical": 0.95,
        "cross_asset": 0.90,
        "macro": 0.85,
        "other": 0.70,
    }
    raw = []
    for family in families:
        value = base.get(family, 0.80)
        if learned_weights:
            value *= learned_weights.get(f"family::{family}", 1.0)
        raw.append(value)
    weights = np.array(raw, dtype=float)
    return weights / weights.sum()


def _weighted_pick(
    pool: list[dict[str, object]],
    *,
    rng: np.random.Generator,
    size: int,
    learned_weights: dict[str, float] | None,
) -> list[dict[str, object]]:
    if not pool:
        return []
    weights = np.array([learned_weights.get(str(item["spec"]), 1.0) if learned_weights else 1.0 for item in pool], dtype=float)
    weights = weights / weights.sum()
    indexes = rng.choice(len(pool), size=min(size, len(pool)), replace=False, p=weights)
    return [pool[int(index)] for index in indexes]


def _learn_spec_weights(rows: list[dict[str, object]]) -> dict[str, float]:
    weights: dict[str, float] = {}
    ranked = sorted(rows, key=_train_first_score, reverse=True)[: max(10, len(rows) // 10)]
    for row in ranked:
        reward = max(0.25, 1.0 + _train_first_score(row) / 25.0)
        specs = _row_specs(row)
        for spec in specs:
            weights[spec] = max(weights.get(spec, 1.0), reward)
            family = _feature_family(spec.split("|", 1)[0])
            weights[f"family::{family}"] = max(weights.get(f"family::{family}", 1.0), min(3.0, reward))
    return weights


def _row_specs(row: dict[str, object]) -> list[str]:
    specs: list[str] = []
    for key in ("long_specs", "short_specs", "feature_specs"):
        value = str(row.get(key, ""))
        specs.extend([part for part in value.split(";") if part])
    return specs


def _train_first_score(row: dict[str, object]) -> float:
    train_calmar = float(row.get("train_calmar", 0.0) or 0.0)
    validation_calmar = float(row.get("validation_calmar", 0.0) or 0.0)
    train_block = float(row.get("train_block_min_calmar", -5.0) or -5.0)
    validation_block = float(row.get("validation_block_min_calmar", -5.0) or -5.0)
    robust = int(row.get("robust_passes", 0) or 0)
    return float(3.0 * train_calmar + validation_calmar + 0.8 * train_block + 0.4 * validation_block + robust)


def _evaluate_walkforward_candidate(
    data: pd.DataFrame,
    params: dict[str, object],
    *,
    initial_cash: float,
    commission_bps: float,
    slippage_bps: float,
) -> dict[str, object]:
    row = evaluate_survival_candidate(
        data,
        params,
        initial_cash=initial_cash,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        criteria=STRICT_CRITERIA,
    )
    if _early_rejection(row) is not None:
        row.update(
            {
                "train_block_min_calmar": -999.0,
                "validation_block_min_calmar": -999.0,
                "train_blocks_positive": 0,
                "validation_blocks_positive": 0,
                "walkforward_passes": 0,
                "walkforward_total": 4,
                "validation_start": VALIDATION_START,
                "validation_end": VALIDATION_END,
                "soft_pass": False,
                "soft_rejection_reason": _early_rejection(row),
                "survival_score": survival_score(row) + _train_first_score(row),
                "locked_opened": False,
            }
        )
        return row
    train, validation = split_train_validation(data)
    train_blocks = _block_metrics(train, params, initial_cash, commission_bps, slippage_bps, blocks=6)
    validation_blocks = _block_metrics(validation, params, initial_cash, commission_bps, slippage_bps, blocks=6)
    row.update(
        {
            "train_block_min_calmar": _min_value(train_blocks, "calmar"),
            "validation_block_min_calmar": _min_value(validation_blocks, "calmar"),
            "train_blocks_positive": _count_at_least(train_blocks, "calmar", 0.0),
            "validation_blocks_positive": _count_at_least(validation_blocks, "calmar", 0.0),
            "walkforward_passes": _walkforward_passes(train_blocks, validation_blocks),
            "walkforward_total": 4,
            "validation_start": VALIDATION_START,
            "validation_end": VALIDATION_END,
        }
    )
    soft_rejection = TRAIN_RESCUE_CRITERIA.rejection_reason(row)
    row["soft_pass"] = soft_rejection is None and _walkforward_soft_rejection(row) is None
    row["soft_rejection_reason"] = soft_rejection or _walkforward_soft_rejection(row)
    hard_walkforward_rejection = _walkforward_hard_rejection(row)
    if row["accepted"] and hard_walkforward_rejection is not None:
        row["accepted"] = False
        row["rejection_reason"] = hard_walkforward_rejection
    row["survival_score"] = survival_score(row) + _train_first_score(row) + float(row["walkforward_passes"]) * 4.0
    row["locked_opened"] = False
    return row


def _early_rejection(row: dict[str, object]) -> str | None:
    if float(row.get("train_calmar", 0.0) or 0.0) < 0.50:
        return "early_train_calmar"
    if abs(float(row.get("train_mdd", 0.0) or 0.0)) > 0.45:
        return "early_train_mdd"
    if float(row.get("trades_per_year", 0.0) or 0.0) < 4.0:
        return "early_too_few_trades"
    if float(row.get("trades_per_year", 0.0) or 0.0) > 150.0:
        return "early_too_many_trades"
    return None


def _block_metrics(
    data: pd.DataFrame,
    params: dict[str, object],
    initial_cash: float,
    commission_bps: float,
    slippage_bps: float,
    *,
    blocks: int,
) -> list[dict[str, float]]:
    chunks = [data.iloc[index_chunk].copy() for index_chunk in np.array_split(np.arange(len(data)), blocks) if len(index_chunk) > 0]
    metrics = []
    for chunk in chunks:
        result = _run_candidate(chunk, params, initial_cash, commission_bps, slippage_bps)
        metrics.append(_survival_metrics(result.equity_curve, result.metrics, chunk))
    return metrics


def _min_value(metrics: list[dict[str, float]], key: str) -> float:
    return float(min((metric[key] for metric in metrics), default=0.0))


def _count_at_least(metrics: list[dict[str, float]], key: str, threshold: float) -> int:
    return int(sum(1 for metric in metrics if metric[key] >= threshold))


def _walkforward_passes(train_blocks: list[dict[str, float]], validation_blocks: list[dict[str, float]]) -> int:
    return sum(
        [
            _min_value(train_blocks, "calmar") >= 0.50,
            _min_value(validation_blocks, "calmar") >= 0.50,
            _count_at_least(train_blocks, "calmar", 0.0) == 6,
            _count_at_least(validation_blocks, "calmar", 0.0) == 6,
        ]
    )


def _walkforward_soft_rejection(row: dict[str, object]) -> str | None:
    if float(row["train_block_min_calmar"]) < 0.0:
        return "soft_train_block_min_calmar"
    if float(row["validation_block_min_calmar"]) < 0.0:
        return "soft_validation_block_min_calmar"
    if int(row["train_blocks_positive"]) < 5:
        return "soft_train_blocks_positive"
    if int(row["validation_blocks_positive"]) < 5:
        return "soft_validation_blocks_positive"
    return None


def _walkforward_hard_rejection(row: dict[str, object]) -> str | None:
    if float(row["train_block_min_calmar"]) < 0.50:
        return "train_block_min_calmar"
    if float(row["validation_block_min_calmar"]) < 0.50:
        return "validation_block_min_calmar"
    if int(row["train_blocks_positive"]) < 6:
        return "train_blocks_positive"
    if int(row["validation_blocks_positive"]) < 6:
        return "validation_blocks_positive"
    return None


def _feature_family(name: str) -> str:
    lowered = name.lower()
    if lowered.startswith("spy_") or lowered in {"return_1d", "return_5d", "return_21d"}:
        return "spy_price"
    if any(token in lowered for token in ("roll", "ma_", "sma", "ema", "drawdown")):
        return "rolling"
    if any(token in lowered for token in ("momentum", "ret_", "trend", "breakout")):
        return "momentum"
    if any(token in lowered for token in ("vix", "vol", "variance", "atr")):
        return "volatility"
    if any(token in lowered for token in ("hyg", "lqd", "credit", "spread")):
        return "credit"
    if any(token in lowered for token in ("tnx", "yield", "rate", "bond", "tlt", "ief", "shy")):
        return "rates"
    if any(token in lowered for token in ("xlu", "xlv", "xlp", "xlk", "xly", "xli", "xlf", "sector")):
        return "defensive_cyclical"
    if any(token in lowered for token in ("gold", "oil", "dxy", "qqq", "iwm", "cross")):
        return "cross_asset"
    if any(token in lowered for token in ("macro", "cpi", "unemployment", "pmi", "fed")):
        return "macro"
    return "other"


if __name__ == "__main__":
    raise SystemExit(main())
