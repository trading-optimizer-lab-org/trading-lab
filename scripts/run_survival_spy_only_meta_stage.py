from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from run_survival_spy_only_adaptive_stage import (  # noqa: E402
    _build_feature_catalog,
    _candidate_stream,
    _evaluate_many,
    _feature_family,
    _row_specs,
    _train_first_score,
)
from run_survival_spy_only_beam_stage import _mutate, _params_from_row  # noqa: E402
from trading_lab.config import load_optimization_config  # noqa: E402
from trading_lab.data_loader import load_market_data  # noqa: E402
from trading_lab.survival import public_feature_columns, split_train_validation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one SPY-only meta-search stage.")
    parser.add_argument("--method", choices=["bayesian", "bandit", "genetic"], required=True)
    parser.add_argument("--config", default="configs/survival_spy_only_github.yaml")
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--total-stages", type=int, default=32)
    parser.add_argument("--candidates-per-stage", type=int, default=900)
    args = parser.parse_args()

    config = load_optimization_config(args.config)
    data = load_market_data(config.data_path)
    train, _ = split_train_validation(data)
    catalog = _build_feature_catalog(train)
    spec_lookup = {str(item["spec"]): item for item in catalog}
    rng = np.random.default_rng(_method_seed(args.method) + args.stage)

    if args.method == "bayesian":
        rows = _run_bayesian_like(data, catalog, rng, args.candidates_per_stage, config)
    elif args.method == "bandit":
        rows = _run_bandit_allocator(data, catalog, rng, args.candidates_per_stage, config)
    else:
        rows = _run_genetic(data, catalog, spec_lookup, rng, args.candidates_per_stage, config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"survival_spy_only_{args.method}_stage_{args.stage}.csv"
    pd.DataFrame(rows).sort_values("survival_score", ascending=False).to_csv(output_path, index=False)
    print(
        json.dumps(
            {
                "method": args.method,
                "stage": args.stage,
                "rows": len(rows),
                "features": len(public_feature_columns(data)),
                "catalog_specs": len(catalog),
                "soft_pass": int(sum(bool(row.get("soft_pass")) for row in rows)),
                "accepted": int(sum(bool(row.get("accepted")) for row in rows)),
                "best_train_calmar": max((float(row.get("train_calmar", 0.0) or 0.0) for row in rows), default=0.0),
                "output_path": str(output_path),
                "locked_opened": False,
            },
            indent=2,
        )
    )
    return 0


def _run_bayesian_like(data, catalog, rng, budget: int, config) -> list[dict[str, object]]:
    warmup = min(max(80, budget // 5), budget)
    rows = _eval(data, _candidate_stream(catalog, rng=rng, count=warmup, learned_weights=None), config)
    remaining = budget - len(rows)
    while remaining > 0:
        weights = _spec_rewards(rows)
        pool = list(_candidate_stream(catalog, rng=rng, count=min(600, max(80, remaining * 4)), learned_weights=weights))
        chosen = sorted(pool, key=lambda params: _expected_improvement(params, weights, rows), reverse=True)[: min(remaining, 120)]
        new_rows = _eval(data, chosen, config)
        for row in new_rows:
            row["meta_method"] = "bayesian"
        rows.extend(new_rows)
        remaining = budget - len(rows)
    return rows


def _run_bandit_allocator(data, catalog, rng, budget: int, config) -> list[dict[str, object]]:
    families = sorted({str(item["family"]) for item in catalog})
    by_family = defaultdict(list)
    for item in catalog:
        by_family[str(item["family"])].append(item)
    rows: list[dict[str, object]] = []
    pulls = {family: 0 for family in families}
    rewards = {family: 0.0 for family in families}
    batch = 30
    while len(rows) < budget:
        family = _ucb_family(families, pulls, rewards, total_pulls=max(1, sum(pulls.values())))
        count = min(batch, budget - len(rows))
        candidates = list(_single_family_candidates(by_family[family], rng=rng, count=count))
        new_rows = _eval(data, candidates, config)
        for row in new_rows:
            row["meta_method"] = "bandit"
            row["allocated_family"] = family
        rows.extend(new_rows)
        pulls[family] += max(1, len(new_rows))
        rewards[family] += sum(max(0.0, _train_first_score(row)) for row in new_rows)
    return rows


def _run_genetic(data, catalog, spec_lookup, rng, budget: int, config) -> list[dict[str, object]]:
    population_size = min(96, max(24, budget // 5))
    population = list(_candidate_stream(catalog, rng=rng, count=population_size, learned_weights=None))
    rows = _eval(data, population, config)
    generation = 0
    while len(rows) < budget:
        generation += 1
        parents = sorted(rows, key=_genetic_score, reverse=True)[:24]
        children = []
        seen = {json.dumps(_params_from_row(row), sort_keys=True) for row in parents}
        while len(children) < min(120, budget - len(rows)):
            first = _params_from_row(parents[int(rng.integers(0, len(parents)))])
            second = _params_from_row(parents[int(rng.integers(0, len(parents)))])
            child = _crossover(first, second, catalog, rng)
            child = _mutate(child, catalog, spec_lookup, rng)
            key = json.dumps(child, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            children.append(child)
        child_rows = _eval(data, children, config)
        for row in child_rows:
            row["meta_method"] = "genetic"
            row["genetic_generation"] = generation
        rows.extend(child_rows)
    return rows


def _eval(data, candidates, config) -> list[dict[str, object]]:
    return _evaluate_many(
        data,
        candidates,
        initial_cash=config.execution.initial_cash,
        commission_bps=config.execution.commission_bps,
        slippage_bps=config.execution.slippage_bps,
    )


def _single_family_candidates(catalog: list[dict[str, object]], *, rng: np.random.Generator, count: int):
    seen: set[str] = set()
    attempts = 0
    while len(seen) < count and attempts < count * 40:
        attempts += 1
        combo_size = int(rng.choice([2, 3, 4, 5], p=[0.25, 0.35, 0.30, 0.10]))
        selected = rng.choice(len(catalog), size=min(combo_size, len(catalog)), replace=False)
        specs = sorted({str(catalog[int(index)]["spec"]) for index in selected})
        if len(specs) < 2:
            continue
        rule = str(rng.choice(["spy_long_short_always", "spy_long_short_score"], p=[0.75, 0.25]))
        confirm_days = int(rng.choice([1, 2, 3, 5, 8]))
        min_hold_days = int(rng.choice([1, 3, 5, 10, 15, 20]))
        if rule == "spy_long_short_score":
            params: dict[str, object] = {
                "rule": rule,
                "feature_specs": ";".join(specs),
                "score_threshold": float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0])),
                "confirm_days": confirm_days,
                "min_hold_days": min_hold_days,
                "combo_size": len(specs),
            }
        else:
            rng.shuffle(specs)
            long_count = max(1, len(specs) // 2)
            params = {
                "rule": "spy_long_short_always",
                "long_specs": ";".join(sorted(specs[:long_count])),
                "short_specs": ";".join(sorted(specs[long_count:])),
                "long_min_votes": 1,
                "short_min_votes": 1,
                "confirm_days": confirm_days,
                "min_hold_days": min_hold_days,
                "combo_size": len(specs),
            }
        key = json.dumps(params, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        yield params


def _spec_rewards(rows: list[dict[str, object]]) -> dict[str, float]:
    rewards: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in sorted(rows, key=_train_first_score, reverse=True)[: max(10, len(rows) // 4)]:
        score = max(0.0, _train_first_score(row))
        for spec in _row_specs(row):
            rewards[spec] = rewards.get(spec, 0.0) + score
            counts[spec] = counts.get(spec, 0) + 1
            family = _feature_family(spec.split("|", 1)[0])
            rewards[f"family::{family}"] = rewards.get(f"family::{family}", 0.0) + score * 0.5
            counts[f"family::{family}"] = counts.get(f"family::{family}", 0) + 1
    return {key: 1.0 + rewards[key] / max(1, counts[key]) / 8.0 for key in rewards}


def _expected_improvement(
    params: dict[str, object],
    weights: dict[str, float],
    rows: list[dict[str, object]],
) -> float:
    specs = _params_specs(params)
    exploitation = sum(weights.get(spec, 1.0) for spec in specs) / max(1, len(specs))
    families = {_feature_family(spec.split("|", 1)[0]) for spec in specs}
    family_score = sum(weights.get(f"family::{family}", 1.0) for family in families) / max(1, len(families))
    tried = {spec for row in rows for spec in _row_specs(row)}
    novelty = sum(1 for spec in specs if spec not in tried) / max(1, len(specs))
    return float(exploitation + 0.6 * family_score + 0.4 * novelty)


def _ucb_family(families: list[str], pulls: dict[str, int], rewards: dict[str, float], *, total_pulls: int) -> str:
    for family in families:
        if pulls[family] == 0:
            return family
    scores = {}
    for family in families:
        average = rewards[family] / pulls[family]
        exploration = np.sqrt(2.0 * np.log(total_pulls + 1) / pulls[family])
        scores[family] = average + exploration
    return max(scores, key=scores.get)


def _genetic_score(row: dict[str, object]) -> float:
    diversity = len({spec.split("|", 1)[0] for spec in _row_specs(row)}) * 0.2
    return _train_first_score(row) + diversity + (20.0 if bool(row.get("soft_pass")) else 0.0)


def _crossover(first: dict[str, object], second: dict[str, object], catalog, rng) -> dict[str, object]:
    specs = sorted(set(_params_specs(first)[:3] + _params_specs(second)[:3]))
    if len(specs) < 2:
        specs.append(str(catalog[int(rng.integers(0, len(catalog)))]["spec"]))
    specs = specs[:5]
    rule = str(rng.choice([first.get("rule", "spy_long_short_always"), second.get("rule", "spy_long_short_score")]))
    confirm_days = int(rng.choice([first.get("confirm_days", 1), second.get("confirm_days", 1), 2, 3, 5]))
    min_hold_days = int(rng.choice([first.get("min_hold_days", 1), second.get("min_hold_days", 1), 5, 10, 15]))
    if rule == "spy_long_short_score":
        return {
            "rule": "spy_long_short_score",
            "feature_specs": ";".join(specs),
            "score_threshold": float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0])),
            "confirm_days": confirm_days,
            "min_hold_days": min_hold_days,
            "combo_size": len(specs),
        }
    rng.shuffle(specs)
    long_count = max(1, len(specs) // 2)
    return {
        "rule": "spy_long_short_always",
        "long_specs": ";".join(sorted(specs[:long_count])),
        "short_specs": ";".join(sorted(specs[long_count:])),
        "long_min_votes": 1,
        "short_min_votes": 1,
        "confirm_days": confirm_days,
        "min_hold_days": min_hold_days,
        "combo_size": len(specs),
    }


def _params_specs(params: dict[str, object]) -> list[str]:
    specs: list[str] = []
    for key in ("long_specs", "short_specs", "feature_specs"):
        value = str(params.get(key, ""))
        specs.extend([part for part in value.split(";") if part and part.lower() != "nan"])
    return specs


def _method_seed(method: str) -> int:
    return {"bayesian": 120_000, "bandit": 130_000, "genetic": 140_000}[method]


if __name__ == "__main__":
    raise SystemExit(main())
