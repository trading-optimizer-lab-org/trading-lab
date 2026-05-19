from __future__ import annotations

import argparse
import json
import sys
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
    _row_specs,
    _train_first_score,
)
from trading_lab.config import load_optimization_config  # noqa: E402
from trading_lab.data_loader import load_market_data  # noqa: E402
from trading_lab.survival import public_feature_columns, split_train_validation  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one beam-search SPY-only survival stage.")
    parser.add_argument("--config", default="configs/survival_spy_only_github.yaml")
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--total-stages", type=int, default=48)
    parser.add_argument("--seed-pool", type=int, default=160)
    parser.add_argument("--beam-width", type=int, default=24)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--mutations-per-parent", type=int, default=8)
    args = parser.parse_args()

    config = load_optimization_config(args.config)
    data = load_market_data(config.data_path)
    train, _ = split_train_validation(data)
    catalog = _build_feature_catalog(train)
    spec_lookup = {str(item["spec"]): item for item in catalog}
    rng = np.random.default_rng(90_000 + args.stage)

    rows: list[dict[str, object]] = []
    seed_candidates = list(_candidate_stream(catalog, rng=rng, count=args.seed_pool, learned_weights=None))
    seed_rows = _evaluate_many(
        data,
        seed_candidates,
        initial_cash=config.execution.initial_cash,
        commission_bps=config.execution.commission_bps,
        slippage_bps=config.execution.slippage_bps,
    )
    rows.extend(seed_rows)
    beam = _select_beam(seed_rows, args.beam_width)
    seen = {json.dumps(_params_from_row(row), sort_keys=True) for row in beam}

    for generation in range(1, args.generations + 1):
        children = []
        for parent in beam:
            for _ in range(args.mutations_per_parent):
                child = _mutate(_params_from_row(parent), catalog, spec_lookup, rng)
                key = json.dumps(child, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                children.append(child)
        child_rows = _evaluate_many(
            data,
            children,
            initial_cash=config.execution.initial_cash,
            commission_bps=config.execution.commission_bps,
            slippage_bps=config.execution.slippage_bps,
        )
        for row in child_rows:
            row["beam_generation"] = generation
        rows.extend(child_rows)
        beam = _select_beam([*beam, *child_rows], args.beam_width)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"survival_spy_only_beam_stage_{args.stage}.csv"
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
                "best_train_calmar": max((float(row.get("train_calmar", 0.0) or 0.0) for row in rows), default=0.0),
                "output_path": str(output_path),
                "locked_opened": False,
            },
            indent=2,
        )
    )
    return 0


def _select_beam(rows: list[dict[str, object]], width: int) -> list[dict[str, object]]:
    sorted_rows = sorted(rows, key=_beam_score, reverse=True)
    selected = []
    seen_signatures: set[str] = set()
    for row in sorted_rows:
        signature = _feature_signature(row)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        selected.append(row)
        if len(selected) >= width:
            break
    return selected


def _beam_score(row: dict[str, object]) -> float:
    soft_bonus = 20.0 if bool(row.get("soft_pass")) else 0.0
    strict_bonus = 50.0 if bool(row.get("accepted")) else 0.0
    return _train_first_score(row) + soft_bonus + strict_bonus


def _feature_signature(row: dict[str, object]) -> str:
    names = sorted({spec.split("|", 1)[0] for spec in _row_specs(row)})
    return ";".join(names[:5])


def _params_from_row(row: dict[str, object]) -> dict[str, object]:
    rule = str(row["rule"])
    if rule == "spy_long_short_score":
        return {
            "rule": rule,
            "feature_specs": str(row["feature_specs"]),
            "score_threshold": float(row.get("score_threshold", 0.0) or 0.0),
            "confirm_days": int(row.get("confirm_days", 1) or 1),
            "min_hold_days": int(row.get("min_hold_days", 1) or 1),
            "combo_size": int(row.get("combo_size", len(_row_specs(row))) or len(_row_specs(row))),
        }
    return {
        "rule": "spy_long_short_always",
        "long_specs": str(row["long_specs"]),
        "short_specs": str(row["short_specs"]),
        "long_min_votes": int(row.get("long_min_votes", 1) or 1),
        "short_min_votes": int(row.get("short_min_votes", 1) or 1),
        "confirm_days": int(row.get("confirm_days", 1) or 1),
        "min_hold_days": int(row.get("min_hold_days", 1) or 1),
        "combo_size": int(row.get("combo_size", len(_row_specs(row))) or len(_row_specs(row))),
    }


def _mutate(
    parent: dict[str, object],
    catalog: list[dict[str, object]],
    spec_lookup: dict[str, dict[str, object]],
    rng: np.random.Generator,
) -> dict[str, object]:
    specs = _params_specs(parent)
    action = str(rng.choice(["replace", "replace", "add", "remove", "memory", "threshold", "rule"]))
    if action == "replace" and specs:
        specs[int(rng.integers(0, len(specs)))] = _nearby_spec(specs[int(rng.integers(0, len(specs)))], catalog, spec_lookup, rng)
    elif action == "add" and len(specs) < 5:
        specs.append(str(catalog[int(rng.integers(0, len(catalog)))]["spec"]))
    elif action == "remove" and len(specs) > 2:
        del specs[int(rng.integers(0, len(specs)))]
    specs = sorted(set(specs))
    if len(specs) < 2:
        specs.append(str(catalog[int(rng.integers(0, len(catalog)))]["spec"]))
    rule = str(parent.get("rule", "spy_long_short_always"))
    if action == "rule":
        rule = "spy_long_short_score" if rule == "spy_long_short_always" else "spy_long_short_always"
    confirm_days = _mutate_choice(int(parent.get("confirm_days", 1) or 1), [1, 2, 3, 5, 8], rng)
    min_hold_days = _mutate_choice(int(parent.get("min_hold_days", 1) or 1), [1, 3, 5, 10, 15, 20], rng)
    if rule == "spy_long_short_score":
        threshold = float(parent.get("score_threshold", 0.0) or 0.0)
        if action == "threshold":
            threshold = float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0, 1.25]))
        return {
            "rule": rule,
            "feature_specs": ";".join(specs[:5]),
            "score_threshold": threshold,
            "confirm_days": confirm_days,
            "min_hold_days": min_hold_days,
            "combo_size": min(5, len(specs)),
        }
    rng.shuffle(specs)
    long_count = max(1, len(specs) // 2)
    long_specs = sorted(specs[:long_count])
    short_specs = sorted(specs[long_count:])
    if not short_specs:
        short_specs = [long_specs.pop()]
    return {
        "rule": "spy_long_short_always",
        "long_specs": ";".join(long_specs),
        "short_specs": ";".join(short_specs),
        "long_min_votes": min(int(parent.get("long_min_votes", 1) or 1), len(long_specs)),
        "short_min_votes": min(int(parent.get("short_min_votes", 1) or 1), len(short_specs)),
        "confirm_days": confirm_days,
        "min_hold_days": min_hold_days,
        "combo_size": len(long_specs) + len(short_specs),
    }


def _params_specs(params: dict[str, object]) -> list[str]:
    specs: list[str] = []
    for key in ("long_specs", "short_specs", "feature_specs"):
        value = str(params.get(key, ""))
        specs.extend([part for part in value.split(";") if part and part.lower() != "nan"])
    return specs


def _nearby_spec(
    spec: str,
    catalog: list[dict[str, object]],
    spec_lookup: dict[str, dict[str, object]],
    rng: np.random.Generator,
) -> str:
    item = spec_lookup.get(spec)
    if item is None or rng.random() < 0.25:
        return str(catalog[int(rng.integers(0, len(catalog)))]["spec"])
    same_family = [candidate for candidate in catalog if candidate["family"] == item["family"]]
    same_name = [candidate for candidate in same_family if candidate["name"] == item["name"]]
    pool = same_name if same_name and rng.random() < 0.70 else same_family
    return str(pool[int(rng.integers(0, len(pool)))]["spec"])


def _mutate_choice(current: int, choices: list[int], rng: np.random.Generator) -> int:
    if rng.random() < 0.50 and current in choices:
        index = choices.index(current)
        index = max(0, min(len(choices) - 1, index + int(rng.choice([-1, 1]))))
        return choices[index]
    return int(rng.choice(choices))


if __name__ == "__main__":
    raise SystemExit(main())
