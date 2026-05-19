from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from run_survival_spy_only_shootout_stage import main as shootout_main  # noqa: E402
from trading_lab.config import load_optimization_config  # noqa: E402


DEFAULT_BATCH_BUDGET = {
    "adaptive": 900,
    "beam": 900,
    "bayesian": 720,
    "bandit": 900,
    "genetic": 540,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one time-boxed SPY-only method marathon chunk.")
    parser.add_argument("--method", choices=["adaptive", "beam", "bayesian", "bandit", "genetic"], required=True)
    parser.add_argument("--config", default="configs/survival_spy_only_github.yaml")
    parser.add_argument("--lane", type=int, required=True)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--minutes", type=float, default=170.0)
    parser.add_argument("--batch-budget", type=int, default=0)
    parser.add_argument("--min-remaining-seconds", type=float, default=180.0)
    args = parser.parse_args()

    config = load_optimization_config(args.config)
    batch_budget = args.batch_budget or DEFAULT_BATCH_BUDGET[args.method]
    deadline = time.monotonic() + args.minutes * 60.0
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    batches_completed = 0
    rows = 0
    started_at = time.time()

    while time.monotonic() < deadline - args.min_remaining_seconds:
        virtual_stage = _virtual_stage(args.method, args.lane, args.chunk, batches_completed)
        before = time.monotonic()
        previous_argv = sys.argv[:]
        sys.argv = [
            "run_survival_spy_only_shootout_stage.py",
            "--method",
            args.method,
            "--config",
            args.config,
            "--stage",
            str(virtual_stage),
            "--total-stages",
            "1000000",
            "--budget",
            str(batch_budget),
        ]
        try:
            shootout_main()
        finally:
            sys.argv = previous_argv

        stage_path = output_dir / f"survival_spy_only_{args.method}_stage_{virtual_stage}.csv"
        if stage_path.exists():
            frame = pd.read_csv(stage_path)
            if not frame.empty:
                frame["marathon_method"] = args.method
                frame["marathon_lane"] = args.lane
                frame["marathon_chunk"] = args.chunk
                frame["marathon_batch"] = batches_completed
                frame["marathon_virtual_stage"] = virtual_stage
                frames.append(frame)
                rows += len(frame)
        batches_completed += 1
        if time.monotonic() - before <= 0.0:
            break

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty and "survival_score" in combined:
        combined = combined.sort_values("survival_score", ascending=False)
    output_path = output_dir / f"survival_spy_only_marathon_{args.method}_lane_{args.lane}_chunk_{args.chunk}.csv"
    combined.to_csv(output_path, index=False)

    print(
        json.dumps(
            {
                "method": args.method,
                "lane": args.lane,
                "chunk": args.chunk,
                "minutes_budget": args.minutes,
                "batch_budget": batch_budget,
                "batches_completed": batches_completed,
                "rows": rows,
                "accepted": int(combined["accepted"].sum()) if "accepted" in combined else 0,
                "soft_pass": int(combined["soft_pass"].sum()) if "soft_pass" in combined else 0,
                "output_path": str(output_path),
                "elapsed_seconds": round(time.time() - started_at, 3),
                "traded_asset": "SPY",
                "cash_allowed": False,
                "always_fully_invested": True,
                "locked_opened": False,
            },
            indent=2,
        )
    )
    return 0


def _virtual_stage(method: str, lane: int, chunk: int, batch: int) -> int:
    method_offset = {
        "adaptive": 100_000,
        "beam": 200_000,
        "bayesian": 300_000,
        "bandit": 400_000,
        "genetic": 500_000,
    }[method]
    return method_offset + lane * 10_000 + chunk * 1_000 + batch


if __name__ == "__main__":
    raise SystemExit(main())
