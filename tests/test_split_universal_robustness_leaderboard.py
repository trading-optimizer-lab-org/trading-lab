from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from split_universal_robustness_leaderboard import split_leaderboard  # noqa: E402


def test_split_leaderboard_can_dedupe_and_reports_n_trials(tmp_path: Path) -> None:
    source = tmp_path / "verified.csv"
    out = tmp_path / "chunks"
    pd.DataFrame(
        {
            "candidate_id": ["a", "b", "a", "c"],
            "method": ["beam", "github_ml", "beam", "genetic"],
            "score": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_csv(source, index=False)

    summary = split_leaderboard(
        source,
        out,
        chunks=2,
        file_prefix="chunk",
        dedupe_key="candidate_id",
    )

    assert summary["input_rows"] == 4
    assert summary["rows"] == 3
    assert summary["duplicates_removed"] == 1
    assert summary["n_trials"] == 3

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert [item["rows"] for item in manifest] == [2, 1]
    assert all(item["n_trials"] == 3 for item in manifest)

    chunk = pd.read_csv(out / "chunk_00.csv")
    assert list(chunk["candidate_id"]) == ["a", "b"]

    written_summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert written_summary == summary


def test_split_leaderboard_preserves_duplicates_by_default(tmp_path: Path) -> None:
    source = tmp_path / "verified.csv"
    out = tmp_path / "chunks"
    pd.DataFrame({"candidate_id": ["a", "a", "b"]}).to_csv(source, index=False)

    summary = split_leaderboard(source, out, chunks=2)

    assert summary["rows"] == 3
    assert summary["duplicates_removed"] == 0
