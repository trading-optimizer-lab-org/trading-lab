from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from aurora.validation.robustness_config import UniversalRobustnessConfig


def assign_duplicate_groups(
    results: pd.DataFrame,
    returns: pd.DataFrame,
    config: UniversalRobustnessConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mark exact and near duplicate strategies.

    Exact duplicates share a rounded-return fingerprint. Near duplicates are
    linked when return correlation is above the configured threshold.
    """

    if results.empty or returns.empty or "candidate_id" not in returns.columns:
        enriched = results.copy()
        enriched["duplicate_group_id"] = ""
        enriched["duplicate_group_size"] = 1
        enriched["duplicate_representative"] = True
        return enriched, pd.DataFrame()

    candidate_ids = [str(value) for value in results["candidate_id"].astype(str).unique()]
    group_parent = {candidate_id: candidate_id for candidate_id in candidate_ids}

    def find(candidate_id: str) -> str:
        parent = group_parent[candidate_id]
        while parent != group_parent[parent]:
            parent = group_parent[parent]
        group_parent[candidate_id] = parent
        return parent

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            group_parent[root_right] = root_left

    pivot = (
        returns.assign(
            candidate_id=returns["candidate_id"].astype(str),
            timestamp=pd.to_datetime(returns["timestamp"], errors="coerce"),
            strategy_return=pd.to_numeric(returns["strategy_return"], errors="coerce"),
        )
        .pivot_table(index="timestamp", columns="candidate_id", values="strategy_return", aggfunc="first")
        .sort_index()
    )
    pivot = pivot.reindex(columns=candidate_ids)

    fingerprints: dict[str, str] = {}
    for candidate_id in candidate_ids:
        series = pivot[candidate_id].fillna(0.0).to_numpy(dtype=float)
        rounded = np.round(series, int(config.duplicate_round_decimals))
        fingerprints[candidate_id] = hashlib.sha256(rounded.tobytes()).hexdigest()[:16]

    by_fingerprint: dict[str, list[str]] = {}
    for candidate_id, fingerprint in fingerprints.items():
        by_fingerprint.setdefault(fingerprint, []).append(candidate_id)
    for members in by_fingerprint.values():
        first = members[0]
        for candidate_id in members[1:]:
            union(first, candidate_id)

    if len(candidate_ids) > 1:
        corr = pivot.corr(min_periods=max(10, int(len(pivot) * 0.5))).fillna(0.0)
        threshold = float(config.near_duplicate_corr_threshold)
        for i, left in enumerate(candidate_ids):
            for right in candidate_ids[i + 1 :]:
                if abs(float(corr.loc[left, right])) >= threshold:
                    union(left, right)

    if "exposure" in returns.columns and len(candidate_ids) > 1:
        exposure_pivot = (
            returns.assign(
                candidate_id=returns["candidate_id"].astype(str),
                timestamp=pd.to_datetime(returns["timestamp"], errors="coerce"),
                exposure=pd.to_numeric(returns["exposure"], errors="coerce"),
            )
            .pivot_table(index="timestamp", columns="candidate_id", values="exposure", aggfunc="first")
            .sort_index()
            .reindex(columns=candidate_ids)
        )
        active = exposure_pivot.fillna(0.0).abs() > 1e-12
        threshold = float(config.position_jaccard_threshold)
        for i, left in enumerate(candidate_ids):
            for right in candidate_ids[i + 1 :]:
                union_count = int((active[left] | active[right]).sum())
                if union_count == 0:
                    continue
                intersection = int((active[left] & active[right]).sum())
                if intersection / union_count >= threshold:
                    union(left, right)

    roots = {candidate_id: find(candidate_id) for candidate_id in candidate_ids}
    root_to_group_id: dict[str, str] = {}
    for index, root in enumerate(sorted(set(roots.values())), start=1):
        root_to_group_id[root] = f"dup_{index:04d}"
    candidate_to_group = {candidate_id: root_to_group_id[root] for candidate_id, root in roots.items()}

    enriched = results.copy()
    enriched["candidate_id"] = enriched["candidate_id"].astype(str)
    enriched["duplicate_group_id"] = enriched["candidate_id"].map(candidate_to_group).fillna("")
    group_sizes = enriched.groupby("duplicate_group_id")["candidate_id"].transform("count")
    enriched["duplicate_group_size"] = group_sizes.fillna(1).astype(int)
    enriched["duplicate_fingerprint"] = enriched["candidate_id"].map(fingerprints).fillna("")

    representatives: set[str] = set()
    for _, group in enriched.groupby("duplicate_group_id", dropna=False):
        ranked = group.copy()
        if "robust_pass" in ranked.columns:
            ranked["_robust_rank"] = ranked["robust_pass"].fillna(False).astype(bool).astype(int)
        else:
            ranked["_robust_rank"] = 0
        ranked["_metric_rank"] = pd.to_numeric(
            ranked.get(str(config.target_metric), ranked.get("cagr", 0.0)),
            errors="coerce",
        ).fillna(-np.inf)
        ranked = ranked.sort_values(["_robust_rank", "_metric_rank"], ascending=[False, False])
        representatives.add(str(ranked.iloc[0]["candidate_id"]))
    enriched["duplicate_representative"] = enriched["candidate_id"].isin(representatives)
    if "correlation_pass" not in enriched.columns:
        enriched["correlation_pass"] = True
    enriched["portfolio_eligible"] = (
        enriched.get("robust_pass", pd.Series(False, index=enriched.index)).fillna(False).astype(bool)
        & enriched["duplicate_representative"].astype(bool)
        & enriched["correlation_pass"].astype(bool)
    )

    duplicates = enriched[
        [
            "candidate_id",
            "duplicate_group_id",
            "duplicate_group_size",
            "duplicate_representative",
            "duplicate_fingerprint",
        ]
    ].copy()
    return enriched, duplicates
