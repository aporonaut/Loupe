# Copyright 2025 Aaron AlAnsari (Aporonaut)
# SPDX-License-Identifier: Apache-2.0

"""Aggregate scoring — weighted arithmetic mean with contribution tracking."""

from __future__ import annotations

from loupe.core.models import AnalyzerResult, ScoringMetadata

MIN_DIMENSIONS_FOR_RELIABLE = 2


def compute_aggregate(
    results: list[AnalyzerResult],
    weights: dict[str, float],
) -> tuple[float, ScoringMetadata]:
    """Compute the aggregate score from per-dimension analyzer results.

    Uses Weighted Arithmetic Mean: ``aggregate = sum(w_i * s_i) / sum(w_i)``
    for available dimensions. Weights are normalized internally.

    Parameters
    ----------
    results : list[AnalyzerResult]
        Results from individual analyzers.
    weights : dict[str, float]
        Relative weights per dimension name. Dimensions not present in
        results are excluded (proportional aggregation).

    Returns
    -------
    tuple[float, ScoringMetadata]
        The aggregate score and metadata about the computation.
    """
    if not results:
        return 0.0, ScoringMetadata(
            weights={},
            contributions={},
            reliable=False,
        )

    # Filter to dimensions that have both a result and a weight
    active: list[tuple[str, float, float]] = []
    for r in results:
        w = weights.get(r.analyzer, 0.0)
        if w > 0:
            active.append((r.analyzer, r.score, w))

    if not active:
        return 0.0, ScoringMetadata(
            weights={},
            contributions={},
            reliable=False,
        )

    total_weight = sum(w for _, _, w in active)
    normalized_weights: dict[str, float] = {}
    contributions: dict[str, float] = {}

    weighted_sum = 0.0
    for name, score, w in active:
        nw = w / total_weight
        normalized_weights[name] = round(nw, 6)
        contribution = nw * score
        contributions[name] = round(contribution, 6)
        weighted_sum += contribution

    aggregate = round(weighted_sum, 6)
    reliable = len(active) >= MIN_DIMENSIONS_FOR_RELIABLE

    return aggregate, ScoringMetadata(
        weights=normalized_weights,
        contributions=contributions,
        reliable=reliable,
    )
