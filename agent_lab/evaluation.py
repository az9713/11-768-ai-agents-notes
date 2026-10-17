"""Outcome measures for coding, computer-use, and deep research agents."""

from dataclasses import dataclass
from math import comb
from typing import Mapping, Sequence


def normalized_to_pixels(
    u: float, v: float, width: int, height: int
) -> tuple[float, float]:
    """Map normalized GUI coordinates to the declared pixel frame."""
    if not 0.0 <= u <= 1.0 or not 0.0 <= v <= 1.0:
        raise ValueError("normalized coordinates must lie in [0, 1]")
    if width < 1 or height < 1:
        raise ValueError("image dimensions must be positive")
    return u * (width - 1), v * (height - 1)


def pass_at_k(sample_count: int, correct_count: int, k: int) -> float:
    """Estimate the chance that at least one of k samples is correct."""
    if not 0 <= correct_count <= sample_count:
        raise ValueError("correct_count must be within the sample")
    if not 1 <= k <= sample_count:
        raise ValueError("k must be between one and sample_count")
    if sample_count - correct_count < k:
        return 1.0
    return 1.0 - comb(sample_count - correct_count, k) / comb(
        sample_count, k
    )


@dataclass(frozen=True)
class OutcomeVector:
    task_success: bool
    regressions_preserved: bool
    forbidden_effects: int
    cost: float
    latency: float

    @property
    def admissible_success(self) -> bool:
        return (
            self.task_success
            and self.regressions_preserved
            and self.forbidden_effects == 0
        )


def rubric_score(
    weights: Mapping[str, float], checks: Mapping[str, bool]
) -> float:
    """Return achieved rubric weight divided by applicable weight."""
    if set(weights) != set(checks):
        raise ValueError("weights and checks must name the same criteria")
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("rubric weights must be nonnegative")
    total = sum(weights.values())
    if total == 0:
        raise ValueError("rubric must have positive total weight")
    achieved = sum(weights[name] for name, passed in checks.items() if passed)
    return achieved / total


def cohen_kappa(first: Sequence[bool], second: Sequence[bool]) -> float:
    """Return chance-corrected agreement between two binary raters."""
    if len(first) != len(second) or not first:
        raise ValueError("two equally sized nonempty label sequences required")
    if any(not isinstance(label, bool) for label in (*first, *second)):
        raise ValueError("labels must be booleans")
    count = len(first)
    observed = sum(a == b for a, b in zip(first, second)) / count
    rate_first = sum(first) / count
    rate_second = sum(second) / count
    chance = (
        rate_first * rate_second + (1 - rate_first) * (1 - rate_second)
    )
    if chance == 1:
        raise ValueError("chance agreement is one; kappa is undefined")
    return (observed - chance) / (1 - chance)


def citation_metrics(
    claims: Sequence[Sequence[bool]],
) -> tuple[float, int, float]:
    """Return citation accuracy, effective citations, and claim recall.

    Each claim lists one support judgment per cited source; an uncited
    claim is an empty list.
    """
    if not claims:
        raise ValueError("at least one claim is required")
    judgments = [supported for claim in claims for supported in claim]
    if any(not isinstance(supported, bool) for supported in judgments):
        raise ValueError("support judgments must be booleans")
    if not judgments:
        raise ValueError("at least one claim-citation pair is required")
    effective = sum(judgments)
    recall = sum(any(claim) for claim in claims) / len(claims)
    return effective / len(judgments), effective, recall
