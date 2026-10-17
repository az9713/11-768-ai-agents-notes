"""Small numerical definitions for Chapters 8 through 10."""

from math import exp, log
from typing import Sequence


def masked_cross_entropy(
    target_probabilities: Sequence[float],
    mask: Sequence[bool],
    weights: Sequence[float] | None = None,
) -> float:
    """Return weighted mean negative log probability on selected tokens."""
    if len(target_probabilities) != len(mask):
        raise ValueError("probabilities and mask must have equal length")
    if weights is None:
        weights = [1.0] * len(mask)
    if len(weights) != len(mask):
        raise ValueError("weights and mask must have equal length")
    if any(not 0.0 < probability <= 1.0 for probability in target_probabilities):
        raise ValueError("target probabilities must lie in (0, 1]")
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be nonnegative")
    terms = [
        (-log(probability), weight)
        for probability, selected, weight in zip(
            target_probabilities, mask, weights
        )
        if selected
    ]
    if not terms:
        raise ValueError("mask must select at least one token")
    normalizer = sum(weight for _, weight in terms)
    if normalizer == 0:
        raise ValueError("selected weights must have positive sum")
    return sum(loss * weight for loss, weight in terms) / normalizer


def reinforce_estimate(
    score_gradients: Sequence[Sequence[float]],
    rewards: Sequence[float],
    baselines: Sequence[float] | None = None,
) -> tuple[float, ...]:
    """Estimate E[(R-b) grad log p] from sampled trajectories."""
    if len(score_gradients) != len(rewards) or not rewards:
        raise ValueError("one nonempty gradient is required per reward")
    if baselines is None:
        baselines = [0.0] * len(rewards)
    if len(baselines) != len(rewards):
        raise ValueError("one baseline is required per reward")
    width = len(score_gradients[0])
    if any(len(gradient) != width for gradient in score_gradients):
        raise ValueError("all gradients must have equal width")
    return tuple(
        sum(
            (reward - baseline) * gradient[j]
            for gradient, reward, baseline in zip(
                score_gradients, rewards, baselines
            )
        ) / len(rewards)
        for j in range(width)
    )


def centered_group_advantages(rewards: Sequence[float]) -> tuple[float, ...]:
    """Center group rewards without standard-deviation normalization."""
    if not rewards:
        raise ValueError("at least one reward is required")
    mean = sum(rewards) / len(rewards)
    return tuple(reward - mean for reward in rewards)


def effective_token_shares(
    sample_probabilities: Sequence[float],
    selected_tokens: Sequence[float],
    mean_weights: Sequence[float],
) -> tuple[float, ...]:
    """Return normalized expected weighted-token contributions."""
    if not (
        len(sample_probabilities) == len(selected_tokens) == len(mean_weights)
    ) or not sample_probabilities:
        raise ValueError("three equally sized nonempty sequences are required")
    if any(value < 0 for values in (
        sample_probabilities, selected_tokens, mean_weights
    ) for value in values):
        raise ValueError("mixture inputs must be nonnegative")
    masses = [s * tokens * weight for s, tokens, weight in zip(
        sample_probabilities, selected_tokens, mean_weights
    )]
    total = sum(masses)
    if total == 0:
        raise ValueError("weighted-token mass must be positive")
    return tuple(mass / total for mass in masses)


def reward_to_go(
    rewards: Sequence[float], gamma: float
) -> tuple[float, ...]:
    """Return discounted reward-to-go for every trajectory position."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1]")
    result = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + gamma * running
        result[index] = running
    return tuple(result)


def degenerate_binary_group_probability(
    success_probability: float, group_size: int
) -> float:
    """Return P(all success or all failure) for an independent group."""
    if not 0.0 <= success_probability <= 1.0:
        raise ValueError("success probability must lie in [0, 1]")
    if group_size < 1:
        raise ValueError("group size must be positive")
    return success_probability**group_size + (1-success_probability)**group_size


def contrastive_loss(scores: Sequence[float], positive_index: int) -> float:
    """Return -log softmax(scores)[positive_index] over the candidates."""
    if not scores:
        raise ValueError("at least one candidate score is required")
    if not 0 <= positive_index < len(scores):
        raise ValueError("positive_index must name a candidate")
    largest = max(scores)
    normalizer = largest + log(sum(exp(score - largest) for score in scores))
    return normalizer - scores[positive_index]
