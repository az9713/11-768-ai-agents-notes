"""Small numerical definitions for Chapters 8 through 12."""

import heapq
import struct
from math import exp, isfinite, log
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


def gae_advantages(
    rewards: Sequence[float],
    values: Sequence[float],
    gamma: float,
    lam: float,
) -> tuple[float, ...]:
    """Return generalized advantage estimates for every trajectory position.

    values[t] estimates V(h_t); values[-1] is the value after the last
    action, zero when the trajectory terminated.
    """
    if len(values) != len(rewards) + 1:
        raise ValueError("one more value than rewards is required")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= lam <= 1.0:
        raise ValueError("gamma and lam must lie in [0, 1]")
    result = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        delta = rewards[index] + gamma * values[index + 1] - values[index]
        running = delta + gamma * lam * running
        result[index] = running
    return tuple(result)


def ratio_moments(
    behavior: Sequence[float], target: Sequence[float]
) -> tuple[float, float, float]:
    """Return E[log rho], Var[log rho], and E[rho**2] under behavior.

    Both arguments are probability vectors over the same finite action set.
    """
    if len(behavior) != len(target) or not behavior:
        raise ValueError("two equal-length nonempty distributions required")
    if any(p < 0 for p in behavior) or any(p < 0 for p in target):
        raise ValueError("probabilities must be nonnegative")
    if abs(sum(behavior) - 1.0) > 1e-9 or abs(sum(target) - 1.0) > 1e-9:
        raise ValueError("each distribution must sum to one")
    if any(m == 0 and p > 0 for m, p in zip(behavior, target)):
        raise ValueError("target support must lie inside behavior support")
    pairs = [(m, p / m) for m, p in zip(behavior, target) if m > 0]
    if any(ratio == 0 for _, ratio in pairs):
        raise ValueError("log ratio is undefined where the target is zero")
    mean = sum(m * log(ratio) for m, ratio in pairs)
    variance = sum(m * (log(ratio) - mean) ** 2 for m, ratio in pairs)
    second_moment = sum(m * ratio**2 for m, ratio in pairs)
    return mean, variance, second_moment


def ppo_clip_term(
    ratio: float, advantage: float, eps_low: float, eps_high: float
) -> tuple[float, float]:
    """Return the PPO term and its coefficient on grad log pi.

    The term is min(rho A, clip(rho, 1-eps_low, 1+eps_high) A). Its gradient
    is the coefficient times the score of the sampled action.
    """
    if ratio < 0:
        raise ValueError("ratio must be nonnegative")
    if not 0.0 <= eps_low < 1.0 or eps_high < 0:
        raise ValueError("eps_low must lie in [0, 1) and eps_high be >= 0")
    clipped = min(max(ratio, 1.0 - eps_low), 1.0 + eps_high)
    raw_term = ratio * advantage
    clipped_term = clipped * advantage
    if clipped_term < raw_term:
        return clipped_term, 0.0
    return raw_term, raw_term


def cispo_coefficient(
    ratio: float,
    advantage: float,
    eps_high: float,
    eps_low: float | None = None,
) -> float:
    """Return the CISPO coefficient sg(clip(rho)) A on grad log pi.

    eps_low=None imposes no lower bound, the setting MiniMax-M1 reports.
    """
    if ratio < 0 or eps_high < 0:
        raise ValueError("ratio and eps_high must be nonnegative")
    lower = 0.0 if eps_low is None else 1.0 - eps_low
    return min(max(ratio, lower), 1.0 + eps_high) * advantage


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Return D_KL(p || q) in nats for distributions on one finite set."""
    if len(p) != len(q) or not p:
        raise ValueError("two equal-length nonempty distributions required")
    if any(value < 0 for value in (*p, *q)):
        raise ValueError("probabilities must be nonnegative")
    if abs(sum(p) - 1.0) > 1e-9 or abs(sum(q) - 1.0) > 1e-9:
        raise ValueError("each distribution must sum to one")
    if any(a > 0 and b == 0 for a, b in zip(p, q)):
        return float("inf")
    return sum(a * log(a / b) for a, b in zip(p, q) if a > 0)


def batch_makespans(
    durations: Sequence[float], slots: int
) -> tuple[float, float]:
    """Return (static, continuous) completion times for queued requests.

    Static batching admits ``slots`` requests at a time and waits for the
    longest one. Continuous batching gives each freed slot the next queued
    request. Each slot runs one request at a fixed duration.
    """
    if slots < 1:
        raise ValueError("slots must be positive")
    if any(duration < 0 for duration in durations):
        raise ValueError("durations must be nonnegative")
    static = sum(
        max(durations[start:start + slots])
        for start in range(0, len(durations), slots)
    )
    free_at = [0.0] * slots
    finish = 0.0
    for duration in durations:
        start = heapq.heappop(free_at)
        finish = max(finish, start + duration)
        heapq.heappush(free_at, start + duration)
    return float(static), finish


def bf16_round(value: float) -> float:
    """Round to bfloat16 (via float32) with round-to-nearest-even."""
    if not isfinite(value):
        raise ValueError("value must be finite")
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack(">f", struct.pack(">I", bits & 0xFFFF0000))[0]


def greedy_encode(text: str, vocabulary: Sequence[str]) -> tuple[int, ...]:
    """Encode text by longest vocabulary match; return token indices."""
    tokens = []
    position = 0
    while position < len(text):
        matches = [
            index for index, piece in enumerate(vocabulary)
            if piece and text.startswith(piece, position)
        ]
        if not matches:
            raise ValueError("text contains a piece outside the vocabulary")
        best = max(matches, key=lambda index: len(vocabulary[index]))
        tokens.append(best)
        position += len(vocabulary[best])
    return tuple(tokens)


def decode(tokens: Sequence[int], vocabulary: Sequence[str]) -> str:
    """Concatenate the vocabulary pieces named by token indices."""
    return "".join(vocabulary[token] for token in tokens)
