"""Safety measures for Chapter 13: cross-run reach, alerts, and outcome rates."""

from typing import Iterable, Mapping


def reachable_effects(
    effects: Mapping[str, frozenset[str]],
    trusts: Mapping[str, Iterable[str]],
    start: str,
) -> frozenset[str]:
    """Return every effect reachable from ``start`` through shared state.

    ``trusts[j]`` names the runs that execute or trust state run ``j`` can
    write. The result is the union of effect sets over all runs reachable
    from ``start``, including ``start`` itself.
    """
    trusts = {run: tuple(later) for run, later in trusts.items()}
    named = {start} | set(trusts)
    named |= {later for runs in trusts.values() for later in runs}
    if named - set(effects):
        raise ValueError("every run needs an effect set")
    reach: set[str] = set()
    seen: set[str] = set()
    pending = [start]
    while pending:
        run = pending.pop()
        if run in seen:
            continue
        seen.add(run)
        reach |= effects[run]
        pending.extend(trusts.get(run, ()))
    return frozenset(reach)


def alert_precision(
    base_rate: float, sensitivity: float, false_positive_rate: float
) -> float:
    """Return the fraction of alerts that flag a truly unsafe decision."""
    for value in (base_rate, sensitivity, false_positive_rate):
        if not 0.0 <= value <= 1.0:
            raise ValueError("rates must lie in [0, 1]")
    true_alerts = base_rate * sensitivity
    false_alerts = (1 - base_rate) * false_positive_rate
    if true_alerts + false_alerts == 0:
        raise ValueError("the monitor raises no alerts")
    return true_alerts / (true_alerts + false_alerts)


def safety_outcomes(
    valid_runs: int,
    completed: int,
    refused: int,
    safety_runs: int,
    attempts: int,
    harms: int,
) -> tuple[float, float, float, float]:
    """Return completion, refusal, unsafe-attempt, and completed-harm rates.

    Completion and refusal divide by valid-task runs; attempts and harms
    divide by safety-test runs. A harmful effect counts as an attempt too.
    """
    if valid_runs < 1 or safety_runs < 1:
        raise ValueError("both run counts must be positive")
    if min(completed, refused, attempts, harms) < 0:
        raise ValueError("counts must be nonnegative")
    if completed + refused > valid_runs:
        raise ValueError("completed and refused runs exceed valid runs")
    if not harms <= attempts <= safety_runs:
        raise ValueError("need harms <= attempts <= safety runs")
    return (
        completed / valid_runs,
        refused / valid_runs,
        attempts / safety_runs,
        harms / safety_runs,
    )


def zero_failure_upper_bound(runs: int, confidence: float = 0.95) -> float:
    """Return the one-sided upper bound on a failure rate after zero failures.

    The runs are assumed independent with a common failure probability.
    """
    if runs < 1:
        raise ValueError("at least one run is required")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    return 1 - (1 - confidence) ** (1 / runs)
