# Verdicts on the Codex review of chapter 13

Each finding in `codex-review-l13.md` was checked against `lecture-13-agent-safety.md` and `agent_lab/safety.py` (as published in commit `06f7221`). "Confirmed" means the defect is real. The severity column gives the checker's severity, which can differ from the one Codex gave. The chapter text and code were not changed before this check.

| # | Line | Verdict | Severity (Codex → checked) | Check |
|---|---|---|---|---|
| 1 | 32 | Confirmed | blocking → minor | Adaptation usually raises the per-attempt rate, but correlated attempts can lower the chance of at least one success (perfect correlation gives 0.001 against 0.993). Fix: "which can raise their odds above this estimate; correlated attempts can also lower them." |
| 2 | 97 | Confirmed | blocking → blocking | Line 97 says the model credential "must be inside", and line 99 describes OpenShell, which keeps the agent inside and the credentials outside. Fix: "With the whole agent inside and no credential broker, the model credential must be inside too". |
| 3 | 195 | Confirmed | minor → minor | The precision identity (line 183) is Bayes' rule for one decision and needs no independence between decisions. Line 245 already states "fixed rates" only. Fix: "The calculation assumes fixed rates." |
| 4 | `agent_lab/safety.py` 17–30 | Confirmed | blocking → minor | Reproduced: with `trusts={"triage": iter(["release"])}` the function returns only the triage effects, because the validation at line 18 consumes the iterator. The chapter's own calls use lists, so no published result is wrong. Fix: copy each adjacency to a tuple before validation, and add a test with iterator adjacency. |

Result: 4 of 4 findings confirmed. Checked severity: 1 blocking (2), 3 minor.

Status, 17 October 2026: every fix above was applied in the commit that adds this line (see `git log -- review/codex-verdicts-l13.md`).
