# Codex review: lecture-13-agent-safety.md

| # | Line | Quoted text | Problem | Evidence | Severity |
|---|---|---|---|---|---|
| 1 | 32 | “Adaptive attackers learn from failures, which raises their odds above this estimate.” | Adaptation does not guarantee success probability above the independent-attempt estimate. Dependence can reduce the probability of at least one success. | Counterexample: every attempt succeeds only when the same hidden vulnerability exists; changing prompts after failures need not help. Computation: perfectly correlated successes give \(P(\text{any success})=0.001\), whereas \(1-(1-0.001)^{5000}=0.993279\). Neither ../lecture-work/lecture-13/lecture-13.txt at 03:33–03:51 nor the stated assumptions establish the claimed increase. | blocking |
| 2 | 97 | “the model credential must be inside too” | This presents credential placement as a necessity, then contradicts it with the brokered architecture in the next paragraph. Model access does not require exposing the upstream credential inside the sandbox. | Line 99 explicitly describes an agent inside with credentials outside. ../lecture-work/lecture-13/lecture-13-citations.md, entry 18, confirms that OpenShell injects real credentials on the trusted side. Distinguish access through a broker from possession of the upstream credential. | blocking |
| 3 | 195, 245 | “The calculation assumes independent decisions and fixed rates.” | Independence between decisions is unnecessary for the precision identity. This confuses a population conditional probability with assumptions needed for sampling uncertainty or concentration of observed counts. | The equation at line 183 follows from Bayes’ rule for one decision. Correlation between successive decisions does not invalidate it. lecture-04-memory-and-skills.md, line 151 and its following equation, derives the same identity without an independence assumption. | minor |
| 4 | agent_lab/safety.py:8, 17–30; chapter 114–121 | “trusts: Mapping[str, Iterable[str]]” | `reachable_effects` silently omits reachable runs when adjacency values are iterators or generators, although its signature accepts these inputs. Validation consumes them before traversal. | Reproduced without file writes: line 18 exhausts `iter(["release"])`; line 30 then finds no successor. With triage effects `{"cache.write"}` and release effects `{"registry.write"}`, list adjacency returns both effects, but iterator adjacency returns only `{"cache.write"}`. tests/test_agent_lab.py uses reusable lists and misses this case. | blocking |

Checked without findings:

Section 0: incident facts, illustrative numerical probabilities, and independent-attempt formula.
Section 1: threat-model examples, authorization cross-references, admissible success, and escalation-score notation.
Section 2: competition counts, ASR arithmetic, PAIR description, and cited training-recipe details.
Section 3: isolation-property assumptions and evaluation-contamination examples.
Section 4: reach definition, induction argument under its stated assumptions, and shared-cache example.
Section 5: least-grant union and minimality argument, account separation, and token-expiry numbers.
Section 6: monitoring recall/precision table, AI Control figures, Bayes derivation, and alert arithmetic.
Section 7: outcome denominators, zero-failure bound, benchmark numbers against the citation check, and all three exercise solutions.
Code verification: all three chapter Python examples and all 15 existing safety tests passed.

Findings: 3 blocking, 1 minor, 0 uncertain.