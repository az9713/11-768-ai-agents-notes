# Verdicts on the Codex review of chapter 12

Each finding in `codex-review-l12.md` was checked against `lecture-12-rl-systems.md` (as published in commit `06f7221`). "Confirmed" means the defect is real. The severity column gives the checker's severity, which can differ from the one Codex gave. The chapter text was not changed before this check.

| # | Line | Verdict | Severity (Codex → checked) | Check |
|---|---|---|---|---|
| 1 | 39, 195 | Confirmed | blocking → blocking | The share of matching token conditionals does not bound the trajectory bias. One early mismatch changes the law of every later prefix. The bias is bounded by the trajectory total-variation distance, which is at most the sum of the per-position distances. Fix: "the bias is small when the per-token distributions are close at every position", and at line 195 "grows with the mismatch at the mixed-weight tokens". |
| 2 | 45 | Confirmed | blocking → blocking | For $\hat A<0$ and $r>1+\epsilon$, the PPO minimum selects the unclipped term $r\hat A$. With $w=2$, $r=100$, $\hat A=-1$, the coefficient is $-200$. The cap limits the mismatch weight only. Fix: "No mismatch weight exceeds $w_{\max}$", and note that a negative-advantage token with a large ratio is still unbounded (the reason for dual clipping). |
| 3 | 91–93, 230 | Confirmed | minor → minor | When $\pi=q_{\rm d}$, the rejection term is $0\cdot0/0$ and the token-count formula is $0/0$. The limit is $n_{\rm draft}+1=5$. Fix: one clause for the case $\alpha_{\rm acc}=1$. |
| 4 | 123 | Confirmed | minor → minor | $\Delta$ does not occur before line 123. Fix: "maps $v+\Delta$, for a weight increment $\Delta$, back to …". |
| 5 | 125 | Confirmed | blocking → blocking | Recomputed: bias-corrected Adam ($\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$) with $g_t=1.01^t$ gives $\lvert\Delta_{2000}\rvert/\eta\approx3.883$, with no zero gradients. The Adam paper gives $\lvert\Delta_t\rvert\lesssim\eta$ only as the usual case, and $\eta(1-\beta_1)/\sqrt{1-\beta_2}\approx3.16\eta$ for the sparse case. Fix: "is usually at most the learning rate". |
| 6 | 213 | Confirmed | minor → minor | Line 27 defines $P_{\rm peak}$ for one GPU. Fix: "divides … by the summed peak rate of the GPUs used". |
| 7 | 239 | Confirmed | blocking → blocking | 61,000 tokens is where attention per position equals the linear cost per position in one sequence. It is not the crossover between harnesses A and B. Recomputed with line 231's model: $C(20{,}000)=4.249\times10^{14}$ is more than $C(11{,}000)+C(11{,}300)=4.220\times10^{14}$. Fix: state that 61,000 tokens is the length where attention becomes the larger term, and that the A-against-B crossover depends on the lengths. |

Result: 7 of 7 findings confirmed. Checked severity: 4 blocking (1, 2, 5, 7), 3 minor.

Status, 17 October 2026: every fix above was applied in the commit that adds this line (see `git log -- review/codex-verdicts-l12.md`).
