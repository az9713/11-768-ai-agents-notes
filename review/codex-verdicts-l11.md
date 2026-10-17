# Verdicts on the Codex review of chapter 11

Each finding in `codex-review-l11.md` was checked against `lecture-11-rl-advanced.md` (as published in commit `06f7221`). "Confirmed" means the defect is real. The severity column gives the checker's severity, which can differ from the one Codex gave. The chapter text was not changed before this check.

| # | Line | Verdict | Severity (Codex → checked) | Check |
|---|---|---|---|---|
| 1 | 69 | Confirmed | blocking → blocking | At $\lambda=1$, $\hat A_t=G_t-V_\phi(h_t)$, so $\mathbb E[\hat A_t\mid h_t,a_t]=Q^\mu-V_\phi\ne A^\mu$ unless $V_\phi=V^\mu$. The GAE paper's "$\gamma$-just regardless of the accuracy of $V$" is a property of the policy-gradient estimate, not of the advantage estimate. Fix: say the gradient estimate is unbiased. |
| 2 | 69 | Confirmed | blocking → minor | The counterexample is valid ($\operatorname{Var}\delta_0=1/4$ against 0). The GAE paper says $\lambda=0$ "typically" has lower variance. Fix: add "typically". |
| 3 | 117 | Confirmed | blocking → minor | With $g\equiv1$, $\mathbb E_\mu[r_t]=1=\mathbb E_{\pi_\theta}[1]$, so "biased whenever" is too strong. Fix: "can be biased when". |
| 4 | 117 | Confirmed (mathematics) | blocking → minor | The counterexample is valid: the suffix ratio adds no variance where $g=0$. Not checked: whether CTPO's Proposition 2 states the extra condition on $g$. Fix: add the condition "where $g$ is nonzero", or quote the proposition. |
| 5 | 156 | Confirmed | blocking → blocking | Line 115 indexes actions from 0, so at the first action the printed interval is $[e^0,e^0]=[1,1]$ and clips every first-token ratio to 1. Fix: use $\sqrt{t+1}$, or state that CTPO counts positions from 1. |
| 6 | 156 | Confirmed | minor → minor | At $T=1$, $x^{1/T}=x$ is linear and $\mathbb E_\mu[w]=1$ for any $w$. Fix: "for $T>1$, with equality only for constant $w(\tau)$". |
| 7 | 193 | Confirmed | blocking → minor | The sentence reports DAPO's own example correctly, but "can raise … only to 0.012" reads as a hard cap and contradicts line 173. Recomputed: one SGD step (rate 1, $\hat A=1$) from 0.01 gives $p\approx0.0265$, because the clip is inactive at ratio 1. Fix: "the objective stops rewarding increases above 0.012". |
| 8 | 205 | Confirmed | minor → minor | The KL term (line 199) and the entropy bonus (line 203) still give gradients for a group with equal rewards. Fix: "the group gives no reward gradient". |
| 9 | 225 | Confirmed | blocking → minor | A test that rejects some correct solutions lowers the measured score of a model that writes them. It caps the score only if it rejects every correct solution. Fix: "can lower the measured score", or limit the cap claim to tasks that no correct solution passes. |
| 10 | 255 | Confirmed | blocking → blocking | For potential-based shaping with $\Phi(h_T)=0$: $Q'=Q-\Phi$ and $V'=V-\Phi$, so $A'=A$ exactly. Shaping changes the per-step rewards and the critic's target, not the exact advantages. Fix: "Shaping changes the per-step rewards and what the critic must learn, so a finite run can stop elsewhere." |

Result: 10 of 10 findings confirmed. Checked severity: 3 blocking (1, 5, 10), 7 minor.

Status, 17 October 2026: every fix above was applied in the commit that adds this line (see `git log -- review/codex-verdicts-l11.md`).
