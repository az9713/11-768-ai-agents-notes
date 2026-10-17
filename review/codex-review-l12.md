# Codex review: lecture-12-rl-systems.md

| # | Line | Quoted text | Problem | Evidence | Severity |
|---|---|---|---|---|---|
| 1 | 39, 195 | “the bias is small when … agree on most tokens”; “grows with the share of mixed-weight tokens” | The fraction of matching token conditionals neither bounds trajectory bias nor determines its growth. One early discrepancy can change the distribution of every later history. | Counterexample: among 1,000 positions, only the first conditional differs, with probabilities 0.99 versus 0.01 for event A. The final token is Bernoulli(0.5) under both policies; reward requires A and final-token success. For the final token’s logit, the token-weighted gradient bias is `(0.99−0.01)×0.5×(1−0.5)=0.245`, despite 999 matching conditionals. | blocking |
| 2 | 45 | “No weight exceeds $w_{\max}$, so no token dominates a batch.” | Capping the mismatch multiplier does not cap a token’s complete gradient contribution. The advantage, score gradient, and unclipped PPO ratio remain relevant. | In line 43, take $\hat A=-1$, $r=100$, $\epsilon=0.2$, and capped mismatch weight 2. PPO selects the unclipped negative-advantage term; the coefficient on $\nabla\log\pi_\theta$ is `2×100×(−1)=−200`. This is also reproduced by `ppo_clip_term` in `agent_lab/training.py`. | blocking |
| 3 | 91–93, 230 | “The output law is exactly $\pi$”; “$(1-\alpha_{\rm acc}^{n_{\rm draft}+1})/(1-\alpha_{\rm acc})$” | The displayed formulas omit the identical-distribution case. When $\pi=q_{\rm d}$, the rejection distribution has zero normalizer and the expected-token formula is $0/0$. The algorithm remains correct, but these expressions require an explicit boundary case. | For $\pi=q_{\rm d}$, rejection probability is zero, so the rejection branch is never evaluated. The token-count limit is $\lim_{\alpha\to1}(1-\alpha^{n+1})/(1-\alpha)=n+1=5$ for four draft tokens. | minor |
| 4 | 123 | “maps $v+\Delta$ back to a bf16 value $v$” | $\Delta$ is used without defining it as the proposed weight increment. | Section 4 defines $v$ but never introduces $\Delta$; the supplied symbol ledger also omits it. | minor |
| 5 | 125 | “effective step is bounded by the learning rate, except after long runs of zero gradient” | This is not a general bound for Adam. Its coordinate update can exceed the learning rate with every gradient strictly positive. | Recomputing bias-corrected Adam with $\beta_1=0.9$, $\beta_2=0.999$, optimizer epsilon $10^{-8}$, zero initial moments, and $g_t=1.01^t$ gives $\lvert\Delta_{2000}\rvert/\eta=\hat m_{2000}/(\sqrt{\hat v_{2000}}+10^{-8})\approx3.883>1$. There are no zero gradients. | blocking |
| 6 | 213 | “MFU … divides the arithmetic the model needs per second by $P_{\rm peak}$” | The definition omits the GPU-count normalization. Line 27 defines $P_{\rm peak}$ for one GPU; aggregate model throughput requires aggregate trainer peak throughput. Alternatively, the numerator must explicitly be per GPU. | With eight trainer GPUs each operating at 50% utilization, the stated aggregate calculation gives `8×0.5P_peak/P_peak=4`, whereas MFU is `4P_peak/(8P_peak)=0.5`. | minor |
| 7 | 239 | “the comparison reverses only above about 61,000 tokens” | The 61,000-token calculation compares attention with linear computation within one sequence. It does not locate the cost crossover between harnesses A and B. | Using line 231’s model, define $C(L)=1.6\times10^{10}L+524{,}288L(L+1)/2$. Keep the task, actions, and summary fixed, but increase each observation to 4,300 tokens: A has length 20,000 and B has lengths 11,000 and 11,300. Then `C(20,000)=4.24863×10^14 > C(11,000)+C(11,300)=4.21998×10^14` operations. B is already cheaper below 61,000 tokens. | blocking |

Checked without findings:

- Section 0: repair-run arithmetic, policy-gradient estimator, and training stages.
- Section 1: workload arithmetic, format sizes, conditional importance identity, and truncation-bias calculation.
- Section 2: KV memory, training-state accounting, prefix-cache counts, scheduling bounds, and nondegenerate speculative-decoding examples.
- Section 3: checkpointing overhead, prefix-sharing arithmetic, parallelism definitions, and pipeline-bubble calculation.
- Section 4: worked bf16 rounding examples, sparse-sync arithmetic, and routing-replay figures.
- Section 5: sequence extension, compaction counts, chat-template claims, and tokenization example.
- Section 6: utilization example, GPU-allocation equation, staleness definitions, and pause modes.
- Section 7: service architecture, LoRA parameter count, and cited benchmark numbers.
- Exercises: solutions 1, 4, and 5; exercise 3’s original pair counts and operation totals.
- Companion verification: seven Chapter 12 tests and three training-code snippets passed in memory.

Not checked: upstream full texts or GPU execution; empirical-source verification relies on the supplied citation check.

Findings: 4 blocking, 3 minor, 0 uncertain.