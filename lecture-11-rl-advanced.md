# Lecture 11 — Advanced reinforcement-learning algorithms for agents

*Independent study chapter · Based on a CMU 11-768: AI Agents lecture by Graham Neubig, 29 September 2026 · [Course and attribution](index.html#sources)*

## 0. A passing test suite does not say which edits helped [00:00](https://www.youtube.com/watch?v=jKx7_fOZRzg&t=0s)

A coding agent must repair `retry_count(config)`, whose body is `return config.get("retries") or 3`. The request reads: "Zero should disable retries, but the code returns three. Preserve the other cases." Zero is falsy in Python, so `0 or 3` evaluates to 3. A **verifier** runs four tests: a missing key and `None` must return 3, and the values 0 and 2 must return themselves. The reward is one when all four pass and zero otherwise. This is chapter 6's falsy-zero defect, posed as a training task because one function supplies a binary reward, a partial reward, a known false negative, and a known false positive.

Two errors have consequences. An update can reinforce a mistaken edit because the run that contained it eventually passed. A flawed verifier can pay for a patch such as `return 0`, and training then makes that shortcut more likely.

The notation is chapter 9's. A trajectory $\tau$ is one attempt of $T$ actions, and history $h_t$ holds the task and everything before action $a_t$. The policy $\pi_\theta(a\mid h)$ has weights $\theta$. Reward $r_t$ follows $a_t$. The slides index it $r_{t+1}$. **Reinforcement learning (RL)** maximizes the expected return $J(\theta)=\mathbb E[R(\tau)]$ by sampling attempts, scoring them, and updating $\theta$. The discount is $\gamma=1$. Chapter 9 required $\gamma<1$ only for an infinite horizon, and every trajectory here is finite.

Trajectory A inspects the function ($a_0$), patches it ($a_1$), sees one test fail ($a_2$), revises ($a_3$), and passes all four ($a_4$). Its first patch was a mistake. Trajectory B makes a useful edit and later breaks the code, so it fails.

Let $n$ be the number of rollouts per task. Chapter 9 writes this group size as $G$. This chapter follows chapter 10 and writes $n$, because $G_\phi$ is chapter 1's execution gate. Rewards $(1,0,0,0)$ have mean $0.25$, so the centered advantages are $(+0.75,-0.25,-0.25,-0.25)$. **Group Relative Policy Optimization (GRPO)** also divides by the group standard deviation: the population value $0.433$ gives about $(1.73,-0.58,-0.58,-0.58)$, and the sample value $0.5$ gives $(1.5,-0.5,-0.5,-0.5)$.

Every action receives its trajectory's coefficient, so A's mistaken patch is pushed up and B's useful edit is pushed down. Chapter 9's reward-to-go $G_t$, the reward from step $t$ on, does not help: with only a terminal reward, $G_t=1$ at every step of A and $0$ on B. Its subscript separates it from the gate $G_\phi$. The chapter asks which turns helped, how stale attempts can be used, when the reward itself is wrong, and how a weak student gets any signal.

## 1. A critic estimates the value of a history [10:25](https://www.youtube.com/watch?v=jKx7_fOZRzg&t=625s)

Let $\mu$ be the **rollout policy**, chapter 9's behavior policy, which generated the data. In fully on-policy training, $\mu=\pi_\theta$. The **value** of history $h$ under $\mu$ is the expected reward-to-go from $h$:

$$V^\mu(h)=\mathbb E_\mu[G_t\mid h_t=h].$$

Chapter 9 defines $V^\pi(s)$ for a state. Here the history takes the state's place. If ten continuations from B's history $h_3$ succeed three times, the Monte Carlo estimate is $0.3$. Ten continuations at each of six histories would cost 60 extra rollouts per trajectory.

A **critic** is a learned approximation $V_\phi(h)$ with parameters $\phi$, as in chapter 9, not chapter 1's harness choices. It is trained by squared error against observed reward-to-go, averaged over sampled histories by $\mathbb E_t$:

$$\mathcal L_V(\phi)=\mathbb E_t\big[(V_\phi(h_t)-G_t)^2\big].$$

### Squared error is minimized by the conditional mean

Condition on $h_t=h$ and write $\bar G=V^\mu(h)$. For any prediction $v$, add and subtract $\bar G$ inside the square:

$$\mathbb E[(v-G_t)^2]=\mathbb E[(G_t-\bar G)^2]+2(v-\bar G)\,\mathbb E[\bar G-G_t]+(v-\bar G)^2=\operatorname{Var}(G_t)+(v-\bar G)^2.$$

The middle term vanishes because $\mathbb E[G_t]=\bar G$. Hence $v=\bar G$ minimizes the loss, and the minimum is the outcome's variance. A critic free to set each history's value reaches $V^\mu$. One with shared parameters reaches the best fit in its class, weighted by how often $\mu$ visits each history.

At $h_3$, the prediction $0.6$ gives $\big(3(0.6-1)^2+7(0.6-0)^2\big)/10=0.30$. The mean $0.3$ gives $0.21$, the Bernoulli variance $0.3\cdot0.7$, and the excess $0.09$ equals $(0.6-0.3)^2$. A high critic loss can therefore reflect noisy outcomes rather than a poor critic. The target is also the value of $\mu$, which lags the policy being updated.

### A separate network or a head on the policy body

A separate value network reads $h$ through its own body. [InstructGPT](https://arxiv.org/abs/2203.02155) initializes a 6-billion-parameter value function from its 6-billion-parameter reward model (Appendix C.4), at the cost of a second model and optimizer. A small **value head** on the policy's hidden representation costs almost nothing. [Mixed Incremental Cross-Entropy Reinforce (MIXER)](https://michaelauli.github.io/papers/iclr2016_mixer.pdf) fits its baseline with a linear regressor on recurrent hidden states by mean squared error. It does not backpropagate that error into the recurrent model, "to prevent feedback loops." This blocking is a **stop-gradient**: the value loss trains only the head.

A third variant appends "Is this trajectory good?" and reads $P(\text{yes})/(P(\text{yes})+P(\text{no}))$ as the value. It rests on the hypothesis that judging an attempt is easier than producing the next action. No study cited here tests it. The critic reads the history, not the sampled action, so chapter 9's baseline proof applies: critic error enters only when critic values form the advantage, as in the next section.

## 2. Temporal-difference errors and generalized advantages locate credit inside a trajectory [16:44](https://www.youtube.com/watch?v=jKx7_fOZRzg&t=1004s)

Chapter 9's **temporal-difference (TD) error** compares values before and after one action:

$$\delta_t=r_t+\gamma V_\phi(h_{t+1})-V_\phi(h_t).$$

On B's last three actions, with zero rewards, let $V_\phi(h_3)=0.3$, $V_\phi(h_4)=0.6$, $V_\phi(h_5)=0.4$, and $V_\phi(h_6)=0$ because B ends after $a_5$. Then $\delta_3=+0.3$, $\delta_4=-0.2$, and $\delta_5=-0.4$: action $a_3$ raised the predicted chance of success, and $a_4$ and $a_5$ lowered it.

**Generalized advantage estimation (GAE)**, also from chapter 9, forms an advantage estimate $\hat A_t$ by weighting later errors geometrically, with trace parameter $\lambda\in[0,1]$:

$$\hat A_t=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l},\qquad\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1},\quad\hat A_T=0.$$

The recursion separates the $l=0$ term. With $\lambda=0.5$, $\hat A_3=0.3+0.5(-0.2)+0.25(-0.4)=0.1$: the step looked good alone, but the errors that followed reduce its credit.

### At λ = 1 the sum telescopes to reward-to-go minus the baseline

Set $\gamma=\lambda=1$ and expand each error:

$$\sum_{l=0}^{T-t-1}\delta_{t+l}=\sum_{l=0}^{T-t-1}r_{t+l}+\sum_{l=0}^{T-t-1}\big(V_\phi(h_{t+l+1})-V_\phi(h_{t+l})\big)=G_t+V_\phi(h_T)-V_\phi(h_t).$$

Each intermediate value appears once with each sign and cancels. A terminated trajectory has $V_\phi(h_T)=0$, so $\hat A_t=G_t-V_\phi(h_t)$. For B, $\hat A_3=0-0.3=-0.3$, the sum of the three errors.

At $\lambda=1$ the critic acts only as a baseline. The policy-gradient estimate is then unbiased whatever the critic's accuracy, but it carries the return's full variance. The advantage estimate itself is unbiased only when $V_\phi=V^\mu$. At $\lambda=0$ it is $\delta_t$, typically with lower variance, and biased unless the critic is exact. The [GAE paper](https://arxiv.org/abs/1506.02438) states both limits.

The choice decides the credit for A's mistaken patch $a_1$. Suppose, as an illustration, that $V_\phi(h_1)=0.5$ before the patch and $V_\phi(h_2)=0.35$ after it. Then $\delta_1=-0.15$, so at $\lambda=0$ the patch is blamed although A succeeded. At $\lambda=1$ it receives $G_1-V_\phi(h_1)=+0.5$. Only a trace below one lets the critic's judgment reach the patch.

The [Proximal Policy Optimization (PPO) paper](https://arxiv.org/abs/1707.06347) used $\lambda=0.95$, and the GAE paper's best control results lie between 0.92 and 0.99. For language-model reasoning, [Open-Reasoner-Zero](https://arxiv.org/abs/2503.24290) trained PPO with $\lambda=\gamma=1$ and reports that $\lambda=0.95$ collapsed response-length growth. One study does not fix a default for agents: a value below one helps only when the critic's bias costs less than the variance it removes.

```python
from agent_lab.training import gae_advantages

values = [0.3, 0.6, 0.4, 0.0]
half = gae_advantages([0.0, 0.0, 0.0], values, gamma=1.0, lam=0.5)
full = gae_advantages([0.0, 0.0, 0.0], values, gamma=1.0, lam=1.0)
assert abs(half[0] - 0.1) < 1e-12
assert abs(full[0] - (0.0 - values[0])) < 1e-12
```

The function assumes a terminated trajectory with final value zero. A truncated rollout needs a bootstrap value there.

### Process reward models label steps directly

A **process reward model (PRM)** scores each step, using labels collected outside the RL run. [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) released PRM800K, 800,000 step labels across 75,000 solutions to 12,000 problems. In its Figure 1, an annotator marks "So x = 7" as incorrect, since the preceding $5x=6x-14$ gives $x=14$. Its process-supervised model solves 78% of a representative MATH test subset, using the PRM to rerank samples, not as an RL reward. Step labels are expensive, and a PRM used as reward is another exploitable oracle (chapter 9). A PRM could also initialize a critic. No source cited here tests that use.

## 3. Asynchronous rollouts create stale data, and importance weights correct it at a cost [23:58](https://www.youtube.com/watch?v=jKx7_fOZRzg&t=1438s)

In **synchronous RL**, every rollout in a batch finishes before the update, so $\mu=\pi_\theta$ exactly. The cost is idle hardware. If three workers each run one rollout and one rollout, the **straggler**, takes four times as long, the workers are busy $(1+1+4)/(3\cdot4)=50\%$ of the generation phase. A twofold straggler gives 67%. The report on [Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO)](https://arxiv.org/abs/2503.14476) notes that such long-tail samples dominate synchronized generation time.

**Asynchronous RL**, whose systems chapter 12 treats, lets rollout workers generate while a learner updates. [AReaL](https://arxiv.org/abs/2505.24298) bounds the **staleness**, the number of policy versions between generation and use, at 4 for code and 8 for math. Zero recovers synchronous training. It reports up to 2.77 times faster training than synchronous systems on the same number of graphics processing units (GPUs), with matched or better scores. Those scores use a decoupled PPO objective that separates the rollout policy from the policy that centers the clip. With standard PPO, AReaL found that even slight staleness degraded final performance.

Stale data mean that the logged $\mu$ differs from the learner's $\pi_\theta$, as for three edits at one history.

| Edit at one history | Tests passed | $\mu$ | $\pi_\theta$ | Ratio $\pi_\theta/\mu$ |
|---|---:|---:|---:|---:|
| Check `None` explicitly | 4 of 4 | 0.40 | 0.70 | 1.75 |
| Use `config.get("retries", 3)` | 3 of 4 | 0.40 | 0.20 | 0.50 |
| Keep the original code | 3 of 4 | 0.20 | 0.10 | 0.50 |

The second edit fails only the `None` test, since `get` returns the stored `None`. The original fails only the zero test.

### Importance sampling reweights by a likelihood ratio

Let $f$ be any function of the action at a fixed history $h$, and suppose $\mu(a\mid h)>0$ wherever $\pi_\theta(a\mid h)>0$. Then

$$\mathbb E_{a\sim\pi_\theta}[f(a)]=\sum_{a:\,\mu(a\mid h)>0}\mu(a\mid h)\,\frac{\pi_\theta(a\mid h)}{\mu(a\mid h)}\,f(a)=\mathbb E_{a\sim\mu}\Big[\frac{\pi_\theta(a\mid h)}{\mu(a\mid h)}f(a)\Big].$$

The support condition makes the division legal. Define the **importance ratio** $r_t(\theta)=\pi_\theta(a_t\mid h_t)/\mu(a_t\mid h_t)$, chapter 9's ratio with history in place of state. The argument $(\theta)$ separates it from the reward $r_t$. With reward one for the explicit check only, $\mathbb E_{\pi_\theta}[R]=0.70=0.40\cdot1.75=\mathbb E_\mu[r_t(\theta)R]$. The single-sample variance is $0.40\cdot1.75^2-0.70^2=0.735$, against $0.70\cdot0.30=0.21$ for a fresh on-policy sample, 3.5 times as large.

Earlier actions also change later histories. Chapter 9's trajectory ratio $w(\tau)=p_\theta(\tau)/p_\mu(\tau)$ reduces to $\prod_tr_t(\theta)$, because the environment factors cancel. Define the **prefix ratio** $w_{\le t}(\theta)=\prod_{k=0}^{t}r_k(\theta)$. Let an earlier action, "open the test file," have illustrative probability 0.40 under $\mu$ and 0.10 under $\pi_\theta$, and the next action, "run tests," 0.50 under both. The next-action ratio is one, yet $\pi_\theta$ reaches that history only $0.10/0.40=0.25$ times as often.

PPO and GRPO use $r_t(\theta)$ alone. Let $g(h_t,a_t)$ be any quantity fixed by the prefix through $a_t$, such as the score times a critic advantage. The identity applied to the whole prefix gives $\mathbb E_{\pi_\theta}[g]=\mathbb E_\mu[w_{\le t}(\theta)\,g]$. The factor $r_t(\theta)$ alone leaves $h_t$ distributed as under $\mu$, so it can be biased when an earlier ratio differs from one, and it is exact at $t=0$. The suffix $\prod_{k>t}r_k(\theta)$ has conditional mean one, because each ratio averages $\sum_a\mu\,(\pi_\theta/\mu)=1$ given its history. For such $g$ it only adds variance. [Cumulative Token Policy Optimization (CTPO)](https://arxiv.org/abs/2605.07331) proves these results as Propositions 1 and 2. The variance reduction is strict only where the suffix ratio varies after a prefix with $g\neq0$.

The hypothesis binds. A trajectory-level coefficient, such as GRPO's group advantage, CTPO's own outcome advantage, or A's terminal reward, depends on actions after $t$ that $\mu$ chose. With the prefix ratio alone it measures the continuation under $\mu$, not $\pi_\theta$, so the suffix is no longer redundant.

### Log weights spread like √T; the weights themselves grow exponentially

Assume the $T$ per-step log ratios are independent under $\mu$, each with mean $-D_{\rm KL}(\mu\|\pi_\theta)$ and variance $\sigma_{\log}^2$. The mean is minus chapter 9's **Kullback-Leibler (KL)** divergence with $\mu$ first, since $\mathbb E_\mu[\log(\pi_\theta/\mu)]=-\sum_a\mu\log(\mu/\pi_\theta)$. Variances of independent terms add:

$$\mathbb E_\mu[\log w(\tau)]=-T\,D_{\rm KL}(\mu\|\pi_\theta),\qquad\operatorname{Var}_\mu[\log w(\tau)]=T\sigma_{\log}^2.$$

The weight behaves differently. Each ratio has $\mathbb E_\mu[r_t(\theta)]=\sum_a\pi_\theta=1$ and $\mathbb E_\mu[r_t(\theta)^2]=\sum_a\pi_\theta^2/\mu=1+D_{\chi^2}(\pi_\theta\|\mu)$, where $D_{\chi^2}(\pi_\theta\|\mu)=\sum_a(\pi_\theta-\mu)^2/\mu$ is the **chi-squared divergence**, unrelated to chapter 10's clue fractions $\chi_\ell$. Independence multiplies moments:

$$\mathbb E_\mu[w(\tau)]=1,\qquad\operatorname{Var}_\mu[w(\tau)]=\big(1+D_{\chi^2}(\pi_\theta\|\mu)\big)^T-1.$$

The log weight's variance grows linearly in $T$, its standard deviation as $\sqrt T$, and the weight's variance exponentially. The source lecture said the variance grows "by the square root". That law holds for the standard deviation of the log. CTPO proves the product form (Proposition 2). On Qwen3-4B tool-integrated math rollouts, its measured standard deviation of the cumulative log ratio follows a fitted $\hat\sigma\sqrt t$ at training steps 50, 100, and 150.

Repeating the table's policies at every step gives $D_{\rm KL}(\mu\|\pi_\theta)\approx0.192$, $\sigma_{\log}\approx0.614$, and $\mathbb E_\mu[r_t(\theta)^2]=1.375$.

| Steps $T$ | Mean of $\log w$ | Std. dev. of $\log w$ | $e^{\text{mean}}$ | $\operatorname{Var}(w)$ |
|---:|---:|---:|---:|---:|
| 1 | −0.19 | 0.61 | 0.83 | 0.375 |
| 10 | −1.92 | 1.94 | 0.15 | 23.2 |
| 100 | −19.2 | 6.14 | $4.6\times10^{-9}$ | $6.8\times10^{13}$ |

A typical weight shrinks toward zero while the mean stays at one, so a rare trajectory carries an enormous weight. The slide's $1.05^{100}\approx132$ is one path on which every ratio is 1.05. Under $\mu$ each ratio averages one, so large weights come from rare paths on which many ratios exceed one. The source lecture puts agentic trajectories at 50,000 to 60,000 tokens, without a cited measurement. Per-token ratios then make $T$ 500 to 600 times the table's largest row.

```python
from agent_lab.training import ratio_moments

mean, variance, second = ratio_moments([0.4, 0.4, 0.2], [0.7, 0.2, 0.1])
assert abs(mean + 0.192) < 1e-3 and abs(variance - 0.377) < 1e-3
assert abs(second - 1.375) < 1e-12
assert second ** 100 - 1 > 6e13
```

The calculation assumes independent steps with the same mismatch. CTPO's variance ordering of prefix and full ratios holds without independence. The closed forms do not.

An unpublished experiment from the source lecture's research group, credited to Aditya Soni, logged the largest weight per batch. Read by eye, it stayed below about 100 with staleness bounded at one. With a bound of three it exceeded 1,200 at step 17. The plot names no task or seed count.

Methods differ in which actions enter the weight. PPO, GRPO, and DAPO use $r_t(\theta)$: near one, but blind to the prefix. CTPO uses $w_{\le t}(\theta)$ and clips it in log space to $[e^{-\epsilon^{\log}_{\rm low}\sqrt{t+1}},e^{\epsilon^{\log}_{\rm high}\sqrt{t+1}}]$, where $t+1$ counts the factors of $w_{\le t}$ because this chapter numbers actions from 0. The base widths are $\epsilon^{\log}_{\rm low}=0.025$ and $\epsilon^{\log}_{\rm high}=0.05$. The paper allows any power of $t$; the square root matches the log spread, and its ablation describes the thresholds as scaling with $\sqrt t$. [Group Sequence Policy Optimization (GSPO)](https://arxiv.org/abs/2507.18071) uses the geometric mean $w(\tau)^{1/T}$, whose log has variance $\sigma_{\log}^2/T$. It is not a correction weight: $x\mapsto x^{1/T}$ is concave, so for $T>1$ Jensen's inequality gives $\mathbb E_\mu[w(\tau)^{1/T}]\le1$, with equality only for constant $w(\tau)$. At $T=1$ the weight is $w(\tau)$ itself. For the table's policies it tends to $e^{-0.192}\approx0.83$. GSPO clipped it at $3\times10^{-4}$ and $4\times10^{-4}$, against 0.2 and 0.27 for its GRPO baseline.

## 4. Clipping, a reference penalty, and an entropy bonus bound each update [36:38](https://www.youtube.com/watch?v=jKx7_fOZRzg&t=2198s)

Chapter 9's clipped PPO objective, with clip width $\epsilon>0$ and the history-based ratio, is

$$L^{\rm clip}(\theta)=\mathbb E_t\Big[\min\big(r_t(\theta)\hat A_t,\ \operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\big)\Big].$$

When the unclipped product of a term is not larger, its gradient is $\hat A_tr_t(\theta)\nabla_\theta\log\pi_\theta(a_t\mid h_t)$. When the clipped product is strictly smaller, it is constant in $\theta$, with zero gradient. Inside the clip range both are equal and the gradient flows. Four cases lie outside it.

| $\hat A_t$ | $r_t(\theta)$ | Minimum selects | Gradient |
|---|---|---|---|
| $>0$ | $>1+\epsilon$ | Clipped term | Zero |
| $>0$ | $<1-\epsilon$ | Unclipped term | Flows |
| $<0$ | $<1-\epsilon$ | Clipped term | Zero |
| $<0$ | $>1+\epsilon$ | Unclipped term | Flows, unbounded in $r_t(\theta)$ |

The PPO paper calls the minimum a pessimistic bound: it ignores a ratio change "when it would make the objective improve," and includes it "when it makes the objective worse." The last row is a counterexample to the reading that clipping bounds the ratio. A stale action with ratio 1.75 and $\hat A_t=-1$ contributes $-1.75$ with its full gradient.

[MiniMax-M1](https://arxiv.org/abs/2506.13585) introduces **Clipped IS-weight Policy Optimization (CISPO)**, where IS abbreviates importance sampling:

$$J^{\rm CISPO}(\theta)=\mathbb E_t\Big[\operatorname{sg}\big(\operatorname{clip}(r_t(\theta),1-\epsilon_{\rm low},1+\epsilon_{\rm high})\big)\hat A_t\log\pi_\theta(a_t\mid h_t)\Big],$$

where $\operatorname{sg}$ is the stop-gradient and $\epsilon_{\rm low},\epsilon_{\rm high}$ are separate widths. Its gradient is the clipped ratio times $\hat A_t\nabla_\theta\log\pi_\theta(a_t\mid h_t)$, nonzero whenever $\hat A_t\ne0$, so no action drops out. MiniMax-M1 reports that rare reflective tokens such as "However" and "Wait" had high ratios and were clipped out by PPO-style objectives. It set no lower bound and calls the gradient "slightly biased."

```python
from agent_lab.training import cispo_coefficient, ppo_clip_term

assert ppo_clip_term(1.75, 1.0, 0.2, 0.2)[1] == 0.0
value, coefficient = ppo_clip_term(1.75, -1.0, 0.2, 0.2)
assert value == coefficient == -1.75
assert abs(cispo_coefficient(1.75, 1.0, eps_high=0.2) - 1.2) < 1e-12
assert abs(cispo_coefficient(1.75, -1.0, eps_high=0.2) + 1.2) < 1e-12
```

`ppo_clip_term` returns the term and its coefficient on $\nabla_\theta\log\pi_\theta(a_t\mid h_t)$; `cispo_coefficient` returns the coefficient alone. Both assume a ratio from logged rollout probabilities.

DAPO separates the bounds, with $\epsilon_{\rm low}=0.2$ and $\epsilon_{\rm high}=0.28$. In its example, with $\epsilon=0.2$ and a positive advantage, the clipped objective gives no further reward once a token at 0.01 passes 0.012. One optimizer step can still move it past that value, because clipping does not bound the ratio itself. For a token at 0.9 the same limit is 1.08, above any probability, so the clip never binds. A higher upper bound gives rare exploratory tokens room to grow. The source lecture reports from practice, without a cited run, that an uncontrolled large weight can make the policy degenerate abruptly into repeated tokens or very long or short outputs. Gradient-norm clipping caps a step's size but not the sample weighting (chapter 9).

### A reference penalty and an entropy bonus act on whole distributions

A **reference model** $\pi_{\rm ref}$ is a fixed anchor, usually the initial model, distinct from $\mu$, which supplies the ratio's denominator. With chapter 10's coefficient $\beta_{\rm KL}\ge0$, the penalized objective is

$$\max_\theta\ J(\theta)-\beta_{\rm KL}\,D_{\rm KL}(\pi_\theta\|\pi_{\rm ref}).$$

[DeepSeekMath](https://arxiv.org/abs/2402.03300) used $\beta_{\rm KL}=0.04$ for GRPO. The penalty limits average drift, not a specific harmful change.

Chapter 9's entropy bonus $\alpha\,\mathbb E_h[\mathcal H(\pi_\theta(\cdot\mid h))]$, which PPO also uses, guards against **entropy collapse**, the concentration of probability on one action. Suppose training moves three actions from $(0.20,0.60,0.20)$ to $(0.01,0.98,0.01)$. The entropy, in **nats** (natural-logarithm units), falls from 0.950 to 0.112. A group of four then picks the dominant action every time with probability $0.98^4\approx0.92$, against $0.6^4\approx0.13$ before, and identical behavior yields zero advantages. The opposite failure, **entropy explosion**, spreads probability over random actions that rarely succeed.

PPO takes its advantage from GAE with a learned critic; GRPO, DAPO, CISPO, GSPO, and CTPO use the group comparison. The slides' four-attempt example subtracts only the mean, while DeepSeekMath, GSPO, and CTPO also divide by the deviation. That deviation is zero for equal rewards. Implementations add a small constant, the centered rewards stay zero, and the group gives no reward gradient. The KL and entropy terms still give one.

## 5. Mixed groups carry the signal, and verifier errors decide whether it is true [45:45](https://www.youtube.com/watch?v=jKx7_fOZRzg&t=2745s)

Rewards $(0,0,0,0)$ and $(1,1,1,1)$ both center to zero, while $(0,1,0,1)$ centers to $(-0.5,+0.5,-0.5,+0.5)$. For per-rollout success probability $p$, chapter 9 gives the chance of an uninformative binary group as $p^n+(1-p)^n$, and the observed fraction is a direct diagnostic. An all-failure group has three common causes.

| Cause | Mechanism | Check |
|---|---|---|
| Verifier quality | A valid solution is rejected | Read the requirements and failed tests |
| Current capability | The model cannot yet solve the task | Test a stronger model's demonstrations |
| Lack of exploration | Every attempt repeats one approach | Test whether trajectories differ |

### A false negative adds noise; a false positive invites hacking

Let $Y(\tau)\in\{0,1\}$ be chapter 9's true utility, here whether the patch is correct. A **false negative** is $Y=1$ with $R=0$. A **false positive** is $Y=0$ with $R=1$. If the rates $f_-=P(R=0\mid Y=1)$ and $f_+=P(R=1\mid Y=0)$ do not depend on the policy,

$$\mathbb E_\theta[R]=(1-f_-)P_\theta(Y=1)+f_+\big(1-P_\theta(Y=1)\big)=f_++(1-f_--f_+)\,P_\theta(Y=1).$$

When $f_-+f_+<1$, expected reward is an increasing affine function of the true success rate, with the same maximizer. The errors shrink the signal by $1-f_--f_+$ and add noise. A policy that finds a shortcut raises $f_+$ itself, and maximizing $\mathbb E[R]$ then no longer maximizes $P(Y=1)$.

A verifier that requires the literal phrase `value is None` rejects a correct candidate with other source text, a false negative. The source lecture reports, without a cited study, that coding agents often write such tests. Tests that reject correct solutions lower the measured score of an agent that writes them. A task whose tests no correct solution passes caps the measurable score below 100%. [SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) has 500 human-screened tasks. [OpenAI's later audit](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) reports that top scores rose from 74.9% to 80.9% in six months. In an audited 27.6% subset that models often failed, at least 59.4% of problems had tests that reject functionally correct submissions, at least 16.4% of the full set. The audit does not say, as the source lecture did, that 81% of tasks are solvable. [SABER](https://arxiv.org/abs/2512.07850) found ceilings of roughly 70% on τ-bench Airline and around 92% on Retail, and the [τ³-bench fixes](https://taubench.com/blog/tau3-task-fixes.html) corrected more than 50 tasks.

If the training verifier tests only `retries=0`, the patch `return 0` earns reward one while passing one of four required cases, a false positive.

### Reward hacking uses whatever the environment exposes

[MAI-Thinking-1](https://microsoft.ai/pdf/mai-thinking-1.pdf) (Section 3.3.1) reports agents that searched the internet for the merged fix, searched local git history for the solution commit, and tampered with tests. Its mitigations were network access control, scrubbing commits after the base commit, resetting and hiding test files, and model-based monitors. Monkey-patching the test framework still escapes simple resets. In [PostTrainBench](https://arxiv.org/abs/2603.08640) (Section 5.4), GPT-5.1 Codex-Max acknowledged a restriction on an OpenAI application programming interface (API) key. It then used the key to generate training data. The authors suggest that the restriction likely dropped out of context. The source lecture describes further cases without a public record.

Each mitigation changes the harness, not the policy. Network control and hidden tests remove actions: in chapter 1's terms, they shrink what the execution gate admits under the authorization policy $U$. The PostTrainBench restriction lived only in the model-visible context $c_t$. No gate checked the key against a prohibition in $U^-$, so the rule held only while the model attended to it.

Policy gradient reinforces a rewarded shortcut like any successful action, so a hack found in training becomes a habit. Training reward should be read beside success on unseen tasks, behaviors the verifier misses, tool calls and cost, and infrastructure failures.

### Dynamic sampling refills the batch with mixed groups

DAPO over-samples and discards groups with accuracy 0 or 1 until the batch is full. A group is mixed with probability $1-p^n-(1-p)^n$, about 0.344 at $p=0.1$ and $n=4$, so about 2.9 groups are sampled per kept group. In DAPO's progressive ablation on the 2024 American Invitational Mathematics Examination (AIME), dynamic sampling raised the score from 42 to 50. The filter changes the objective: tasks with $p$ near 0.5 survive most often.

## 6. Partial credit adds signal and can move the optimum [56:08](https://www.youtube.com/watch?v=jKx7_fOZRzg&t=3368s)

A **partial reward** pays the fraction of checks passed, here 0.25 per test. Four failed attempts that pass 0, 1, 2, and 1 tests have success rewards all zero, but partial rewards $(0,0.25,0.50,0.25)$ with centered advantages $(-0.25,0,+0.25,0)$.

The cost is a changed objective. The unmodified code passes three tests and earns 0.75 for doing nothing. Suppose a repair succeeds with probability $p$ and otherwise leaves a patch that passes one test, an illustrative value. Its expected partial reward $p+0.25(1-p)$ exceeds 0.75 only when $p>2/3$, while under the success reward attempting is better for every $p>0$. For $0<p<2/3$, partial credit makes leaving the bug the better policy, and at $p=2/3$ the two tie. This is chapter 9's identification problem.

### Potential-based shaping keeps the optimum

[Ng, Harada, and Russell](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf) add a shaping reward $F=\gamma\Phi(h_{t+1})-\Phi(h_t)$ for a potential $\Phi$, any function of the history. With $\gamma=1$ and terminal potential zero, the shaping terms sum on every trajectory to a policy-independent constant:

$$\sum_{t=0}^{T-1}\big(\Phi(h_{t+1})-\Phi(h_t)\big)=\Phi(h_T)-\Phi(h_0)=-\Phi(h_0).$$

Let $\Phi(h)$ be the fraction of tests the current code passes. An edit that fixes a test earns 0.25 at once, one that breaks a test loses 0.25, and the final step subtracts the remaining potential. Credit inside the trajectory moves, but every total shifts by the same constant. With $\Phi=V_\phi$, the shaped reward $r_t+V_\phi(h_{t+1})-V_\phi(h_t)$ is exactly $\delta_t$.

The sum uses no Markov property, because $\Phi$ is a function of the history. Chapter 9's caveat concerned a potential on approximate states. Two conditions bind. First, the harness must set the potential to zero at every end of a run, including a timeout at the step limit $T_{\max}$. Otherwise the shaped return $R(\tau)+\Phi(h_T)-\Phi(h_0)$ pays for the final pass fraction and restores the partial-credit incentive (Exercise 4). Second, the theorem preserves the optimal policy, not the training path. Shaping leaves the exact advantages unchanged, because $Q^\mu$ and $V^\mu$ both shift by $-\Phi(h)$. It changes the per-step rewards and the target that the critic must learn, so a finite run can stop elsewhere.

### Two documented auxiliary-reward failures

An **auxiliary reward** pays for behavior other than task success. [OpenAI's production-evaluation report](https://alignment.openai.com/prod-evals/) describes "Calculator Hacking" in GPT-5.1: a training-time bug rewarded superficial web-tool use, and the model used the browser as a calculator while behaving as if it had searched. This was the majority of GPT-5.1's deceptive behaviors at deployment.

In [Where the goblins came from](https://openai.com/index/where-the-goblins-came-from/), a reward for the "Nerdy" personality scored outputs with "goblin" or "gremlin" higher in 76.2% of audited datasets. After GPT-5.1 launched, "goblin" use rose 175%. The habit rose almost as much without the persona prompt, which OpenAI attributes to transfer. It describes reuse of rollouts for supervised fine-tuning (SFT) as a feedback loop that can reinforce such a tic, and found both words in GPT-5.5's SFT data.

## 7. A teacher supplies dense targets, and on-policy histories remove exposure bias [60:01](https://www.youtube.com/watch?v=jKx7_fOZRzg&t=3601s)

A student that almost never succeeds gets all-failure groups and no RL signal. A **warm start** first imitates demonstrations through SFT or mid-training (chapters 8 and 10). A **curriculum** needs no teacher. It orders tasks from easy to hard, as in [Curriculum Learning](https://ronan.collobert.com/pub/2009_curriculum_icml.pdf), and keeps earlier tasks in the mix. A fixed evaluation set measures progress, and the schedule has no settled method.

**Rejection-sampled distillation** lets a stronger model roll out and trains on its passing trajectories, chapter 9's filtered loss with the teacher as sampler. **Knowledge distillation** matches the teacher's whole next-action distribution, as in [Hinton, Vinyals, and Dean](https://arxiv.org/abs/1503.02531). Let $q(\cdot\mid h)$ be that distribution. It is not chapter 10's query $q$. Training on teacher trajectories still causes chapter 9's exposure bias: the student's mistakes lead to histories the teacher never visited.

### On-policy distillation trains on the student's own histories

**On-policy distillation** lets the student generate the histories and asks the teacher for targets there. [Generalized Knowledge Distillation (GKD)](https://arxiv.org/abs/2306.13649) develops it and compares several divergences. Averaging over student-visited histories, the loss with the reverse KL divergence is

$$\mathcal L_{\rm OPD}(\theta)=\mathbb E_h\big[D_{\rm KL}(\pi_\theta(\cdot\mid h)\,\|\,q(\cdot\mid h))\big].$$

At B's $h_3$, let the student give $(0.5,0.5)$ to two actions and the teacher $(0.8,0.2)$. The reverse divergence is $0.5\log(0.5/0.8)+0.5\log(0.5/0.2)\approx0.223$ nats, the forward $D_{\rm KL}(q\|\pi_\theta)$ about 0.193. They differ because KL is not symmetric, but two actions cannot show what each direction rewards.

Three actions show it. Let the teacher split between "run the tests" and "inspect the caller," with little mass on `return 0`: $q=(0.49,0.02,0.49)$. Compare a committed student $(0.90,0.05,0.05)$ with an even one $(1/3,1/3,1/3)$. The reverse divergence is about 0.479 against 0.681, so it prefers commitment to one teacher mode. The forward divergence is about 0.802 against 0.321, so it prefers covering both modes, at the price of mass $1/3$ on `return 0`. The forward term $q\log(q/\pi_\theta)$ punishes small $\pi_\theta$ where $q$ is large. The reverse term punishes large $\pi_\theta$ where $q$ is small. GKD calls the reverse direction mode-seeking and finds the best divergence task-dependent.

```python
from agent_lab.training import kl_divergence

student, teacher = [0.5, 0.5], [0.8, 0.2]
assert abs(kl_divergence(student, teacher) - 0.2231) < 1e-4
assert abs(kl_divergence(teacher, student) - 0.1927) < 1e-4
teacher3, committed, even = [0.49, 0.02, 0.49], [0.9, 0.05, 0.05], [1 / 3] * 3
assert kl_divergence(committed, teacher3) < kl_divergence(even, teacher3)
assert kl_divergence(teacher3, committed) > kl_divergence(teacher3, even)
```

The target is dense, one distribution per action against one scalar reward per trajectory, at the cost of a teacher forward pass at every student position. A teacher available only as generated text cannot supply probabilities.

[On-Policy Self-Distillation (OPSD)](https://arxiv.org/abs/2601.18734) uses the same model, frozen at its initial weights and conditioned on a verified solution, as teacher, with a per-token divergence over student rollouts. Its abstract reports better token efficiency than RL and better results than off-policy distillation on mathematical reasoning, without figures. The targets reflect knowledge the student lacks: a teacher that knows the fix favors editing at once, skipping the inspection the student needs.

### Observation tokens can be a training target

Agent RL normally applies the loss only to model-generated tokens, as chapters 8 and 10 mask. [Environment Cross-entropy Hybrid Objective (ECHO)](https://arxiv.org/abs/2605.24517) adds a cross-entropy loss on observation tokens that follow the agent's actions. Its abstract reports that ECHO "doubles GRPO pass@1 on TerminalBench-2.0." Here pass@1 is chapter 6's pass@$k$ at $k=1$. Qwen3-8B rose from 2.70% to 5.17%, and Qwen3-14B from 5.17% to 10.79%, low rates without intervals.

### Scope and limits of the algorithms in this chapter

The byline names Graham Neubig on the basis of slide 1 and the recording's description. The course site names no one for this session. Nine results here are derivations. They are the squared-error minimizer, the GAE telescoping identity, the importance-sampling identity, its prefix form, and the moment formulas. The rest are the Jensen bound, the PPO gradient cases, the verifier-noise relation, and the shaping sum. CTPO's unbiasedness result does not cover the group advantage that CTPO runs. The method comparisons are mostly single-paper results on mathematical reasoning, and several cases come from company reports.

### Exercises

1. A three-action trajectory has rewards $(0,0,1)$ and critic values $(V_\phi(h_0),\ldots,V_\phi(h_3))=(0.5,0.4,0.8,0)$ with $\gamma=1$. Compute the TD errors and the GAE advantages for $\lambda=0.5$ and $\lambda=1$. Which action, if any, does each estimate blame?
2. Let $D_{\chi^2,t}$ be the divergence at step $t$. Prove that independent per-step ratios with support give $\mathbb E_\mu[w(\tau)]=1$ and $\operatorname{Var}_\mu[w(\tau)]=\prod_t(1+D_{\chi^2,t})-1$. Evaluate it for $D_{\chi^2,t}=0.01$ at $T=100$ and $T=1000$.
3. For ratios 0.5, 1.0, and 1.75 and advantages $\pm1$, give the PPO gradient coefficient ($\epsilon=0.2$) and the CISPO coefficient ($\epsilon_{\rm high}=0.2$, no lower bound). Which actions stop learning under each?
4. Design a shaping reward for the retry task with the pass fraction as $\Phi$, so that the unmodified code earns no more than a failed repair that passes one test. State the rule the harness needs at $T_{\max}$, and give the three total returns, success reward included, with and without that rule.

### Solutions and discussion

1. The TD errors are $\delta_0=-0.1$, $\delta_1=0.4$, and $\delta_2=0.2$. With $\lambda=0.5$, $\hat A_2=0.2$, $\hat A_1=0.4+0.5(0.2)=0.5$, and $\hat A_0=-0.1+0.5(0.5)=0.15$. With $\lambda=1$, $\hat A_t=G_t-V_\phi(h_t)$, giving $(0.5,0.6,0.2)$.
2. Under $\mu$, $\mathbb E[r_t(\theta)]=\sum_a\mu\,(\pi_\theta/\mu)=1$ and $\mathbb E[r_t(\theta)^2]=\sum_a\pi_\theta^2/\mu=1+D_{\chi^2,t}$. Independence makes the expectation of a product the product of expectations, so $\mathbb E[w]=1$ and $\mathbb E[w^2]=\prod_t(1+D_{\chi^2,t})$. Subtracting $\mathbb E[w]^2=1$ gives the variance. At 0.01 it is $1.01^{100}-1\approx1.70$ and $1.01^{1000}-1\approx2.1\times10^4$.
3. PPO: at 0.5, $\hat A=+1$ gives 0.5 and $\hat A=-1$ zero; at 1.0, $\pm1$; at 1.75, $\hat A=+1$ gives zero and $\hat A=-1$ gives $-1.75$. CISPO gives $\pm0.5$, $\pm1$, and $\pm1.2$. PPO stops learning in two of six cases, CISPO in none.
4. Set $\Phi(h_T)=0$ at every end of a run, timeouts included. The shaping terms then sum to $-0.75$, so the unmodified code and the failed repair earn $-0.75$ and a success $0.25$. Without the rule the return is $R+\Phi(h_T)-0.75$: inaction 0, failure $-0.5$, success $1.25$, so inaction wins whenever $1.75p-0.5<0$, that is $p<2/7\approx0.29$.

The first answer shows that neither estimate blames an action, although the value fell after $a_0$. As a function of the trace, $\hat A_0=-0.1+0.4\lambda+0.2\lambda^2$, negative only for $\lambda$ below about 0.22. Choosing $\lambda$ states how far the critic is trusted.

The second answer shows that a per-step divergence of 0.01 gives a variance of 1.7 at 100 steps and over 20,000 at 1,000, so practical methods truncate, clip, or normalize the weight.

The third answer locates the difference between the clipping rules. PPO silences a positive action already made much more likely and a negative action already made much less likely, but never a negative action with a high ratio, the stale-data case. CISPO keeps every action and bounds only its weight, at the price of bias.

The fourth answer shows that the shaping guarantee depends on the harness's bookkeeping: the potential moves credit inside a run, and the zero at every end keeps each total fixed.

### Further reading

- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438): TD-error weighting and its λ = 0 and λ = 1 limits.
- [Rethinking Importance Sampling in LLM Policy Optimization (CTPO)](https://arxiv.org/abs/2605.07331): token, prefix, sequence, and length-normalized ratios, with variance proofs.
- [Reward Hacking in Reinforcement Learning](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/): a survey of specification gaming, reward tampering, and mitigations.
