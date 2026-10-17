# Lecture 8 — Supervised fine-tuning for agents

*Independent study chapter · Based on a CMU 11-768: AI Agents lecture by Yueqi Song, 17 September 2026 · [Course and attribution](index.html#sources)*

## 0. A trajectory becomes a training sequence [00:00](https://www.youtube.com/watch?v=O3HSU0AoILc&t=0s)

An agent successfully repairs a repository after inspecting files, making a mistaken edit, observing a failing test, reverting the mistake, and producing a correct patch. Supervised fine-tuning can increase the probability of every recorded assistant action in that trajectory. This includes the mistake unless the data representation or loss explicitly excludes it.

**Supervised fine-tuning (SFT)** updates a pretrained model by maximizing the likelihood of target tokens in labeled sequences. For an agent, a training example is often a **trajectory**: an ordered record of system instructions, user messages, assistant reasoning or messages, tool calls, tool observations, and a final response.

Let trajectory $i$ become token sequence $x^{(i)}=(x_1^{(i)},\ldots,x_{T_i}^{(i)})$. Let $m_t^{(i)}\in\{0,1\}$ be a **loss mask**. A value of one means prediction of token $x_t^{(i)}$ contributes to the loss. Let $w_t^{(i)}\ge0$ be an optional token weight. For model parameters $\theta$, the normalized masked negative log-likelihood is

$$\mathcal L_{\rm SFT}(\theta)=
-\frac{
\sum_i\sum_{t=1}^{T_i}m_t^{(i)}w_t^{(i)}
\log p_\theta(x_t^{(i)}\mid x_{<t}^{(i)})
}{
\sum_i\sum_{t=1}^{T_i}m_t^{(i)}w_t^{(i)}
}.$$

The notation $x_{<t}^{(i)}$ denotes all earlier tokens in example $i$. The probability $p_\theta$ is the model's predicted probability for the recorded target token. Minimizing the loss increases probabilities of selected recorded tokens. The equation does not evaluate whether the recorded actions were good.

SFT can teach tool-call syntax, recovery patterns, long-horizon continuation, and task-specific behavior. Reinforcement learning (RL) can subsequently optimize sampled behavior against a reward. It updates model weights, so one training run can affect many tasks and harnesses. This reach makes weight updates powerful and harder to localize than editing a prompt or tool adapter.

### The training target is selected behavior, not the recorded outcome

Consider the repair trajectory described above. Let assistant action sequence be $(a_1,a_2,a_3,a_4)$: inspect a file, make a wrong edit, run a failing test, and make the correct edit. Suppose the final repository passes. Ordinary assistant-token SFT assigns positive likelihood pressure to all four recorded actions. The terminal success label does not retroactively distinguish the wrong edit from the recovery.

Let $q_t\in\{0,1\}$ be a step-quality label and let $m_t$ select assistant tokens belonging to action $a_t$. A step-filtered objective can replace $m_t$ by $m_tq_t$. This removes targets labeled undesirable. It also changes the conditioning history: later correct actions are still trained after the wrong action appears in context. Removing the wrong event from the serialized history would create a different, possibly incoherent trajectory.

Three data operations must therefore be distinguished. **Outcome filtering** keeps or removes an entire trajectory according to its terminal result. **Step masking** retains the history but removes selected target tokens from the loss. **Trajectory editing** changes the history itself. They produce different conditional distributions and should not share one label such as “cleaning.”

The gradient makes this local effect explicit. For one selected token $x_t$, its contribution is

$$-w_t\nabla_\theta\log p_\theta(x_t\mid x_{<t}).$$

A positive weight increases the target token's log probability under gradient descent. A zero weight supplies context without direct target pressure. A negative weight would define a different objective and is not ordinary maximum-likelihood SFT. Quality labels, masks, and weights must be audited before training because the optimizer follows them literally.

Weight updates also broaden the blast radius. A prompt edit changes runs that include the prompt. A tool-adapter edit changes calls through that adapter. SFT changes any deployment using the checkpoint. Hold-out evaluation must therefore include capabilities and safety properties outside the target task family.

## 1. Chat templates determine what the model sees [08:00](https://www.youtube.com/watch?v=O3HSU0AoILc&t=480s)

A **chat template** serializes structured messages into tokens. It inserts role markers, tool-call delimiters, observation boundaries, and end-of-turn tokens. The same semantic conversation can yield different token sequences under different templates.

Let structured trajectory $\tau$ contain messages and tool events. Let $q$ be a template. The serialization function $S_q$ produces tokens and mask:

$$S_q(\tau)=(x_{1:T},m_{1:T}).$$

Template choice is part of the learned interface. Training with one delimiter and serving with another changes the conditional distribution presented to the model. [Hugging Face chat-template documentation](https://huggingface.co/docs/transformers/en/chat_templating) describes how tokenizer templates convert role-structured messages to model-specific strings.

The usual agent mask selects assistant-produced tokens and excludes system, user, and tool-observation tokens. This trains the model to predict actions from context rather than reproduce observations. A different objective may deliberately train a world model to predict tool results. That objective must label the change; otherwise evaluation may confuse action modeling with environment modeling.

Reasoning tokens present another choice. Including them teaches their observed distribution and gives them weight proportional to length. Excluding them trains only visible messages or tool actions. Partial weighting can rebalance components, but a weight is an optimization decision rather than a semantic guarantee.

The companion function evaluates the chapter's finite-token loss from target probabilities:

```python
from agent_lab.training import masked_cross_entropy

loss = masked_cross_entropy(
    target_probabilities=[0.8, 0.5, 0.25, 0.9],
    mask=[False, True, True, False],
)
```

Only the second and third probabilities contribute, so the loss is $-[\log(0.5)+\log(0.25)]/2\approx1.040$. The function demonstrates masking. It is not an automatic-differentiation training framework.

### Serialization has invariants

A valid serialized agent trajectory should satisfy at least four invariants. Every tool result refers to a preceding tool call. Roles occur in an order accepted by the serving runtime. Special tokens are inserted by one component only. The loss mask aligns exactly with token positions after tokenization.

Let $x_{1:T}=S_q(\tau)$ be the serialized tokens and let $r_t$ be the role associated with token $x_t$. A common mask is $m_t=\mathbf1[r_t=\text{assistant}]$, where $\mathbf1[\cdot]$ equals one when its condition is true. Tool-call arguments then receive loss because the assistant produced them. Tool observations remain context with zero loss because the environment produced them.

The rule needs exceptions. A template can place role delimiters adjacent to assistant content. Some delimiters should be predicted because the model must emit them at inference. Others are injected by the runtime. Duplicating a beginning-of-sequence or end-of-turn token during serving creates a distribution shift even when the visible text matches.

A serialization unit test should use a tiny hand-constructed trajectory. It checks the exact token identifiers, decoded boundaries, loss-mask positions, and round trip through the serving parser. The test should include one malformed call and one error observation because recovery syntax often differs from successful syntax.

[Hugging Face chat templates](https://huggingface.co/docs/transformers/en/chat_templating) expose the model-specific rendering function used by many open checkpoints. The documentation is operational evidence for template mechanics. A model paper or tokenizer configuration remains the authority for the intended special tokens of a particular checkpoint.

## 2. Samples and tokens induce different mixtures [18:30](https://www.youtube.com/watch?v=O3HSU0AoILc&t=1110s)

Suppose dataset domain $d$ contains $N_d$ trajectories with mean selected-token count $\bar T_d$. Sampling domains in proportion to example counts gives nominal share

$$\alpha_d^{\rm sample}=\frac{N_d}{\sum_jN_j}.$$

If every selected token has equal weight, the approximate share of loss terms is

$$\alpha_d^{\rm token}=\frac{N_d\bar T_d}{\sum_jN_j\bar T_j}.$$

These shares can differ sharply. If coding and general-assistant datasets each contain 10,000 trajectories, but their average selected lengths are 8,000 and 800 tokens, coding supplies about $8000/(8000+800)=90.9\%$ of selected tokens. A sample-balanced mixture is not token-balanced.

The training objective should reflect the intended capability distribution. Per-domain sampling, per-token weights, and sequence-length caps alter the effective objective. Report both example and token counts, the mask policy, and the normalization rule.

Long trajectories also magnify local errors. If a successful run contains a wrong edit followed by recovery, ordinary SFT treats both assistant actions as targets. A process label can mask the wrong action. A step weight can reduce it. Filtering the entire trajectory discards the recovery lesson as well. The choice depends on whether the model should learn imitation, error recovery, or only verified actions.

### Effective mixture includes sampling, length, mask, and weight

Let domain $d$ be sampled with probability $s_d$. Let $M_d$ be the expected number of selected tokens in one sampled trajectory, and let $\bar w_d$ be their mean weight. The approximate share of total weighted loss contributed by the domain is

$$\alpha_d=\frac{s_dM_d\bar w_d}{\sum_j s_jM_j\bar w_j}.$$

This expression generalizes sample and token shares. A domain can be oversampled but contribute little if most of its tokens are masked. A short high-weight safety dataset can influence the objective more than its raw token count suggests. Report empirical weighted-token totals from the actual preprocessed batches rather than relying only on configuration values.

Suppose coding and dialogue are sampled equally. Coding trajectories contain 8,000 tokens, of which 50 percent are selected. Dialogue trajectories contain 1,000 tokens, of which 80 percent are selected. With unit weights, their expected selected-token counts are 4,000 and 800. Coding therefore contributes $4000/(4000+800)=5/6\approx0.833$ of the loss, despite equal trajectory sampling.

Batch formation adds variance. If long trajectories occupy most of a batch, the number of independent tasks per update falls. Gradient estimates can become dominated by a few task instances. Logging per-domain weighted tokens and per-batch task counts reveals this concentration.

The companion calculation uses sampling probability, selected-token count, and mean token weight:

```python
from agent_lab.training import effective_token_shares

shares = effective_token_shares(
    sample_probabilities=[0.5, 0.5],
    selected_tokens=[4000, 800],
    mean_weights=[1.0, 1.0],
)
assert shares == (5 / 6, 1 / 6)
```

These are expected loss shares. Actual batches fluctuate around them and can differ when packing or truncation changes the realized selected-token counts.

## 3. Data selection is a causal intervention on behavior [22:30](https://www.youtube.com/watch?v=O3HSU0AoILc&t=1350s)

Agent SFT data can come from humans, a stronger teacher model, the same model, or a mixture. A trajectory can be kept in full, filtered by outcome, filtered by shape, or relabeled at the step level. Outcome filtering removes failed runs but does not ensure every action in a successful run was desirable.

Let $D$ be a candidate distribution of recorded trajectories. Let selection rule $F(\tau)\in\{0,1\}$ decide whether to retain trajectory $\tau$. The selected training distribution is

$$D_F(\tau)=\frac{D(\tau)F(\tau)}{\mathbb E_{\tau'\sim D}[F(\tau')]},$$

provided the denominator is positive. The rule changes both data quality and coverage. Filtering for short successful traces may remove recovery behavior. Filtering for long traces may reward wandering.

Teacher quality is multidimensional. A teacher can solve more tasks while producing trajectories that are difficult for a smaller student to imitate. Tool conventions, reasoning style, action granularity, and context length affect transfer. Evaluate the trained student rather than using teacher benchmark rank as a proxy.

[Self-Instruct](https://arxiv.org/abs/2212.10560) bootstraps instruction data from model-generated instructions and filters invalid or redundant generations. [UltraChat](https://arxiv.org/abs/2305.14233) scales synthetic multi-turn conversations. These projects show how synthetic data can expand coverage. They also make the generator, filters, and deduplication rules part of the learned distribution.

The [Flan Collection](https://arxiv.org/abs/2301.13688) reports ablations of task balancing and prompt mixtures for instruction tuning. [LIMA](https://arxiv.org/abs/2305.11206) studies a much smaller, curated alignment set. Their settings and claims differ. Together they rule out a simple law that more raw examples always produce a better instruction-following model. Quality, diversity, base-model competence, and evaluation distribution interact.

Task-source diversity can matter more than repeated trajectories from the same distribution. [Agent Data Protocol](https://arxiv.org/abs/2510.24702) defines a common representation for heterogeneous agent datasets and reports multi-domain SFT results. A common format reduces conversion work; it does not make source tasks statistically equivalent. Preserve original observations, action types, provenance, licenses, and verifier semantics.

### Demonstration shift and DAgger

SFT on expert demonstrations conditions on histories visited by the expert. At deployment, the learned policy visits its own histories. Small mistakes can therefore move it outside the demonstration distribution. This is **covariate shift** over histories.

[DAgger](https://proceedings.mlr.press/v15/ross11a.html) addresses this imitation-learning problem by repeatedly running the learner, querying an expert for actions on learner-visited states, aggregating those labeled states, and retraining. For expensive language-model agents, full expert relabeling may be infeasible. The principle remains useful: collect training examples where the current policy actually fails, not only ideal expert traces.

### Filtering changes the question answered by the dataset

Selection on success induces bias toward tasks and histories that the collector already handles. Let task variable $X$ affect both selection $F$ and trajectory behavior $\tau$. The selected distribution is $P(\tau,X\mid F=1)$, not the deployment distribution $P(\tau,X)$. If difficult tasks rarely succeed, outcome filtering removes them precisely when more learning signal is needed.

Inverse-probability weighting can correct selection only when the selection probability is known and nonzero. If $e(\tau)=P(F=1\mid\tau)$, a retained trajectory can receive weight proportional to $1/e(\tau)$. In agent data, $e$ is rarely known and can be extremely small. Large weights create high variance. The equation is mainly a warning: filtering is a distributional intervention, not a free improvement in quality.

Data curricula can preserve coverage while raising quality. One stratum contains clean expert demonstrations. Another contains successful self-generated traces. A third contains corrected failures or recovery fragments. Sampling weights state the intended balance. Evaluation by task difficulty and failure type determines whether the curriculum improved capability rather than only the average easy case.

[SWE-Gym](https://arxiv.org/abs/2412.21139), a training environment for software-engineering agents, constructs training environments and trajectories for software agents. [BalanceSFT](https://arxiv.org/abs/2505.20192) studies balanced training signals and data hardness in supervised fine-tuning for function calling. Their shared methodological lesson is to measure the resulting student under the target interaction protocol. Teacher quality scores and dataset size do not substitute for that experiment.

## 4. A common trajectory protocol needs typed semantics [38:00](https://www.youtube.com/watch?v=O3HSU0AoILc&t=2280s)

Datasets may store screenshots, HTML, accessibility trees, shell output, patches, or structured API results. Flattening all fields to anonymous text loses the distinction between action and observation. A common protocol should preserve an event type, actor, timestamp, content representation, tool identifier, result status, and provenance.

Define event $e_t=(r_t,k_t,c_t,p_t)$, where $r_t$ is its role, $k_t$ its event kind, $c_t$ its content, and $p_t$ its provenance. A conversion from source dataset $D_j$ to common protocol $A$ is $f_j:D_j\to A$. A harness-specific renderer $g_h:A\to X_h$ produces the training sequence used by harness $h$.

The factorization avoids writing a converter between every dataset and every harness. With $n$ datasets and $m$ harnesses, pairwise converters require up to $nm$ paths. An interlingua requires $n+m$ converters. This count measures engineering interfaces, not information preservation. If $f_j$ discards a screenshot or uncertainty label, no downstream renderer can recover it.

Validate conversion with round-trip or semantic checks. Count event types before and after conversion. Confirm tool arguments and results retain their association. Ensure lower-trust webpage content does not become a system message.

### A protocol is useful only if conversion is information-preserving

Let source record $z\in D_j$ and converted event sequence $f_j(z)\in A$. Define a source-semantic query $Q_k(z)$, such as “which tool produced this observation?” For every query required by training or audit, an equivalent query $\widetilde Q_k$ over the common protocol should satisfy

$$Q_k(z)=\widetilde Q_k(f_j(z)).$$

If no such query exists because conversion discarded the tool identifier or screenshot reference, the conversion is not semantics-preserving for that use. Round-trip text equality is insufficient when source formats encode type, trust, or binary media outside the text.

The [Agent Data Protocol](https://arxiv.org/abs/2510.24702) is a primary example of an interlingua for heterogeneous agent trajectories. An interlingua reduces the number of adapters, but it also creates a common failure point. Version schemas. Preserve unknown fields. Record the original dataset and record identifier. Make lossy transformations explicit.

Validation should include structural counts and semantic fixtures. Count roles, event kinds, tool calls, errors, and media references before and after conversion. For selected records, reconstruct the original action-observation pairing and compare it manually or with a source-specific checker. Reject an orphan result instead of guessing which call produced it.

## 5. Packing and optimization preserve trajectory boundaries [43:30](https://www.youtube.com/watch?v=O3HSU0AoILc&t=2610s)

**Packing** concatenates multiple training examples into a fixed-length sequence to reduce padding. If attention is allowed across packed examples, one conversation can leak into another. Position handling and attention masks must preserve example boundaries.

Let capacity be $L$ tokens and trajectory lengths be $T_1,\ldots,T_n\le L$. Packing assigns each trajectory to a bin without splitting it. Minimizing padding is a bin-packing problem. Best-fit heuristics place each trajectory into the bin with the smallest remaining capacity that can hold it. They improve utilization but do not change the statistical objective unless padding or cross-example attention was mishandled.

Truncating a trajectory can remove the final reward evidence or separate a tool call from its result. For agent data, retaining whole conversations is often preferable. If a trajectory exceeds maximum length, define a principled segmentation or compaction policy and record the lost dependencies.

Mixture-of-experts models add routing behavior. Narrow fine-tuning data can concentrate traffic on a subset of experts. Report routing statistics and load-balancing losses when they affect stability. A decreasing token loss alone does not establish that the deployed agent improved.

### Packing requires block-diagonal attention

Suppose two trajectories with lengths $T_1$ and $T_2$ are packed into one sequence. Let attention mask $A_{ij}$ equal one when token $i$ may attend to token $j$. To prevent cross-example leakage, $A$ should be block diagonal in example identity and causal within each block:

$$A_{ij}=\mathbf1[e(i)=e(j)]\,\mathbf1[j\le i],$$

where $e(i)$ is the packed-example identifier of token $i$. Position identifiers may reset at each example or continue according to the model's training design. The serving and training conventions must agree.

Without the first indicator, the second trajectory can condition on the first trajectory's user data, tool outputs, and final answer. The loss may improve because the packed neighbor leaks useful patterns. Deployment will not provide that neighbor. Attention-mask tests should inspect actual model inputs rather than only the data collator's configuration.

Truncation has similar semantic consequences. Left truncation can remove the user goal or system constraints. Right truncation can remove the successful resolution. Middle truncation can separate a tool call from its result. A compaction procedure should preserve goal, hard constraints, open calls, selected evidence, and final labels. Report how many trajectories were dropped, truncated, split, or compacted.

### Optimization statistics need behavioral interpretation

Let batch $B$ contain selected tokens with individual negative log-likelihoods $\ell_j$. The reported batch loss is often their weighted mean. Its decrease can arise because the model improved on common easy tokens while rare tool arguments remain poor. Log loss by role, domain, trajectory-length bucket, and action type. Tool-call validity and exact argument accuracy are closer to deployed behavior than aggregate perplexity alone.

Gradient accumulation changes the number of microbatches contributing to one optimizer step. If microbatch $b$ contains weighted-token mass $M_b$, averaging each microbatch loss equally gives short and long microbatches equal influence. Aggregating the numerator and denominator across microbatches gives every weighted token equal influence. Both are defensible objectives. They are not equivalent.

Mixed-precision training adds a numerical layer. Overflow, underflow, and dynamic loss scaling can skip or distort updates. Record skipped steps, gradient norms, learning rate, token throughput, and checkpoint hashes. A reproducible report distinguishes a data-objective change from an optimizer or systems failure.

Checkpoint selection should use a declared rule chosen before examining the final test set. Selecting the lowest held-out token loss is appropriate only when that metric matches the intended claim. Selecting by interactive success requires repeated rollouts and uncertainty estimates because the metric is noisier and more expensive.

## 6. Training loss and agent success measure different distributions [48:30](https://www.youtube.com/watch?v=O3HSU0AoILc&t=2910s)

Training loss evaluates recorded target tokens under recorded histories. Deployment evaluates actions under histories produced by the current policy and environment. Low loss can coexist with poor task success because the model encounters unseen states, calls tools with invalid arguments, or fails to recover.

Before a large run, an **overfit test** trains on a tiny set until loss drops substantially and predictions match the targets. Failure indicates a likely pipeline bug in serialization, masks, labels, optimization, or checkpoint loading. Passing the test establishes only that the implementation can fit those examples.

Evaluation should use the target harness and checkable outcomes. Split data according to the generalization claim. To claim repository transfer, keep repositories disjoint. To claim site transfer, keep sites disjoint. Keeping only individual trajectories disjoint permits leakage through repeated tasks and environments.

Let $L_D(\theta)$ be held-out token loss on data distribution $D$ and $S_E(\theta)$ be task success in interactive environment distribution $E$. Neither is a deterministic function of the other. Monitor both, plus regression capabilities, forbidden effects, cost, and latency.

### Validation should separate pipeline health from behavioral gain

A tiny overfit test answers whether the implemented objective can fit a few examples. It should drive selected-token loss close to zero and reproduce target tool syntax under greedy decoding. If it fails, likely causes include a shifted mask, wrong labels, missing trainable parameters, excessive regularization, or checkpoint-loading errors.

A held-out teacher-forced loss test answers whether the model predicts recorded actions under recorded prefixes. An offline action test answers whether it selects the reference action from a fixed history. An interactive evaluation answers whether the closed-loop agent completes tasks under its own histories. These tests form a ladder. Passing an earlier rung is necessary for many pipelines but insufficient for the next.

Let $S_{e,h}(\theta)$ denote success in environment family $e$ under harness $h$. A robustness matrix evaluates several pairs $(e,h)$ rather than one diagonal configuration. Large off-diagonal drops reveal learned dependence on tool names, error wording, viewport conventions, or prompt templates. This dependence may be acceptable for one fixed product. It must be measured before claiming a general agent capability.

Regression evaluation should include the base model's non-agent tasks when the checkpoint will serve them. Catastrophic forgetting is a change in the capability vector, not merely a rise in one validation loss. The acceptance policy can require minimum retained performance for named capabilities.

### Harness robustness

Trajectories collected in one harness teach its prompt wording, tool names, error messages, and action granularity. If training harness is $h$ and evaluation harness is $h'$, the performance difference

$$\Delta_{h\to h'}=S_{h'}(\theta_h)-S_h(\theta_h)$$

measures a joint distribution shift rather than model quality alone. Train and evaluate a matrix of harnesses when the deployment interface may change.

## 7. SFT initializes a policy; it does not finish the objective [53:30](https://www.youtube.com/watch?v=O3HSU0AoILc&t=3210s)

SFT maximizes likelihood of selected actions. It does not directly maximize task reward. Its role before reinforcement learning is often to establish syntax, basic tool use, and a policy capable of generating some successful trajectories.

Too little SFT may leave reinforcement learning without positive samples. Too much narrow SFT can reduce exploration or overfit harness conventions. Select an SFT checkpoint using downstream learning behavior as well as its immediate score. The best standalone SFT checkpoint need not be the best initialization for later reinforcement learning.

### The SFT checkpoint determines the support available to RL

Let $\mathcal T^+$ be the set of trajectories receiving positive reward. If an SFT policy assigns negligible probability to every trajectory in $\mathcal T^+$, on-policy reinforcement learning may observe no successes. Define initial success probability

$$p_0=\Pr_{\tau\sim\pi_{\rm SFT}}[R(\tau)>0].$$

With $N$ independent rollouts, the probability of observing at least one positive trajectory is $1-(1-p_0)^N$. If $p_0=0.001$ and $N=100$, this probability is only about $0.095$. SFT can make RL feasible by raising support on valid tool syntax and minimally successful behaviors.

Too much imitation can narrow diversity. A policy concentrated on one demonstrated strategy may have high SFT accuracy but poor exploration. Checkpoint selection for RL should therefore examine success rate, entropy or diversity under controlled sampling, invalid-action rate, and learning progress after a small RL pilot. The downstream objective determines the useful cold start.

[DeepSeek-R1](https://arxiv.org/abs/2501.12948) contrasts a direct-RL system with a later pipeline that uses curated cold-start data before further reinforcement learning. The evidence is specific to its reasoning tasks and training system. It supports the existence of a trade-off, not a universal recipe for the quantity of SFT.

This conclusion is empirical and pipeline-dependent. Reports such as [DeepSeek-R1](https://arxiv.org/abs/2501.12948) compare cold-start and reinforcement-learning stages under specific models and tasks. They do not imply one universal amount of SFT.

### Teacher forcing creates an on-policy mismatch

During supervised fine-tuning, **teacher forcing** conditions the model on the reference prefix $y_{<t}^*$ when predicting reference token $y_t^*$. At deployment, the model conditions on its own sampled prefix $\hat y_{<t}$. The corresponding context distributions are generally different:

$$y_{<t}^*\sim d_{\rm data},\qquad
\hat y_{<t}\sim d_{\pi_\theta}.$$

Here $d_{\rm data}$ is the distribution of prefixes in the demonstration data, and $d_{\pi_\theta}$ is the distribution induced by policy $\pi_\theta$. An early deployment error may create a prefix that never appears in the training set. Later predictions are then evaluated under an unfamiliar context. This accumulation is called **exposure bias**.

[DAgger](https://proceedings.mlr.press/v15/ross11a.html), short for Dataset Aggregation, addresses the analogous problem in imitation learning. It executes the current policy, obtains expert labels on states the policy actually visits, and adds those labeled states to the dataset. If $D_i$ is the dataset before iteration $i$ and $d_{\pi_i}$ is the state distribution of the current policy, the update is conceptually

$$D_{i+1}=D_i\cup\{(s,\pi^*(s)):s\sim d_{\pi_i}\},$$

where $\pi^*$ is the expert policy. For an agent, the expert might be a human, a stronger model, or a verified repair procedure. The method is valuable only when the resulting labels are trustworthy and the rollout environment is safe.

The symbol $\cup$ denotes set union. The notation $s\sim d_{\pi_i}$ means that state $s$ is sampled from the state distribution induced by policy $\pi_i$.

### Loss weights define the behavior being imitated

Suppose the dataset contains task families $k=1,\ldots,K$. Let $n_k$ be the number of supervised tokens from family $k$, and let $w_k\ge0$ be its explicit weight. The effective token share is

$$\alpha_k=\frac{w_kn_k}{\sum_{j=1}^{K}w_jn_j}.$$

The denominator sums weighted token counts across all families. Sampling an equal number of trajectories per family does not make $\alpha_k$ equal when trajectory lengths differ. Long computer-use traces can dominate short classification or formatting examples. Reporting only the number of examples conceals this mixture.

Weights can also distinguish token roles. A tool result may be included as context but assigned mask value zero, while the following assistant action receives positive weight. If observation tokens were treated as prediction targets, the model would spend capacity imitating the environment rather than choosing the next action. Conversely, masking every argument token would prevent it from learning tool parameters. The objective must state exactly which roles and spans contribute loss.

### Parameter-efficient adaptation changes capacity, not the objective

Low-Rank Adaptation, or [LoRA](https://arxiv.org/abs/2106.09685), freezes a pretrained weight matrix $W_0$ and learns a low-rank update

$$W=W_0+BA,$$

where $A\in\mathbb R^{r\times d}$, $B\in\mathbb R^{k\times r}$, and rank $r$ is much smaller than $d$ and $k$. The matrices $A$ and $B$ reduce the number of trainable parameters. They do not change the masked cross-entropy target. LoRA can lower memory and storage costs, but a rank that is too small can limit adaptation, and a poorly constructed dataset remains poor supervision regardless of parameter efficiency.

Selection should therefore report at least three objects separately: the training objective, the effective data mixture, and downstream agent performance. A lower held-out token loss can coexist with worse task completion if the validation set resembles teacher-forced demonstrations more closely than deployment trajectories.

[QLoRA](https://arxiv.org/abs/2305.14314) combines a frozen quantized base model with low-rank adapters to reduce memory requirements. Quantization changes the numerical substrate and adapter capacity. It does not repair a wrong loss mask or biased dataset. [Tulu 3](https://arxiv.org/abs/2411.15124) publishes a broader post-training recipe that includes SFT, preference optimization, reinforcement learning with verifiable rewards, data curation, and evaluation. Its openness is valuable because stage interactions can be inspected rather than inferred from a final checkpoint alone.

### Scope and limits of masked likelihood

Masked likelihood makes the supervised signal explicit. It does not capture optimizer dynamics, distributed failures, numerical precision, curriculum order, or model-specific chat-template details. Data rights, privacy, and contamination also constrain what may be trained.

A reproducible SFT report should state the base checkpoint, template, masks, token weights, datasets and licenses, example and token mixtures, sequence length, packing policy, optimizer, learning-rate schedule, hardware, held-out split, target harness, and outcome evaluation.

### Exercises

1. Compute the masked loss for target probabilities $(0.8,0.5,0.25,0.9)$ and mask $(0,1,1,0)$.
2. Two domains each contain 1,000 trajectories. Their mean selected lengths are 500 and 5,000 tokens. Compute sample and token shares.
3. A successful trajectory contains one destructive exploratory action followed by recovery. Compare whole-trajectory inclusion, exclusion, and step-level masking.
4. Design a split for claiming that an SFT coding agent generalizes to unseen repositories and unseen issue families.

### Solutions and discussion

1. The loss is $-[\log0.5+\log0.25]/2\approx1.040$.
2. Each domain has sample share $0.5$. Their token shares are $500/(500+5000)\approx0.091$ and $0.909$.
3. Inclusion teaches both the destructive action and recovery. Exclusion loses the useful recovery. Step-level masking can remove the destructive target while retaining later actions, but it requires a reliable label and may create a history the model did not itself produce.
4. Group by repository so no repository appears across splits. Within the test repositories, select issue families absent from training using a declared taxonomy. Deduplicate code and issue text across time and project forks. Report both repository and issue-family criteria.

The first exercise is a masked geometric mean in log space. Only probabilities $0.5$ and $0.25$ contribute. Their losses are approximately $0.693$ and $1.386$, whose mean is $1.040$. The zero-mask positions still provide preceding context if they occur before a selected token.

The second exercise separates sample balance from optimization balance. Both domains contribute half of sampled trajectories. The long domain contributes ten times as many selected tokens and therefore ten times the unweighted loss mass. Equalizing trajectory counts does not equalize gradient contribution.

For the third exercise, whole-trajectory inclusion preserves a coherent recovery history but reinforces the destructive step. Exclusion removes both the bad action and the useful recovery. Step masking retains the bad action as context without making it a target. A stronger alternative is to construct a corrected trajectory, but that introduces counterfactual history that must be validated for coherence.

The fourth split needs two independent grouping rules. Repository grouping tests transfer across codebases. Issue-family grouping tests transfer across defect types. Forks and near-duplicate projects belong to one repository family. A time boundary can add realism, but it does not replace content decontamination.

### Further reading

- [Agent Data Protocol](https://arxiv.org/abs/2510.24702): a typed interlingua for heterogeneous agent trajectories.
- [DAgger](https://proceedings.mlr.press/v15/ross11a.html): imitation learning on learner-visited states.
- [SWE-Gym](https://arxiv.org/abs/2412.21139): training environments and trajectories for software agents.
- [BalanceSFT](https://arxiv.org/abs/2505.20192): balanced training signals and data hardness in function-calling supervised fine-tuning.
- [Hugging Face chat templates](https://huggingface.co/docs/transformers/en/chat_templating): model-specific conversation serialization.
- [Transformer Reinforcement Learning (TRL) SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer): an implementation reference for supervised fine-tuning.
