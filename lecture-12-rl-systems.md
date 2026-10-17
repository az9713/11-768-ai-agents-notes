# Lecture 12 — Reinforcement-learning systems for agents

*Independent study chapter · Based on a CMU 11-768: AI Agents lecture by Apurva Gandhi, 1 October 2026 · [Course and attribution](index.html#sources)*

## 0. One policy-gradient step runs on two engines [00:00](https://www.youtube.com/watch?v=dwR6NP9nI-8&t=0s)

A laboratory trains an 8-billion-parameter coding agent with reinforcement learning (RL) on repository-repair tasks. Each step takes 8 tasks and samples $n=8$ rollouts per task. A **rollout** is one sampled trajectory, as in chapter 9. A step therefore has 64 trajectories, each capped at 32,768 tokens. A typical trajectory has a 2,000-token task prompt and four turns of a 200-token action and a 1,500-token tool observation, $2{,}000+4\cdot1{,}700=8{,}800$ tokens. The consequential error is quiet: the trainer computes a gradient for tokens or contexts that the sampled policy never used. The update is then biased, and the run can stall or collapse after many graphics processing unit (GPU) hours.

Call this the **repair run**, the running case of the chapter. Chapter 10 writes $n$ for rollouts per task; chapter 9 writes $G$, reserved here for chapter 1's gate $G_\phi$. With total return $R(\tau_i)$ attributed to every action, chapter 9's estimator with a baseline reads

$$\widehat g_b=\frac1N\sum_{i=1}^N\sum_t\big(R(\tau_i)-b(h_{i,t})\big)\nabla_\theta\log\pi_\theta(a_{i,t}\mid h_{i,t}).$$

For the repair run, $N=64$. The baseline $b(h_t)$ is chapter 9's, not chapter 1's budget $b_t$. Each factor names an operation. A **sample** operation generates $\tau_i$ through the agent harness. A **verify** operation computes $R(\tau_i)$ from tests or tool outcomes. A **forward** pass computes $\log\pi_\theta$, a **backward** pass its gradient, and an **optimizer step** changes $\theta$. The last three map onto ordinary PyTorch calls, so one training loop could implement everything.

Production frameworks split the work, because sampling and training stress different resources (section 1). An **inference engine**, such as vLLM or SGLang, runs sample. A **trainer engine**, such as Megatron or Fully Sharded Data Parallel version 2 (FSDP2), runs forward and backward. A **weight sync** copies new parameters to the inference engine. Rollout data, tokens and their logged probabilities, flows back.

### A training step has five stages and one hidden quantity

(1) The inference engine samples trajectories while the harness executes actions. (2) The environment verifies them. (3) The trainer recomputes token log probabilities with gradient tracking. (4) The backward pass computes $\widehat g_b$. (5) The optimizer updates $\theta$. The weights go to the inference engine and are **resharded**, split again to match how that engine divides the model across GPUs (section 3).

Let $\mu(\cdot\mid x,\tau_{i,<j})$ be the distribution from which the inference engine actually drew token $\tau_{i,j}$, with $x$ the task and $\tau_{i,<j}$ the earlier tokens. If the engine samples at temperature 1 without top-$k$ or top-$p$ truncation, and both engines compute exactly, then $\mu=\pi_{\theta_{\rm old}}$, the policy before the update. Temperature or truncation changes $\mu$ by design (chapter 9, section 1); finite precision changes it by accident (section 1 below). Either way $\mu$ is hidden unless the inference engine logs it. Sections 2 and 3 make each engine cheaper. Sections 4 to 6 keep $\mu$ observed and close to the trainer's distribution, and section 7 moves its record behind a service boundary.

## 1. Sampling and training are different workloads [02:54](https://www.youtube.com/watch?v=dwR6NP9nI-8&t=174s)

**Decoding** generates one token per model call, because each token conditions on the previous one. **Prefill** processes known tokens in one parallel pass (chapter 3). A trainer forward pass resembles prefill, because the whole trajectory is known.

Let a model have $N_{\rm par}$ parameters in the 2-byte **bfloat16 (bf16)** format: 1 sign bit, 8 exponent bits, and 7 stored fraction (mantissa) bits. One decoding step for $b_{\rm dec}$ concurrent sequences reads $2N_{\rm par}$ weight bytes and performs about $2N_{\rm par}b_{\rm dec}$ operations. The **arithmetic intensity** of a computation is operations per byte read: about $b_{\rm dec}$ here, and about $\ell$ for a prefill of $\ell$ tokens. A GPU with peak rate $P_{\rm peak}$ operations per second and memory bandwidth $\beta_{\rm mem}$ bytes per second is bandwidth-bound below intensity $P_{\rm peak}/\beta_{\rm mem}$. [NVIDIA lists the H100 SXM](https://www.nvidia.com/en-us/data-center/h100/) at 1,979 bf16 teraFLOPS (trillion floating-point operations per second) with sparsity and 3.35 terabytes per second (TB/s). Section 3's ZoRRo post takes half this rate, 989 teraFLOPS, as the dense bf16 peak of the H200, which NVIDIA lists at the same rate. The ratio is then about 295. The repair run decodes at most 64 sequences at once, so decoding is bandwidth-bound, while an 8,800-token trainer pass is arithmetic-bound. This count ignores attention and caches.

Inference needs no autograd graph, so it spends memory on the key/value (KV) cache. Training must keep activations, or recompute them. Separate engines optimize each constraint, at the price of two implementations.

### Two engines compute two probabilities

A **kernel** is one compiled GPU routine, such as a matrix product. Kernels in two engines can sum numbers in different orders, and floating-point addition is not associative: in 64-bit arithmetic, $(0.1+0.2)+0.3$ gives $0.6000000000000001$ and $0.1+(0.2+0.3)$ gives $0.6$. [A Thinking Machines study of nondeterminism in large language model (LLM) inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) traces one source. Many inference kernels lack **batch invariance**: a request's output depends on how many requests share its batch. [A 2025 study of the training-inference mismatch](https://arxiv.org/abs/2510.26788) locates the root cause in bf16 rounding. It reports that 16-bit floating point (FP16), with 10 fraction bits, effectively eliminates the mismatch. FP16 pays with range: its 5 exponent bits cap values near $6.6\times10^4$.

So $\pi_{\theta_{\rm old}}$ and $\mu$ differ even with identical weights. An action $a_{i,t}$ is a run of tokens, so $\log\pi_\theta(a_{i,t}\mid h_{i,t})$ is the sum of its tokens' log probabilities, which the trainer computes. Chapter 9, section 7, corrects data from a behavior policy $\mu$ with an importance ratio. At token level, define

$$w_{i,j}=\frac{\pi_{\theta_{\rm old}}(\tau_{i,j}\mid x,\tau_{i,<j})}{\mu(\tau_{i,j}\mid x,\tau_{i,<j})}.$$

Fix one position and its prefix, and write $\pi$ for $\pi_{\theta_{\rm old}}$ there. For any function $f$ of the token, $\mathbb E_\mu[wf]=\sum_a\mu(a)\frac{\pi(a)}{\mu(a)}f(a)=\mathbb E_\pi[f]$, provided $\mu(a)>0$ wherever $\pi(a)>0$. The identity corrects one token's conditional law, not the prefix, which $\mu$ also generated. Chapter 9's trajectory ratio $w(\tau)$ multiplies all token ratios, and its variance grows with length (chapter 11, section 3). The token form is therefore biased across a trajectory. The bias is small when the token distributions of $\mu$ and $\pi_{\theta_{\rm old}}$ are close at every position. Agreement on most tokens is not enough, because one early mismatch changes every later prefix.

Chapter 10's ratio $r_{i,j}=\pi_\theta/\pi_{\theta_{\rm old}}$ measures the update, and $w_{i,j}$ measures the engine mismatch. Their product, $\pi_\theta/\mu$, is the token form of chapter 9's $r_t(\theta)$. Separate factors let the clip act on the update and a cap act on the mismatch. **Truncated importance sampling** replaces $w_{i,j}$ by $\min(w_{i,j},w_{\max})$ for a cap $w_{\max}$. The FP16 study's Eq. (9), which credits the method to [IMPALA](https://arxiv.org/abs/1802.01561) and Yao et al. (2025), weights each token's gradient as

$$\min(w_{i,j},w_{\max})\,\nabla_\theta\min\big(r_{i,j}\hat A_i,\ \operatorname{clip}(r_{i,j},1-\epsilon,1+\epsilon)\hat A_i\big).$$

IMPALA's V-trace target uses the same truncated weights. No mismatch weight exceeds $w_{\max}$. The cap does not bound a token's whole term: for $\hat A_i<0$ and $r_{i,j}>1+\epsilon$, the minimum keeps the unclipped $r_{i,j}\hat A_i$, so a large ratio still gives a large term. The bias has an exact form at one position:

$$\mathbb E_\pi[f]-\mathbb E_\mu[\min(w,w_{\max})f]=\sum_{a:\,w(a)>w_{\max}}\big(\pi(a)-w_{\max}\mu(a)\big)f(a).$$

Only tokens above the cap contribute. If $\mu=0.50$ and the trainer gives $0.55$, then $w=1.1$, an ordinary correction. If $\mu=0.01$ and the trainer gives $0.20$, then $w=20$. With $w_{\max}=2$, its weight falls to 2, and its bias term is $(0.20-2\cdot0.01)f=0.18f$. The correction needs $\mu$ for every sampled token, so the inference engine must log it; recomputing in the trainer returns $\pi_{\theta_{\rm old}}$.

## 2. Inference throughput comes from reuse and scheduling [10:16](https://www.youtube.com/watch?v=dwR6NP9nI-8&t=616s)

Chapter 3 gives the KV-cache size $B_{KV}=2Lnh_{KV}d_kb$ for $L$ layers, $n$ stored tokens, $h_{KV}$ KV heads, head width $d_k$, and $b$ bytes per entry. Here $n$ is not the group size, and $b$ is not the baseline. [The Llama 3 report](https://arxiv.org/abs/2407.21783) gives the 8B model 32 layers, 8 KV heads, and width 4,096 over 32 attention heads, so $d_k=128$. In bf16, $b=2$.

```python
from agent_lab.context import kv_cache_bytes

per_token = kv_cache_bytes(32, 1, 8, 128, 2)
assert per_token == 131_072
assert kv_cache_bytes(32, 32_768, 8, 128, 2) == 4 * 2**30
assert 64 * kv_cache_bytes(32, 32_768, 8, 128, 2) == 256 * 2**30
```

One token costs 131,072 bytes, 128 kibibytes (KiB). A 32,768-token trajectory needs exactly 4 gibibytes (GiB), and 64 need 256 GiB, before allocator overhead. The repair run's 8,800-token trajectories need exactly 68.75 GiB (73.8 gigabytes, GB).

[ZeRO](https://arxiv.org/abs/1910.02054) counts 16 bytes per parameter for mixed-precision Adam training. Weights and gradients take 2 bytes each (fp16 in the paper; bf16 has the same size). The other 12 bytes are 32-bit floating point (FP32) state: a master weight copy plus Adam's momentum and variance. For 8 billion parameters that is 128 GB. On eight 80 GB accelerators, an assumed configuration, trainer state, a full 256 GiB cache (274.9 GB), and 16 GB of inference weights leave about 221 GB of 640 GB for activations and buffers. The realized cache leaves about 422 GB. Larger models and batches remove this margin.

**Prefix caching** stores KV states for a prefix and reuses them in later requests that share it. In agentic RL each turn is a separate request, and a group shares the task prompt. Chapter 3's `submitted_input_tokens(4, 2000, 1700)` counts 18,200 prefill tokens per rollout without a cache, 145,600 for the group. With every prefix cached, the group computes the prompt once and each rollout three observations: $2{,}000+8\cdot3\cdot1{,}500=38{,}000$ tokens, 3.8 times fewer. The fourth observation follows the last request, and action tokens already have KV states from decoding. SGLang's RadixAttention (chapter 3) and [vLLM's Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/) implement this reuse. It needs resubmitted tokens to equal the cached ones exactly (section 5). When a harness edits earlier history, the saving stops at the first edited token.

### Continuous batching removes the wait for the longest request

**Static batching** frees a batch's slots only when its longest request finishes. **Continuous batching**, the iteration-level scheduling of [Orca](https://www.usenix.org/conference/osdi22/presentation/yu), reschedules after every decoding step, so a queued request takes a freed slot at once. Take eight rollouts of 10, 120, 30, 40, 20, 30, 60, and 10 seconds on four slots.

```python
from agent_lab.training import batch_makespans

durations = [10, 120, 30, 40, 20, 30, 60, 10]
static, continuous = batch_makespans(durations, slots=4)
assert (static, continuous) == (180.0, 120.0)
assert continuous >= max(max(durations), sum(durations) / 4)
```

Static batching takes $120+60=180$ seconds; continuous batching takes 120, the straggler's length. No schedule beats $\max(\max_id_i,\sum_id_i/N_{\rm slot})$ for durations $d_i$ on $N_{\rm slot}$ slots. For an upper bound, let request $j$ finish last and start at $s_j$. Every slot was busy until $s_j$, so $s_j\le(\sum_id_i-d_j)/N_{\rm slot}$, and completion is at most $\sum_id_i/N_{\rm slot}+(1-1/N_{\rm slot})\max_id_i$, here 170 seconds. With $B$ equal requests of duration $d_r$ on $m=N_{\rm slot}$ slots, refilling finishes at $\lceil B/m\rceil d_r$, chapter 9's $Bd_r/m$ rounded up to whole waves. The model assumes slot speed does not depend on batch size, which real decoding violates.

A long prompt prefilled in one pass stalls every running decode. **Chunked prefill**, as in [Sarathi-Serve](https://arxiv.org/abs/2403.02310), interleaves its pieces with decode steps. **Disaggregated inference**, as in [DistServe](https://arxiv.org/abs/2401.09670), runs prefill and decode on separate GPU pools and transfers the KV cache. Each paper reports gains only for its own models and latency targets. Separate pools also allow hardware suited to each phase, bandwidth for decoding and arithmetic for prefill, a benefit no source here measures.

### Speculative decoding preserves the distribution but loses speed as the policy drifts

**Speculative decoding** lets a cheap **drafter**, a small model or a **multi-token prediction (MTP) head**, propose $n_{\rm draft}$ tokens that the policy scores in one pass. At one position, let $\pi(a)$ and $q_{\rm d}(a)$ be the policy's and drafter's probabilities. [Leviathan et al.](https://arxiv.org/abs/2211.17192) accept a drafted $a$ with probability $\min(1,\pi(a)/q_{\rm d}(a))$ and on rejection sample from $\max(0,\pi-q_{\rm d})$ normalized. The output law is exactly $\pi$:

$$\Pr(a)=q_{\rm d}(a)\min\Big(1,\frac{\pi(a)}{q_{\rm d}(a)}\Big)+\Big(1-\sum_{a'}\min(\pi,q_{\rm d})(a')\Big)\frac{\max(0,\pi(a)-q_{\rm d}(a))}{\sum_{a'}\max(0,\pi-q_{\rm d})(a')}.$$

The first term is $\min(\pi(a),q_{\rm d}(a))$. Since $\sum\min(\pi,q_{\rm d})+\sum\max(0,\pi-q_{\rm d})=1$, the second is $\max(0,\pi(a)-q_{\rm d}(a))$, and the sum is $\pi(a)$. The acceptance probability is $\alpha_{\rm acc}=\sum_a\min(\pi,q_{\rm d})(a)$. The **total-variation distance** $\tfrac12\sum_a|\pi(a)-q_{\rm d}(a)|$ equals $\sum_a\max(0,\pi-q_{\rm d})(a)$ (exercise 2), so $\alpha_{\rm acc}$ is one minus it. Under independent acceptances, the paper's Eq. (1) gives $(1-\alpha_{\rm acc}^{n_{\rm draft}+1})/(1-\alpha_{\rm acc})$ tokens per policy pass: about 3.36 at $\alpha_{\rm acc}=0.8$ and 1.94 at $0.5$, with $n_{\rm draft}=4$. When $\pi=q_{\rm d}$, every draft is accepted, the rejection branch never runs, and the count is its limit $n_{\rm draft}+1=5$. The count ignores the drafter's own cost.

Training moves $\pi$ while a frozen $q_{\rm d}$ stays fixed, so nothing keeps the distance small, and $\alpha_{\rm acc}$ falls as the policy drifts. Correctness survives; speed does not. One remedy trains the drafter online, for example by distillation from the current policy.

## 3. The trainer spends computation to save memory [21:00](https://www.youtube.com/watch?v=dwR6NP9nI-8&t=1260s)

**Activation checkpointing** stores only each layer's input in the forward pass and recomputes the layer's activations when the backward pass reaches it. Memory falls to one input per layer plus one layer's activations. For each weight matrix, the backward pass computes two products, the gradients for input and weight, so it costs about twice the forward cost $F_{\rm fwd}$. A step costs $3F_{\rm fwd}$, and full recomputation adds one $F_{\rm fwd}$: about 33% more arithmetic. [The Megatron-LM paper](https://arxiv.org/abs/2104.04473) uses the same 2:1 simplification. [Chen et al.](https://arxiv.org/abs/1604.06174) measured 30% extra running time for a 1,000-layer residual network. **Selective recomputation**, from [Korthikanti et al.](https://arxiv.org/abs/2205.05198), recomputes only memory-heavy attention parts; with sequence parallelism it cuts the recomputation overhead by over 90%.

### Shared prefixes can be computed once in training

**Prefix sharing** places one prompt followed by all its responses in one sequence. A **tree attention mask** blocks attention between responses, and each response keeps its standalone position indices. Each response token then attends to exactly what it saw standalone, so its logits are unchanged, and the prompt's gradient is the sum of the branch gradients. Snowflake's [ZoRRo post](https://www.snowflake.com/en/blog/engineering/zorro-enterprise-rl-training/) implements the same sharing with split attention: causal self-attention runs once per unique prompt, then each response attends to the prompt and itself. It gives the idealized work ratio for $n$ rollouts with prompt length $\ell_{\rm p}$ and response length $\ell_{\rm r}$:

$$\frac{n(\ell_{\rm p}+\ell_{\rm r})}{\ell_{\rm p}+n\,\ell_{\rm r}}.$$

The duplicated share of tokens is $(n-1)\ell_{\rm p}/\big(n(\ell_{\rm p}+\ell_{\rm r})\big)$. The post reports about 6 times faster actor updates on a text-to-SQL (Structured Query Language) workload with 16 rollouts per prompt. It gives average prompt and response lengths of 10.1K and 800 tokens. With both figure notes, the formula gives a ratio of 7.6 and a duplicated share of 87%.

For the repair run, with a 2,000-token prompt and 6,800-token responses, the ratio is only $70{,}400/56{,}400\approx1.25$, because long tool observations dominate. Multi-turn trajectories also share prefixes across turns, which a prefix tree captures. On $\tau^2$-bench rollouts, [AReaL-DTA](https://arxiv.org/abs/2602.00482) reports a full-tree compression of 9.43 times. Its depth-first traversal runs forward and backward along one branch, releases its activations, then moves on, so peak activation memory is one root-to-leaf path. It reports up to 8.31 times the throughput of dense training and 1.70 times that of sparse tree-mask training.

### Parallelism divides memory and adds communication

A **rank** is one process of a distributed job, usually driving one GPU; a **node** is one machine, whose GPUs share a fast interconnect. **Data parallelism** keeps a replica on each rank and splits the batch into **microbatches** by a token budget. A 40,000-token budget holds four repair trajectories, so 64 trajectories form 16 microbatches. FSDP and ZeRO shard parameters, gradients, and optimizer state across data-parallel ranks.

**Pipeline parallelism** places consecutive layers on different ranks, called stages, and streams microbatches through them. Let each microbatch take time $t$ per stage. Stage $s$ starts at $(s-1)t$, so the last of $n_{\rm stage}$ stages finishes $n_{\rm micro}$ microbatches at $(n_{\rm stage}-1+n_{\rm micro})t$. Each stage is busy for $n_{\rm micro}t$ and idle for $(n_{\rm stage}-1)t$. Megatron-LM reports idle over busy time, $(n_{\rm stage}-1)/n_{\rm micro}$; the idle share of wall-clock time is $(n_{\rm stage}-1)/(n_{\rm stage}-1+n_{\rm micro})$. Four stages and 16 microbatches give $3/16=18.75\%$ of busy time and $3/19\approx15.8\%$ of wall-clock time.

**Tensor parallelism** splits each matrix product across ranks and exchanges partial results in every layer for every token, so it usually stays within one node. **Expert parallelism** places the experts of a **Mixture-of-Experts (MoE)** layer, whose router sends each token to a few of many feed-forward blocks, on different ranks. **Context parallelism** splits sequence positions (chapter 3). NVIDIA's [parallelisms guide](https://docs.nvidia.com/nemo-framework/user-guide/25.02/nemotoolkit/features/parallelisms.html) describes all five, with figures for all but context parallelism. A common heuristic maximizes data parallelism and minimizes tensor-parallel communication.

## 4. Weight synchronization exploits bf16 rounding [29:05](https://www.youtube.com/watch?v=dwR6NP9nI-8&t=1745s)

A one-trillion-parameter model in bf16 occupies 2 TB. At an assumed 12.5 GB/s per link, one copy to an inference rank takes 160 seconds; the repair run's 16 GB takes about 1.3 seconds.

A bf16 number is the top 16 bits of an FP32 number. An implicit leading one gives 8 bits of precision, which is why some papers count 8 mantissa bits. Between consecutive powers of two bf16 has $2^7=128$ values, so the spacing near a value $v$ lies between $|v|/256$ and $|v|/128$. Round-to-nearest-even maps $v+\Delta$, for a weight increment $\Delta$, back to a bf16 value $v$ when $|\Delta|$ is under half the spacing. A value between grid points, such as an FP32 master weight, changes its bf16 copy when it crosses the nearest rounding boundary, which can be closer.

[PULSE](https://arxiv.org/abs/2602.03839) measures approximately 99% of per-step weight updates as invisible after the bf16 cast, across Qwen, Llama, and Gemma models of 0.5B to 7B parameters. Following [mixed-precision training](https://arxiv.org/abs/1710.03740), Adam updates FP32 master weights, and forward passes use a bf16 cast. [Adam's](https://arxiv.org/abs/1412.6980) effective step is usually at most about the learning rate. It can be larger, for example after long runs of zero gradient or while gradients grow steadily. PULSE gives typical RL learning rates near $3\times10^{-6}$.

```python
from agent_lab.training import bf16_round

assert bf16_round(1.0 + 1e-3) == 1.0
v = bf16_round(0.0177)
assert bf16_round(v + 3e-6) == v
master = 0.0177
steps = 0
while bf16_round(master) == v:
    master += 3e-6
    steps += 1
assert steps == 21
```

PULSE's median $|v|$ for Qwen2.5-1.5B is 0.0177, where the spacing is $2^{-13}\approx1.22\times10^{-4}$; a step of $3\times10^{-6}$ is 2.5% of it. With an FP32 master, 21 same-sign steps cross the boundary and change the bf16 copy: the update is delayed, not lost. With bf16 storage alone, each step would round back and the signal would be lost. If a master sits at a uniformly random point of its cell, one step changes the copy with probability about 2.5%. That estimate is the chapter's; it does not predict PULSE's rate, because Adam's steps vary. Rounding through FP32 first matters only near a bf16 tie, and an FP32 master also needs 21 steps.

Delta synchronization sends only changed values and their positions. At 1% of $10^{12}$ parameters, values take 20 GB. Store positions as gaps between changed indices, as PULSE does. If parameters change independently with probability 0.01, gaps are geometric with mean 100. A **variable-length integer** (varint) stores 7 bits per byte, so a gap of 128 or more, probability $0.99^{127}\approx0.279$, takes two bytes. Positions then average 1.28 bytes, 12.8 GB, for 32.8 GB in all; an entropy coder at 8.08 bits per gap gives 30.1 GB. The source lecture's estimate of "maybe 20 to 30 GB" lies just below this arithmetic, and no measurement covers that scale. For 7B models, PULSE's reference figure is 140 megabytes of patches against 14 GB of weights. Its measured Qwen2.5-7B deployment averages 108 megabytes per upload, about 130-fold less, and checksums confirm bit-identical reconstruction.

Do the same parameters change at every step? PULSE reports sparsity above 98% over windows of up to 8 steps after learning-rate warmup. Independent random 1% changes would alter $1-0.99^8\approx7.7\%$ in 8 steps. Either the same parameters change repeatedly, or values return to earlier bf16 values within the window. Both are inferences from PULSE's numbers; the paper tests neither.

### Routing replay removes a discrete source of mismatch

An MoE router selects the top few experts from scores, so a last-bit difference or a tie can select a different expert. [Rollout Routing Replay (R3)](https://arxiv.org/abs/2510.11370) measures that about 10% of routers choose different experts in training than in inference. In one MoE run with Group Relative Policy Optimization (GRPO), after 60 steps, more than 10% of tokens differed in probability by over a factor of two between engines. Collapsed runs, the authors report, almost always show such values. R3 records the inference engine's routing and replays it in the trainer's forward pass, giving about ten times fewer such tokens and stable training. When an MoE run's loss or reward stops improving, a first check is whether R3 is enabled.

## 5. Agent trajectories must remain token-exact prefixes [37:42](https://www.youtube.com/watch?v=dwR6NP9nI-8&t=2262s)

Let $c_k$ be the token sequence the policy sees at turn $k$, and $\Vert$ concatenation. A harness has the **sequence extension property** when $c_k\Vert a_k$ is a prefix of $c_{k+1}$ for every turn. [The Tinker tutorial on sequence extension](https://tinker-docs.thinkingmachines.ai/tutorials/advanced/sequence-extension/) merges such turns into one training datum. Under the property, one causal forward pass over the trajectory reproduces every turn's conditional probabilities. Chapter 8's loss mask sets $m=1$ on action tokens, and chapter 10's trajectory advantage $\hat A_i$ multiplies each of them.

Harness A appends every action and observation. Harness B keeps the task and, after every two observations, replaces earlier turns with a 300-token summary, treated as context a separate summarizer wrote. Harness A yields one 8,800-token datum, $800/8{,}800\approx9.1\%$ of it with loss. Harness B breaks the prefix at turn 3 and yields two data: $2{,}000+2\cdot1{,}700=5{,}400$ and $2{,}000+300+2\cdot1{,}700=5{,}700$ tokens. The trainer handles 11,100 positions, 26% more, and the repeated task tokens cost computation without loss. Production harnesses such as Codex, Claude Code, and OpenHands compact long contexts, so such harnesses must sometimes be trained as they are. Tree-based execution, as in AReaL-DTA, recovers the shared prompt.

### Chat templates can break the prefix silently

A chat template (chapter 8) renders messages to tokens. [No Token Left Behind](https://www.lmsys.org/blog/2026-05-13-no-token-left-behind/) describes a **cut-thinking boundary**: the default templates of reasoning models such as Qwen3 and Kimi K2 drop assistant reasoning before the last user message. The rendered context then loses earlier reasoning, so $c_k\Vert a_k$ is no longer a prefix of $c_{k+1}$.

Three responses exist. Splitting each turn into its own datum, with the context the policy saw, is on-policy but repeats computation. Concatenating the trajectory with reasoning kept, after a rollout under the default template, trains on contexts the policy never saw: the repair run's quiet error. Changing the template so the policy sees its earlier reasoning during the rollout restores the prefix. Frameworks therefore ship templates, called **renderers**, with options that keep reasoning across turns.

### Token-in, token-out keeps the sampled tokens

Early frameworks passed rollouts to the trainer as strings and retokenized them. **Token-in, token-out (TITO)** passes the sampled token identifiers instead. For a lossless tokenizer, encoding is one-to-one: decoding a string's canonical sequence returns the string. Encoding is not onto, and decoding is many-to-one, so the policy can sample a non-canonical sequence. The LMSYS example uses tokens `Hel`(3), `lo`(7), and `Hello`(4).

```python
from agent_lab.training import decode, greedy_encode

vocab = ("H", "e", "l", "Hel", "Hello", "o", "el", "lo")
sampled = (3, 7)
text = decode(sampled, vocab)
assert text == "Hello"
assert greedy_encode(text, vocab) == (4,)
assert greedy_encode(text, vocab) != sampled
```

After the round trip the trainer evaluates token 4, which the policy never sampled, while the logged $\mu$ belongs to tokens 3 and 7, so $w_{i,j}$ is undefined. An inserted space in a template causes the same failure. Greedy longest match stands in for real merge rules. LMSYS warns that such discrepancies "can eventually lead to erratic updates"; it gives no measurement.

## 6. Asynchronous RL trades staleness for utilization [47:10](https://www.youtube.com/watch?v=dwR6NP9nI-8&t=2830s)

**Synchronous RL** alternates phases, typically on shared GPUs: the trainer runs on all eight, sleeps, and the inference engine wakes. With rollout time $T_{\rm roll}$, update time $d_u$, and sync time $d_{\rm sync}$, a step takes $T_{\rm roll}+d_u+d_{\rm sync}$, extending chapter 9's bound $Bd_r/m+d_u$, in which $m$ counts parallel environments, not chapter 8's mask. The straggler sets $T_{\rm roll}$. In section 2's schedule, three slots finish at 50, 60, and 90 seconds and idle until 120, a utilization of $320/480\approx67\%$.

**Asynchronous RL** splits GPUs between engines and overlaps them, so finished rollouts feed the next update while stragglers continue. It needs continuous batching and in-flight weight updates. Let $W_{\rm r}$ and $W_{\rm u}$ be the GPU-seconds of rollout and update work per step, and let a fraction $f_{\rm inf}$ of $N_{\rm gpu}$ GPUs run inference. If work scales linearly with GPU count, the phases take equal time when $W_{\rm r}/(f_{\rm inf}N_{\rm gpu})=W_{\rm u}/((1-f_{\rm inf})N_{\rm gpu})$, so

$$f_{\rm inf}=\frac{W_{\rm r}}{W_{\rm r}+W_{\rm u}}.$$

A 3:1 inference-to-training split corresponds to $W_{\rm r}=3W_{\rm u}$; it is one practitioner's experience, not a benchmark.
The cost is **staleness**. Let a token's version lag be the number of optimizer steps between the weights that sampled it and the weights being updated. Synchronous RL with one optimizer step per rollout batch has lag zero. If a batch feeds several minibatch steps, the last trains on data lagged by the earlier steps. With one step of overlap, repair-run tokens have lag 0 or 1, and a rollout spanning an in-flight update carries both. [AReaL](https://arxiv.org/abs/2505.24298) bounds staleness by balancing rollout and training workers. It uses a staleness-enhanced variant of Proximal Policy Optimization (PPO) and reports up to 2.77 times faster training than synchronous systems. [PipelineRL](https://arxiv.org/abs/2509.19128) reports about twice as fast learning with in-flight updates. For a stale token, section 1's ratio applies with $\mu$ from the sampling weights; each paper states its own objective.

### In-flight updates change weights in the middle of a rollout

An **in-flight weight update** pauses a running request, loads new weights, and continues. [SGLang's RL interface](https://docs.sglang.io/docs/advanced_features/sglang_for_rl) offers three pause modes. `abort` returns pending requests to the caller. `retract` requeues running requests so their KV cache can be recomputed under the new weights. `in_place` keeps request state, so generation continues on KV states from the old weights.

Under `in_place`, later tokens come from a mixture that matches neither policy. The rollout stays usable if the engine logs, per token, the probability it actually sampled from; that value is $\mu$ by definition. The prefix-level bias of section 1 remains, and it grows with the size of the mismatch at the mixed-weight tokens. SGLang can report per-token weight versions, so lag is defined per token.

In [Recursive Agent Optimization](https://arxiv.org/abs/2605.06639), whose harness resembles that of [Recursive Language Models](https://arxiv.org/abs/2512.24601), a root agent delegates subtasks to new instances of itself. A tree of sub-agent trajectories can enter the batch as one rollout or as separate ones, which changes how the outcome advantage is shared.

## 7. Service boundaries make the training loop replaceable [53:00](https://www.youtube.com/watch?v=dwR6NP9nI-8&t=3180s)

A **harness-agnostic proxy** sits between harness and inference engine and emulates the chat-completions and responses interfaces, the Hypertext Transfer Protocol (HTTP) formats through which harnesses call hosted models. It forwards requests and stores token identifiers, logged $\mu$, and routing decisions, so a new harness needs no framework change. [Agent Lightning](https://arxiv.org/abs/2508.03680) reports such decoupling with almost no harness code changes. Chapter 1's execution gate must still authorize actions during training rollouts.

**Single Program Multiple Data (SPMD)** runs one program on every rank and branches on the rank number. One program gives direct control, but every component shares one code base and release cycle. The source lecture reports that AReaL used SPMD before version 1.0.

A **single controller** instead runs a lightweight central processing unit (CPU) loop that calls services for inference, agents, training, weight sync, and environment reward. The [Open Reward Standard](https://openrewardstandard.io/) is an open HTTP-based protocol for connecting models to RL environments. The [Model Context Protocol](https://modelcontextprotocol.io/specification/2025-06-18) (MCP) connects harnesses to tools. The Tinker training interface exposes [`forward_backward`, `optim_step`, and sampling clients](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/trainingclient/), the split this chapter began with. According to the source lecture, Miles and SkyRL implement it on owned hardware. Services can join during a run, run on different vendors' hardware, and serve several policies. [AstraFlow](https://arxiv.org/abs/2605.15565) reports 77.65 seconds per iteration on a math solver-verifier workflow against 212.64 for a verl-based multi-agent pipeline, 2.7 times less. Network latency, version consistency, and retries are costs no source here measures.

### Online and multi-tenant training reuse the same services

A deployed harness can supply rollouts. Cursor reports [online RL for Tab](https://cursor.com/blog/tab-rl) at 1.5 to 2 hours per checkpoint cycle. For [Composer](https://cursor.com/blog/real-time-rl-for-composer) it reports a checkpoint behind its Auto setting "as often as every five hours". User feedback is then the reward, with chapter 9's identification problem.

**Low-Rank Adaptation (LoRA)** adds a trained product $B_{\rm L}A_{\rm L}$ to a frozen $d_{\rm out}\times d_{\rm in}$ matrix, with rank $r_{\rm L}$, $B_{\rm L}\in\mathbb R^{d_{\rm out}\times r_{\rm L}}$, and $A_{\rm L}\in\mathbb R^{r_{\rm L}\times d_{\rm in}}$, so it trains $r_{\rm L}(d_{\rm out}+d_{\rm in})$ parameters per matrix ([LoRA](https://arxiv.org/abs/2106.09685)). A batched kernel such as Punica's Segmented Gather Matrix-Vector Multiplication (SGMV), or a grouped general matrix-matrix multiplication (GEMM), runs many adapters in one pass; [Punica](https://arxiv.org/abs/2310.18547) reports 12 times the serving throughput. [Trajectory's multi-LoRA study](https://www.trajectory.ai/field-notes/multi-lora-training-for-continual-learning) ran eight Qwen3-4B experiments on one H200 node with four inference and four training GPUs. Eight concurrent runs finished in 5,433 seconds against 15,244 serially, 2.81 times faster, but the first run finished 1.97 times later and step time rose 2.62 times.

Eight debugging signals cover these failure modes. $\max w_{i,j}$ (section 1) detects engine mismatch. Average staleness is the mean version lag. Stall time is the time the trainer waits for rollouts. Model FLOPs utilization (MFU) divides the arithmetic the model needs per second by the summed $P_{\rm peak}$ of the GPUs that do it. Training reward against held-out evaluation separates learning from fitting the training tasks. pass@$k$ against pass@1 (chapter 6) separates what the policy can reach from what it reaches reliably. The **entropy** $-\sum_a\pi_\theta(a)\log\pi_\theta(a)$ falls as the policy concentrates. The **gradient norm** is the length of $\widehat g_b$ before clipping.

### Scope and limits of the systems analysis

Three results are exact under stated assumptions. Importance weighting is exact per token, given its prefix and support. Speculative rejection sampling outputs $\pi$. Sequence extension makes one forward pass equal to per-turn passes. The cost formulas assume no allocator overhead, a 2:1 backward-to-forward ratio, equal-speed slots, and linear scaling. Throughput figures come from single papers or vendor posts. The places where $\mu$ and $\pi_{\theta_{\rm old}}$ separate are kernels, routing, templates, and retokenization. A systems claim should report hardware, batch composition, rollout lengths, precision, and how $\mu$ was logged.

### Exercises

1. A model has 48 layers, 8 KV heads, head width 128, and bf16 entries. Compute the KV bytes per token and the number of 32,768-token rollouts that fit in 320 GB of cache.
2. Prove that speculative rejection sampling outputs $\pi$, and that its acceptance probability is one minus the total-variation distance between $\pi$ and $q_{\rm d}$.
3. For the repair run, count causal query-key pairs, $\ell(\ell+1)/2$ for length $\ell$, under harnesses A and B. Explain why compaction can lower attention cost yet raise total training cost.
4. Weights 0.0031 and 0.0177 have FP32 masters and bf16 copies, and Adam adds $3\times10^{-6}$ per step. How many same-sign steps change each copy? With masters at uniformly random points of their cells, what share of steps changes each copy? What happens with bf16 storage alone?
5. Design the per-token log for an asynchronous run with `in_place` updates and R3. State what each field protects against.

### Solutions and discussion

1. A token needs $2\cdot48\cdot8\cdot128\cdot2=196{,}608$ bytes, and a rollout $196{,}608\cdot32{,}768=6\cdot2^{30}\approx6.44\times10^9$. Then $320\times10^9/6.44\times10^9\approx49.7$, so 49 fit.
2. Proposing and accepting $a$ has probability $q_{\rm d}(a)\min(1,\pi(a)/q_{\rm d}(a))=\min(\pi,q_{\rm d})(a)$. Rejection has probability $1-\sum\min(\pi,q_{\rm d})=\sum\max(0,\pi-q_{\rm d})$, which cancels the normalizer, leaving $\max(0,\pi-q_{\rm d})(a)$. If $\pi=q_{\rm d}$, rejection has probability zero and the branch is never used. Since $\min(x,y)+\max(0,x-y)=x$, the masses sum to $\pi(a)$. Acceptance is $1-\sum_a\max(0,\pi-q_{\rm d})(a)$; because $\sum(\pi-q_{\rm d})=0$, positive and negative parts are equal, so this sum is half the $L_1$ distance.
3. A has $8{,}800\cdot8{,}801/2=38{,}724{,}400$ pairs and B $5{,}400\cdot5{,}401/2+5{,}700\cdot5{,}701/2=30{,}830{,}550$, about 20% fewer, but B has 26% more positions. Projections and feed-forward layers cost about $2N_{\rm par}=1.6\times10^{10}$ forward operations per position. Per pair, attention computes a query-key product and a weighted value sum, $4\cdot4{,}096$ operations per layer, 524,288 over 32 layers. A costs $8{,}800\cdot1.6\times10^{10}+38{,}724{,}400\cdot524{,}288\approx1.61\times10^{14}$, 12.6% attention; B costs $\approx1.94\times10^{14}$, 8.3% attention, about 20% more. Backward scales both terms alike. Per-position cost exceeds mean attention per position, $524{,}288(\ell+1)/2$, until $\ell\approx61{,}000$.
4. The weight 0.0031 has spacing $2^{-16}\approx1.53\times10^{-5}$, grid point $203\cdot2^{-16}\approx0.0030975$, and a boundary $5.16\times10^{-6}$ above it: two steps. The weight 0.0177 has spacing $2^{-13}$ and a boundary $6.12\times10^{-5}$ above it: 21 steps, as section 4's code checks. The share of changing steps is about step over spacing, 19.7% and 2.5%. With bf16 storage alone, each sum rounds back and neither weight moves.
5. Log the token identifier, the sampled probability $\mu$, the weight version, and the MoE routing. Identifiers prevent retokenization errors; logged $\mu$ gives $w_{i,j}$ its true denominator under mixed versions; versions measure staleness; routing lets the trainer replay experts.

The first answer shows linear scaling in layers and the GB-versus-GiB effect.

The second answer shows why speculative decoding needs no correction in RL; only its speed depends on $\alpha_{\rm acc}$.

The third answer separates two cost terms that the 26% figure merges. For this model, attention per position overtakes the per-position linear cost only above about 61,000 tokens. That length is not where B becomes cheaper than A; that point depends on the lengths in both harnesses. Compaction still shortens the KV cache that decoding reads.

The fourth answer shows that the FP32 master keeps small updates, and that small weights change their bf16 copies far more often than large ones.

The fifth answer makes $\mu$ observed, which every correction here needs.

### Further reading

- [PULSE](https://arxiv.org/abs/2602.03839): bf16 update absorption and lossless sparse weight synchronization.
- [Rollout Routing Replay](https://arxiv.org/abs/2510.11370): training–inference routing mismatch in MoE RL and its replay fix.
- [No Token Left Behind](https://www.lmsys.org/blog/2026-05-13-no-token-left-behind/): token-in, token-out, cut-thinking boundaries, and prefix checks.
- [AReaL](https://arxiv.org/abs/2505.24298): a fully asynchronous RL system with staleness control.
