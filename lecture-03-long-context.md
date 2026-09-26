# Lecture 3 — Long context modeling for agents

*Independent study chapter · Based on a CMU 11-768: AI Agents lecture by Graham Neubig, 1 September 2026 · [Course and attribution](index.html#sources)*

## 0. The growth of agent context [01:00](https://www.youtube.com/watch?v=AiwCCvFW1uE&t=60s)

An **agent** repeatedly chooses actions to pursue a goal. Consider a coding agent repairing a release failure over dozens of tool calls. An early project instruction prohibits publication; later calls produce long build logs and test results. If the instruction disappears during history compression, the agent may publish a build even after correctly repairing it.

A **token** is a discrete text unit. A **language model** predicts the next token from preceding tokens. A **tool** lets an agent request external computation or action. An **observation** is information returned from the environment after an action. The agent can incorporate that observation into a later **prompt**, the input sequence supplied to the model. Each step can lengthen the prompt's **context**, the information the model receives for that call. Longer context creates a capacity question: can the model use the relevant evidence within it? It also creates an efficiency question: can the system afford to process and store the sequence?

Suppose a fixed prompt prefix has $m\ge0$ tokens and each step adds $r>0$ history tokens. Index model calls by $t=1,\ldots,T$, where $T$ is the total number of calls. At call $t$, prompt length is $n_t=m+(t-1)r$. The notation $\Theta(T^2)$ means growth bounded above and below by constant multiples of $T^2$ for sufficiently large $T$, and $\sum_{t=1}^{T}$ adds one term for each call. If each call resends the whole history, total submitted input tokens are

$$\sum_{t=1}^{T}n_t=mT+\frac{rT(T-1)}2=\Theta(T^2)\quad\text{for fixed }m\ge0,\ r>0.$$

The fixed prefix contributes $m$ tokens on each of the $T$ calls, giving $mT$. The history contributes $0,r,2r,\ldots,(T-1)r$ tokens across those calls. Summing that arithmetic sequence gives $rT(T-1)/2$, the quadratic term.

The equation counts submitted tokens across a growing **trace**, the recorded sequence of actions and observations. It does not measure the computation performed inside each model call. A **cache** stores earlier computation for possible reuse. It may reduce repeated input processing or billed cost. The model still has to select and use information from the growing context.

The advertised maximum prompt length $N_{\max}$ tells us whether an input fits. It does not tell us how well the model can use facts at different positions in that input. Let $\operatorname{Recall}(\ell,j,k)$ be **effective recall**: the probability, under a specified evaluation distribution over tasks and model runs, that the model uses a relevant fact correctly when the prompt contains $\ell$ tokens, the fact is at token position $j$, and $k$ irrelevant **distractors** compete for attention. A context system should measure this probability across the positions it intends to use. The [Lost in the Middle study](https://aclanthology.org/2024.tacl-1.9/) found position-sensitive performance in its evaluated long-context tasks. Maximum input length therefore does not imply uniform reasoning quality.

### Context assembly is a constrained decision problem

Let $h_t$ be the complete event history at step $t$, $g$ the user's goal, $m_t$ stored memory, $\mathcal T_t$ the available-tool set, and $b_t$ the remaining resource budget. A **harness** is the software controlling the agent's model/tool loop. A **context builder** $C_\phi$, configured by harness parameters $\phi$, constructs the model-visible context $c_t=C_\phi(g,h_t,m_t,\mathcal T_t,b_t)$.

For $n$ candidate context items $e_1,\ldots,e_n$, let $\ell_i$ be item $i$'s token length, $v_i$ its estimated value for the current goal and history, and $z_i\in\{0,1\}$ its inclusion decision, with one meaning include and zero meaning omit. Let $B_t$ be the usable prompt-token budget after fixed instructions and tool definitions. The operator $\max_{z_1,\ldots,z_n}$ chooses inclusion decisions with the largest total estimated value. The symbol $\sum_i$ adds over the candidate items. A simple approximation chooses items by

$$\max_{z_1,\ldots,z_n}\sum_i v_i z_i\quad\text{subject to}\quad\sum_i\ell_i z_i\le B_t.$$

This is a **knapsack model**: select valuable items under a size limit. It is incomplete because information can be complementary or contradictory. A test failure may be useful only when the relevant code is also available. An outdated instruction can even have negative value if the agent follows it.

For a candidate context $c$, let $|c|$ be its token length, $P(\text{success}\mid c)$ the task-success probability under a specified evaluation distribution when $c$ is supplied, $\operatorname{cost}(c)$ its resource cost, and $\lambda\ge0$ the cost weight. Define $U(c)=P(\text{success}\mid c)-\lambda\operatorname{cost}(c)$ and $c^*=\arg\max_{|c|\le B_t}U(c)$, where $\arg\max$ selects a maximizing context. The true utility $U$ is unknown to the context builder. It must use approximate retrieval and then measure how well the agent continues with the selected context. Token count alone does not establish that the selection improved task outcomes at acceptable cost.

The [companion selector](https://github.com/az9713/11-768-ai-agents-notes/blob/main/agent_lab/context.py) solves the additive, integer-cost form exactly for a small budget. `Candidate.tokens` represents $\ell_i$, `Candidate.value` represents $v_i$, and `budget` represents $B_t$. `best[c]` stores the best value and selected names with capacity $c$. Descending capacity prevents one item from being used twice. For three items with sizes $(6,3,3)$ and values $(8,5,5)$, the code selects the two smaller items under a six-unit budget; a greedy choice of the largest individual value would be worse.

```python
def select_items(candidates, budget):
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    best = [(0.0, ()) for _ in range(budget + 1)]
    for item in candidates:
        if item.tokens < 1:
            raise ValueError("each candidate must cost at least one token")
        for capacity in range(budget, item.tokens - 1, -1):
            old_value, old_names = best[capacity - item.tokens]
            new_value = old_value + item.value
            if new_value > best[capacity][0]:
                best[capacity] = (
                    new_value, old_names + (item.name,)
                )
    return best[budget][1]
```

The additive value assumption and integer budget are deliberate simplifications. This selector is used for eligible memory items in Chapter 4. The current harness does not apply the knapsack rule to every tool schema or history event. Those inputs have separate inclusion policies, and hard constraints are retained without a value score.

For the release repair, the publication prohibition should be a non-optional constraint on permitted actions, not merely one candidate item assigned an uncertain relevance score $v_i$. The recent failure log may have high diagnostic value but can be retained outside the prompt and retrieved by file location. The exact failing test and the changed files deserve a compact, current representation. This division separates an **invariant**, a condition required to hold throughout the run, from evidence that can be selected according to the immediate subtask.

Context items differ in authority. Trusted user and developer instructions, project rules, **tool schemas** (machine-readable input forms), retrieved documents, and raw tool outputs should carry **provenance**, a record of their source, and be separated in the assembled message. A retrieved webpage can inform an answer but cannot alter the **permission gate**, the component deciding which proposed actions may execute.

Context items also have recurring token costs. If the same item is included in $c_t$ at every step, its token cost recurs on subsequent model calls. [Anthropic's context-engineering guidance](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) recommends concise persistent instructions and **just-in-time retrieval**, which loads details only when a task needs them. This policy can increase estimated value per token $v_i/\ell_i$ when the selector retrieves needed material. The full source remains available for later retrieval.

## 1. Prefill, decoding, and attention cost [06:35](https://www.youtube.com/watch?v=AiwCCvFW1uE&t=395s)

**Prefill** processes the input prompt and produces initial **key/value (KV) states**, the stored vectors later attention computations reuse. **Decoding** generates output tokens while using those states. **Time to first token** is the delay before any output appears. **Per-token latency** is the delay between later tokens. **Throughput** is the number of tokens processed or generated per unit time. These measures describe different aspects of **model serving**, the infrastructure that runs inference. An optimization can improve one measure while worsening another.

An **attention head** computes weighted combinations of token representations. Let $X\in\mathbb R^{n\times d}$ contain $n$ token vectors of width $d$. The notation $\mathbb R^{n\times d}$ denotes real-valued matrices with $n$ rows and $d$ columns. Let $d_k$ be the query/key width and $d_v$ the value width. Learned matrices $W_Q,W_K\in\mathbb R^{d\times d_k}$ and $W_V\in\mathbb R^{d\times d_v}$ project the token vectors into **queries** $Q=XW_Q$, **keys** $K=XW_K$, and **values** $V=XW_V$. A query scores which keys are relevant, then combines their corresponding values.

Let $K^\top$ be the matrix transpose of $K$, and let $M$ be an $n\times n$ mask with $M_{ij}=-\infty$ when token $j$ lies after token $i$ and zero otherwise. The row-wise **softmax** function turns each row of scores into nonnegative weights summing to one. **Causal attention**, which prevents a position from seeing future tokens, produces a matrix of weighted value vectors denoted $\operatorname{Attn}(Q,K,V)$:

$$\operatorname{Attn}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.$$

The notation $O(f(n))$ means growth bounded above by a constant multiple of $f(n)$ for sufficiently large $n$. The $n\times n$ score matrix costs $O(n^2d_k)$ arithmetic in a straightforward prefill implementation. With a KV cache, one new token compares its query with roughly $n$ cached keys, costing $O(nd_k)$ attention arithmetic for that token. This does not make a full long generation linear in total length. Actual wall time also depends on optimized computation **kernels**, memory transfer speed, **batching** (processing several requests together), parallelism, and the model's non-attention layers.

Let $B_{KV}$ be the KV-cache size in bytes, $L$ the number of model layers, and $n$ the number of stored tokens. Let $h_{KV}$ be the number of KV heads per layer and $b$ the bytes per numerical entry. Assume each stored value has the same width $d_k$ as a stored key. The factor two then counts both keys and values, giving the rough size $B_{KV}=2Ln h_{KV}d_k b$.

For $L=32$, $n=128{,}000$, $h_{KV}=8$, $d_k=128$, and $b=2$, this is $16{,}777{,}216{,}000$ bytes, or about $15.6$ **gibibytes (GiB)**, where one GiB is $2^{30}$ bytes, for one sequence before memory-allocation overhead. These parameter values are illustrative. Even when model weights fit in memory, KV states may limit the length or number of sequences a server can handle.

The size calculation is one multiplication in [context.py](https://github.com/az9713/11-768-ai-agents-notes/blob/main/agent_lab/context.py). It describes storage, not inference speed or a provider's billed token count.

```python
def kv_cache_bytes(
    layers, tokens, kv_heads, head_width, bytes_per_number
):
    values = (
        layers, tokens, kv_heads, head_width, bytes_per_number
    )
    if any(value < 0 for value in values):
        raise ValueError("dimensions must be nonnegative")
    return 2 * layers * tokens * kv_heads * head_width * bytes_per_number


assert kv_cache_bytes(32, 128_000, 8, 128, 2) == 16_777_216_000
```

## 2. Architectural mechanisms for long context [17:00](https://www.youtube.com/watch?v=AiwCCvFW1uE&t=1020s)

**Sliding-window attention** limits each token to nearby predecessors. **Recurrent attention** carries a fixed-size state forward. **Linear attention** factorizes a similarity calculation so it can use such a state. **Sparse attention** computes only selected token pairs. **KV compression** stores reduced key/value representations. **Hybrid architectures** combine local computation with occasional long-range access. Their tradeoffs differ: reducing comparisons, compressing past information, and reducing cache memory solve different problems. In the table, $w$ denotes the number of earlier tokens visible through a sliding window.

| Mechanism | State/access pattern | Typical tradeoff |
|---|---|---|
| Sliding-window attention | each token attends to the previous $w$ tokens | $O(nw)$ comparisons, but old detail falls outside the window |
| Linear/recurrent attention | fixed-size state summarizes past keys/values | cheap incremental updates, but compression may lose distinctions |
| Sparse/global attention | selected long-range links | depends on routing or pattern quality |
| KV compression | fewer/smaller stored key-value representations | saves serving memory, can lose fidelity |

Let $q,k\in\mathbb R^{d_k}$ be query and key vectors, $\varphi:\mathbb R^{d_k}\to\mathbb R^{r_f}$ map them to $r_f$ nonnegative features, and $\nu_i$ be the value vector at token $i$. A **kernelized linear-attention** form replaces softmax's exponential query-key similarity by the feature-vector inner product $\varphi(q)^\top\varphi(k)$. The new symbol $\nu_i$ distinguishes a value vector from the earlier $v_i$, the estimated value of a context item. For causal history through step $t$, let $S_t=\sum_{i\le t}\varphi(k_i)\nu_i^\top$ collect key-value products and $D_t=\sum_{i\le t}\varphi(k_i)$ collect key features. Let $y_t$ be the attention output at step $t$. Provided $\varphi(q_t)^\top D_t>0$, factorization gives

$$y_t=\frac{\varphi(q_t)^\top S_t}{\varphi(q_t)^\top D_t},\qquad S_t=S_{t-1}+\varphi(k_t)\nu_t^\top,\quad D_t=D_{t-1}+\varphi(k_t).$$

The factorization removes the explicit $t\times t$ score matrix. The result is generally different from softmax attention. Its feature map changes or approximates the similarity function used by softmax. The fixed state can also forget which of many distinct keys produced a summed contribution. A **recurrent neural network (RNN)** updates a state as it reads successive tokens. [Transformers are RNNs](https://arxiv.org/abs/2006.16236) derives the recurrent perspective. DeltaNet and gated variants use error-correcting updates or decay to make fixed-state memory more selective. [Gated Delta Networks](https://arxiv.org/abs/2412.06464) describes one specific architecture.

**Multi-head attention** uses several query/key/value projections in parallel. **Grouped-query attention** lets multiple query heads share a smaller number of KV heads. **Multi-query attention** is the limiting case with one KV head. The cache formula above makes the memory saving explicit. [Grouped-Query Attention (GQA)](https://arxiv.org/abs/2305.13245) studies the conversion and quality tradeoff. [PagedAttention](https://arxiv.org/abs/2309.06180) tackles a different problem: allocating and sharing KV blocks in pages, small managed units of memory. It does not reduce the mathematical amount of attention required for a long sequence by itself.

## 3. Training for longer sequences [36:35](https://www.youtube.com/watch?v=AiwCCvFW1uE&t=2195s)

**Length extension** increases the sequence length a model can use effectively. A **short-to-long curriculum** trains on shorter examples first and introduces longer ones later. Coherent long documents provide dependencies that the model can learn to track. **Context parallelism** distributes sequence positions across computing devices during training. A **position embedding** gives a token representation information about where it occurs. Absolute position embeddings encode a position directly. Rotary position embeddings (RoPE) rotate query and key coordinates according to position. Models without explicit position embeddings must obtain order information from other structure, including the attention mask.

For one two-dimensional query/key coordinate pair, let $R(\alpha)$ be a rotation through angle $\alpha$, $\omega$ a fixed angular frequency, $q$ and $k$ the unrotated query and key vectors, and $i,j$ their token positions. RoPE rotates the query by $i\omega$ and the key by $j\omega$. Since rotations preserve dot products and $R(i\omega)^\top R(j\omega)=R((j-i)\omega)$,

$$[R(i\omega)q]^\top[R(j\omega)k]=q^\top R((j-i)\omega)k.$$

The identity explains dependence on **relative displacement**, the position difference $j-i$. For an extension factor $s>1$, **position interpolation** uses an effective position near $i/s$ in place of raw position $i$, keeping rotation angles closer to those encountered during training. [The positional-interpolation paper](https://arxiv.org/abs/2306.15595) and [YaRN](https://arxiv.org/abs/2309.00071) develop more specific schemes. Scaling position indices changes the range of positions the model can process. It does not, by itself, teach the model to reason over long-range dependencies. Suitable data and continued training also matter.

In **bidirectional, unmasked attention**, every position may read every other position. Permuting input tokens without changing their embeddings permutes outputs, so absolute order is unavailable without a positional mechanism. In a causal model, the **triangular mask** sets attention to future positions to zero: token $t$ can attend to positions $\{1,\ldots,t\}$ but not later ones. Thus order information can enter through the mask even without explicit position vectors. This representation argument shows how a causal mask can convey order information. It does not prove that removing RoPE is optimal. The [NoPE length-generalization study](https://arxiv.org/abs/2305.19466) examines that question.

Context parallelism partitions sequence positions across devices and exchanges the needed blocks or states. This mechanism addresses training memory and throughput. It does not decide which information an agent should include in its prompt. [Ring Attention](https://arxiv.org/abs/2310.01889) and [Megatron's context-parallelism documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html) provide implementation details.

## 4. KV caching and stable prefixes [52:30](https://www.youtube.com/watch?v=AiwCCvFW1uE&t=3150s)

Several serving mechanisms should be kept distinct. A KV cache avoids recomputing already-processed tokens. **Paged allocation** manages that memory in reusable blocks. **Prefix sharing** reuses computation for identical initial token sequences across requests. **Cache-aware routing** sends a request to a server worker likely to hold the needed cached state. [SGLang](https://arxiv.org/abs/2312.07104) describes RadixAttention, a prefix-sharing approach. [Prompt Cache](https://arxiv.org/abs/2311.04934) develops another approach to reuse.

Let $p_u$ be the price of an uncached input token, $p_c$ the price of a cached token, $N$ the number of submitted input tokens, $h\in[0,1]$ the fraction billed as cached, and $C_{\rm input}$ the total input-token charge. Then $C_{\rm input}=N[(1-h)p_u+hp_c]$. A **partial derivative** measures how an output changes with one input while others stay fixed. Thus $\partial C_{\rm input}/\partial h=N(p_c-p_u)$ measures how the charge changes when $h$ rises. The derivative is negative when cached tokens are cheaper. Actual prices, eligibility, retention, and accounting depend on the provider and can change.

Cache reuse generally requires an **identical prefix** at the relevant boundary. Stable system instructions and tool schemas should precede variable information when the prompt's meaning permits that order. A changing timestamp near the beginning can invalidate a large shared prefix. Measure *actual* cached-token counts under the target provider because cache policy and routing are service-specific. Place mutable material later when doing so preserves the intended meaning of the prompt. Correctness takes precedence over cache hit rate.

### Prefix stability and disclosure policy

Let $p$ be a stable prompt prefix, $d_t$ its task-dependent suffix, $\Vert$ sequence concatenation, and $|p|$ the prefix length in tokens. The prompt is $c_t=p\Vert d_t$. If a service caches $p$ and charges $p_c<p_u$, each reuse saves approximately $|p|(p_u-p_c)$ relative to a full uncached prefill, subject to that service's cache rules. Inserting a changing status line near the beginning shortens the common prefix even if most of the visible text is unchanged. Instructions repeated on every call consume tokens and may be costly. A safety-critical instruction should still be retained even if its presence reduces cache reuse. Cache optimization should operate inside a correctness-preserving context policy.

[Cursor's project-rule system](https://docs.cursor.com/context/rules-for-ai) gives a disclosure spectrum: rules can be always included, attached when a file path matches a pattern, requested by the agent, or invoked manually. Its `.cursor/rules` files are version-controlled context controls. An `AGENTS.md` file provides a simpler form of project instruction. Recall that $z_i\in\{0,1\}$ is the inclusion decision for item $i$; let $\mathbf1\{A\}$ equal one if condition $A$ holds and zero otherwise, and let $\pi_i(h_t)$ denote an agent-selected inclusion decision based on history $h_t$. The three modes correspond to $z_i=1$, $z_i=\mathbf1\{\text{path matches}\}$, and $z_i=\pi_i(h_t)$.

An always-on rule is available on every call and cannot be missed by retrieval. It consumes tokens and may interfere with unrelated tasks each time. A conditional rule saves those tokens when irrelevant, but its selector can fail to include it when needed. The appropriate policy depends on recurrence, selection accuracy, and the consequence of omission.

Hermes exposes the fixed-prefix burden through [`hermes prompt-size`](https://hermes-agent.nousresearch.com/docs/guides/tips), which reports components such as system prompt, skills index, memory, and tool schemas. Its guidance to discover deeper `AGENTS.md` files as tools encounter subdirectories is a just-in-time policy. These are operational examples of context selection. They do not establish that any product has solved the general problem of choosing the right information for every task.

OpenAI's [harness-engineering account](https://openai.com/index/harness-engineering/) describes a related **repository**, a versioned collection of project files: a short `AGENTS.md` maps into a structured `docs/` knowledge base containing architecture, plans, and validated references. This is **two-stage retrieval**: the agent receives a compact index, then reads task-relevant documents.

For document $D_i$, let $\ell_i^{\rm doc}$ be its full length, $\rho_i$ the probability of selecting it on a future task, and $r_i$ the length of its index description. The expected prompt burden is approximately $\sum_i r_i+\sum_i\rho_i\ell_i^{\rm doc}$ rather than $\sum_i\ell_i^{\rm doc}$, assuming all index entries are always loaded and selection incurs no other prompt cost. The saving is meaningful when $\rho_i\ll1$ (much less than one) for most documents. A selector that misses a necessary document can nevertheless impose a large task cost. Navigable names and fresh links help the agent locate the correct document. Checks that documentation still matches code help ensure that the retrieved document is trustworthy.

Tool schemas obey the same economics. For $k$ available tools, let $s_i$ be the prompt-token length of tool $i$'s schema. If all schemas appear on each of $T$ model calls, submitted schema volume is $T\sum_{i=1}^{k}s_i$. Disabling irrelevant tools can reduce cost and tool-selection ambiguity. The selection rule must still retain every tool needed for the task. A **staged interface** first exposes a short catalog, then loads a specific schema when the agent selects a capability. Its selection error must be evaluated against the savings. A schema change may also invalidate a cached prefix. Measure prompt construction and tool discovery together when estimating the change's cost.

## 5. Compaction as state estimation [66:45](https://www.youtube.com/watch?v=AiwCCvFW1uE&t=4005s)

Compaction replaces a long interaction history with a smaller continuation state. It must preserve constraints, decisions, and unresolved work that will matter later. Repeated compaction can distort those details. A fluent summary may still omit a hard constraint, causing a later task failure.

Let $h_t$ be the full event history before continuation and $Z_t=f(h_t)$ a compact state produced by compression function $f$. Let $Y$ be a future outcome the agent must predict or act upon. The expression $P(Y\mid h_t)$ means the conditional probability distribution of $Y$ given history $h_t$. A **sufficient statistic** retains all information in $h_t$ relevant to $Y$, expressed as $P(Y\mid h_t)=P(Y\mid Z_t)$. Real summaries are imperfect. For $J$ potentially needed facts, let $w_j\ge0$ be the consequence of losing fact $j$, and let $\mathbf1\{A\}$ equal one if condition $A$ holds and zero otherwise. Let $\mathbb E$ mean averaging over possible histories and future needs. The expected weighted retention loss $\mathcal L(f)$ is

$$\mathcal L(f)=\mathbb E\left[\sum_{j=1}^{J}w_j\,\mathbf 1\{\text{fact }j\text{ needed later but absent or wrong in }Z_t\}\right].$$

The loss function gives greater weight to facts whose omission could change a later decision. Preserve credential *references* rather than secret values, along with hard constraints, unresolved errors, file names, tests, and decisions. A smooth summary is less valuable if it loses any of these items. Maintain external artifacts or searchable logs for details that do not fit the compact state.

Evaluate compaction by what the agent can do after it resumes. The fluency or elegance of the summary is not a sufficient outcome measure. Plant facts and decisions at known points in long traces, compact, then require the agent to complete downstream tasks that depend on them. Measure task success and constraint violations against an uncompacted baseline at matched cost. Repeated compaction deserves separate testing because error can compound.

### Sufficient statistics, recoverability, and compaction error

The full history and the continuation state serve different purposes. A durable event log can preserve $h_t$ even when the model sees only $Z_t=f(h_t)$. Anthropic's [Managed Agents architecture](https://www.anthropic.com/engineering/managed-agents) separates the session event log from the harness's transformation of selected events into model context. The event log permits selective replay after a failure. The compact summary need not be the only surviving record. This architecture does not guarantee that $Z_t$ is sufficient. It makes an omission recoverable when the relevant event remains in the log and can be found again.

Suppose each compaction independently drops a needed fact with probability $p_{\rm drop}$. After $k$ successive lossy summaries, retention is $(1-p_{\rm drop})^k$. Compaction errors are rarely independent in practice. The expression nevertheless shows how repeatedly summarizing the latest summary can amplify loss. A safer structure retains immutable source events and regenerates a compact view from them, or stores explicit pointers to exact artifacts. The summary should distinguish confirmed facts, hypotheses, decisions, active constraints, pending actions, and call outcomes. An unresolved **side effect** is an external state change whose outcome is not yet known. Retain its **idempotency key**, an identifier that lets the service deduplicate retries, and its `unknown` status. Dropping either may cause the resumed agent to repeat an action that already committed.

Compaction quality can also be expressed as **decision regret**, the expected loss from acting with compressed rather than full information. Let $\pi^*(h_t)$ choose the best next action from full history, $\pi(Z_t)$ choose from compact state, and $V_{\rm act}(h_t,a)$ be the expected value of taking action $a$ when the full history is $h_t$. Define $\mathcal R(f)=\mathbb E[V_{\rm act}(h_t,\pi^*(h_t))-V_{\rm act}(h_t,\pi(f(h_t)))]$. Decision regret is a conceptual quantity and is not directly observable in ordinary deployments. Paired downstream tasks can approximate it by comparing continuation with full and compacted histories. A compact state need not retain every sentence. It must retain distinctions that could change a safe future action.

### When long context is the wrong remedy

Increasing $N_{\max}$ helps only if needed evidence is available, placed in the prompt, and used correctly. A context failure can originate at three stages. *Retrieval* may fail to find the source. *Assembly* may drop or de-prioritize it. The model may fail to use the evidence even after it appears in the prompt. Instrument these stages separately.

A **needle-in-a-haystack test** asks the model to recover one planted fact from a long prompt. It primarily probes whether the model can use information already included in that prompt. A real coding task also depends on file discovery, tool output quality, and action planning. If a fact is authoritative and changes rarely, a small rule or typed state field may be more reliable than embedding an entire history. If it changes rapidly, a fresh **application programming interface (API)** query—a programmatic request for current data—is safer than retaining a long stale transcript.

The distinction matters for cost as well. For a prompt-token budget $B$, let $p_{\rm miss}(B)$ be the probability that assembly omits critical evidence, and $p_{\rm use}(B)$ the probability of correct model use *conditional on inclusion*. By the probability chain rule, the probability of both including and using that evidence is $(1-p_{\rm miss}(B))p_{\rm use}(B)$; this factorization does not assume independence.

Raising $B$ can reduce the probability that assembly omits evidence. It may also increase distraction or latency. The probability of correct use need not improve. Evaluate downstream task success against cost. Context length and isolated recall scores are useful diagnostics, but neither measures the whole task.

### Worked budget example: caching versus compaction

Consider a 100-step agent run with a fixed 2,000-token instruction/tool prefix and 300 new history tokens per step. Without compaction, the agent submits $2{,}000(100)+300(100)(99)/2=1{,}685{,}000$ input tokens. The final prompt is $2{,}000+99(300)=31{,}700$ tokens. A perfectly stable prefix may make repeated tokens cheaper to process or bill. The final model call still contains 31,700 tokens of potentially relevant and irrelevant history.

Suppose the harness compacts after every 20 steps and replaces the accumulated segment with a 1,000-token state summary. Each subsequent block begins with the fixed 2,000-token prefix and the saved state. It then grows by 300 history tokens per step. Cumulative input falls. At each boundary, however, the agent must rely on a summary that may have lost information. The engineering objective is two-dimensional: reduce computation and billing while controlling continuation error. Cache reuse cannot recover stale evidence or a hard constraint lost during compaction. A compact state can therefore lower cost while degrading correctness.

Under the stated convention, the summary replaces the preceding block's summary rather than accumulating with it. The first 20-call block submits $20(2{,}000)+300(0+\cdots+19)=97{,}000$ tokens. Each of the next four blocks submits $20(2{,}000+1{,}000)+300(0+\cdots+19)=117{,}000$ tokens. The total is $97{,}000+4(117{,}000)=565{,}000$ submitted input tokens, about $66.5\%$ below the uncompacted count. The last call contains $2{,}000+1{,}000+19(300)=8{,}700$ tokens. These counts exclude tokens and computation used to create summaries, any cached-token price reduction, and any accuracy loss. A comparison of billing or quality must include those terms separately.

The two arithmetic functions in [context.py](https://github.com/az9713/11-768-ai-agents-notes/blob/main/agent_lab/context.py) implement the respective sums. `calls` is the number $T$ of model calls, `prefix` the fixed instruction and tool-schema length, `growth` the new history length after each call, `block` the calls between compactions, and `summary` the replacement summary length. This function requires $T$ to be a multiple of the block length, matching the worked example.

```python
def submitted_input_tokens(calls, prefix, growth):
    if min(calls, prefix, growth) < 0:
        raise ValueError("token counts must be nonnegative")
    return prefix * calls + growth * calls * (calls - 1) // 2


def compacted_input_tokens(
    calls, prefix, growth, block, summary
):
    if block < 1 or calls < 0 or calls % block:
        raise ValueError("calls must be a multiple of block")
    if min(prefix, growth, summary) < 0:
        raise ValueError("token counts must be nonnegative")
    if calls == 0:
        return 0
    return (
        prefix * calls
        + growth * calls * (block - 1) // 2
        + summary * (calls - block)
    )


assert submitted_input_tokens(100, 2000, 300) == 1_685_000
assert compacted_input_tokens(100, 2000, 300, 20, 1000) == 565_000
```

In the release repair, the summary must preserve the exact “do not publish” constraint, the failing test identifier, changes already made, and the location of full logs. If it retains only “release issue resolved,” the next action may be wrong even though the history is shorter and the prose reads fluently. A continuation test should ask the agent to repair and report, then inspect whether it attempted publication.

### A concrete compaction schema

Rather than ask for a generic narrative summary, define typed slots: `goal`, `hard_constraints`, `confirmed_facts`, `open_hypotheses`, `actions_completed`, `artifacts`, `pending_actions`, and `evidence_locations`. Mark each item with its source step and confidence. Keep exact identifiers and commands when subsequent steps depend on them. Store bulky logs outside the prompt with stable handles. On continuation, test that the compact state entails known invariants—for example, “do not publish” or “the target file is X”—before allowing further action. The summary is a form of state estimate: it must preserve information needed to choose future actions. A polished description of the past is insufficient when it omits an unresolved obligation.

The implemented [compact state](https://github.com/az9713/11-768-ai-agents-notes/blob/main/agent_lab/context.py) is narrower than this proposed production schema. It preserves the goal and hard constraints exactly, retains the most recent observations, and stores identifiers for every observation in the complete in-process history. `keep_recent` is the number retained in the prompt. The full history remains in `RunState`; the current package does not persist that history across a process restart or expose a retrieval tool for old identifiers.

```python
def compact_state(state, keep_recent):
    if keep_recent < 0:
        raise ValueError("keep_recent must be nonnegative")
    recent = tuple(state.observations[-keep_recent:]) if keep_recent else ()
    return CompactState(
        state.goal,
        state.hard_constraints,
        recent,
        tuple(item.call_id for item in state.observations),
    )
```

`ContextBuilder.build` assembles the goal, exact constraints, selected memories, and recent observations for the next model proposal. Its `estimated_tokens` counts whitespace-separated units as a teaching proxy. A deployed adapter must use the model provider's tokenizer and account for tool-schema serialization, role framing, and multimodal inputs before enforcing a real context limit.

### Exercises

1. Derive the $\Theta(T^2)$ cumulative input-token result for a growing history. Recalculate when compaction resets the history to $c$ tokens every $K$ steps.
2. Use the KV formula to compare 32 versus 8 KV heads, holding all else fixed. Why can a paged allocator improve throughput without changing this theoretical byte count?
3. Construct a long-context continuation test where the relevant constraint appears in the middle. Specify a failure metric stronger than exact-string recall.
4. Show with a two-dimensional vector why the RoPE dot product depends on $j-i$. What changes if the position index is interpolated by $s$?

### Solutions and discussion

1. With fixed prefix $m$, growth $r$ per call, and $T$ calls, the submitted total is $\sum_{t=1}^T[m+(t-1)r]=mT+rT(T-1)/2$. For fixed positive $r$, the quadratic term gives $\Theta(T^2)$. If $T$ is divisible by block size $K$, each block resets its newly accumulated history to zero, and a fixed $c$-token summary is carried into every block after the first, the total is $mT+rT(K-1)/2+c(T-K)$. For the worked values $T=100$, $K=20$, $m=2{,}000$, $r=300$, and $c=1{,}000$, this gives $565{,}000$. If summaries accumulate rather than replace one another, the expression changes and can again grow poorly.
2. The cache formula $B_{KV}=2Ln h_{KV}d_k b$ is linear in the number $h_{KV}$ of key/value heads. Replacing 8 heads with 32 multiplies the illustrative $15.6$ GiB by four, yielding about $62.5$ GiB before allocator overhead. A paged allocator reduces wasted or stranded blocks and can let more requests fit concurrently. It does not change the number of bytes required to represent one fully occupied sequence under the same numerical format.
3. Put the prohibition “do not publish” midway through a long repair trace. After compaction, ask the agent to fix the build and deliver a patch. Measure whether the final artifact passes tests and whether any deployment or publication tool was proposed or invoked. A correct quotation of the prohibition is weaker evidence than complying with it throughout the continuation. Include changed distractor logs and repeat runs to test whether the result is stable.
4. Write $q=(q_1,q_2)$ and $k=(k_1,k_2)$ as two-dimensional real vectors. Rotating them at positions $i$ and $j$ gives $[R(i\omega)q]^\top[R(j\omega)k]=q^\top R(i\omega)^\top R(j\omega)k=q^\top R((j-i)\omega)k$. The middle equality uses the fact that the transpose of a rotation reverses its angle; the last uses addition of rotation angles. If positions are interpolated by factor $s>1$, the relative angle becomes $(j-i)\omega/s$. The representation then changes more slowly with token distance. This identity alone does not establish preservation of long-range task accuracy.

### Further reading

- [Lost in the Middle](https://aclanthology.org/2024.tacl-1.9/): effective context versus nominal length.
- [Transformers are RNNs](https://arxiv.org/abs/2006.16236), [Gated Delta Networks](https://arxiv.org/abs/2412.06464), and [Grouped-Query Attention (GQA)](https://arxiv.org/abs/2305.13245): stateful and compressed attention.
- [Position Interpolation](https://arxiv.org/abs/2306.15595) and [YaRN](https://arxiv.org/abs/2309.00071): length extension.
- [PagedAttention](https://arxiv.org/abs/2309.06180), [SGLang](https://arxiv.org/abs/2312.07104), and [Prompt Cache](https://arxiv.org/abs/2311.04934): serving and reuse.
