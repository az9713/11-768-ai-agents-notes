# Lecture 4 — Memory and skills for agents

*Independent study chapter · Based on a CMU 11-768: AI Agents lecture by Daniel Fried, 3 September 2026 · [Course and attribution](index.html#sources)*

## 0. The cross-task learning problem [01:00](https://www.youtube.com/watch?v=6zigF2a-2Pw&t=60s)

A **language model** predicts the next token, a discrete text unit, from preceding tokens. An agent repeatedly chooses actions toward a goal using that model. Suppose it has found return policies for many online stores and is now asked to find one on a new site. A past page address may be stale; a general procedure for locating and verifying the current policy may transfer.

The agent's **context** is the information supplied to the model for one call. **Compaction** compresses a long task history into a shorter continuation state. A **persistent memory store** retains selected information across separate tasks. A **skill** is a reusable procedure for a class of tasks. The problem is to decide what past experience generalizes, how to represent it, and when to retrieve it.

Index successive tasks by $i=1,2,\ldots$ and write $\tau_i$ for task $i$. Let $P(\tau)$ be the probability distribution over future tasks; it assigns relative likelihoods to tasks. Let $e_i$ be the recorded experience from task $\tau_i$, $M$ the durable store, and $u$ the current authority state. Three distinct **policies**, or decision rules, govern memory. The write policy $W(e_i)$ proposes an item to store. The retrieval policy $R(\tau_j,M,u)$ selects accessible items for a later task $\tau_j$. The application policy $\operatorname{Apply}(\tau_j,R(\tau_j,M,u))$ uses the selected information during that task.

The objective is not to maximize the quantity of stored text. A candidate memory should increase expected future task benefit enough to justify its storage, retrieval, distraction, privacy, and maintenance costs.

Within one run, let $t$ index decision steps, $g$ be the user's goal, $h_t$ the event history, $m_t$ retrieved persistent memory, $\mathcal T_t$ available tools, $b_t$ remaining resources, and $u_t$ current authorization. A **harness** is the surrounding software that manages model calls, tools, and execution rules. Its **context builder** $C_\phi$, configured by harness choices $\phi$, produces model input $c_t=C_\phi(g,h_t,m_t,\mathcal T_t,b_t)$. A skill may also influence the **action policy**, the rule for choosing the next operation.

These are different interventions. A fact such as “the project uses Python 3.12” changes the information available about the environment. A skill such as “run the focused test, inspect the diff, then run the package suite” changes *how* the agent acts. Neither should modify execution authority $u_t$. A memory recording past approval is evidence about a past authorization. The current **permission gate**, the harness check that admits or rejects an action, must determine whether authorization applies now.

### A decision rule for writing memory

Let $m$ be a candidate memory item and $U(\tau\mid m)$ the task-outcome value when the system may retrieve and use $m$ under a specified retrieval policy. Let $U(\tau\mid\varnothing)$ be the corresponding value without the item, where $\varnothing$ denotes no retrieved item. These values exclude the storage, retrieval, and risk costs accounted for below. Let $\mathbb E_\tau$ average over the future-task distribution $P(\tau)$. The item's gross expected benefit is $B(m)=\mathbb E_\tau[U(\tau\mid m)-U(\tau\mid\varnothing)]$. A selector that often fails to retrieve $m$ reduces this benefit.

Let $C_{\rm store}(m)$ include recurring prompt tokens, retrieval computation, review, and maintenance, and $C_{\rm risk}(m)$ expected harm from stale or misapplied content. Express all three quantities in the same utility units. Write $m$ only if $B(m)>C_{\rm store}(m)+C_{\rm risk}(m)$ under defensible estimates. This inequality is a decision criterion, not a claim that a live system knows these quantities exactly. A hard-to-recall user constraint may deserve durable storage because forgetting it has a high cost. A verbose trace from a one-off task may be better kept in a searchable archive.

The difference inside $B(m)$ asks how much the item changes the outcome of a future task. Averaging over $P(\tau)$ gives more weight to tasks that occur often. A useful item can still be a poor choice for always-on memory if its recurring token and maintenance costs exceed that expected benefit.

Let $N$ be the number of future runs, $\ell_m$ the item's token length, and $p_{\rm tok}$ the cost per input token. Loading it on every run costs roughly $N\ell_m p_{\rm tok}$. **On-demand retrieval** selects an item only when the current task appears to need it. This lowers recurring prompt cost. The selector can still miss a useful item or include a harmful one. The same always-on versus conditional tradeoff applies to project rules in Chapter 3.

## 1. Memory representations [05:00](https://www.youtube.com/watch?v=6zigF2a-2Pw&t=300s)

A **memory item** records information from experience. A skill encodes a reusable way to carry out a class of tasks. An **episode** is one complete attempt at a task. A **trajectory** is the ordered sequence of actions and observations within an episode. An extracted fact, a lesson from feedback, and an executable routine preserve different information and support different retrieval decisions. A skill may be written by a person or **induced**, meaning inferred from several trajectories, and may consist of instructions, examples, or code.

The return-policy example yields distinct candidates. “Store A's policy page had address X on a particular date” is a time-bound fact. “Check that the page belongs to the merchant and has a current effective date” is a reusable verification rule. “Open this exact menu, then click the third link” is a site-dependent procedure. Treating all three as undifferentiated text obscures their different validity periods and transfer risks.

| Representation | Example | Retrieval question | Main risk |
|---|---|---|---|
| Episode | full browser trace for a past purchase | Is this case similar? | high context cost; brittle details |
| Fact | user prefers a particular format | Is it current and authorized to retain? | staleness or privacy |
| Lesson | a past search strategy failed on dynamic pages | Does this failure mode apply? | overgeneralization |
| Text skill | “review Python changes in this order” | Does this task match its scope? | irrelevant instruction interference |
| Code skill | tested `sort_listings(site, order)` routine | Does this implementation fit the site? | breakage, dependencies, side effects |

Store each item with **metadata**, structured fields that describe its content and use. The `kind` field classifies the item, while `content` holds its claim or procedure. The fields `source_task` and `created_at` record origin and time. Use `scope` to state where the item applies, `confidence` to record the strength of its support, and `permissions` to limit access. The `expiry` field marks when to reconsider the item, and `evidence` points to supporting observations.

These fields prevent several category errors. A fact about a user does not become a universal instruction. One failure does not establish a general rule. Storing a skill does not authorize it to run. **Typed** metadata distinguishes these categories explicitly rather than leaving the model to infer them from prose.

### Provenance and temporal validity

**Provenance** records where a claim came from and when it was valid. Let $p$ name a property, $v$ its claimed value, $[t_0,t_1)$ the time interval including $t_0$ but excluding $t_1$, $\sigma$ the source, and $\gamma$ an evidence grade or confidence estimate. Represent the fact by $(p,v,[t_0,t_1),\sigma,\gamma)$. A later correction can close the old interval instead of erasing history. A configuration fact valid before a migration should not be retrieved as current after it. A versioned procedure also records the tool and environment versions under which its preconditions and postconditions were tested. **Semantic similarity**, closeness of meaning between texts, cannot establish that those preconditions still hold.

An instruction-like memory is particularly sensitive to provenance. A webpage's claim that the agent should reveal **credentials**, secrets used to authenticate access, must remain source content, even if stored and retrieved later. Promoting it to a system instruction is **memory poisoning**, the persistent analogue of **prompt injection**, an attempt to make lower-trust content act as a higher-authority instruction. A **write gate**, a check before durable storage, should require trusted origin for policy-like content and store ordinary retrieved material as quoted evidence with its source. Deletion and correction are security controls as well as quality controls.

The [companion memory type](https://github.com/az9713/11-768-ai-agents-notes/blob/main/agent_lab/memory.py) records provenance and scope as fields rather than burying them in the text. `kind` distinguishes a fact from a procedure. `source` states its origin. `valid_from` and `valid_until` are Unix timestamps, measured in seconds from 1 January 1970 Coordinated Universal Time (UTC); the latter may be `None` for no preset expiry. `trigger_terms` are simple case-insensitive substrings in the task goal. `required_effects` describes the capabilities needed to follow a skill. This trigger is intentionally elementary; it can miss paraphrases and match irrelevant uses of a word.

```python
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MemoryItem:
    item_id: str
    kind: Literal["fact", "skill"]
    content: str
    source: str
    scope: str
    valid_from: float
    valid_until: float | None
    token_cost: int
    estimated_gain: float
    estimated_risk: float
    trigger_terms: tuple[str, ...] = ()
    required_effects: frozenset[str] = frozenset()
```

The [file-backed store](https://github.com/az9713/11-768-ai-agents-notes/blob/main/agent_lab/memory.py) writes its list of records as JSON, a structured text format, through a temporary file followed by `os.replace`. This prevents a reader from seeing a partially written file during an ordinary single-process replacement. It does not implement a trusted-origin write gate, version history, deletion workflow, or cross-process coordination. In this example, the application developer writes memory; the model cannot write to the store through a registered tool. A later extension that grants model-initiated writes must add those controls before exposing that operation.

## 2. Authored skills and progressive disclosure [15:00](https://www.youtube.com/watch?v=6zigF2a-2Pw&t=900s)

An authored skill can be stored as a package whose `SKILL.md` contains instructions and pointers to supplementary material. A compact **index** contains skill names and short descriptions. The agent sees this index first and loads full instructions only for skills that appear relevant. This **progressive disclosure** controls context cost while retaining access to detailed procedures.

Let $K$ be the number of skills, $\ell_i$ the token length of skill $i$'s full instructions, $d_i$ its index-description length, and $j$ the selected skill. The symbol $\sum_i$ means summing over all $K$ skills. The relation $d_i\ll\ell_i$ means that an index description is much shorter than a skill body. Loading everything costs $\sum_i\ell_i$ tokens, while loading the index and one selected skill costs $\sum_i d_i+\ell_j$. Retrieval errors can cancel the saving. A longer description may help the agent select the right skill. It also increases index cost and may distract the model.

A skill is most useful when demand recurs, success can be tested, and the reusable procedure is substantial relative to task-specific details. A one-off preference, unverified tactic, or instructions tied to unstable selectors are poor candidates. Derive procedures from observed successes and failures. Specify when a skill should activate, what prerequisites must hold, what artifact it should produce, how to check the result, and when to stop. Version the skill and evaluate each revision. The [OpenHands guide to creating effective skills](https://www.openhands.dev/blog/20260227-creating-effective-agent-skills) develops practical design criteria.

For a `find_return_policy` skill, the index description should state that it applies to locating a merchant's current return terms. The loaded body can then specify a procedure: search the merchant's own site, verify that the page concerns the requested country and product category, record the effective date, and cite the page used. It should stop and report uncertainty if the merchant has conflicting policies. An old policy web address can be a search hint, but it cannot be the verifier. This separation lets the compact index save tokens without making the skill's validity depend on a single historical webpage.

The [SkillsBench paper](https://arxiv.org/abs/2602.12670) reports 87 tasks across eight domains and, in its matched evaluation of 18 model-harness configurations, average pass rates of 33.9% without curated skills and 50.5% with them. The difference is **16.6 percentage points** for that benchmark and its curated-skill condition. It does not establish the effect of an arbitrary skill on an arbitrary agent. The paper also reports that focused bundles outperform larger or exhaustive ones. Skill selection is part of the intervention being evaluated. A comparison should account for selection errors rather than assume that an oracle always picks the right skill.

### Four implemented memory and skill policies

| System | Stored representation and loading policy | Consequence for the formal model |
|---|---|---|
| [Hermes Agent memory](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory) and [skills](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) | Bounded `MEMORY.md` and `USER.md` are loaded at session start; longer procedural skills are discovered and loaded on demand. Memory writes persist to disk but do not alter the already-assembled session prompt. | Always-on facts consume fixed context; skill retrieval trades prompt cost for selection risk. A frozen prompt snapshot requires explicit refresh or a new session to affect current context. |
| [OpenClaw memory](https://docs.openclaw.ai/concepts/memory) and [skills](https://docs.openclaw.ai/tools/skills) | Workspace Markdown files, written in a lightweight text-formatting syntax, separate profile, long-term, and dated notes; file-backed skills have defined discovery and precedence rules. The documentation explicitly separates memory from tool policy. | Durable text can inform $C_\phi$, but capability state $u_t$ remains a separate enforcement surface. Precedence and workspace scope must be tested. |
| [Cursor rules](https://docs.cursor.com/context/rules-for-ai) | Project rules can be always included, path attached, requested, or manually invoked. | A rule is a context-selection policy, not automatically a learned memory or executable skill. Its inclusion depends on metadata and task path. |
| [OpenAI Agents software development kit (SDK)](https://openai.com/index/the-next-evolution-of-the-agents-sdk/) | The SDK describes configurable memory, sandbox-aware orchestration, filesystem tools, Model Context Protocol (MCP) tools, skills, and `AGENTS.md` as distinct primitives. | A useful harness exposes separate controls for persistent state, context disclosure, tool access, and execution isolation. |

These implementations demonstrate different design choices. The examples alone do not establish which implementation performs better. A comparison should hold model, task distribution, tool rights, and budget constant. Otherwise, a gain may be attributed to memory when it actually comes from another component.

## 3. External memory and conflicts [33:30](https://www.youtube.com/watch?v=6zigF2a-2Pw&t=2010s)

External stores permit an agent to read and write memories beyond its current context window. [MemGPT](https://arxiv.org/abs/2310.08560) explores memory management through model-mediated operations. [Mem0](https://arxiv.org/abs/2504.19413) studies extraction and maintenance. A durable store needs update and deletion operations as well as insertion, because facts change and contradictory entries cannot be resolved by indefinite accumulation.

For a candidate fact $f$ and existing memory $m$, distinguish four cases. If the facts are independent, add $f$. If they duplicate one another, retain the better-supported version. If $f$ refines $m$, merge them while preserving provenance. If they contradict one another, examine time, source, and authority before resolving the conflict. “I work at X” from 2024 and “I now work at Y” from 2026 form a time-indexed update. A validity interval $[t_{\rm start},t_{\rm end})$ includes its start time $t_{\rm start}$ and excludes its end time $t_{\rm end}$. Correction and deletion are necessary because a false fact can otherwise become a persistent bias.

For return policies, a later dated merchant page can supersede an earlier one for future purchases. It does not erase which policy applied to an earlier order. The memory store should preserve the dates and source addresses of both claims. If two pages conflict without a clear effective-date rule, mark the current policy unresolved and retrieve the authoritative source again. A semantic search score cannot adjudicate the conflict.

A retrieval system should answer “Is this item needed for this task?” before “Is it semantically similar?” Personal data requires retention limits and access scope. Even a correct memory can be inappropriate to surface in a new context. Benchmark accuracy does not compensate for leaking private context.

The `applicable` predicate in [memory.py](https://github.com/az9713/11-768-ai-agents-notes/blob/main/agent_lab/memory.py) implements a necessary, mechanically checked subset of $\operatorname{access}(m,\tau,u)$. It tests scope membership, required effects, the validity interval, and the trigger. `authority.memory_scopes` is the set of memory scopes permitted for this run; it is separate from `authority.effects`, the executable tool rights. Passing this filter does not establish that the content is true or useful.

```python
def applicable(self, goal, authority, now):
    if self.scope not in authority.memory_scopes:
        return False
    if not self.required_effects <= authority.effects:
        return False
    if now < self.valid_from:
        return False
    if self.valid_until is not None and now >= self.valid_until:
        return False
    lowered = goal.casefold()
    return not self.trigger_terms or any(
        term.casefold() in lowered for term in self.trigger_terms
    )
```

For memory $m$, task $\tau$, and authority state $u$, let $\operatorname{access}(m,\tau,u)$ be one when this task may use the item and zero otherwise. The notation $m\in M$ means item $m$ belongs to store $M$, and $A\subseteq B$ means every member of set $A$ also belongs to $B$. The retrieval policy $R(\tau,M,u)$ should return only accessible items: $R(\tau,M,u)\subseteq\{m\in M:\operatorname{access}(m,\tau,u)=1\}$. Among accessible items it should prefer those with positive expected benefit. This allows a private preference to be available in a personal assistant yet excluded from a shared work session. A retrieval model must not turn semantic similarity into permission.

## 4. Skill induction from trajectories [44:30](https://www.youtube.com/watch?v=6zigF2a-2Pw&t=2670s)

Repeated website tasks may share a subroutine such as “search for a product.” A system can extract that subroutine from successful episodes and represent it as text with examples or as executable code. [Agent Workflow Memory](https://arxiv.org/abs/2409.07429) induces reusable workflows from web-navigation experience and evaluates transfer. [Voyager](https://arxiv.org/abs/2305.16291) develops an executable skill library in a different environment.

Let $E$ be a collection of past episodes and $s$ a proposed skill. The **minimum-description-length principle** favors a short reusable representation that leaves little episode detail unexplained. Define $\operatorname{length}(s)$ as the complexity of skill $s$. The term $\operatorname{residual}(E\mid s)$ measures episode detail not explained by the skill. Let $\operatorname{failure}_{\rm heldout}(s)$ be its failure rate on separate **held-out** tasks, which were not used to create the skill. Let $\lambda_{\rm fit},\mu_{\rm transfer}\ge0$ weight the latter two penalties. The operator $\arg\min_s$ chooses a skill $s$ that minimizes the expression. A conceptual objective is

$$s^*=\arg\min_s\bigl[\operatorname{length}(s)+\lambda_{\rm fit}\operatorname{residual}(E\mid s)+\mu_{\rm transfer}\operatorname{failure}_{\rm heldout}(s)\bigr].$$

The held-out term penalizes a long **macro**, a fixed sequence of actions, that merely memorizes one trajectory. This is a conceptual objective, not the exact optimization used in the cited systems.

Text and code fail differently. Text can describe intent and adapt to changed interfaces. Each execution still requires the model to interpret the text again. Code can be tested and may execute cheaply. It binds the skill to software libraries, **selectors** (rules identifying webpage elements), and a particular permission scope. A practical design separates an **abstract interface**, which states the required operation, from environment-specific implementations. [PolySkill](https://arxiv.org/abs/2510.15863) studies such polymorphic abstraction. A `sort_listings` operation might use a dropdown on one retailer's site and a **uniform resource locator (URL) query parameter**, a value encoded in the web address, on another. The caller requests the abstract capability. A selector chooses the implementation compatible with the current site.

For skill $s$ and task $\tau$, let the **trigger predicate** $T_s(\tau)$ state whether the task matches the skill. The **precondition** $P_s(x)$ states whether current run state $x$ permits its use. The **effect set** $\epsilon_s$ lists external changes the skill may make. The **execution policy** $\pi_s$ chooses actions, and the **verifier** $V_s$ checks the outcome. Skill application requires $T_s(\tau)\land P_s(x)$, where $\land$ means both conditions hold. The run's allowed capabilities must also include every effect in $\epsilon_s$.

A prose skill can recommend actions, but it cannot establish that $P_s$ holds in the current environment. The harness or agent must check the relevant facts before acting. A code skill can test specific **postconditions**, conditions expected after successful execution, but still needs adaptation when the environment changes. The contract should state when the skill must not be used. It should also describe failure states and recovery, in addition to the successful procedure.

## 5. Failure, judging, and retrieval error [67:00](https://www.youtube.com/watch?v=6zigF2a-2Pw&t=4020s)

Failed traces can yield useful strategic corrections but may also encode task-specific accidents. An **automatic judge** is a program or model that labels an outcome or memory item. **Label noise** means that some labels are wrong. Excessive retrieval can crowd out relevant evidence or prompt the wrong procedure. [ReasoningBank](https://arxiv.org/abs/2509.25140) studies reusable reasoning experiences. [Reflexion](https://arxiv.org/abs/2303.11366) studies verbal feedback after failed attempts.

Let $Z=1$ mean a stored skill truly helps on a future task, $Z=0$ mean it does not, $J=1$ mean a judge labels it useful, and $J=0$ mean it does not. The notation $P(A\mid B)$ means the probability of event $A$ given event $B$. **Sensitivity** $P(J=1\mid Z=1)$ is the chance of approving a helpful skill. The **false-positive rate** $P(J=1\mid Z=0)$ is the chance of approving an unhelpful one. If helpful skills are rare, even a small false-positive rate can fill memory with harmful entries. Bayes' rule gives the **precision**, the probability an approved item is truly useful:

$$P(Z=1\mid J=1)=\frac{P(J=1\mid Z=1)P(Z=1)}{P(J=1\mid Z=1)P(Z=1)+P(J=1\mid Z=0)P(Z=0)}.$$

For example, suppose 10% of candidate skills are helpful, sensitivity is 90%, and the false-positive rate is 10%. Then 9% of all candidates are helpful skills that the judge approves: $0.10(0.90)=0.09$. Another 9% are unhelpful skills that the judge approves: $0.90(0.10)=0.09$. Half of the approved skills are therefore helpful, since $0.09/(0.09+0.09)=0.50$. The numbers are illustrative. Independent validation, provenance, and expiry keep the store from becoming a self-confirming archive.

If only 1% of candidates are helpful while sensitivity and false-positive rate stay at 90% and 10%, the approved helpful fraction is $0.01(0.90)=0.009$ and the approved unhelpful fraction is $0.99(0.10)=0.099$. Precision falls to $0.009/(0.009+0.099)=1/12$, about $8.3\%$. A judge that appears accurate on balanced examples can therefore contaminate a store dominated by weak candidate skills. The prior frequency of genuinely useful skills is an empirical quantity that must be estimated for the actual induction pipeline.

For retrieved item $m$ and task $\tau$, define its **marginal value** $\Delta(m,\tau)=P(\text{success}\mid\tau,m)-P(\text{success}\mid\tau,\varnothing)$: the change in task-success probability when $m$ is supplied. Semantic similarity does not imply $\Delta>0$. A memory may discuss the right topic and still be stale, redundant, or misleading. Evaluate **paired runs** on the same tasks with and without the item. Keep the model, harness, and budget matched, and record latency and token cost as well as success. Test **hard negatives** (similar-looking but inapplicable items), tasks from other domains, changed interfaces, and conflicting facts. [SkillsBench](https://arxiv.org/abs/2602.12670) illustrates matched no-skill and curated-skill conditions.

### Retrieval is a policy, not a search score

For $n$ candidate memories $m_1,\ldots,m_n$, let $z_i\in\{0,1\}$ indicate whether item $i$ is retrieved, $z=(z_1,\ldots,z_n)$ the complete selection, $c_i$ its prompt-token cost, and $\lambda_{\rm token}\ge0$ a weight that converts token cost into task-utility units. Let $U_{\rm sel}(\tau,z)$ be the task-outcome value when selection $z$ is supplied, before charging that token cost, and let $\mathbb E$ average over future tasks and stochastic outcomes. The selector should maximize $\mathbb E[U_{\rm sel}(\tau,z)]-\lambda_{\rm token}\sum_i c_i z_i$ subject to access rules and a stated context-token budget.

A **similarity ranker** orders items by topical closeness. An **embedding** is a numerical vector representing content. A “top $k$ by embedding similarity” rule selects the $k$ nearest vectors. This similarity rule is a baseline for retrieval. It does not account for interactions between items: two individually useful memories may conflict. A skill may also help only when the current **tool schema**, the machine-readable specification of tool arguments, matches its assumptions.

The store applies the eligibility filter first. It then calls Chapter 3's `select_items` with each eligible item's estimated net value $\widehat{\Delta}_i-\widehat{r}_i-\lambda_{\rm token}c_i$, where $\widehat{\Delta}_i$ is `estimated_gain` and $\widehat{r}_i$ is `estimated_risk`. Interpret all three terms in comparable utility units. The estimates are inputs to the teaching example, not calibrated probabilities. The returned names select records for the next prompt. A negative-value item is omitted because the empty selection has value zero in this additive model.

```python
eligible = [
    item for item in self._items.values()
    if item.applicable(goal, authority, current)
]
candidates = [
    Candidate(
        item.item_id,
        item.token_cost,
        item.estimated_gain
        - item.estimated_risk
        - token_price * item.token_cost,
    )
    for item in eligible
]
selected = set(select_items(candidates, budget))
return tuple(item for item in eligible if item.item_id in selected)
```

In [harness.py](https://github.com/az9713/11-768-ai-agents-notes/blob/main/agent_lab/harness.py), `self.memory.retrieve(goal, authority)` supplies selected text to `ContextBuilder.build` before each model proposal. The model sees that text in `Prompt.memories`; the `Authority` object remains unchanged. The [behavioral check](https://github.com/az9713/11-768-ai-agents-notes/blob/main/tests/test_agent_lab.py) confirms both properties. Thus a stored procedure can influence a proposal, while the Chapter 2 tool gate still decides whether any proposed action can execute. Because the deterministic demonstration model does not interpret prose instructions, the test checks prompt inclusion rather than claiming an improvement in task success. A real model adapter would require paired evaluation of runs with and without the memory.

The effect of a memory item is causal only under a defined intervention. Holding the same task and model fixed, compare outcomes when the item is supplied versus withheld while keeping all other context and budgets comparable. Adding an item changes prompt length and may displace other information. A measured treatment effect should include this displacement because it is part of what happens in deployment. For retrieval-policy comparison, evaluate the entire selector on held-out tasks rather than giving the experimental condition an oracle that already knows which skill applies. Report false retrieval as well as missed retrieval. A useful confusion matrix places true applicability on one axis and the retrieval decision on the other. Downstream task success then distinguishes harmless false retrievals from ones that caused failure.

### Skill maintenance as software maintenance

An executable skill has dependencies and side effects. An instructional skill has assumptions and can still steer unsafe behavior. Both need a versioned contract. Let $s^{(r)}$ denote revision $r$ of a skill and $s^{(r+1)}$ its successor. Keep a **regression set**, previously supported tasks used to detect new failures, changed-environment tests, and explicit negative cases where the skill should not activate. Passing the original training trajectory is weak evidence because it can reward memorization. The maintenance decision is whether the revision improves expected utility on future tasks without violating its effect boundary.

Hermes's [skills documentation](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) allows agent-managed creation and update of procedural skills, with an optional approval gate for writes. This makes memory modification part of the agent's action space. Let $\pi_{t+1}$ denote its future action-selection policy after a write at time $t$. Because that write can change $\pi_{t+1}$, risk extends beyond the current task. Review, versioning, and **rollback**, restoring an earlier revision, matter more for a self-modifying skill than for a transient answer. OpenClaw's [skills loading rules](https://docs.openclaw.ai/tools/skills) define **precedence**, the rule for choosing among same-named skills across workspace, personal, managed, and bundled sources. Test that order after an upgrade or workspace change.

## 6. The memory lifecycle

A memory system must learn, represent, retrieve, use, judge, and maintain its entries. These operations form a lifecycle:

1. **Observe:** preserve task goal, actions, outcomes, and verifier evidence.
2. **Propose:** extract a candidate fact, lesson, or procedure with scope and provenance.
3. **Validate:** test factual support or run a code skill. Evaluate it on distinct tasks.
4. **Store:** version it, set permissions and expiry, and keep rollback possible.
5. **Retrieve:** choose only items predicted to help the current task.
6. **Measure:** compare with a no-memory baseline at matched budget.
7. **Consolidate:** merge duplicates, repair stale entries, and delete harmful ones.

A system can remember an episode accurately without benefiting from it later. Evidence of transfer requires a comparison of future decisions with and without that memory. Transfer requires selective retrieval, outcome measurement, and maintenance.

### Worked experiment: does a learned browser skill really transfer?

Suppose an agent induces a `find_return_policy` skill from 30 successful shopping-site trajectories. Randomly splitting *steps* from those same trajectories into train and test would leak the same page structure into both sides. Split evaluation by website and by time. Learn from a set of sites at initial version $v_{\rm site}$. Then test on unseen sites and on changed versions of known sites. Include a no-skill **baseline** (comparison condition), an authored-skill baseline, and an induced-skill condition. Hold the model, harness, task instructions, and execution budget fixed. Randomize or pair tasks across conditions and record task success, steps, latency, token cost, unauthorized actions, and whether the cited policy was actually current.

The skill might raise success on same-site tasks but lower it on unseen sites because it assumes a particular menu name. This pattern identifies the skill's scope. The procedure may be useful only on sites that share the navigation structure it learned. Revise its metadata from “shopping sites” to the supported website family, or induce an abstract procedure that first discovers site navigation and then uses a site-specific implementation. Keep the failed external cases as *counterexamples* for the next revision. Report within-domain and cross-domain performance separately. Pooling them into one average can conceal a failure to transfer.

Running each task once under each condition yields a paired success difference for that task. Repeating **stochastic runs**, whose outputs may vary even under the same setup, reveals variation due to the model and environment. On 20 tasks with one binary success result per condition, one additional success changes the measured rate by five **percentage points**. That gain is weaker evidence than the same rate change on hundreds of independent tasks. A **confidence interval** is a range constructed by a procedure with stated long-run coverage. An **error taxonomy** groups observed failures by cause. Report both alongside the point estimate so readers can judge uncertainty and failure mode. Published benchmarks can suggest hypotheses and experimental designs. The intended application determines which task distribution is relevant.

### Exercises

1. Write a schema for a stored user preference and for an executable skill. Which fields differ? How would you represent a preference superseded later?
2. Derive the 50% judge precision in the illustrative Bayes example. Recompute it if helpful skills have a 1% prior rate. What does this imply for automatic memory writing?
3. Design a held-out test for a browser skill induced from one shopping site. Prevent task leakage and test both changed selectors and genuinely new domains.
4. A retrieved skill improves success from 60% to 64% but adds 20 seconds and $0.20$ per run. State a utility function under which it is or is not worthwhile.

### Solutions and discussion

1. A user-preference item needs the preference value, scope, user-authorized source, valid-from time, expiry or review time, and access policy. An executable skill additionally needs a trigger, preconditions, allowed effects, implementation or instructions, dependencies, expected output, verifier, version, and rollback path. To supersede a preference, close its validity interval at the correction time and create a new version with its own provenance. Do not silently rewrite the old record if historical interpretation matters.
2. With a 10% helpful prior, approved helpful and unhelpful fractions are both $0.09$, giving precision $0.09/(0.09+0.09)=0.50$. With a 1% prior, they are $0.009$ and $0.099$, giving precision $0.009/(0.108)=1/12\approx0.0833$. Automatic writing should therefore use independent outcome checks, a high-precision acceptance rule, and removal or expiry of items that fail on future tasks. The calculation does not determine the optimal acceptance threshold without the costs of false approval and false rejection.
3. Train the browser skill on complete trajectories from one site. Hold out entire tasks collected later, after the site's interface has changed, and tasks from other merchant domains. Never place steps from the same trajectory in both sets. Compare no-skill, authored-skill, and induced-skill conditions under matched model, tools, permissions, and budget. Report success on the original interface, changed selectors, and unseen domains separately. Record incorrect policy citations and unauthorized actions as well as completion.
4. Let $V$ be the value of one successful run, $\lambda_D$ the cost of one second of delay, and $\lambda_C$ the cost of one dollar of spending, all measured in the same utility units. The skill's incremental expected utility is $0.04V-20\lambda_D-0.20\lambda_C$. It is worthwhile under this model exactly when that expression is positive, provided risk is unchanged. For $V=100$, $\lambda_D=0.10$, and $\lambda_C=1$, the gain is $4-2-0.2=1.8$ utility units. At $\lambda_D=0.25$ with the other values fixed, it is $4-5-0.2=-1.2$. If the skill also increases privacy or authorization risk, that outcome needs its own constraint or penalty rather than being hidden in task success.

### Further reading

- [SkillsBench](https://arxiv.org/abs/2602.12670): paired measurement of curated skills.
- [MemGPT](https://arxiv.org/abs/2310.08560) and [Mem0](https://arxiv.org/abs/2504.19413): external-memory systems.
- [Agent Workflow Memory](https://arxiv.org/abs/2409.07429), [Voyager](https://arxiv.org/abs/2305.16291), and [PolySkill](https://arxiv.org/abs/2510.15863): induced and reusable skills.
- [ReasoningBank](https://arxiv.org/abs/2509.25140) and [Reflexion](https://arxiv.org/abs/2303.11366): experience, feedback, and selection.
