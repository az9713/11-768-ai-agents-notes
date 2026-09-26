# Lecture 1 — What are agents, and how do they work?

*Independent study chapter · Based on a CMU 11-768: AI Agents lecture by Daniel Fried and Graham Neubig, 25 August 2026 · [Course and attribution](index.html#sources)*

## 0. Autonomy and consequence [03:55](https://www.youtube.com/watch?v=UwfjzyLnvMg&t=235s)

An **agent** is a system that selects actions in response to observations while pursuing a goal. Consider a customer-support agent asked to cancel an order if it has not shipped and otherwise explain the return options. A stale confirmation email cannot establish the current shipping status. A wrong cancellation may affect fulfillment, while a wrong answer may cause the customer to miss a return deadline. The agent must obtain evidence, choose an action that advances the request, and remain within the authority the customer granted. The consequences also depend on **reversibility**, whether an action's effects can be reliably undone.

The **harness** is the software that checks a proposed action, invokes permitted tools, and records their results. Let $a$ be a proposed action, including its operation and target. Let $s$ be the actual state of the external environment and $e$ the trusted evidence currently available to the harness. Represent the user's authorization policy as $U=(U^+,U^-)$, where $U^+$ contains the actions covered by positive grants for this task, before specific exceptions, and $U^-$ contains explicitly prohibited actions. The superscripts $+$ and $-$ label grants and denials. The symbol $\in$ means “belongs to.” The predicate $\operatorname{inScope}(a,U)$ is true when $a\in U^+$; $\operatorname{forbidden}(a,U)$ is true when $a\in U^-$. The predicate $\operatorname{preconditions}(a,e)$ is true when the evidence establishes the facts required before $a$ may be attempted. The symbols $\land$ and $\neg$ mean logical “and” and “not.” The admission predicate $\operatorname{allow}$ is

$$\operatorname{allow}(a,e,U)=\operatorname{inScope}(a,U)\land\operatorname{preconditions}(a,e)\land\neg\operatorname{forbidden}(a,U).$$

Both predicates take the same $a$ and $U$, but they inspect different components of $U$. The sets $U^+$ and $U^-$ may overlap, and an action may belong to neither. Suppose the user says, “Organize the files in Folder A, but do not delete any files.” Renaming a file in Folder A belongs only to $U^+$. Deleting one belongs to both sets: it falls under the broad organizing task but violates the specific ban. Reading a file in Folder B belongs to neither set. For this last action, $\neg\operatorname{forbidden}(a,U)$ is true while $\operatorname{inScope}(a,U)$ is false. The absence of a ban is not a grant. If $U^-$ instead contained exactly the actions outside $U^+$, the two predicates would be complements by definition, and the separate prohibition check would be redundant. The companion harness checks positive grants for tool effects; it does not implement a separate deny list or a general target-scope check.

This gate is necessary, but it does not guarantee a safe outcome. The harness evaluates preconditions against evidence $e$ because it usually cannot inspect the actual state $s$ directly. Let $L(a)$ be the loss that could result from executing $a$; uncertainty about the state and downstream effects makes $L(a)$ a random quantity. The conditional expectation $\mathbb E[L(a)\mid e]$ is its mean given the available evidence.

A system may set two consequence thresholds, $\tau_0<\tau_1$, for actions that already pass the admission gate. It may execute automatically when estimated loss is below $\tau_0$ and request confirmation between the thresholds. Above $\tau_1$, it refuses the action. An explicit prohibition also requires refusal, regardless of the estimate. Expected loss is difficult to estimate, so high-impact categories such as payments, deletion, disclosure, and publication require auditable rules. Before confirmation, the user should be able to inspect the proposed action and its supporting evidence. The thresholds are design parameters, not universal constants.

For the order request, evidence $e$ should include the order identifier, a status returned by an authoritative service, and the time of that response. The cancellation gate can require that these fields match the requested order and are fresh enough for the service's stated consistency rules. Even then, the order might ship between the status read and the cancellation request. The cancellation service must check the shipping precondition again as part of the same **atomic transaction** that commits cancellation. An atomic transaction checks the condition and changes the state without another operation intervening. This prevents a **race condition**, an error caused by the unpredictable ordering of concurrent operations. A client-side status check limits uninformed proposals; the server-side check addresses the race between observation and execution.

## 1. From language models to agents [15:25](https://www.youtube.com/watch?v=UwfjzyLnvMg&t=925s)

A **language model** maps a sequence of tokens—discrete text units—to probabilities for the next token. Its learned numerical parameters, or **weights**, cannot directly read a file or change a browser. A **tool** is an interface through which the model requests external computation or action. The harness builds a **prompt** (the input sequence supplied to the model), exposes tools, validates and executes calls, returns observations, and determines when the run ends.

Index successive decisions by $t=0,1,\ldots$. Let $\mathcal S$, $\mathcal O$, $\mathcal M$, and $\mathcal A$ be the possible environment states, observations, internal memories, and actions. At step $t$, the hidden environment state is $s_t\in\mathcal S$ and the received observation is $o_t\in\mathcal O$. The agent's memory is $m_t\in\mathcal M$, and its action is $a_t\in\mathcal A$. The symbol $\in$ means “belongs to.”

The environment **transition kernel** $P_{\rm env}(\cdot\mid s_t,a_t)$ is a probability distribution over the next state conditional on the current state and action. Thus $s_{t+1}\sim P_{\rm env}(\cdot\mid s_t,a_t)$, where $\sim$ means “is sampled from.” For goal $g$, the **policy** $\pi$ assigns probabilities to actions conditional on observations through $t$ (written $o_{\le t}$), earlier actions (written $a_{<t}$), memory, and goal: $\pi(a_t\mid o_{\le t},a_{<t},m_t,g)$.

In a language-model agent, the model and harness jointly implement this policy. The model proposes a response or tool call. The harness decides whether the proposal is admissible, dispatches any allowed call, and records the result. If a text answer counts as an action, a chatbot is an agent under the broad definition above. The engineering question is whether the system can observe consequences and choose subsequent actions.

```text
user goal + instructions + tool specifications + history
                         ↓
                   model proposes action
                         ↓
               harness validates / authorizes
                         ↓
                   tool or environment
                         ↓
               observation appended to history
                         ↺
```

### The harness as a controlled stochastic process

The useful unit of analysis is a **run**: the sequence of decisions, tool operations, and observations from a user request until termination. In addition to environment state $s_t$, let $h_t$ be its append-only event history, $m_t$ the memory available at step $t$, $b_t$ the remaining time/token/tool budget, and $u_t$ the current authorization and capability state. Write the run state as

$$x_t=(s_t,h_t,m_t,b_t,u_t).$$

Let $\theta$ denote model weights and $\phi$ harness choices, such as context selection and admission rules. The harness's **context builder** $C_\phi$ selects the model-visible context $c_t$ from goal $g$, history $h_t$, memory $m_t$, available-tool set $\mathcal T_t$, and budget $b_t$: $c_t=C_\phi(g,h_t,m_t,\mathcal T_t,b_t)$. The model distribution $M_\theta(\cdot\mid c_t)$ generates a proposal $z_t$.

Let $e_t$ be the trusted evidence that the harness has extracted from observations and authoritative records by step $t$. The **execution gate** $G_\phi$ considers the proposal, evidence, current authority, and budget. Its decision is the action $a_t=G_\phi(z_t,e_t,u_t,b_t)$. The action may be a tool operation, a clarification request, a rejection, or a final answer. If a tool action executes, the environment supplies the next state and observation according to its transition and observation laws. The harness then appends the action, result, and status to $h_t$, updates $b_t$, and may update $m_t$.

The decomposition is conceptual. A product may distribute these components across an application programming interface (API), a local process, a gateway, and remote **sandboxes**. A sandbox is an execution environment that restricts file, process, or network access. The model can be asked to choose well. A trusted execution gate and sandbox can enforce an effect boundary if they cover every route to that effect. A system that copies a permission rule only into a prompt has changed $C_\phi$; it has not necessarily changed what $G_\phi$ permits.

Define a **trajectory** $\omega=(x_0,z_0,a_0,o_1,\ldots,x_T)$ as one realized run. Let $T$ be its first completion or abort step. It is a **stopping time** when the decision to stop at step $t$ depends only on information available through that step. Let $R_g(\omega)$ measure success on goal $g$, $K(\omega)$ measure resource cost, and $F(\omega)$ equal one if any forbidden event occurs and zero otherwise. The notation $\mathbb E$ averages over possible trajectories and $P(E)$ gives the probability of event $E$. Let $\lambda\ge0$ price resource use, $\delta\in[0,1]$ be the maximum tolerated forbidden-event probability, and $T_{\max}$ the step limit. The operator $\max_{\phi,\theta}$ chooses model and harness settings that maximize the following objective. One design problem is

$$\max_{\phi,\theta}\ \mathbb E[R_g(\omega)-\lambda K(\omega)]\quad\text{subject to}\quad P(F(\omega)=1)\le\delta,\quad T\le T_{\max}.$$

The objective balances expected task success against resource cost. The two constraints play a different role: they limit forbidden events and run length regardless of how much success the system might gain. A high success score cannot compensate for violating either constraint.

The **chance constraint** $P(F(\omega)=1)\le\delta$ limits the probability of a forbidden event. It expresses a design requirement. It does not imply that current systems can estimate the probability accurately. Let $\operatorname{execute}(a_t)$ mean the harness actually invokes action $a_t$, and $\operatorname{authorized}(a_t,u_t)$ mean the current authority state permits it. In temporal logic, $\mathbf G$ means “at every step” and $\Rightarrow$ means implication. Where permission can be checked mechanically, enforce the stronger per-step **invariant** $\mathbf G(\operatorname{execute}(a_t)\Rightarrow\operatorname{authorized}(a_t,u_t))$. Uncertain semantic conditions require calibrated evidence and sometimes human review.

### Workflows, policies, and the boundary of autonomy

A fixed **workflow** is a directed graph of prescribed steps with branches on predefined true/false conditions, or **predicates**. An agent chooses its next operation from observations and may generate a new plan. Both workflows and agents can contain model calls. They differ in who controls the next transition. Let $q_t$ be the workflow's engineered state and $f$ its fixed transition rule: its next action is $a_t=f(q_t,o_t)$. An agentic controller instead samples from an adaptable action policy $\pi_\phi(\cdot\mid h_t,g)$ over a broader action set. More autonomy is justified when the environment varies enough to make enumerated branches costly and when actions remain observable and recoverable.

Anthropic's [agent architecture taxonomy](https://www.anthropic.com/engineering/building-effective-agents) distinguishes prompt chaining (sequential model calls), routing (choosing a branch), parallelization (independent calls), orchestrator-worker (a controller delegates subtasks), evaluator-optimizer (iterative generation and feedback), and open-ended agents. These patterns allocate control flow and feedback differently. Their placement in a taxonomy does not establish a quality ranking.

Most agents do not observe the true environment state. A browser screenshot, log excerpt, or search result is only partial evidence. A **partially observable Markov decision process** assumes the next state depends on the current state and action, while observations provide incomplete evidence about that state. The notation $P(A\mid B)$ means the probability of event $A$ given event $B$. Its **belief distribution** $\beta_t(s)=P(s_t=s\mid o_{\le t},a_{<t})$ assigns a probability to each possible state $s$ after the observed history. This is a distribution over states, not an assertion of subjective certainty. The belief distribution $\beta_t$ is distinct from the resource budget $b_t$.

Let $O(o\mid s',a)$ be the probability of observing $o$ after reaching state $s'$ through action $a$. A prime marks a candidate next state, and $\sum_s$ sums over possible prior states. The symbol $\propto$ means “proportional to.” Bayes' rule gives relative weights for the candidate next states:

$$\beta_{t+1}(s')\propto O(o_{t+1}\mid s',a_t)\sum_sP_{\rm env}(s'\mid s,a_t)\beta_t(s).$$

The sum first predicts how likely each next state is after action $a_t$. The observation factor then increases or decreases that likelihood according to how well state $s'$ explains what the agent observed. Normalize the weights so that they sum to one over all candidate next states; the result is the updated belief distribution.

Practical agents rarely compute this distribution exactly, but the equation explains why an unverified assumption can compromise a long run. A reliable system preserves observations and uncertainty and seeks discriminating evidence before costly actions.

In the order case, suppose the prior probability that an order is unshipped is $0.40$. Suppose a status API reports “unshipped” with probability $0.95$ when the order really is unshipped, but also with probability $0.02$ when it has shipped. Bayes' rule gives $P(\text{unshipped}\mid\text{report unshipped})=0.95(0.40)/[0.95(0.40)+0.02(0.60)]\approx0.969$. The report is strong evidence, but it is not certainty. The client can use it to choose the cancellation branch. The service still needs the atomic precondition check because a probabilistic belief does not prevent a later state change.

## 2. The interaction loop [24:35](https://www.youtube.com/watch?v=UwfjzyLnvMg&t=1475s)

A compact coding agent, such as [mini-swe-agent](https://github.com/swe-agent/mini-swe-agent), constructs messages, requests a model response, executes a validated tool call, appends the resulting observation, and repeats. Its sequence of actions and observations becomes part of the next prompt. Each boundary in this loop has a different meaning. Model output is a proposal. A tool result is evidence about what happened. Neither, by itself, establishes that the task succeeded.

For a request to repair a failing software test, a sound trace has a clear sequence. First inspect the **repository**, the collection of project files and version history, and identify the test command. Run the failing test and read the relevant code. Form a hypothesis, make a focused edit, and rerun the test. Finally, inspect the **diff**, the recorded before/after file changes, and report which checks passed. Each tool output is an *observation*, not an instruction. This distinction becomes crucial when the output is an untrusted webpage or malicious file.

The [companion Python package](agent-lab.html) makes the mathematical separation between proposal and execution explicit. `ToolCall` is a proposed action with a name, arguments, and a call identifier (ID). `FinalAnswer` is a proposed termination. `Authority` is the set of effects granted for one run; it is supplied by the caller and cannot be enlarged by a tool result. These are data types, not commands to the external service.

```python
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class Authority:
    effects: frozenset[str]
    memory_scopes: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Mapping[str, Any]
    call_id: str
    idempotency_key: str | None = None


@dataclass(frozen=True)
class FinalAnswer:
    text: str
```

The complete definitions live in [types.py](https://github.com/az9713/11-768-ai-agents-notes/blob/main/agent_lab/types.py). A `Model` adapter returns either `ToolCall` or `FinalAnswer`. The harness records the proposal before it evaluates the execution gate. In [harness.py](https://github.com/az9713/11-768-ai-agents-notes/blob/main/agent_lab/harness.py), the decisive transition is:

```python
proposal = model.propose(prompt)
label = (
    f"{proposal.name}:{proposal.call_id}"
    if isinstance(proposal, ToolCall)
    else "final answer"
)
state.events.append(Event(step, "proposed", label))
if isinstance(proposal, FinalAnswer):
    if self.verifier(state, proposal):
        state.events.append(Event(step, "finish", "verified"))
        return RunResult(
            proposal.text,
            tuple(state.events),
            tuple(state.observations),
            step,
            state.submitted_input_tokens,
        )
    state.events.append(
        Event(step, "rejected", "unverified answer")
    )
    continue
```

This implements a stopping rule: a final answer terminates the run only after the verifier accepts it. The next branch in the same loop calls `tools.admit(proposal, authority, state.observations)` before `tools.invoke`. Admission implements the predicate $\operatorname{allow}(a,e,U)$ from Section 0. The model never calls the external service directly. On each completed invocation, the harness appends an `Observation` and an event, making the next proposal conditional on the returned evidence. A step limit bounds the run; an unverified answer consumes a step rather than becoming a false success.

The package is executable with `python -m agent_lab.order_demo`. Its deterministic proposal source stands in for a language-model API so the example needs no credentials. The first case produces `get_order_status:ok`, `cancel_order:unknown`, and `get_order_status:ok`: the cancellation committed, its response was lost, and the agent reconciled by reading the authoritative state. The second case reads a shipped status and selects the return-policy branch. [The executable checks](https://github.com/az9713/11-768-ai-agents-notes/blob/main/tests/test_agent_lab.py) exercise permission rejection, verification failure, and both outcomes.

A deployed loop also needs time and money budgets. It needs **retries**, which repeat an operation after failure, and **cancellation**, which stops in-flight work. Unique **call IDs** pair requests with results. **Concurrency controls** govern overlapping actions. **Checkpoints** save state for resumption, while **trace capture** records the action sequence. Sandboxing limits the effects of executed code. These requirements cannot be inferred from the short loop alone. This package implements a step budget and an in-process event record; it does not claim durable recovery after process failure or provide a production sandbox.

A robust implementation represents each invocation as a state transition `proposed → admitted → running → succeeded | failed | uncertain`. The `uncertain` state matters when a **timeout**, an enforced waiting limit, occurs after a remote service might have committed a **side effect**, a change outside the model's text. A **server-recognized idempotency key** is a transaction identifier that makes repeated requests count as one operation. A **reconciliation query** checks the authoritative state before a retry. After an uncertain result, the harness needs one of these protections before repeating a consequential call.

The run itself needs the states `active → waiting_for_user | completed | failed | cancelled`. **Streaming** sends partial output before the turn ends. A streamed sentence does not establish that a tool action or the run has completed. [OpenClaw's agent-loop documentation](https://docs.openclaw.ai/concepts/agent-loop) describes a serialized per-session loop, lifecycle events, session persistence, streaming, timeout handling, and a separate completed-turn signal for Codex-backed runs.

Let $n$ be the number of indispensable steps and $p$ their common success probability. If their successes are independent, meaning one step's result does not alter the probability of another's, then $P(\text{whole task succeeds})=p^n$. At $p=0.97$ and $n=30$, this is about $0.40$. Independence is an oversimplification: errors can be correlated, and recovery can repair a mistake. Nevertheless, local accuracy alone cannot establish whole-task reliability. Verification, reversible actions, retries informed by new evidence, and checkpoints change the effective error process.

## 3. Six dimensions of agent capability [30:45](https://www.youtube.com/watch?v=UwfjzyLnvMg&t=1845s)

An interaction loop is easy to implement. Reliability requires more: accurate tool calls, coherence over long histories, adaptation to user and project requirements, task management, and sound interpretation of the environment. The agent must also respect safety boundaries. Safety is a property of the full system: obtaining the requested result by violating a user's authorization is failure.

| Capability | Observable failure | Engineering test |
|---|---|---|
| Tool calling | wrong function, invalid arguments, ignored result | tool-choice and argument test cases |
| Long-context coherence | forgets a constraint or repeats work | continuation after long histories |
| Customizability | ignores a project rule or user preference | matched runs with and without the rule |
| Task management | loses dependencies or stops early | end-to-end multi-step verifier |
| Environment understanding | mistakes stale or partial observations for current state | hidden-state and changed-state scenarios |
| Safety | performs an unauthorized or dangerous action | adversarial and permission-boundary tests |

Let $S$ be a numerical task-success score, $C$ monetary and computational cost, $D$ delay, and $R$ a numerical measure of harm or risk. Let nonnegative weights $\lambda_C,\lambda_D,\lambda_R$ set the relative importance of these quantities, and let $J$ denote overall expected utility. One design objective is $J=\mathbb E[S]-\lambda_C\mathbb E[C]-\lambda_D\mathbb E[D]-\lambda_R\mathbb E[R]$, where $\mathbb E$ denotes an average over possible runs.

The weighted objective describes preferences among permitted outcomes. It does not replace a hard constraint: impose $P(\text{forbidden action})\le\epsilon$ separately, where $\epsilon$ is a specified maximum probability. A trusted execution gate can enforce mechanically checkable prohibitions; a model promise cannot. Evaluation should also report the outcome vector $(S,C,D,R)$ before applying the weights. Separate success, cost, delay, and risk measurements show which trade-offs produced an apparent improvement. A single leaderboard rank conceals those trade-offs.

## 4. Training and harness engineering [35:40](https://www.youtube.com/watch?v=UwfjzyLnvMg&t=2140s)

An agent's capability can reside in its model weights or in its **runtime system**, the software that operates during a run. **Pretraining** learns weights from large general data. **Supervised fine-tuning** adjusts those weights using examples with desired outputs. **Reinforcement learning** adjusts them using rewards for sampled behavior. **Harness engineering** changes instructions, tools, memory, workflow, validation, and execution environment without changing weights. A new failure can often be addressed in the harness immediately. Training may be justified when the failure is frequent, general, and difficult to correct with runtime controls. The relevant choice depends on quality, **latency** (elapsed response time), maintenance, and deployment scale.

Suppose a harness fix costs $F_H$ once and adds $c_H$ per task, while a training change costs $F_T$ once and adds $c_T$ per task. Training is cheaper over $N$ tasks only when $F_T+Nc_T<F_H+Nc_H$, subject to equal quality and risk. The threshold is $N>(F_T-F_H)/(c_H-c_T)$ when $c_H>c_T$. These quantities include evaluation and maintenance, not just compute. Local instructions and tools can be attractive for a solo developer, while a model provider serving millions of runs may justify training. Neither approach is intrinsically more capable.

The **reasoning-and-acting (ReAct)** approach interleaves reasoning with environment actions. Its [original paper](https://arxiv.org/abs/2210.03629) provides a historical bridge: interleaved reasoning and environment actions can improve performance over either alone in the evaluated settings. [Toolformer](https://arxiv.org/abs/2302.04761) takes a different route, training a model to decide when and how to insert API calls. These papers illustrate changes to model behavior through inference-time interaction and training. Their benchmark gains do not transfer automatically to every modern agent.

## 5. The agent system [52:00](https://www.youtube.com/watch?v=UwfjzyLnvMg&t=3120s)

An agent consists of more than a model and a loop. Its execution sandbox bounds what generated code can touch. **Model serving** is the infrastructure that runs inference, the computation of model outputs from inputs. Serving determines latency and **throughput**, the amount of work completed per unit time. **Observability** records requests, calls, outputs, costs, and failures. Training systems need reproducible environments and outcome signals. A **trust boundary** separates inputs or components with different authority, such as a user's request and a retrieved webpage. A system diagram is useful only if it exposes these responsibilities and boundaries.

Treat model output as a proposal, tool output as data, and the user's instruction as authority. For every tool, specify a **contract** $\mathcal C=(\text{schema},\text{preconditions},\text{side effects},\text{permissions},\text{postconditions})$. The schema describes argument shape, and preconditions state what must hold before the call. Side effects list possible external changes. Permissions specify allowed callers and actions. Postconditions describe the result to verify. The harness validates the proposal against $\mathcal C$ before execution and verifies postconditions afterward. Sandboxing can limit damage if the model or a tool is wrong. It cannot make malicious retrieved text trustworthy. A **threat model** identifies who controls each input, what secrets the tool can access, and what irreversible side effects it can produce.

### Three production decompositions

| Implementation | Durable state and control | Engineering consequence |
|---|---|---|
| [OpenAI Codex agent loop](https://openai.com/index/unrolling-the-codex-agent-loop/) | A harness repeatedly assembles model input, interprets tool calls, executes tools, and records results; [Codex harness internals](https://openai.com/index/unlocking-the-codex-harness/) describe typed events and sandbox-aware execution. | The model's proposed command and the process allowed to execute it are separate components. Trace events can identify the step that failed. |
| [Anthropic Managed Agents](https://www.anthropic.com/engineering/managed-agents) | A durable session event log is separate from context transformations in the harness and from execution environments exposed as tools. | Replayable history and disposable execution environments permit recovery and selective context construction; attaching a container only when needed reduces startup cost. |
| [OpenClaw runtime](https://docs.openclaw.ai/concepts/agent) | A configured agent has a workspace, bootstrap files, session store, tool policy, and optional sandbox. | Files such as `AGENTS.md` and `MEMORY.md` contribute context, while tool policy and sandbox determine actual executable authority. |

These products differ in deployment, trust boundary, and persistence format. Their common pattern is a controller that owns state transitions and a model that proposes the next step. The architecture suggests a practical design rule: write an event record **before and after** a consequential action, with a stable call ID, precondition evidence, and observed outcome. Without those records, a resumed agent may confuse “proposed,” “sent,” and “committed.”

### Delegation changes the control plane

Multiple agents introduce a second policy: the parent must decide which subtask to **delegate** (assign to another agent), which context to transmit, which capabilities to grant, and how to verify the result. Define a directed task graph $\mathcal D=(\mathcal V,\mathcal E)$, where each node $v\in\mathcal V$ is a subtask and each edge $i\to j$ in $\mathcal E$ means that task $j$ depends on task $i$. A delegation contract for node $v$ is $(g_v,c_v,u_v,\operatorname{verify}_v)$: its local goal, supplied context, authority state, and result checker.

Let $\operatorname{cap}(u)$ denote the rights contained in authority state $u$. **Least privilege** requires $\operatorname{cap}(u_v)\subseteq\operatorname{cap}(u_{\rm parent})$, where $A\subseteq B$ means every right in $A$ is also in $B$. Task coherence imposes a separate requirement: every dependency result needed by $v$ must be present in or retrievable from $c_v$. These conditions are necessary, not sufficient. Parallel writes can conflict, and a passing local checker may still break the integrated artifact.

The parent should treat a child result as an observation with **provenance**, a record of its source, not as an instruction to expand its own permissions. The **critical path** of $\mathcal D$ is its longest dependency chain and bounds the possible speedup. Parallelization helps only when the saved wall time exceeds coordination, duplication, and integration costs. Anthropic's [building-effective-agents patterns](https://www.anthropic.com/engineering/building-effective-agents) use orchestrator-worker designs for tasks whose subtasks are not predictable in advance. OpenAI's [Symphony orchestration description](https://openai.com/index/open-source-codex-orchestration-symphony/) illustrates a different scale: a task board acts as a control plane that assigns ongoing coding work to agents. Neither pattern eliminates the need for a final verifier over the combined result.

### Harness changes as causal interventions

If a run fails, the trace should help locate the failure. The model may have selected a poor proposal, or the context builder may have omitted needed evidence. A **tool adapter**, software translating a model call to an external API, may have misrepresented that API. Execution may have failed despite a valid call. The verifier may also have misclassified the result. A prompt rewrite is a plausible **intervention**, a deliberately changed system component, only when the failure is attributable to information or instructions that the prompt controls.

For task $i$ and harness configuration $\phi$, let $Y_i(\phi)$ equal one when the run succeeds and zero otherwise, with model, environment, and budget held fixed. Let $\mathbb E_i$ mean averaging over the evaluated tasks. The **estimand**, the quantity an experiment seeks to measure, is the average effect $\Delta_H=\mathbb E_i[Y_i(\phi_1)-Y_i(\phi_0)]$ of changing configuration $\phi_0$ to $\phi_1$. Pairing the same tasks across both conditions reduces variation due to task difficulty. Repeated runs estimate **stochastic variance**, variation caused by random outputs or environments. Report side effects, cost, and latency alongside $\Delta_H$. A new prompt that improves average success while increasing unauthorized actions is not an unqualified improvement.

### Verification as a stopping rule

Define $\operatorname{verify}(h_t,g)$ as a checker that inspects history $h_t$ for evidence that goal $g$ is complete and returns `pass`, `fail`, or `unknown`. A harness should allow success termination only if $\operatorname{verify}(h_t,g)=\text{pass}$ for a goal with a machine-checkable outcome. For a coding task, verification may require passing tests and an inspected diff. For a factual answer, it may require sources that support each claim. For a payment, it may require a transaction receipt. A verifier is itself fallible, so report both the artifact and the evidence class. OpenAI's [harness-engineering account](https://openai.com/index/harness-engineering/) emphasizes making repository guidance and validation legible to the agent. This can be read formally as reducing uncertainty in $C_\phi$ and improving the checker's ability to distinguish success from failure.

When modifying a harness, compare old and new on the *same task distribution* with the same model, tool versions, and budgets. Record full trajectories, not only success flags, to distinguish model-choice failure from tool-runtime failure. For stochastic runs, report a **confidence interval**, a range constructed by a procedure with stated long-run coverage, or repeated paired trials. Compare $P(\text{task success}\mid\text{correct tool choice})$ with $P(\text{correct tool choice})$ to locate the weak layer.

## 6. What this abstraction captures and misses

The loop model captures feedback, action, observation, and the software required to affect an environment. It does not require the agent to expose private reasoning, use one model, or hold its entire history in a single prompt. The six capabilities also interact: poor observation can appear to be a planning failure, while an unsafe tool boundary can turn a minor model error into a serious incident. Tool design, context management, and cross-task memory determine how these interactions play out in a working system.

### Worked design exercise: a customer-support agent

Consider an agent asked, “Cancel my order if it has not shipped; otherwise tell me the return options.” Let the actual order state be $s\in\{\text{unshipped},\text{shipped}\}$. The status `unknown` describes what the agent has observed; it is not a third actual order state. A naive language model may guess from an old confirmation email. The agent instead calls `get_order_status(order_id)`, where `order_id` identifies the requested order, before any side effect. If the returned status is unshipped, `cancel_order(order_id)` changes the order. If the status is shipped, the agent retrieves the current return policy and responds. If the status request times out, the correct next action is to retry or report uncertainty, not to choose a branch by confidence in its own prose.

The following trace separates external state, observed evidence, and permissible next action. “Unknown” is an evidence state, not an external shipping status.

| Trusted evidence after a step | Permissible next operation | Required check |
|---|---|---|
| No current status for the requested order | `get_order_status(order_id)` | The identifier matches the user's request. |
| Fresh “unshipped” status | `cancel_order(order_id)` | The user's authorization is current; the service atomically checks that the order remains unshipped. |
| Fresh “shipped” status | Retrieve the current return policy | The answer cites the applicable policy and does not claim cancellation. |
| Timeout after a cancellation request | Query transaction or order state | Do not retry a potentially committed cancellation on a guess. |

A safety invariant can be expressed in temporal logic. Let $C_{\rm cancel}$ mean “a cancellation is executed,” $E_{\rm unshipped}$ mean “a fresh status observation says unshipped,” and $A_{\rm cancel}$ mean “the user authorized cancellation of this order.” The desired property is $\mathbf G(C_{\rm cancel}\Rightarrow(E_{\rm unshipped}\land A_{\rm cancel}))$, where $\mathbf G$ means “at every step.” A second invariant, $\mathbf G(\text{answer shipped}\Rightarrow\text{fresh shipped evidence})$, rules out confidently reporting a stale state. The model can *propose* a cancellation, but the harness can enforce the first invariant if it stores trusted status evidence and validates the call. The second depends partly on response checking and is harder to enforce perfectly.

The observation model may be unreliable or delayed. A `get_order_status` response is evidence about $s$, not $s$ itself. For ordinary orders, a fresh API status may be enough. For a costly or irreversible action, require stronger evidence or a confirmation step. The appropriate amount of verification depends on consequence, observability, and reversibility.

### Exercises

1. Formalize the difference between a search tool that returns text and a purchase tool that creates an order. Which fields belong in the action contract? Which checks belong before and after execution?
2. Starting from $p^n$, derive the whole-task success probability when every failed step is independently retried once. Explain why correlation and irreversible actions invalidate the simple result.
3. Design an experiment that distinguishes a poor model from a poor harness for a coding agent. Name a measurable intervention, an outcome, and a confounder.

### Solutions and discussion

1. A search call has a query schema, read permission, a result provenance field, and a postcondition about receiving a well-formed response. It should not mutate external state. A purchase call needs an order identifier, item identities, quantity, total price, payment method reference, and a server-recognized idempotency key. Before execution, check the user's purchase authorization and the displayed price against the proposed order. After execution, reconcile the transaction identifier and receipt with the seller's authoritative order record. A syntactically valid order payload proves none of these semantic or authorization conditions.
2. Assume each indispensable step succeeds with probability $p$ on one attempt, a failed first attempt is retried once, and attempts and steps are mutually independent. A step succeeds either on the first attempt, with probability $p$, or after one failure and a successful retry, with probability $(1-p)p$. Its success probability is therefore $p+(1-p)p=1-(1-p)^2$. For $n$ steps, whole-task success is $[1-(1-p)^2]^n$. At $p=0.97$ and $n=30$, this is approximately $0.973$. Correlated failures make the product invalid. Retrying a non-idempotent action may create a second side effect rather than repair the first attempt.
3. Hold the model, coding tasks, tool versions, and resource budgets fixed. Compare a harness that supplies the relevant test command and captures tool errors with one that omits those aids. The intervention changes context assembly and error handling; the outcome is verified task success, recorded separately from cost and unauthorized actions. Pair tasks across conditions and repeat stochastic runs. A simultaneous repository change is a confounder because it can change test difficulty independently of the harness.

### Further reading

- [ReAct](https://arxiv.org/abs/2210.03629): interleaving reasoning and acting.
- [Toolformer](https://arxiv.org/abs/2302.04761): learned API selection and use.
- [mini-swe-agent repository](https://github.com/swe-agent/mini-swe-agent): a compact implementation of a coding-agent loop.
