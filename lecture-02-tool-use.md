# Lecture 2 — Tool use for language-model agents

*Independent study chapter · Based on a CMU 11-768: AI Agents lecture by Graham Neubig, 27 August 2026 · [Course and attribution](index.html#sources)*

## 0. Tools extend the action space [00:30](https://www.youtube.com/watch?v=jXChFB4JSyw&t=30s)

A **language model** predicts text one **token** (a discrete text unit) at a time. A **tool** is an interface through which it requests an external program's operation. Consider an agent asked to prepare a grocery cart for review. Finding products requires **information retrieval**, or locating stored material. Creating a cart changes the shopping application's state. Placing the order would make a financial commitment that the request has not authorized.

A **capability** is an operation the system can perform; a **side effect** is a change to external state. Select tools according to the capabilities the task needs and the side effects the user permits. An **application programming interface (API)** is a programmatic collection of operations. Exposing every available API increases the number of choices the model must make and the operations the harness must control.

The model chooses tokens. A **harness** is the software that controls the agent's model/tool loop and interprets some model output as a proposed invocation. Let $q$ be that proposal and $\mathcal T$ the set of available tools. A **dispatcher**, the component that selects and invokes tool implementations, maps $q$ to $(\tau,x)$, where $\tau\in\mathcal T$ is a tool, $\in$ means “belongs to,” and $x$ is its argument. Execution produces an **observation** $y$ or an error. Proposing, validating, and executing are separate operations. A proposed call may be well-formed JavaScript Object Notation (JSON), a structured text representation. Its valid syntax does not grant permission to act.

Let $s_t\in\mathcal S$ be the environment state at step $t$. Let $h_t$ be the event history, $m_t$ retained memory, $b_t$ the remaining resource budget, and $u_t$ the current authority. Together they form the run state $x_t=(s_t,h_t,m_t,b_t,u_t)$. Let $\mathcal X_\tau$ be tool $\tau$'s argument space, $\mathcal Y_\tau$ its successful-result space, and $\mathcal E_\tau$ its error-result space. For sets $A$ and $B$, write $A\sqcup B$ for a tagged union whose members retain which set they came from. Write $\Delta(Z)$ for the set of probability distributions on $Z$. A possibly stochastic tool has the type

$$\tau:\mathcal S\times\mathcal X_\tau\to\Delta\!\left(\mathcal S\times(\mathcal Y_\tau\sqcup\mathcal E_\tau)\right).$$

The successful operation is **partial**: some proposed arguments produce errors rather than a successful result. Including state in either outcome allows for a timeout or failure after an external change. A **pure** calculator preserves $s_t$ when it succeeds. A file edit or purchase changes the environment state. An **adapter**, software translating an external API into a model-facing tool, chooses names, descriptions, argument schemas, error representations, and **credentials**, secrets used to authenticate calls. Those choices affect model behavior even when the underlying API is unchanged.

## 1. Narrow interfaces and code as a meta-tool [10:35](https://www.youtube.com/watch?v=jXChFB4JSyw&t=635s)

Many narrow API calls can be replaced by one **code-execution interface**, a tool that runs a supplied program. Code expresses loops, variables, library calls, and **parallelism**, overlapping independent operations, within an action. [CodeAct](https://arxiv.org/abs/2402.01030) and [What Are Tools Anyway?](https://arxiv.org/abs/2403.15452) examine this expressive **action space**, the possible actions the agent can select. Its advantage in composition comes with a larger security and resource footprint.

| Interface | Strong when | Main failure mode |
|---|---|---|
| Narrow typed tool | action has stable fields and consequential side effects | a long sequence requires many model turns |
| General code interpreter | task needs control flow and composition | much larger action space and dependency/sandbox risk |
| Hybrid | code handles data processing; narrow tools handle consequential actions | boundary design and duplicate capabilities |

Suppose a tool has $k$ independent argument fields and field $i$ permits $r_i$ discrete choices. The notation $\prod_{i=1}^k$ means multiply one factor for each field from $1$ through $k$. The number of possible argument combinations is $\prod_{i=1}^k r_i$. General-purpose code has a much larger set of possible programs. This expressivity can reduce model/tool **round trips**, which are repeated model requests separated by tool execution. It also expands the set of reachable side effects. The second effect matters even when the generated program is short.

A hybrid design gives code access to pure or **sandboxed** computations—operations restricted by an execution environment—and reserves payments, publication, deployment, and data disclosure for constrained calls with explicit policy checks. CodeAct's measured gains are specific to its evaluated tasks and risk profile.

An **effect** is an externally observable operation that a tool may perform. Let $\epsilon(\tau)$ be tool $\tau$'s possible-effect set. Let $r$ name a resource such as a file, $d$ a network destination or disclosure recipient, and $v$ an amount of money. Here $A\subseteq B$ means every member of set $A$ belongs to set $B$, and $A\cup B$ means the union of their members. One effect vocabulary is $\epsilon(\tau)\subseteq\{\operatorname{read}(r),\operatorname{write}(r),\operatorname{network}(d),\operatorname{spend}(v),\operatorname{disclose}(d)\}$.

A code interpreter with unrestricted filesystem and network access has a broad effect set even if the generated program looks harmless. A restricted sandbox can narrow the interpreter's effective set. A typed payment tool can separately enforce amount and recipient checks. If $\tau_2\circ\tau_1$ means executing $\tau_1$ followed by $\tau_2$, its possible effects lie in $\epsilon(\tau_1)\cup\epsilon(\tau_2)$ *provided* both descriptions account for every effect, including effects of failure and retries. A list of tool names does not reveal what each call can touch. A capability inventory should record the resources and destinations reachable through every tool.

For the grocery request, `search_products` has a read effect and `create_grocery_cart` has a cart-write effect. `place_order` has at least a spending effect. The user's requested effect set contains cart creation but does not contain spending. A broad browser or shell tool may technically reach a checkout page, yet that reachable capability is outside the request's authority. A narrow cart-creation interface makes the intended effect boundary easier to enforce and test.

## 2. Tool-call protocols and contracts [16:00](https://www.youtube.com/watch?v=jXChFB4JSyw&t=960s)

A tool definition supplies a name, description, and **argument schema**, a machine-readable rule for the form of its input. The model emits a call in its provider-specific message format. The harness resolves and executes the call. A result message refers to the call's unique **identifier** (ID). The ID is essential when multiple calls return out of order. Provider encodings differ, but a reliable harness preserves the association between each call and its result.

```json
{
  "name": "get_weather",
  "description": "Get current weather for a city",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string"},
      "units": {"type": "string", "enum": ["C", "F"]}
    },
    "required": ["city", "units"],
    "additionalProperties": false
  }
}
```

In this JSON schema, `city` must be text and `units` must be either `C` or `F`. Both fields are required. The setting `additionalProperties: false` forbids any other fields. A **predicate** is a true/false condition. For argument $x$, user goal $g$, and authority state $u$, define $V_{\rm syn}(x)$ to mean syntactically valid, $V_{\rm sem}(x,g)$ to mean suitable for $g$, and $V_{\rm auth}(x,u)$ to mean permitted by $u$. Execute only if all three hold. The schema checks the call's shape. It cannot establish that “Pittsburgh” is the intended city. It also cannot establish the caller's authority or the accuracy of the returned weather. A model may help interpret intent, but the permission boundary should live in trusted code.

### A contract with proof obligations

A **Hoare-style contract** relates a **precondition**, which must hold before execution, to a **postcondition**, which must hold after successful execution. Let $\operatorname{Pre}(s,x,u)$ be the precondition for environment state $s$, argument $x$, and authority $u$. Let $s'$ and $y$ be the resulting state and returned value. The predicate $\operatorname{Post}(s',y)$ states the required postcondition. The notation $\{\operatorname{Pre}\}\ \tau(x)\ \{\operatorname{Post}\}$ specifies that if $\operatorname{Pre}$ holds and tool $\tau$ terminates successfully, $\operatorname{Post}$ must hold. This is a partial-correctness claim: it does not assert that the tool terminates.

The specification also leaves timeouts, compromised implementations, and other errors outside its success case. A full contract therefore defines allowed effects, error classes, and evidence for checking $\operatorname{Post}$. For `cancel_order(id)`, where `id` identifies an order, $\operatorname{Pre}$ might require a current status response for that order and authority scoped to cancellation. $\operatorname{Post}$ could be a confirmed cancelled status. A terminal conflict, such as “already shipped,” is an error outcome rather than proof of $\operatorname{Post}$. A text response saying “cancelled” does not establish $\operatorname{Post}$.

**Capability-based authorization** grants explicit rights to perform classes of operations. Let $\operatorname{cap}(u)$ be the rights granted to this run and $\operatorname{req}(\tau,x)$ the rights required by a call. The harness admits the call only when $\operatorname{req}(\tau,x)\subseteq\operatorname{cap}(u)$. A broad `shell` capability—permission to execute operating-system commands—is hard to reason about because its required rights depend on the command and environment. Sandboxing, **directory allowlists** (permitted filesystem paths), **network egress restrictions** (permitted outbound destinations), and narrow **wrappers** (programs that expose only selected operations) make the effective rights more inspectable. The model may argue that an action is useful. That argument does not enlarge $\operatorname{cap}(u)$.

An operation exposed through Hypertext Transfer Protocol (HTTP), a **Model Context Protocol (MCP)** tool, and a local shell command can all have contracts. **OpenAPI** documents HTTP service operations. **JSON Schema** provides rules for validating the shape of JSON values. Neither establishes user consent or correct business semantics. MCP exposes agent-facing operations through a client-server protocol; its server still has to enforce the upstream API's permissions.

The enforcement point also depends on who executes the operation. [OpenAI's account of equipping the Responses API with a computer environment](https://openai.com/index/equip-responses-api-computer-environment/) describes a **callback boundary**: with custom tools, the API yields a proposal and the client-side harness executes it. An implementation that treats the returned call as already authorized collapses that boundary.

Let $id$ be a unique call ID, $\tau$ the selected tool, $x$ its argument, and $\kappa$ an optional **idempotency key**: a server-recognized identifier that makes repeated submissions of one logical operation count as one transaction. Represent a call by $(id,\tau,x,\kappa)$. If a network timeout occurs after execution, a blind retry can duplicate a purchase or email. For a side-effecting operation, either the downstream service must honor $\kappa$, or the harness must **reconcile** the state by querying an authoritative record before retrying. Record tool version, time, arguments (with secret fields removed), outcome, and verification evidence. A completed network exchange does not, by itself, confirm the postcondition. The harness needs evidence about the state the downstream service actually reached.

The admission rule is implemented in [tools.py](https://github.com/az9713/11-768-ai-agents-notes/blob/main/agent_lab/tools.py). `spec.fields` maps argument names to required Python types. `spec.effects` is the set $\operatorname{req}(\tau,x)$ for this simplified tool specification; `authority.effects` is $\operatorname{cap}(u)$. `spec.precondition` is a function of the proposed call and previously recorded observations. The code rejects an unknown tool, an altered argument shape, an ungranted effect, or missing evidence *before* calling the handler.

```python
def admit(self, call, authority, evidence):
    spec = self._tools.get(call.name)
    if spec is None:
        raise CallRejected("unknown tool")
    if set(call.arguments) != set(spec.fields):
        raise CallRejected("argument fields do not match schema")
    for field, expected_type in spec.fields.items():
        if type(call.arguments[field]) is not expected_type:
            raise CallRejected(f"wrong type for {field}")
    if not spec.effects <= authority.effects:
        raise CallRejected("effect not authorized")
    if not spec.precondition(call, evidence):
        raise CallRejected("precondition lacks trusted evidence")
    return spec
```

The order example registers `cancel_order` with the effect `order.cancel` and with a precondition requiring a matching `get_order_status` observation. This client-side check establishes *admissibility from evidence*. It does not establish that the order remains unshipped. In [order_demo.py](https://github.com/az9713/11-768-ai-agents-notes/blob/main/agent_lab/order_demo.py), the service therefore holds a lock while checking its current status and revision and changing the state:

```python
with self._lock:
    old = self._committed.get(key)
    if old is not None:
        old_order_id, old_revision, old_result = old
        if (old_order_id, old_revision) != (
            order_id, expected_revision
        ):
            raise ToolFailure(
                "idempotency key reused with new arguments"
            )
        return old_result
    if self.order.status != "unshipped":
        raise ToolFailure("order no longer unshipped")
    if self.order.revision != expected_revision:
        raise ToolFailure("status observation is stale")
    self.order.status = "cancelled"
    self.order.revision += 1
```

`expected_revision` is the revision number read from the status service. The equality check implements **optimistic concurrency control**: a write based on an older revision is rejected. The lock makes the check and update atomic in this in-memory example. A real distributed service needs its own atomic database transaction or conditional update; a client-process lock cannot protect a remote database. The committed-request ledger keyed by $\kappa$ prevents a repeated *identical* cancellation from changing state twice. The ledger itself is only in memory and would need durable storage in a service that survives restarts.

Suppose `create_grocery_cart(items)` times out after the server creates cart `C17`. The client has observed no result, but the server has changed state. A blind retry might create a second cart `C18`. If the service supports idempotency, the harness resends the same key $\kappa$ and the same arguments; the service should return the transaction associated with `C17`. If it does not, the harness queries the user's active carts and reconciles the intended item list before issuing another write. This trace explains why `timeout` and `not committed` are different states.

### Retry semantics and exactly-once illusions

Classify calls as **read-only** (no intended state change), **idempotent writes** (repetition has the same final state as one execution), or **non-idempotent writes**. For a deterministic successful state update $f_x(s)$ caused by fixed argument $x$ in state $s$, idempotence means $f_x(f_x(s))=f_x(s)$. This equation concerns the state update, not the tool's returned message or error. `set_status(order, "cancelled")` may be idempotent because repeating it leaves the same final status. `charge_card(amount)` usually is not: repeating it can create a second charge.

A network timeout means the client does not know whether the server **committed**, or made durable, the action. The result is an *epistemic* `unknown`: the client lacks knowledge of the downstream outcome. It has no evidence that the operation failed. For non-idempotent operations, use a server-recognized key $\kappa$ or query the authoritative record before retrying. Even with a key, test its **retention window**, the interval during which repeated requests are recognized as duplicates, and the server's exact matching rule.

The event log should store `call_id`, $\kappa$, a **request hash** (a compact fingerprint of the submitted arguments), an attempt number, a **transport outcome** (whether the network exchange completed), and a downstream transaction identifier. On restart, the harness can reconcile an unfinished call rather than asking the model to guess. Model text is an unreliable substitute for an external side-effect ledger.

The runnable example distinguishes a *known pre-commit rejection* from an outcome whose state is unknown to the caller. `ToolFailure` represents the first case, and `OutcomeUnknown` represents the second. An unexpected handler exception is conservatively classified as `unknown`: after an arbitrary exception, the harness cannot prove that no external effect occurred. The corresponding [tool adapter](https://github.com/az9713/11-768-ai-agents-notes/blob/main/agent_lab/tools.py) turns both classes into observations rather than silently treating an exception as success.

```python
try:
    data = spec.handler(call)
    return Observation(call.call_id, call.name, "ok", data)
except OutcomeUnknown as exc:
    return Observation(
        call.call_id, call.name, "unknown", {"error": str(exc)}
    )
except ToolFailure as exc:
    return Observation(
        call.call_id, call.name, "error", {"error": str(exc)}
    )
except Exception as exc:
    return Observation(
        call.call_id, call.name, "unknown", {"error": str(exc)}
    )
```

In the simulated timeout, `OrderService.cancel` changes the state and then raises `OutcomeUnknown`. The next proposal is a status read, not another cancellation. The final-answer verifier checks the service's current state. This is an executable counterexample to the mistaken implication `timeout ⇒ no commit`.

## 3. Grammar-constrained generation [27:00](https://www.youtube.com/watch?v=jXChFB4JSyw&t=1620s)

JSON Schema describes allowed argument forms. **Grammar-constrained decoding** blocks next tokens that cannot lead to a complete output in a specified formal language, such as a restricted JSON format. A **finite-state machine** has only finitely many internal states. A **stack-based parser** additionally records a potentially unbounded sequence of open structures. Arbitrarily nested JSON requires the latter in the general case. [XGrammar](https://arxiv.org/abs/2411.15100) describes preparing a grammar in advance and maintaining parser stacks during token generation.

Let $L$ be the set of complete valid outputs, $p$ the output prefix generated so far, $v$ a candidate next token, and $z$ any possible continuation. Juxtaposition $pvz$ means concatenating these strings. Candidate $v$ is allowed when some $z$ makes $pvz\in L$. Let $P(v\mid p)$ be the model's original next-token probability, let $w$ range over all candidate tokens, and let $\mathbf 1[A]$ equal one if condition $A$ is true and zero otherwise. The symbol $\exists$ means “there exists,” and $\sum_w$ sums over candidate tokens. Let $P'$ denote the constrained distribution obtained by zeroing forbidden tokens and renormalizing the remaining probabilities:

$$P'(v\mid p)=\frac{P(v\mid p)\mathbf 1[\exists z:pvz\in L]}{\sum_wP(w\mid p)\mathbf 1[\exists z:pwz\in L]}.$$

The numerator retains the model's original probability for a token that can lead to a valid completion and sets it to zero otherwise. The denominator is the total probability of all still-allowed tokens. Dividing by it makes the retained probabilities sum to one, provided at least one continuation is available.

A **pushdown automaton** is a state machine with a stack. Its stack can remember arbitrarily deep bracket structure. **Tokenization**, the conversion between text and model tokens, complicates the test: a single token may contain several bytes or punctuation characters, so the grammar checker must account for the token's entire byte sequence.

Grammar masking can guarantee *syntactic membership* in the chosen language if implemented correctly and generation reaches a complete accepting state. It cannot guarantee factual arguments, tool appropriateness, authorization, safe effects, or completion before a token limit. A run may end with an incomplete but valid prefix of JSON. Parse errors and business-rule errors therefore require separate evaluation.

## 4. HTTP APIs, OpenAPI, and MCP [35:45](https://www.youtube.com/watch?v=jXChFB4JSyw&t=2145s)

An **HTTP API** exposes operations through network addresses called **endpoints**. HTTP is the Hypertext Transfer Protocol used for web requests. OpenAPI documents those operations and their schemas. An MCP server exposes agent-facing tools and resources through a client-server protocol. **Authentication** establishes the caller's identity. **Authorization** decides what that caller may do. These layers have related but distinct contracts: MCP packages access and discovery without replacing upstream authentication or authorization.

If an MCP server calls a weather or payment API, distinguish the agent-to-MCP credential from the MCP-to-upstream credential. Let $K_M$ authorize the client to call the MCP endpoint and $K_U$ authorize the server to call the upstream API. Keeping $K_U$ server-side limits accidental exposure to model context. Apply least privilege at both boundaries. The server must still check a syntactically valid call against the operation the user permitted. Consult the current [MCP tools specification](https://modelcontextprotocol.io/specification/draft/server/tools), [OpenAPI specification](https://spec.openapis.org/oas/latest.html), and [JSON Schema validation specification](https://json-schema.org/draft/2020-12/json-schema-validation.html) when implementing because protocol details evolve.

## 5. Parallel execution and dependencies [45:00](https://www.youtube.com/watch?v=jXChFB4JSyw&t=2700s)

**Concurrent** calls overlap in time. **Independent** calls need neither one another's results nor conflicting changes to shared state. **Dependent** calls must wait for a predecessor's result or effect. Weather in two cities can be fetched together, whereas a flight booking cannot occur before the traveler chooses among returned flights. Call IDs preserve the relationship between requests and results even when completion order differs from submission order.

Represent the $n$ calls by a **directed acyclic graph**. Its nodes $V=\{1,\ldots,n\}$ are calls. An arrow $i\to j$ means that call $j$ requires the result or side effect of call $i$. The graph has no directed cycles. Let $d_i$ be call $i$'s duration and $\mathcal P$ the set of dependency paths, each a sequence of calls linked by arrows. The **critical path** is the path with greatest total duration. The notation $\sum_{i\in\gamma}d_i$ adds durations along path $\gamma$, and $\max$ selects the largest path sum. Any valid schedule, which respects all arrows, has latency at least $\max_{\gamma\in\mathcal P}\sum_{i\in\gamma}d_i$.

This lower bound concerns elapsed time under the stated dependencies. Running every available call at once may still increase risk or cost if calls share mutable state, **rate limits** (service-imposed request caps), or irreversible effects. Two writes to the same file may conflict even when their arguments are separately known.

The dependency graph should include *conflict edges*, arrows that prevent simultaneous operations with incompatible effects, as well as data edges. For calls $i$ and $j$ with deterministic successful state-update functions $f_i$ and $f_j$, **commutativity** means $f_i(f_j(s))=f_j(f_i(s))$ for every relevant state $s$. If neither call depends on the other's result and their updates commute, concurrent execution may be safe with respect to final state. This condition is stronger than having disjoint arguments.

Two reads of one file commute. A read and edit of that file may not commute because the read can see different content. Even writes to different files can conflict if a build or deployment observes one change without the other. If commutativity cannot be established, **serialize** the calls (run one after the other) or isolate their state. Concurrency is a correctness decision before it is a latency optimization.

In the grocery case, two independent product searches can run concurrently. Cart creation depends on their product identifiers and must wait for both results. If two workers can write the same cart, impose an ordering or use a service operation with conflict detection. Otherwise, the final cart may omit one worker's items even when both calls return success. Parallel latency is bounded by the search dependency path, while correctness also depends on the cart's concurrency semantics.

Python is a programming language. The following extension takes `calls`, an iterable of `ToolCall` objects with unique `call_id` fields, and `dispatch`, an asynchronous function that executes one read-only call. `asyncio.gather` waits for all supplied operations and returns their results. It is separate from the sequential harness because that harness has no dependency scheduler or concurrent event-log merge rule.

```python
import asyncio

async def run_independent(calls, dispatch):
    async def one(call):
        return call.call_id, await dispatch(call)
    results = await asyncio.gather(*(one(c) for c in calls))
    return dict(results)  # IDs keep results attached to the right call.
```

The snippet omits cancellation and partial failure. In practice, bound concurrency, impose per-call timeouts, classify retryable errors, and avoid retrying a side effect without idempotency or reconciliation.

## 6. Evaluation across the tool stack [49:50](https://www.youtube.com/watch?v=jXChFB4JSyw&t=2990s)

Tool choice, argument correctness, call order, end-to-end success, latency, token use, and operational failures measure different properties. The [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) provides tool-use tasks, but an end-to-end system can also fail in its inference provider, authentication layer, dispatcher, or target service.

Define $A$ = correct tool, $B$ = correct arguments given tool, $C$ = successful execution given the request, and $D$ = task success given execution. Here $P(A)$ is the probability of $A$, $P(B\mid A)$ is the conditional probability of $B$ given $A$, and $\cap$ means that all listed events occur. By the probability chain rule,

$$P(A\cap B\cap C\cap D)=P(A)P(B\mid A)P(C\mid A,B)P(D\mid A,B,C).$$

Each factor measures a different conditional stage. For example, $P(C\mid A,B)$ asks how often execution succeeds after the correct tool and arguments have already been chosen. A low value at that stage points toward credentials, dispatch, or the downstream service rather than tool selection.

This factorization is exact and does **not** assume independence. It also gives a diagnostic plan. Log each stage and estimate its conditional success rate. If malformed calls are frequent, test schema/decoding. If valid calls fail, inspect credentials and runtime. If all calls succeed but tasks fail, inspect planning or the task verifier. A benchmark score measures performance on the benchmark's test distribution. A domain-specific end-to-end test is still needed to estimate performance in the intended environment.

### Tool surfaces in deployed agents

[Cursor's tool documentation](https://docs.cursor.com/en/agent/tools) separates search, edit, terminal execution, MCP access, and guardrail controls. These tools have different effects and therefore need different controls. An editor integration can also expose a reviewable diff and local [checkpoints](https://docs.cursor.com/en/agent/chat/checkpoints). Checkpoints help restore local work, but they are not permanent version control. [OpenClaw's tool documentation](https://docs.openclaw.ai/tools) similarly distinguishes tool availability from sandbox and policy. A tool appearing in a model prompt therefore should not be interpreted as an unrestricted execution right. For a new tool, test the model proposal, schema validation, policy admission, execution, and postcondition verification as **separate** stages. A single end-to-end pass can hide which stage is fragile.

[Claude Code's permission documentation](https://code.claude.com/docs/en/permissions) gives a concrete multi-layer gate: allow, ask, and deny rules determine tool admission; pre-tool hooks can block or request a prompt; shell sandboxing imposes operating-system boundaries. A hook is useful for a deterministic condition such as “reject writes outside this directory,” but its correctness depends on complete input inspection and on the runtime actually invoking it for the relevant event. Sandbox policy constrains shell subprocess effects. Other tools require their own controls. The [hooks guide](https://code.claude.com/docs/en/hooks-guide) also describes post-tool and stop events.

In a tool contract, pre-tool checks can enforce preconditions, sandbox and permission settings can bound effects, and tests or post-tool checks can verify postconditions. These layers address different failure modes. An allow rule cannot repair a faulty verifier.

### Prompt injection as an information-flow violation

**Prompt injection** is an attempt to make lower-trust content act as an instruction with higher authority. An **information-flow violation** occurs when information crosses a boundary that was meant to restrict how it can influence actions. Let $L$ denote untrusted observations, such as retrieved pages or repository files. The harness may legitimately let $L$ affect task data, for example a quoted fact in an answer. It must not let $L$ modify authorization $u_t$ or the enforcement function $G_\phi$.

An **authorization-safety property** makes this boundary testable: for every possible $L$, each executed action must remain within $\operatorname{cap}(u_t)$, the rights granted by the current authority state. This property allows a page to change which facts the agent reports. It does not allow the page to grant an upload capability. A prompt instruction telling the model to ignore attacks cannot enforce this property by itself.

For a coding agent, test this boundary with a repository file containing a false instruction to exfiltrate credentials. The file may be read as evidence about code. The model may nevertheless propose the network call requested by the malicious text. The decisive check is whether the execution policy rejects the call and whether the trace records the rejection. A behavioral benchmark that only scores the final answer can miss this attempted boundary crossing. One rejected malicious proposal is not proof that the boundary holds across every tool path. Enumerate the available capabilities and test each execution surface.

### Worked threat model: a retrieved page becomes a false command

Suppose an agent uses `search_web` to answer a chemistry question. A returned page contains a sentence that claims to be a new system instruction and asks the agent to run `execute_code` to upload local files. The page's text is an *observation supplied by a third party*. It has no authority to redefine the user's request or the tool policy. The attack succeeds only if the system promotes source content into instruction authority or lets the model make an unchecked call across that boundary.

Write $A\succ B$ to mean that instruction source $A$ has higher authority than source $B$. The trust ordering is user/developer policy $\succ$ tool-output data. Tool output may contain true and useful information about the world. Its truth does not give it authority over the harness. Enforce a narrow code sandbox without access to unrelated files or network destinations; separate browsing and execution permissions; and inspect intended side effects before high-impact calls. A **classifier**, a program that labels inputs by category, may help flag prompt injection, but the core protection is architectural: the retrieved page cannot mutate trusted policy, and a forbidden upload cannot pass the authorization predicate.

### A small test matrix for a new tool

For a `create_grocery_cart(items)` tool, test more than a single happy path. **Syntax tests** include missing `items`, wrong types, and extra fields. **Semantic tests** include ambiguous item names and unavailable products. **Authorization tests** distinguish preparing a cart from placing an order. **Runtime tests** include a timeout after the cart was actually created, upstream rate limiting, and a response attached to the wrong call ID. **Adversarial tests** include a product description that asks the agent to change payment details. The expected outcome is not always a successful call. In some cases, the correct behavior is to decline, ask for clarification, or report uncertainty. This matrix separates model quality from protocol and product quality.

### Exercises

1. Design a `transfer_funds` tool contract. State its schema, semantic checks, permission check, idempotency key, and postcondition. Explain which checks constrained decoding cannot enforce.
2. Given durations $d_A=2$, $d_B=3$, $d_C=4$ seconds, with $A\to C$ and $B\to C$, calculate sequential and optimal parallel latency. Add a shared write conflict and revise the schedule.
3. Suppose 90% of calls choose the right tool, 80% of those have correct arguments, 95% execute successfully, and 85% of these produce task success. Compute whole-task success and identify the most promising measured intervention.

### Solutions and discussion

1. A `transfer_funds` call should include source account, destination account, amount with currency, and a client transaction identifier. The schema can require well-formed fields and a positive amount. Semantic checks must compare the destination and amount with the user's reviewed instruction. The permission check must establish that this user may transfer from the source account and that the amount is within the granted limit. A server-recognized idempotency key prevents a timeout retry from becoming a second transfer. After execution, query the authoritative transaction record and verify recipient, amount, and final status. Constrained decoding can enforce a syntactic form; it cannot establish consent, account ownership, available funds, or that a remote transfer committed.
2. Sequential execution takes $2+3+4=9$ seconds. Without conflicts and with enough workers, run $A$ and $B$ together, then $C$ after both finish. The elapsed time is $\max(2,3)+4=7$ seconds, matching the critical path through $B$. If $A$ and $B$ make conflicting writes, order them before $C$; elapsed time returns to $9$ seconds under these durations. The conflict edge is a correctness constraint, not a data dependency inherent in their arguments.
3. Treat the four given percentages as conditional probabilities in order. Whole-task success is $0.90(0.80)(0.95)(0.85)=0.5814$, or $58.14\%$. Argument correctness is the lowest conditional rate, so it deserves diagnosis. It is not automatically the best investment: the achievable improvement and implementation cost matter. Raising that rate from $0.80$ to $0.90$, while holding the others fixed, would increase success by $0.90(0.10)(0.95)(0.85)=0.072675$, or about $7.27$ percentage points. Measure the same intervention's latency, cost, and unsafe-call rate before choosing it.

### Further reading

- [What Are Tools Anyway?](https://arxiv.org/abs/2403.15452) and [CodeAct](https://arxiv.org/abs/2402.01030): expressive tool interfaces and code actions.
- [XGrammar](https://arxiv.org/abs/2411.15100): the grammar-engine implementation path.
- [JSON Schema](https://json-schema.org/draft/2020-12/json-schema-validation.html), [OpenAPI](https://spec.openapis.org/oas/latest.html), and [MCP tools](https://modelcontextprotocol.io/specification/draft/server/tools): evolving protocol contracts.
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html): tool-use evaluation, to be read with its methodology.
