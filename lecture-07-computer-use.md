# Lecture 7 — Computer-use agents

*Independent study chapter · Based on a CMU 11-768: AI Agents lecture by JY Koh, 15 September 2026 · [Course and attribution](index.html#sources)*

## 0. Acting through the human interface [01:30](https://www.youtube.com/watch?v=jwGluLrrqjQ&t=90s)

A user asks an agent to buy a blue mug for less than twenty dollars. The agent observes a rendered screen, finds candidate products, opens a result, checks color and price, and adds an item to a cart. A visually plausible click can hit the wrong control. A correct cart can still contain an unauthorized item. The interface exposes the same pixels a human sees, but it does not supply the semantics and guarantees a programmatic application programming interface (API) would provide.

A **computer-use agent (CUA)** acts through a graphical user interface (GUI). Let hidden environment state at step $t$ be $s_t\in\mathcal S$. The agent receives observation $o_t=O(s_t)$, where $O$ may produce a screenshot, an accessibility tree, document structure, or a combination. It chooses action $a_t\sim\pi_\theta(\cdot\mid g,o_{\le t},a_{<t})$ for goal $g$. The environment then changes according to

$$s_{t+1}\sim P_{\rm env}(\cdot\mid s_t,a_t).$$

The observation is partial. A screenshot does not expose hidden application state, pending network requests, or whether a visual update has committed. The same pixels can correspond to different states, and small rendering changes can move a control.

Common action types include clicking a point, typing text, pressing a key, scrolling, dragging, waiting, and calling a non-GUI tool. Represent a click as $a=(\text{click},x,y)$, where $(x,y)$ is a coordinate in a declared reference frame. If the model predicts normalized coordinates $(u,v)\in[0,1]^2$ for an image of width $W$ and height $H$, the corresponding pixel coordinate is

$$x=u(W-1),\qquad y=v(H-1).$$

The subtraction by one maps $u=1$ to the last pixel index. Device scaling, browser zoom, window movement, and remote-display transforms can invalidate this conversion. A harness should record the screenshot dimensions and transformation used for every coordinate action.

The companion function implements only this declared reference-frame conversion:

```python
from agent_lab.evaluation import normalized_to_pixels

point = normalized_to_pixels(0.25, 0.75, width=1440, height=900)
assert point == (359.75, 674.25)
```

It deliberately returns a continuous coordinate. Rounding and conversion to device-independent pixels belong to the GUI adapter because those conventions depend on the execution environment.

The loop ends when a verifier accepts the resulting state, the agent requests completion without sufficient evidence, or a budget expires. Model self-declaration is not an outcome verifier.

### The mug task as a partially observed control problem

The user goal contains positive requirements and exclusions. Let terminal state $s_T$ contain the cart and checkout state. Let $b(s_T)$ mean that exactly one blue mug is in the cart. Let $p(s_T)\le20$ mean that its listed unit price is at most twenty dollars. Let $c(s_T)=0$ mean checkout has not occurred. A suitable terminal predicate is

$$\phi_g(s_T)=b(s_T)\land[p(s_T)\le20]\land[c(s_T)=0].$$

The square brackets group a Boolean comparison. The predicate rejects a red mug, two blue mugs, an expensive mug, or a completed purchase. A verifier that checks only whether the page displays the words “blue mug” tests an observation, not the underlying goal state.

The agent cannot evaluate $\phi_g$ directly from every screenshot. It maintains a belief $\beta_t(s)=P(s_t=s\mid h_t)$ over possible hidden states given history $h_t$. A click on “Add to cart” may produce no visible change while a request is pending. Repeating the click can add a second item. The correct next operation may be to wait, inspect cart state, or query a structured interface. The action depends on uncertainty about the transition, not only on the visual location of the button.

Three boundaries should therefore surround each effectful GUI action. **Admission** checks that the action is authorized and that its coordinates or target reference are valid for the current observation. **Execution** performs the input event in an isolated session. **Reconciliation** observes the resulting state and compares it with the expected effect. An admitted click is not a successful click until reconciliation supplies evidence.

The human interface is sometimes the only available interface, but it need not be the only source of verification. A cart database, browser accessibility tree, downloaded artifact, or application state file can provide stronger evidence than pixels. The verifier should be independent of the same visual interpretation used to choose the action whenever possible.

## 1. Environments became more realistic [06:00](https://www.youtube.com/watch?v=jwGluLrrqjQ&t=360s)

Early GUI environments restricted tasks and interfaces to make interaction reproducible. [MiniWoB++](https://arxiv.org/abs/1802.08802) extends the earlier MiniWoB benchmark with small browser tasks that add stochastic environments and varied natural-language goals. [WebShop](https://arxiv.org/abs/2207.01206) models language-guided shopping. Such environments isolate learning problems but simplify state, layout, and consequences.

[WebArena](https://arxiv.org/abs/2307.13854) provides reproducible, functional websites and executable goals across several domains. [VisualWebArena](https://arxiv.org/abs/2401.13649) adds visually grounded tasks. [OSWorld](https://arxiv.org/abs/2404.07972), a benchmark for operating-system tasks, uses real desktop applications, initial-state setup, and task-specific execution checks. These benchmark designs move evaluation closer to deployed computer use, while still resetting and controlling the environment.

Real-world use introduces drift. Websites change, sessions expire, pop-ups appear, and applications update. Let $D_{\rm eval}$ be the distribution of states in an evaluation and $D_{\rm deploy}$ the deployment distribution. A reported benchmark success rate estimates performance on $D_{\rm eval}$. It predicts deployment only to the extent that the relevant state-action structure transfers to $D_{\rm deploy}$.

The correct comparison unit is a complete system: model, observation representation, action schema, prompt, memory, retry policy, environment version, and verifier. Two systems using the same model can differ materially because their coordinate conventions or state representations differ.

### Benchmark realism is a vector

“Realistic” is not a total ordering. Environments differ in visual complexity, state persistence, application fidelity, network variability, task horizon, user history, and consequence. A benchmark can use visually realistic pages while simplifying account state. Another can use genuine applications while replacing irreversible actions with resettable mocks.

Let realism profile $r(E)$ for environment $E$ be a vector

$$r(E)=(r_{\rm visual},r_{\rm state},r_{\rm dynamics},r_{\rm consequence},r_{\rm history}).$$

Each component is defined by a measurement protocol rather than by appearance alone. The vector prevents a single adjective from hiding which dimensions transfer to deployment. [Mini World of Bits++ (MiniWoB++)](https://arxiv.org/abs/1802.08802) isolates many interface skills in small tasks. [WebShop](https://arxiv.org/abs/2207.01206) adds language-guided product search. [WebArena](https://arxiv.org/abs/2307.13854), [VisualWebArena](https://arxiv.org/abs/2401.13649), and [OSWorld](https://arxiv.org/abs/2404.07972) increase site or application fidelity and use executable end-state checks.

Benchmark progression also changes the source of failure. In a toy environment, a wrong click may dominate. In a persistent desktop, the same error can alter files that affect later steps. In an environment with personal history, the observation policy must decide which emails, calendar events, or documents are relevant without disclosing unrelated data. Capability and control become inseparable.

A reproducible report names the environment commit or image, initial-state generator, account fixtures, viewport, scaling, observation mode, action schema, reset procedure, task version, and verifier. “Model X on OSWorld” omits most of the experimental system.

## 2. Static grounding and end-to-end success answer different questions [12:00](https://www.youtube.com/watch?v=jwGluLrrqjQ&t=720s)

A **static evaluation** scores an action against a recorded state without executing it. A **grounding task** asks the model to identify the interface region corresponding to a textual instruction. If the reference rectangle is $B$ and the predicted rectangle is $\widehat B$, their intersection-over-union (IoU) is

$$\operatorname{IoU}(B,\widehat B)=
\frac{|B\cap\widehat B|}{|B\cup\widehat B|},$$

where $|\cdot|$ denotes pixel area. IoU equals one for identical nonempty boxes and zero for disjoint boxes. Point-based benchmarks instead accept a click when it lies inside a target region. [ScreenSpot-Pro](https://arxiv.org/abs/2504.07981) evaluates grounding in professional interfaces.

Static action accuracy is inexpensive and reproducible, but it does not measure recovery or task completion. Several different actions can be valid. A recorded reference action can become inappropriate after an earlier deviation.

An **end-to-end evaluation** executes the agent from an initial state and checks the resulting environment. Let $\phi_g(s_T)$ be a task-specific predicate that returns one when terminal state $s_T$ satisfies goal $g$ and zero otherwise. Binary success is

$$Y=\phi_g(s_T).$$

This measure permits different successful trajectories. Its validity depends on whether $\phi_g$ captures the full goal and exclusions. For “add one blue mug below twenty dollars,” the verifier should check product color, unit price, quantity, cart membership, and absence of checkout. Checking only that a cart is nonempty accepts the wrong state.

### Grounding accuracy compounds across a trajectory

Suppose a task requires $T$ indispensable grounding decisions and each is correct with conditional probability $q_t$. If errors are conditionally independent and no recovery is possible, trajectory grounding success is

$$P(\text{all grounded correctly})=\prod_{t=1}^{T}q_t.$$

With twenty decisions at $q_t=0.97$, the product is about $0.544$. The calculation is not a calibrated law because errors are correlated and recovery can occur. It does show why a small per-step improvement can matter over long horizons and why end-to-end success cannot be inferred from static grounding accuracy.

IoU can also misrepresent click quality. A predicted bounding box may overlap a large button strongly while placing its chosen point on a neighboring control. Conversely, a small point target can be clicked correctly even when the predicted box has low IoU. Report the metric that matches the executed action. For point clicks, useful quantities include inside-target accuracy and distance to the target boundary.

Let intended point be $(x,y)$ and executed point be $(x+\epsilon_x,y+\epsilon_y)$, where $\epsilon_x$ and $\epsilon_y$ represent scaling and motor error. If the nearest target boundary is only two pixels away, a visually correct point is fragile. Selecting a point near the center of a verified target region increases the error margin. Re-observation remains necessary when the layout can move between localization and execution.

[ScreenSpot-Pro](https://arxiv.org/abs/2504.07981) uses professional interfaces to make static grounding harder. Such a benchmark measures an important component. It does not include state transitions, recovery, or terminal verification. Component and system evaluations should be reported together.

### Outcome, process, and evidence

Programmatic checks inspect structured application state. A human or vision-language judge can assess open-ended evidence. A **trajectory-aware** verifier also examines how the result was produced. These methods answer different questions.

| Verifier | Strength | Failure mode |
|---|---|---|
| Programmatic state check | precise for encoded properties | omits unencoded requirements |
| Screenshot judge | flexible for visual outcomes | can be fooled by appearance |
| Human review | handles ambiguity | costly and variable |
| Trajectory audit | detects forbidden process | long traces and hidden effects |

For high-consequence tasks, use independent state evidence rather than only the agent's final screenshot. A screenshot can show “Saved” even when a later synchronization fails.

A verifier should distinguish an outcome from evidence about that outcome. Let hidden proposition $Y$ mean “the file is durably saved.” A screenshot message $E_1$ and a successful file read $E_2$ are evidence variables. The posterior $P(Y\mid E_1,E_2)$ depends on the reliability and dependence of both channels. Two screenshots generated by the same application event are not independent confirmations.

For the mug task, a robust verifier can inspect cart structure and compare product identifier, variant, quantity, price, and checkout status. It should also check that unrelated cart items were not added. For a desktop document, it can reopen the saved file from disk and inspect its content. For a sent message, it can query the sent-mail folder and message identifier. Each procedure verifies a different state transition.

Trajectory audit is required when process restrictions matter. A terminal artifact can be correct even though the agent exposed a secret, visited a prohibited site, or used an unauthorized account on the way. Let $F(\tau)$ count forbidden effects in trajectory $\tau$. Admissible success is $\phi_g(s_T)=1\land F(\tau)=0$. The terminal state and the process trace are jointly necessary.

## 3. Long-horizon tasks need partial but non-gameable feedback [33:30](https://www.youtube.com/watch?v=jwGluLrrqjQ&t=2010s)

Hours-long tasks often produce meaningful intermediate artifacts. A binary terminal reward discards that information. A **rubric** contains criteria $j=1,\ldots,m$, weights $w_j\ge0$, and checks $c_j\in\{0,1\}$. Its normalized score is

$$R_{\rm rubric}=\frac{\sum_{j=1}^m w_jc_j}{\sum_{j=1}^m w_j},$$

provided the denominator is positive. The score lies in $[0,1]$. Each criterion must describe an outcome that can be inspected independently.

Suppose a spreadsheet task awards weights 4 for correct values, 2 for formulas rather than pasted constants, 1 for formatting, and 3 for preserving unrelated sheets. An agent satisfying the first three but damaging another sheet receives $7/10$. Whether that score should count as success is a separate decision. Preservation may be a hard constraint rather than compensable reward.

The companion function checks that weights and criteria have identical names:

```python
from agent_lab.evaluation import rubric_score

score = rubric_score(
    {"values": 4, "formulas": 2, "format": 1, "preserve": 3},
    {"values": True, "formulas": True, "format": True,
     "preserve": False},
)
assert score == 0.7
```

Rubric design can create reward hacking. If a criterion checks only that a file exists, the agent may create an empty or corrupt file. If intermediate rewards are given for opening the right application, the agent can repeat that step. Criteria should be outcome-based, nonduplicative, and tested against adversarial near-misses.

[Odysseys](https://odysseys-website.pages.dev/) studies long-horizon live-web tasks with rubric-based judging. Later desktop benchmarks add partial checks for complex artifacts. Results from a dated live-web benchmark can drift as sites change, so the environment and evaluation date are part of the claim.

### Partial credit must not buy permission

Rubric weights define substitution among scored criteria. If an agent can compensate for deleting user data by formatting a spreadsheet well, the rubric encodes the wrong decision rule. Separate hard constraints $H_k(\tau,s_T)$ from soft criteria $c_j(\tau,s_T)$. Define admissible reward

$$R(\tau,s_T)=\begin{cases}
\displaystyle\frac{\sum_jw_jc_j}{\sum_jw_j},&\bigwedge_k H_k=1,\\
\bot,&\text{otherwise},
\end{cases}$$

where $\bot$ denotes an inadmissible outcome rather than a low numerical score. Training code may map inadmissibility to a scalar penalty, but the evaluator should retain the constraint violation as a separate event.

Rubric criteria should be mutually interpretable. If “correct formulas” implies “correct values,” awarding both can double-count one achievement unless that dependence is intended. Let $C$ be the vector of criterion results. Analyze correlations and adversarial examples before assigning weights. A criterion that is almost always satisfied or that duplicates another adds little diagnostic value.

For a presentation task, partial criteria might inspect slide count, required topics, readable text, citation presence, and file validity. Preservation of existing files and absence of external publication can remain hard constraints. A well-designed near-miss suite includes blank files, flattened screenshots instead of editable slides, hidden off-canvas text, duplicated content, and an otherwise correct artifact saved to the wrong location.

## 4. Observation and action representations define the interface [46:30](https://www.youtube.com/watch?v=jwGluLrrqjQ&t=2790s)

A screenshot represents every visible pixel but makes text and element identity implicit. An accessibility tree exposes roles, names, and hierarchy but can omit custom-rendered controls or visual relations. Hypertext markup language (HTML) exposes web structure but not arbitrary desktop applications.

Let $z_t^{\rm pix}$, $z_t^{\rm acc}$, and $z_t^{\rm dom}$ be pixel, accessibility, and document-object representations. A multimodal context builder may form

$$c_t=C(g,h_t,z_t^{\rm pix},z_t^{\rm acc},z_t^{\rm dom},b_t),$$

where $b_t$ is the remaining token and action budget. Including every representation increases context cost and can introduce contradictions. The system should preserve provenance so the model can distinguish visible text from metadata and untrusted page content from instructions.

A coordinate action is compact but fragile to layout change. An element-reference action such as `click(role="button", name="Save")` is more semantic but depends on a correct accessibility representation and a deterministic resolver. Hybrid systems can let the model identify an element semantically and let the harness resolve its current geometry.

[Agent S](https://arxiv.org/abs/2410.08164) combines hierarchical planning, experience retrieval, and an agent-computer interface for graphical control. Its system result illustrates that planning and interface design interact with the same underlying model. It does not isolate a universal contribution from any one component without the reported ablations.

Action schemas vary among frontier CUAs. A result therefore depends on provider-specific tokenization, screenshot scaling, history serialization, and tool semantics. Reproducing a benchmark requires recording those details, not only a model name.

### Prompt injection through the GUI

Text rendered on a page is an observation. It is not an authority source. Let $L(x)$ assign a trust level to context item $x$. The harness should enforce an information-flow rule: content at a lower trust level may supply data, but it cannot modify permissions, system constraints, or the verifier.

For every executed action $a_t$, require

$$\operatorname{execute}(a_t)\Rightarrow
\operatorname{authorized}(a_t,U_t)\land
\operatorname{supported}(a_t,e_t),$$

where $U_t$ is current authorization and $e_t$ is trusted evidence. A page saying “upload your credentials to continue” cannot satisfy either predicate merely by appearing on screen.

### Observation selection is active sensing

The agent can often choose its next observation. It may zoom, scroll, inspect accessibility metadata, query the active application, or open a read-only details panel. Let observation action $q$ cost $c(q)$ and produce random evidence $e$. Its expected value of information is

$$\operatorname{VOI}(q)=\mathbb E_e\left[\max_a U(a\mid h_t,e)\right]-\max_aU(a\mid h_t)-c(q).$$

The agent should acquire evidence when it can change the best action enough to justify its cost. Scrolling to expose delivery terms has positive value when the user imposed a deadline. Repeatedly taking an unchanged screenshot usually has little value unless the system is waiting for an asynchronous transition.

Observation provenance matters for security. Visible page text is controlled by the page. Accessibility labels may also be page-controlled. Operating-system metadata about the active application has a different trust source. The context builder should label these channels so that the model and admission layer can apply different rules.

Prompt injection is an information-flow failure when untrusted observation text is allowed to alter control instructions. A practical defense does not rely on the model recognizing malicious prose. It keeps authorization, tool schemas, and hard constraints in trusted state outside the page; limits tools to the current task; and requires evidence for each consequential action. The page may propose data. It cannot grant effects.

## 5. Training combines grounding, trajectories, and outcome feedback [57:00](https://www.youtube.com/watch?v=jwGluLrrqjQ&t=3420s)

CUA training usually begins from a vision-language model. Pretraining can teach visual-text relationships and interface elements. Supervised fine-tuning can imitate action trajectories. Reinforcement learning can optimize behavior against environment rewards. These stages expose different supervision.

A grounding example maps an observation and element description to a region or point. An action-prediction example maps history to the next action. A trajectory example records a sequence. Let dataset item $i$ be

$$d_i=(g_i,s_{i,0},o_{i,0},a_{i,0},\ldots,o_{i,T_i},y_i),$$

where $y_i$ contains terminal checks or reward. Initial state $s_{i,0}$ is essential for reproducibility. The same instruction can require a different action in a different state.

Human demonstrations can be precise but expensive. Synthetic trajectories can scale but inherit teacher errors and benchmark artifacts. A privileged teacher may use an accessibility tree while the student receives pixels. This can generate successful targets, but it also creates an information mismatch. The student must infer actions from less informative observations.

### Training data defines the observation policy

Behavior cloning minimizes action loss on recorded histories. Let expert data distribution be $d_{\pi^*}(h)$ and learned deployment distribution be $d_{\pi_\theta}(h)$. Even low error under expert histories can compound because the learned policy visits states that the expert did not. This is the same covariate-shift problem developed in Lecture 8.

Computer-use data adds another mismatch: the collector may observe more than the deployed policy. If a teacher selects elements from the document object model while the student receives only pixels, the target action can be correct but poorly identifiable from the student's input. Such **privileged information** can be useful during training, but the experiment must test whether it yields a policy that succeeds under the deployment observation channel.

[Mind2Web](https://arxiv.org/abs/2306.06070) provides human demonstrations across websites and domains for generalist web agents. Offline element-ranking and action-prediction metrics make iteration cheaper. They do not reproduce consequences of earlier mistakes. [OpenCUA](https://arxiv.org/abs/2508.09123) develops open models and data for computer use. These resources should be compared by observation channels, action schemas, trajectory provenance, and deployment interface, not only by example count.

[WebLINX](https://arxiv.org/abs/2402.05930) adds multi-turn dialogue to website navigation and reports 100,000 interactions across 2,300 expert demonstrations. Its HTML-pruning component makes context selection part of the agent. [WebVoyager](https://arxiv.org/abs/2401.13919) evaluates an end-to-end multimodal agent on live websites and uses an automatic multimodal evaluator checked against human judgment. Together they expose two additional variables: whether the user can revise the goal during execution and whether open-ended terminal states require a learned judge.

Training should mix atomic grounding, short recovery fragments, and complete trajectories according to the desired behavior. Atomic data supplies dense supervision. Recovery fragments teach what to do after an error or unexpected dialog. Full trajectories teach long-range state management. Their token and sample weights determine which behavior dominates.

## 6. Resettable environments make reinforcement learning possible [64:00](https://www.youtube.com/watch?v=jwGluLrrqjQ&t=3840s)

Online training requires repeated interaction, reward computation, and recovery from failures. Real websites are expensive, nondeterministic, and capable of charging money or affecting other users. A **resettable environment** has a known initial-state generator and can restore state after a trajectory.

Let $\mathcal E=(\mu_0,P,O,r,\operatorname{reset})$, where $\mu_0$ is an initial-state distribution, $P$ the transition law, $O$ the observation function, $r$ the reward, and `reset` a procedure that restores a valid starting state. A training task is useful only when its setup, transition behavior, and reward agree.

[CUA-Gym](https://arxiv.org/abs/2605.25624) generates mock applications, tasks, initial states, and reward logic. [Gym-Anything](https://arxiv.org/abs/2604.06126) targets broader software environments. Synthetic scale does not remove validation: the generator can create impossible tasks, leaking verifiers, or shortcuts absent from real applications.

Separate environment-generation tests from agent training. For each generated task, execute a known successful trace, execute adversarial failing traces, verify reset determinism within stated limits, and confirm that forbidden external effects are unreachable.

A reset operation is part of the experimental semantics. Let $s_0\sim\mu_0$ be an initial state and let $\mathcal R(s_T,\zeta)$ be reset with random seed $\zeta$. A useful reset guarantee is distributional:

$$\mathcal R(s_T,\zeta)\sim\mu_0,$$

independently of the preceding terminal state $s_T$ for all allowed trajectories. Exact byte-for-byte reset is stronger and often unnecessary. The required equivalence classes must include every state variable that can affect the task or reveal a previous rollout.

Reset leakage can make training and evaluation misleading. A cart item, cached login, modified file, or accumulated notification may simplify or obstruct later tasks. Random seeds should control initial fixtures while preserving a held-out set of states. Environment tests should run before policy experiments and after every environment update.

The reward implementation also needs unit tests. A known successful trace must receive the intended score. Near-misses must fail the exact criteria they violate. The reward must not be readable by the agent through an unintended file or page field. A generated environment without these checks scales annotation errors as efficiently as it scales useful experience.

## 7. Deployment adds speed, personalization, and control [72:00](https://www.youtube.com/watch?v=jwGluLrrqjQ&t=4320s)

CUA latency accumulates across screenshots, model calls, rendering, and actions. If step $t$ has observation latency $d_t^{\rm obs}$, inference latency $d_t^{\rm model}$, and action latency $d_t^{\rm act}$, total time is

$$D=\sum_{t=0}^{T-1}
(d_t^{\rm obs}+d_t^{\rm model}+d_t^{\rm act}).$$

Parallel workers reduce time only for independent state. Two agents controlling one cursor can interfere. Chapter 5's work–span bound applies after the plan includes shared-interface constraints.

Personalization supplies user preferences and context. It also increases privacy risk and the chance that stale memory affects action. Store provenance, scope, and expiration. A preference does not imply permission to transact.

Proactive background use needs visible state, bounded authority, an interruption mechanism, and durable evidence. The user should be able to inspect what is running, what action is proposed, and how to stop it. A polished GUI animation is not a substitute for an auditable event record.

Latency is a control variable because stale observations make actions less reliable. Let $\delta_t$ be the time between capturing observation $o_t$ and executing action $a_t$. Let $m_t$ be the probability that the relevant interface geometry changes during that interval. In dynamic applications, $m_t$ generally increases with $\delta_t$. The harness can impose a freshness limit and re-observe before coordinate actions when that limit is exceeded.

Parallelism is safe only across independent interfaces or isolated sessions. If two workers share a cursor, keyboard focus, clipboard, or browser profile, their actions interact even when their goals are logically separate. These shared resources create exclusion constraints in the plan. A worker may research another product in a separate browser profile, but it should not type while another worker controls the active checkout window.

Personalization should use scoped retrieval. A preference such as “avoid fragile items” may be relevant to shopping. An unrelated medical document is not. Every retrieved item should retain source, scope, and expiration. Retrieval provides context for a decision. It never creates authority to reveal or act on the information.

### The observation channel creates an information bottleneck

Let the hidden interface state be $s_t$. The agent receives observation $o_t=\phi(s_t)$ through an observation function $\phi$. A screenshot, an accessibility tree, and a document-object-model snapshot use different functions $\phi$ and therefore preserve different information. A screenshot preserves pixels and spatial layout but does not directly expose semantic roles. An accessibility tree exposes roles, names, and hierarchy but may omit canvas content or custom controls. A structured web representation exposes machine-readable elements but can diverge from what the user sees.

An observation is **sufficient** for a decision when it preserves all state distinctions that can change the best action. Formally, for two hidden states $s$ and $s'$ with the same observation, $\phi(s)=\phi(s')$, sufficiency requires the same optimal action set:

$$\operatorname*{arg\,max}_{a}Q(s,a)
=\operatorname*{arg\,max}_{a}Q(s',a),$$

where $Q(s,a)$ is the expected future task value after taking action $a$ in state $s$. This condition usually fails for a single screenshot. Two identical-looking dialogs may belong to different applications, accounts, or security contexts. The harness can restore information by adding trusted metadata, querying the active application, or requiring confirmation before a consequential action.

### Coordinate actions amplify small grounding errors

Suppose the intended target occupies rectangle $B$ and the agent predicts click point $(\hat x,\hat y)$. A strict grounding score is

$$g=\mathbf 1[(\hat x,\hat y)\in B],$$

where $\mathbf 1[\cdot]$ equals $1$ when its condition is true and $0$ otherwise. This score hides the margin to failure. Define $d((\hat x,\hat y),\partial B)$ as the distance from the predicted point to the target boundary $\partial B$. A positive, large margin is more robust to display scaling, layout motion, and motor noise than a click barely inside the box.

Before execution, the agent should re-observe when the interface may have moved. After execution, it should verify the resulting state rather than infer success from the click itself. This creates a three-step micro-policy: locate, act, verify. The same pattern applies to typing into a field, dragging an object, or selecting a menu item.

### Control should depend on consequence, uncertainty, and reversibility

Let $p_t$ be the estimated probability that the proposed action is correct, let $L_t$ be the loss if it is wrong, and let $r_t\in[0,1]$ measure reversibility, with $1$ meaning fully reversible. A simple escalation score is

$$E_t=(1-p_t)L_t(1-r_t).$$

The score is a decision aid, not a safety proof. A policy may execute automatically only below threshold $\eta_1$, request a fresh observation between $\eta_1$ and $\eta_2$, and require human approval above $\eta_2$. Some actions, such as disclosing a secret or confirming a purchase, should remain hard-gated regardless of the score because their constraints are noncompensable.

Benchmarks such as [WebArena](https://arxiv.org/abs/2307.13854) and [OSWorld](https://arxiv.org/abs/2404.07972) provide reproducible environments and end-state evaluators for web and desktop tasks. Their controlled accounts and reset mechanisms are part of the measurement apparatus. Deployment adds live credentials, personalized state, changing layouts, prompt injection, and actions that cannot simply be reset. The benchmark score therefore estimates capability under its environment distribution. It does not establish safe autonomy in a user's actual computer.

### Scope and limits of the partially observed control model

The partially observed control model explains why screenshot interpretation, action grounding, and verification are separate problems. It omits human motor conventions, application-specific recovery, adversarial interfaces, and social consequences. Benchmarks further simplify account state, policy restrictions, and irreversible actions.

Evaluation should report at least task success, partial progress, forbidden effects, latency, model and environment cost, human intervention, and verifier uncertainty. A benchmark score can establish performance only under its stated environment and verifier.

### Exercises

1. A model predicts normalized click $(0.25,0.75)$ on a 1440-by-900 screenshot. Compute the continuous pixel coordinate under the chapter's convention.
2. Design a verifier for “create an empty repository named `agent-notes`.” Give one insufficient check and one adversarial near-miss.
3. A rubric has weights $(5,3,2)$ and checks $(1,0,1)$. Compute its score. Which criterion might need to be a hard constraint instead?
4. Compare screenshot-only, accessibility-only, and hybrid representations for a custom canvas application.

### Solutions and discussion

1. The coordinate is $(0.25\times1439,0.75\times899)=(359.75,674.25)$. The harness must specify how it rounds to integer device coordinates.
2. Check repository existence, exact name, owner, visibility, emptiness, and absence of unrelated mutations. Checking only that a page contains the name is insufficient. A text file or issue titled `agent-notes` is an adversarial near-miss.
3. The score is $(5+0+2)/10=0.7$. If the failed criterion represents “no unauthorized transaction” or preservation of existing data, it should be a hard constraint rather than compensable reward.
4. Pixels capture the canvas but hide element semantics. Accessibility metadata may expose no useful canvas children. A hybrid can use pixels for grounding and semantic metadata for surrounding controls, with provenance and fallback behavior.

The first calculation produces a continuous coordinate. An implementation must declare whether it rounds, floors, or uses device-independent pixels. It must also apply browser zoom and operating-system scale exactly once. A one-pixel convention is harmless on a large button and material on a narrow target.

The repository verifier in the second exercise should query the hosting service's structured state when possible. It should confirm the owner and visibility as well as the name. Emptiness means no commits or files according to a declared service semantics. A screenshot of a repository page cannot by itself exclude a hidden initial commit or the creation of an unrelated repository.

The third exercise has arithmetic score $0.7$. The substantive answer depends on the meaning of the failed criterion. Preservation, authorization, and privacy conditions should generally gate admissibility. Formatting quality may be compensable. The rubric must state this classification before evaluation.

For the canvas comparison, the hybrid policy needs a fallback when semantic and pixel channels disagree. It can treat pixels as evidence of what is rendered, use accessibility metadata for named surrounding controls, and request human input when the consequential target remains ambiguous. Provenance lets the verifier diagnose which channel caused an error.

### Further reading

- [WebArena](https://arxiv.org/abs/2307.13854): reproducible web environments with functional evaluation.
- [VisualWebArena](https://arxiv.org/abs/2401.13649): visually grounded web tasks.
- [OSWorld](https://arxiv.org/abs/2404.07972): real desktop applications and execution-based checks.
- [Mind2Web](https://arxiv.org/abs/2306.06070): generalist web-agent data and offline evaluation.
- [OpenCUA](https://arxiv.org/abs/2508.09123): open foundations for computer-use agents.
- [CUA-Gym](https://arxiv.org/abs/2605.25624): generated, resettable environments for training.
