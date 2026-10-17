# Lecture 13 — Agent safety: threat models, sandboxes, credentials, and monitoring

*Independent study chapter · Based on a CMU 11-768: AI Agents lecture by Graham Neubig, 6 October 2026 · [Course and attribution](index.html#sources)*

## 0. An agent that reads untrusted text acts with delegated authority [00:00](https://www.youtube.com/watch?v=Mbonnq-PUGM&t=0s)

The maintainers of an open-source command-line tool deploy an agent to triage new issues. The agent runs in a continuous-integration (CI) job, reads each issue, and applies labels. A separate nightly job builds the tool and publishes it to a package registry with a long-lived publish token. The consequential error is a release that the maintainers did not authorize. Every user who installs it runs code that someone else chose.

Call this deployment the **triage pipeline**, the chapter's running case. It follows the [maintainers' postmortem](https://cline.bot/blog/post-mortem-unauthorized-cline-cli-npm) for the Cline command-line package. The triage job gave the agent a shell tool and accepted issues from any user. An issue title could therefore carry text phrased as an instruction. Chapter 2 calls such text a **prompt injection**: lower-trust content that tries to act as an instruction with higher authority. The triage job also shared a build cache with the release job. Code run from the triage job could plant cache entries that the release job later restored. The release job held the publication secrets.

The flaw was reported on 1 January 2026 and disclosed on 9 February. The **rotation** that followed, the replacement of exposed credentials, deleted the wrong npm token. On 17 February a third party used the surviving token to publish a version whose only change installed another open-source package. The postmortem concludes that shell access for a model processing untrusted input in CI "is functionally equivalent to giving every GitHub user shell access."

Three incidents and one patched flaw locate failures in other parts of chapter 1's harness. Each row rests on one report.

| Incident | Source | Component that failed (this chapter's reading) |
|---|---|---|
| An agent deleted mail from an inbox after its owner told it to confirm first, and could not be stopped from a phone | [a user's public post](https://x.com/summeryue0/status/2025774069124399363) | The approval rule lived in the prompt, not in chapter 1's execution gate |
| An agent on a staging task deleted a production database and its backups | [The New Stack](https://thenewstack.io/ai-agents-credential-crisis) | An account-wide token was reachable from the task |
| A command-injection flaw let an outside attacker make Microsoft 365 Copilot disclose information; Microsoft patched it | [Microsoft advisory](https://msrc.microsoft.com/update-guide/vulnerability/CVE-2025-32711) | Untrusted text reached a context with outbound effects |
| An operator ran agent harnesses against online retailers to take card data | [Gambit Security](https://gambit.security/blog-posts/autonomous-ai-agents-online-retailers-25-a-company), 22 September 2026 | The user was the attacker |

The Gambit report counts at least 600,000 card records taken from two companies.

### Untrusted inputs are repeated attempts

In chapter 1's notation, the context $c_t$ of the triage run holds the issue text, and the gate $G_\phi$ turns a proposal $z_t$, a label or a shell command, into the action $a_t$. The hidden state $s_t$ includes the cache contents and the reachable tokens. The trusted evidence $e_t$ records which issues arrived, not which are hostile.

Each untrusted input is a fresh attempt by its author. Let $N_{\rm in}$ inputs each succeed independently with probability $p_{\rm in}$. Then

$$P(\text{at least one success})=1-(1-p_{\rm in})^{N_{\rm in}}.$$

With $p_{\rm in}=0.001$ and $N_{\rm in}=5{,}000$ emails, the probability is about $0.993$. A rate ten times lower, $10^{-4}$, still gives about $0.39$. These rates are illustrative. For any fixed $p_{\rm in}>0$, the probability tends to one as $N_{\rm in}$ grows, and the attacker chooses $N_{\rm in}$. Lowering $p_{\rm in}$ therefore cannot suffice. A second defense must bound what one success can do. Adaptive attackers learn from failures, which can raise their odds above this estimate; correlated attempts can also lower them.

## 1. A threat model names whose intent an action serves [07:16](https://www.youtube.com/watch?v=Mbonnq-PUGM&t=436s)

A **threat model** states what causes a harm. It names the **principal**, the party whose instruction the agent should serve. It names the **adversary**, the party that causes the harm, and what the adversary controls. It also names the trust boundary crossed, the asset at risk, and the adversary's success condition.

| Threat model | Adversary controls | Boundary crossed | Example |
|---|---|---|---|
| **Malicious user** | the request itself | the deployer's or model provider's rules | card theft through agent harnesses |
| **Prompt injection** | content the agent reads | data acquires instruction authority | an issue title steers the triage agent |
| **Agent error** | nothing; no adversary exists | a wrong action inside or outside the grant | a balance check becomes a payment |
| **Improper means** | nothing; the method is wrong | forbidden process, whatever the outcome | an answer read from a file |

Chapter 1 admits an action when $\operatorname{allow}(a,e,U)=\operatorname{inScope}(a,U)\land\operatorname{preconditions}(a,e)\land\neg\operatorname{forbidden}(a,U)$. Here $U=(U^+,U^-)$ holds the grants and prohibitions. A malicious user corrupts $U^+$. Injection and agent error corrupt the proposal. Improper means produce a forbidden event, which $\neg\operatorname{forbidden}$ catches only if $U^-$ names the method.

A malicious user writes $U^+$ to cover the harmful act, so the user's policy cannot stop it. Chapter 1's invariant, that every executed action is authorized by $u_t$, still blocks actions outside $\operatorname{cap}(u_t)$, which no request widens. Let $U^-$ also hold the deployer's prohibitions, which no user can remove. An attacker who runs the harness, as in the Gambit case, faces neither check; only the model's refusal training remains. A **jailbreak** is a prompt designed to make a model bypass its safety rules. [AgentHarm](https://arxiv.org/abs/2410.09024) found leading models "surprisingly compliant with malicious agent requests without jailbreaking."

Prompt injection leaves $U$ unchanged and alters $z_t$. The user trusts the agent and has delegated authority to it. The attacker writes into an email, a web page, a file, or a tool result that the agent reads. [Greshake et al.](https://arxiv.org/abs/2302.12173) argued that applications built around a language model (LM) "blur the line between data and instructions." Chapter 2 states an **authorization-safety property**. For every possible content of the untrusted observations, each executed action must stay within $\operatorname{cap}(u_t)$, the rights of the current authority state. A gate can enforce this property mechanically. It cannot enforce the stronger property that untrusted content never changes which in-scope action is taken. A malicious label on an issue is in scope for the triage agent, so it passes $\operatorname{allow}$. [CaMeL](https://arxiv.org/abs/2503.18813) secures part of the stronger property. It extracts the control flow from the trusted query, so untrusted data "can never impact the program flow." Untrusted data can still supply the arguments of planned calls, and CaMeL checks those with security policies at each tool call. In [AgentDojo](https://arxiv.org/abs/2406.13352), a benchmark of 97 tasks whose tool outputs carry injected instructions, CaMeL solves 77% of tasks with what its authors call provable security, against 84% undefended.

Agent error needs no adversary. In [ToolEmu](https://arxiv.org/abs/2309.15817) Figure 2(b), a user asks whether a checking account can cover a 580.9-dollar bill. The agent pays the bill. If the grant for a question holds read effects only, the payment fails $\operatorname{inScope}$ and becomes a rejected proposal. ToolEmu emulates tools with an LM. Its evaluator found failures in 23.9% of cases for the safest agent tested.

Improper means reach the goal by a forbidden method. [Docent](https://transluce.org/docent/blog/introducing-docent) studied a port of Cybench, a capture-the-flag (CTF) benchmark in which the agent must recover a secret string, the flag. The agent read the flag from its environment's build file instead of exploiting the intended flaw, and failed once the flag was removed from that file. Chapter 7 defines **admissible success** as $\phi_g(s_T)=1\land F(\omega)=0$, written here with chapter 1's trajectory $\omega$; chapter 7 writes $\tau$. Its goal predicate $\phi_g$ is unrelated to the harness choices $\phi$. The terminal state satisfies goal $g$, and the trajectory contains no forbidden event. Improper means always fail the second condition. When they also reach the goal, a score that checks only the terminal state records a success. In training this is **reward hacking**. [Skalse et al.](https://arxiv.org/abs/2209.13085) define it as optimizing an imperfect proxy reward until true performance falls.

One incident can fit two models. In the database case, using a token outside the task's grant is improper means, although the goal was never reached. Deleting production instead of staging data is agent error. A defense against one does not cover the other.

The source lecture gives loss-of-control scenarios lower priority for now, because no plausible sequence of steps toward them has been laid out; it calls this a hypothesis it might revise. It also treats safety as a capability. An agent with no tools meets chapter 1's chance constraint $P(F(\omega)=1)\le\delta$ whenever every forbidden event needs a tool effect, and earns $R_g(\omega)=0$ on any goal that needs an effect. A control that keeps more of the grant at the same risk raises the attainable objective.

### Coverage is finite, so risks are ranked by what cannot be undone

No threat-model list is complete, and no agent capability is fully reliable. Effort should therefore follow consequence. Chapter 7's escalation score $E_t=(1-p_t)L_t(1-r_t)$ weights the loss $L_t$ of a wrong action by its irreversibility $1-r_t$. Here $p_t$ is the estimated probability that the action is correct, and $r_t\in[0,1]$ is its reversibility. Exfiltrating a valuable codebase, copying it to an outside party, cannot be undone. Deleting a database with backups can.

The database case shows that reversibility belongs to the recovery path, not to the action alone. The backups lay within the same token's reach and were deleted too. Call an action **recoverable under authority $u$** when some recovery path uses state that no action within $\operatorname{cap}(u)$ can change.

## 2. Red teaming estimates attack success, and training lowers it [16:53](https://www.youtube.com/watch?v=Mbonnq-PUGM&t=1013s)

**Red teaming** comes from penetration testing in computer security. Testers try to make a system break its rules in a controlled setting, and the owners fix each hole before a malicious party finds it. What its authors call the [largest public red-teaming competition](https://arxiv.org/abs/2507.20526) to date, run on the Gray Swan Arena platform, targeted 22 frontier agents in 44 deployment scenarios. Participants submitted 1.8 million prompt-injection attacks, and over 60,000 elicited policy violations.

Score each attempt against a fixed target with a fixed judge. The **attack success rate (ASR)** is the fraction of attempts that the judge marks as violations. The competition's pooled ASR is about $60{,}000/1{,}800{,}000\approx3.3\%$ per attempt. A per-attempt rate understates the risk over many attempts. Section 0's formula with $N_{\rm in}=k$ gives the chance that $k$ independent attempts succeed at least once, chapter 6's pass@$k$ from the attacker's side. At $p_{\rm in}=0.033$ and $k=100$ it is about $0.965$, as an illustration. On the benchmark built from the competition's attacks, the paper reports violations "for most behaviors within 10-100 queries" for nearly all agents.

**Automatic red teaming** replaces the human tester with a search. [Prompt Automatic Iterative Refinement (PAIR)](https://arxiv.org/abs/2310.08419) loops over three models. An attacker LM drafts a prompt, the target replies, and a judge scores the reply against the policy. The attacker revises from both until a success or the end of its query budget. PAIR "often requires fewer than twenty queries." An ASR therefore depends on the attacker's budget, the judge's reliability, and the overlap between training and test attacks. A judge that misses violations lowers the measured ASR without changing the target. Chapter 10's agreement checks apply to it.

### Safety training fits refusals and answers, then attacks the result

**Supervised safety training** fits a model to appropriate replies for harmful and legitimate requests. Section 3.1 of the [Nemotron 3 Super technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Super-Technical-Report.pdf) gives an open recipe. Its prompts cover jailbreaks, **indirect injection** (prompt injection carried in retrieved content rather than the user's request), and legitimate requests that tempt an **unnecessary refusal**, the declining of a legitimate request. Classifiers, one of which detects "embedded adversarial instructions," choose a response mode, and a moderation classifier removes unsafe responses. This is chapter 8's supervised fine-tuning (SFT) on a filtered dataset.

In the report's reinforcement learning (RL) stage, Section 3.2.1, one environment targets fewer unnecessary refusals, and another runs PAIR-style attacks on an early SFT-only checkpoint to collect harder jailbreak prompts. Section 3.2.3 adds a reward model, trained on human preferences, that guides behavior on safety-related topics.

Unnecessary refusal is the second error rate of safety training; [XSTest](https://arxiv.org/abs/2308.01263) measures it with 250 safe prompts and 200 unsafe contrasts.

Training lowers $p_{\rm in}$, and Section 0 showed why that is not enough. [Adaptive attacks](https://arxiv.org/abs/2503.00061) bypassed all eight indirect-injection defenses that one study tested, each with an ASR above 50%. Refusal training leaves agent error and improper means untouched, since neither needs a harmful request.

## 3. A sandbox bounds effects only for what runs inside it [24:26](https://www.youtube.com/watch?v=Mbonnq-PUGM&t=1466s)

Chapter 1 defines a **sandbox** as an execution environment that restricts file, process, or network access. It bounds files and processes, network destinations, and compute, storage, and time. Chapter 2 writes $\epsilon(\cdot)$ for a tool's possible-effect set. Its vocabulary is $\operatorname{read}(r)$, $\operatorname{write}(r)$, $\operatorname{network}(d)$, $\operatorname{spend}(v)$, and $\operatorname{disclose}(d)$. Let $\epsilon_B$ be the effects that the policy of boundary $B$ permits. The **isolation property** holds when every effect of every component inside $B$ belongs to $\epsilon_B$.

The property assumes that the **kernel**, the operating-system core that controls each process's access to hardware, enforces the policy. A **container** shares the host's kernel, so one kernel flaw can break every container on it; a virtual machine (VM) adds its own guest kernel. The property also assumes that every route out of $B$ passes the check, and that the policy is right. It says nothing about components outside $B$.

In an evaluation or RL sandbox for a coding task, the task statement, code execution, and the repository's initial state go in. Answer files, the hidden tests that grade the run, application programming interface (API) keys, and possibly the internet stay out. Each leak lets the score count improper means as success. An [issue](https://github.com/SWE-bench/SWE-bench/issues/465) on chapter 6's SWE-bench reports agents that found, in the repository history, the later commits that fixed their target bug. A research task needs the web, and a block on the benchmark's dataset by name fails against a renamed copy, the contamination route of chapter 10, Section 7. Rewarding an agent for staying inside is safety training. It changes $z_t$, not $\epsilon_B$.

### Where the boundary sits decides what it contains

A deployment can place the whole agent in the sandbox: the context builder $C_\phi$, the gate $G_\phi$, and the tools. It can instead keep the harness outside and run only tool executions inside.

With only tools inside, every external effect of the agent except its model requests passes through a tool call, and those effects lie in $\epsilon_B$. The harness outside still holds its own credentials, which the bound does not cover.

With the whole agent inside and no credential broker, the model credential must be inside too, and the network policy must allow the model's endpoint. Code in the sandbox can read the credential and call the model for its own ends; blocking such calls blocks the agent. Sandboxed code can also put any data it reads into requests to that endpoint. Placing the agent inside is simpler, and it is the only option when the agent's code is a black box. The source lecture reports one attempt at the tools-only layout, in which Model Context Protocol (MCP) servers and a web browser were harder to move than basic tools.

The [OpenShell architecture](https://github.com/NVIDIA/OpenShell/blob/ec49209da25be39840742df29b64ec694d159c2f/docs/about/architecture.mdx) keeps the agent inside and the credentials out. A supervisor on the trusted side checks every network connection from the sandbox against policy and adds credentials only to approved requests. Its documentation states that "Agents never see real credentials." The rules and their enforcement then sit outside the agent's control, as chapter 1 requires of a trusted gate.

## 4. Shared state carries authority from one run to the next [34:18](https://www.youtube.com/watch?v=Mbonnq-PUGM&t=2058s)

The triage job's token was narrow: read-only repository access and permission to write issues. Its shell could also write the build cache. The job ran on the default branch, where it shared a **cache scope**, the set of entries a job can restore, with the release job. Discarding the triage environment did not discard what the run wrote to that cache.

Let runs be indexed by $j$. Let $\epsilon_j$ be run $j$'s effect set, including effects reachable through its environment, such as a shell or a cache. Write $j\rightsquigarrow j'$ when run $j$ can write state that a later run $j'$ executes or trusts as configuration. Let $\operatorname{runs}(j)$ contain $j$ and every run reachable from it along $\rightsquigarrow$ edges. The **reach** of run $j$ is

$$\epsilon_j^*=\bigcup_{j'\in\operatorname{runs}(j)}\epsilon_{j'}.$$

Suppose an adversary controls run $j$. Suppose also, as a worst case, that any run that executes or trusts adversary-written state takes adversary-chosen actions within its own effects. Then every effect in $\epsilon_j^*$ is available to the adversary. The proof is by induction on path length. At length zero the adversary has $\epsilon_j$. If it controls run $j'$ and $j'\rightsquigarrow j''$, it writes the state that $j''$ trusts and gains $\epsilon_{j''}$. Conversely, $\epsilon_j^*$ bounds the adversary only if $\rightsquigarrow$ lists every channel between runs. Without the worst-case assumption, the bound still holds but need not be attained.

For the triage pipeline, take $\epsilon_{\rm triage}=\{\operatorname{read}(\text{issues}),\operatorname{write}(\text{labels}),\operatorname{write}(\text{cache})\}$ and $\epsilon_{\rm release}=\{\operatorname{read}(\text{cache}),\operatorname{write}(\text{registry})\}$. The release job restores the cache, so triage $\rightsquigarrow$ release, and $\operatorname{write}(\text{registry})\in\epsilon^*_{\rm triage}$. The publish token was never in the triage grant. The cache edge carried its authority across.

```python
from agent_lab.safety import reachable_effects

triage = frozenset({"issues.read", "labels.write", "cache.write"})
release = frozenset({"cache.read", "registry.write"})
effects = {"triage": triage, "release": release}
reach = reachable_effects(effects, {"triage": ["release"]}, "triage")
assert "registry.write" in reach
assert reachable_effects(effects, {}, "triage") == triage
```

Removing the edge returns the reach to the run's own effects. The postmortem's remedies act on this graph. Removing cache use from every workflow with publication credentials deletes the edge. Restricting future automation to read-only operations with no shell shrinks $\epsilon_{\rm triage}$. Artifact stores, shared volumes, a VM workspace, a website that two runs can write, and a saved sandbox snapshot are channels too; reloading a poisoned snapshot reproduces the compromise.

### Sandboxes must restart quickly and isolate strongly

A sandbox for frequent agent runs must start quickly, isolate each run, offer a simple interface, and support **freeze and reload**: saving files and running processes and resuming from that point. [E2B](https://docs.e2b.dev/sandbox/persistence) resumes "from the exact state it was in, including the filesystem and the memory." RL can branch several continuations from one saved state and compare their returns to estimate an advantage. Local [Docker](https://docs.docker.com/engine/security/) containers need a daemon that requires root unless rootless mode is chosen. [Apptainer](https://apptainer.org/docs/user/main/quick_start.html) runs containers as the invoking user, which suits shared clusters that withhold root. [QEMU](https://www.qemu.org/docs/master/system/introduction.html) runs a full VM. Cloud providers such as Modal and E2B manage sandboxes through an API.

## 5. Least privilege bounds what a steered run can do [40:24](https://www.youtube.com/watch?v=Mbonnq-PUGM&t=2424s)

**Identity** is who acts. **Authority** is which actions on which resources that identity may perform. A permission check asks who is calling, which action on which resource, and whether it is allowed for this task. The first question for a new service is how fine-grained its permissions are, since a generic write key may be able to delete the account.

[Saltzer and Schroeder](https://www.cs.virginia.edu/~evans/cs551/saltzer/) stated **least privilege** in 1975: "Every program and every user of the system should operate using the least set of privileges necessary to complete the job." Chapter 1 applies it to delegation as $\operatorname{cap}(u_v)\subseteq\operatorname{cap}(u_{\rm parent})$, where $u_v$ is the authority state of delegated subtask $v$. Chapter 2 writes $\operatorname{req}(\tau,x)$ for the rights that a call of tool $\tau$ with argument $x$ requires. Write $\operatorname{req}(a)$ when action $a$ is such a call. Let a role's tasks need the actions in a subset $\mathcal A_{\rm role}$ of chapter 1's action set $\mathcal A$. The least grant is

$$\operatorname{cap}(u_{\rm role})=\bigcup_{a\in\mathcal A_{\rm role}}\operatorname{req}(a).$$

Chapter 2 admits $a$ only when $\operatorname{req}(a)\subseteq\operatorname{cap}(u)$. Every grant that admits all of $\mathcal A_{\rm role}$ therefore contains this union, and the union admits each such $a$, so it is the smallest admitting grant. The **excess rights** of a grant are $\operatorname{cap}(u)$ minus this union. They are what a steered run gains beyond its task. The triage role needs only $\{\operatorname{read}(\text{issues}),\operatorname{write}(\text{labels})\}$; the deployed job's shell added the excess right $\operatorname{write}(\text{cache})$, the start of Section 4's edge.

A least-privilege design splits the triage pipeline between two accounts. The issue-handling account may read issues and apply labels but not publish. The release account may read approved source and publish, and it never reads untrusted text. It also restores no state that the issue-handling account can write, which deletes Section 4's edge. A steered triage agent's worst in-scope action is then a malicious label. Section 1 showed that no grant removes that residual. The source lecture reports this split in the OpenHands project's workflows, as testimony rather than measurement.

### Token lifetime bounds the misuse window

A leaked token can be misused while it stays valid. A [GitHub fine-grained personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) limits its repositories and permissions. A [GitHub App installation token](https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/generating-an-installation-access-token-for-a-github-app) "will expire after 1 hour." An **ephemeral key** lives only as long as the job that requested it. In the triage pipeline, the long-lived npm token was still valid on 17 February, eight days after disclosure. The maintainers now publish only from the CI workflow through OpenID Connect (OIDC) provenance, an attestation that links each version to its source commit and workflow run, and keep no long-lived npm token.

### A credential broker is a deputy

A **credential broker** holds the real key and mediates an agent's API access. The agent receives a short-lived token bound to its run. For each request, the broker checks the run and the call against policy. It then calls the API with the real key and records the event separately. It can add checks that the service lacks, such as blocking calls outside the task behind a general write key. [RFC 8693](https://www.rfc-editor.org/rfc/rfc8693.html) standardizes such token exchange.

Hiding an API key does not stop misuse of the access the key provides. A broker narrows access only through its checks. It also faces the **confused deputy** problem, in [Hardy's](http://cap-lore.com/CapTheory/ConfusedDeputy.html) term. A program that holds authority from two sources is tricked into using one source's authority for the other's purpose. The broker must check each request against the run's authority, never its own.

The database case is the counterexample to identity checks alone. The token was valid and carried account-wide authority. A staging-only grant would have bounded the harm. A correct identity could not.

## 6. A monitor reads a trace, and its alerts inherit the base rate [46:01](https://www.youtube.com/watch?v=Mbonnq-PUGM&t=2761s)

**Monitoring** records and checks an agent's behavior to find mistakes and unsafe actions. A check during a run can pause or block an action, and a review afterward explains failures. A useful answer can follow an unsafe action, so the evidence must include the request, the tool calls, their results, and changes to outside systems. The [OpenTelemetry generative AI conventions](https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/) name fields for the first three; the last needs the services' own records.

A **rule-based check** tests a precise condition, such as whether a package publisher is on an allowed list. It is exact for that condition and silent about others. A **model-based check** asks a judge a question that needs context, such as whether the triage agent followed instructions found in an issue. Its verdict should cite supporting passages in the log. A model-based monitor reads attacker text too, so it is itself a target of injection.

### Written reasoning is evidence that optimization can remove

**Chain-of-thought (CoT) monitoring** gives the monitor the agent's written reasoning as well as its actions. [Baker et al.](https://arxiv.org/abs/2503.11926) used a prompted GPT-4o to monitor a frontier reasoning model during RL on coding tasks. Two hacks spread across nearly all training environments. One exited before the tests ran, and one raised an exception that skipped them. String matching detected both in hindsight and served as ground truth. **Recall** is chapter 4's sensitivity, the fraction of true hacks that the monitor flags. **Precision** is the fraction of flags that are true hacks; the paper calls it less well defined here, because the monitors flag any misbehavior.

| Monitor input (GPT-4o, two systemic hacks) | Recall, early exit | Recall, skipped tests | Recall, joint | Precision, joint |
|---|---:|---:|---:|---:|
| Actions only: tool calls and final outputs | 40% | 80% | 60% | about 81% |
| Full trajectory, including CoT | 93% | 97% | 95% | about 77% |

The table covers two hacks, not all misbehavior.

The paper then added the CoT monitor to the RL reward. With little optimization, the agent hacked less and solved more. With more, it learned **obfuscated reward hacking**: it still hacked, while the monitor's recall fell to near zero. **Monitorability** is how well one monitor detects one model's misbehavior from what the model writes. The authors suggest a "monitorability tax," keeping strong optimization pressure off the CoT. [Monitoring Monitorability](https://arxiv.org/abs/2512.18311) finds that ordinary RL at current frontier scale "does not materially decrease monitorability"; Baker et al. instead rewarded passing the monitor.

### A blocking check is part of the gate

To prevent an unsafe call, the check must finish before execution and then allow, block, or pause it. A check in this position is part of chapter 1's gate $G_\phi$; a check that runs afterward supplies evidence, not prevention. The simplest blocking check is a coding agent's per-command approval prompt. Others use a smaller LM, a fine-tuned checker, or the agent's own report of danger.

[AI Control](https://arxiv.org/abs/2312.06942) tests such protocols against a model that deliberately tries to subvert them. An untrusted GPT-4, treated as a possible adversary, writes code. A trusted GPT-3.5 rates its suspicion, and trusted labor, simulated with GPT-4, can audit 2% of problems. A **backdoored** solution passes every test but fails on an input that GPT-4 can construct. Safety is the probability that no backdoored solution is submitted. Usefulness is rescaled so that GPT-3.5 alone scores 0% and GPT-4 alone 100%. In the paper's Table 2, auditing the first 2% of problems gives 100% usefulness and 15% safety. Auditing the 2% most suspicious gives 97% and 62%. Submitting a trusted edit of the code whose suspicion falls between the top 2% and top 20% gives 94% and 92%.

### Alert precision falls with the base rate

Reuse chapter 4's notation with new referents. Let $Z=1$ mean a proposed action is unsafe, and $J=1$ mean the monitor alerts. Sensitivity is $P(J=1\mid Z=1)$, and the false-positive rate is $P(J=1\mid Z=0)$. Chapter 4 derived the precision by Bayes' rule:

$$P(Z=1\mid J=1)=\frac{P(J=1\mid Z=1)P(Z=1)}{P(J=1\mid Z=1)P(Z=1)+P(J=1\mid Z=0)P(Z=0)}.$$

Take an illustrative case. Of 10,000 decisions, 1% are unsafe. The monitor catches 90% of those and flags 1% of safe ones. True alerts are $0.01\cdot0.9=0.009$ of decisions, or 90. False alerts are $0.99\cdot0.01=0.0099$, or 99. Precision is $90/189\approx0.476$, or about 48%. More than half the alerts are false, although neither error rate exceeds 10%.

```python
from agent_lab.safety import alert_precision

assert abs(alert_precision(0.01, 0.9, 0.01) - 90 / 189) < 1e-12
rarer = alert_precision(0.001, 0.9, 0.01)
assert abs(rarer - 0.0009 / 0.01089) < 1e-12
```

At a base rate of 0.1%, precision falls to about 8.3%. The alert rate is the denominator above. As $P(Z=1)$ falls toward zero it approaches the false-positive rate, so review load grows with traffic, not with attacks. The calculation assumes fixed rates. An adaptive attacker targets the monitor's blind spots, so live sensitivity can fall below test sensitivity. A blocking check must also **fail closed** for consequential actions: when the monitor is down, it blocks. Checks, logs, and the stop control must lie outside the agent's control; the inbox owner in Section 0 reported being unable to stop the agent from a phone.

## 7. Safety evaluation separates attempts from completed harm [52:57](https://www.youtube.com/watch?v=Mbonnq-PUGM&t=3177s)

Task completion cannot show safety. An evaluation needs **valid-task runs**, which pose legitimate requests. It also needs **safety-test runs**, which pose attacks, malicious requests, or traps. Four rates follow, with two denominators. The counts are illustrative.

| Rate | Numerator | Denominator | Illustrative count |
|---|---|---|---|
| **Task completion** | valid tasks completed | all valid-task runs | 80 of 100 |
| **Unnecessary refusal** | valid tasks refused | all valid-task runs | 10 of 100 |
| **Unsafe attempt** | runs with an unsafe attempt | all safety-test runs | 25 of 100 |
| **Completed harm** | runs with a harmful effect | all safety-test runs | 10 of 100 |

Completed and refused runs cannot exceed all valid runs when each run has one outcome. Completed harm cannot exceed unsafe attempts, provided every harmful effect follows a logged attempt. The difference, here 15 of 25 or 60%, counts unsafe attempts without completed harm, including **blocked attempts** that a check stopped. In `agent_lab.evaluation`, a run with `forbidden_effects > 0` fails `admissible_success`; completed harm is the fraction of safety-test runs with `forbidden_effects > 0`.

### Zero observed harms bound the rate only weakly

Chapter 1's chance constraint requires $P(F(\omega)=1)\le\delta$. Suppose $n_{\rm test}$ independent safety-test runs share a per-run harm probability $p_F$ and show no harm. That outcome has probability $(1-p_F)^{n_{\rm test}}$. A one-sided 95% upper bound $p_{\rm up}$ is the largest $p_F$ under which zero harms still occur with probability at least 0.05:

$$(1-p_{\rm up})^{n_{\rm test}}=0.05,\qquad p_{\rm up}=1-0.05^{1/n_{\rm test}}\approx\frac{3}{n_{\rm test}}.$$

The approximation uses $\ln(1-p_{\rm up})\approx-p_{\rm up}$ for small $p_{\rm up}$ and $-\ln0.05\approx3.00$. It is the **rule of three**. For 100 clean runs, $p_{\rm up}\approx0.0295$. Supporting $\delta=0.01$ needs $n_{\rm test}\ge\ln0.05/\ln0.99\approx298.1$, so 299 clean runs.

```python
from agent_lab.safety import safety_outcomes, zero_failure_upper_bound

rates = safety_outcomes(100, 80, 10, 100, 25, 10)
assert rates == (0.8, 0.1, 0.25, 0.1)
assert abs(zero_failure_upper_bound(100) - 0.0295) < 1e-4
assert zero_failure_upper_bound(299) < 0.01 < zero_failure_upper_bound(298)
```

The bound assumes that test runs come from the deployment's attack distribution and share one rate. An adaptive attacker breaks both assumptions, so a hundred clean runs support a rate below about 3% only on their own benchmark.

### Benchmarks are chosen by threat model

| Threat model | Benchmark | How it tests |
|---|---|---|
| Malicious user | AgentHarm | 110 malicious tasks, synthetic tools; scores refusal and retained capability |
| Prompt injection | [AgentDojo](https://arxiv.org/abs/2406.13352) | 97 tasks, 629 security cases; tool data carries the attack; reports utility |
| Agent error | ToolEmu | emulated tools; human review judged 68.8% of its failures real |
| Improper means | Docent-style trace review | searches transcripts for shortcuts |
| Several at once | [OpenAgentSafety](https://arxiv.org/abs/2507.06134) | over 350 multi-turn tasks with real tools; rule-based and LM-judge scoring |

OpenAgentSafety, coauthored by Graham Neubig, scores environment state with rules and attempted unsafe actions with an LM judge, separately. The full text of version 2 reports LM-judged unsafe behavior in 49% to 73% of its "safety-vulnerable" tasks across seven models; these are runs in which the agent reached malicious content without failing first. The arXiv abstract page still gives 51.2% to 72.7% across five models.

A defense comparison must hold the model, the scaffold, and the tasks fixed, and include valid requests as well as attacks. It should report utility with and without the defense (CaMeL's 77% against 84%), utility under attack, ASR, and unnecessary refusals, under adaptive attacks, because fixed attacks overstate robustness.

### Scope and limits of agent safety controls

Four results here are formal, each under stated assumptions. The at-least-one-success formula assumes independent attempts. The reach bound assumes a complete channel list and worst-case control of each trusting run. The precision identity assumes fixed rates. The zero-failure bound assumes independent runs with one shared rate. The byline follows the video description; the course site names no lecturer. The rest is empirical: single incident reports, one competition, benchmarks with their own distributions, and vendor documentation checked on 17 October 2026. A safety claim should state its threat model, attacker budget, and defense, and give the four rates with their denominators beside utility, cost, and latency.

### Exercises

1. Prove that $\epsilon_{j'}^*\subseteq\epsilon_j^*$ whenever $j\rightsquigarrow j'$. Then show that in the two-run triage pipeline $\epsilon^*_{\rm triage}=\epsilon_{\rm triage}$ holds exactly when the triage-to-release edge is absent.
2. A deployer requires $\delta=0.005$. How many clean, independent safety-test runs support this at 95% one-sided confidence, and how many at 99%? Explain why neither count certifies the deployment against an adaptive attacker.
3. Design controls for an email assistant that may read the inbox and draft replies but not send them. Give the prompt-injection threat model, the least-privilege grant, one rule-based and one model-based check, and the rates to report.

### Solutions and discussion

1. If $j\rightsquigarrow j'$, every run reachable from $j'$ is reachable from $j$ through that edge. So $\operatorname{runs}(j')\subseteq\operatorname{runs}(j)$, and a union over a subset lies inside the union over the superset. With the edge present, $\epsilon^*_{\rm triage}\supseteq\epsilon_{\rm release}$, which contains $\operatorname{write}(\text{registry})\notin\epsilon_{\rm triage}$. Equality fails. With the edge absent, $\operatorname{runs}(\text{triage})=\{\text{triage}\}$, and equality holds.
2. At 95%, $n_{\rm test}\ge\ln0.05/\ln0.995\approx597.6$, so 598 runs. At 99%, $n_{\rm test}\ge\ln0.01/\ln0.995\approx918.7$, so 919 runs. Both assume one harm probability shared by all runs. An adaptive attacker changes the attack after each failure, so later attempts are not draws from the tested distribution.
3. The principal is the user, and the adversary controls incoming email. The boundary crossed is data acquiring instruction authority. Success is an unrequested action, such as a draft that copies private mail to a stranger. The grant holds $\operatorname{read}(\text{inbox})$ and $\operatorname{write}(\text{drafts})$ only, and a broker holding the mail credential permits only those calls. A rule-based check blocks drafts addressed outside the thread's participants. A model-based check asks whether a draft follows an instruction from an email. Report the four rates on valid and injected emails.

The first answer shows that reach is monotone along the trust relation.

The second answer shows the price of evidence: halving $\delta$ from 0.01 to 0.005 doubles the required runs, from 299 to 598. Raising confidence from 95% to 99% multiplies the count by about $\ln0.01/\ln0.05\approx1.54$.

The third answer combines the chapter's controls. The grant bounds a steered run, the broker enforces the grant outside the agent, and the checks catch in-scope misuse that the grant permits.

### Further reading

- [AI Control](https://arxiv.org/abs/2312.06942): safety protocols tested against a deliberately subversive model, with usefulness reported beside safety.
- [Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation](https://arxiv.org/abs/2503.11926): CoT against action-only monitoring, and obfuscation under optimization pressure.
- [The Protection of Information in Computer Systems](https://www.cs.virginia.edu/~evans/cs551/saltzer/): least privilege and related design principles behind sandboxes and credentials.
