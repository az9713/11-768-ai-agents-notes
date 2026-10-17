# Companion Python harness

This standard-library package accompanies the ten current study chapters. It
runs without a model account or network connection. The proposal source in
`order_demo.py` is deterministic so the mechanics can be tested separately from
model variability.

| Chapter | Code | Mathematical object |
|---|---|---|
| 1, agents | `types.py`, `harness.py` | Run state, proposal, admission, observation, verified stopping |
| 2, tools | `tools.py`, `order_demo.py` | Effect set, schema, precondition, atomic conditional update, uncertain outcome |
| 3, context | `context.py` | Token-volume equations, KV-cache size, knapsack, compact state |
| 4, memory | `memory.py` | Scope and validity predicates, budgeted retrieval, file-backed persistence |
| 5, planning | `planning.py` | Dependency graph, ready set, work/span, reliability product, and replanning rule |
| 6–7, evaluation | `evaluation.py` | pass@k, multidimensional outcomes, weighted rubrics, and coordinate transforms |
| 8–9, training | `training.py` | Masked loss, effective mixtures, reward-to-go, REINFORCE, and group diagnostics |
| 10, deep research | `evaluation.py`, `training.py` | Chance-corrected judge agreement, citation accuracy and recall, contrastive retrieval loss |

From the repository root, run:

```sh
python -m agent_lab.order_demo
python -m unittest discover -s tests -v
```

The complete loop is `Harness.run`. Its `Model` protocol accepts a `Prompt`
and returns a `ToolCall` or `FinalAnswer`. A provider adapter can implement
that protocol, but must map provider responses to these types and treat tool
output as lower-trust data. It must not execute a proposed call itself. The
application supplies `Authority`, registered `ToolSpec` objects, and a
verifier. The model cannot change any of them.

The order service is in memory. Its lock and request ledger illustrate atomic
conditional cancellation and idempotency within one process; they do not
survive a restart. The event record and memory store are not an audit-grade
checkpoint system. The context builder uses whitespace-separated units as a
teaching proxy, not a provider tokenizer. There is no network sandbox,
concurrent dependency scheduler, optimizer, payment tool, or model-initiated memory write.
Those features require service-specific interfaces, permissions, and tests.
Before connecting a remote tool, its adapter must classify uncertain outcomes
and avoid putting secrets from arguments or exception messages into traces.

The package is a coherent **single-run educational harness**, not a production
agent runtime. Later course chapters can extend the same boundaries without
changing the central rule: a model proposes, the harness authorizes, an
environment executes, observations update the next decision, and a verifier
decides whether the run has completed.
