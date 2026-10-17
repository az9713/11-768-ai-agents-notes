"""Behavioral checks for the equations and boundaries used in the chapters."""

import math
import tempfile
import unittest
from pathlib import Path

from agent_lab.context import (
    Candidate,
    ContextBuilder,
    compact_state,
    compacted_input_tokens,
    kv_cache_bytes,
    select_items,
    submitted_input_tokens,
)
from agent_lab.evaluation import (
    OutcomeVector,
    citation_metrics,
    cohen_kappa,
    normalized_to_pixels,
    pass_at_k,
    rubric_score,
)
from agent_lab.harness import RunState
from agent_lab.memory import MemoryItem, MemoryStore
from agent_lab.order_demo import (
    OrderService,
    ScriptedOrderModel,
    build_harness,
)
from agent_lab.planning import (
    TaskNode,
    plan_success_probability,
    ready_tasks,
    should_replan,
    topological_order,
    work_span_bound,
)
from agent_lab.safety import (
    alert_precision,
    reachable_effects,
    safety_outcomes,
    zero_failure_upper_bound,
)
from agent_lab.tools import CallRejected
from agent_lab.training import (
    batch_makespans,
    bf16_round,
    centered_group_advantages,
    cispo_coefficient,
    contrastive_loss,
    decode,
    degenerate_binary_group_probability,
    effective_token_shares,
    gae_advantages,
    greedy_encode,
    kl_divergence,
    masked_cross_entropy,
    ppo_clip_term,
    ratio_moments,
    reinforce_estimate,
    reward_to_go,
)
from agent_lab.types import Authority, FinalAnswer, Observation, ToolCall


TRIAGE = frozenset({"issues.read", "labels.write", "cache.write"})
RELEASE = frozenset({"cache.read", "registry.write"})



class HarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.goal = "Cancel order O-1 if unshipped; otherwise explain returns."
        self.rights = Authority(
            frozenset({"order.read", "order.cancel", "policy.read"})
        )

    def test_timeout_after_commit_is_reconciled_by_read(self) -> None:
        service = OrderService("unshipped", fail_after_commit=True)
        result = build_harness(service).run(
            self.goal, ScriptedOrderModel(), self.rights
        )
        self.assertEqual(service.read_status("O-1")["status"], "cancelled")
        self.assertEqual(len(service._committed), 1)
        self.assertEqual(
            [item.status for item in result.observations],
            ["ok", "unknown", "ok"],
        )
        self.assertEqual(result.events[-1].detail, "verified")

    def test_shipped_order_selects_read_only_branch(self) -> None:
        service = OrderService("shipped")
        result = build_harness(service).run(
            self.goal, ScriptedOrderModel(), self.rights
        )
        self.assertIn("return", result.answer)
        self.assertEqual(service.read_status("O-1")["status"], "shipped")
        self.assertNotIn(
            "cancel_order", [item.tool for item in result.observations]
        )

    def test_permission_gate_rejects_call_before_execution(self) -> None:
        service = OrderService("unshipped")
        tools = build_harness(service).tools
        status = Observation(
            "read-1", "get_order_status", "ok", service.read_status("O-1")
        )
        call = ToolCall(
            "cancel_order",
            {"order_id": "O-1", "expected_revision": 0},
            "cancel-1",
            "key-1",
        )
        read_only = Authority(frozenset({"order.read"}))
        with self.assertRaisesRegex(CallRejected, "not authorized"):
            tools.admit(call, read_only, [status])
        self.assertEqual(service.read_status("O-1")["status"], "unshipped")

    def test_service_rechecks_state_after_client_admission(self) -> None:
        service = OrderService("unshipped")
        tools = build_harness(service).tools
        status = Observation(
            "read-1", "get_order_status", "ok", service.read_status("O-1")
        )
        call = ToolCall(
            "cancel_order",
            {"order_id": "O-1", "expected_revision": 0},
            "cancel-1",
            "key-1",
        )
        spec = tools.admit(call, self.rights, [status])
        service.ship()
        result = tools.invoke(spec, call)
        self.assertEqual(result.status, "error")
        self.assertEqual(service.read_status("O-1")["status"], "shipped")

    def test_idempotency_key_deduplicates_same_order(self) -> None:
        service = OrderService("unshipped")
        first = service.cancel("O-1", 0, "one-logical-cancel")
        second = service.cancel("O-1", 0, "one-logical-cancel")
        self.assertEqual(first, second)
        self.assertEqual(service.read_status("O-1")["revision"], 1)
        with self.assertRaisesRegex(ValueError, "new arguments"):
            service.cancel("O-1", 1, "one-logical-cancel")

    def test_final_answer_requires_verification(self) -> None:
        class UnsupportedModel:
            def propose(self, prompt):
                return FinalAnswer("Order O-1 was cancelled.")

        service = OrderService("unshipped")
        harness = build_harness(service)
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            harness.run(
                self.goal, UnsupportedModel(), self.rights, max_steps=2
            )
        self.assertEqual(service.read_status("O-1")["status"], "unshipped")

    def test_proposal_trace_does_not_copy_arguments(self) -> None:
        class SensitiveModel:
            def __init__(self) -> None:
                self.first = True
                self.delegate = ScriptedOrderModel()

            def propose(self, prompt):
                if self.first:
                    self.first = False
                    return ToolCall(
                        "get_order_status",
                        {"order_id": "O-1", "secret": "private-value"},
                        "rejected-1",
                    )
                return self.delegate.propose(prompt)

        result = build_harness(OrderService("shipped")).run(
            self.goal, SensitiveModel(), self.rights
        )
        self.assertNotIn("private-value", repr(result.events))


class ContextTests(unittest.TestCase):
    def test_chapter_three_token_equations(self) -> None:
        self.assertEqual(submitted_input_tokens(100, 2000, 300), 1_685_000)
        self.assertEqual(
            compacted_input_tokens(100, 2000, 300, 20, 1000), 565_000
        )
        self.assertEqual(
            kv_cache_bytes(32, 128_000, 8, 128, 2), 16_777_216_000
        )

    def test_context_knapsack_checks_combinations(self) -> None:
        items = [
            Candidate("large", 6, 8.0),
            Candidate("small-a", 3, 5.0),
            Candidate("small-b", 3, 5.0),
        ]
        self.assertEqual(
            select_items(items, 6), ("small-a", "small-b")
        )

    def test_compaction_preserves_constraints(self) -> None:
        state = RunState(
            "Repair the build",
            Authority(frozenset()),
            ("Do not publish.",),
        )
        state.observations.extend(
            [
                Observation("call-1", "read", "ok", {"value": 1}),
                Observation("call-2", "test", "ok", {"value": 2}),
            ]
        )
        compact = compact_state(state, keep_recent=1)
        self.assertEqual(compact.hard_constraints, ("Do not publish.",))
        self.assertEqual(compact.evidence_ids, ("call-1", "call-2"))
        self.assertEqual(compact.recent_observations[0].call_id, "call-2")
        prompt = ContextBuilder(keep_recent=1).build(state, ())
        self.assertIn("Do not publish.", prompt.hard_constraints)


class MemoryTests(unittest.TestCase):
    def test_skill_enters_prompt_without_granting_effects(self) -> None:
        store = MemoryStore()
        store.write(
            MemoryItem(
                "return-skill", "skill",
                "Read the current return policy before answering.",
                "trusted authored procedure", "shopping", 0.0, None,
                8, 0.3, 0.01, ("return",),
                frozenset({"policy.read"}),
            )
        )

        class CapturingModel:
            def __init__(self) -> None:
                self.prompts = []

            def propose(self, prompt):
                self.prompts.append(prompt)
                return FinalAnswer(
                    "Order O-1 shipped; check its return terms."
                )

        model = CapturingModel()
        service = OrderService("shipped")
        authority = Authority(
            frozenset({"policy.read"}), frozenset({"shopping"})
        )
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            build_harness(service, store).run(
                "Find return terms", model, authority, max_steps=1
            )
        self.assertEqual(
            model.prompts[0].memories,
            ("Read the current return policy before answering.",),
        )
        self.assertEqual(authority.effects, frozenset({"policy.read"}))

    def test_skill_obeys_scope_time_effects_and_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            store = MemoryStore(path)
            store.write(
                MemoryItem(
                    "return-skill",
                    "skill",
                    "Find the merchant's current dated return policy.",
                    "authored and tested procedure",
                    "shopping",
                    0.0,
                    200.0,
                    10,
                    0.3,
                    0.02,
                    ("return",),
                    frozenset({"policy.read"}),
                )
            )
            store.write(
                MemoryItem(
                    "stale-policy",
                    "fact",
                    "Old return deadline",
                    "old merchant page",
                    "shopping",
                    0.0,
                    50.0,
                    5,
                    0.5,
                    0.01,
                    ("return",),
                )
            )
            reopened = MemoryStore(path)
            authority = Authority(
                frozenset({"policy.read"}), frozenset({"shopping"})
            )
            found = reopened.retrieve(
                "Find return terms", authority, budget=10, now=100.0
            )
            self.assertEqual(
                [item.item_id for item in found], ["return-skill"]
            )
            no_right = Authority(frozenset(), frozenset({"shopping"}))
            self.assertEqual(
                reopened.retrieve("Find return terms", no_right, now=100.0),
                (),
            )
            self.assertEqual(
                reopened.retrieve("Find return terms", authority, now=201.0),
                (),
            )


class PlanningEvaluationTrainingTests(unittest.TestCase):
    def test_dependency_plan_and_work_span_bound(self) -> None:
        tasks = (
            TaskNode("inspect", 2),
            TaskNode("edit", 4, frozenset({"inspect"})),
            TaskNode("test", 3, frozenset({"edit"})),
            TaskNode("docs", 2, frozenset({"inspect"})),
        )
        self.assertEqual(topological_order(tasks)[0], "inspect")
        self.assertEqual(ready_tasks(tasks, frozenset()), ("inspect",))
        self.assertEqual(work_span_bound(tasks, 2), (11, 9, 9))
        with self.assertRaisesRegex(ValueError, "cycle"):
            topological_order((
                TaskNode("a", 1, frozenset({"b"})),
                TaskNode("b", 1, frozenset({"a"})),
            ))

    def test_plan_reliability_and_replan_rules(self) -> None:
        self.assertAlmostEqual(
            plan_success_probability([0.97] * 10), 0.97**10
        )
        self.assertFalse(should_replan(0.8, 0.85, 0.1,
                                       current_plan_valid=True))
        self.assertTrue(should_replan(0.8, 0.81, 0.1,
                                      current_plan_valid=False))

    def test_outcome_metrics_keep_distinct_questions_distinct(self) -> None:
        self.assertAlmostEqual(pass_at_k(10, 2, 3), 8 / 15)
        outcome = OutcomeVector(True, True, 0, 0.12, 4.5)
        self.assertTrue(outcome.admissible_success)
        self.assertAlmostEqual(
            rubric_score({"saved": 2, "verified": 1},
                         {"saved": True, "verified": False}),
            2 / 3,
        )
        self.assertEqual(
            normalized_to_pixels(0.25, 0.75, 1440, 900),
            (359.75, 674.25),
        )

    def test_training_equations_match_worked_examples(self) -> None:
        self.assertAlmostEqual(
            masked_cross_entropy([0.5, 0.25], [True, True]),
            1.03972077084,
        )
        self.assertEqual(
            reinforce_estimate([[0.5], [-0.5]], [1, 0], [0.5, 0.5]),
            (0.25,),
        )
        self.assertEqual(
            centered_group_advantages([1, 0, 2]),
            (0, -1, 1),
        )
        self.assertEqual(reward_to_go([0.2, 1.0], 0.9), (1.1, 1.0))
        self.assertEqual(
            effective_token_shares([0.5, 0.5], [4000, 800], [1, 1]),
            (5/6, 1/6),
        )
        self.assertAlmostEqual(
            degenerate_binary_group_probability(0.01, 8),
            0.01**8 + 0.99**8,
        )


class DeepResearchTests(unittest.TestCase):
    def test_citation_accuracy_can_rise_while_recall_does_not(self) -> None:
        cited = [[True], [True], [False], [True, True], []]
        self.assertEqual(citation_metrics(cited), (0.8, 4, 0.6))
        uncited = [[True], [True], [], [True, True], []]
        self.assertEqual(citation_metrics(uncited), (1.0, 4, 0.6))
        with self.assertRaisesRegex(ValueError, "pair"):
            citation_metrics([[], []])

    def test_citation_metrics_rejects_non_boolean_judgments(self) -> None:
        with self.assertRaisesRegex(ValueError, "booleans"):
            citation_metrics([[1, 2]])

    def test_kappa_removes_agreement_expected_by_chance(self) -> None:
        self.assertAlmostEqual(
            cohen_kappa([True, True, True, False],
                        [True, True, False, False]),
            0.5,
        )
        with self.assertRaisesRegex(ValueError, "undefined"):
            cohen_kappa([True, True], [True, True])

    def test_kappa_rejects_non_boolean_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "booleans"):
            cohen_kappa([1, 0], [1, 1])

    def test_contrastive_loss_is_softmax_cross_entropy(self) -> None:
        self.assertAlmostEqual(
            contrastive_loss([2.0, 1.0, 0.0], 0), 0.40760596444
        )
        self.assertAlmostEqual(
            contrastive_loss([12.0, 11.0, 10.0], 0),
            contrastive_loss([2.0, 1.0, 0.0], 0),
        )
        self.assertGreater(
            contrastive_loss([2.0, 1.0, 0.0], 2),
            contrastive_loss([2.0, 1.0, 0.0], 0),
        )


class Ch11GaeTest(unittest.TestCase):
    rewards = [0.0, 0.0, 0.0]
    values = [0.3, 0.6, 0.4, 0.0]

    def test_half_trace_matches_worked_example(self):
        advantages = gae_advantages(self.rewards, self.values, 1.0, 0.5)
        expected = (0.1, -0.4, -0.4)
        for got, want in zip(advantages, expected):
            self.assertAlmostEqual(got, want, places=12)

    def test_lambda_zero_is_td_error(self):
        advantages = gae_advantages(self.rewards, self.values, 1.0, 0.0)
        for got, want in zip(advantages, (0.3, -0.2, -0.4)):
            self.assertAlmostEqual(got, want, places=12)

    def test_lambda_one_telescopes_to_return_minus_value(self):
        advantages = gae_advantages(self.rewards, self.values, 1.0, 1.0)
        for t, got in enumerate(advantages):
            self.assertAlmostEqual(got, sum(self.rewards[t:]) - self.values[t])

    def test_rejects_wrong_value_count(self):
        with self.assertRaises(ValueError):
            gae_advantages([0.0], [0.0], 1.0, 0.5)


class Ch11RatioMomentsTest(unittest.TestCase):
    def test_slide_policies(self):
        mean, variance, second = ratio_moments([0.4, 0.4, 0.2],
                                               [0.7, 0.2, 0.1])
        kl_behavior_target = kl_divergence([0.4, 0.4, 0.2], [0.7, 0.2, 0.1])
        self.assertAlmostEqual(mean, -kl_behavior_target, places=12)
        self.assertAlmostEqual(second, 1.375, places=12)
        self.assertAlmostEqual(variance, 0.37666, places=4)

    def test_equal_policies_have_unit_weight(self):
        mean, variance, second = ratio_moments([0.5, 0.5], [0.5, 0.5])
        self.assertEqual((mean, variance, second), (0.0, 0.0, 1.0))

    def test_rejects_missing_support(self):
        with self.assertRaises(ValueError):
            ratio_moments([1.0, 0.0], [0.5, 0.5])


class Ch11ClipTest(unittest.TestCase):
    def test_positive_advantage_above_cap_has_no_gradient(self):
        value, coefficient = ppo_clip_term(1.75, 1.0, 0.2, 0.2)
        self.assertAlmostEqual(value, 1.2)
        self.assertEqual(coefficient, 0.0)

    def test_negative_advantage_above_cap_is_unbounded(self):
        value, coefficient = ppo_clip_term(1.75, -1.0, 0.2, 0.2)
        self.assertAlmostEqual(value, -1.75)
        self.assertAlmostEqual(coefficient, -1.75)

    def test_negative_advantage_below_floor_has_no_gradient(self):
        value, coefficient = ppo_clip_term(0.5, -1.0, 0.2, 0.2)
        self.assertAlmostEqual(value, -0.8)
        self.assertEqual(coefficient, 0.0)

    def test_inside_interval_keeps_raw_term(self):
        self.assertEqual(ppo_clip_term(1.1, 2.0, 0.2, 0.2), (2.2, 2.2))

    def test_cispo_keeps_every_token(self):
        self.assertAlmostEqual(cispo_coefficient(1.75, 1.0, 0.2), 1.2)
        self.assertAlmostEqual(cispo_coefficient(1.75, -1.0, 0.2), -1.2)
        self.assertAlmostEqual(cispo_coefficient(0.5, -1.0, 0.2), -0.5)
        self.assertAlmostEqual(cispo_coefficient(0.5, -1.0, 0.2, 0.2), -0.8)


class Ch11KlTest(unittest.TestCase):
    def test_distillation_example_and_asymmetry(self):
        student, teacher = [0.5, 0.5], [0.8, 0.2]
        self.assertAlmostEqual(kl_divergence(student, teacher), 0.2231, 4)
        self.assertAlmostEqual(kl_divergence(teacher, student), 0.1927, 4)

    def test_three_action_mode_seeking_example(self):
        teacher = [0.49, 0.02, 0.49]
        committed, even = [0.9, 0.05, 0.05], [1 / 3] * 3
        self.assertAlmostEqual(kl_divergence(committed, teacher), 0.479, 3)
        self.assertAlmostEqual(kl_divergence(even, teacher), 0.681, 3)
        self.assertAlmostEqual(kl_divergence(teacher, committed), 0.802, 3)
        self.assertAlmostEqual(kl_divergence(teacher, even), 0.321, 3)

    def test_zero_only_for_equal_distributions(self):
        self.assertEqual(kl_divergence([0.3, 0.7], [0.3, 0.7]), 0.0)
        self.assertTrue(math.isinf(kl_divergence([0.5, 0.5], [1.0, 0.0])))


class Ch12RLSystemsTest(unittest.TestCase):
    def test_static_batches_wait_for_their_longest_request(self):
        static, continuous = batch_makespans([10, 120, 30, 40], slots=2)
        self.assertEqual(static, 160.0)
        self.assertEqual(continuous, 120.0)

    def test_continuous_batching_respects_lower_and_upper_bounds(self):
        durations = [10, 120, 30, 40, 20, 30, 60, 10]
        _, continuous = batch_makespans(durations, slots=4)
        lower = max(max(durations), sum(durations) / 4)
        upper = sum(durations) / 4 + (1 - 1 / 4) * max(durations)
        self.assertTrue(lower <= continuous <= upper)

    def test_equal_durations_take_whole_waves(self):
        self.assertEqual(batch_makespans([5.0] * 9, slots=4), (15.0, 15.0))

    def test_makespan_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            batch_makespans([1.0], slots=0)
        with self.assertRaises(ValueError):
            batch_makespans([-1.0], slots=1)

    def test_bf16_keeps_seven_fraction_bits(self):
        self.assertEqual(bf16_round(1.0 + 2**-7), 1.0 + 2**-7)
        self.assertEqual(bf16_round(1.0 + 2**-9), 1.0)
        # A tie rounds to the even neighbour.
        self.assertEqual(bf16_round(1.0 + 2**-8), 1.0)
        self.assertEqual(bf16_round(1.0 + 3 * 2**-8), 1.0 + 2**-6)
        self.assertEqual(bf16_round(-2.5), -2.5)
        with self.assertRaises(ValueError):
            bf16_round(float("nan"))

    def test_bf16_absorbs_update_below_half_spacing(self):
        w = bf16_round(0.0177)
        self.assertEqual(w, 145 * 2**-13)
        self.assertEqual(bf16_round(w + 3e-6), w)
        self.assertNotEqual(bf16_round(w + 2**-13), w)

    def test_retokenization_loses_sampled_tokens(self):
        vocab = ("H", "e", "l", "Hel", "Hello", "o", "el", "lo")
        self.assertEqual(decode((3, 7), vocab), "Hello")
        self.assertEqual(greedy_encode("Hello", vocab), (4,))
        self.assertEqual(greedy_encode("Hel", vocab), (3,))
        with self.assertRaises(ValueError):
            greedy_encode("x", vocab)


class ReachableEffectsTests(unittest.TestCase):
    def test_shared_cache_carries_publish_to_triage(self) -> None:
        effects = {"triage": TRIAGE, "release": RELEASE}
        reach = reachable_effects(effects, {"triage": ["release"]}, "triage")
        self.assertIn("registry.write", reach)
        self.assertEqual(reach, TRIAGE | RELEASE)

    def test_removing_the_trust_edge_restores_the_run_bound(self) -> None:
        effects = {"triage": TRIAGE, "release": RELEASE}
        self.assertEqual(reachable_effects(effects, {}, "triage"), TRIAGE)

    def test_reach_is_transitive_and_cycles_terminate(self) -> None:
        effects = {
            "a": frozenset({"x"}),
            "b": frozenset({"y"}),
            "c": frozenset({"z"}),
        }
        trusts = {"a": ["b"], "b": ["c"], "c": ["a"]}
        self.assertEqual(
            reachable_effects(effects, trusts, "a"), frozenset("xyz")
        )

    def test_trust_does_not_flow_backward(self) -> None:
        effects = {"triage": TRIAGE, "release": RELEASE}
        reach = reachable_effects(effects, {"triage": ["release"]}, "release")
        self.assertEqual(reach, RELEASE)

    def test_unknown_run_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            reachable_effects({"triage": TRIAGE}, {"triage": ["x"]}, "triage")


class AlertPrecisionTests(unittest.TestCase):
    def test_slide_example_gives_ninety_of_189(self) -> None:
        self.assertAlmostEqual(
            alert_precision(0.01, 0.9, 0.01), 90 / 189, places=12
        )

    def test_rarer_unsafe_actions_lower_precision(self) -> None:
        self.assertLess(
            alert_precision(0.001, 0.9, 0.01),
            alert_precision(0.01, 0.9, 0.01),
        )
        self.assertAlmostEqual(
            alert_precision(0.001, 0.9, 0.01), 0.0009 / 0.01089, places=12
        )

    def test_rates_outside_unit_interval_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            alert_precision(1.5, 0.9, 0.01)

    def test_silent_monitor_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            alert_precision(0.01, 0.0, 0.0)


class SafetyOutcomeTests(unittest.TestCase):
    def test_slide_counts_use_two_denominators(self) -> None:
        self.assertEqual(
            safety_outcomes(100, 80, 10, 100, 25, 10),
            (0.8, 0.1, 0.25, 0.1),
        )

    def test_harm_without_attempt_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            safety_outcomes(100, 80, 10, 100, 5, 10)

    def test_overlapping_completion_and_refusal_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            safety_outcomes(100, 95, 10, 100, 25, 10)


class ZeroFailureBoundTests(unittest.TestCase):
    def test_hundred_clean_runs_bound_about_three_percent(self) -> None:
        self.assertAlmostEqual(zero_failure_upper_bound(100), 0.0295, 4)

    def test_299_runs_are_needed_for_one_percent(self) -> None:
        self.assertLess(zero_failure_upper_bound(299), 0.01)
        self.assertGreater(zero_failure_upper_bound(298), 0.01)

    def test_invalid_arguments_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            zero_failure_upper_bound(0)
        with self.assertRaises(ValueError):
            zero_failure_upper_bound(10, confidence=1.0)


if __name__ == "__main__":
    unittest.main()
