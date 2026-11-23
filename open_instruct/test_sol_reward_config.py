import ast
import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import ray.cloudpickle as ray_cloudpickle

from open_instruct import data_loader, data_types, ground_truth_utils


class TestSolRewardConfig(unittest.TestCase):
    def _infos(self, count):
        return SimpleNamespace(
            timeouts=[False] * count,
            tool_errors=[False] * count,
            tool_outputs=[[] for _ in range(count)],
            tool_calleds=[False] * count,
            rollout_states=[{} for _ in range(count)],
        )

    def test_reward_config_uses_explicit_correctness_and_runs_before_group_filter(self):
        verifier_mock = AsyncMock(return_value=([0.0, 0.0], [{}, {}], [True, True]))
        reward_fn = ground_truth_utils.RewardConfig(use_sol_reward=True).build()
        with patch.object(ground_truth_utils, "apply_verifiable_reward", new=verifier_mock):
            scores, metrics = asyncio.run(
                reward_fn(
                    responses=[[1] * 10, [2] * 20],
                    decoded_responses=["a", "b"],
                    ground_truths=["x", "x"],
                    datasets=["test", "test"],
                    finish_reasons=["stop", "stop"],
                    infos=self._infos(2),
                )
            )

        self.assertEqual(scores, [2.0, 1.99])
        self.assertGreater(np.std(scores), 0.0)
        self.assertEqual(metrics["_sol_verifier_correctness"], [True, True])
        self.assertTrue(verifier_mock.await_args.kwargs["return_correctness"])

    def test_disabled_feature_preserves_verifier_rewards(self):
        verifier_mock = AsyncMock(return_value=([0.0, 10.0], [{}, {}]))
        reward_fn = ground_truth_utils.RewardConfig(use_sol_reward=False).build()
        with patch.object(ground_truth_utils, "apply_verifiable_reward", new=verifier_mock):
            scores, metrics = asyncio.run(
                reward_fn(
                    responses=[[1], [2, 3]],
                    decoded_responses=["a", "b"],
                    ground_truths=["x", "x"],
                    datasets=["test", "test"],
                    finish_reasons=["stop", "stop"],
                    infos=self._infos(2),
                )
            )

        self.assertEqual(scores, [0.0, 10.0])
        self.assertNotIn("_sol_verifier_correctness", metrics)
        self.assertFalse(verifier_mock.await_args.kwargs["return_correctness"])

    def test_real_binary_verifier_correctness_flows_into_sol(self):
        reward_fn = ground_truth_utils.RewardConfig(
            use_sol_reward=True,
            verifier_functions={"gsm8k": ground_truth_utils.GSM8KVerifier()},
        ).build()
        scores, metrics = asyncio.run(
            reward_fn(
                responses=[[1] * 10, [2] * 20, [3] * 15],
                decoded_responses=["The answer is 42.", "The answer is 42.", "The answer is 43."],
                ground_truths=["42", "42", "42"],
                datasets=["gsm8k"] * 3,
                finish_reasons=["stop"] * 3,
                infos=self._infos(3),
            )
        )

        self.assertEqual(metrics["_sol_verifier_correctness"], [True, True, False])
        self.assertEqual(scores, [2.0, 1.99, -0.005])

    def test_config_rejects_ambiguous_reward_combinations(self):
        with self.assertRaises(ValueError):
            ground_truth_utils.RewardConfig(use_sol_reward=True, apply_r1_style_format_reward=True).build()
        with self.assertRaises(ValueError):
            ground_truth_utils.RewardConfig(use_sol_reward=True, reward_aggregator="sum").build()

    def test_sol_coefficients_are_validated_during_configuration_construction(self):
        invalid_values = [
            (-1.0, 0.001),
            (float("inf"), 0.001),
            ("2.0", 0.001),
            (2.0, -0.001),
            (2.0, float("nan")),
            (2.0, "0.001"),
        ]
        for alpha, beta in invalid_values:
            with self.subTest(alpha=alpha, beta=beta):
                with self.assertRaises(ValueError):
                    data_loader.StreamingDataLoaderConfig(use_sol_reward=True, sol_alpha=alpha, sol_beta=beta)
                with self.assertRaises(ValueError):
                    ground_truth_utils.RewardConfig(use_sol_reward=True, sol_alpha=alpha, sol_beta=beta)

    def test_feature_off_defaults_and_both_entrypoint_propagation(self):
        streaming_config = data_loader.StreamingDataLoaderConfig()
        reward_config = ground_truth_utils.RewardConfig()
        self.assertFalse(streaming_config.use_sol_reward)
        self.assertEqual(streaming_config.sol_alpha, 2.0)
        self.assertEqual(streaming_config.sol_beta, 0.001)
        self.assertFalse(reward_config.use_sol_reward)
        self.assertEqual(reward_config.sol_alpha, 2.0)
        self.assertEqual(reward_config.sol_beta, 0.001)

        repo_root = Path(__file__).resolve().parents[1]
        for entrypoint in ("grpo.py", "grpo_fast.py"):
            with self.subTest(entrypoint=entrypoint):
                source = (repo_root / "open_instruct" / entrypoint).read_text()
                syntax_tree = ast.parse(source)
                reward_config_calls = [
                    node
                    for node in ast.walk(syntax_tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "RewardConfig"
                ]
                self.assertEqual(len(reward_config_calls), 1)
                forwarded = {keyword.arg: ast.unparse(keyword.value) for keyword in reward_config_calls[0].keywords}
                self.assertEqual(forwarded["use_sol_reward"], "streaming_config.use_sol_reward")
                self.assertEqual(forwarded["sol_alpha"], "streaming_config.sol_alpha")
                self.assertEqual(forwarded["sol_beta"], "streaming_config.sol_beta")

    def test_verifier_correctness_survives_ray_serialization_round_trip(self):
        result = data_types.GenerationResult(
            responses=[[1], [2]],
            finish_reasons=["stop", "stop"],
            masks=[[1], [1]],
            request_info=data_types.RequestInfo(
                num_calls=[0, 0],
                timeouts=[False, False],
                tool_errors=["", ""],
                tool_outputs=["", ""],
                tool_runtimes=[0.0, 0.0],
                tool_calleds=[False, False],
            ),
            index=0,
            prompt_id="sol-serialization",
            verifier_correctness=[True, False],
        )

        restored = ray_cloudpickle.loads(ray_cloudpickle.dumps(result))

        self.assertEqual(restored.verifier_correctness, [True, False])
        self.assertEqual(restored.responses, result.responses)


if __name__ == "__main__":
    unittest.main()
