import unittest
from types import SimpleNamespace

from open_instruct import data_loader, data_types
from open_instruct.dataset_transformation import (
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    RAW_PROMPT_KEY,
    VERIFIER_SOURCE_KEY,
)
from open_instruct.sol_utils import compute_sol_rewards


class TestSolRewardIntegration(unittest.TestCase):
    def _process_group(self, result, dataset):
        class Tokenizer:
            eos_token_id = 0

            def batch_decode(self, responses, skip_special_tokens=False):
                return [str(response) for response in responses]

        return data_loader.process_group(
            result=result,
            generation_config=SimpleNamespace(n=2),
            tokenizer=Tokenizer(),
            dataset=dataset,
            max_possible_score=10.0,
            no_resampling_pass_rate=None,
            iter_dataloader=None,
            filter_zero_std_samples=True,
            replenish_prompts=False,
            param_prompt_Q=None,
            base_env_config=None,
        )

    def _result(self, index, correctness, lengths, token):
        responses = [[token + i] * length for i, length in enumerate(lengths)]
        return data_types.GenerationResult(
            responses=responses,
            finish_reasons=["stop", "stop"],
            masks=[[1] * length for length in lengths],
            request_info=data_types.RequestInfo(
                num_calls=[0, 0],
                timeouts=[False, False],
                tool_errors=["", ""],
                tool_outputs=["", ""],
                tool_runtimes=[0.0, 0.0],
                tool_calleds=[False, False],
            ),
            index=index,
            prompt_id=f"p{index}",
            token_statistics=data_types.TokenStatistics(1, sum(lengths), 0.1),
            logprobs=[[-0.1] * length for length in lengths],
            reward_scores=compute_sol_rewards(correctness, lengths),
            verifier_correctness=correctness,
        )

    def test_sized_group_alignment_survives_group_filtering(self):
        result = data_types.GenerationResult(
            responses=[[1] * 10, [2] * 20],
            finish_reasons=["stop", "stop"],
            masks=[[1] * 10, [1] * 20],
            request_info=data_types.RequestInfo(
                num_calls=[0, 0],
                timeouts=[False, False],
                tool_errors=["", ""],
                tool_outputs=["", ""],
                tool_runtimes=[0.0, 0.0],
                tool_calleds=[False, False],
            ),
            index=0,
            prompt_id="p0",
            token_statistics=data_types.TokenStatistics(1, 30, 0.1),
            logprobs=[[-0.1] * 10, [-0.1] * 20],
            reward_scores=[2.0, 1.99],
            verifier_correctness=[True, True],
        )

        class Tokenizer:
            eos_token_id = 0

            def batch_decode(self, responses, skip_special_tokens=False):
                return [str(response) for response in responses]

        result.reward_scores = [2.0, 2.0]
        filtered_group = data_loader.process_group(
            result=result,
            generation_config=SimpleNamespace(n=2),
            tokenizer=Tokenizer(),
            dataset=[
                {
                    INPUT_IDS_PROMPT_KEY: [7],
                    GROUND_TRUTHS_KEY: "x",
                    VERIFIER_SOURCE_KEY: "test",
                    RAW_PROMPT_KEY: "question",
                }
            ],
            max_possible_score=10.0,
            no_resampling_pass_rate=None,
            iter_dataloader=None,
            filter_zero_std_samples=True,
            replenish_prompts=False,
            param_prompt_Q=None,
            base_env_config=None,
        )
        self.assertIsNone(filtered_group)

        result.reward_scores = [2.0, 1.99]
        group = data_loader.process_group(
            result=result,
            generation_config=SimpleNamespace(n=2),
            tokenizer=Tokenizer(),
            dataset=[
                {
                    INPUT_IDS_PROMPT_KEY: [7],
                    GROUND_TRUTHS_KEY: "x",
                    VERIFIER_SOURCE_KEY: "test",
                    RAW_PROMPT_KEY: "question",
                }
            ],
            max_possible_score=10.0,
            no_resampling_pass_rate=None,
            iter_dataloader=None,
            filter_zero_std_samples=True,
            replenish_prompts=False,
            param_prompt_Q=None,
            base_env_config=None,
        )

        self.assertIsNotNone(group)
        self.assertEqual(group.reward_scores, [2.0, 1.99])
        self.assertEqual(group.decoded_responses, [str([1] * 10), str([2] * 20)])
        self.assertAlmostEqual(group.percent_solved, 1.0)

        combined_result, batch, _, _ = data_loader.make_batch_from_groups(
            [group], generation_config=SimpleNamespace(n=2), training_step=0
        )
        packed = data_loader.pack_sequences(
            queries=batch.queries,
            responses=combined_result.responses,
            masks=combined_result.masks,
            pack_length=64,
            pad_token_id=0,
            vllm_logprobs=combined_result.logprobs,
        )
        self.assertEqual(batch.scores, [2.0, 1.99])
        self.assertEqual(
            list(zip(combined_result.responses, batch.scores)),
            [([1] * 10, 2.0), ([2] * 20, 1.99)],
        )
        self.assertEqual(packed.original_responses, combined_result.responses)

    def test_multi_prompt_alignment_survives_filtering_and_packing(self):
        dataset = [
            {
                INPUT_IDS_PROMPT_KEY: [7 + index],
                GROUND_TRUTHS_KEY: f"answer-{index}",
                VERIFIER_SOURCE_KEY: "test",
                RAW_PROMPT_KEY: f"question-{index}",
            }
            for index in range(3)
        ]
        results = [
            self._result(0, [True, False], [10, 20], 10),
            self._result(1, [False, True], [30, 10], 20),
            self._result(2, [True, True], [15, 15], 30),
        ]

        groups = [self._process_group(result, dataset) for result in results]

        self.assertIsNotNone(groups[0])
        self.assertIsNotNone(groups[1])
        self.assertIsNone(groups[2])
        self.assertEqual(groups[0].reward_scores, [2.0, -0.01])
        self.assertEqual(groups[1].reward_scores, [-0.02, 2.0])
        self.assertEqual(groups[0].result.verifier_correctness, [True, False])
        self.assertEqual(groups[1].result.verifier_correctness, [False, True])

        combined_result, batch, _, _ = data_loader.make_batch_from_groups(
            [groups[0], groups[1]], generation_config=SimpleNamespace(n=2), training_step=0
        )
        packed = data_loader.pack_sequences(
            queries=batch.queries,
            responses=combined_result.responses,
            masks=combined_result.masks,
            pack_length=128,
            pad_token_id=0,
            vllm_logprobs=combined_result.logprobs,
        )

        expected_responses = results[0].responses + results[1].responses
        expected_scores = [2.0, -0.01, -0.02, 2.0]
        self.assertEqual(combined_result.responses, expected_responses)
        self.assertEqual(batch.scores, expected_scores)
        self.assertEqual(list(zip(combined_result.responses, batch.scores)), list(zip(expected_responses, expected_scores)))
        self.assertEqual(packed.original_responses, expected_responses)

if __name__ == "__main__":
    unittest.main()
