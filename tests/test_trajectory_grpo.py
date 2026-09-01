from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from counterfeint.training.rollout import RecordingHFInvestigator
from counterfeint.training.trajectory_grpo import (
    TrajectoryRecord,
    TrajectoryTurn,
    compute_group_advantages,
    optimise_trajectory_group,
)


def test_group_advantages_are_standardised() -> None:
    advantages = compute_group_advantages([1.0, 2.0, 3.0, 4.0])
    assert sum(advantages) == pytest.approx(0.0)
    variance = sum(value * value for value in advantages) / len(advantages)
    assert math.sqrt(variance) == pytest.approx(1.0)


def test_equal_group_rewards_produce_zero_advantage() -> None:
    assert compute_group_advantages([0.7, 0.7, 0.7, 0.7]) == [0.0] * 4


def test_recording_policy_marks_parse_fallback_and_keeps_tokens() -> None:
    class ParseFallbackPolicy:
        fallback_count = 0
        call_count = 0
        last_prompt = None
        last_completion = None
        last_prompt_token_ids = None
        last_completion_token_ids = None

        def reset(self) -> None:
            self.fallback_count = 0
            self.call_count = 0

        def act(self, _observation):
            self.call_count += 1
            self.last_prompt = "review ad_1"
            self.last_completion = "not-json"
            self.last_prompt_token_ids = [1, 2]
            self.last_completion_token_ids = [3, 4]
            self.fallback_count += 1
            return {"action_type": "verdict", "ad_id": "ad_1"}

    recorder = RecordingHFInvestigator(ParseFallbackPolicy())
    recorder.reset()
    recorder.act({})
    row = recorder.step_records[0]
    assert row["fallback_used"] is True
    assert row["prompt_token_ids"] == [1, 2]
    assert row["completion_token_ids"] == [3, 4]


def test_trajectory_group_update_backpropagates_over_action_tokens() -> None:
    class TinyLM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 6)
            self.head = torch.nn.Linear(6, 8)

        def forward(self, input_ids, use_cache=False):
            del use_cache
            return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

    model = TinyLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    records = []
    for idx, (completion, advantage) in enumerate((([3, 4], 1.0), ([5, 6], -1.0))):
        records.append(
            TrajectoryRecord(
                task_id="task_3",
                seed=31,
                group_id="task_3_31",
                trajectory_idx=idx,
                grader_score=0.5,
                investigator_reward=float(advantage),
                raw_investigator_reward=float(advantage),
                advantage=advantage,
                turns=[
                    TrajectoryTurn(
                        prompt="prompt",
                        completion="completion",
                        prompt_token_ids=[1, 2],
                        completion_token_ids=completion,
                    )
                ],
            )
        )

    before = model.head.weight.detach().clone()
    result = optimise_trajectory_group(
        model=model,
        optimizer=optimizer,
        records=records,
        max_prompt_tokens=16,
    )
    assert result["updated_turns"] == 2
    assert result["grad_norm"] > 0
    assert not torch.equal(before, model.head.weight.detach())
