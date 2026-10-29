# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("tensordict")
pytest.importorskip("verl")

import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto

from agentlightning.verl.agl_rollout_manager import CompletedRollout, Triplet
from agentlightning.verl.rollout_adapter import RolloutAdapter
from agentlightning.verl.trainer import AgentLightningRayPPOTrainer


class ValidationTrainer(AgentLightningRayPPOTrainer):
    """Evaluate real adapter metrics without starting rollout servers or Ray workers."""

    def __init__(self, batch_sizes: list[int], *, n: int = 1, legacy_manager: bool = False) -> None:
        self.config = OmegaConf.create(
            {"actor_rollout_ref": {"rollout": {"val_kwargs": {"n": n, "do_sample": True, "temperature": 0.7}}}}
        )
        self.tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0)
        self.global_steps = 7
        if legacy_manager:
            self.async_rollout_manager = SimpleNamespace(server_addresses=["localhost:8000"])
        else:
            self.llm_server_manager = SimpleNamespace(get_addresses=lambda: ["localhost:8000"])
        self.val_dataloader = []
        offset = 0
        for size in batch_sizes:
            self.val_dataloader.append({"sample_id": np.arange(offset, offset + size)})
            offset += size
        self.adapter = RolloutAdapter(
            max_prompt_length=16,
            max_response_length=16,
            device=torch.device("cpu"),
            pad_token_id=0,
            reward_fillna_value=0.25,
        )
        self.rollout_batch_sizes: list[int] = []

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        return batch

    def _rollout(self, gen_batch: DataProto, is_train: bool = True) -> tuple[DataProto, dict[str, Any]]:
        assert not is_train
        assert gen_batch.meta_info["validate"] is True
        self.rollout_batch_sizes.append(len(gen_batch))
        rewards = [0.0, 1.0, None, None, 1.0]
        response_lengths = [[2], [4, 6], [], [8], [1, 3, 5]]
        rollouts = [
            CompletedRollout(
                rollout_id=f"r{index}",
                data_id=str(sample_id),
                step=self.global_steps,
                sample_idx_in_step=index,
                enqueue_time=0.0,
                final_reward=rewards[sample_id],
                triplets=[
                    Triplet(prompt={"token_ids": [1]}, response={"token_ids": [3] * length})
                    for length in response_lengths[sample_id]
                ],
            )
            for index, sample_id in enumerate(gen_batch.non_tensor_batch["sample_id"])
        ]
        metrics = self.adapter.get_test_metrics(rollouts, global_steps=self.global_steps)
        metrics["training/async/proxy_inflight_at_pause"] = len(gen_batch)
        return DataProto(), metrics


@pytest.mark.parametrize("batch_sizes", [[5], [4, 1], [2, 3]])
@pytest.mark.parametrize("n", [1, 3])
@pytest.mark.parametrize("legacy_manager", [False, True], ids=["verl-0.8", "legacy-verl"])
def test_validation_metrics_cover_all_batches(batch_sizes: list[int], n: int, legacy_manager: bool) -> None:
    trainer = ValidationTrainer(batch_sizes, n=n, legacy_manager=legacy_manager)

    metrics = trainer._validate()

    # Rewards include 0.25 for each missing reward, including the rollout without a trace.
    assert metrics["val/reward"] == pytest.approx(2.5 / 5)
    assert metrics["val/n_rollouts"] == 5 * n
    assert metrics["val/n_rollouts_w_trace"] == 4 * n
    assert metrics["val/n_rollouts_w_reward"] == 3 * n
    # Trace means are over four rollouts, retaining the adapter's per-rollout semantics.
    assert metrics["val/mean_response_length_per_turn"] == pytest.approx((2 + 5 + 8 + 3) / 4)
    assert metrics["val/mean_total_response_length_per_rollout"] == pytest.approx((2 + 10 + 8 + 9) / 4)
    assert metrics["val/turn_count"] == pytest.approx((1 + 2 + 1 + 3) / 4)
    assert trainer.rollout_batch_sizes == [size * n for size in batch_sizes]
    assert metrics["training/async/proxy_inflight_at_pause"] == batch_sizes[-1] * n


def test_validation_metrics_empty_dataloader() -> None:
    trainer = ValidationTrainer([])

    assert trainer._validate() == {}
    assert trainer.rollout_batch_sizes == []
