# Copyright 2024 Bytedance Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.utils.reward_score.mt_score import compute_score_val_bleu

@register("mt_train")
class MtTrainRewardManager(AbstractRewardManager):
    """Reward manager for machine translation (training stage)."""

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
    ) -> None:
        """
        Initialize the MtTrainRewardManager.

        Args:
            tokenizer: Tokenizer used to decode token IDs into text.
            num_examine: Number of batches of decoded responses to print for debugging.
            compute_score: Function to compute reward scores.
            reward_fn_key: Key to access data source in non-tensor batch data.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """Compute MT rewards for each sample in the batch."""
        if "rm_scores" in data.batch:
            raise NotImplementedError(
                "MT reward model score combination not supported; use BLEU or COMET scores instead."
            )

        reward_tensor = torch.zeros(
            data.batch["responses"].shape,
            dtype=torch.float32,
            device=data.batch["responses"].device,
        )
        reward_extra_info = defaultdict(list)
        printed_data_sources: dict[str, int] = {}

        for i, data_item in enumerate(data):
            # --- Decode tokens ---
            prompt_ids = data_item.batch["prompts"]
            response_ids = data_item.batch["responses"]
            attn_mask = data_item.batch["attention_mask"]

            prompt_len = prompt_ids.shape[-1]
            valid_prompt_len = int(attn_mask[:prompt_len].sum())
            valid_response_len = int(attn_mask[prompt_len:].sum())

            prompt_str = self.tokenizer.decode(
                prompt_ids[-valid_prompt_len:], skip_special_tokens=True
            )
            response_str = self.tokenizer.decode(
                response_ids[:valid_response_len], skip_special_tokens=True
            )

            non_tensor = data_item.non_tensor_batch
            ground_truth = non_tensor["reward_model"]["ground_truth"]
            data_source = non_tensor[self.reward_fn_key]
            translation_raw = non_tensor["last_response"] if non_tensor.get("last_response") else None
            lg_pair = f"{non_tensor['extra_info']['src_lang']}-{non_tensor['extra_info']['tgt_lang']}"

            metric_scores = [
                float(v) for k, v in data_item.batch.items() if k.endswith("_comet_score")
            ]

            # --- Compute reward ---
            score = self.compute_score(
                metric_scores=metric_scores,
                lang_pair=lg_pair,
                prompt_str=prompt_str,
                solution_str=response_str,
                ground_truth=ground_truth,
                translation_raw=translation_raw
            )

            if isinstance(score, dict):
                reward = score.get("score", 0.0)
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_len - 1] = reward

            printed_data_sources.setdefault(data_source, 0)
            if printed_data_sources[data_source] < self.num_examine:
                printed_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        return (
            {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            if return_dict
            else reward_tensor
        )


@register("mt_val")
class MtValRewardManager(AbstractRewardManager):
    """Reward manager for machine translation (validation stage)."""

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
    ) -> None:
        """
        Initialize the MtValRewardManager.

        During validation, BLEU or similar metric is computed directly.

        Args:
            tokenizer: Tokenizer used to decode token IDs into text.
            num_examine: Number of batches of decoded responses to print for debugging.
            compute_score: Function to compute validation score (e.g., BLEU).
            reward_fn_key: Key to access the data source in the non-tensor batch.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        # we do not need ant params 
        # if rewardmanager can use settings from config, it is easy
        # now we use custom func args to control score caclulation's params in training
        self.compute_score = compute_score_val_bleu 
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """Compute validation BLEU or other metric-based reward."""
        if "rm_scores" in data.batch:
            raise NotImplementedError(
                "MT validation with reward model score not supported; use BLEU or COMET."
            )

        reward_tensor = torch.zeros(
            data.batch["responses"].shape,
            dtype=torch.float32,
            device=data.batch["responses"].device,
        )
        reward_extra_info = defaultdict(list)
        printed_data_sources: dict[str, int] = {}

        for i, data_item in enumerate(data):
            prompt_ids = data_item.batch["prompts"]
            response_ids = data_item.batch["responses"]
            attn_mask = data_item.batch["attention_mask"]

            prompt_len = prompt_ids.shape[-1]
            valid_prompt_len = int(attn_mask[:prompt_len].sum())
            valid_response_len = int(attn_mask[prompt_len:].sum())

            prompt_str = self.tokenizer.decode(
                prompt_ids[-valid_prompt_len:], skip_special_tokens=True
            )
            response_str = self.tokenizer.decode(
                response_ids[:valid_response_len], skip_special_tokens=True
            )

            non_tensor = data_item.non_tensor_batch
            ground_truth = non_tensor["reward_model"]["ground_truth"]
            data_source = non_tensor[self.reward_fn_key]
            lg_pair = f"{non_tensor['extra_info']['src_lang']}-{non_tensor['extra_info']['tgt_lang']}"

            score = self.compute_score(
                solution_str=response_str,
                ground_truth=ground_truth,
                lang_pair=lg_pair,
            )

            if not isinstance(score, dict):
                score = {"score": score}

            # Attach any validation metrics
            for key, value in data_item.batch.items():
                if key.endswith("_valid"):
                    score[key] = float(value)

            reward = score.get("score")
            for key, value in score.items():
                reward_extra_info[key].append(value)

            reward_tensor[i, valid_response_len - 1] = reward

            printed_data_sources.setdefault(data_source, 0)
            if printed_data_sources[data_source] < self.num_examine:
                printed_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                for key, value in score.items():
                    print(f"[{key}]", value)

        return (
            {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            if return_dict
            else reward_tensor
        )
