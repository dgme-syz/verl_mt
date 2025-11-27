# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement a multiprocess PPOCritic
"""

import torch
import re

from verl import DataProto
from verl.workers.comet import BaseCOMETModel


__all__ = ['DataParallelCOMET']


class DataParallelCOMET(BaseCOMETModel):

    def __init__(self, config, comet_module, tokenizer):
        super().__init__(config=config)
        self.comet_model = comet_module 
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        self.tokenizer = tokenizer

        
        
    def _forward_micro_batch(self, micro_batch):
        # response_length = micro_batch['responses'].size(-1)

        batch_size = len(micro_batch)
        print(f"dp_comet.py forward micro_batch: {batch_size}")
        comet_output = self.comet_model.predict(micro_batch, batch_size=batch_size, gpus=1)
        scaled_scores = [round(float(score) * 100, 2) for score in comet_output.scores]
        
        # return comet_output.scores #for example: [0.84, 0.77, ...]
        return scaled_scores

    def extract_translation(self, solution_str: str):
        """
        Extracts the final answer from the model's response string.
        
        Args:
            solution_str: Raw response string from the language model
            
        Returns:
            Tuple containing (extracted_answer, processed_string)
        """
        processed_str = solution_str
        # --- Remove all <think>...</think> blocks ---
        processed_str = re.sub(r"<think>.*?</think>", "", processed_str, flags=re.DOTALL).strip()
        return processed_str

    def compute_comet_rm(self, data: DataProto) -> torch.Tensor:

        reward_tensor = torch.zeros((len(data.batch['responses']), 1), dtype=torch.float32)
        triplet_list = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = valid_response_ids
            sequences_str = self.tokenizer.decode(sequences, skip_special_tokens=True)
            answer_text = self.extract_translation(sequences_str)

            src_text = data_item.non_tensor_batch['extra_info']['src']
            tgt_text = data_item.non_tensor_batch['reward_model']['ground_truth']

            if answer_text:
                new_item = {"src":src_text, "mt":answer_text, "ref":tgt_text}
            else:
                new_item = {"src":src_text, "mt":"None", "ref":tgt_text}
            triplet_list.append(new_item)

        micro_batch_size = data.meta_info['micro_batch_size']
        micro_batches = [triplet_list[i:i + micro_batch_size] for i in range(0, len(triplet_list), micro_batch_size)]
        # print(f"dp_comet.py compute micro_batches: {micro_batches}")
        values_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                scores = self._forward_micro_batch(micro_batch)
            values_lst.extend(scores)

        for i in range(len(data)):
            reward_tensor[i] = values_lst[i]

        return reward_tensor

    def compute_valid_comet(self, data: DataProto) -> torch.Tensor:

        reward_tensor = torch.zeros((len(data.batch['responses']), 1), dtype=torch.float32)
        triplet_list = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]


            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = valid_response_ids
            sequences_str = self.tokenizer.decode(sequences, skip_special_tokens=True)
            answer_text, processed_str = self.extract_translation(sequences_str)

            src_text = data_item.non_tensor_batch['extra_info']['src']
            tgt_text = data_item.non_tensor_batch['reward_model']['ground_truth']

            if answer_text:
                new_item = {"src":src_text, "mt":answer_text, "ref":tgt_text}
            else:
                new_item = {"src":src_text, "mt":processed_str, "ref":tgt_text}
            triplet_list.append(new_item)

        # micro_batches = [triplet_list[i:i + micro_batch_size] for i in range(0, len(triplet_list), micro_batch_size)]
        micro_batches = [triplet_list[i:i + self.config.val_batch_size] for i in range(0, len(triplet_list), self.config.val_batch_size)]

        values_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                scores = self._forward_micro_batch(micro_batch)
            values_lst.extend(scores)

        for i in range(len(data)):
            reward_tensor[i] = values_lst[i]

        return reward_tensor