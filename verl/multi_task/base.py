from abc import ABC
from typing import Any, Dict, List, Tuple
import re
import uuid
import copy
import textwrap

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from verl import DataProto
from verl.utils.fs import copy_to_local
from verl.utils import hf_tokenizer
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask


class MultiTaskWorkflow(ABC):
    """Abstract base class for defining multi-task workflows."""

    actor_rollout_wg: Any
    config: Any
    gen_batch_output: DataProto

    def __init__(self, config: Any):
        # gen_batch_output: generated batch from language model, bsz*n
        self.config = config

    def set(
        self,
        gen_batch_output: DataProto,
        actor_rollout_wg: Any,
    ):
        self.gen_batch_output = gen_batch_output
        self.actor_rollout_wg = actor_rollout_wg

    def work(self, **kwargs) -> DataProto:
        raise NotImplementedError


class MTWorkflow(MultiTaskWorkflow):
    """
    Implements a multi-turn machine translation workflow:
    1. Initial translation (MT).
    2. Post-editing/Refinement of the translation.
    """

    def __init__(self, config: Any):
        super().__init__(config)
        tokenizer_path = copy_to_local(self.config.workflow.get("tokenizer_path"))
        self.tokenizer: PreTrainedTokenizerBase = hf_tokenizer(tokenizer_path)
        self.repeat_times = self.config.workflow.get("repeat_times")
        self.max_prompt_length = self.config.data.get("max_prompt_length", 1024)
        self.truncation = self.config.data.get("truncation", "error")

    @staticmethod
    def _extract_translation(solution_str: str) -> str | None:
        """
        Extracts the final answer from the model's response string by removing <think> blocks.
        """
        # --- Remove all <think>...</think> blocks ---
        processed_str = re.sub(r"<think>.*?</think>", "", solution_str, flags=re.DOTALL).strip()
        return processed_str if "<think>" not in processed_str else None

    @staticmethod
    def _build_recheck_prompt(target_lang: str, src_text: str, pred_text: str) -> List[Dict[str, str]]:
        """Builds the prompt for the recheck/refinement step."""
        prompts = {
            "zh": textwrap.dedent(
                f"""
                给定源文：'{src_text}'，和它的翻译草稿：'{pred_text}'
                必须先理解源文，然后参考以下标准对草稿进行进一步修改润色

                1. 草稿翻译可能漏翻，请不要遗漏原文的含义
                2. 保证翻译文段读起来流畅，通顺，符合人类表达，可以调整句子的顺序
                3. 请注意仔细理解语境，选择书面语还是口语化表达
                4. 请再检查每个词翻译的意思，是否符合整个语境和现实社会
                5. 请再检查每个句翻译的意思，是否符合整个语境和现实社会
                6. 注意你的润色对象是翻译后的草稿，不是源文
                7. 当你觉得语句读起来困惑的时候，尝试从源文本重新思考
                8. 如果翻译草稿的语言并非中文，请确保你的润色文本为中文
                9. 注意检查，不要遗漏源文的含义，也不要添加补充，也不要尝试在翻译中使用过分的比喻
                10. 可以有思考过程，不要无限思考下去，最终回复中仅输出你润色后的内容

                请返回你最后的润色翻译文本，不要输出多余内容。
                """
            ).strip(),
            "en": textwrap.dedent(
                f"""
                Given the source text: '{src_text}' and its draft translation: '{pred_text}', please refine and polish the draft translation according to the following guidelines:

                1. The draft translation may have omissions — do not leave out any meaning from the source text.
                2. Ensure the translation reads smoothly and naturally, consistent with human expression; you may adjust sentence order if needed.
                3. Carefully understand the context, and decide whether to use formal or colloquial language.
                4. Recheck the meaning of each word to ensure it fits the overall context and real-world usage.
                5. Recheck the meaning of each sentence to ensure it fits the overall context and real-world usage.
                6. Note that your task is to polish the translated draft, not the source text.
                7. When a sentence feels confusing, reconsider it from the perspective of the source text.
                8. If the draft translation is not in English, make sure your refined version is in English.
                9. Do not include any extra reasoning or commentary — only output your polished translation.

                Please return only the final refined translation text, without any additional content.
                """
            ).strip(),
        }

        lang_key = next((key for key in prompts if key in target_lang), None)
        if lang_key is None:
            raise ValueError(f"Unsupported target language: {target_lang}. Supported: {list(prompts.keys())}")

        return [{"role": "user", "content": prompts[lang_key]}]

    def _prepare_recheck_inputs(
        self, gen_batch_output: DataProto
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[int]]:
        """
        Process the initial generation output to prepare inputs for the recheck step.
        This involves decoding, extracting answers, and re-tokenizing for the next turn.
        """
        input_ids_list, attention_mask_list, position_ids_list = [], [], []
        last_responses, valid_indices = [], []

        for i, data_item in enumerate(gen_batch_output):
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequences_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            answer_str = self._extract_translation(sequences_str)

            if answer_str and valid_response_length < self.max_prompt_length:
                valid_indices.append(i)
            else:
                answer_str = "null"
                print(f"Warning: sample {i: }, len={valid_response_length} {sequences_str} \n has no extracted translation, set to default 'null'")

            src_text = data_item.non_tensor_batch["extra_info"]["src"]
            tgt_lang = data_item.non_tensor_batch["extra_info"]["tgt_lang"]

            def recheck_tokenized(answer_str):

                recheck_chat = self._build_recheck_prompt(
                    target_lang=tgt_lang, src_text=src_text, pred_text=answer_str
                )
                recheck_text = self.tokenizer.apply_chat_template(recheck_chat, add_generation_prompt=True, tokenize=False)
                recheck_prompt_tokenized = self.tokenizer(recheck_text, return_tensors="pt", add_special_tokens=False)
                return recheck_prompt_tokenized, recheck_text

            recheck_prompt_tokenized, recheck_text = recheck_tokenized(answer_str)
            if i == 0:
                print(f"Preview:\nanswer_str:\n{answer_str}\nsource text:\n{src_text}\nReCheck prompt example:\n{recheck_text}")
            if recheck_prompt_tokenized['input_ids'].shape[-1] > self.max_prompt_length:
                print(f"Warning: sample {i: } recheck prompt length {recheck_prompt_tokenized['input_ids'].shape[-1]} exceeds max_prompt_length {self.max_prompt_length}")
                recheck_prompt_tokenized, _ = recheck_tokenized("null")
                answer_str = "null"
            
            last_responses.append(answer_str)
            x, y = verl_F.postprocess_data(
                input_ids=recheck_prompt_tokenized['input_ids'],
                attention_mask=recheck_prompt_tokenized['attention_mask'],
                max_length=self.max_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.truncation,
            )
            z = compute_position_id_with_mask(y)
            input_ids_list.append(x[0])
            attention_mask_list.append(y[0])
            position_ids_list.append(z[0])

        return (
            torch.stack(input_ids_list),
            torch.stack(attention_mask_list),
            torch.stack(position_ids_list),
            last_responses,
            valid_indices,
        )

    def _align_batch_for_dp(self, indices: List[int], repeat_times: int) -> List[int]:
        """Aligns the batch of indices to be divisible by the data-parallel world size."""
        dp_world_size = self.actor_rollout_wg.world_size
        if dp_world_size <= 0:
            return indices

        num_samples = len(indices) * repeat_times
        remainder = num_samples % dp_world_size
        if remainder == 0:
            return indices

        samples_to_drop = remainder // repeat_times + (1 if remainder % repeat_times != 0 else 0)
        print(f"Aligning batch for DP: dropping {samples_to_drop} samples to make batch size divisible by {dp_world_size}")
        return indices[:-samples_to_drop]

    def _build_post_edit_batch(
        self,
        original_batch: DataProto,
        recheck_input_ids: torch.Tensor,
        recheck_attn_mask: torch.Tensor,
        recheck_pos_ids: torch.Tensor,
        last_responses: List[str],
        uuid,
    ) -> DataProto:
        """Constructs the DataProto object for the post-editing generation step."""
        # Use shallow copy for efficiency, as we are only modifying top-level dicts
        post_edit_batch = copy.copy(original_batch)
        post_edit_batch.batch = copy.copy(original_batch.batch)
        post_edit_batch.non_tensor_batch = copy.copy(original_batch.non_tensor_batch)

        post_edit_batch.batch.pop("prompts")
        post_edit_batch.batch.pop("responses")
        post_edit_batch.batch.update({
            "input_ids": recheck_input_ids,
            "attention_mask": recheck_attn_mask,
            "position_ids": recheck_pos_ids,
        })

        batch_size = len(recheck_input_ids)
        post_edit_batch.non_tensor_batch.update({
            "last_response": np.array(last_responses, dtype=object),
            "uid": uuid,
            "depth": np.full(batch_size, 2, dtype=np.int32)
        })
        return post_edit_batch

    def work(self, repeat_times: int | None = None, concat: bool = True, test: bool = False) -> DataProto:
        """Executes the full multi-turn workflow."""
        repeat_times = repeat_times if repeat_times is not None else self.repeat_times

        # 1. Initial Generation (MT)
        gen_batch_output = self.actor_rollout_wg.generate_sequences(self.gen_batch_output)
        gen_batch_output.non_tensor_batch["own_uid"] = np.array([str(uuid.uuid4()) for _ in range(len(gen_batch_output.batch))], dtype=object)
        # 2. Prepare for Post-Editing Step
        (
            recheck_input_ids,
            recheck_attn_mask,
            recheck_pos_ids,
            last_responses,
            valid_indices,
        ) = self._prepare_recheck_inputs(gen_batch_output)

        if not test:
            valid_indices = self._align_batch_for_dp(valid_indices, repeat_times)
        else:
            valid_indices = list(range(len(recheck_input_ids))) # auto pad
        # Filter all inputs based on valid indices
        recheck_input_ids = recheck_input_ids[valid_indices]
        recheck_attn_mask = recheck_attn_mask[valid_indices]
        recheck_pos_ids = recheck_pos_ids[valid_indices]
        filtered_last_responses = [last_responses[i] for i in valid_indices]
        
        # Build the batch for the post-editing step
        post_edit_batch = self._build_post_edit_batch(
            original_batch=gen_batch_output[valid_indices],
            recheck_input_ids=recheck_input_ids,
            recheck_attn_mask=recheck_attn_mask,
            recheck_pos_ids=recheck_pos_ids,
            last_responses=filtered_last_responses,
            uuid=gen_batch_output.non_tensor_batch["own_uid"][valid_indices]
        )

        # Set metadata for the original generation batch
        gen_batch_output.non_tensor_batch["depth"] = np.ones(len(gen_batch_output.batch), dtype=np.int32)
        gen_batch_output.non_tensor_batch["last_response"] = np.full(len(gen_batch_output.batch), None, dtype=object)

        print(f"Post-edit batch size (before repeat): {len(post_edit_batch.batch)}")
        print(f"Preview of post-editing:\n{gen_batch_output[:2]}")
        
        # 3. Post-Editing Generation
        post_edit_batch = post_edit_batch.repeat(repeat_times=repeat_times, interleave=True)
        post_edit_output = self.actor_rollout_wg.generate_sequences(post_edit_batch)
        post_edit_output.non_tensor_batch["own_uid"] = np.full(len(post_edit_output.batch), "-1", dtype=object)

        # 4. Combine results
        if concat:
            gen_batch_output.meta_info.pop("timing", None)
            final_output = DataProto.concat([gen_batch_output, post_edit_output])
        else:
            final_output = post_edit_output

        print(f"Final combined batch size: {len(final_output.batch)}")
        return final_output



