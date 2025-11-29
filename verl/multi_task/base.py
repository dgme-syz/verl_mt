from abc import ABC
from typing import Any
import re
import uuid
import pickle
import copy

import numpy as np
import torch

from verl import DataProto
from verl.utils.fs import copy_to_local
from verl.utils import hf_tokenizer
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask


class MutiTaskWorkflow(ABC):
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
    
    def work(self) -> DataProto:
        raise NotImplementedError
    

class MTWorkflow(MutiTaskWorkflow):
    def __init__(self, config):
        super().__init__(config)
        tokenizer_path = copy_to_local(self.config.workflow.get('tokenizer_path'))
        self.tokenizer = hf_tokenizer(tokenizer_path)
        self.repeat_times = self.config.workflow.get('repeat_times')

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
        return processed_str if "<think>" not in processed_str else None
    
    def recheck_prompt(self, target_lang: str, src_text: str, pred_text: str):
        if "zh" in target_lang:
            user_input = (
                f"给定源文：'{src_text}'，和它的翻译草稿：'{pred_text}'"
                f"必须先理解源文，然后参考以下标准对草稿进行进一步修改润色\n\n"
                f"1. 草稿翻译可能漏翻，请不要遗漏原文的含义\n\n"
                f"2. 保证翻译文段读起来流畅，通顺，符合人类表达，可以调整句子的顺序\n\n"
                f"3. 请注意仔细理解语境，选择书面语还是口语化表达\n\n"
                f"4. 请再检查每个词翻译的意思，是否符合整个语境和现实社会\n\n"
                f"5. 请再检查每个句翻译的意思，是否符合整个语境和现实社会\n\n"
                f"6. 注意你的润色对象是翻译后的草稿，不是源文\n\n"
                f"7. 当你觉得语句读起来困惑的时候，尝试从源文本重新思考\n\n"
                f"8. 如果翻译草稿的语言并非中文，请确保你的润色文本为中文\n\n"
                f"9. 注意检查，不要遗漏源文的含义，也不要添加补充，也不要尝试在翻译中使用过分的比喻\n\n"
            )
            user_input += (
                f"10. 可以有思考过程，不要无限思考下去，最终回复中仅输出你润色后的内容\n\n"
                f"请返回你最后的润色翻译文本，不要输出多余内容。"
            )
        elif "en" in target_lang:
            # dev
            user_input = (
                f"Given the source text: '{src_text}' and its draft translation: '{pred_text}', "
                f"please refine and polish the draft translation according to the following guidelines:\n\n"
                f"1. The draft translation may have omissions — do not leave out any meaning from the source text.\n\n"
                f"2. Ensure the translation reads smoothly and naturally, consistent with human expression; you may adjust sentence order if needed.\n\n"
                f"3. Carefully understand the context, and decide whether to use formal or colloquial language.\n\n"
                f"4. Recheck the meaning of each word to ensure it fits the overall context and real-world usage.\n\n"
                f"5. Recheck the meaning of each sentence to ensure it fits the overall context and real-world usage.\n\n"
                f"6. Note that your task is to polish the translated draft, not the source text.\n\n"
                f"7. When a sentence feels confusing, reconsider it from the perspective of the source text.\n\n"
                f"8. If the draft translation is not in English, make sure your refined version is in English.\n\n"
                f"9. Do not include any extra reasoning or commentary — only output your polished translation.\n\n"
                f"Please return only the final refined translation text, without any additional content."
            )
        else:
            raise ValueError(
                f"Now we just support lang=['Chinese', 'English'], we get {target_lang}"
            )

        return [{"role": "user", "content": user_input}]


    def work(self, repeat_times: int | None = None, concat: bool = True) -> DataProto:
        if repeat_times is None:
            repeat_times = self.repeat_times

        # 1. mt
        gen_batch_output = self.actor_rollout_wg.generate_sequences(self.gen_batch_output)
        # 2. post-edit
        ## 2.1 tokenize and prepare prompts_str
        rec_texts = []
        lst_texts = []
        normal_idx = []
        input_ids = []
        attention_mask = []
        position_ids = []
        for i in range(len(gen_batch_output)):
            data_item = gen_batch_output[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # 2.1 decode
            sequences = valid_response_ids
            sequences_str = self.tokenizer.decode(sequences, skip_special_tokens=True)
            answer_str = self.extract_translation(sequences_str)
            
            if answer_str:
                normal_idx.append(i)
            else:
                answer_str = "."
                print(f"Warning: sample {i} has no extracted translation, set to default '.'")

            lst_texts.append(answer_str)

            src_text = data_item.non_tensor_batch['extra_info']['src']
            tgt_lang = data_item.non_tensor_batch['extra_info']['tgt_lang']
            rec_text = self.recheck_prompt(target_lang=tgt_lang, src_text=src_text, pred_text=answer_str)
            rec_text = self.tokenizer.apply_chat_template(
                rec_text, add_generation_prompt=True, tokenize=False
            )
            if i == 0:
                print(
                    f"Preview:"
                    f"answer_str:\n{answer_str}\n"
                    f"source text:\n{src_text}\n"
                    f"ReCheck prompt example:\n{rec_text}"  
                )
            rec_prompt = self.tokenizer(rec_text, return_tensors="pt", add_special_tokens=False)
            
            x, y = verl_F.postprocess_data(
                input_ids=rec_prompt['input_ids'],
                attention_mask=rec_prompt['attention_mask'],
                max_length=self.config.data.get("max_prompt_length", 1024),
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation=self.config.data.get("truncation", "error")
            )
            z = compute_position_id_with_mask(y)
            input_ids.append(x[0])
            attention_mask.append(y[0])
            position_ids.append(z[0])
            
        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        position_ids = torch.stack(position_ids, dim=0)

        lst_texts = np.array(lst_texts, dtype=object)
        ## 2.2 batch_decode
        print(
            f"ReCheck input_ids shape: {input_ids.shape}\n"
            f"Preview input_ids:\n{input_ids[0][:min(10, input_ids.shape[1])]}\n"
            f"pad_token_id: {self.tokenizer.pad_token_id}\n"
            f"Preview attention_mask:\n{attention_mask[0][:min(10, input_ids.shape[1])]}"
        )
        
        # post_edit_batch_output = copy.deepcopy(gen_batch_output) # deepcopy
        post_edit_batch_output = copy.deepcopy(gen_batch_output)
        post_edit_batch_output.batch.pop('prompts')
        post_edit_batch_output.batch.pop('responses')
        post_edit_batch_output.batch.update({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        })

        print(
            f"Before ReCheck, total batch size: {len(gen_batch_output.batch)}\n"
            f"Preview:\n\n{gen_batch_output[[0, 1]]}\n\n"
            f"Normal indices for ReCheck: {normal_idx}"
        )

        post_edit_batch_output.non_tensor_batch.update({"last_response": lst_texts})
        gen_batch_output.non_tensor_batch["last_response"] = np.array(
            [None for _ in range(len(gen_batch_output.batch))], dtype=object
        )
        post_edit_batch_output.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(post_edit_batch_output.batch))], dtype=object
        )
        post_edit_batch_output.non_tensor_batch["depth"] = np.array(
            [2 for _ in range(len(post_edit_batch_output.batch))], dtype=np.int32
        )
        ### know who am I
        gen_batch_output.non_tensor_batch["depth"] = np.array(
            [1 for _ in range(len(gen_batch_output.batch))], dtype=np.int32
        )
        gen_batch_output.non_tensor_batch["own_uid"] = post_edit_batch_output.non_tensor_batch["uid"]
        # just use normal batch
        post_edit_batch_output = post_edit_batch_output[normal_idx]

        print(
            f"After filtering, total batch size: {len(post_edit_batch_output.batch)}\n"
            f"Preview:\n\n{post_edit_batch_output[[0, 1]]}"
        )

        ## 2.3 repeat for post-edit generation
        post_edit_batch_output
        post_edit_batch_output = post_edit_batch_output.repeat(
            repeat_times=repeat_times, interleave=True
        )
        post_edit_batch_output = self.actor_rollout_wg.generate_sequences(post_edit_batch_output)

        ### leaves do not need own_uid, set all -1
        post_edit_batch_output.non_tensor_batch["own_uid"] = np.array(
            ["-1" for _ in range(len(post_edit_batch_output.batch))], dtype=object
        )
        if concat:
            gen_batch_output.meta_info.pop("timing")
            gen_batch_output = DataProto.concat([gen_batch_output, post_edit_batch_output])
        else:
            gen_batch_output = post_edit_batch_output
        print(
            f"After ReCheck, total batch size: {len(gen_batch_output.batch)}"
            f"Preview:\n\n{gen_batch_output[0]}"      
        )
        return gen_batch_output


            
            