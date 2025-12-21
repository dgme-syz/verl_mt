#!/usr/bin/env bash
set -euo pipefail
# 确保实验在 verl_syz 根目录下运行


# 1). 实验设置
## 消融验证实验 - 翻译任务中不使用 QE 指标，但是使用来自后编辑的反馈

# 2). 需调整参数

model_path="/data/models/Qwen3-4B"
ppo_mini_batch_size=32 # 大显存： 64
ppo_micro_batch_size_per_gpu=16 # 大显存：32
project_name="shenyz"
save_freq=50

# 3). 其余参数可以不用太调整，主要是模型权重会存在当前文件夹

train_batch_size=128
rollout_num=8
num_gpus=8
datetime=$(date +%Y%m%d_%H%M%S)
mul_times=1
model_tag="Qwen3-4B-en2fi-ablation_NoQE_high_data"
exp_name="shenyz_bs@${train_batch_size}_n@${rollout_num}_m@${mul_times}_@${datetime}_@${model_tag}_@${num_gpus}gpus"
dir=./data

# train_file_path="[$dir/train/train_mixed_en-zh_CN.parquet,$dir/train/Qwen3-4B_mixed_en_zh.parquet]"
# test_file_path="[$dir/test/wmt24_en-zh_CN.parquet,$dir/test/challenge_set_en-zh_CN.parquet,$dir/test/flores_en2zh.parquet]"

# fi
train_file_path=$dir/train/wmt_en2fi_7k.parquet
test_file_path="[$dir/test/wmt24_en-fi_FI.parquet,$dir/test/flores_en2fi.parquet]"
export RAY_raylet_start_wait_time_s=120 


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    ++algorithm.leaf_score_only=False \
    ++algorithm.grpo_child_score_merge_fn=mean \
    ++algorithm.qe_weight=0.0 \
    ++algorithm.merge_weight=1.0 \
    ++algorithm.remove_runtime_qe=False \
    data.train_files=$train_file_path \
    data.val_files=$test_file_path \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=768 \
    data.max_response_length=3072 \
    ++data.apply_chat_template_kwargs.enable_thinking=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.92 \
    actor_rollout_ref.rollout.n=${rollout_num} \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    ++reward_model.train_reward_manager="mt_train" \
    ++reward_model.val_reward_manager="mt_val" \
    ++ray_kwargs.ray_init.ignore_reinit_error=True \
    ++workflow.repeat_times=4 \
    ++workflow.data_divisor=1 \
    ++workflow.mt_only=False \
	++workflow.test_mt_only=False \
    ++workflow.tokenizer_path=$model_path \
    ++workflow.use_test_prompt=True \
    ++workflow.dynamic_mode=False \
    custom_reward_function.reward_kwargs.mul_times=${mul_times} \
    custom_reward_function.reward_kwargs.thinking_check=True \
    trainer.val_before_train=True \
    trainer.logger="[console,tensorboard]" \
    trainer.project_name=$project_name \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=$num_gpus \
    trainer.nnodes=1 \
    trainer.default_local_dir="${exp_name}" \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=$save_freq \
    trainer.test_freq=5 \
    trainer.total_epochs=15
