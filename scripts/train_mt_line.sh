#!/usr/bin/env bash
set -euo pipefail

train_batch_size=128
rollout_num=10
num_gpus=4
datetime=$(date +%Y%m%d_%H%M%S)
mul_times=1
model_tag="Qwen3-4B-mix-plus-added-data_17-20-line"
# model_path="/home/nfs05/shenyz/translation/verl/bs@2_@20251118_211452/global_step_140/huggingface" # sft
model_path="/data/models/Qwen3-4B"
exp_name="bs@${train_batch_size}_n@${rollout_num}_m@${mul_times}_@${datetime}_@${model_tag}_@${num_gpus}gpus"

train_file_dir=/data/dataset/recheck
#train_file_path="[$train_file_dir/Qwen3-0.6B_wmt24_en_zh.parquet,$train_file_dir/Qwen2.5-7B-Instruct_wmt24_en_zh.parquet,$train_file_dir/Qwen3-4B_wmt24_en_zh.parquet,$train_file_dir/Qwen3-0.6B_mixed_en_zh.parquet,$train_file_dir/Qwen2.5-7B-Instruct_mixed_en_zh.parquet,$train_file_dir/Qwen3-4B_mixed_en_zh.parquet]"
#train_file_path="[$train_file_dir/Qwen3-0.6B_wmt24_en_zh.parquet,$train_file_dir/Qwen2.5-7B-Instruct_wmt24_en_zh.parquet,$train_file_dir/Qwen3-4B_wmt24_en_zh.parquet,$train_file_dir/wmt24_en-zh_CN.parquet]"
# train_file_path="[$train_file_dir/Qwen3-0.6B_mixed_en_zh.parquet,$train_file_dir/Qwen2.5-7B-Instruct_mixed_en_zh.parquet,$train_file_dir/Qwen3-4B_mixed_en_zh.parquet,$train_file_dir/train_mixed_en-zh_CN.parquet]"
train_file_path=$train_file_dir/train_mixed_en-zh_CN.parquet
test_file_path="[$train_file_dir/wmt24_en-zh_CN.parquet,$train_file_dir/challenge_set_en-zh_CN.parquet,$train_file_dir/flores_en2zh.parquet]"

export TMPDIR=/home/nfs05/shenyz/temp
export WANDB_API_KEY=428f855d211a1e71e0dc27c8675469476d8c22a3
check_free_gpus() {
    free_gpus=()
    gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -n 1)
    for ((gpu=0; gpu<gpu_count; gpu++)); do
        mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu)
        mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $gpu)
        util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $gpu)

        if [ "$mem_used" -lt 400 ] && [ "$util" -lt 6 ]; then
            free_gpus+=($gpu)
        fi
    done
    echo "${free_gpus[@]}"
}
export TRANSFORMERS_OFFLINE=1
export RAY_raylet_start_wait_time_s=120 
# export RAY_memory_monitor_refresh_ms=0
while true; do
    free_gpus=($(check_free_gpus))
    if [ "${#free_gpus[@]}" -ge $num_gpus ]; then
        cuda_list=""
        for ((i=0; i<num_gpus; i++)); do
            if [ $i -eq 0 ]; then
                cuda_list="${free_gpus[$i]}"
            else
                cuda_list="${cuda_list},${free_gpus[$i]}"
            fi
        done
        export CUDA_VISIBLE_DEVICES="$cuda_list"
        echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

        echo "检测到至少 $num_gpus 张空闲 GPU: ${free_gpus[@]:0:$num_gpus}, 启动训练..."
        python3 -m verl.trainer.main_ppo \
            algorithm.adv_estimator=grpo \
            ++algorithm.leaf_score_only=True \
            ++algorithm.grpo_child_score_merge_fn=max \
            data.train_files=$train_file_path \
            data.val_files=$test_file_path \
            data.train_batch_size=$train_batch_size \
            data.max_prompt_length=3072 \
            data.max_response_length=4096 \
            ++data.apply_chat_template_kwargs.enable_thinking=True \
            actor_rollout_ref.model.path=$model_path \
            actor_rollout_ref.actor.optim.lr=5e-7 \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.actor.ppo_mini_batch_size=32 \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
            actor_rollout_ref.actor.use_kl_loss=False \
            actor_rollout_ref.actor.kl_loss_coef=0.01 \
            actor_rollout_ref.actor.entropy_coeff=0.0 \
            actor_rollout_ref.actor.kl_loss_type=low_var_kl \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.fsdp_config.param_offload=True \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
            actor_rollout_ref.rollout.n=${rollout_num} \
            actor_rollout_ref.rollout.val_kwargs.top_k=20 \
            actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
            actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
            actor_rollout_ref.rollout.val_kwargs.do_sample=True \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
            actor_rollout_ref.ref.fsdp_config.param_offload=True \
            ++reward_model.train_reward_manager="mt_train" \
            ++reward_model.val_reward_manager="mt_val" \
            ++ray_kwargs.ray_init.ignore_reinit_error=True \
            ++workflow.repeat_times=1 \
            ++workflow.tokenizer_path=$model_path \
            custom_reward_function.reward_kwargs.mul_times=${mul_times} \
            trainer.val_before_train=True \
            trainer.logger="[console,wandb]" \
            trainer.project_name="ReCheck" \
            trainer.experiment_name="${exp_name}" \
            trainer.n_gpus_per_node=$num_gpus \
            trainer.nnodes=1 \
            trainer.default_local_dir="${exp_name}" \
            trainer.default_hdfs_dir=null \
            trainer.save_freq=100 \
            trainer.test_freq=5 \
            trainer.total_epochs=15
        break
    else
	echo "空闲GPU编号: ${free_gpus[@]}"
        echo "空闲 GPU 不足 $num_gpus 张，10 秒后重试..."
        sleep 10
    fi
done
