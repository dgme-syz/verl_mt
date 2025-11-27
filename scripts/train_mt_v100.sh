#!/usr/bin/env bash
set -euo pipefail

train_batch_size=32
rollout_num=4
datetime=$(date +%Y%m%d_%H%M%S)
exp_name="bs@${train_batch_size}_n@${rollout_num}_@${datetime}"

train_file_dir=/home/nfs06/shenyz/data/recheck
train_file_path="[$train_file_dir/Qwen3-0.6B_wmt24_en_zh.parquet,$train_file_dir/Qwen2.5-7B-Instruct_wmt24_en_zh.parquet,$train_file_dir/Qwen3-4B_wmt24_en_zh.parquet,$train_file_dir/Qwen3-8B_wmt24_en_zh.parquet]"
model_path="/home/nfs06/shenyz/models/Qwen3-0.6B"
export TMPDIR=/home/nfs05/shenyz/temp

# 检查空闲 GPU（自动检测系统里所有 GPU）
check_free_gpus() {
    free_gpus=()
    gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -n 1)
    for ((gpu=0; gpu<gpu_count; gpu++)); do
        mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu)
        mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $gpu)
        util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $gpu)

        # 判断条件：显存使用 < 500MB 且 GPU 利用率 < 10%
        if [ "$mem_used" -lt 500 ] && [ "$util" -lt 10 ]; then
            free_gpus+=($gpu)
        fi
    done
    echo "${free_gpus[@]}"
}
export RAY_TMPDIR=/home/nfs05/shenyz/ray_tmp
export TMPDIR=/home/nfs05/shenyz/temp
export TRANSFORMERS_OFFLINE=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

while true; do
    free_gpus=($(check_free_gpus))
    if [ "${#free_gpus[@]}" -ge 4 ]; then
        export CUDA_VISIBLE_DEVICES="${free_gpus[0]},${free_gpus[1]},${free_gpus[2]},${free_gpus[3]}"
        echo $CUDA_VISIBLE_DEVICES

        echo "检测到至少 4 张空闲 GPU: ${free_gpus[@]:0:4}, 启动训练..."
        TMPDIR=/home/nfs05/shenyz/temp python3 -m verl.trainer.main_ppo \
            algorithm.adv_estimator=grpo \
            data.train_files=$train_file_path \
            data.val_files=$train_file_path \
            data.train_batch_size=$train_batch_size \
            data.max_prompt_length=1024 \
            data.max_response_length=512 \
            actor_rollout_ref.model.path=$model_path \
	        actor_rollout_ref.rollout.dtype=float16 \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.actor.ppo_mini_batch_size=4 \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
            actor_rollout_ref.actor.use_kl_loss=False \
            actor_rollout_ref.actor.kl_loss_coef=0.01 \
            actor_rollout_ref.actor.entropy_coeff=0.0 \
            actor_rollout_ref.actor.kl_loss_type=low_var_kl \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.fsdp_config.param_offload=True \
            actor_rollout_ref.actor.fsdp_config.model_dtype=float16 \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
            actor_rollout_ref.rollout.n=${rollout_num} \
            actor_rollout_ref.rollout.enable_chunked_prefill=false \
            actor_rollout_ref.rollout.enable_prefix_caching=false \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
            actor_rollout_ref.ref.fsdp_config.param_offload=True \
            reward_model.reward_manager="mt_train" \
            ray_kwargs.ray_init.num_cpus=8 \
            trainer.val_before_train=False \
            trainer.logger="[console]" \
            trainer.project_name="ReCheck" \
            trainer.experiment_name="${exp_name}" \
            trainer.n_gpus_per_node=4 \
            trainer.nnodes=1 \
            trainer.default_local_dir="${exp_name}" \
            trainer.default_hdfs_dir=null \
            trainer.save_freq=100 \
            trainer.test_freq=0 \
            trainer.total_epochs=5
        break
    else
        echo "空闲 GPU 不足 2 张，10 秒后重试..."
        sleep 10
    fi
done
