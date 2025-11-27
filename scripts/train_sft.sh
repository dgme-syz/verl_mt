#!/usr/bin/env bash
set -euo pipefail
num_gpus=2
train_batch_size_per_gpu=2
datetime=$(date +%Y%m%d_%H%M%S)
exp_name="bs@${train_batch_size_per_gpu}_@${datetime}"

train_file_dir=/home/nfs06/shenyz/data/recheck_sft
train_file_path="[$train_file_dir/dpsk-3.2-exp-qwen3-0.6b-wmt24-en2zh-sft.parquet]"

model_path="/home/nfs06/shenyz/models/Qwen3-0.6B"
export TMPDIR=/home/nfs05/shenyz/temp
export WANDB_API_KEY=428f855d211a1e71e0dc27c8675469476d8c22a3

check_free_gpus() {
    free_gpus=()
    gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -n 1)
    for ((gpu=0; gpu<gpu_count; gpu++)); do
        mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu)
        mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $gpu)
        util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $gpu)

        if [ "$mem_used" -lt 600 ] && [ "$util" -lt 6 ]; then
            free_gpus+=($gpu)
        fi
    done
    echo "${free_gpus[@]}"
}
export TRANSFORMERS_OFFLINE=1

while true; do
    free_gpus=($(check_free_gpus))
    if [ "${#free_gpus[@]}" -ge $num_gpus ]; then
        for ((i=0; i<num_gpus; i++)); do
            if [ $i -eq 0 ]; then
                CUDA_VISIBLE_DEVICES="${free_gpus[i]}"
            else
                CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES},${free_gpus[i]}"
            fi
        done
        export CUDA_VISIBLE_DEVICES
        echo $CUDA_VISIBLE_DEVICES

        echo "检测到至少 $num_gpus 张空闲 GPU: ${free_gpus[@]:0:num_gpus}, 启动训练..."
        torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
            -m verl.trainer.fsdp_sft_trainer \
            data.train_files=$train_file_path \
            data.val_files=$train_file_path \
            data.train_batch_size=128 \
            data.max_length=6144 \
            optim.lr=1e-5 \
            data.prompt_key=prompt \
            data.response_key=response \
            data.micro_batch_size_per_gpu=$train_batch_size_per_gpu \
            model.partial_pretrain=$model_path \
            model.fsdp_config.model_dtype=bfloat16 \
            model.fsdp_config.offload_params=True \
            model.enable_gradient_checkpointing=True \
            trainer.default_local_dir="${exp_name}" \
            trainer.project_name=mt-sft \
            trainer.test_freq=10 \
            trainer.save_freq=70 \
            trainer.experiment_name="${exp_name}" \
            trainer.total_epochs=20 \
            trainer.logger='["console", "wandb"]' $@
        break
    else
	    echo "空闲GPU编号: ${free_gpus[@]}"
        echo "空闲 GPU 不足 ${num_gpus} 张，10 秒后重试..."
        sleep 10
    fi
done
