import os
import json
import argparse
from datasets import load_dataset

# 旧文本和新文本
OLD_TEXT = (
    "10. 不要输出多余思考内容，仅输出你润色后的内容\n\n"
    "请返回你最后的润色翻译文本，不要输出多余内容。"
)
NEW_TEXT = (
    "10. 可以有思考过程，不要无限思考下去，最终回复中仅输出你润色后的内容\n\n"
    "请返回你最后的润色翻译文本，不要输出多余内容。"
)


def modify_prompt(prompts):
    """遍历 prompts 列表，替换 content 中的旧文本"""
    if isinstance(prompts, list):
        for p in prompts:
            if isinstance(p, dict) and "content" in p and isinstance(p["content"], str):
                if OLD_TEXT in p["content"]:
                    p["content"] = p["content"].replace(OLD_TEXT, NEW_TEXT)
    return prompts


def process_parquet_file(file_path, output_dir):
    # 用 datasets 读取 parquet，同时指定独立缓存目录，避免锁阻塞
    ds = load_dataset("parquet", data_files=file_path, split="train", cache_dir="/tmp/datasets_cache")

    # 修改 prompt 列
    if "prompt" in ds.column_names:
        ds = ds.map(lambda x: {"prompt": modify_prompt(x["prompt"])}, batched=False, num_proc=1)

    # 保存到输出目录
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(file_path))
    ds.to_parquet(output_path)
    print(f"Processed {file_path} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Parquet 文件 prompt 内容批量替换 (datasets)")
    parser.add_argument("parquet_dir", help="Parquet 文件所在目录")
    parser.add_argument("output_dir", help="输出目录")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for fname in os.listdir(args.parquet_dir):
        if fname.endswith(".parquet"):
            file_path = os.path.join(args.parquet_dir, fname)
            process_parquet_file(file_path, args.output_dir)


if __name__ == "__main__":
    main()

