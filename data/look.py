import argparse
import pyarrow.parquet as pq
import json
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Parquet 文件信息查看（JSON 输出）")
    parser.add_argument("file", help="Parquet 文件路径")
    parser.add_argument("-n", "--num-rows", type=int, default=5, help="显示前几行")
    parser.add_argument("-c", "--columns", nargs="+", help="只显示指定列")
    args = parser.parse_args()

    parquet_file = pq.ParquetFile(args.file)

    info = {
        "file": args.file,
        "num_rows": parquet_file.metadata.num_rows,
        "num_columns": parquet_file.metadata.num_columns,
        "columns": [],
        "preview": []
    }

    # 列信息
    for field in parquet_file.schema:
        dtype = getattr(field, "logical_type", None) or field.physical_type
        info["columns"].append({
            "name": field.name,
            "type": str(dtype)
        })

    # 读取前几行
    row_group = parquet_file.read_row_group(0, columns=args.columns)
    df = row_group.to_pandas()

    # 转换为 JSON 可序列化对象
    info["preview"] = df.head(args.num_rows).replace({pd.NA: None}).to_dict(orient="records")

    # 自定义 JSON 序列化函数
    def default_serializer(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif obj is pd.NA:
            return None
        else:
            return str(obj)

    print(json.dumps(info, indent=4, ensure_ascii=False, default=default_serializer))

if __name__ == "__main__":
    main()

