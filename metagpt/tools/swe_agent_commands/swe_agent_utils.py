from pathlib import Path

import numpy as np
from datasets import load_dataset, load_from_disk


def extract_patch(command_output):
    """从命令输出中提取补丁（patch）内容。

    参数:
        command_output (str): 命令的输出内容，包含差异（diff）信息。

    返回:
        str: 提取的补丁内容。
    """
    patch_lines = []
    recording = False
    # 遍历命令输出的每一行
    for line in command_output.split("\n"):
        # 如果行以 "diff --git" 开头，表示补丁的开始
        if line.startswith("diff --git"):
            recording = True
        if recording:
            # 记录补丁的每一行
            patch_lines.append(line)
    return "\n".join(patch_lines)


def load_hf_dataset(dataset_name_or_path: str, cache_dir, split: str = "test", existing_ids: list = []):
    """加载 Hugging Face 数据集，并按需过滤已有的实例 ID。

    参数:
        dataset_name_or_path (str): 数据集的名称或路径。
        cache_dir (Path): 缓存目录路径，用于存储加载的数据集。
        split (str): 数据集的分割，默认为 "test"。可以是 "train"、"validation" 或 "test"。
        existing_ids (list): 已存在的实例 ID 列表，用于过滤掉已存在的数据。

    返回:
        dataset: 返回处理后的数据集。
    """
    data_dir = cache_dir / dataset_name_or_path  # 生成数据集缓存路径
    if Path(data_dir).exists():
        # 如果缓存目录存在，则从磁盘加载数据集
        dataset = load_from_disk(data_dir)
    else:
        # 如果缓存目录不存在，则从 Hugging Face 加载数据集，并保存到磁盘
        dataset = load_dataset(dataset_name_or_path)
        dataset.save_to_disk(data_dir)

    print(dataset)  # 打印数据集信息
    # 如果指定的分割不存在于数据集中，抛出异常
    if split not in dataset:
        raise ValueError(f"Invalid split {split} for dataset {dataset_name_or_path}")

    dataset = dataset[split]  # 获取指定的分割
    np.array(list(map(len, dataset["instance_id"])))  # 获取每个实例 ID 的长度

    # 如果提供了已存在的实例 ID 列表，则过滤掉这些实例
    if existing_ids:
        dataset = dataset.filter(
            lambda x: x["instance_id"] not in existing_ids,  # 过滤已存在的 ID
            desc="Filtering out existing ids",  # 过滤描述信息
            load_from_cache_file=False,  # 禁用缓存文件加载
        )

    return dataset
