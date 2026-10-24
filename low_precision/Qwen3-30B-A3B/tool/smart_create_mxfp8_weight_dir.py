#!/usr/bin/env python3
"""
智能目录复制工具
- 小文件（<=10M）：直接复制
- 大文件（>10M）：创建软链接
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def get_file_size(file_path):
    """获取文件大小（字节）"""
    return os.path.getsize(file_path)


def format_size(size_bytes):
    """格式化文件大小显示"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def smart_copy_directory(src_dir, dst_dir, size_threshold=10 * 1024 * 1024, verbose=True):
    """
    智能复制目录

    Args:
        src_dir: 源目录路径
        dst_dir: 目标目录路径
        size_threshold: 大小阈值（字节），默认10MB
        verbose: 是否显示详细信息
    """
    src_path = Path(src_dir).resolve()
    dst_path = Path(dst_dir).resolve()

    if not src_path.exists():
        print(f"错误：源目录不存在: {src_path}")
        return False

    if not src_path.is_dir():
        print(f"错误：源路径不是目录: {src_path}")
        return False

    # 统计信息
    stats = {"copied": 0, "linked": 0, "dirs": 0, "copied_size": 0, "linked_size": 0}

    print(f"源目录: {src_path}")
    print(f"目标目录: {dst_path}")
    print(f"大小阈值: {format_size(size_threshold)}")
    print("-" * 60)

    # 遍历源目录
    for root, dirs, files in os.walk(src_path):
        root_path = Path(root)
        relative_path = root_path.relative_to(src_path)
        target_dir = dst_path / relative_path

        # 创建目标子目录
        if not target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            stats["dirs"] += 1
            if verbose:
                print(f"[目录] 创建: {relative_path}")

        # 处理文件
        for file in files:
            src_file = root_path / file
            dst_file = target_dir / file
            file_size = get_file_size(src_file)
            file_rel_path = src_file.relative_to(src_path)

            # 如果目标文件已存在，先删除
            if dst_file.exists() or dst_file.is_symlink():
                dst_file.unlink()

            if file_size > size_threshold:
                # 大文件：创建软链接
                try:
                    # 使用相对路径创建软链接，更便于移植
                    # dst_file.symlink_to(src_file)  # 绝对路径链接
                    os.symlink(str(src_file), str(dst_file))
                    stats["linked"] += 1
                    stats["linked_size"] += file_size
                    if verbose:
                        print(f"[链接] {file_rel_path} ({format_size(file_size)})")
                except OSError as e:
                    print(f"[错误] 无法创建链接 {file_rel_path}: {e}")
            else:
                # 小文件：直接复制
                try:
                    shutil.copy2(src_file, dst_file)
                    stats["copied"] += 1
                    stats["copied_size"] += file_size
                    if verbose:
                        print(f"[复制] {file_rel_path} ({format_size(file_size)})")
                except OSError as e:
                    print(f"[错误] 无法复制 {file_rel_path}: {e}")

    # 打印统计信息
    print("-" * 60)
    print("完成统计:")
    print(f"  创建目录: {stats['dirs']} 个")
    print(f"  复制文件: {stats['copied']} 个 (共 {format_size(stats['copied_size'])})")
    print(f"  链接文件: {stats['linked']} 个 (共 {format_size(stats['linked_size'])})")
    print(f"  总计文件: {stats['copied'] + stats['linked']} 个")

    return True


def smart_create_quant_model_json(
    src_dir,
    dst_dir,
    quant_pattern=None,
    quant_config="W8A8_MXFP8",
    quant_method="ascend",
):
    import json
    import re

    quant_pattern = quant_pattern or [
        ".*.down_proj.weight",
        ".*.gate_proj.weight",
        ".*.up_proj.weight",
        ".*.q_proj.weight",
        ".*.k_proj.weight",
        ".*.v_proj.weight",
        ".*.o_proj.weight",
    ]
    print(f"  quant_pattern: {quant_pattern}")

    with open(f"{src_dir}/model.safetensors.index.json") as f:
        data = json.load(f)
        weight_map = data["weight_map"]
        quant_mapping = {"quant_method": quant_method}
        for key, val in weight_map.items():
            res = [re.match(pattern, key) is not None for pattern in quant_pattern]
            if any(res):
                quant_mapping[key] = quant_config
            else:
                quant_mapping[key] = "FLOAT"
        with open(f"{dst_dir}/quant_model_description.json", "w") as f2:
            json.dump(quant_mapping, f2, indent=4, sort_keys=True)
            print("  创建文件：quant_model_description.json ")


def main():
    parser = argparse.ArgumentParser(
        description="智能目录复制工具：小文件直接复制，大文件创建软链接",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s /path/to/src /path/to/dst
  %(prog)s /path/to/src /path/to/dst --threshold 20M
  %(prog)s /path/to/src /path/to/dst -t 5242880  # 5MB (字节)
        """,
    )

    parser.add_argument("src", help="源目录路径")
    parser.add_argument("dst", help="目标目录路径")
    parser.add_argument(
        "-t",
        "--threshold",
        default="10M",
        help="大小阈值，超过此大小的文件将创建软链接 (默认: 10M)。支持格式: 10M, 1G, 5242880)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="安静模式，减少输出信息")
    parser.add_argument(
        "--quant_pattern",
        action="append",
        default=[],
        help="需要量化的权重正则表达式（可多次指定），例如 --quant_pattern '.*.down_proj.weight'",
    )
    parser.add_argument("--quant_config", type=str, default="W8A8_MXFP8", help="量化配置，如 W8A8_MXFP8")
    parser.add_argument("--quant_method", type=str, default="ascend", help="量化方法，如 ascend")

    args = parser.parse_args()

    # 解析大小阈值
    threshold_str = args.threshold.upper()
    if threshold_str.endswith("K"):
        size_threshold = int(float(threshold_str[:-1]) * 1024)
    elif threshold_str.endswith("M"):
        size_threshold = int(float(threshold_str[:-1]) * 1024 * 1024)
    elif threshold_str.endswith("G"):
        size_threshold = int(float(threshold_str[:-1]) * 1024 * 1024 * 1024)
    else:
        size_threshold = int(threshold_str)

    # 执行复制
    success = smart_copy_directory(
        args.src,
        args.dst,
        size_threshold=size_threshold,
        verbose=not args.quiet,
    )
    if success:
        smart_create_quant_model_json(
            args.src,
            args.dst,
            args.quant_pattern,
            args.quant_config,
            args.quant_method,
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
