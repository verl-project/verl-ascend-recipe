#!/usr/bin/env python3
"""
智能目录复制工具（单文件双引擎）
- 小文件（<=10M）：直接复制；大文件（>10M）：创建软链接
- 生成 quant_model_description.json

引擎 1（默认，正则引擎）：30B 版逻辑原样保留——按 --quant_pattern 正向正则
从 model.safetensors.index.json 生成量化名单，默认 7 条投影正则，行为与旧版
逐字节一致。

引擎 2（--profile qwen3_5_moe，结构感知引擎）：面向 Qwen3.5-35B-A3B 这类
hybrid（GatedDeltaNet + full attention）+ 融合 MoE 架构。正则引擎从索引
生成不出该架构的目标 json（120 个 vLLM FusedMoE 查表合成键在索引中不存在、
80 个融合专家键无 .weight 后缀、768 个 MTP 专家键会被默认正则误纳），故按
模型结构语义生成：
  * 作用域：仅 model.language_model.layers.{*} 内，其余（visual / lm_head /
    embed_tokens / mtp / 各类 norm）一律 FLOAT
  * 融合专家 mlp.experts.{gate_up,down}_proj（3D）-> 量化，并合成
    mlp.experts.0.{gate,up,down}_proj.weight 查表键
  * nn.Linear 且 min(in,out) >= 64 才量化（对齐 FSDPTurbo quantize.linear.mx
    的 _SMALL_LINEAR_DIM=64；mlp.gate 路由器为裸 Parameter、shared_expert_gate
    等 out=1/32 的小投影保持 FLOAT）

--exclude_pattern（两引擎通用）：负向正则，可多次，优先级最高，命中一律
FLOAT。动机：Qwen3.5-35B-A3B 的 linear_attn（gated delta net）三投影
in_proj_qkv / in_proj_z / out_proj 对 MXFP8 量化敏感——递归式线性注意力的
投影误差沿序列递归累积放大，实测（GRPO, geo3k, 601 题 val）量化这 90 项使
step0 val 从 bf16 簇 403~409 掉到 373~387（-26 题）；而 include 侧
`.*.out_proj.weight` 会同时命中 self_attn.out_proj（应量化）与
linear_attn.out_proj（不能量化），纯正向无法区分。

用法:
  # 30B 等（正则引擎，默认）
  %(prog)s /model/Qwen3-30B-A3B /model/Qwen3-30B-A3B-MXFP8
  # Qwen3.5-35B-A3B（结构感知引擎 + 排除 linear_attn）
  %(prog)s /model/Qwen3.5-35B-A3B /model/Qwen3.5-35B-A3B-MXFP8 \
      --profile qwen3_5_moe --exclude_pattern '.*linear_attn\\..*'
"""

import argparse
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path

QUANT_DESC_FILENAME = "quant_model_description.json"
INDEX_FILENAME = "model.safetensors.index.json"

PROFILES = ("qwen3_5_moe",)


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

    # 目标目录在源目录内部会导致无限递归复制（walk 会把刚写入的目标再次复制）
    if dst_path == src_path or dst_path.is_relative_to(src_path):
        print(f"错误：目标目录位于源目录内部（{dst_path} ⊂ {src_path}），已拒绝")
        return False

    # 统计信息
    stats = {"copied": 0, "linked": 0, "dirs": 0, "copied_size": 0, "linked_size": 0, "failed": 0}

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
                try:
                    dst_file.unlink()
                except OSError as e:
                    stats["failed"] += 1
                    print(f"[错误] 无法替换已存在文件 {file_rel_path}: {e}")
                    continue

            if file_size > size_threshold:
                # 大文件：创建软链接
                try:
                    os.symlink(str(src_file), str(dst_file))
                    stats["linked"] += 1
                    stats["linked_size"] += file_size
                    if verbose:
                        print(f"[链接] {file_rel_path} ({format_size(file_size)})")
                except OSError as e:
                    stats["failed"] += 1
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
                    stats["failed"] += 1
                    print(f"[错误] 无法复制 {file_rel_path}: {e}")

    # 打印统计信息
    print("-" * 60)
    print("完成统计:")
    print(f"  创建目录: {stats['dirs']} 个")
    print(f"  复制文件: {stats['copied']} 个 (共 {format_size(stats['copied_size'])})")
    print(f"  链接文件: {stats['linked']} 个 (共 {format_size(stats['linked_size'])})")
    print(f"  总计文件: {stats['copied'] + stats['linked']} 个")
    if stats["failed"]:
        print(f"  失败: {stats['failed']} 个（复制/链接出错，目标目录不完整）")

    return stats["failed"] == 0


# ============================================================================
# 引擎 1：正则引擎（30B 逻辑原样保留，只增不改）
# ============================================================================
def smart_create_quant_model_json(
    src_dir,
    dst_dir,
    quant_pattern=None,
    exclude_pattern=None,
    quant_config="W8A8_MXFP8",
    quant_method="ascend",
    group_size=None,
):
    # 默认与 30B 旧版完全一致的 7 条正则
    quant_pattern = quant_pattern or [
        ".*.down_proj.weight",
        ".*.gate_proj.weight",
        ".*.up_proj.weight",
        ".*.q_proj.weight",
        ".*.k_proj.weight",
        ".*.v_proj.weight",
        ".*.o_proj.weight",
    ]
    exclude_pattern = exclude_pattern or []
    print(f"  quant_pattern: {quant_pattern}")
    print(f"  exclude_pattern: {exclude_pattern}")

    with open(f"{src_dir}/model.safetensors.index.json") as f:
        data = json.load(f)
        weight_map = data["weight_map"]
        quant_mapping = {"quant_method": quant_method}
        # group_size 仅在显式指定时写入（vllm-ascend W8A8_MXFP8 读取，默认 32）；
        # 缺省不写入以保持与旧版工具输出逐字节一致
        if group_size is not None:
            quant_mapping["group_size"] = group_size
        n_quant = n_float = 0
        for key, val in weight_map.items():
            # exclude 优先：命中负向正则的一律 FLOAT，不再看 include
            if any(re.match(pattern, key) is not None for pattern in exclude_pattern):
                quant_mapping[key] = "FLOAT"
                n_float += 1
                continue
            res = [re.match(pattern, key) is not None for pattern in quant_pattern]
            if any(res):
                quant_mapping[key] = quant_config
                n_quant += 1
            else:
                quant_mapping[key] = "FLOAT"
                n_float += 1
        with open(f"{dst_dir}/quant_model_description.json", "w") as f2:
            json.dump(quant_mapping, f2, indent=4, sort_keys=True)
            print("  创建文件：quant_model_description.json ")
        print(f"  量化项: {n_quant}，FLOAT 项: {n_float}")


# ============================================================================
# 引擎 2：结构感知引擎（--profile qwen3_5_moe）
# 收编自验证过的 create_qwen3_5_mxfp8_rollout.py 键名语义：
# 取值依据 FSDPTurbo config_fp8.yaml 量化段（mxfp8/E4M3/block32，
# converters=[quantize.linear.mx, quantize.moe.mx]，
# quant_apply_modules=[model.language_model.layers.{*}]）
# ============================================================================

# FSDPTurbo fsdp_turbo/quantization/utils.py: _SMALL_LINEAR_DIM
SMALL_LINEAR_DIM = 64

# vllm-ascend modelslim_config.py: packed_modules_model_mapping["qwen3_5_moe"]["experts"]
# FusedMoE 用老式逐专家分片键查表（transformers 5.x 融合格式检查点需合成这些键）
EXPERT_SHARD_NAMES = ("gate_proj", "up_proj", "down_proj")

# hf_to_vllm_mapper（vllm qwen3_vl.py / qwen3_5.py）：描述键变换规则
HF_TO_VLLM_PREFIX = (
    ("model.visual.", "visual."),
    ("lm_head.", "language_model.lm_head."),
    ("model.language_model.", "language_model.model."),
)

# vllm-ascend packed_modules_model_mapping["qwen3_5_moe"]：
# vLLM 融合模块名 -> 检查点分片名（各分片取值必须一致）
PACKED_MODULES = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
    "in_proj_qkvz": ("in_proj_qkv", "in_proj_z"),
    "in_proj_ba": ("in_proj_b", "in_proj_a"),
    "experts": tuple(f"experts.0.{n}" for n in EXPERT_SHARD_NAMES),
}

LANG_LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
FUSED_EXPERT_RE = re.compile(r"^mlp\.experts\.(gate_up_proj|down_proj)$")


def load_safetensors_shapes(model_dir):
    """读取所有分片的 safetensors 头部，返回 {权重名: shape}（不加载权重数据）。"""
    shapes = {}
    for path in sorted(Path(model_dir).glob("*.safetensors")):
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        for name, meta in header.items():
            if name != "__metadata__":
                shapes[name] = tuple(meta["shape"])
    return shapes


def load_model_config(model_dir):
    """读取 config.json，取 text_config 的关键结构信息。"""
    with open(Path(model_dir) / "config.json") as f:
        config = json.load(f)
    text = config.get("text_config", config)
    return {
        "model_type": config.get("model_type"),
        "num_hidden_layers": text.get("num_hidden_layers"),
        "layer_types": text.get("layer_types"),
        "num_experts": text.get("num_experts"),
    }


def classify_qwen3_5_weight(name, shape):
    """按量化作用域给单个权重定值，返回 (是否量化, 原因)。"""
    m = LANG_LAYER_RE.match(name)
    if not m:
        # visual / lm_head / embed_tokens / language_model.norm / mtp.*：
        # 不在 quant_apply_modules（model.language_model.layers.{*}）内，保持 bf16
        return False, "out-of-scope"
    suffix = m.group(2)

    if FUSED_EXPERT_RE.match(suffix):
        # quantize.moe.mx：融合 3D 专家（(E, 2I, H) / (E, H, I)），无维度下限检查
        return True, "moe.mx fused experts"

    if suffix == "mlp.gate.weight":
        # 路由器是裸 nn.Parameter（Qwen3_5MoeTopKRouter），非 nn.Linear，不被转换
        return False, "router (raw Parameter, not nn.Linear)"

    if name.endswith(".weight") and len(shape) == 2:
        # quantize.linear.mx 只转换 nn.Linear，且 in/out >= 64（module_filter_fn）
        if min(shape) >= SMALL_LINEAR_DIM:
            return True, "linear.mx in-scope"
        return False, f"linear dim < {SMALL_LINEAR_DIM} (shape {list(shape)})"

    # 各类 norm（1D）、conv1d（3D）、A_log / dt_bias（1D）等非 Linear 权重
    return False, "non-linear weight"


def create_qwen3_5_quant_model_description(
    src_dir,
    dst_dir,
    exclude_pattern=None,
    quant_config="W8A8_MXFP8",
    quant_method="ascend",
    group_size=32,
):
    """结构感知引擎：生成 quant_model_description.json。

    步骤：读 index 的 weight_map -> 读各分片 safetensors 头部取 shape ->
    逐权重按结构语义分类 -> exclude 覆盖为 FLOAT -> 为融合专家层合成
    逐专家查表键（同样受 exclude 约束）-> 写 json（含 quant_method/group_size）。
    """
    exclude_pattern = exclude_pattern or []
    src_path = Path(src_dir)
    index_path = src_path / INDEX_FILENAME
    if not index_path.exists():
        raise FileNotFoundError(f"缺少 {INDEX_FILENAME}：{index_path}")

    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    cfg = load_model_config(src_path)
    shapes = load_safetensors_shapes(src_path)
    missing_shape = [k for k in weight_map if k not in shapes]
    if missing_shape:
        raise RuntimeError(f"以下权重在 safetensors 头部中未找到: {missing_shape[:5]} ...")

    description = {"quant_method": quant_method, "group_size": group_size}
    n_quant = n_float = n_excluded = synthesized = 0

    def excluded(key):
        return any(re.match(p, key) is not None for p in exclude_pattern)

    for key in weight_map:
        is_quant, _reason = classify_qwen3_5_weight(key, shapes[key])
        if is_quant and excluded(key):
            # exclude 优先：命中负向正则的一律 FLOAT
            description[key] = "FLOAT"
            n_excluded += 1
            n_float += 1
        elif is_quant:
            description[key] = quant_config
            n_quant += 1
        else:
            description[key] = "FLOAT"
            n_float += 1

    # 合成 FusedMoE 查表键：vllm-ascend 对 "experts" 前缀按老式逐专家键查表
    for key in weight_map:
        m = LANG_LAYER_RE.match(key)
        if not (m and FUSED_EXPERT_RE.match(m.group(2))):
            continue
        layer_prefix = key[: key.index("mlp.experts")]
        for shard in EXPERT_SHARD_NAMES:
            synth = f"{layer_prefix}mlp.experts.0.{shard}.weight"
            if synth not in description:
                if excluded(synth):
                    description[synth] = "FLOAT"
                    n_excluded += 1
                    n_float += 1
                else:
                    description[synth] = quant_config
                    n_quant += 1
                synthesized += 1

    print(f"  profile: qwen3_5_moe (model_type={cfg['model_type']}, "
          f"layers={cfg['num_hidden_layers']}, experts/layer={cfg['num_experts']})")
    print(f"  exclude_pattern: {exclude_pattern}")
    out_path = Path(dst_dir) / QUANT_DESC_FILENAME
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(description, f, indent=4, sort_keys=True)
    print("  创建文件：quant_model_description.json ")
    print(f"  量化项: {n_quant}（含合成专家键 {synthesized}），FLOAT 项: {n_float}"
          f"（其中 exclude 覆盖 {n_excluded}）")
    return description, cfg


def validate_qwen3_5_description(description, cfg):
    """模拟 vllm-ascend AscendModelSlimConfig 的查表，提前暴露缺键/分片不一致。

    复现引擎侧行为：先做 hf_to_vllm 前缀变换，再按 vLLM 模块前缀 +
    packed 分片映射查 quant_description[shard + ".weight"]。
    返回错误列表（空 = 通过）。
    """
    mapped = {}
    for key, value in description.items():
        new_key = key
        for old, new in HF_TO_VLLM_PREFIX:
            if key.startswith(old):
                new_key = new + key[len(old):]
                break
        mapped[new_key] = value

    errors = []

    def lookup(prefix):
        proj = prefix.rsplit(".", 1)[-1]
        shards = PACKED_MODULES.get(proj)
        if shards:
            values = []
            for shard in shards:
                shard_key = f"{prefix[: prefix.rfind(proj)]}{shard}.weight"
                if shard_key not in mapped:
                    errors.append(f"缺键: {shard_key}")
                    return None
                values.append(mapped[shard_key])
            if len(set(values)) > 1:
                errors.append(f"融合模块 {prefix} 分片取值不一致: {values}")
                return None
            return values[0]
        key = f"{prefix}.weight"
        if key not in mapped:
            errors.append(f"缺键: {key}")
            return None
        return mapped[key]

    num_layers = cfg["num_hidden_layers"] or 0
    layer_types = cfg["layer_types"] or []
    for i in range(num_layers):
        lt = layer_types[i] if i < len(layer_types) else "full_attention"
        prefix = f"language_model.model.layers.{i}"
        if lt == "full_attention":
            lookup(f"{prefix}.self_attn.qkv_proj")
            lookup(f"{prefix}.self_attn.o_proj")
        else:
            lookup(f"{prefix}.linear_attn.in_proj_qkvz")
            lookup(f"{prefix}.linear_attn.in_proj_ba")
            lookup(f"{prefix}.linear_attn.out_proj")
        lookup(f"{prefix}.mlp.gate")
        lookup(f"{prefix}.mlp.shared_expert_gate")
        lookup(f"{prefix}.mlp.shared_expert.gate_up_proj")
        lookup(f"{prefix}.mlp.shared_expert.down_proj")
        lookup(f"{prefix}.mlp.experts")

    # 词表 / lm_head / 视觉塔（视觉保持 FLOAT，走 AscendUnquantizedLinearMethod）
    lookup("language_model.model.embed_tokens")
    lookup("language_model.lm_head")
    for v in ("attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2"):
        lookup(f"visual.blocks.0.{v}")
    for v in ("linear_fc1", "linear_fc2"):
        lookup(f"visual.merger.{v}")

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="智能目录复制工具：小文件直接复制，大文件创建软链接；双引擎生成量化描述",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s /path/to/src /path/to/dst                     # 正则引擎（默认，30B 行为）
  %(prog)s /path/to/src /path/to/dst --threshold 20M
  %(prog)s /path/to/src /path/to/dst --exclude_pattern '.*linear_attn\\..*'
  %(prog)s /model/Qwen3.5-35B-A3B /model/Qwen3.5-35B-A3B-MXFP8 \\
      --profile qwen3_5_moe --exclude_pattern '.*linear_attn\\..*'
        """,
    )

    parser.add_argument("src", help="源目录路径")
    parser.add_argument("dst", help="目标目录路径")
    parser.add_argument(
        "-t",
        "--threshold",
        default="10M",
        help="大小阈值，超过此大小的文件将创建软链接 (默认: 10M)。支持格式: 10M, 1G, 5242880",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="安静模式，减少输出信息")
    parser.add_argument(
        "--profile",
        choices=PROFILES,
        default=None,
        help="结构感知引擎（默认不启用=正则引擎）。qwen3_5_moe: Qwen3.5-35B-A3B 的 "
        "hybrid+融合 MoE 架构，按模型结构语义生成（含 FusedMoE 合成查表键、"
        "融合专家键、MTP 全 FLOAT）；同架构新模型直接复用，新架构需新增 profile",
    )
    parser.add_argument(
        "--quant_pattern",
        action="append",
        default=[],
        help="[正则引擎] 需要量化的权重正则表达式（可多次指定），例如 --quant_pattern '.*.down_proj.weight'",
    )
    parser.add_argument(
        "--exclude_pattern",
        action="append",
        default=[],
        help="排除量化的权重正则表达式（可多次指定，两引擎通用，优先级最高），"
        "例如 --exclude_pattern '.*linear_attn\\..*' 排除所有 linear_attn 权重",
    )
    parser.add_argument("--quant_config", type=str, default="W8A8_MXFP8", help="量化配置，如 W8A8_MXFP8")
    parser.add_argument("--quant_method", type=str, default="ascend", help="量化方法，如 ascend")
    parser.add_argument(
        "--group_size",
        type=int,
        default=None,
        help="MXFP8 块大小。正则引擎缺省不写入（与旧版一致）；"
        "--profile qwen3_5_moe 缺省按 32 写入（对齐该架构训练侧 block_size）",
    )

    args = parser.parse_args()

    if args.profile and args.quant_pattern:
        parser.error("--quant_pattern 仅用于正则引擎，与 --profile 互斥")

    # 解析大小阈值（支持 10M / 1G / 512K / 纯字节数）
    threshold_str = args.threshold.upper()
    try:
        if threshold_str.endswith("K"):
            size_threshold = int(float(threshold_str[:-1]) * 1024)
        elif threshold_str.endswith("M"):
            size_threshold = int(float(threshold_str[:-1]) * 1024 * 1024)
        elif threshold_str.endswith("G"):
            size_threshold = int(float(threshold_str[:-1]) * 1024 * 1024 * 1024)
        else:
            size_threshold = int(threshold_str)
    except ValueError:
        parser.error(f"--threshold 无法解析: {args.threshold!r}（支持格式: 10M, 1G, 512K, 5242880）")

    # 执行复制
    success = smart_copy_directory(
        args.src,
        args.dst,
        size_threshold=size_threshold,
        verbose=not args.quiet,
    )
    if not success:
        sys.exit(1)

    if args.profile == "qwen3_5_moe":
        description, cfg = create_qwen3_5_quant_model_description(
            args.src,
            args.dst,
            exclude_pattern=args.exclude_pattern,
            quant_config=args.quant_config,
            quant_method=args.quant_method,
            group_size=args.group_size if args.group_size is not None else 32,
        )
        errors = validate_qwen3_5_description(description, cfg)
        if errors:
            print("校验失败（模拟 vllm-ascend 查表）:")
            for e in errors[:20]:
                print(f"  {e}")
            sys.exit(1)
        print("校验通过：模拟 vllm-ascend 对全部 Linear/FusedMoE/Embedding 前缀"
              "的查表无缺键、无融合分片取值冲突。")
    else:
        smart_create_quant_model_json(
            args.src,
            args.dst,
            args.quant_pattern,
            args.exclude_pattern,
            args.quant_config,
            args.quant_method,
            args.group_size,
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
