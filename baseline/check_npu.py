import argparse
import re

# ==================== 命令行参数 ====================
parser = argparse.ArgumentParser()
parser.add_argument("--log", required=True, help="日志文件路径")
parser.add_argument("--base", required=True, help="基线txt路径")
args = parser.parse_args()

# ==================== 指标配置 ====================
MEMORY_KEY = "actor/perf/max_memory_allocated_gb"
PERF_KEYS = ["perf/throughput", "timing_s/step"]
ACC_KEYS = ["critic/rewards/mean", "training/rollout_probs_diff_mean"]

# ==================== 正则（兼容科学计数法和 np.float64(xxx) 格式） ====================
STEP_PATTERN = re.compile(r"step:(\d+)")
# 匹配整数、小数和科学计数法，例如 -123.45、1e-5、-2.3E+4
NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
# 匹配两种格式：key:np.float64(1e-5)  或 key:-2.3E+4
METRIC_PATTERN = re.compile(rf"([\w/]+):(?:np\.float64\()?({NUMBER_PATTERN})(?:\))?")


# ==================== 1. 解析日志文件 ====================
def parse_log(log_path):
    step_data = {}
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            step_match = STEP_PATTERN.search(line)
            if not step_match:
                continue
            step = int(step_match.group(1))

            metrics = {}
            for m in METRIC_PATTERN.finditer(line):
                k = m.group(1)
                num_str = m.group(2)
                try:
                    metrics[k] = float(num_str)
                except Exception:
                    continue
            if metrics:
                step_data[step] = metrics

    if not step_data:
        raise ValueError("日志中无有效step")

    # 内存：最后一步
    last_step = max(step_data.keys())
    mem_val = step_data[last_step].get(MEMORY_KEY, None)
    if mem_val is None:
        raise ValueError(f"最后一步{last_step}未读取到内存指标 {MEMORY_KEY}")

    # 5~14步
    target_steps = [s for s in range(5, 15) if s in step_data]
    if not target_steps:
        raise ValueError("无5~14步数据")

    # 均值函数
    def avg(key):
        vals = []
        for s in target_steps:
            val = step_data[s].get(key)
            if val is not None:
                vals.append(val)
        if not vals:
            raise ValueError(f"日志5-14步中未找到指标: {key}")
        return sum(vals) / len(vals)

    A = {
        MEMORY_KEY: mem_val,
        PERF_KEYS[0]: avg(PERF_KEYS[0]),
        PERF_KEYS[1]: avg(PERF_KEYS[1]),
        ACC_KEYS[0]: avg(ACC_KEYS[0]),
        ACC_KEYS[1]: avg(ACC_KEYS[1]),
    }
    return A


# ==================== 2. 解析基线文件 ====================
def parse_base(base_path):
    B = {}
    with open(base_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k_part, v_str = line.split(":", 1)
            k = k_part.strip()
            # 基线文件如果也存在np.float64格式一并兼容处理
            num_match = re.search(rf"(?:np\.float64\()?({NUMBER_PATTERN})(?:\))?", v_str)
            if not num_match:
                continue
            num_str = num_match.group(1)
            try:
                B[k] = float(num_str.strip())
            except ValueError:
                continue

    # 兼容旧基线key映射
    if "perf/max_memory_allocated_gb" in B and MEMORY_KEY not in B:
        B[MEMORY_KEY] = B["perf/max_memory_allocated_gb"]

    # 校验基线所有必需指标存在
    required_keys = [MEMORY_KEY] + PERF_KEYS + ACC_KEYS
    miss_keys = [k for k in required_keys if k not in B]
    if miss_keys:
        raise ValueError(f"基线文件缺失指标: {', '.join(miss_keys)}")
    return B


# ==================== 3. 校验逻辑（不exit，抛异常） ====================
def check(A, B):
    fail = []
    print("\n==================== 指标校验结果 ====================")

    # 1. 内存：绝对值差值 <=1
    key = MEMORY_KEY
    a = A[key]
    b = B[key]
    diff = a - b
    print(f"【内存，绝对值差值<=1】{key:<40} | A={a:.6f} B={b:.6f} | 差值={diff:.6f}")
    if abs(diff) > 1:
        fail.append(f"{key} 差值 {diff:.6f}，超出阈值1")

    # 2. perf/throughput：变化率 <=5%
    key = PERF_KEYS[0]
    a = A[key]
    b = B[key]
    pct = (a - b) / b * 100
    print(f"【性能，变化率<=5%】{key:<44} | A={a:.6f} B={b:.6f} | 变化={pct:.2f}%")
    if abs(pct) > 5:
        fail.append(f"{key} 变化率 {pct:.2f}%，超出阈值5%")

    # 3. timing_s/step：变化率 <=5%
    key = PERF_KEYS[1]
    a = A[key]
    b = B[key]
    pct = (a - b) / b * 100
    print(f"【性能，变化率<=5%】{key:<44} | A={a:.6f} B={b:.6f} | 变化={pct:.2f}%")
    if abs(pct) > 5:
        fail.append(f"{key} 变化率 {pct:.2f}%，超出阈值5%")

    # 4. critic/rewards/mean：绝对值差值 <=0.05
    key = ACC_KEYS[0]
    a = A[key]
    b = B[key]
    diff = a - b
    print(f"【精度，绝对值差值<=0.05】{key:<36} | A={a:.6f} B={b:.6f} | 差值={diff:.6f}")
    if abs(diff) > 0.05:
        fail.append(f"{key} 差值 {diff:.6f}，超出阈值0.05")

    # 5. training/rollout_probs_diff_mean：变化率 <=5%
    key = ACC_KEYS[1]
    a = A[key]
    b = B[key]
    pct = (a - b) / b * 100
    print(f"【精度，变化率<=5%】{key:<40} | A={a:.6f} B={b:.6f} | 变化={pct:.2f}%")
    if abs(pct) > 5:
        fail.append(f"{key} 变化率 {pct:.2f}%，超出阈值5%")

    print("========================================================\n")

    # 校验不通过，抛出异常
    if fail:
        err_msg = "指标校验失败，不达标项：\n" + "\n".join(f"  - {item}" for item in fail)
        raise AssertionError(err_msg)
    print("✅ 所有指标均达标！")


# ==================== 主入口 ====================
if __name__ == "__main__":
    try:
        A = parse_log(args.log)
        B = parse_base(args.base)
        check(A, B)
    except Exception as e:
        # 捕获异常并抛出，便于上层调用识别错误
        raise RuntimeError(f"执行校验失败: {e}") from e
