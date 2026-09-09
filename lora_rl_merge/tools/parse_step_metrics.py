#!/usr/bin/env python3
"""从 verl console logger 的 step 行抽取关键指标。用法：parse_step_metrics.py <log> [--all-keys]"""

import re
import sys

log = sys.argv[1]
show_keys = "--all-keys" in sys.argv
KEYS = [
    "timing_s/step",
    "timing_s/gen",
    "timing_s/old_log_prob",
    "timing_s/ref",
    "timing_s/update_actor",
    "timing_s/update_weights",
    "timing_s/generate_sequences",
    "perf/throughput",
    "perf/total_num_tokens",
    "perf/max_memory_allocated_gb",
    "perf/max_memory_reserved_gb",
    "critic/score/mean",
    "critic/rewards/mean",
    "response_length/mean",
    "response_length/clip_ratio",
    "actor/pg_loss",
    "actor/kl_loss",
    "actor/grad_norm",
    "actor/lr",
]
ansi = re.compile(r"\x1b\[[0-9;]*m")
for line in open(log, errors="replace"):
    line = ansi.sub("", line)
    m = re.search(r"TaskRunnerV1.*?step:(\d+) - (.*)", line)
    if not m:
        continue
    kv = {}
    for part in m.group(2).split(" - "):
        if ":" in part:
            k, v = part.rsplit(":", 1)
            kv[k.strip()] = v.strip()
    out = []
    for k in KEYS:
        if k in kv:
            short = (
                k.replace("timing_s/", "t_")
                .replace("perf/", "")
                .replace("critic/", "")
                .replace("actor/", "a_")
                .replace("response_length/", "resp_")
            )
            out.append(f"{short}={kv[k]}")
    print("step", m.group(1), " ".join(out))
    if show_keys and m.group(1) == "1":
        print("   keys:", ", ".join(sorted(kv.keys())))
