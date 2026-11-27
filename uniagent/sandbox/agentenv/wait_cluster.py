"""Wait until the two private workers have registered their NPU resources."""

import os
import time

import ray

ray.init(address=os.environ["RAY_ADDRESS"], logging_level="ERROR")
try:
    for _ in range(120):
        nodes = [n for n in ray.nodes() if n["Alive"]]
        if len(nodes) == 2 and all(n["Resources"].get("NPU", 0) >= 16 for n in nodes):
            break
        time.sleep(5)
    else:
        raise RuntimeError("Timed out waiting for two 16-NPU workers")
finally:
    ray.shutdown()
