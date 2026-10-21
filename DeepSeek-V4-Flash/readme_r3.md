# R3 (Router Replay)

R3（Router Replay，路由重放）用于保证 MoE 模型 RL 训练与 rollout 阶段的专家路由一致性：rollout 阶段记录 router top-k 决策，训练阶段按记录重放，消除训练/推理路由不一致带来的偏差。

## 1. 安装

R3 默认关闭，安装方式参考readme.md

```bash
bash install.sh            # 不含 R3
bash install.sh --r3       # 启用 R3（等价于 INSTALL_R3=1 bash install.sh）
```

## 2. 使用

安装启用 R3 后，通过训练配置开启：

```bash
# 本 recipe 使用 mindspeed 后端，注意配置节点是 actor.mindspeed 而非 actor.megatron
actor_rollout_ref.actor.mindspeed.router_replay.mode=R3
actor_rollout_ref.rollout.enable_rollout_routing_replay=True
```

