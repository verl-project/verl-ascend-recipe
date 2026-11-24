# Qwen3-8B RLOO Ascend 910B/910C 训练配方

本配方使用 FSDP2 和 vLLM Ascend 运行 Qwen3-8B RLOO 训练。启动脚本读取 GSM8K 的 train.parquet 和 test.parquet；verl 版本固定在 [REQUIRED_VERL.txt](REQUIRED_VERL.txt)。

## 运行方法

在仓库根目录安装指定版本的 verl：

    ./install_verl.sh --recipe rloo --method git --dest ./verl

使用 verl 的 examples/data_preprocess/gsm8k.py 准备 GSM8K 数据，然后启动训练：

    bash rloo/run_qwen3_8b_rloo_fsdp_npu.sh --data_path=/path/to/gsm8k --model_path=/path/to/Qwen3-8B

数据目录需包含 train.parquet 和 test.parquet。两个路径参数之后可以继续传入 Hydra 覆盖参数。脚本默认使用一张可见 NPU；多卡运行时需设置 n_gpus_per_node 和设备可见性。启动前请加载 CANN 环境。

脚本使用 RLOO 优势估计、FSDP2 Actor/Reference、BF16、vLLM rollout 和 DAPO reward。为匹配 GSM8K 的严格答案格式，脚本关闭 Qwen3 thinking；相关模型配置已写入脚本，无需额外的模型补丁。

## 验证材料

[Issue #79 实践报告](../verl_ascend_practice/reports/rloo/README.md)记录了一次独立的 8 卡 Ascend 910B、100 步训练。[完整训练日志](../verl_ascend_practice/reports/rloo/evidence/train.log)、[reward 曲线](../verl_ascend_practice/reports/rloo/reward_curve.png)和[材料索引](../verl_ascend_practice/reports/rloo/evidence/README.md)均随本 PR 提交。

实测前 10 步与后 10 步的平均 reward 分别为 0.578812 和 0.741040。8 卡合计训练吞吐为 1588.43 tokens/s，平均每卡 198.55 tokens/s；该口径包含训练步中的 prompt 和 response token。实验覆盖参数、验证集范围与启动脚本退出状态见实践报告。

**本次实验使用的运行脚本**：[run_qwen3_8b_rloo_fsdp_npu.sh](run_qwen3_8b_rloo_fsdp_npu.sh)。

实验环境还应用了尚未纳入本 PR 的向量化 RLOO 优势计算改动，且服务器上的启动脚本在训练期间被修改。因此，上述性能数据是该实验配置的记录，不能直接视作当前 PR 脚本与未修改 verl 的性能基准。
