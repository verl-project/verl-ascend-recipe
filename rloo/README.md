# Qwen3-8B RLOO Ascend 910B/910C 训练配方

本配方使用 FSDP2 和 vLLM Ascend 运行 Qwen3-8B RLOO 训练。启动脚本读取 GSM8K 的 train.parquet 和 test.parquet；verl 版本固定在 [REQUIRED_VERL.txt](REQUIRED_VERL.txt)。

## 运行方法

在仓库根目录安装指定版本的 verl：

    ./install_verl.sh --recipe rloo --method git --dest ./verl

使用 verl 的 examples/data_preprocess/gsm8k.py 准备 GSM8K 数据，然后启动训练：

    bash rloo/run_qwen3_8b_rloo_fsdp_npu.sh --data_path=/path/to/gsm8k --model_path=/path/to/Qwen3-8B

数据目录需包含 train.parquet 和 test.parquet。两个路径参数之后可以继续传入 Hydra 覆盖参数。多卡运行时需设置 n_gpus_per_node 和设备可见性。

