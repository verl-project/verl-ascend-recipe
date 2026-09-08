# Ascend A2/A3 + VeRL Qwen3 SFT/RL Full Training Guide
## 1\. Introduction

This document provides an end-to-end standardized practice workflow 
for **Supervised Fine-Tuning (SFT)** and **Reinforcement Learning (RL)** of Qwen3-series models. 
The entire training pipeline is built on **Ascend A2/A3 hardware platforms** and the **VeRL v0.7.1** training framework.

Taking the classic mathematical reasoning dataset GSM8K as the training example, 
this tutorial adapts to the Ascend software and hardware ecosystem. 
It covers the full process of environment configuration, data preprocessing, model training, 
and metric visualization, which can be directly deployed for fine-tuning and alignment of Qwen3-series models.

## 2. Environment Requirements

This chapter details the verified hardware specifications, software versions, and container images for stable VeRL training and Qwen3 model adaptation.

### 2.1 Hardware Environment

This training pipeline supports 2 types of Ascend devices. Users can select the appropriate hardware according to model scale:

- **Ascend A2**: 8 NPUs per node.

- **Ascend A3**: 16 NPUs per node.

### 2.2 Software Environment

All software versions are fully compatible and pre-configured in the official container images. No additional dependency installation is required for out-of-the-box deployment.

|Component| Version                                                                                                                                                        |Description|
|---|----------------------------------------------------------------------------------------------------------------------------------------------------------------|---|
|CANN| 8.5.0                                                                                                                                                          |Core driver and operator library for Ascend computing|
|Python| 3.11                                                                                                                                                           |Stable version adapted for VeRL and model training scripts|
|VeRL Framework| 0.7.1                                                                                                                                                          |Unified training framework for LLM SFT and RL alignment|
|A3 Docker Image| [quay.io/ascend/verl:verl-8.5.0-a3-ubuntu22.04-py3.11-v0.7.1](https://quay.io/repository/ascend/verl?tab=tags&tag=verl-8.5.0-a3-ubuntu22.04-py3.11-v0.7.1)     |Optimized for Ascend A3, integrated with complete VeRL project|
|A2 Docker Image| [quay.io/ascend/verl:verl-8.5.0-910b-ubuntu22.04-py3.11-v0.7.1](https://quay.io/repository/ascend/verl?tab=tags&tag=verl-8.5.0-910b-ubuntu22.04-py3.11-v0.7.1) |Optimized for Ascend A2, integrated with complete VeRL project|

## 3. Training Dataset

### 3.1 Dataset Overview

This tutorial adopts the **GSM8K (Grade School Math 8K)** dataset, a classic open-source benchmark 
for evaluating the mathematical reasoning capabilities of large language models. It consists of a large number of 
elementary math word problems and is widely used for LLM supervised fine-tuning and reinforcement learning alignment tasks.

Dataset Repository:[https://huggingface.co/datasets/openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)

### 3.2 Dataset Adaptation

The VeRL official preprocessing scripts are built into the provided Docker images. 
The pipeline automatically completes dataset format conversion to generate standard data for SFT and RL training.

## 4. Full Model Training Pipeline

This chapter includes 2 core training modules: **SFT** and **RL**. We provide standardized pipelines based on Qwen3-0.6B and Qwen3-8B, 
which can be extended to other Qwen3 Dense and MoE models by modifying relevant configuration parameters.

### 4.1 SFT (Qwen3-0.6B + GSM8K)

We perform supervised fine-tuning on the lightweight Qwen3-0.6B model using the GSM8K dataset to enhance its fundamental mathematical reasoning ability. 
The pipeline includes data preprocessing, task launching, and training metric visualization.

#### 4.1.1 Data Preprocessing

Use the built-in VeRL script to standardize and clean the GSM8K dataset, generating formatted data for SFT training. No extra engineering deployment is required.

```bash
# GSM8K SFT data preprocessing
# local_dataset_path: Path for raw dataset storage
# local_save_dir: Output path for processed SFT data
python verl/examples/data_preprocess/gsm8k_multiturn_sft.py \
--local_dataset_path "${HOME}/data/gsm8k" \
--local_save_dir "${HOME}/data/gsm8k_sft"
```

#### 4.1.2 Launch SFT Training

After data preprocessing, execute the training script to start distributed SFT on Ascend devices.

```bash
# Launch Qwen3-0.6B SFT training on Ascend NPU
bash run_sft_qwen3_0_6b_npu.sh
```

#### 4.1.3 Loss Curve

Training loss is recorded in real time during the training process. 

<img src="figures/qwen3_0_6b_gsm8k_sft_loss_curve.png" style="width: 50%;" alt="Loss Curve">

### 4.2 RL (Qwen3-8B + GSM8K)

We perform RL alignment on the Qwen3-8B model. 
The reward-driven optimization refines the model’s reasoning logic and improves the accuracy and normalization of mathematical reasoning outputs.

#### 4.2.1 Data Preprocessing

Run the RL preprocessing script to generate RL dataset, supporting reward calculation.

```bash
# GSM8K RL data preprocessing
python verl/examples/data_preprocess/gsm8k.py \
--local_dataset_path "${HOME}/data/gsm8k" \
--local_save_dir "${HOME}/data/gsm8k_rl"
```

#### 4.2.2 Launch RL Training

Start the RL training task for Qwen3-8B after data preparation is completed.

```bash
# Launch Qwen3-8B RL training on Ascend NPU
bash run_rl_qwen3_8b_npu.sh
```

#### 4.2.3 Reward Curve

The core monitoring metric for RL training is reward value. 
A normal convergence trend presents a steady increase in average reward that finally plateaus, 
which demonstrates continuous optimization of the model’s reasoning policy.

<img src="figures/qwen3_8b_gsm8k_rl_reward_curve.png" style="width: 50%;" alt="Reward Curve">

### 4.3 ReMax (Qwen3-8B + GSM8K/MATH)

This recipe runs Qwen3-8B ReMax training with an FSDP actor/reference model and a
vLLM-Ascend rollout backend. It was validated on an Atlas 800T A2 node with 8 x
Ascend 910B3 64 GB NPUs.

Corresponding task:
[verl-ascend-recipe #23](https://github.com/verl-project/verl-ascend-recipe/issues/23)

#### 4.3.1 Adaptation

ReMax generates sampled responses together with one greedy baseline for each prompt,
then uses their reward difference as the policy-gradient advantage. The combined
rollout path is already backend-independent in VeRL, so vLLM-Ascend can be used
without modifying the trainer or rollout implementation.

```text
GSM8K / MATH prompts
          |
          v
vLLM-Ascend sampled rollout + greedy baseline
          |
          v
sampled reward - baseline reward
          |
          v
ReMax advantage + KL-in-reward
          |
          v
FSDP actor update and rollout weight synchronization
```

The launch script applies the following Ascend-specific settings:

- `trainer.device=npu` with the vLLM-Ascend rollout backend.
- Torch compile is disabled for the actor and reference model.
- Actor parameter/optimizer offload and reference parameter offload are enabled to
  leave HBM for the colocated rollout engine.
- Rollout memory utilization defaults to 0.6 because ReMax generates sampled
  responses plus a greedy baseline.
- Full-decode-only ACL Graph capture, HCCL timeouts, task queue, and CPU affinity
  settings are enabled.

#### 4.3.2 Validated Environment and Configuration

| Component | Version or configuration |
| --- | --- |
| Hardware | Atlas 800T A2, 8 x Ascend 910B3 64 GB |
| CANN | 25.5.1 |
| torch-npu | 2.9.0.post2 |
| vLLM | 0.18 |
| vLLM-Ascend | 0.18.1.dev41 |
| transformers | 5.3.0.dev0 |
| Model | Qwen3-8B |
| Algorithm | ReMax with KL-in-reward, `kl_coef=0.001` |
| Training backend | FSDP |
| Rollout backend | vLLM-Ascend, TP=2, n=4 |
| Training batch / mini batch | 128 / 32 |
| Prompt / response length | 1024 / 1024 |
| Actor learning rate | `1e-6` |

#### 4.3.3 Data Preprocessing

Prepare the default GSM8K and MATH datasets from the VeRL repository root:

```bash
python3 examples/data_preprocess/gsm8k.py \
--local_save_dir "${HOME}/data/gsm8k"

python3 examples/data_preprocess/math_dataset.py \
--local_save_dir "${HOME}/data/math"
```

The default dataset directories can be overridden with `GSM8K_DATA_DIR` and
`MATH_DATA_DIR`.

#### 4.3.4 Launch ReMax Training

Run the recipe from the VeRL repository root:

```bash
TRAIN_BATCH_SIZE=128 \
PPO_MINI_BATCH_SIZE=32 \
ROLLOUT_N=4 \
ROLLOUT_TP=2 \
ROLLOUT_GPU_MEM_UTIL=0.6 \
MAX_RESPONSE_LENGTH=1024 \
MODEL_PATH=/data/models/Qwen3-8B \
bash /path/to/verl-ascend-recipe/verl_ascend_practice/run_qwen3_8b_remax_fsdp_npu.sh \
    'trainer.logger=["console"]'
```

Model, data, cluster size, batch size, sequence length, rollout parallelism, memory
utilization, save frequency, and test frequency can be overridden with environment
variables. Additional arguments are forwarded to `verl.trainer.main_ppo` as Hydra
overrides.

#### 4.3.5 Validation and Performance

| Test | Result |
| --- | --- |
| Smoke test | Qwen3-0.6B completed 2 steps on 8 NPUs; ReMax advantage and `temperature=0` greedy baseline requests were observed |
| Stability test | Qwen3-8B completed 5 consecutive steps with batch size 128 and response length 1024, without OOM or HCCL timeout |
| Training run | Qwen3-8B completed 60 steps on GSM8K in approximately 2 hours 25 minutes |
| Reward trend | `critic/score/mean` increased from approximately 0.26 to the 0.60-0.76 range, peaking at 0.764 |
| Throughput | Average cluster throughput was approximately 381 tokens/s, with a 337-425 tokens/s range |
| Stress test | Response length 2048 completed 3 steps without OOM at 343-411 tokens/s |

The measured throughput is above the 100 tokens/s target, and the reward shows a
clear upward trend.

#### 4.3.6 Reward Curve

The following figure shows `critic/rewards/mean` during the training run:

<img src="https://github.com/user-attachments/assets/45428637-fe4b-45ba-a63a-96cac151e719" style="width: 70%;" alt="Qwen3-8B ReMax Reward Curve">


## 5. Extension and Compatibility

- **Model Extension**: This pipeline is compatible with all Qwen3-series Dense models (1.7B, 4B, 14B, etc.) and MoE models. 
Only model parameters and configuration files need replacement for migration.

- **Hardware Adaptation**: Small-scale models are recommended for Ascend A2 training. 
8B+ dense models and MoE models are recommended to run on Ascend A3 for higher throughput and training stability.

- **Dataset Migration**: The pipeline supports custom reasoning and dialogue datasets. 
Users only need to modify the corresponding preprocessing scripts and training configurations.

## 6. Troubleshooting

- **Environment Mismatch**: Strictly use the specified Docker images and CANN version. 
Version inconsistency will cause operator errors and abnormal NPU training.

- **Data Path Error**: Ensure sufficient read/write permissions for the dataset directory to avoid data loading failures during preprocessing and training.

- **Convergence Issues**: If training loss fails to decrease or reward fails to increase, 
adjust hyperparameters including learning rate, batch size, and training epochs.
