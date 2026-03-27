<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/logo.svg?raw=true" width="60%" alt="DeepSeek-V3" />
</div>
<hr>
<div align="center" style="line-height: 1;">
  <a href="https://www.deepseek.com/"><img alt="Homepage"
    src="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/badge.svg?raw=true"/></a>
  <a href="https://chat.deepseek.com/"><img alt="Chat"
    src="https://img.shields.io/badge/🤖%20Chat-DeepSeek%20V3-536af5?color=536af5&logoColor=white"/></a>
  <a href="https://huggingface.co/deepseek-ai"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DeepSeek%20AI-ffc107?color=ffc107&logoColor=white"/></a>
  <br>
  <a href="https://discord.gg/Tc7c45Zzu5"><img alt="Discord"
    src="https://img.shields.io/badge/Discord-DeepSeek%20AI-7289da?logo=discord&logoColor=white&color=7289da"/></a>
  <a href="https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/qr.jpeg?raw=true"><img alt="Wechat"
    src="https://img.shields.io/badge/WeChat-DeepSeek%20AI-brightgreen?logo=wechat&logoColor=white"/></a>
  <a href="https://twitter.com/deepseek_ai"><img alt="Twitter Follow"
    src="https://img.shields.io/badge/Twitter-deepseek_ai-white?logo=x&logoColor=white"/></a>
  <br>
  <a href="LICENSE" style="margin: 2px;">
    <img alt="License" src="https://img.shields.io/badge/License-Apache 2.0-f5de53?&color=f5de53" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <br>
</div>

## 1. Introduction

This repository contains the official implementation for the paper: **[Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](Engram_paper.pdf)**.

> **Abstract:** While Mixture-of-Experts (MoE) scales capacity via conditional computation, Transformers lack a native primitive for knowledge lookup. To address this, we explore **conditional memory** as a complementary sparsity axis, instantiated via **Engram**, a module that modernizes classic $N$-gram embeddings for $\mathcal{O}(1)$ lookup.

**Key Contributions:**
- **Sparsity Allocation:** We formulate the trade-off between neural computation (MoE) and static memory (Engram), identifying a U-shaped scaling law that guides optimal capacity allocation.
- **Empirical Verification:** Under strict iso-parameter and iso-FLOPs constraints, the Engram-27B model demonstrates consistent improvements over MoE baselines across knowledge, reasoning, code and math domains.
- **Mechanistic Analysis:** Our analysis suggests that Engram relieves early layers from static pattern reconstruction, potentially preserving effective depth for complex reasoning.
- **System Efficiency:** The module employs deterministic addressing, enabling the offloading of massive embedding tables to host memory with minimal inference overhead.


## 2. Architecture

The Engram module augments the backbone by retrieving static $N$-gram memory and fusing it with dynamic hidden states. The architecture is shown below ([drawio provided](drawio/Engram.drawio)):

<p align="center">
  <img width="75%" src="figures/arch.png" alt="Engram Architecture">
</p>

## 3. Evaluation

### Scaling Law
<p align="center">
  <img width="90%" src="figures/scaling_law.png" alt="Scaling Law">
</p>

---

### Large Scale Pre-training
<p align="center">
  <img width="80%" src="figures/27b_exp_results.png" alt="Pre-training Results">
</p>

---

### Long-context Training
<p align="center">
  <img width="80%" src="figures/long_context_results.png" alt="Long Context Results">
</p>


## 4. Case Study of Engram
<p align="center">
  <img width="80%" src="figures/case.png" alt="Long Context Results">
</p>

## 5. Quick Start

We recommend using Python 3.8+ and PyTorch.
```bash
pip install torch numpy transformers sympy
```
We provide a standalone implementation to demonstrate the core logic of the Engram module:
```bash
python engram_demo_v1.py
```

> ⚠️ **Note:** The provided code is a demonstration version intended to illustrate the data flow. It mocks standard components (like Attention/MoE/mHC) to focus on the Engram module. 


## 6. Engram AutoResearch: 自动循环验证实验框架

基于 [autoresearch](https://github.com/karpathy/autoresearch) 方法论，我们设计了一套针对 Engram 知识注入的自动循环验证框架，支持 AI 代理自主迭代优化 Engram 超参数和知识格式。

### 6.1 硬件要求与模型选型

| 硬件配置 | 推荐规格 |
|---------|----------|
| GPU | NVIDIA A10 (24GB VRAM) 或更高 |
| CPU | 16 核+ |
| 内存 | 60GB+ |

**推荐基础模型（冻结主干，仅训练 Engram 参数）**：

| 模型 | HuggingFace ID | bf16 显存 | 适用场景 |
|------|---------------|-----------|----------|
| Qwen3-0.6B | `Qwen/Qwen3-0.6B` | ~1.2 GB | 超快速原型验证 |
| Qwen3-1.7B | `Qwen/Qwen3-1.7B` | ~3.4 GB | 快速实验迭代（推荐） |
| Qwen3-4B | `Qwen/Qwen3-4B` | ~8 GB | 正式实验（推荐） |
| Qwen3-8B | `Qwen/Qwen3-8B` | ~16 GB | 高质量验证 |

> Engram 训练冻结主干模型，仅训练 Engram 参数（约 10-15M），显存开销主要来自冻结模型的推理。24GB A10 可舒适运行 Qwen3-4B 级别模型。

### 6.2 知识数据格式

定义了四种知识类型，覆盖不同的知识注入场景：

**Type A: 指令-命令映射 (command_mapping)**
```
## 指令: {action_description}
命令: {command_template}
参数: {param1}={description1}, {param2}={description2}
示例:
>>> {example_input}
{example_output}
```
适用: CLI 工具、API 调用、操作手册。N-gram 触发模式: "指令:" + 动作词。

**Type B: 事实-知识对 (fact_pair)**
```
问: {question}
答: {answer}

{entity} 是 {definition}。{entity} 的 {attribute} 是 {value}。
```
适用: 百科知识、实体属性、定义说明。N-gram 触发模式: 实体名 + "是"。

**Type C: 流程-步骤文档 (procedure)**
```
## {task_name}
步骤 1: {step_description}
  操作: {operation}
  预期: {expected_result}
步骤 2: {step_description}
  操作: {operation}
  前置: 步骤 1 完成
```
适用: 运维流程、排障指南。N-gram 触发模式: "步骤" + 数字。

**Type D: 结构化配置 (structured_config)**
```
{system_name}:
  类型: {type}
  地址: {address}
  端口: {port}
  依赖: [{dep1}, {dep2}]
```
适用: 基础设施配置、系统拓扑。N-gram 触发模式: 配置键名 + ":"。

> 每条知识生成 2-3 个改写变体用于数据增强，确保 N-gram 多角度覆盖。建议 command_mapping 占 40-50%，fact_pair 占 20-25%，procedure 占 15-20%，structured_config 占 10-15%。

### 6.3 实验框架

```
autoresearch/
├── prepare.py          # 评估函数 + 常量（只读）
├── train.py            # 训练脚本（唯一可修改的文件）
├── knowledge_format.py # 知识格式定义 + 数据增强
├── program.md          # 实验协议
├── run_experiment.sh   # 单次实验执行脚本
└── results.tsv         # 实验结果记录
```

**实验循环**（模仿 autoresearch 的自主迭代模式）：
1. 修改 `train.py` 中的超参数或结构
2. `git commit`
3. 运行: `./run_experiment.sh`（固定 10 分钟训练预算）
4. 评估: 关键词召回率 (recall_score) + 验证困惑度 (val_ppl)
5. recall_score 提升则 keep，否则 discard + git reset
6. 循环至收敛

**可调超参数**：
```python
ENGRAM_VOCAB_SIZE = [80000, 80000]   # 2-gram/3-gram 哈希表大小
N_EMBED_PER_NGRAM = 128              # 每个 ngram 嵌入维度
N_HEAD_PER_NGRAM = 4                 # 哈希头数量
KERNEL_SIZE = 4                      # 短卷积核大小
MAX_NGRAM_SIZE = 3                   # 最大 N-gram 阶数
ENGRAM_LAYERS = []                   # Engram 插入层（空=自动）
LEARNING_RATE = 1e-3                 # 学习率
BATCH_SIZE = 4                       # 批大小
SEQ_LEN = 256                        # 序列长度
```

**4 阶段搜索策略**：

| 阶段 | 内容 | 预计轮次 |
|------|------|----------|
| Phase 1 | 基线建立 + 学习率/层选择扫描 | ~10 轮 |
| Phase 2 | Engram 结构搜索（vocab/embed/head/kernel） | ~15 轮 |
| Phase 3 | 知识格式优化（文本类型/增强/长度） | ~10 轮 |
| Phase 4 | 升级到更大模型验证迁移性 | ~5 轮 |

### 6.4 快速开始

```bash
cd autoresearch/

# 安装依赖
pip install torch numpy transformers sympy

# 运行单次实验
./run_experiment.sh

# 或手动运行
python train.py > run.log 2>&1
grep "^recall_score:\|^val_ppl:\|^peak_vram_mb:" run.log
```

## 7. Local Knowledge Injection Demo

冻结预训练主干（Qwen/Llama/...），在指定层插入 Engram 模块，只训练 Engram 参数（约 10-15M），实现高效知识注入：

```bash
# 用内置示例知识训练
python engram_local_demo.py --epochs 20

# 用本地 txt/md 文件训练
python engram_local_demo.py --data_dir ./my_texts --epochs 20

# 换用更大模型
python engram_local_demo.py --model Qwen/Qwen3-1.7B --epochs 20

# 对比无 Engram 的原始模型
python engram_local_demo.py --no_engram
```

## 8. License
The use of Engram models is subject to [the Model License](LICENSE).

## 9. Contact

If you have any questions, please raise an issue or contact us at [service@deepseek.com](mailto:service@deepseek.com).