"""
Engram AutoResearch - 训练脚本
==============================
这是实验循环中唯一可修改的文件。
修改超参数和结构后运行: python train.py

用法: python train.py > run.log 2>&1

依赖: pip install torch transformers sympy numpy
"""

import os
import sys
import math
import time
import warnings

# 抑制 NumPy 兼容性警告
warnings.filterwarnings("ignore", message=".*NumPy.*")
os.environ.setdefault("NUMPY_EXPERIMENTAL_DTYPE_API", "1")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# ---------------------------------------------------------------------------
# 路径设置：添加父目录以导入核心模块
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从核心模块导入
from engram_local_demo import (
    EngramConfig, CompressedTokenizer, NgramHashMapping,
    MultiHeadEmbedding, ShortConv, Engram, EngramLM, TextDataset,
    _detect_device, _select_dtype
)

# 从本地模块导入
from prepare import (
    TIME_BUDGET, MAX_SEQ_LEN, BASE_MODEL,
    evaluate_recall, evaluate_ppl, setup_hf_mirror
)
from knowledge_format import (
    RECALL_TESTS, build_training_text, build_validation_text
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Engram architecture
ENGRAM_VOCAB_SIZE = [80000, 80000]   # 2-gram/3-gram 哈希表大小
N_EMBED_PER_NGRAM = 128              # 每个 ngram 的嵌入维度
N_HEAD_PER_NGRAM = 4                 # 哈希头数量
KERNEL_SIZE = 4                      # 短卷积核大小
MAX_NGRAM_SIZE = 3                   # 最大 N-gram 阶数
ENGRAM_LAYERS = []                   # 空=自动选择, 或指定如 [4, 12]

# Optimization
LEARNING_RATE = 1e-3                 # Engram 参数学习率
BATCH_SIZE = 4                       # 训练 batch size
SEQ_LEN = 256                        # 训练序列长度（不超过 MAX_SEQ_LEN）
WEIGHT_DECAY = 0.01                  # AdamW weight decay

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()

# 设置 HF 镜像
setup_hf_mirror()

# 设备和 dtype
device = _detect_device()
dtype = _select_dtype(device.type)

print("=" * 60)
print("Engram AutoResearch Training")
print("=" * 60)
print(f"Device:           {device}")
print(f"Dtype:            {dtype}")
print(f"Time budget:      {TIME_BUDGET}s")
print(f"Base model:       {BASE_MODEL}")
print()

# ---------------------------------------------------------------------------
# Print hyperparameters
# ---------------------------------------------------------------------------

print("Hyperparameters:")
print(f"  ENGRAM_VOCAB_SIZE:  {ENGRAM_VOCAB_SIZE}")
print(f"  N_EMBED_PER_NGRAM:  {N_EMBED_PER_NGRAM}")
print(f"  N_HEAD_PER_NGRAM:   {N_HEAD_PER_NGRAM}")
print(f"  KERNEL_SIZE:        {KERNEL_SIZE}")
print(f"  MAX_NGRAM_SIZE:     {MAX_NGRAM_SIZE}")
print(f"  ENGRAM_LAYERS:      {ENGRAM_LAYERS if ENGRAM_LAYERS else 'auto'}")
print(f"  LEARNING_RATE:      {LEARNING_RATE}")
print(f"  BATCH_SIZE:         {BATCH_SIZE}")
print(f"  SEQ_LEN:            {SEQ_LEN}")
print(f"  WEIGHT_DECAY:       {WEIGHT_DECAY}")
print()

# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

ecfg = EngramConfig(
    engram_vocab_size=ENGRAM_VOCAB_SIZE,
    max_ngram_size=MAX_NGRAM_SIZE,
    n_embed_per_ngram=N_EMBED_PER_NGRAM,
    n_head_per_ngram=N_HEAD_PER_NGRAM,
    kernel_size=KERNEL_SIZE,
    engram_layer_ids=ENGRAM_LAYERS.copy() if ENGRAM_LAYERS else [],
)

model = EngramLM(BASE_MODEL, ecfg, device=device, dtype=dtype)
model.to(device)

# 统计 Engram 参数量
n_engram = sum(p.numel() for p in model.engrams.parameters())
n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nEngram parameters: {n_engram / 1e6:.2f}M")
print(f"Trainable params:  {n_trainable / 1e6:.2f}M")

# ---------------------------------------------------------------------------
# Prepare data
# ---------------------------------------------------------------------------

print("\nPreparing training data...")
text = build_training_text()

# 确保 SEQ_LEN 不超过 model.max_seq_len 和 MAX_SEQ_LEN
seq_len = min(SEQ_LEN, model.max_seq_len, MAX_SEQ_LEN)
if seq_len != SEQ_LEN:
    print(f"  Adjusted SEQ_LEN from {SEQ_LEN} to {seq_len}")

tokens = model.tokenizer.encode(text)
print(f"  Corpus: {len(text):,} chars → {len(tokens):,} tokens")

# 处理文本太短需要重复的情况
if len(tokens) <= seq_len:
    print(f"  [WARN] Text too short ({len(tokens)} tokens), repeating to fill...")
    while len(tokens) <= seq_len:
        tokens = tokens + tokens
    print(f"  Expanded to {len(tokens):,} tokens")

dataset = TextDataset(tokens, seq_len)
print(f"  Training samples: {len(dataset):,} (seq_len={seq_len})")

# ---------------------------------------------------------------------------
# Setup optimizer and scheduler
# ---------------------------------------------------------------------------

params = [p for p in model.parameters() if p.requires_grad]
if not params:
    print("\nNo trainable parameters! Exiting.")
    sys.exit(1)

optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# 估算 total_steps（基于时间预算）
# 假设每个 epoch 大约需要的时间，先用一个初始估计
estimated_steps = max(1000, len(dataset) // BATCH_SIZE * 10)  # 粗略估计
scheduler = CosineAnnealingLR(optimizer, T_max=estimated_steps)

# ---------------------------------------------------------------------------
# Training loop (time-budgeted)
# ---------------------------------------------------------------------------

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

print(f"\n{'=' * 60}")
print(f"  Training: time budget = {TIME_BUDGET}s, bs={BATCH_SIZE}")
print(f"  Batches per epoch: {len(loader)}")
print(f"{'=' * 60}\n")

model.train()
# 保持 base model 在 eval 模式（避免 dropout 等影响）
model.base.eval()

t_start_training = time.time()
total_training_time = 0
total_steps = 0
epoch = 0

while True:
    epoch += 1
    total_loss, num_batches = 0.0, 0
    
    for batch in loader:
        t_step_start = time.time()
        
        batch = batch.to(device)
        _, loss = model(batch, labels=batch)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        total_steps += 1
        
        # 同步并计算时间
        if device.type == 'cuda':
            torch.cuda.synchronize()
        dt = time.time() - t_step_start
        total_training_time += dt
        
        # 检查时间预算
        if total_training_time >= TIME_BUDGET:
            break
    
    # 计算 epoch 统计
    avg_loss = total_loss / max(num_batches, 1)
    lr = scheduler.get_last_lr()[0]
    elapsed = total_training_time
    
    # 门控统计
    gs = "  ".join(f"L{k}={v:.4f}" for k, v in model.gate_stats().items())
    
    print(f"  Epoch {epoch:3d} | Loss {avg_loss:.4f} | LR {lr:.1e} | Gate[{gs}] | {elapsed:.0f}s/{TIME_BUDGET}s")
    
    # 检查时间预算
    if total_training_time >= TIME_BUDGET:
        print(f"\n  Time budget reached ({TIME_BUDGET}s)")
        break

print(f"\nTraining completed in {total_training_time:.1f}s ({total_steps} steps)")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

print("\nEvaluating...")

model.eval()

# 召回评估
print("  Computing recall score...")
recall_score = evaluate_recall(model, device, RECALL_TESTS)

# PPL 评估
print("  Computing validation PPL...")
val_text = build_validation_text()
val_ppl = evaluate_ppl(model, device, val_text)

# 获取 peak VRAM
if device.type == 'cuda':
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
else:
    peak_vram_mb = 0.0

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

total_time = time.time() - t_start

print("---")
print(f"recall_score:     {recall_score:.6f}")
print(f"val_ppl:          {val_ppl:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {total_time:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"engram_params_M:  {n_engram / 1e6:.1f}")
print(f"base_model:       {BASE_MODEL}")
print(f"total_steps:      {total_steps}")
print(f"final_loss:       {avg_loss:.6f}")
