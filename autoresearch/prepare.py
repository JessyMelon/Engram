"""
Engram AutoResearch - 数据准备与评估模块（只读）
=================================================
此文件包含固定的评估函数和常量。
在实验循环中，只修改 train.py，不修改此文件。

依赖: pip install torch transformers
"""

import os
import math
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------
TIME_BUDGET = 600          # 训练时间预算：10 分钟
MAX_SEQ_LEN = 512          # 最大训练序列长度
BASE_MODEL = "Qwen/Qwen2.5-1.5B"  # 快速迭代阶段的基础模型

# ---------------------------------------------------------------------------
# HuggingFace mirror setup
# ---------------------------------------------------------------------------

def setup_hf_mirror():
    """若未设置 HF_ENDPOINT 且直连不通，自动切换国内镜像"""
    endpoint = os.environ.get("HF_ENDPOINT", "")
    if endpoint:
        print(f"HF mirror: {endpoint}")
        return
    # 尝试探测 huggingface.co 是否可达
    try:
        import urllib.request
        urllib.request.urlopen("https://huggingface.co", timeout=5)
    except Exception:
        mirror = "https://hf-mirror.com"
        os.environ["HF_ENDPOINT"] = mirror
        print(f"[Auto] HuggingFace unreachable, using mirror: {mirror}")


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE - these are fixed metrics)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_recall(model, device, recall_tests) -> float:
    """
    知识召回评估。
    
    对每个 recall_test：
    1. 将 prompt encode 为 token ids
    2. 用 model.generate() 生成回复（max_new=100, temperature=0.3, top_k=10）
    3. decode 生成文本
    4. 检查 must_contain 关键词是否出现在生成文本中
    5. 可选检查 must_not_contain
    6. 计算每条测试的得分：命中关键词数 / 总关键词数
    7. 返回所有测试的平均得分 (0.0 ~ 1.0)
    
    Args:
        model: EngramLM 实例，需要有 .tokenizer 和 .generate() 方法
        device: torch device
        recall_tests: list of dict，每个 dict 包含 prompt/must_contain/must_not_contain/score_method
    
    Returns:
        float: 平均召回得分 0.0 ~ 1.0
    """
    if not recall_tests:
        return 0.0
    
    model.eval()
    scores = []
    
    for test in recall_tests:
        try:
            prompt = test.get("prompt", "")
            must_contain = test.get("must_contain", [])
            must_not_contain = test.get("must_not_contain", [])
            
            if not prompt:
                scores.append(0.0)
                continue
            
            # 1. encode prompt
            prompt_ids = model.tokenizer.encode(prompt)
            if not prompt_ids:
                scores.append(0.0)
                continue
            
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            
            # 2. generate response
            try:
                output_ids = model.generate(
                    prompt_tensor,
                    max_new=100,
                    temperature=0.3,
                    top_k=10
                )
            except Exception:
                scores.append(0.0)
                continue
            
            # 3. decode generated text
            if output_ids is None or output_ids.numel() == 0:
                scores.append(0.0)
                continue
            
            generated_text = model.tokenizer.decode(
                output_ids[0].tolist(),
                skip_special_tokens=True
            )
            
            # 获取生成的部分（去掉 prompt）
            if len(generated_text) > len(prompt):
                response = generated_text[len(prompt):]
            else:
                response = generated_text
            
            response_lower = response.lower()
            
            # 4. check must_contain keywords (case insensitive)
            if not must_contain:
                # 无需检查，满分
                test_score = 1.0
            else:
                hits = 0
                for keyword in must_contain:
                    if keyword.lower() in response_lower:
                        hits += 1
                test_score = hits / len(must_contain)
            
            # 5. check must_not_contain (if any keyword appears, halve the score)
            if must_not_contain:
                for bad_word in must_not_contain:
                    if bad_word.lower() in response_lower:
                        test_score *= 0.5
                        break  # 只减半一次
            
            scores.append(test_score)
            
        except Exception:
            # 任何异常都给 0 分，确保健壮性
            scores.append(0.0)
    
    # 7. 返回平均得分
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


@torch.no_grad()
def evaluate_ppl(model, device, val_text) -> float:
    """
    验证集困惑度评估。
    
    将 val_text tokenize 后，用滑窗方式计算困惑度。
    
    Args:
        model: EngramLM 实例
        device: torch device
        val_text: str，验证文本
    
    Returns:
        float: 困惑度值
    """
    if not val_text or not val_text.strip():
        return float("inf")
    
    model.eval()
    
    try:
        # Tokenize validation text
        tokens = model.tokenizer.encode(val_text)
        if not tokens or len(tokens) < 2:
            return float("inf")
        
        tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
        
        # 滑窗计算，stride = MAX_SEQ_LEN // 2
        stride = MAX_SEQ_LEN // 2
        seq_len = MAX_SEQ_LEN
        
        total_loss = 0.0
        total_tokens = 0
        
        # 滑窗遍历
        for start in range(0, len(tokens) - 1, stride):
            end = min(start + seq_len, len(tokens))
            
            # 获取窗口内的 tokens
            window_tokens = tokens_tensor[start:end]
            
            if len(window_tokens) < 2:
                continue
            
            # 准备输入和目标
            input_ids = window_tokens[:-1].unsqueeze(0)  # [1, T-1]
            target_ids = window_tokens[1:].unsqueeze(0)   # [1, T-1]
            
            # 调用模型获取 logits
            # EngramLM.forward 返回 (logits, loss)
            # 这里我们需要手动计算 loss 以便精确控制
            logits, _ = model(input_ids)
            
            # logits shape: [1, T-1, vocab_size]
            # target_ids shape: [1, T-1]
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = target_ids.view(-1)
            
            # 计算 cross-entropy loss (per token)
            loss = F.cross_entropy(logits_flat, targets_flat, reduction="sum")
            
            # 计算实际贡献的 token 数
            # 对于重叠部分，只计算非重叠部分（第一个窗口除外）
            if start == 0:
                num_tokens = len(window_tokens) - 1
            else:
                # 只计算 stride 范围内的 loss，避免重复计算
                overlap_start = max(0, seq_len - stride - 1)
                num_tokens = end - start - 1 - overlap_start
                if num_tokens <= 0:
                    continue
                # 重新计算非重叠部分的 loss
                logits_non_overlap = logits[0, overlap_start:, :]
                targets_non_overlap = target_ids[0, overlap_start:]
                if logits_non_overlap.size(0) == 0:
                    continue
                loss = F.cross_entropy(
                    logits_non_overlap.view(-1, logits_non_overlap.size(-1)),
                    targets_non_overlap.view(-1),
                    reduction="sum"
                )
            
            total_loss += loss.item()
            total_tokens += num_tokens
            
            # 如果已经到达末尾，退出
            if end >= len(tokens):
                break
        
        if total_tokens == 0:
            return float("inf")
        
        # PPL = exp(average_loss)
        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)
        
        return ppl
        
    except Exception:
        return float("inf")
