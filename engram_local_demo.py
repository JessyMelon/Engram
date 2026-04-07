"""
=======================================================================
Engram × Pretrained LLM — 基于预训练模型的本地知识注入 Demo
=======================================================================
冻结预训练主干（Qwen / Llama / …），在指定层插入 Engram 模块，
只训练 Engram 参数（~10-15M），实现高效知识注入。

自动适配: MPS (Mac M-series) / CUDA (Linux GPU) / CPU
自动选型: bf16 / fp16 / fp32

用法:
  # ① 用内置示例知识训练（开箱即用）
  python engram_local_demo.py --epochs 20

  # ② 用本地 txt/md 文件训练
  python engram_local_demo.py --data_dir ./my_texts --epochs 20

  # ③ 换用更大模型（需要更多显存/内存）
  python engram_local_demo.py --model Qwen/Qwen2.5-1.5B --epochs 20

  # ④ 加载已保存的 Engram 权重
  python engram_local_demo.py --load_engram ./engram_weights.pt

  # ⑤ 对比无 Engram 的原始模型
  python engram_local_demo.py --no_engram

依赖: pip install torch transformers sympy numpy
=======================================================================
"""

# ── Imports ──────────────────────────────────────────────────────────
import os, glob, math, time, argparse, inspect, warnings, importlib.util, sys
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

# 抑制 NumPy 1.x/2.x 兼容性警告
warnings.filterwarnings("ignore", message=".*NumPy.*")
os.environ.setdefault("NUMPY_EXPERIMENTAL_DTYPE_API", "1")  # 缓解部分兼容性问题

# PyTorch CUDA 显存管理优化
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sympy import isprime
from transformers import AutoTokenizer, AutoModelForCausalLM
from tokenizers import normalizers, Regex

from knowledge_data import EXAMPLE_KNOWLEDGE, RECALL_PROMPTS


def _setup_hf_mirror():
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Engram 配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class EngramConfig:
    engram_vocab_size: List[int] = field(default_factory=lambda: [80000, 80000])
    max_ngram_size: int = 3  # 2-gram + 3-gram
    n_embed_per_ngram: int = 128
    n_head_per_ngram: int = 4  # 每种 ngram 4 个哈希头
    kernel_size: int = 4
    seed: int = 42
    engram_layer_ids: List[int] = field(default_factory=list)  # 空=自动选择


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. CompressedTokenizer — Token 归一化（The/the/THE → 同一 ID）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CompressedTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        S = "\uE000"
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(), normalizers.NFD(),
            normalizers.StripAccents(), normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), S),
            normalizers.Strip(), normalizers.Replace(S, " "),
        ])
        self.lookup, self.size = self._build()

    def __len__(self):
        return self.size

    def _build(self):
        o2n, k2n, new = {}, {}, []
        for tid in range(len(self.tokenizer)):
            txt = self.tokenizer.decode([tid], skip_special_tokens=False)
            k = (self.tokenizer.convert_ids_to_tokens(tid) if "\ufffd" in txt
                 else (self.normalizer.normalize_str(txt) or txt))
            nid = k2n.get(k)
            if nid is None:
                nid = len(new);
                k2n[k] = nid;
                new.append(k)
            o2n[tid] = nid
        return np.array([o2n[i] for i in range(len(self.tokenizer))], np.int64), len(new)

    def __call__(self, ids):
        a = np.asarray(ids, np.int64);
        o = a.copy();
        m = a >= 0
        o[m] = self.lookup[a[m]];
        return o


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. NgramHashMapping — O(1) 确定性哈希
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _next_prime(s, seen):
    c = s + 1
    while True:
        if isprime(c) and c not in seen: return c
        c += 1


class NgramHashMapping:
    def __init__(self, cfg: EngramConfig, ctok: CompressedTokenizer, pad_id: int):
        self.cfg, self.ctok = cfg, ctok
        self.pad_id = int(ctok.lookup[pad_id]) if pad_id < len(ctok.lookup) else 0
        vs = len(ctok)
        half = max(1, int(np.iinfo(np.int64).max // vs) // 2)
        self.mults = {}
        for lid in cfg.engram_layer_ids:
            g = np.random.default_rng(cfg.seed + 10007 * lid)
            self.mults[lid] = g.integers(0, half, (cfg.max_ngram_size,), np.int64) * 2 + 1
        self.vocab_sizes = self._primes()

    def _primes(self):
        seen, res = set(), {}
        for lid in self.cfg.engram_layer_ids:
            layer = []
            for n in range(2, self.cfg.max_ngram_size + 1):
                heads, s = [], self.cfg.engram_vocab_size[n - 2] - 1
                for _ in range(self.cfg.n_head_per_ngram):
                    p = _next_prime(s, seen);
                    seen.add(p);
                    heads.append(p);
                    s = p
                layer.append(heads)
            res[lid] = layer
        return res

    def hash(self, ids_np, layer_id):
        x = self.ctok(ids_np);
        B, T = x.shape
        ms = self.mults[layer_id]

        def shift(k):
            return x if k == 0 else np.pad(x, ((0, 0), (k, 0)), constant_values=self.pad_id)[:, :T]

        shifts = [shift(k) for k in range(self.cfg.max_ngram_size)]
        hashes = []
        for n in range(2, self.cfg.max_ngram_size + 1):
            mix = shifts[0] * ms[0]
            for k in range(1, n):
                mix = np.bitwise_xor(mix, shifts[k] * ms[k])
            for mod in self.vocab_sizes[layer_id][n - 2]:
                hashes.append((mix % mod).astype(np.int64))
        return np.stack(hashes, axis=2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Engram 核心组件
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MultiHeadEmbedding(nn.Module):
    def __init__(self, sizes: List[int], dim: int):
        super().__init__()
        off = [0]
        for s in sizes[:-1]: off.append(off[-1] + s)
        self.register_buffer("offsets", torch.tensor(off, dtype=torch.long))
        self.emb = nn.Embedding(sum(sizes), dim)

    def forward(self, ids):
        return self.emb(ids + self.offsets)


class ShortConv(nn.Module):
    def __init__(self, dim, ks=4, dil=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, ks, groups=dim, bias=False,
                              padding=(ks - 1) * dil, dilation=dil)
        self.act = nn.SiLU()

    def forward(self, x):
        T = x.size(1)
        return self.act(self.conv(self.norm(x).transpose(1, 2))[..., :T]).transpose(1, 2)


class Engram(nn.Module):
    """hash → multi-head embedding → gated injection + short conv"""

    def __init__(self, layer_id, cfg: EngramConfig, hmap: NgramHashMapping, hidden_size: int):
        super().__init__()
        self.layer_id, self.hmap = layer_id, hmap
        sizes = [s for ng in hmap.vocab_sizes[layer_id] for s in ng]
        hd = cfg.n_embed_per_ngram // cfg.n_head_per_ngram
        self.mh_emb = MultiHeadEmbedding(sizes, hd)
        edim = (cfg.max_ngram_size - 1) * cfg.n_embed_per_ngram
        self.val_proj = nn.Linear(edim, hidden_size)
        self.key_proj = nn.Linear(edim, hidden_size)
        self.nk = nn.LayerNorm(hidden_size)
        self.nq = nn.LayerNorm(hidden_size)
        self.conv = ShortConv(hidden_size, cfg.kernel_size, cfg.max_ngram_size)
        self.last_gate = 0.0

    def forward(self, h, ids_np):
        hids = torch.tensor(self.hmap.hash(ids_np, self.layer_id), dtype=torch.long, device=h.device)
        e = self.mh_emb(hids).flatten(-2)
        g = (self.nk(self.key_proj(e)) * self.nq(h)).sum(-1) / math.sqrt(h.size(-1))
        g = g.abs().clamp_min(1e-6).sqrt() * g.sign()
        g = g.sigmoid().unsqueeze(-1)
        self.last_gate = g.mean().item()
        v = g * self.val_proj(e)
        return v + self.conv(v)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. EngramLM — 预训练主干 + Engram 注入
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _select_dtype(device_type: str):
    """根据设备选择最佳 dtype"""
    if device_type == "cuda":
        return torch.bfloat16
    # MPS fp16 存在 mps.select 类型不匹配 bug，用 fp32 更稳定
    return torch.float32


def _detect_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_config_attr(config, attr_name):
    """Get config attribute, handling nested configs (e.g., Gemma4's text_config).
    
    多模态模型（如 Gemma 4）的配置结构可能不同于纯文本模型：
    - 标准文本模型: config.hidden_size
    - 多模态模型: config.text_config.hidden_size 或 config.llm_config.hidden_size
    """
    # 直接在顶层查找
    if hasattr(config, attr_name):
        val = getattr(config, attr_name)
        # 确保不是 None（有些模型顶层有属性但值为 None）
        if val is not None:
            return val
    # 在 text_config 子配置中查找（多模态模型如 Gemma4）
    if hasattr(config, 'text_config') and hasattr(config.text_config, attr_name):
        val = getattr(config.text_config, attr_name)
        if val is not None:
            return val
    # 在 llm_config 子配置中查找
    if hasattr(config, 'llm_config') and hasattr(config.llm_config, attr_name):
        val = getattr(config.llm_config, attr_name)
        if val is not None:
            return val
    # 处理属性名映射（如 d_model -> hidden_size）
    attr_aliases = {
        'hidden_size': ['d_model', 'model_dim', 'embed_dim'],
        'num_hidden_layers': ['n_layer', 'num_layers', 'n_layers'],
        'num_attention_heads': ['n_head', 'num_heads', 'n_heads'],
    }
    if attr_name in attr_aliases:
        for alias in attr_aliases[attr_name]:
            if hasattr(config, alias):
                val = getattr(config, alias)
                if val is not None:
                    return val
            if hasattr(config, 'text_config') and hasattr(config.text_config, alias):
                val = getattr(config.text_config, alias)
                if val is not None:
                    return val
    raise AttributeError(f"Cannot find '{attr_name}' in config or its sub-configs: {config}")


def _find_transformer_layers(model):
    """Find transformer decoder layers in various model architectures.
    
    支持多种模型架构：
    - 标准 HF 模型 (Qwen, Llama, Mistral): model.model.layers
    - GPT-2 / GPT-Neo 系列: model.transformer.h
    - 多模态模型 (Gemma 4 ForConditionalGeneration): model.model.language_model.layers
    - 多模态模型 (旧版): model.language_model.model.layers
    - 其他可能的路径
    """
    # 常见路径列表，按优先级排序
    candidates = [
        # 标准 HF 模型 (Qwen, Llama, Mistral, Gemma4ForCausalLM)
        lambda m: m.model.layers,
        # GPT-2 / GPT-Neo 系列
        lambda m: m.transformer.h,
        # 多模态模型 (Gemma4ForConditionalGeneration) - model.language_model.layers
        lambda m: m.model.language_model.layers,
        # 多模态模型 (旧版结构) - language_model.model.layers
        lambda m: m.language_model.model.layers,
        # 其他可能的路径
        lambda m: m.model.decoder.layers,
        lambda m: m.model.model.layers,
        lambda m: m.decoder.layers,
    ]
    for fn in candidates:
        try:
            layers = fn(model)
            if layers is not None and len(layers) > 0:
                return layers
        except (AttributeError, TypeError):
            continue
    return None


def _find_backbone_component(model, component_name):
    """Find backbone component (embed_tokens, norm, lm_head, rotary_emb) in various architectures.
    
    Args:
        model: 模型实例
        component_name: 组件名称，支持 'embed_tokens', 'norm', 'lm_head', 'rotary_emb'
    """
    # 定义每个组件可能的属性名和路径
    component_paths = {
        'embed_tokens': [
            # 标准 HF 模型 (Qwen, Llama, Mistral, Gemma4ForCausalLM)
            lambda m: m.model.embed_tokens,
            # GPT-2 系列
            lambda m: m.transformer.wte,
            # 多模态模型 (Gemma4ForConditionalGeneration)
            lambda m: m.model.language_model.embed_tokens,
            # 多模态模型 (旧版结构)
            lambda m: m.language_model.model.embed_tokens,
            lambda m: m.model.model.embed_tokens,
        ],
        'norm': [
            # 标准 HF 模型 (Qwen, Llama, Mistral, Gemma4ForCausalLM)
            lambda m: m.model.norm,
            # GPT-2 系列
            lambda m: m.transformer.ln_f,
            # 多模态模型 (Gemma4ForConditionalGeneration)
            lambda m: m.model.language_model.norm,
            # 多模态模型 (旧版结构)
            lambda m: m.language_model.model.norm,
            lambda m: m.model.model.norm,
        ],
        'lm_head': [
            lambda m: m.lm_head,
            # 多模态模型 (Gemma4ForConditionalGeneration)
            lambda m: m.model.lm_head,
            lambda m: m.language_model.lm_head,
        ],
        'rotary_emb': [
            # 标准 HF 模型
            lambda m: m.model.rotary_emb,
            # 多模态模型 (Gemma4ForConditionalGeneration)
            lambda m: m.model.language_model.rotary_emb,
            # 多模态模型 (旧版结构)
            lambda m: m.language_model.model.rotary_emb,
            lambda m: m.model.model.rotary_emb,
        ],
    }
    
    paths = component_paths.get(component_name, [])
    for fn in paths:
        try:
            component = fn(model)
            if component is not None:
                return component
        except (AttributeError, TypeError):
            continue
    return None


def _setup_tensor_parallel(gpu_devices_str: str):
    """设置多GPU环境，返回 (主设备, 设备列表)"""
    if not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to single device")
        return _detect_device(), [_detect_device()]
    
    device_ids = [int(d.strip()) for d in gpu_devices_str.split(",") if d.strip()]
    n_gpu = torch.cuda.device_count()
    valid_ids = [did for did in device_ids if 0 <= did < n_gpu]

    if not valid_ids:
        print(f"[WARN] No requested GPUs {device_ids} available (found {n_gpu} GPUs). "
              f"Falling back to cuda:0.")
        valid_ids = [0]
    elif len(valid_ids) < len(device_ids):
        dropped = [d for d in device_ids if d not in valid_ids]
        print(f"[WARN] GPU(s) {dropped} not available (found {n_gpu} GPUs). "
              f"Using {valid_ids} only.")

    devices = [torch.device(f"cuda:{did}") for did in valid_ids]
    primary_device = devices[0]
    
    print(f"[Setup] Tensor Parallel: {len(devices)} GPUs = {device_ids}")
    for dev in devices:
        total_mem = torch.cuda.get_device_properties(dev.index).total_memory / 1e9
        print(f"  {dev}: {total_mem:.1f} GB")
    
    return primary_device, devices


class EngramLM(nn.Module):
    def __init__(self, model_name: str, ecfg: EngramConfig,
                 use_engram=True, device=None, dtype=None,
                 tensor_parallel_devices=None, engram_device_strategy="distribute"):
        super().__init__()

        if device is None:
            device = _detect_device()
        if dtype is None:
            dtype = _select_dtype(device.type)

        self.device_target = device
        self.dtype = dtype
        self.tensor_parallel_devices = tensor_parallel_devices or [device]
        self.engram_device_strategy = engram_device_strategy
        self.n_devices = len(self.tensor_parallel_devices)

        # ── 加载预训练模型 ──
        print(f"[1/4] Loading {model_name} ({dtype})...")
        if self.n_devices > 1:
            try:
                self.base = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=dtype, trust_remote_code=True,
                    device_map="auto",
                    attn_implementation="flash_attention_2",
                )
                print(f"  Using Flash Attention 2.")
            except Exception:
                self.base = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=dtype, trust_remote_code=True,
                    device_map="auto",
                )
                print(f"  Using standard attention (Flash Attention 2 not available).")
            print(f"  Using device_map='auto' across {self.n_devices} GPUs")
        else:
            try:
                self.base = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=dtype, trust_remote_code=True,
                    attn_implementation="flash_attention_2",
                )
                print(f"  Using Flash Attention 2.")
            except Exception:
                self.base = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=dtype, trust_remote_code=True,
                )
                print(f"  Using standard attention (Flash Attention 2 not available).")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── 启用梯度检查点以节省显存 ──
        if hasattr(self.base, 'gradient_checkpointing_enable'):
            self.base.gradient_checkpointing_enable()
            print("  Gradient checkpointing enabled.")

        # ── 冻结主干 ──
        print("[2/4] Freezing backbone...")
        for p in self.base.parameters():
            p.requires_grad = False
        self.base.eval()

        # ── 探测模型内部结构（支持多种架构，包括多模态模型如 Gemma 4）──
        self._layers = _find_transformer_layers(self.base)
        if self._layers is None:
            raise ValueError(
                f"Cannot find transformer layers in {type(self.base).__name__}. "
                f"Supported architectures: model.model.layers, model.transformer.h, "
                f"model.language_model.model.layers, etc. "
                f"Please check model architecture and update _find_transformer_layers()."
            )
        self._embed = _find_backbone_component(self.base, 'embed_tokens')
        self._norm = _find_backbone_component(self.base, 'norm')
        self._lm_head = _find_backbone_component(self.base, 'lm_head')
        self._rotary = _find_backbone_component(self.base, 'rotary_emb')

        # 使用辅助函数获取配置属性，兼容多模态模型（如 Gemma4）
        hs = _get_config_attr(self.base.config, 'hidden_size')
        nl = _get_config_attr(self.base.config, 'num_hidden_layers')
        self.max_seq_len = min(
            getattr(self.base.config, "max_position_embeddings", 
                    getattr(getattr(self.base.config, 'text_config', None), 
                            'max_position_embeddings', 2048) if hasattr(self.base.config, 'text_config') else 2048), 
            1024
        )

        # 探测 decoder layer 接受的参数
        sig = inspect.signature(self._layers[0].forward)
        self._lp = set(sig.parameters.keys())

        # ── 自动选择 Engram 插入层 ──
        if not ecfg.engram_layer_ids:
            ecfg.engram_layer_ids = [max(1, nl // 6), nl // 2]
        ecfg.engram_layer_ids = [l for l in ecfg.engram_layer_ids if l < nl]

        # ── 构建 Engram ──
        self.engrams = nn.ModuleDict()
        if use_engram:
            print(f"[3/4] Building Engram (layers {ecfg.engram_layer_ids}, "
                  f"hidden={hs})...")
            pad_id = self.tokenizer.pad_token_id or 0
            ctok = CompressedTokenizer(self.tokenizer)
            hmap = NgramHashMapping(ecfg, ctok, pad_id)
            for lid in ecfg.engram_layer_ids:
                self.engrams[str(lid)] = Engram(lid, ecfg, hmap, hs)
            self.engrams = self.engrams.to(dtype=dtype)
            if self.n_devices > 1:
                self._allocate_engram_devices()
        else:
            print("[3/4] Engram disabled (baseline mode)")

        # ── 统计 ──
        n_base = sum(p.numel() for p in self.base.parameters())
        n_eng = sum(p.numel() for p in self.engrams.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[4/4] Ready!")
        print(f"  Base model   : {n_base / 1e6:.0f}M (frozen)")
        print(f"  Engram       : {n_eng / 1e6:.1f}M (trainable)")
        print(f"  Trainable    : {n_train / 1e6:.1f}M / {(n_base + n_eng) / 1e6:.0f}M "
              f"({n_train / (n_base + n_eng) * 100:.2f}%)")
        print(f"  Device={device}  dtype={dtype}")
        mem_gb = (n_base * (2 if dtype != torch.float32 else 4)
                  + n_eng * 4 * 3) / 1e9  # 粗估: model + grad + opt
        print(f"  Est. memory  : ~{mem_gb:.1f} GB")

    def _allocate_engram_devices(self):
        """根据策略分配 Engram 模块到多张 GPU
        
        注意：这只是初始放置的 hint。由于 HF device_map='auto' 可能把 backbone 层
        放到不同 GPU，forward 中的 _make_hook 会在运行时动态迁移 Engram 到与隐藏态
        相同的设备，作为最终的安全网。
        """
        strategy = self.engram_device_strategy
        layer_ids = sorted([int(k) for k in self.engrams.keys()])
        
        if strategy == "distribute":
            for idx, lid in enumerate(layer_ids):
                target = self.tensor_parallel_devices[idx % self.n_devices]
                self.engrams[str(lid)].to(target)
                print(f"  Engram[L{lid}] -> {target}")
        elif strategy == "gpu0":
            target = self.tensor_parallel_devices[0]
            for lid in layer_ids:
                self.engrams[str(lid)].to(target)
            print(f"  All Engrams -> {target}")
        elif strategy == "gpu1":
            target = self.tensor_parallel_devices[min(1, self.n_devices - 1)]
            for lid in layer_ids:
                self.engrams[str(lid)].to(target)
            print(f"  All Engrams -> {target}")

    # ── Forward ──────────────────────────────────────────────────────

    def forward(self, input_ids, labels=None):
        ids_np = np.array(input_ids.cpu().tolist(), dtype=np.int64)

        # 用 hook 注入 Engram，让模型走自己的 forward 处理 RoPE / Attention
        hooks = []
        for lid_str, engram in self.engrams.items():
            layer = self._layers[int(lid_str)]
            def _make_hook(eng):
                def _hook(module, args):
                    h = args[0]
                    # 使用通用方式获取 Engram 模块的当前设备
                    try:
                        eng_device = next(eng.parameters()).device
                        if eng_device != h.device:
                            eng.to(h.device)
                    except StopIteration:
                        pass
                    engram_out = eng(h, ids_np)
                    engram_out = engram_out.to(h.device)
                    return (h + engram_out,) + args[1:]
                return _hook
            hooks.append(layer.register_forward_pre_hook(_make_hook(engram)))

        # 多卡时确保输入在正确设备
        if self.n_devices > 1:
            input_ids = input_ids.to(self.tensor_parallel_devices[0])

        try:
            # 不传 labels，只获取 logits（避免在 GPU 1 上计算 cross_entropy 导致 OOM）
            outputs = self.base(input_ids=input_ids, use_cache=False)
            logits = outputs.logits
        finally:
            for h in hooks:
                h.remove()

        loss = None
        if labels is not None:
            # 将 logits 和 labels 移到同一设备，优先选择显存更充足的设备（通常是 GPU 0）
            # Shift: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # 多卡时将 loss 计算移到 GPU 0（显存通常更充足，因为 lm_head 在 GPU 1）
            if self.n_devices > 1:
                target_device = self.tensor_parallel_devices[0]
                shift_logits = shift_logits.to(target_device)
                shift_labels = shift_labels.to(target_device)
            else:
                shift_labels = shift_labels.to(shift_logits.device)

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    # ── 门控统计 ──

    def gate_stats(self):
        return {f"layer_{k}": m.last_gate for k, m in self.engrams.items()}

    # ── 生成 ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, prompt_ids, max_new=150, temperature=0.7, top_k=50):
        self.eval()
        ids = prompt_ids.clone()
        if self.n_devices > 1:
            ids = ids.to(self.tensor_parallel_devices[0])
        for _ in range(max_new):
            crop = ids[:, -self.max_seq_len:]
            logits, _ = self(crop)
            if self.n_devices > 1:
                logits = logits.to(self.tensor_parallel_devices[0])
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            ids = torch.cat([ids, nxt], dim=1)
            if nxt.item() in (self.tokenizer.eos_token_id,):
                break
        return ids


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. 数据集
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TextDataset(Dataset):
    """滑窗切分 token 序列，每条样本同时作为 input 和 label"""

    def __init__(self, ids, seq_len):
        self.data = torch.tensor(ids, dtype=torch.long)
        self.sl = seq_len
        self.n = max(1, len(ids) - seq_len)

    def __len__(self): return self.n

    def __getitem__(self, i): return self.data[i: i + self.sl]


def load_texts(data_dir):
    texts = []
    for ext in ("*.txt", "*.md"):
        for f in sorted(glob.glob(os.path.join(data_dir, "**", ext), recursive=True)):
            c = open(f, encoding="utf-8", errors="ignore").read().strip()
            if c:
                texts.append(c)
                print(f"  {f}  ({len(c):,} chars)")
    return "\n\n".join(texts) if texts else None


def load_eval_data(data_dir: str) -> Dict[str, Any]:
    """
    扫描 data_dir 下的 *_eval.py 文件，动态导入并提取评估数据。
    
    返回字典包含:
      - recall_prompts: 召回测试提示词列表（合并自所有 *_RECALL_PROMPTS 变量）
      - expected_keywords: 期望关键词列表（合并自所有 *_EXPECTED_KEYWORDS 变量）
      - prompt_to_keyword_idx: 每个提示词对应的关键词组索引列表
    """
    result = {
        "recall_prompts": [],
        "expected_keywords": [],
        "prompt_to_keyword_idx": [],
    }
    
    # 扫描 *_eval.py 文件
    eval_files = sorted(glob.glob(os.path.join(data_dir, "*_eval.py")))
    if not eval_files:
        return result
    
    print(f"\nFound {len(eval_files)} evaluation file(s):")
    
    for eval_file in eval_files:
        print(f"  Loading: {eval_file}")
        
        # 动态导入模块
        module_name = os.path.splitext(os.path.basename(eval_file))[0]
        spec = importlib.util.spec_from_file_location(module_name, eval_file)
        if spec is None or spec.loader is None:
            print(f"    [WARN] Cannot load {eval_file}, skipping")
            continue
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"    [WARN] Error loading {eval_file}: {e}, skipping")
            continue
        
        # 遍历模块属性，查找匹配的变量
        found_prompts = []
        found_keywords = []
        
        for attr_name in dir(module):
            if attr_name.startswith("_"):
                continue
            attr_val = getattr(module, attr_name)
            
            # 匹配 *_RECALL_PROMPTS
            if attr_name.endswith("_RECALL_PROMPTS") and isinstance(attr_val, list):
                found_prompts.append((attr_name, attr_val))
                print(f"    Found {attr_name}: {len(attr_val)} prompts")
            
            # 匹配 *_EXPECTED_KEYWORDS
            elif attr_name.endswith("_EXPECTED_KEYWORDS") and isinstance(attr_val, list):
                found_keywords.append((attr_name, attr_val))
                print(f"    Found {attr_name}: {len(attr_val)} keyword groups")
        
        # 合并数据并建立对应关系
        # 假设同一模块中的 RECALL_PROMPTS 和 EXPECTED_KEYWORDS 配对
        # EXPECTED_KEYWORDS 按知识条目分组，RECALL_PROMPTS 是展开的
        # 需要从模块的 KNOWLEDGE_ENTRIES 中推断对应关系，或根据注释的条目分组来计算
        
        for prompts_name, prompts_list in found_prompts:
            # 尝试找到对应的 keywords
            prefix = prompts_name.replace("_RECALL_PROMPTS", "")
            keywords_name = f"{prefix}_EXPECTED_KEYWORDS"
            keywords_list = None
            for kw_name, kw_val in found_keywords:
                if kw_name == keywords_name:
                    keywords_list = kw_val
                    break
            
            if keywords_list is None:
                # 没有配对的关键词，只添加 prompts
                base_idx = len(result["expected_keywords"])
                for prompt in prompts_list:
                    result["recall_prompts"].append(prompt)
                    result["prompt_to_keyword_idx"].append(-1)  # -1 表示无对应关键词
            else:
                # 有配对的关键词，需要建立对应关系
                # 尝试从 KNOWLEDGE_ENTRIES 推断每组的提示词数量
                entries_name = "KNOWLEDGE_ENTRIES"
                entries_list = getattr(module, entries_name, None)
                
                base_kw_idx = len(result["expected_keywords"])
                result["expected_keywords"].extend(keywords_list)
                
                if entries_list and isinstance(entries_list, list):
                    # 从 KNOWLEDGE_ENTRIES 获取每个条目的提示词数量
                    prompt_idx = 0
                    for entry_idx, entry in enumerate(entries_list):
                        entry_prompts = entry.get("recall_prompts", [])
                        n_prompts = len(entry_prompts)
                        for _ in range(n_prompts):
                            if prompt_idx < len(prompts_list):
                                result["recall_prompts"].append(prompts_list[prompt_idx])
                                result["prompt_to_keyword_idx"].append(base_kw_idx + entry_idx)
                                prompt_idx += 1
                    # 处理剩余的 prompts（如果有）
                    while prompt_idx < len(prompts_list):
                        result["recall_prompts"].append(prompts_list[prompt_idx])
                        result["prompt_to_keyword_idx"].append(-1)
                        prompt_idx += 1
                else:
                    # 没有 KNOWLEDGE_ENTRIES，尝试平均分配
                    n_prompts = len(prompts_list)
                    n_groups = len(keywords_list)
                    prompts_per_group = n_prompts // n_groups if n_groups > 0 else n_prompts
                    
                    for i, prompt in enumerate(prompts_list):
                        result["recall_prompts"].append(prompt)
                        if n_groups > 0:
                            group_idx = min(i // prompts_per_group, n_groups - 1)
                            result["prompt_to_keyword_idx"].append(base_kw_idx + group_idx)
                        else:
                            result["prompt_to_keyword_idx"].append(-1)
    
    if result["recall_prompts"]:
        print(f"\nTotal: {len(result['recall_prompts'])} prompts, "
              f"{len(result['expected_keywords'])} keyword groups")
    
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. 训练
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_model(model, dataset, epochs, lr, bs, device, tensor_parallel_devices=None,
                accumulation_steps=1):
    """
    训练模型。
    
    Args:
        accumulation_steps: 梯度累积步数，effective batch size = bs × accumulation_steps
    """
    # 清理显存
    torch.cuda.empty_cache()
    
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=True)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        print("No trainable parameters! (--no_engram mode, skipping training)")
        return

    primary_device = (tensor_parallel_devices or [device])[0]
    n_gpus = len(tensor_parallel_devices) if tensor_parallel_devices else 1

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    total_steps = epochs * len(loader)
    # scheduler 按实际 optimizer.step() 次数调度
    scheduler_steps = total_steps // accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(scheduler_steps, 1))

    effective_bs = bs * accumulation_steps
    print(f"\n{'=' * 60}")
    print(f"  Training: {epochs} epochs × {len(loader)} steps, bs={bs}")
    if accumulation_steps > 1:
        print(f"  Gradient Accumulation: {accumulation_steps} steps, effective bs={effective_bs}")
    print(f"  Trainable: {sum(p.numel() for p in params) / 1e6:.1f}M params")
    print(f"  Device: {device}" + (f" (Tensor Parallel: {n_gpus} GPUs)" if n_gpus > 1 else ""))
    print(f"{'=' * 60}\n")

    model.train()
    # 保持 base model 在 eval 模式（避免 dropout 等影响）
    model.base.eval()
    t0 = time.time()

    for ep in range(1, epochs + 1):
        total_loss, nb = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        
        for step_idx, batch in enumerate(loader):
            batch = batch.to(primary_device)
            _, loss = model(batch, labels=batch)
            
            # 梯度累积：loss 需要除以累积步数
            if accumulation_steps > 1:
                loss = loss / accumulation_steps
            
            loss.backward()
            total_loss += loss.item() * accumulation_steps  # 恢复原始 loss 值用于统计
            nb += 1

            # 每 accumulation_steps 步执行一次 optimizer.step()
            if (step_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        
        # 处理 epoch 末尾未满 accumulation_steps 的剩余梯度
        if len(loader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # 多卡时同步
        if n_gpus > 1:
            torch.cuda.synchronize()

        avg = total_loss / nb
        gs = "  ".join(f"L{k}={v:.4f}" for k, v in model.gate_stats().items())
        print(f"  Epoch {ep:3d}/{epochs} | Loss {avg:.4f} | "
              f"LR {scheduler.get_last_lr()[0]:.1e} | Gate[{gs}] | "
              f"{time.time() - t0:.0f}s")

    print(f"\nTraining done in {time.time() - t0:.1f}s")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. 知识召回测试 + 交互
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_recall(model, device, recall_prompts: Optional[List[str]] = None,
                expected_keywords: Optional[List[List[str]]] = None,
                prompt_to_keyword_idx: Optional[List[int]] = None):
    """
    知识召回测试。
    
    Args:
        model: 模型实例
        device: 设备
        recall_prompts: 可选的召回提示词列表，不提供则使用默认的 RECALL_PROMPTS
        expected_keywords: 可选的期望关键词列表（按知识条目分组）
        prompt_to_keyword_idx: 每个提示词对应的关键词组索引列表
    """
    # 使用提供的提示词或默认提示词
    prompts = recall_prompts if recall_prompts is not None else RECALL_PROMPTS
    
    print(f"\n{'=' * 60}")
    print("  Knowledge Recall Test")
    if expected_keywords:
        print(f"  ({len(prompts)} prompts, {len(expected_keywords)} keyword groups)")
    print(f"{'=' * 60}")
    
    # 统计信息
    total_prompts = len(prompts)
    total_keywords_hit = 0
    total_keywords_expected = 0
    per_group_stats = {}  # group_idx -> {"hit": int, "total": int, "prompts": int}
    
    for i, prompt in enumerate(prompts):
        ids = torch.tensor(
            [model.tokenizer.encode(prompt)], device=device
        )
        out = model.generate(ids, max_new=60, temperature=0.3, top_k=10)
        text = model.tokenizer.decode(out[0], skip_special_tokens=True)
        # 只显示生成的部分
        generated = text[len(prompt):].strip()
        
        print(f"\n  [{prompt}]")
        print(f"  → {generated[:200]}")
        
        # 关键词评估
        if expected_keywords and prompt_to_keyword_idx:
            kw_idx = prompt_to_keyword_idx[i] if i < len(prompt_to_keyword_idx) else -1
            if kw_idx >= 0 and kw_idx < len(expected_keywords):
                keywords = expected_keywords[kw_idx]
                hits = []
                misses = []
                # 检查每个关键词是否出现在生成文本中
                generated_lower = generated.lower()
                for kw in keywords:
                    if kw.lower() in generated_lower:
                        hits.append(kw)
                    else:
                        misses.append(kw)
                
                hit_count = len(hits)
                total_count = len(keywords)
                total_keywords_hit += hit_count
                total_keywords_expected += total_count
                
                # 更新每组统计
                if kw_idx not in per_group_stats:
                    per_group_stats[kw_idx] = {"hit": 0, "total": 0, "prompts": 0}
                per_group_stats[kw_idx]["hit"] += hit_count
                per_group_stats[kw_idx]["total"] += total_count
                per_group_stats[kw_idx]["prompts"] += 1
                
                # 打印关键词命中情况
                hit_rate = hit_count / total_count * 100 if total_count > 0 else 0
                print(f"  📊 Keywords: {hit_count}/{total_count} ({hit_rate:.0f}%)")
                if hits:
                    print(f"     ✓ Hit: {', '.join(hits)}")
                if misses:
                    print(f"     ✗ Miss: {', '.join(misses)}")
    
    # 打印总体评估摘要
    if expected_keywords and prompt_to_keyword_idx and total_keywords_expected > 0:
        print(f"\n{'=' * 60}")
        print("  Recall Evaluation Summary")
        print(f"{'=' * 60}")
        
        # 总体命中率
        overall_hit_rate = total_keywords_hit / total_keywords_expected * 100
        print(f"\n  Overall Keyword Hit Rate: {total_keywords_hit}/{total_keywords_expected} "
              f"({overall_hit_rate:.1f}%)")
        
        # 每组统计
        print(f"\n  Per-Group Statistics:")
        for group_idx in sorted(per_group_stats.keys()):
            stats = per_group_stats[group_idx]
            group_hit_rate = stats["hit"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"    Group {group_idx + 1}: {stats['hit']}/{stats['total']} hits "
                  f"({group_hit_rate:.0f}%) across {stats['prompts']} prompt(s)")
        
        print(f"\n{'=' * 60}")


def interactive(model, device):
    # readline 支持方向键 / 历史记录 / 中文输入
    try:
        import readline  # noqa: F401  导入即可生效
        readline.parse_and_bind("set editing-mode emacs")
        readline.parse_and_bind(r'"\e[A": previous-history')
        readline.parse_and_bind(r'"\e[B": next-history')
        readline.parse_and_bind(r'"\e[C": forward-char')
        readline.parse_and_bind(r'"\e[D": backward-char')
    except ImportError:
        pass

    print(f"\n{'=' * 60}")
    print("  Interactive Mode  (type 'quit' to exit)")
    print(f"{'=' * 60}")
    while True:
        try:
            prompt = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            break
        ids = torch.tensor(
            [model.tokenizer.encode(prompt)], device=device
        )
        out = model.generate(ids, max_new=200, temperature=0.7, top_k=50)
        text = model.tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"\nModel > {text}")
    print("\nBye!")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    ap = argparse.ArgumentParser(description="Engram × Pretrained LLM Demo")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                    help="HuggingFace 模型 ID")
    ap.add_argument("--data_dir", type=str, default=None,
                    help="本地知识文件目录（.txt/.md）")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seq_len", type=int, default=128,
                    help="训练序列长度")
    ap.add_argument("--save_engram", type=str, default="./engram_weights.pt",
                    help="保存 Engram 权重路径")
    ap.add_argument("--load_engram", type=str, default=None,
                    help="加载已保存的 Engram 权重")
    ap.add_argument("--no_engram", action="store_true",
                    help="禁用 Engram（对比基线）")
    ap.add_argument("--engram_layers", type=str, default=None,
                    help="Engram 插入层 ID，逗号分隔，如 '4,12'")
    ap.add_argument("--fp32", action="store_true",
                    help="强制使用 FP32")
    ap.add_argument("--mirror", type=str, default=None,
                    help="HuggingFace 镜像地址，如 https://hf-mirror.com")
    ap.add_argument("--tensor_parallel", action="store_true",
                    help="启用张量并行（多GPU）")
    ap.add_argument("--gpu_devices", type=str, default="0,1",
                    help="指定GPU设备索引（0-based，受CUDA_VISIBLE_DEVICES影响），逗号分隔，如 '0,1'")
    ap.add_argument("--engram_device", type=str, default="distribute",
                    choices=["distribute", "gpu0", "gpu1"],
                    help="Engram模块设备策略: distribute=随层分布, gpu0=全GPU0, gpu1=全GPU1")
    ap.add_argument("--accumulation_steps", type=int, default=1,
                    help="梯度累积步数，effective batch size = batch_size × accumulation_steps")
    args = ap.parse_args()

    # ── HF 镜像 ──
    if args.mirror:
        os.environ["HF_ENDPOINT"] = args.mirror
        print(f"HF mirror: {args.mirror}")
    else:
        _setup_hf_mirror()

    # ── 设备 ──
    if args.tensor_parallel:
        device, devices = _setup_tensor_parallel(args.gpu_devices)
    else:
        device = _detect_device()
        devices = [device]
    dtype = torch.float32 if args.fp32 else None
    print(f"\nDevice: {device}")

    # ── Engram 配置 ──
    ecfg = EngramConfig()
    if args.engram_layers:
        ecfg.engram_layer_ids = [int(x) for x in args.engram_layers.split(",")]

    # ── 构建模型 ──
    model = EngramLM(
        args.model, ecfg,
        use_engram=not args.no_engram,
        device=device, dtype=dtype,
        tensor_parallel_devices=devices if args.tensor_parallel else None,
        engram_device_strategy=args.engram_device if args.tensor_parallel else "distribute",
    )

    # ── 加载已有 Engram 权重 ──
    if args.load_engram:
        print(f"\nLoading Engram weights from {args.load_engram}...")
        state = torch.load(args.load_engram, map_location=device, weights_only=True)
        model.engrams.load_state_dict(state)
        print("Engram weights loaded.")
        if not args.tensor_parallel:
            model.to(device)
        
        # 加载评估数据（如果指定了 data_dir）
        eval_data = None
        if args.data_dir:
            eval_data = load_eval_data(args.data_dir)
        
        if eval_data and eval_data.get("recall_prompts"):
            test_recall(model, device,
                       recall_prompts=eval_data["recall_prompts"],
                       expected_keywords=eval_data["expected_keywords"],
                       prompt_to_keyword_idx=eval_data["prompt_to_keyword_idx"])
        else:
            test_recall(model, device)
        interactive(model, device)
        return

    # ── 准备数据 ──
    text = None
    eval_data = None
    if args.data_dir:
        print(f"\nLoading texts from {args.data_dir}...")
        text = load_texts(args.data_dir)
        # 加载评估数据
        eval_data = load_eval_data(args.data_dir)
    if text is None:
        print("\nUsing built-in example knowledge...")
        text = EXAMPLE_KNOWLEDGE

    tokens = model.tokenizer.encode(text)
    print(f"Corpus: {len(text):,} chars → {len(tokens):,} tokens")

    seq_len = min(args.seq_len, model.max_seq_len)
    if len(tokens) <= seq_len:
        print(f"[WARN] Text too short, repeating to fill...")
        while len(tokens) <= seq_len:
            tokens = tokens + tokens

    dataset = TextDataset(tokens, seq_len)
    print(f"Training samples: {len(dataset):,}  (seq_len={seq_len})")

    # ── 训练 ──
    if not args.tensor_parallel:
        model.to(device)
    train_model(model, dataset, args.epochs, args.lr, args.batch_size, device,
                tensor_parallel_devices=devices if args.tensor_parallel else None,
                accumulation_steps=args.accumulation_steps)

    # ── 保存 Engram 权重 ──
    if model.engrams:
        torch.save(model.engrams.state_dict(), args.save_engram)
        print(f"Engram weights saved to {args.save_engram}")

    # ── 知识召回 + 交互 ──
    if eval_data and eval_data.get("recall_prompts"):
        test_recall(model, device,
                   recall_prompts=eval_data["recall_prompts"],
                   expected_keywords=eval_data["expected_keywords"],
                   prompt_to_keyword_idx=eval_data["prompt_to_keyword_idx"])
    else:
        test_recall(model, device)
    interactive(model, device)


if __name__ == "__main__":
    main()