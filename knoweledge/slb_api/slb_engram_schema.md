# SLB → Engram JSONL Schema

这份 schema 兼容你当前 `JessyMelon/Engram` 仓库里的 `autoresearch/knowledge_format.py` 使用方式：
当前训练文本构造主要消费 `content`、`paraphrases`、`recall_prompts` 三个字段；其余字段用于后续筛选、评估和增量扩展。仓库当前把知识分成 `command_mapping / fact_pair / procedure / structured_config` 四类，并在 `build_training_text()` 中把原文、改写和 QA 拼接到训练文本里；本地 demo 也会把 `.txt/.md` 文本拼接成训练数据。citeturn685934view0turn758471view0turn758471view1

## 一条 JSONL 记录的字段

```json
{
  "id": "slb-create_loadbalancer-default-eip_type",
  "type": "structured_config",
  "action": "create_loadbalancer",
  "title": "create_loadbalancer.eip_type 默认值",
  "anchor": "[Action:create_loadbalancer]/[Default:eip_type]",
  "tags": ["slb", "api", "default", "param"],
  "content": "[Action:create_loadbalancer]/[Default:eip_type]\n在接口 create_loadbalancer 中，参数 eip_type 的默认值是 internet。",
  "paraphrases": [
    "如果 create_loadbalancer 未显式传入 eip_type，默认使用 internet。",
    "create_loadbalancer 的 eip_type 缺省值为 internet。"
  ],
  "recall_prompts": [
    "create_loadbalancer 的 eip_type 默认值是什么？",
    "不传 eip_type 时 create_loadbalancer 会用什么值？"
  ],
  "must_contain": ["create_loadbalancer", "eip_type", "internet"],
  "metadata": {
    "sample_kind": "default",
    "param_name": "eip_type",
    "default": "internet"
  }
}
```

## 字段说明

- `id`: 全局唯一 ID。
- `type`: 对齐当前仓库里的四种知识类型。
- `action`: 接口名或流程名，便于按接口切分。
- `title`: 人类可读标题。
- `anchor`: 强路径锚点，适合 Engram 的 2/3-gram 触发。
- `tags`: 检索与筛选标签。
- `content`: 训练主文本，建议保留英文 action / 参数名。
- `paraphrases`: 同义改写，用于增强召回。
- `recall_prompts`: 用于生成 QA 风格训练块或召回测试。
- `must_contain`: 用于关键词命中式评测。
- `metadata`: 结构化补充信息，例如 `sample_kind/default/enum_values/error_code`。

## 建议的 sample_kind

- `overview`: 动作概览
- `param`: 参数说明
- `default`: 默认值
- `constraint`: 条件约束 / 互斥规则 / 数量限制
- `error`: 错误码
- `struct_overview`: 结构体概览
- `struct_field`: 结构体字段
- `workflow`: 流程链路
- `contrast`: 相似接口对比

## 与当前仓库的最小对接方法

### 方法 1：直接喂给 autoresearch

读取 JSONL 后，仅保留：
- `type`
- `content`
- `paraphrases`
- `recall_prompts`

再作为 `KNOWLEDGE_ENTRIES` 使用即可。当前仓库的 `build_training_text()` 会把这几项自动拼成训练文本。citeturn758471view0

### 方法 2：转成 txt / md 给本地 demo

把 `content + paraphrases + recall_prompts` 展平成一份 `.txt` 或 `.md`，放到 demo 的知识目录即可。本地 demo 会递归读取 `.txt/.md` 文件并拼接成训练文本。citeturn758471view1

## 为什么这个 schema 比纯 FAQ 更适合 Engram

因为你当前仓库的 Engram 侧重固定路径、参数名、命令名和局部 N-gram 命中；同时 `CompressedTokenizer` 会做 lowercasing 和空白归一化，所以 `anchor + 英文字段名 + 短逻辑句` 会比冗长自然语言段落更稳定。citeturn685934view3turn758471view1
