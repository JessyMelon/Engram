# Engram 与 Agent-Memory-Paper-List 中 knowledge_format 的定义、示例与可落地升级方案研究报告

## 执行摘要

本次研究聚焦于两个位于 entity["company","GitHub","code hosting"] 的仓库：`JessyMelon/Engram` 与 `JessyMelon/Agent-Memory-Paper-List`，目标是梳理 Engram 当前“知识库/knowledge_format”定义与示例、识别其变体与兼容性缺陷，并结合 Agent Memory 代表性论文提出可落地的优化方案与替代格式。citeturn34view0turn33view0turn35view0

核心发现与结论如下（面向落地）：

Engram 的“knowledge_format”在仓库内至少存在三套互相并行的表达：  
其一是 `autoresearch/knowledge_format.py` 中以 **`List[Dict]`** 形式维护的结构化知识条目（含 `type/content/paraphrases/recall_prompts/expected_keywords`）与 `RECALL_TESTS`（prompt+must_contain+must_not_contain）。citeturn24view4turn7view0turn26view0turn23view0  
其二是根目录 `knowledge_data.py` 的 **纯文本知识语料 `EXAMPLE_KNOWLEDGE`** 与 `RECALL_PROMPTS`，被 `engram_local_demo.py` 以“拼接多个 `.txt/.md` 或回退到内置文本”的方式直接 token 化训练。citeturn13view2turn13view3turn16view0turn16view2  
其三是 README 中给出的四类知识模板（Type A–D）与 **N-gram 触发模式**、增强与配比建议，作为“作者侧规范”，但并未统一落入同一数据模式与加载入口。citeturn34view0turn34view1  

当前体系的主要优点：  
结构化版本（autoresearch）把“知识条目、数据增强（paraphrases）、召回评测（RECALL_TESTS）”放在同一处，能快速迭代 recall_score/val_ppl；README 进一步明确了 Type A–D 的适用场景与 N-gram 触发模板，有利于在 Engram 的 N-gram 记忆机制下“可控触发”。citeturn33view0turn26view0turn34view0turn34view1  

主要问题集中在“格式碎片化与缺少工程化元数据”：  
同一仓库同时存在“结构化 Dict 列表”和“纯文本语料”两条数据通路，字段命名与评测结构不统一（`recall_prompts` vs `RECALL_PROMPTS`；`expected_keywords` vs `must_contain`），并且缺少稳定的 `id/版本/来源/生效范围/安全分级/语言/时间` 等元数据，导致知识复用、增量更新、冲突处理、跨规模检索与审计都较难扩展。citeturn24view3turn13view3turn23view0turn16view0  

结合 `Agent-Memory-Paper-List` 的统一分类框架（Forms/Functions/Dynamics）与代表性论文（图结构、层级结构、时间维度、写入-巩固-检索生命周期等），本报告给出 4 个可落地方案：  
A) **KBEntry v2 JSONL（最推荐的渐进式升级）**：把现有字段补齐为可版本化、可索引、可向量化的稳定 Schema，并提供编译器兼容输出旧版训练文本；  
B) **Dual-View（源数据结构化 + 训练/推理可控渲染）**：将“触发友好的文本视图”从数据中剥离为可配置渲染层，保留 Engram 训练优势；  
C) **Temporal/Property Graph（面向关系与时间推理）**：借鉴 Zep/MAGMA/HippoRAG 等，把事实与事件显式图化，适合 10k–1M+ 规模与复杂检索；  
D) **Hierarchical Tree（面向长文档与流程）**：借鉴 MemTree，把 procedure/手册类知识树化、分层摘要与嵌入，提升检索与上下文拼装质量。citeturn35view0turn30search0turn30search2turn31search1turn31search3  

## 研究对象与方法

研究对象与关键证据来自以下源码与文档入口：

Engram 仓库侧：  
README 的 “6.2 知识数据格式” 给出 Type A–D 的模板、适用场景与 N-gram 触发模式，并提出“每条知识 2–3 个改写变体”和类型占比建议；同一 README 还定义 AutoResearch 循环以 `recall_score` 与 `val_ppl` 作为核心指标。citeturn34view0turn34view1  
`autoresearch/knowledge_format.py` 定义结构化知识条目 `KNOWLEDGE_ENTRIES`、召回测试 `RECALL_TESTS`，并提供 `build_training_text/build_validation_text`、格式校验 `validate_entry` 等函数。citeturn24view4turn26view0turn33view0turn24view3  
`autoresearch/prepare.py` 固化 recall 评测方式：对每条测试 prompt 生成并检查 must_contain/must_not_contain，计算“命中关键词数 / 总关键词数”的均值。citeturn23view0turn23view2  
根目录 `knowledge_data.py` 的 `EXAMPLE_KNOWLEDGE` 与 `RECALL_PROMPTS` 作为内置样例语料；`engram_local_demo.py` 会加载本地 `.txt/.md` 或回退该内置语料进行 token 训练与召回测试。citeturn13view3turn16view0turn16view2  

论文侧：  
`Agent-Memory-Paper-List` README 给出 Agent Memory 的三维统一分类（Forms/Functions/Dynamics）与大量论文条目；本报告从其中选择 8 篇具有代表性的“知识/记忆表示格式”工作，优先采用 entity["organization","arXiv","preprint server"]、entity["organization","ACL Anthology","nlp paper archive"]、以及论文官方代码仓库作为引用来源。citeturn35view0turn29view0turn31search6turn31search21turn32search4  

## Engram 当前 knowledge_format 盘点与变体比较

### 结构化版 knowledge_format（autoresearch/knowledge_format.py）

`autoresearch/knowledge_format.py` 的文件头明确其“定义 Engram 训练所需的知识数据格式、数据增强方法和召回测试数据”，并将知识分为 Type A–D：`command_mapping / fact_pair / procedure / structured_config`。citeturn6view2  

其中 `KNOWLEDGE_ENTRIES` 的每个条目是一个字典；以 GetRegions 为例，该条目含 `type/content/paraphrases/recall_prompts/expected_keywords`。citeturn24view4turn7view0  

此外，`validate_entry()` 指出必填字段为 `["type","content"]`，并约束 `type` 必须属于四个枚举；同时对 `paraphrases`（至少 2 个变体）与 `expected_keywords` 的类型进行校验。citeturn24view3  

`build_training_text()` 的拼装逻辑是：添加统一 header，然后对每条知识拼接“原文 content + paraphrases（以‘补充说明’列表形式）+ recall_prompts 生成的 QA（只取前两个问题，并对 command_mapping 给出固定回答模板）”，最后以双换行分隔。citeturn33view0turn33view1turn33view2  

### 评测版 knowledge_format（RECALL_TESTS 与 evaluate_recall）

同一文件中 `RECALL_TESTS` 定义为 `List[Dict]`，字段包含：

- `prompt: str`  
- `must_contain: List[str]`  
- `must_not_contain: List[str]`  
- `score_method: str`（当前示例为 `"keyword_match"`）

并给出多条测试用例（如 GetRegions、CheckoutCluster、DockerLogs 等）。citeturn26view0turn25view0  

`autoresearch/prepare.py` 的 `evaluate_recall()` 将每条测试的得分定义为“must_contain 命中数 / must_contain 总数”，并取所有测试均值（0–1）。citeturn23view0turn23view2  

训练脚本 `autoresearch/train.py` 直接导入 `RECALL_TESTS/build_training_text/build_validation_text` 并在训练后计算 `recall_score` 与 `val_ppl`。citeturn21view0turn21view3  

### 纯文本版 knowledge_format（knowledge_data.py 与 engram_local_demo.py）

与 autoresearch 的结构化条目不同，`knowledge_data.py` 将知识作为一个“内置示例知识文本”`EXAMPLE_KNOWLEDGE` 存储（包含大量 Markdown 结构、命令模板、示例输入输出等），并单独维护 `RECALL_PROMPTS: List[str]`。citeturn13view3  

`engram_local_demo.py` 提供两种数据入口：若用户传 `--data_dir`，则递归读取目录下 `.txt/.md` 并拼接为训练文本；否则回退到 `EXAMPLE_KNOWLEDGE`，然后直接 tokenizer 编码与滑窗训练。citeturn16view0turn16view2  

这意味着 local demo 的“知识格式”实际上是“任何可读文本”，而不是结构化条目，这与 autoresearch 管线形成明显分叉。citeturn16view0turn24view4  

### README 层的“规范型 knowledge_format”（模板 + N-gram 触发）

README “6.2 知识数据格式”给出四类知识模板，并为每类指定 N-gram 触发模式（例如 Type A 由 `"指令:" + 动作词` 触发，Type C 由 `"步骤" + 数字` 触发）；同时建议每条知识生成 2–3 个改写变体，并给出四类知识的占比建议。citeturn34view0turn34view1  

### 变体比较表：字段、必选/可选、示例与兼容性问题

下表将仓库中三条“knowledge_format 表达”并列比较（结构化条目 / 纯文本语料 / README 模板规范）。表中示例均来自仓库源码与 README。citeturn24view3turn13view3turn34view0  

| 维度/字段 | autoresearch/knowledge_format.py（结构化条目） | knowledge_data.py + local demo（纯文本语料） | README 6.2（模板规范） | 兼容性/风险点 |
|---|---|---|---|---|
| 知识单位 | `Dict` 条目（多条组成 List）citeturn24view4 | 单个大字符串 `EXAMPLE_KNOWLEDGE` 或拼接的 `.txt/.md` 文本citeturn13view3turn16view0 | 模板片段（作者侧规范）citeturn34view0 | 结构化与文本化双轨：难以共享工具链（校验/索引/增量更新） |
| `type` | 必填；枚举：`command_mapping/fact_pair/procedure/structured_config`citeturn24view3turn6view2 | 无显式字段；通过文本内容隐含 | 明确四类 Type A–D 与适用场景citeturn34view0 | local demo 无法直接按 type 做过滤/检索/配比控制 |
| `content` | 必填；Markdown+示例块citeturn24view4turn7view0 | 全部内容都在文本中 | 模板用占位符描述 content 结构citeturn34view0 | lack of canonical content：同一知识在不同入口可能出现格式漂移 |
| `paraphrases` | 可选但校验要求 ≥2（如存在）citeturn24view3turn7view0 | 无字段；可在文本中自行写改写段落 | 建议每条知识 2–3 改写变体citeturn34view0 | 改写策略无法统一治理（质量、去重、覆盖率） |
| `recall_prompts` / `RECALL_PROMPTS` | 条目内 `recall_prompts: List[str]`citeturn7view0turn27view3 | 单独维护 `RECALL_PROMPTS: List[str]`citeturn13view3 | 无 | 字段命名与存放位置不一致，评测集难复用 |
| `expected_keywords` | 可选；用于关键词覆盖/评测设计citeturn7view0turn24view3 | 无（只能从文本抽取） | 无 | 关键词治理缺失：难做自动生成 RECALL_TESTS |
| 评测结构 | `RECALL_TESTS`: prompt + must_contain + must_not_contain + score_methodciteturn26view0turn25view0 | local demo 仅用 prompt 列表做“输出观测”，无 must_contain/负样本约束citeturn16view1turn13view3 | README 定义 recall_score/val_ppl 作为指标citeturn34view1 | 评测粒度不一致：autoresearch 可量化；local demo 更像“演示” |
| 训练文本生成 | `build_training_text()`：content+改写+QA 拼装citeturn33view0turn33view1 | 直接将文本 token 化训练citeturn16view0 | 只给模板与触发建议 | 数据准备逻辑分叉，导致同一知识在不同路径下“记忆效果不可比” |

## 论文启发：代表性 Agent Memory 格式与原则

`Agent-Memory-Paper-List` 给出一个三维统一框架：  
Forms（Token-level/Parametric/Latent），Functions（Factual/Experiential/Working），Dynamics（Formation/Evolution/Retrieval）。这一结构非常适合用来反推“知识库格式应该携带哪些元数据”（如时间、演化、检索策略、存储介质）。citeturn35view0  

本报告从该列表中选择 8 篇代表性工作（覆盖“扁平事实、图结构、树结构、时间结构、写入-巩固-检索生命周期、生产实践”），并说明选择理由与关键启发（每条均来自列表与论文原文/官方实现）。citeturn29view0turn35view0  

### 入选论文与理由

- **MemGPT: Towards LLMs as Operating Systems**：典型的“分层记忆/虚拟上下文管理”，强调把记忆看作可调度资源（与 Engram 的“可控触发”理念互补）。citeturn29view1turn31search0turn31search20  
- **HippoRAG**：把知识图谱与 Personalized PageRank 融合为长期记忆检索框架，代表“结构化关系 + 图算法检索”路线。citeturn29view1turn31search1turn31search21  
- **EMG-RAG（Editable Memory Graph）**：以“可编辑记忆图”组织个人数据，强调可编辑性与选择性，适合解释“为什么需要显式 schema/可追溯”。citeturn29view0turn31search2turn31search6  
- **MemTree（Dynamic Tree Memory Representation）**：以动态树结构存储聚合文本、embedding 与抽象层级，代表“层级知识库/分层摘要”路线。citeturn29view0turn31search3turn31search11  
- **Zep（Temporal Knowledge Graph/Graphiti）**：以时间感知知识图为核心，强调“动态整合对话与结构化数据、保留历史关系”，代表“时间维度 + 图化 + 低延迟服务化”。citeturn29view0turn30search0turn30search4  
- **Mem0**：面向生产的“抽取-巩固-检索”流水线，并提出图记忆增强版本，给出延迟与 token 成本等工程指标。citeturn29view0turn32search0turn32search4turn32search25  
- **MAGMA（Multi-Graph Agentic Memory）**：将每条记忆映射到语义/时间/因果/实体四张正交图，并以策略引导遍历检索，代表“多视图解耦 + 可解释检索路径”。citeturn35view0turn30search2turn30search6  
- **EverMemOS**：提出 MemCells/MemScenes 等记忆对象与“形成-巩固-回忆”的生命周期，代表“可演化、可重构的记忆操作系统”式组织。citeturn35view0turn30search3turn30search7  

### 各论文对“知识/记忆表示格式”的直接启发

1) **层级/多级存储与“写入/读取接口”要显式化**：MemGPT 把上下文管理类比 OS 的分层内存，并通过系统机制在不同层级间搬运信息，意味着知识库格式应区分“热区（工作记忆）/冷区（长期记忆）”，并显式记录内容如何进入/退出不同层级。citeturn31search0turn31search20  

2) **图结构适合表达实体关系、跨多跳推理与可解释检索**：HippoRAG 将 LLM、知识图谱与 Personalized PageRank 结合，强调“关联性/多跳检索的效率提升”。这提示：对 fact_pair 与跨条目引用密集的知识，图化（实体-关系）比纯文本更可控。citeturn31search1turn31search21  

3) **“可编辑性/选择性”需要结构化节点与变更语义**：EMG-RAG 明确提出 Editable Memory Graph，并用强化学习处理数据收集、可编辑性与选择性挑战；这要求知识库对象具备“可定位、可修改、可比较”的稳定标识与字段边界。citeturn31search2turn31search6  

4) **树结构适合长对话/长文档的“抽象层级”与渐进检索**：MemTree 强调每个节点封装聚合文本、对应语义 embedding 以及不同抽象层级，动态更新树结构以更好组织与检索。对 Engram 的 procedure/手册类知识，这是非常直接的替代格式灵感。citeturn31search3turn31search11  

5) **时间维度是长期记忆系统的第一等公民**：Zep 以时间感知知识图引擎 Graphiti 作为核心，面向“多轮对话 + 业务数据”的动态整合场景，并强调历史关系与低延迟。对于需要“版本/生效期/冲突”的知识库，时间字段与时间边（temporal edges）应进入 Schema。citeturn30search0turn30search4  

6) **工程指标必须纳入 Schema 设计的权衡**：Mem0 在论文摘要中强调 p95 延迟、token 成本、对比基线的可复现实验指标；同时其代码与研究页面面向生产落地。对 Engram 而言，这意味着知识格式不能只满足训练，还要为“索引/检索/更新/审计”提供足够元数据以支持 SLA。citeturn32search0turn32search4turn32search25  

7) **多视图解耦有助于降低“语义相似检索”在因果/时间/实体上的混叠**：MAGMA 指出现有方法把时间/因果/实体“纠缠”在单一记忆库里，提出四图正交表示与策略引导遍历。这对 Engram 当前仅靠文本 N-gram 触发的方式是重要补全：结构化视图可以与 N-gram 触发并行，使检索更稳健、更可解释。citeturn30search2turn30search6  

8) **生命周期对象化（MemCells/MemScenes）有助于规模化管理冲突与演化**：EverMemOS 将对话流转化为 MemCells，再组织为 MemScenes，并以“形成-巩固-回忆”三阶段运作；这提示：knowledge_format 应支持“原始事件/提炼事实/归并主题”多种粒度对象，并记录“由谁、何时、基于哪些证据”生成与更新。citeturn30search3turn30search7  

## 可落地的优化与替代知识库格式方案

本节提出 4 个方案（满足“字段定义、示例、适用场景、优缺点、兼容性、迁移与注意事项”），并以统一指标进行评估。所有方案默认“无特定约束”，但在评估中分别讨论小规模（<10k）、中等（10k–1M）、大规模（>1M）影响。评估指标与打分用于工程决策，属建议性结论。citeturn34view1turn32search0turn30search0  

### 评估指标体系

结合 Engram 当前 recall 评测方式（关键词命中率）与论文中常见工程维度（延迟、成本、可解释性），建议采用以下指标组：

- 查询效率：支持 `type/tags/time` 过滤、关键字检索、向量检索、图遍历等能力；复杂度与索引可达性。citeturn23view0turn30search2turn31search3  
- 可解释性：能否回答“为什么召回这条知识”，是否可提供路径/证据（如图路径、树路径、版本记录）。citeturn30search0turn30search2turn31search3  
- 可扩展性：数据量从 10k→1M→>1M 时，存储、索引、更新与检索是否可水平扩展。citeturn32search0turn30search0turn30search2  
- 存储成本：字段冗余、embedding 多份存储、图边数量增长等对存储的影响。citeturn31search3turn30search2  
- 检索延迟：p50/p95，尤其是“组合上下文”的端到端延迟；与 Mem0/Zep 等强调的延迟指标对齐。citeturn32search0turn30search0  
- 向量化兼容性：是否易于生成 embedding（单文本、多 chunk、分层摘要）、是否支持多表示（paraphrases/summary）。citeturn31search3turn34view0turn33view0  
- 与现有 Engram 管线兼容性：能否无缝生成 `build_training_text()` 所需“触发友好文本”，并兼容 `RECALL_TESTS` 的 must_contain 评测逻辑。citeturn33view0turn26view0turn34view0  

### 方案 A：KBEntry v2 JSONL（渐进式、以兼容为优先）

**定位**：把 `KNOWLEDGE_ENTRIES` 映射为稳定的 JSONL/JSON Schema，补齐工程化元数据；通过“编译/渲染器”输出旧版训练文本与旧版 `RECALL_TESTS`，最适合先落地。其设计理念延续 Engram README 的 Type A–D 与改写增强策略。citeturn34view0turn24view3turn33view0  

**字段定义（建议）**：  
- `id: string`（稳定 ID，推荐 content 哈希或 ULID）  
- `type: enum`（四类）citeturn24view3turn34view0  
- `title: string`（人类可读短标题）  
- `content: { format: "markdown"|"text", body: string }`（原文）citeturn7view0turn24view4  
- `paraphrases: string[]`（改写）citeturn24view3turn33view0  
- `prompts: { recall: string[], qa?: {q:string,a:string}[] }`（统一 recall_prompts/RECALL_PROMPTS）citeturn7view0turn13view3  
- `keywords: { expected: string[], negative?: string[] }`（统一 expected_keywords 与 must_not_contain）citeturn26view0turn7view0  
- `triggers: { ngram_phrases: string[], pattern?: string }`（把 README 的 N-gram 触发显式化）citeturn34view0  
- `provenance: { source: string, author?: string, created_at: string, updated_at: string, version: int }`  
- `embeddings?: { model: string, dims: int, vectors: {field:string, chunk:int, v: number[]}[] }`（可选）  
- `security?: { pii: boolean, confidentiality: "public"|"internal"|"restricted" }`（可选）

**示例模板（YAML，作者侧）**
```yaml
id: "kb_01HZX2..."
type: "command_mapping"
title: "GetRegions - 获取 Region 列表"
content:
  format: "markdown"
  body: |
    ## GetRegions - 获取 Region 列表
    tianji GetRegions 命令用于获取当前天基环境下所有可用的 Region 列表及其编号(region no)。
    命令格式:
    ```
    tianji GetRegions
    __func__ python3 -m cli.OpenAPI.api tianji Action GetRegions
    ```
paraphrases:
  - "使用 tianji GetRegions 可以列出所有可用区域及其 API/Portal 地址。"
  - "查看天基有哪些 Region：执行 GetRegions 获取区域列表。"
prompts:
  recall:
    - "获取当前天基环境下所有可用的 Region 列表及其编号"
keywords:
  expected: ["GetRegions", "tianji", "Region"]
triggers:
  ngram_phrases: ["指令: 获取 Region 列表", "tianji GetRegions"]
provenance:
  source: "engram/autoresearch/knowledge_format.py"
  created_at: "2026-03-28T00:00:00Z"
  updated_at: "2026-03-28T00:00:00Z"
  version: 1
```

**示例模板（JSON，API/存储侧）**
```json
{
  "id": "kb_01HZX2...",
  "type": "command_mapping",
  "title": "GetRegions - 获取 Region 列表",
  "content": {"format": "markdown", "body": "## GetRegions ..."},
  "paraphrases": ["...", "..."],
  "prompts": {"recall": ["获取当前天基环境下所有可用的 Region 列表及其编号"]},
  "keywords": {"expected": ["GetRegions", "tianji", "Region"]},
  "triggers": {"ngram_phrases": ["指令: 获取 Region 列表", "tianji GetRegions"]},
  "provenance": {"source": "JessyMelon/Engram", "version": 1, "created_at": "...", "updated_at": "..."}
}
```

**适用场景**：  
- 现有 Engram AutoResearch（需要继续产出训练文本 + RECALL_TESTS）citeturn33view0turn26view0turn34view1  
- 小规模到中等规模（<10k–1M）知识库的“可治理化”改造（字段齐全后可上全文检索与向量检索）citeturn32search0  

**优点**：  
- 与现有 `KNOWLEDGE_ENTRIES` 映射直观，迁移成本最低；可继续复用 `validate_entry` 的思想并升级为 JSON Schema 校验。citeturn24view3turn24view4  
- 把 README 中隐含的触发模式变成显式字段，可系统化做触发覆盖率分析，而不是靠经验写文本。citeturn34view0turn33view0  

**缺点**：  
- 仍是“扁平条目”，复杂关系（实体-因果-时间）只能放在文本或 keywords 里，难以得到图检索级别的表达力（这是后续方案 C 的补足）。citeturn30search2turn30search0  

**与现有系统兼容性**：  
- 通过编译器生成旧版 `build_training_text()` 所需拼装文本（header+content+paraphrases+qa），并可按规则从 `keywords.expected/negative` 自动生成 `RECALL_TESTS`。citeturn33view0turn26view0turn23view0  

**迁移步骤要点**（概要，详见最后一节“落地实施”）：  
- AST 解析 `knowledge_format.py` 提取条目并写 JSONL；  
- 将 `knowledge_data.py` 作为“单文档”导入或进一步切分；  
- 引入 Schema 校验与回归测试，启用双写与可回滚导出。citeturn24view4turn13view3turn34view1  

---

### 方案 B：Dual-View（源数据结构化 + 可控渲染为训练/推理文本）

**定位**：把“知识源数据（结构化）”与“用于 Engram 训练/触发的文本视图”解耦。A 方案把 trigger 放进字段，但 B 方案更进一步：把渲染规则抽成配置（每种 type 一套模板），从而能像 README 6.2 那样精细控制 N-gram 触发片段、示例块、QA 结构等，同时保持存储层稳定。其动机来自 Engram 对 “触发模式”高度敏感的设定（Type A–D 的触发建议）。citeturn34view0turn33view0  

**字段定义（在 A 基础上额外增加）**：  
- `rendering: { templates: {...}, compile_targets: ["engram_train","rag_doc","eval_prompt"] }`  
- `compiled: { engram_train_text: string, eval_snippets: string[] }`（可缓存，或按需生成）

**示例模板（渲染规则 YAML）**
```yaml
render_templates:
  command_mapping:
    header: |
      ## 指令: {{action}}
      命令: {{command}}
    body: |
      参数: {{params}}
      示例:
      >>> {{example_in}}
      {{example_out}}
    trigger_hints:
      - "指令:"
      - "{{action_verb}}"
  procedure:
    body: |
      ## {{task_name}}
      步骤 1: {{step1}}
      步骤 2: {{step2}}
```

**适用场景**：  
- 需要频繁 A/B 实验“知识表达方式”（模板、触发短语、示例选择、改写策略）以优化 recall_score，而又不希望改动存储 Schema。citeturn34view1turn33view0turn23view0  

**优点**：  
- 把 `build_training_text()` 的逻辑从 Python 代码硬编码，提升为可配置渲染层，减少“代码即数据”的耦合。citeturn33view0turn24view4  
- 支持对同一条知识生成多种“投影”（Engram 训练投影 / RAG 文档投影 / 评测投影），更贴近论文中“不同记忆形式、不同功能”统一治理的方向。citeturn35view0turn31search0turn32search0  

**缺点**：  
- 需要维护模板版本与渲染一致性测试，否则可能出现“源数据没变但渲染后文本变化导致 recall 波动”。（可通过 compiled 缓存与 golden tests 控制。）

**兼容性**：  
- 可 1:1 生成现有 `build_training_text()` 的输出结构（甚至保持 header 文本一致），以保证与历史实验可比。citeturn33view0turn33view1  

---

### 方案 C：Temporal/Property Graph（时间感知知识图谱）

**定位**：将一部分知识（尤其是 fact_pair、用户画像、跨条目引用逻辑、跨会话演化）以“实体-关系-事件”图谱显式表达，并保留时间维度与版本演化。该方案借鉴 Zep 的 Temporal Knowledge Graph（Graphiti）与 MAGMA 的多图正交表示思想，用于解决扁平文本在“时间/因果/实体关系”上的混叠。citeturn30search0turn30search2turn31search1  

**字段定义（核心概念）**：  
- `nodes: [{id, kind: "entity"|"event"|"doc"|"command", props...}]`  
- `edges: [{src, dst, rel, valid_from, valid_to, confidence}]`  
- `evidence: [{node_or_edge_id, source_entry_id, snippet, timestamp}]`  
- `views: { semantic_graph, temporal_graph, causal_graph, entity_graph }`（可选，MAGMA 风格）citeturn30search2turn30search6  

**示例模板（JSON，片段）**
```json
{
  "graph": {
    "nodes": [
      {"id": "cmd:GetRegions", "kind": "command", "name": "GetRegions"},
      {"id": "param:ProjectName", "kind": "entity", "name": "ProjectName"},
      {"id": "doc:tianji_cli", "kind": "doc", "name": "tianji 命令行"}
    ],
    "edges": [
      {"src": "cmd:GetRegions", "dst": "doc:tianji_cli", "rel": "belongs_to", "valid_from": "2026-01-01"},
      {"src": "cmd:GetRegions", "dst": "param:ProjectName", "rel": "does_not_require", "valid_from": "2026-01-01"}
    ]
  }
}
```

**适用场景**：  
- 中等到大规模（10k–>1M）需要“可解释检索路径、跨 session 演化、冲突检测”的 Agent Memory；  
- 强时间属性领域（用户偏好变化、配置版本、事件序列）。citeturn30search0turn32search0turn30search3  

**优点**：  
- 天然支持 explainability：可输出“图路径/时间线”，与 MAGMA 的“policy-guided traversal 可解释路径”一致。citeturn30search2turn30search6  
- 天然支持“时间与历史关系”：与 Zep 的 Temporal KG 方向一致，利于跨会话推理与冲突处理。citeturn30search0turn30search4  

**缺点**：  
- 工程复杂度更高：需要图存储/图索引/图遍历策略；对小规模（<10k）可能性价比不如 A/B。  
- 仍需“文本投影”：Engram 的训练仍需要可触发的文本片段（因此建议与方案 B 的渲染层组合）。

**兼容性**：  
- 通过“Graph→Text 渲染器”输出训练文本（包含关键节点名、关系短语与模板化示例），保持 N-gram 触发覆盖。citeturn34view0turn33view0turn30search2  

---

### 方案 D：Hierarchical Tree（分层知识树 + 多粒度摘要/向量）

**定位**：针对 procedure/长文档/对话纪要等，采用 MemTree 风格的动态树表示：节点包含聚合文本、embedding、抽象层级，并支持随新信息到来进行插入与重平衡。该方案直接对齐 MemTree 的“hierarchical schemas”与节点封装信息的做法。citeturn31search3turn31search11turn34view0  

**字段定义（建议）**：  
- `tree_id: string`  
- `nodes: [{node_id, parent_id, depth, text, summary, embedding_ref, tags, time_range}]`  
- `routing: { insert_policy, merge_policy, split_policy }`  
- `exports: { engram_train_text, rag_chunks }`

**示例模板（YAML，树节点）**
```yaml
tree_id: "tianji_ops_procedures"
nodes:
  - node_id: "root"
    parent_id: null
    depth: 0
    summary: "天基运维知识总览"
  - node_id: "cluster_query"
    parent_id: "root"
    depth: 1
    title: "天基集群信息查询流程"
    text: |
      步骤 1: 获取可用区域（GetRegions）
      步骤 2: 列出产品集群（ListClusters）
      步骤 3: 获取集群服务实例（ListServiceInstance）
    tags: ["procedure", "cluster"]
```

**适用场景**：  
- 以“流程、手册、长文档”为主的知识库；  
- 需要“先粗后细”检索与上下文拼装（先召回高层摘要，再下钻叶子节点）。citeturn31search3turn27view1turn27view2  

**优点**：  
- 对长文本更友好：天然支持多粒度摘要与分层 embedding，减少一次性塞满上下文导致的噪声；与 MemTree 的设计一致。citeturn31search3turn31search11  
- 对 Engram 也友好：可在每层节点生成不同粒度的“触发片段”（比如根节点输出命令分类、叶子节点输出具体命令格式）。citeturn27view2turn34view0  

**缺点**：  
- 更依赖“插入/合并策略”的质量；需要额外测试以控制树退化（过深/过宽）与频繁重平衡带来的成本。  

**兼容性**：  
- 可将树节点“逐节点渲染”为旧版训练文本块，保持 `build_training_text()` 的“多块拼接”语义。citeturn33view0turn33view2  

---

### 方案对比评估表（定性/半量化）

评分 1–5：越高越优（存储成本维度为“越高越省”）。评分依据上节指标与各论文/现有 Engram 管线的能力假设（例如 Zep/Mem0 对延迟和时间维度的强调、MemTree 对层级结构优势）。citeturn30search0turn32search0turn31search3turn34view1  

| 指标 | 现状（结构化+纯文本双轨） | 方案 A（KBEntry v2） | 方案 B（Dual-View） | 方案 C（Temporal Graph） | 方案 D（Hierarchical Tree） |
|---|---:|---:|---:|---:|---:|
| 查询效率（过滤+检索） | 2 | 4 | 4 | 5 | 4 |
| 可解释性 | 2 | 3 | 3 | 5 | 4 |
| 可扩展性（到 >1M） | 2 | 4 | 4 | 4–5（取决于图存储） | 4 |
| 存储成本（越省越高） | 3 | 4 | 3–4（compiled 可缓存） | 2–3（边多） | 3–4 |
| 检索延迟（p95） | 3（local 直拼文本） | 4 | 4 | 4（可做图裁剪/缓存） | 4 |
| 向量化兼容性 | 2–3 | 4 | 5（多投影） | 4 | 5 |
| 与现有 Engram 兼容性 | 3 | 5 | 5 | 4（需渲染） | 4–5（需渲染） |
| 小规模 <10k 推荐度 | 中 | 高 | 高 | 中 | 中 |
| 中等 10k–1M 推荐度 | 低–中 | 高 | 高 | 高 | 高 |
| 大规模 >1M 推荐度 | 低 | 中–高 | 中–高 | 高 | 高（需良好策略） |

综合建议：优先落地 **方案 A + 方案 B**（最小风险地统一 Schema 与数据通路），在“关系/时间推理强”的业务上再引入 **方案 C**，在“流程/长文档强”的业务上引入 **方案 D**。这一渐进路线与 Engram README 的 AutoResearch“阶段 3：知识格式优化”思路一致。citeturn34view1turn30search0turn31search3  

## 落地实施：迁移、索引、API、测试与回滚

本节给出可直接执行的工程落地建议，目标是：统一知识格式（消除双轨）、保留现有训练/评测可比性（能生成旧版训练文本与 RECALL_TESTS）、并为后续规模化检索（全文/向量/图/树）预留扩展点。citeturn33view0turn26view0turn16view0turn34view1  

### 迁移脚本思路（从现状到方案 A/B 的 JSONL）

**现有输入源**：
- `autoresearch/knowledge_format.py`: `KNOWLEDGE_ENTRIES` + `RECALL_TESTS`citeturn24view4turn26view0  
- `knowledge_data.py`: `EXAMPLE_KNOWLEDGE` + `RECALL_PROMPTS`citeturn13view3  
- 目录 `.txt/.md`：local demo 会递归读取拼接citeturn16view0  

**迁移策略建议**（先“保真导入”，再“结构化切分”）：
- 第一步：把 `KNOWLEDGE_ENTRIES` 逐条转为 JSONL（字段几乎 1:1），生成 `id` 与 `provenance`，并把 `expected_keywords` 映射到 `keywords.expected`。citeturn7view0turn24view3  
- 第二步：把 `RECALL_TESTS` 直接作为 `eval_tests.jsonl` 导入；同时提供生成器：若某条 KBEntry 缺少显式测试，则从 `prompts.recall` 与 `keywords.expected` 自动合成 must_contain。citeturn26view0turn23view0  
- 第三步：对 `EXAMPLE_KNOWLEDGE` 先作为 `type=structured_config/doc` 的单条记录导入（避免解析失败）；随后再做可选的“命令级切分”（基于 `## <Command> -` 标题与 `__func__` 标记分段），把切分出的单元转为 `command_mapping`。citeturn13view3turn34view0  
- 第四步：对外部 `.txt/.md`（local demo 入口）统一走“导入器”：每个文件成为一条 `doc` 或按规则切分为多条 entry，并记录 file path 作为 provenance.source。citeturn16view0  

**关键实现注意事项**：
- 迁移脚本要做“文本规范化”但保持可控：例如保留 README 指定的触发短语（`"指令:"`、`"步骤"` 等）以避免 recall 大幅波动。citeturn34view0turn33view0  
- 为保证实验可比性，先实现 `compile_to_engram_text()` 完全复刻 `build_training_text()`（header + content + paraphrases 列表 + QA），并提供 golden 输出测试。citeturn33view0turn33view1  

### 索引与向量化策略（按规模分层）

**小规模（<10k）**：  
建议“单库即可”：PostgreSQL JSONB + 全文索引（或 SQLite FTS）+ 可选的向量索引（如 HNSW/FAISS）。此时方案 A/B 的收益主要在“统一治理与可追溯”，而不是极致性能。citeturn32search0turn33view0  

**中等规模（10k–1M）**：  
推荐“混合检索”：BM25（关键词）+ 向量检索（语义）+ 结构过滤（type/time/tags）。  
- 索引字段：`type`、`provenance.updated_at`、`keywords.expected`、`triggers.ngram_phrases`、`content.body`、`paraphrases`。citeturn24view3turn34view0  
- 向量化策略：  
  - 最少两路 embedding：`content.body`（原文）与 `paraphrases`（增强召回）；  
  - 对 procedure/doc 使用方案 D 的“层级摘要 embedding”可显著降低长文噪声（参考 MemTree）。citeturn31search3turn33view0  

**大规模（>1M）**：  
- 若以“个人记忆/会话事件”为主：优先考虑方案 C（Temporal Graph）或“图+向量”的混合架构，以降低仅靠向量相似度带来的时间/因果混叠（MAGMA 的动机）。citeturn30search2turn30search0  
- 若以“长文档/流程”为主：使用方案 D 的树化分层，并在叶节点做向量索引、在高层做摘要索引，可显著改善检索延迟与拼装 token 成本（MemTree 的动机）。citeturn31search3turn31search11  
- 工程上需要：分片、冷热分层、批量写入、增量重建索引；这些与 Mem0 对“延迟/成本”强调一致。citeturn32search0turn32search25  

### API 设计要点（面向方案 A/B 的最小闭环）

建议先提供 5 个核心接口（REST 或 gRPC 均可）：

- `POST /kb/entries:upsert`：写入/更新 KBEntry（支持 version++、乐观锁）  
- `POST /kb/search`：混合检索（filters + bm25 + vector），输出 `entry_id + score + evidence`  
- `POST /kb/compile`：把若干 entry 编译为 `engram_train_text`（支持指定模板版本）  
- `GET /kb/entries/{id}`：获取原始 entry 与 provenance 以便审计  
- `POST /kb/eval_tests:generate`：从 entry 集合生成 `RECALL_TESTS`（默认沿用 must_contain/negative）citeturn26view0turn23view0turn33view0  

返回结构建议包含：
- `why`: 命中的字段（title/content/paraphrases/keywords/graph_path/tree_path）  
- `evidence_snippets`: 被召回的文本片段（用于可解释性与调试）  
这与 Zep/MAGMA 强调的“可解释检索路径/证据”方向一致。citeturn30search0turn30search2  

### 测试用例与回滚策略

**测试用例（建议最小集合）**：
- Schema 校验测试：对每条 entry 验证必填字段与类型约束（等价于 `validate_entry` 的升级版）。citeturn24view3  
- 编译一致性测试：同一输入 entry，`compile_to_engram_text()` 输出应稳定；并与历史 `build_training_text()` 输出对齐（golden files）。citeturn33view0turn33view1  
- Recall 回归测试：使用现有 `RECALL_TESTS` 计算 recall_score，确保升级不引入系统性下降；评测逻辑直接复用 `evaluate_recall` 的关键词命中定义。citeturn26view0turn23view0turn21view3  
- 负样本污染测试：确保 `must_not_contain` 能抑制明显跑偏（例如把 CheckoutService 答成 CheckoutCluster）。citeturn26view1turn25view0  

**回滚策略（建议）**：
- 双写阶段：一段时间内同时产出“新 JSONL + 旧训练文本”，训练/评测同时跑，出现 recall_score 回退则切回旧通路；这一策略与 README 描述的 AutoResearch “提升则 keep，否则 discard + git reset”精神一致。citeturn34view1turn21view3  
- 数据快照：所有导入/变更记录写入 append-only 日志（即便暂不实现完整事件溯源，也至少能恢复到任意版本）。  

### 推荐流程图（从知识维护到训练与检索的统一闭环）

```mermaid
flowchart TD
  A[作者/数据源\nREADME模板/py条目/txt&md] --> B[导入器\nparse+normalize+assign id]
  B --> C[KBEntry v2 Store\nJSONL/DB + provenance/version]
  C --> D[索引层\nBM25 + Vector + (Graph/Tree可选)]
  C --> E[渲染/编译层\nDual-View templates]
  E --> F[Engram训练文本\nheader+content+paraphrases+QA]
  C --> G[评测集生成器\nRECALL_TESTS]
  F --> H[训练/微调\nEngram AutoResearch]
  G --> I[Recall评测\nmust_contain/negative]
  H --> I
  I --> J[指标面板\nrecall_score/val_ppl/latency/token]
  J --> K[迭代与回滚\nkeep/discard]
  K --> C
```

该闭环把 Engram 当前“知识格式定义+训练文本拼装+召回评测”三件事统一到同一数据源与渲染层中，并保留现有 recall_score/val_ppl 指标体系。citeturn33view0turn26view0turn23view0turn34view1