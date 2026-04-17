# AGENTS.md 使用说明流程

这份说明面向两类 agent：
- Codex
- OpenCode

目标：把 `AGENTS.md` 作为仓库级操作手册，让 agent 在你的 Engram 样本工程里稳定执行“抽取 → 扩写 → 校验 → 评测”的闭环。

---

## 一、放置位置

把 `AGENTS.md` 放在仓库根目录：

```text
repo/
├── AGENTS.md
├── canonical_facts.jsonl
├── prompts/
├── schemas/
├── scripts/
└── data/
```

提交到 Git，让后续 agent 会话共享同一套规则。

---

## 二、推荐目录约定

建议把工程整理成下面这样：

```text
repo/
├── AGENTS.md
├── docs/
│   └── slb_api.md
├── data/
│   ├── canonical/
│   │   └── canonical_facts.jsonl
│   ├── expanded/
│   │   └── engram_samples.jsonl
│   └── eval/
│       └── eval_samples.jsonl
├── prompts/
│   └── engram_sample_expansion_prompt.md
├── schemas/
│   └── engram_sample_schema.json
├── scripts/
│   ├── extract_canonical_facts.py
│   ├── expand_engram_samples.py
│   ├── validate_engram_samples.py
│   └── stats_engram_samples.py
└── reports/
```

---

## 三、Codex 使用流程

### 方式 A：Codex App
1. 打开 Codex。
2. 选择你的项目目录。
3. 确认仓库根目录里有 `AGENTS.md`。
4. 给 Codex 明确任务。

推荐首条任务：

```text
读取 AGENTS.md。检查当前仓库的数据流水线是否符合其中规则。
先不要改训练脚本，先补齐 canonical facts → expansion → validation 的最小可运行链路。
最后给出变更摘要和待人工确认项。
```

### 方式 B：Codex CLI / IDE 扩展
进入仓库目录后直接给任务。

推荐任务模板：

```text
按照 AGENTS.md 执行：
1. 从 docs/slb_api.md 抽取 canonical facts 到 data/canonical/canonical_facts.jsonl
2. 用结构化输出脚本补全 expanded 样本
3. 运行 validator
4. 输出一份 stats 报告到 reports/
不要发明任何 action、field、error_code。
```

### 在 Codex 中适合做的事情
- 扫描代码库并补脚本
- 生成和修复 schema
- 跑验证
- 生成统计报告
- 形成小而可审查的 PR

### 不建议让 Codex 直接做的事情
- 脱离 source doc 自由发明训练样本
- 直接混改 train/eval 边界
- 大规模覆写人工标注文件且不报告差异

---

## 四、OpenCode 使用流程

### 初始化
进入仓库：

```bash
cd /path/to/repo
opencode
```

如果仓库里还没有 `AGENTS.md`，可以先运行：

```text
/init
```

如果已经有我们这份 `AGENTS.md`，优先保留仓库版本，不要让 `/init` 覆盖掉项目特定规则。

### 推荐操作顺序
先用 `Plan` 模式做分解，再切到 `Build` 模式执行。

示例：

```text
先根据 AGENTS.md 制定计划：
- 哪些脚本缺失
- 哪些数据层还没有分开
- 哪些验证还没做
然后只实现第一阶段最小闭环。
```

再切执行：

```text
按照计划执行：
- 增加 extract_canonical_facts.py
- 对接 expand_engram_samples.py
- 运行 validate_engram_samples.py
- 生成 reports/sample_stats.md
保持 diff 尽量小。
```

### OpenCode 中很适合的用法
- 用 Plan 先整理多步任务
- 用 Build 改代码和跑命令
- 用项目里的 AGENTS.md 约束每次执行

---

## 五、建议的日常工作流

### 流程 1：新增一份 API 文档
1. 把源文档放到 `docs/`
2. 让 agent 解析并生成 `canonical_facts.jsonl`
3. 运行样本扩写脚本
4. 运行 JSON Schema 校验
5. 检查 stats 报告
6. 人工抽查高价值 action
7. 合并到训练集

### 流程 2：优化已有样本
1. 让 agent 找出短样本、低多样性样本、冲突样本
2. 只重生成 expression layer
3. 不改 canonical facts
4. 比较前后统计和评测结果
5. 保留更优版本

### 流程 3：做相似接口消歧
1. 选一组容易混淆的 API
2. 让 agent 生成 contrast samples
3. 增加 negative_terms
4. 重新跑 validator 和 confusion-focused eval

---

## 六、推荐给 agent 的常用任务指令

### 任务 A：补最小闭环
```text
读取 AGENTS.md。
检查仓库里是否具备 canonical facts、expanded samples、validator 三层。
缺什么补什么，但保持最小改动。
完成后输出执行步骤、结果文件、风险项。
```

### 任务 B：新增 SLB 样本
```text
读取 AGENTS.md。
从 docs/slb_api.md 提取事实到 canonical_facts.jsonl。
再把默认值、条件约束、错误码恢复、相似接口对比 四类样本扩写到 expanded 数据集。
运行校验并输出统计。
```

### 任务 C：提升 Engram 热知识质量
```text
读取 AGENTS.md。
从 expanded 数据集中筛选高频、高稳定、高约束价值的样本，生成一个 hotset。
优先保留 defaults、constraints、error recovery、contrast。
不要包含低置信度或 needs_review 条目。
```

### 任务 D：生成 PR 摘要
```text
根据本次 diff，生成 PR 说明：
- 数据来源
- 新增/修改样本数
- 校验结果
- 统计变化
- 待人工审核点
```

---

## 七、人工审核重点

每次 agent 跑完后，重点人工检查：
- action / field / error_code 是否被改写
- 条件约束是否被“合理化脑补”
- 相似接口对比是否准确
- train/eval 是否混了数据
- must_contain 是否真能命中正文

---

## 八、推荐落地顺序

第一阶段：
- 固定 AGENTS.md
- 跑通 canonical → expansion → validation

第二阶段：
- 增加 stats 和对比评测
- 增加 contrast / negative samples

第三阶段：
- 自动生成 hotset
- 接入训练与回归评估

---

## 九、最小成功标准

这个流程成功，不是看 agent 写了多少字，而是看：
- 是否保住事实正确性
- 是否把样本变得更适合 Engram 触发
- 是否能自动校验
- 是否能稳定复现
