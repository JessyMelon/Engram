# Prompt Templates

## Inventory
```text
目标：为当前仓库建立初始认知地图，产出 inventory 文档。

任务范围：
- 全仓库
- 重点关注 README、构建配置、部署配置、主入口、顶层目录结构

输出文件：
- coder-llm-wiki/01-inventory/repo-map.md
- coder-llm-wiki/01-inventory/tech-stack.md
- coder-llm-wiki/01-inventory/entrypoints.md
- coder-llm-wiki/01-inventory/module-candidates.json

要求：
1. 不要做空泛总结，尽量依据代码和配置。
2. 明确列出主要应用、服务、包、目录职责。
3. entrypoints 需包含 HTTP、CLI、worker、cron、consumer 等入口。
4. 如果有不确定项，单独标注。
5. 所有关键结论尽量附文件路径。
```

## Module Analysis
```text
目标：分析模块 <MODULE_NAME>，生成高质量模块文档和证据文档。

任务范围：
- 目录：<PATHS>
- 相关测试目录：<TEST_PATHS>
- 如有必要可查看调用它的入口代码和上游调用方

输出文件：
- coder-llm-wiki/03-modules/<MODULE_NAME>.md
- coder-llm-wiki/08-evidence/<MODULE_NAME>.refs.md
- coder-llm-wiki/09-review/<MODULE_NAME>.questions.md

要求：
1. 使用固定模板写模块文档。
2. 区分事实与推断。无法直接从代码验证的内容放到 Open Questions 或 Risks。
3. 需要引用具体文件和行号作为 evidence。
4. 除模块本身外，必要时查看入口、配置和测试，以补足理解。
5. 重点识别调用方、依赖、数据读写、隐式约束和高风险改动点。
```

## Flow Analysis
```text
目标：分析流程 <FLOW_NAME>，生成流程文档。

任务范围：
- 从这些入口出发：<ENTRY_FILES>
- 允许沿调用链向下追踪
- 允许查看相关模块文档和测试

输出文件：
- coder-llm-wiki/04-flows/<FLOW_NAME>.md

要求：
1. 按模板写：Trigger、Entry Points、Preconditions、Main Path、Failure Paths、State Changes、External Calls、Related Modules、Risks、Tests、Evidence、Open Questions。
2. 主路径要按顺序描述，不要只列函数名。
3. 明确失败路径和补偿机制。
4. 标注外部调用和副作用。
5. 用具体文件和行号提供 evidence。
```

## Review
```text
目标：对已有 coder llm wiki 做一致性和证据校验。

检查范围：
- coder-llm-wiki/03-modules/
- coder-llm-wiki/04-flows/
- coder-llm-wiki/08-evidence/

输出文件：
- coder-llm-wiki/09-review/conflict-log.md
- coder-llm-wiki/09-review/open-questions.md
- coder-llm-wiki/09-review/human-review.md

要求：
1. 检查模块文档和流程文档是否互相一致。
2. 检查关键结论是否有 evidence 支撑。
3. 检查 evidence 是否足够具体。
4. 检查术语命名是否统一。
5. 将问题分成明确错误、证据不足、推断过强、待人工确认。
```

## Incremental Update
```text
目标：根据最近变更更新受影响的 coder llm wiki 文档。

任务范围：
- 读取当前 git diff
- 找出受影响模块和流程
- 更新对应文档与证据

输出：
- 修改已有 coder-llm-wiki 文档
- 在 coder-llm-wiki/09-review/ 中记录新增不确定项

要求：
1. 只更新受影响部分，不要重写整个 wiki。
2. 变更必须反映到模块文档、流程文档或 evidence 文档。
3. 如果现有文档与代码不一致，优先修正文档，并记录冲突来源。
```
