"""
=======================================================================
SLB Rule 知识条目评估数据
=======================================================================
保存 SLB Rule 相关的 recall_prompts 和 expected_keywords，
用于知识召回测试和评估。

配套的知识文本文件位于 ./knowledge/ 目录下：
- 01_rule_login_flow.txt      : Rule系统登录流程
- 02_rule_modify_config.txt   : Rule界面变更配置流程
- 03_rule_add_config.txt      : Rule界面新增配置流程
- 04_rule_sql_insert.txt      : Rule数据库SQL新增规则
- 05_rule_db_schema.txt       : Rule数据库表结构与数据规范
- 06_rule_tag_naming.txt      : Tag标签命名规范

使用方式:
  python engram_local_demo.py --data_dir ./knowledge --epochs 20
=======================================================================
"""

# ── SLB Rule 知识召回测试提示词 ─────────────────────────────────────────

SLB_RULE_RECALL_PROMPTS = [
    # 条目 1: Rule 系统登录流程
    "如何登录Rule系统？",
    "Rule系统的入口在哪里？",
    "怎么找到Rule配置界面？",
    "webapp-rule服务怎么访问？",
    "openapi.endpoint在哪里找？",
    # 条目 2: Rule 变更配置流程
    "如何修改Rule规则配置？",
    "怎么变更SLB的Rule配置？",
    "ruleConfig页面怎么使用？",
    "修改Rule配置有什么注意事项？",
    # 条目 3: Rule 新增配置流程
    "如何新增Rule配置？",
    "怎么添加SLB的规则配置？",
    "ruleConfigInsert页面怎么用？",
    "新增配置的用户类型怎么选？",
    # 条目 4: SQL 新增规则
    "如何用SQL新增SLB Rule配置？",
    "rule_config表的INSERT语句怎么写？",
    "怎么通过数据库给租户添加quota规则？",
    "SLB规则配置的SQL模板是什么？",
    # 条目 5: 数据库表结构
    "rule_config表的结构是什么？",
    "SLB Rule数据库叫什么？",
    "rule_config有哪些关键字段？",
    "action字段有什么规范？",
    # 条目 6: Tag 命名规范
    "Rule系统的tag怎么写？",
    "SLB tag命名规范是什么？",
    "default、safe、uid开头的tag分别代表什么？",
    "如何给特定租户配置tag？",
    "region_id在tag中怎么使用？",
]


# ── SLB Rule 知识召回期望关键词 ─────────────────────────────────────────

SLB_RULE_EXPECTED_KEYWORDS = [
    # 条目 1: Rule 系统登录流程
    ["天基", "webapp-rule", "openapi.endpoint", "注册变量", "浏览器"],
    # 条目 2: Rule 变更配置流程
    ["ruleConfig", "slb", "action", "config", "刷缓存"],
    # 条目 3: Rule 新增配置流程
    ["ruleConfigInsert", "config", "slb", "用户类型", "user_id", "刷缓存"],
    # 条目 4: SQL 新增规则
    ["INSERT", "rule_config", "slb", "config", "user_id", "tag", "vm_lbs_quota"],
    # 条目 5: 数据库表结构
    ["rule_db", "rule_config", "product", "action", "config_type", "is_deleted"],
    # 条目 6: Tag 命名规范
    ["default", "safe", "uid", "region_id", "tag"],
]


# ── SLB Rule 完整知识条目（结构化数据，用于详细评估）──────────────────────

KNOWLEDGE_ENTRIES = [
    # 条目 1: Rule 系统界面登录流程 (Type C: procedure)
    {
        "content": "## Rule系统登录流程\n步骤 1: 登陆天基，搜索服务webapp-rule\n  操作: 在天基服务搜索框中搜索webapp-rule\n步骤 2: 查找openapi.endpoint\n  操作: 在服务的注册变量中查找openapi.endpoint的值\n步骤 3: 浏览器访问\n  操作: 在浏览器中输入openapi.endpoint地址，进入欢迎页",
        "paraphrases": [
            "登录Rule系统的方法：先从天基搜索webapp-rule服务，然后在注册变量里找到openapi.endpoint，最后用浏览器访问该地址进入欢迎页",
            "如何访问Rule配置系统：天基中搜索webapp-rule → 获取注册变量openapi.endpoint → 浏览器打开该URL"
        ],
        "recall_prompts": [
            "如何登录Rule系统？",
            "Rule系统的入口在哪里？",
            "怎么找到Rule配置界面？",
            "webapp-rule服务怎么访问？",
            "openapi.endpoint在哪里找？"
        ],
        "expected_keywords": ["天基", "webapp-rule", "openapi.endpoint", "注册变量", "浏览器"]
    },
    # 条目 2: Rule 界面变更配置流程 (Type C: procedure)
    {
        "content": "## Rule系统变更配置\n步骤 1: 进入配置页\n  操作: 在欢迎页URL后面加上/ruleConfig\n步骤 2: 搜索目标规则\n  操作: 产品code填slb，action填规则配置文档给出的action\n步骤 3: 修改配置\n  操作: 只修改config内容，其他字段不能变\n  注意: 其他字段修改可能导致配置失效\n步骤 4: 刷新缓存\n  操作: 修改完成后刷缓存使新配置生效",
        "paraphrases": [
            "变更Rule配置的方法：访问/ruleConfig路径，用product=slb和action筛选规则，仅修改config值，改完刷缓存生效",
            "修改SLB Rule规则：URL加/ruleConfig进入配置页，按产品code和action搜索，注意只能改config内容不能改其他字段"
        ],
        "recall_prompts": [
            "如何修改Rule规则配置？",
            "怎么变更SLB的Rule配置？",
            "ruleConfig页面怎么使用？",
            "修改Rule配置有什么注意事项？"
        ],
        "expected_keywords": ["ruleConfig", "slb", "action", "config", "刷缓存"]
    },
    # 条目 3: Rule 界面新增配置流程 (Type C: procedure)
    {
        "content": "## Rule系统新增配置\n步骤 1: 进入新增页\n  操作: 在欢迎页URL后面加上/ruleConfigInsert\n步骤 2: 填写配置\n  操作: 配置类型选config，产品code填slb，action填文档给出的action\n步骤 3: 选择用户类型\n  操作: 四选一，对于已有配置不能选默认配置否则会不生效或影响已有默认配置，建议选择针对具体用户填入user_id\n步骤 4: 刷新缓存\n  操作: 新增完成后刷缓存生效\n  注意: 重复配置只会生效一个",
        "paraphrases": [
            "新增Rule配置：访问/ruleConfigInsert，配置类型选config，产品code=slb，用户类型建议选具体用户填user_id避免影响默认配置，完成后刷缓存",
            "添加SLB Rule规则的步骤：URL加/ruleConfigInsert → 类型选config → 产品slb → 选用户类型（推荐具体用户）→ 刷缓存生效"
        ],
        "recall_prompts": [
            "如何新增Rule配置？",
            "怎么添加SLB的规则配置？",
            "ruleConfigInsert页面怎么用？",
            "新增配置的用户类型怎么选？"
        ],
        "expected_keywords": ["ruleConfigInsert", "config", "slb", "用户类型", "user_id", "刷缓存"]
    },
    # 条目 4: Rule 数据库 SQL 新增规则 (Type A: command_mapping)
    {
        "content": "## 指令: 通过SQL新增SLB Rule配置\n命令: INSERT IGNORE INTO rule_config (`product`, `component`, `action`, `user_id`, `bid`, `tag`, `config`, `config_var_type`, `config_type`, `concurrent_count_var`, `error_code`, `memo`, `gmt_created`, `gmt_modified`, `creator`, `modifier`, `is_deleted`) VALUES ('slb', NULL, '{action}', '{user_id}', NULL, '{tag}', '{config_value}', NULL, 'config', NULL, NULL, NULL, now(), now(), '{creator}', NULL, 0);\n参数: action=规则动作名(如vm_lbs_quota), user_id=租户UID, tag=标签格式(如uid_cn-gz-csa-d01), config=配置值, creator=修改人\n示例:\n>>> 给租户1773267149977932添加quota规则为290\nINSERT IGNORE INTO rule_config (`product`, `component`, `action`, `user_id`, `bid`, `tag`, `config`, `config_var_type`, `config_type`, `concurrent_count_var`, `error_code`, `memo`, `gmt_created`, `gmt_modified`, `creator`, `modifier`, `is_deleted`) VALUES ('slb', NULL, 'vm_lbs_quota', '1773267149977932', NULL, 'uid_cn-gz-csa-d01', '290', NULL, 'config', NULL, NULL, NULL, now(), now(), 'zfl.slb', NULL, 0);",
        "paraphrases": [
            "在rule_db数据库的rule_config表中用INSERT IGNORE INTO插入SLB规则配置，product=slb，config_type=config，is_deleted=0",
            "通过SQL直接往rule_config表插入规则，关键字段有product、action、user_id、tag、config、creator"
        ],
        "recall_prompts": [
            "如何用SQL新增SLB Rule配置？",
            "rule_config表的INSERT语句怎么写？",
            "怎么通过数据库给租户添加quota规则？",
            "SLB规则配置的SQL模板是什么？"
        ],
        "expected_keywords": ["INSERT", "rule_config", "slb", "config", "user_id", "tag", "vm_lbs_quota"]
    },
    # 条目 5: Rule 数据库表结构与数据规范 (Type D: structured_config)
    {
        "content": "rule_config表结构:\n  数据库: rule_db\n  表名: rule_config\n  product: slb\n  action: 规则动作名(统一小写，如vrouter_entry)\n  config_type: config(固定字符串)\n  config: 配置值(整数对应的字符串类型)\n  is_deleted: 0(有效)/1(删除)\n  creator: 修改人\n  gmt_created: 创建时间戳\n  gmt_modified: 修改时间戳",
        "paraphrases": [
            "SLB Rule数据库结构：数据库rule_db，表rule_config，核心字段包括product=slb、action(全小写)、config_type=config、config(字符串值)、is_deleted=0",
            "rule_config表关键字段说明：product标识产品slb，action用小写命名，config_type固定config，config是字符串格式的整数值"
        ],
        "recall_prompts": [
            "rule_config表的结构是什么？",
            "SLB Rule数据库叫什么？",
            "rule_config有哪些关键字段？",
            "action字段有什么规范？"
        ],
        "expected_keywords": ["rule_db", "rule_config", "product", "action", "config_type", "is_deleted"]
    },
    # 条目 6: Tag 标签命名规范 (Type B: fact_pair)
    {
        "content": "问: SLB Rule系统中tag标签的命名规范是什么？\n答: tag格式为${类型}_${region_id}，共三种类型：\n默认水位配置: default_${region_id}，例如default_cn_hangzhou\n安全水位配置: safe_${region_id}，例如safe_cn_hangzhou\n具体租户限制: uid_${region_id}，例如uid_cn_hangzhou\n注意: uid是固定字符串前缀，如uid_cn_beijin_1, uid_cn_region_hangzhou_1。Rule系统会对default、safe、uid这些特殊前缀进行识别。",
        "paraphrases": [
            "Rule系统tag有三种格式：default_regionId表示默认水位，safe_regionId表示安全水位，uid_regionId表示单租户限制，uid时需要配合user_id字段",
            "SLB tag命名规范：类型前缀加下划线加region_id，类型有default(默认)、safe(安全水位)、uid(具体租户)"
        ],
        "recall_prompts": [
            "Rule系统的tag怎么写？",
            "SLB tag命名规范是什么？",
            "default、safe、uid开头的tag分别代表什么？",
            "如何给特定租户配置tag？",
            "region_id在tag中怎么使用？"
        ],
        "expected_keywords": ["default", "safe", "uid", "region_id", "tag"]
    },
]