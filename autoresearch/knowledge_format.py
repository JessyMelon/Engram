"""
===========================================================================
Engram 知识格式定义和数据增强模块
===========================================================================
本文件定义了 Engram 训练所需的知识数据格式、数据增强方法和召回测试数据。

知识类型分类：
- Type A: command_mapping  - 指令-命令映射（如 GetRegions, ListClusters 等）
- Type B: fact_pair        - 事实-知识对（如实体定义、属性说明）
- Type C: procedure        - 流程-步骤文档（如操作流程）
- Type D: structured_config - 结构化配置（如系统配置）

主要导出：
- KNOWLEDGE_ENTRIES: 知识条目列表
- RECALL_TESTS: 召回测试数据
- build_training_text(): 构建训练文本
- build_validation_text(): 构建验证文本
===========================================================================
"""

from typing import List, Dict, Optional

# ── 知识条目列表 ─────────────────────────────────────────────────────────

KNOWLEDGE_ENTRIES: List[Dict] = [
    # ========== Type A: 指令-命令映射 ==========
    
    # 1. GetRegions 命令
    {
        "type": "command_mapping",
        "content": """## GetRegions - 获取 Region 列表

tianji GetRegions 命令用于获取当前天基环境下所有可用的 Region 列表及其编号(region no)。

命令格式:
```
tianji GetRegions
__func__  python3 -m cli.OpenAPI.api tianji Action GetRegions
```

示例输出:
```
>>> tianji GetRegions
{
    "cn-wulan-env82-d01": {
        "ApiIP": "http://cn-wulan-env82-d01-api.tianji.intra.env82.shuguang.com",
        "PortalIP": "http://cn-wulan-env82-d01.tianji.intra.env82.shuguang.com"
    }
}
```

返回值说明: 返回一个字典，键为 Region 名称，值包含 ApiIP 和 PortalIP 两个地址。""",
        "paraphrases": [
            "使用 tianji GetRegions 可以列出当前天基平台所有可用区域的信息，包括 API 接口地址和 Portal 门户地址。",
            "要查看天基环境中有哪些 Region 可用，执行 GetRegions 命令即可获取所有区域列表及对应的服务端点。",
            "GetRegions 是天基命令行工具的基础命令，用于发现当前环境下所有可用的地域配置信息。"
        ],
        "recall_prompts": [
            "获取当前天基环境下所有可用的 Region 列表及其编号",
            "查看天基有哪些可用区域",
            "列出所有 region 信息"
        ],
        "expected_keywords": ["GetRegions", "tianji", "Region", "ApiIP", "PortalIP"]
    },
    
    # 2. ListClusters 命令
    {
        "type": "command_mapping",
        "content": """## ListClusters - 获取集群列表

tianji ListClusters 命令用于根据产品名获取该产品下所有集群列表。

命令格式:
```
tianji ListClusters <ProjectName>
  <ProjectName>  产品名称(必填)，用于筛选该产品下的集群列表
__func__  python3 -m cli.OpenAPI.api tianji Action ListClusters ProjectName {ProjectName}
```

示例:
```
>>> tianji ListClusters ProjectName slb
[
    "slbCluster-A-20260108-0079",
    "slbCluster-B-20260108-0235",
    "slb_three_zone-A-20260110-025e",
    "slb_three_zone-C-20260111-026a",
    "slb_yaochi_Cluster-A-20260108-007a"
]

>>> tianji ListClusters ProjectName vpc
[
    "cenAzoneCluster-A-20260108-00cb",
    "vpcAzoneCluster-A-20260108-00c9",
    "vpcRegionAgCluster-A-20260108-00c6",
    "vpcRegionCluster-A-20260108-00cc"
]
```

参数说明: ProjectName 是必填参数，指定要查询的产品名称（如 slb、vpc 等）。""",
        "paraphrases": [
            "要查看某个产品下有哪些集群，使用 ListClusters 命令并指定 ProjectName 参数，如查询 slb 产品的所有集群。",
            "ListClusters 命令需要提供产品名作为必填参数，返回该产品下所有集群的名称列表。",
            "通过 tianji ListClusters ProjectName xxx 可以获取指定产品下的全部集群清单。"
        ],
        "recall_prompts": [
            "获取该产品下所有集群列表，产品名称为 slb",
            "查看 vpc 产品有哪些集群",
            "列出某个产品下的所有集群"
        ],
        "expected_keywords": ["ListClusters", "ProjectName", "tianji", "集群"]
    },
    
    # 3. ListServiceInstance 命令
    {
        "type": "command_mapping",
        "content": """## ListServiceInstance - 获取服务实例列表

tianji ListServiceInstance 命令用于获取指定集群下的服务实例列表，可选择是否返回详细信息。

命令格式:
```
tianji ListServiceInstance [ProjectName] <ClusterName> [detail]
  [ProjectName]  产品名称(可选)，不填时系统自动根据集群名推断
  <ClusterName>  集群名称(必填)，指定要查询服务实例的目标集群
  [detail]       是否返回详细信息(可选)，true 返回详细信息
__func__  python3 -m cli.OpenAPI.api tianji Action ListServiceInstance [ProjectName {ProjectName}] ClusterName {ClusterName} [detail {detail}]
```

示例:
```
>>> tianji ListServiceInstance ProjectName slb ClusterName slbCluster-A-20260108-0079
{
    "Version": "f5ccf4577e7dff5fa188912fb699ae315abcf4dc",
    "ClusterState": "normal",
    "ServiceInstances": [
        "tianji-hostservice", "slb-lvs", "slb-keyserver", "slb-monitor",
        "slb-controller", "os", "tianji", "hids-client",
        "tianji-sshtunnel-client", "slb-proxy", "tianji-dockerdaemon"
    ]
}

>>> tianji ListServiceInstance ClusterName slbCluster-A-20260108-0079 detail true
{
    "ExpectedInstances": {
        "slb-controller": { "State": "upgrading", "ExpectedServerRoleCount": 5 }
    }
}
```

返回值: 包含 Version、ClusterState、ServiceInstances 等字段，detail=true 时返回更详细信息。""",
        "paraphrases": [
            "ListServiceInstance 用于查询某个集群中部署了哪些服务实例，可以加 detail true 获取详细状态。",
            "要获取集群的服务列表，使用 ListServiceInstance 并指定 ClusterName，ProjectName 可选。",
            "查看集群下有哪些服务在运行，用 tianji ListServiceInstance ClusterName xxx 命令。"
        ],
        "recall_prompts": [
            "获取指定集群下的服务实例列表，集群名称为 slbCluster-A-20260108-0079",
            "查看集群 slbCluster-A-20260108-0079 有哪些服务",
            "列出某个集群下部署的所有服务实例"
        ],
        "expected_keywords": ["ListServiceInstance", "ClusterName", "ServiceInstances", "Version", "ClusterState"]
    },
    
    # 4. ListMachineByCluster 命令
    {
        "type": "command_mapping",
        "content": """## ListMachineByCluster - 根据集群查询机器列表

tianji ListMachineByCluster 命令用于根据集群名称查询该集群下的所有机器列表。

命令格式:
```
tianji ListMachineByCluster [ProjectName] <ClusterName>
  [ProjectName]  产品名称(可选)，不填时系统自动根据集群名推断
  <ClusterName>  集群名称(必填)，指定要查询机器列表的目标集群
__func__  python3 -m cli.OpenAPI.api tianji Action ListMachineByCluster [ProjectName {ProjectName}] ClusterName {ClusterName}
```

示例:
```
>>> tianji ListMachineByCluster ProjectName vpc ClusterName slbCluster-A-20260108-0079
[
    "c28d12002.cloud.d12.amtest82",
    "c28d12003.cloud.d12.amtest82",
    "c28d12004.cloud.d12.amtest82",
    "c28d12005.cloud.d12.amtest82",
    "c28d12010.cloud.d12.amtest82",
    "vm010082008063",
    "vm010082013189"
]
```

说明: 通过 /api/machines?cluster= 接口查询，返回机器 hostname 列表。""",
        "paraphrases": [
            "ListMachineByCluster 用于获取某个集群下所有机器的主机名列表，只需指定 ClusterName。",
            "要查看集群里有哪些机器，使用 ListMachineByCluster 命令并传入集群名称。",
            "获取集群机器清单可以用 tianji ListMachineByCluster ClusterName xxx。"
        ],
        "recall_prompts": [
            "根据集群名称查询该集群下的所有机器列表",
            "获取集群 slbCluster-A-20260108-0079 的机器列表",
            "查看某个集群有哪些机器"
        ],
        "expected_keywords": ["ListMachineByCluster", "ClusterName", "hostname", "机器"]
    },
    
    # 5. CheckoutCluster 命令
    {
        "type": "command_mapping",
        "content": """## CheckoutCluster - 检出集群配置

tianji CheckoutCluster 命令用于检出(查看)指定集群的配置信息，返回集群配置和版本号。

命令格式:
```
tianji CheckoutCluster [ProjectName] <ClusterName> [version] [extend] [plain]
  [ProjectName]  产品名称(可选)
  <ClusterName>  集群名称(必填)，指定要检出配置的目标集群
  [version]      版本号(可选)，指定检出的配置版本，不填则获取最新版本
  [extend]       是否展开模板继承(可选)，true 表示展开扩展配置
  [plain]        是否以纯文本模式返回(可选)
__func__ python3 -m cli.OpenAPI.api tianji Action CheckoutCluster [ProjectName {ProjectName}] ClusterName {ClusterName} [version {version}] [extend {extend}] [plain {plain}]
```

示例:
```
>>> tianji CheckoutCluster ProjectName slb ClusterName slb_three_zone-A-20260110-025e
[
    {
        "cluster.conf": "CloneTemplateBasis: MachineGroup\\nName: slb_three_zone-A-20260110-025e\\nProject: slb\\n",
        "kv.conf": "{ \\"KeyValues\\": { \\"CLUSTER_TYPE\\": \\"NETFRAME\\" } }",
        "machine_group.conf": "MachineGroups:\\n  SlbLvs:\\n  - c28d12006.cloud.d12.amtest82"
    },
    "5866c1475208c1f462bb76309e329ec6d9af1601"
]
```

返回值: [配置字典, 版本号] 数组，配置包含 cluster.conf、kv.conf、machine_group.conf 等。""",
        "paraphrases": [
            "CheckoutCluster 用于获取集群的基线配置信息，包括 cluster.conf、kv.conf 等配置文件内容。",
            "要查看某个集群的配置详情，使用 CheckoutCluster 命令，可以指定版本号获取历史配置。",
            "检出集群配置用 tianji CheckoutCluster ClusterName xxx，返回配置字典和版本哈希值。"
        ],
        "recall_prompts": [
            "获取指定集群的基线信息，集群名称为 slbCluster-A-20260108-0079",
            "检出集群 slb_three_zone-A-20260110-025e 的配置",
            "查看某个集群的配置文件内容"
        ],
        "expected_keywords": ["CheckoutCluster", "ClusterName", "cluster.conf", "kv.conf", "Version"]
    },
    
    # 6. CheckoutService 命令
    {
        "type": "command_mapping",
        "content": """## CheckoutService - 检出服务配置

tianji CheckoutService 命令用于检出(查看)指定服务实例的配置信息，返回集群配置、服务配置和版本号。

命令格式:
```
tianji CheckoutService [ProjectName] <ClusterName> <ServiceName> [version] [extend] [plain]
  [ProjectName]  产品名称(可选)
  <ClusterName>  集群名称(必填)，指定服务实例所在的集群
  <ServiceName>  服务名称(必填)，指定要检出配置的目标服务
  [version]      版本号(可选)，指定检出的配置版本
  [extend]       是否展开模板继承(可选)
  [plain]        是否以纯文本模式返回(可选)
__func__ python3 -m cli.OpenAPI.api tianji Action CheckoutService [ProjectName {ProjectName}] ClusterName {ClusterName} ServiceName {ServiceName} [version {version}] [extend {extend}] [plain {plain}]
```

示例:
```
>>> tianji CheckoutService ClusterName slbCluster-A-20260108-0079 ServiceName slb-controller
[
    { "cluster.conf": "...", "kv.conf": "...", "machine_group.conf": "..." },
    {
        "dependency.conf": "{ \\"Dependency\\": { \\"ServiceTest#\\": {...} } }",
        "role.conf": "ServerRoles:\\n  SlbAg#:\\n  - vm010082008063",
        "template.conf": "BaseTemplate:\\n  TemplateName: TMPL-SLB-CONTROLLER"
    },
    "f5ccf4577e7dff5fa188912fb699ae315abcf4dc"
]
```

返回值: [集群配置, 服务配置, 版本号] 数组，服务配置包含 dependency.conf、role.conf 等。""",
        "paraphrases": [
            "CheckoutService 用于获取指定服务实例的完整配置，包括服务依赖、角色配置、模板信息等。",
            "要查看某个服务的配置详情，使用 CheckoutService 命令并同时指定 ClusterName 和 ServiceName。",
            "检出服务配置用 tianji CheckoutService ClusterName xxx ServiceName yyy，返回服务级别配置。"
        ],
        "recall_prompts": [
            "获取指定服务实例的配置信息，集群名称为 slbCluster-A-20260108-0079，服务名称为 slb-controller",
            "检出服务 slb-controller 在集群 slbCluster-A-20260108-0079 的配置",
            "查看某个服务的基线配置"
        ],
        "expected_keywords": ["CheckoutService", "ClusterName", "ServiceName", "dependency.conf", "role.conf"]
    },
    
    # 7. GetClusterResources 命令
    {
        "type": "command_mapping",
        "content": """## GetClusterResources - 获取集群资源信息

tianji GetClusterResources 命令用于获取集群的资源信息，包含 dns、db 等资源类型。

命令格式:
```
tianji GetClusterResources [ProjectName] <ClusterName>
  [ProjectName]  产品名称(可选但推荐填写，否则可能报错)
  <ClusterName>  集群名称(必填)，指定要查询资源信息的目标集群
__func__ python3 -m cli.OpenAPI.api tianji Action GetClusterResources [ProjectName {ProjectName}] ClusterName {ClusterName}
```

示例:
```
>>> tianji GetClusterResources ClusterName slbCluster-A-20260108-0079 ProjectName slb
{
    "code": 200,
    "data": [
        {
            "instance_id": "14a9975940ede77286d3d9d337c4f448b0ef835d",
            "app": "slb-ops-api",
            "name": "slbopsapi",
            "parameters": "{\\"domain\\":\\"slbopsapi.intra.env82.shuguang.com\\"}",
            "result": "{\\"dns\\":\\"slbopsapi.intra.env82.shuguang.com\\"}",
            "status": "done",
            "type": "dns"
        },
        {
            "app": "slb-control-master",
            "name": "stats",
            "type": "db",
            "result": "{\\"db_host\\":\\"stats.mysql.minirds.intra.env82.shuguang.com\\"}"
        }
    ]
}
```

返回值: 包含 code、data、message，data 是资源列表，每个资源有 type(dns/db)、status 等属性。""",
        "paraphrases": [
            "GetClusterResources 用于查询集群关联的资源，如 DNS 域名、数据库连接等基础设施资源。",
            "要获取集群的 dns、db 等资源信息，使用 GetClusterResources 命令，建议同时填写 ProjectName。",
            "查看集群资源可以用 tianji GetClusterResources ClusterName xxx ProjectName yyy。"
        ],
        "recall_prompts": [
            "获取集群资源信息，集群名称为 slbCluster-A-20260108-0079",
            "查看集群 slbCluster-A-20260108-0079 的 dns 和 db 资源",
            "获取某个集群的资源列表"
        ],
        "expected_keywords": ["GetClusterResources", "ClusterName", "ProjectName", "dns", "db", "type"]
    },
    
    # 8. DockerLogs 命令
    {
        "type": "command_mapping",
        "content": """## DockerLogs - 获取 Docker 容器日志

tianji DockerLogs 命令用于获取指定集群中某个服务实例应用的 Docker 容器日志。

命令格式:
```
tianji DockerLogs <ProjectName> <ClusterName> <HostName> <ServiceName> <ServerRoleName> <AppName> <Machine> <Count>
  <ProjectName>     产品名称(必填)，指定目标产品
  <ClusterName>     集群名称(必填)，指定目标集群
  <HostName>        主机名称(必填)，指定目标机器的 hostname
  <ServiceName>     服务名称(必填)，指定目标服务实例
  <ServerRoleName>  服务角色名称(必填)，指定目标 ServerRole
  <AppName>         应用名称(必填)，指定目标 App
  <Machine>         机器标识(必填)，指定要获取日志的机器
  <Count>           日志行数(必填)，指定返回的日志条数
__func__ python3 -m cli.OpenAPI.api tianji Action DockerLogs ProjectName {ProjectName} ClusterName {ClusterName} HostName {HostName} ServiceName {ServiceName} ServerRoleName {ServerRoleName} AppName {AppName} Machine {Machine} Count {Count}
```

说明: 该命令需要较多参数，用于精确定位某个 Docker 容器并获取其日志内容。参数包括产品、集群、主机、服务、角色、应用、机器和日志行数。""",
        "paraphrases": [
            "DockerLogs 命令用于获取指定容器的日志，需要提供完整的定位信息：集群、主机、服务、角色、应用等。",
            "要查看 Docker 容器日志，使用 DockerLogs 并指定所有必填参数，Count 指定获取的日志行数。",
            "获取容器日志用 tianji DockerLogs，需要同时提供 ProjectName、ClusterName、HostName、ServiceName 等参数。"
        ],
        "recall_prompts": [
            "获取 Docker 容器日志",
            "查看某个服务的 Docker 日志",
            "获取指定应用的容器日志"
        ],
        "expected_keywords": ["DockerLogs", "ProjectName", "ClusterName", "HostName", "ServiceName", "ServerRoleName", "Count"]
    },
    
    # ========== Type B: 事实-知识对 ==========
    
    # 9. 通用参数说明
    {
        "type": "fact_pair",
        "content": """## tianji 命令行通用参数说明

天基命令行工具的通用参数定义:

| 参数名 | 说明 |
|--------|------|
| region_no | 地域ID，标识天基环境中的特定区域 |
| bid | 渠道ID，用于区分不同的访问渠道 |
| idkp | 阿里云云帐号ID，也称为 aliUid |
| ProjectName | 产品名称，如 slb、vpc 等 |
| ClusterName | 集群名称，格式通常为 {product}{type}-{zone}-{date}-{id} |
| ServiceName | 服务名称，如 slb-controller、slb-proxy 等 |
| ServerRole | 服务角色，如 slb-controller.SlbAg# |
| HostName | 主机名称，机器的 hostname |

命令行由三部分组成:
1. 模板 - 用于匹配命令行(使用缩进表示层级)
2. 参数 - 用于匹配命令行参数(包括 <必填参数> 和 [可选参数])
3. 实际运行命令 - 由 __func__ 标记""",
        "paraphrases": [
            "天基命令行参数包括 region_no(地域)、ProjectName(产品)、ClusterName(集群)、ServiceName(服务)等。",
            "tianji 命令的参数格式中，<> 表示必填参数，[] 表示可选参数，__func__ 是实际执行的命令。"
        ],
        "recall_prompts": [
            "天基命令行的通用参数有哪些",
            "tianji 命令参数说明"
        ],
        "expected_keywords": ["region_no", "ProjectName", "ClusterName", "ServiceName", "__func__"]
    },
    
    # 10. 集群命名规范
    {
        "type": "fact_pair",
        "content": """## 天基集群命名规范

天基集群名称的命名格式和示例:

命名格式: {产品名}{类型}-{可用区}-{日期}-{编号}

示例:
- slbCluster-A-20260108-0079: slb 产品的 Cluster 类型，A 可用区，创建日期 20260108
- slb_three_zone-A-20260110-025e: slb 三可用区部署集群
- vpcAzoneCluster-A-20260108-00c9: vpc 产品的 AzoneCluster 类型
- cenAzoneCluster-A-20260108-00cb: cen 产品的集群

常见产品代码:
- slb: 负载均衡服务
- vpc: 虚拟私有云
- cen: 云企业网
- ecs: 云服务器""",
        "paraphrases": [
            "天基集群名称格式为 产品名+类型-可用区-日期-编号，如 slbCluster-A-20260108-0079。",
            "集群命名包含产品信息、可用区和创建时间，便于识别和管理。"
        ],
        "recall_prompts": [
            "天基集群的命名规范是什么",
            "集群名称格式"
        ],
        "expected_keywords": ["slbCluster", "vpcAzoneCluster", "产品", "可用区", "日期"]
    },
    
    # ========== Type C: 流程-步骤文档 ==========
    
    # 11. 查询集群信息流程
    {
        "type": "procedure",
        "content": """## 天基集群信息查询流程

完整查询某个集群信息的步骤:

### 步骤 1: 获取可用区域
```
tianji GetRegions
```
获取当前环境所有可用的 Region 列表。

### 步骤 2: 列出产品集群
```
tianji ListClusters ProjectName {产品名}
```
根据产品名(如 slb)获取该产品下所有集群。

### 步骤 3: 获取集群服务实例
```
tianji ListServiceInstance ClusterName {集群名}
```
获取指定集群下部署的服务实例列表。

### 步骤 4: 获取集群机器列表
```
tianji ListMachineByCluster ClusterName {集群名}
```
获取集群下所有机器的 hostname。

### 步骤 5: 获取集群资源
```
tianji GetClusterResources ProjectName {产品名} ClusterName {集群名}
```
获取集群的 dns、db 等资源信息。

### 步骤 6: 获取集群配置
```
tianji CheckoutCluster ClusterName {集群名}
```
检出集群的详细配置信息。""",
        "paraphrases": [
            "查询集群信息的完整流程：先 GetRegions 查区域，再 ListClusters 查集群列表，然后逐步深入查看详情。",
            "天基集群查询通常从 GetRegions 开始，依次使用 ListClusters、ListServiceInstance、ListMachineByCluster 等命令。"
        ],
        "recall_prompts": [
            "如何完整查询一个集群的信息",
            "天基集群查询流程"
        ],
        "expected_keywords": ["GetRegions", "ListClusters", "ListServiceInstance", "ListMachineByCluster", "CheckoutCluster"]
    },
    
    # ========== Type D: 结构化配置 ==========
    
    # 12. 命令分类总览
    {
        "type": "structured_config",
        "content": """## tianji 命令行命令分类

天基命令行工具的命令按功能分类:

| 分类 | 命令 | 说明 |
|------|------|------|
| Region 信息 | GetRegions | 获取所有可用 Region 列表 |
| 集群管理 | ListClusters | 根据产品名获取集群列表 |
| 服务实例 | ListServiceInstance | 获取集群下的服务实例列表 |
| 机器查询 | ListMachineByCluster | 根据集群名查询机器列表 |
| 机器查询 | ListClusterMachine | 获取集群下的机器列表(另一接口) |
| 机器查询 | ListMachines | 获取机器详细信息 |
| 角色查询 | ListClusterServerRoles | 获取集群下所有 ServerRole |
| 角色查询 | ListMachineSRByCluster | 根据集群和角色查询机器 |
| 基线配置 | CheckoutCluster | 检出集群配置信息 |
| 基线配置 | CheckoutService | 检出服务实例配置 |
| 集群资源 | GetClusterResources | 获取集群资源(dns、db) |
| 文件目录 | FetchFileList | 获取指定路径文件列表 |
| 文件目录 | FetchAllDir | 获取应用下所有目录结构 |
| Docker | DockerLogs | 获取 Docker 容器日志 |
| 实例信息 | GetInstanceInfo | 获取组件实例列表信息 |""",
        "paraphrases": [
            "tianji 命令分为 Region、集群管理、服务实例、机器查询、基线配置、资源、Docker 等几大类。",
            "天基命令行工具提供了完整的集群、服务、机器、配置查询能力。"
        ],
        "recall_prompts": [
            "tianji 有哪些命令",
            "天基命令行命令分类"
        ],
        "expected_keywords": ["GetRegions", "ListClusters", "CheckoutCluster", "GetClusterResources", "DockerLogs"]
    },
]

# ── 召回测试数据 ─────────────────────────────────────────────────────────

RECALL_TESTS: List[Dict] = [
    # 1. GetClusterResources 召回测试
    {
        "prompt": "获取集群资源信息,集群名称为 slbCluster-A-20260108-0079",
        "must_contain": ["GetClusterResources", "ProjectName", "ClusterName"],
        "must_not_contain": ["ListClusters", "CheckoutCluster"],
        "score_method": "keyword_match",
    },
    
    # 2. GetRegions 召回测试
    {
        "prompt": "获取当前天基环境下所有可用的 Region 列表及其编号(region no)",
        "must_contain": ["GetRegions", "tianji", "Region"],
        "must_not_contain": ["ListClusters", "ProjectName"],
        "score_method": "keyword_match",
    },
    
    # 3. ListClusters 召回测试
    {
        "prompt": "获取该产品下所有集群列表，产品名称为 slb",
        "must_contain": ["ListClusters", "ProjectName", "slb"],
        "must_not_contain": ["GetRegions", "CheckoutCluster"],
        "score_method": "keyword_match",
    },
    
    # 4. ListServiceInstance 召回测试
    {
        "prompt": "获取指定集群下的服务实例列表,集群名称为 slbCluster-A-20260108-0079",
        "must_contain": ["ListServiceInstance", "ClusterName", "ServiceInstances"],
        "must_not_contain": ["ListClusters", "GetClusterResources"],
        "score_method": "keyword_match",
    },
    
    # 5. CheckoutCluster 召回测试
    {
        "prompt": "获取指定集群的基线信息，集群名称为 slbCluster-A-20260108-0079",
        "must_contain": ["CheckoutCluster", "ClusterName", "cluster.conf"],
        "must_not_contain": ["CheckoutService", "GetClusterResources"],
        "score_method": "keyword_match",
    },
    
    # 6. CheckoutService 召回测试
    {
        "prompt": "获取指定服务实例的配置信息，集群名称为 slbCluster-A-20260108-0079，服务名称为 slb-controller",
        "must_contain": ["CheckoutService", "ClusterName", "ServiceName"],
        "must_not_contain": ["CheckoutCluster", "ListServiceInstance"],
        "score_method": "keyword_match",
    },
    
    # 7. ListMachineByCluster 召回测试
    {
        "prompt": "根据集群名称查询该集群下的所有机器列表",
        "must_contain": ["ListMachineByCluster", "ClusterName", "hostname"],
        "must_not_contain": ["ListClusters", "GetRegions"],
        "score_method": "keyword_match",
    },
    
    # 8. DockerLogs 召回测试
    {
        "prompt": "获取 Docker 容器日志，需要指定服务和机器",
        "must_contain": ["DockerLogs", "ServiceName", "HostName"],
        "must_not_contain": ["ListMachineByCluster", "CheckoutService"],
        "score_method": "keyword_match",
    },
    
    # 9. 查询流程测试
    {
        "prompt": "如何完整查询一个集群的信息",
        "must_contain": ["GetRegions", "ListClusters", "ListServiceInstance"],
        "must_not_contain": [],
        "score_method": "keyword_match",
    },
    
    # 10. 命令分类测试
    {
        "prompt": "天基命令行有哪些命令分类",
        "must_contain": ["Region", "集群", "服务", "机器"],
        "must_not_contain": [],
        "score_method": "keyword_match",
    },
    
    # 11. 参数说明测试
    {
        "prompt": "tianji 命令的通用参数有哪些",
        "must_contain": ["region_no", "ProjectName", "ClusterName"],
        "must_not_contain": [],
        "score_method": "keyword_match",
    },
    
    # 12. 集群命名规范测试
    {
        "prompt": "天基集群的命名格式是什么",
        "must_contain": ["slbCluster", "产品", "可用区"],
        "must_not_contain": [],
        "score_method": "keyword_match",
    },
]


# ── 构建训练文本函数 ─────────────────────────────────────────────────────

def build_training_text(entries: Optional[List[Dict]] = None) -> str:
    """
    将知识条目组装为训练文本。
    
    对每条知识，拼接：原文 + 改写变体 + QA 对（如果有）。
    用 \\n\\n 分隔不同知识块。
    
    Args:
        entries: 知识条目列表，默认使用 KNOWLEDGE_ENTRIES
        
    Returns:
        完整训练文本字符串
    """
    if entries is None:
        entries = KNOWLEDGE_ENTRIES
    
    text_blocks = []
    
    # 添加文档头部
    header = """# tianji 命令行注册模板
通用参数说明, region_no : 地域ID; bid : 渠道ID; idkp : 阿里云云帐号ID,或者叫 aliUid;
命令行由：模板，参数，实际运行命令三部分组成。模板用来匹配命令行(使用缩进表示层级)，参数用来匹配命令行参数(包括：<必填参数>,[可选参数])，实际运行命令(由 __func__ 标记)。
"""
    text_blocks.append(header)
    
    for entry in entries:
        # 添加主内容
        text_blocks.append(entry["content"])
        
        # 添加改写变体作为额外训练数据
        if "paraphrases" in entry and entry["paraphrases"]:
            paraphrase_section = "\n### 补充说明\n"
            for i, para in enumerate(entry["paraphrases"], 1):
                paraphrase_section += f"- {para}\n"
            text_blocks.append(paraphrase_section)
        
        # 如果有召回提示词，可以生成 QA 对
        if "recall_prompts" in entry and entry["recall_prompts"]:
            qa_section = "\n### 常见问题\n"
            for prompt in entry["recall_prompts"][:2]:  # 只取前两个
                qa_section += f"Q: {prompt}\n"
                # 从 content 中提取答案的关键部分
                if entry["type"] == "command_mapping":
                    qa_section += f"A: 使用 tianji 命令行工具可以完成此操作。\n\n"
            text_blocks.append(qa_section)
    
    # 用双换行分隔不同知识块
    full_text = "\n\n".join(text_blocks)
    
    return full_text


def build_validation_text(entries: Optional[List[Dict]] = None, ratio: float = 0.3) -> str:
    """
    构建验证文本（取知识的子集或不同改写）。
    
    Args:
        entries: 知识条目列表，默认使用 KNOWLEDGE_ENTRIES
        ratio: 用于验证的比例，默认 0.3
        
    Returns:
        验证文本字符串
    """
    if entries is None:
        entries = KNOWLEDGE_ENTRIES
    
    text_blocks = []
    
    # 选取部分条目用于验证
    num_validation = max(1, int(len(entries) * ratio))
    validation_entries = entries[:num_validation]
    
    for entry in validation_entries:
        # 使用改写变体而不是原始内容（如果有）
        if "paraphrases" in entry and entry["paraphrases"]:
            # 使用不同的改写版本
            text_blocks.append(f"## 知识点\n{entry['paraphrases'][-1]}")
        else:
            # 如果没有改写，使用原始内容的摘要
            content_lines = entry["content"].split("\n")
            summary = "\n".join(content_lines[:10])  # 只取前10行作为验证
            text_blocks.append(summary)
        
        # 添加验证问题
        if "recall_prompts" in entry and entry["recall_prompts"]:
            qa_section = "\n验证问题:\n"
            for prompt in entry["recall_prompts"]:
                qa_section += f"- {prompt}\n"
            text_blocks.append(qa_section)
    
    return "\n\n".join(text_blocks)


def get_knowledge_stats() -> Dict:
    """
    获取知识数据的统计信息。
    
    Returns:
        包含统计信息的字典
    """
    stats = {
        "total_entries": len(KNOWLEDGE_ENTRIES),
        "by_type": {},
        "total_paraphrases": 0,
        "total_recall_prompts": 0,
        "total_tests": len(RECALL_TESTS),
    }
    
    for entry in KNOWLEDGE_ENTRIES:
        entry_type = entry.get("type", "unknown")
        stats["by_type"][entry_type] = stats["by_type"].get(entry_type, 0) + 1
        
        if "paraphrases" in entry:
            stats["total_paraphrases"] += len(entry["paraphrases"])
        
        if "recall_prompts" in entry:
            stats["total_recall_prompts"] += len(entry["recall_prompts"])
    
    return stats


def validate_entry(entry: Dict) -> List[str]:
    """
    验证单个知识条目的格式是否正确。
    
    Args:
        entry: 知识条目字典
        
    Returns:
        错误信息列表，如果为空表示验证通过
    """
    errors = []
    
    required_fields = ["type", "content"]
    for field in required_fields:
        if field not in entry:
            errors.append(f"缺少必填字段: {field}")
    
    valid_types = ["command_mapping", "fact_pair", "procedure", "structured_config"]
    if entry.get("type") not in valid_types:
        errors.append(f"无效的类型: {entry.get('type')}, 应为 {valid_types} 之一")
    
    if "paraphrases" in entry:
        if not isinstance(entry["paraphrases"], list):
            errors.append("paraphrases 应为列表类型")
        elif len(entry["paraphrases"]) < 2:
            errors.append("paraphrases 应至少包含 2 个变体")
    
    if "expected_keywords" in entry:
        if not isinstance(entry["expected_keywords"], list):
            errors.append("expected_keywords 应为列表类型")
    
    return errors


def validate_all_entries() -> Dict[int, List[str]]:
    """
    验证所有知识条目的格式。
    
    Returns:
        字典，键为条目索引，值为错误信息列表
    """
    all_errors = {}
    for i, entry in enumerate(KNOWLEDGE_ENTRIES):
        errors = validate_entry(entry)
        if errors:
            all_errors[i] = errors
    return all_errors


# ── 模块测试入口 ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 打印统计信息
    stats = get_knowledge_stats()
    print("=" * 60)
    print("知识数据统计信息")
    print("=" * 60)
    print(f"知识条目总数: {stats['total_entries']}")
    print(f"按类型分布: {stats['by_type']}")
    print(f"改写变体总数: {stats['total_paraphrases']}")
    print(f"召回提示词总数: {stats['total_recall_prompts']}")
    print(f"召回测试总数: {stats['total_tests']}")
    
    # 验证格式
    errors = validate_all_entries()
    if errors:
        print("\n格式验证发现问题:")
        for idx, err_list in errors.items():
            print(f"  条目 {idx}: {err_list}")
    else:
        print("\n格式验证通过 ✓")
    
    # 构建训练文本并打印大小
    training_text = build_training_text()
    print(f"\n训练文本大小: {len(training_text)} 字符 ({len(training_text.encode('utf-8'))} 字节)")
    
    validation_text = build_validation_text()
    print(f"验证文本大小: {len(validation_text)} 字符 ({len(validation_text.encode('utf-8'))} 字节)")
    
    print("\n" + "=" * 60)
    print("训练文本预览 (前 500 字符):")
    print("=" * 60)
    print(training_text[:500])
