"""
=======================================================================
知识数据 & 召回提示词配置
=======================================================================
将 EXAMPLE_KNOWLEDGE 和 RECALL_PROMPTS 独立出来，方便修改和扩展。

- EXAMPLE_KNOWLEDGE: 内置示例知识文本，用于训练 Engram
- RECALL_PROMPTS:    知识召回测试的提示词列表
=======================================================================
"""

# ── 内置示例知识 ─────────────────────────────────────────────────────

EXAMPLE_KNOWLEDGE = """
# tianji 命令行注册模板
通用参数说明, region_no : 地域ID; bid : 渠道ID; idkp : 阿里云云帐号ID,或者叫 aliUid;
命令行由：模板，参数，实际运行命令三部分组成。模板用来匹配命令行(使用缩进表示层级)，参数用来匹配命令行参数(包括：<必填参数>,[可选参数])，实际运行命令(由 __func__ 标记)。

## 大纲

| 分类 | 命令 | 说明 |
|------|------|------|
| Region 信息 | GetRegions | 获取当前天基环境下所有可用的 Region 列表 |
| 集群管理 | ListClusters | 根据产品名获取该产品下所有集群列表 |
| 服务实例 | ListServiceInstance | 获取指定集群下的服务实例列表 |
| 机器与角色查询 | ListMachineByCluster | 根据集群名称查询该集群下的所有机器列表 |
| | ListClusterMachine | 获取指定集群下的机器列表（另一接口） |
| | ListMachines | 获取指定集群下所有机器的详细信息 |
| | ListClusterServerRoles | 获取指定集群下所有 ServerRole 列表 |
| | ListMachineSRByCluster | 根据集群和服务角色查询机器列表 |
| | ListMachineByProjectAndSR | 根据产品名和服务角色查询机器列表 |
| | ListMachineByIP | 根据 IP 地址查询对应的机器信息 |
| | GetClusterNamesOfService | 获取指定服务所关联的所有集群名称 |
| | ListInstanceInService | 获取指定服务下的所有服务实例列表 |
| 基线配置 | CheckoutCluster | 检出指定集群的配置信息 |
| | CheckoutService | 检出指定服务实例的配置信息 |
| 集群资源 | GetClusterResources | 获取集群的资源信息（dns、db 等） |
| 文件与目录 | FetchFileList | 获取指定路径下的文件列表 |
| | FetchAllDir | 获取应用下的所有目录结构 |
| Docker | DockerLogs | 获取 Docker 容器日志 |
| 实例信息 | GetInstanceInfo | 获取指定组件的实例列表信息 |

## 获取当前环境的region信息

```test
tianji    天基(Tianji)配置管理平台命令行工具，用于查询和管理集群、服务、机器等资源信息
  GetRegions  获取当前天基环境下所有可用的 Region 列表及其编号(region no)
    __func__  python3 -m cli.OpenAPI.api tianji Action GetRegions
    __help__ 
```

示例:
```
>>> tianji GetRegions
{
    "cn-wulan-env82-d01": {
        "ApiIP": "http://cn-wulan-env82-d01-api.tianji.intra.env82.shuguang.com",
        "PortalIP": "http://cn-wulan-env82-d01.tianji.intra.env82.shuguang.com"
    }
}
```


### 获取集群列表

```
  ListClusters  根据产品名获取该产品下所有集群列表
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
    "slb_yaochi_Cluster-A-20260108-007a",
    "slb_yaochi_Cluster-B-20260108-0236",
    "slb_yaochi_location_only-C-20260111-0269"
]

>>> tianji ListClusters ProjectName vpc
[
    "cenAzoneCluster-A-20260108-00cb",
    "vpcAzoneCluster-A-20260108-00c9",
    "vpcAzoneCluster-A-20260110-025f",
    "vpcRegionAgCluster-A-20260108-00c6",
    "vpcRegionCluster-A-20260108-00cc",
    ...
]
```

### 获取服务实例列表

```
  ListServiceInstance  获取指定集群下的服务实例列表，可选择是否返回详细信息
    [ProjectName]  产品名称(可选)，不填时系统自动根据集群名推断
    <ClusterName>  集群名称(必填)，指定要查询服务实例的目标集群
    [detail]  是否返回详细信息(可选)，true 返回详细信息，false 或不填返回基本列表
    __func__  python3 -m cli.OpenAPI.api tianji Action ListServiceInstance [ProjectName {ProjectName}]  ClusterName {ClusterName} [detail {detail}]
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
        "tianji-sshtunnel-client", "slb-proxy", "tianji-dockerdaemon",
        "staragent-client", "slb-ops-api", "srs-client"
    ],
    "ExpectedInstances": null,
    "DeletingInstances": null
}

>>> tianji ListServiceInstance ProjectName slb ClusterName slbCluster-A-20260108-0079 detail true
{
    "Version": "f5ccf4577e7dff5fa188912fb699ae315abcf4dc",
    "ClusterState": "normal",
    "ServiceInstances": [...],
    "ExpectedInstances": {
        "hids-client": { "State": "good", "ExpectedServerRoleCount": 1, "DeletingServerRoleCount": 0, "GoodExpectedServerRoleCount": 1 },
        "slb-controller": { "State": "upgrading", "ExpectedServerRoleCount": 5, "DeletingServerRoleCount": 0, "GoodExpectedServerRoleCount": 4 },
        "slb-proxy": { "State": "good", "ExpectedServerRoleCount": 1, "DeletingServerRoleCount": 0, "GoodExpectedServerRoleCount": 1 },
        ...
    },
    "DeletingInstances": {}
}
```

### 获取集群，服务，服务角色，机器信息

```
  ListMachineByCluster  根据集群名称查询该集群下的所有机器列表(通过 /api/machines?cluster= 接口)
    [ProjectName]  产品名称(可选)，不填时系统自动根据集群名推断
    <ClusterName>  集群名称(必填)，指定要查询机器列表的目标集群
    __func__  python3 -m cli.OpenAPI.api tianji Action ListMachineByCluster [ProjectName {ProjectName}] ClusterName {ClusterName}

  ListClusterMachine  获取指定集群下的机器列表(通过 /api/projects/{project}/clusters/{cluster}/machines 接口)
    [ProjectName]  产品名称(可选)，不填时系统自动根据集群名推断
    <ClusterName>  集群名称(必填)，指定要查询机器的目标集群
    __func__  python3 -m cli.OpenAPI.api tianji Action ListClusterMachine [ProjectName {ProjectName}] ClusterName {ClusterName}

  ListMachines  获取指定集群下所有机器的详细信息列表(通过 /api/projects/{project}/clusters/{cluster}/machinesinfo 接口，包含机器的完整属性)
    [ProjectName]  产品名称(可选)，不填时系统自动根据集群名推断
    <ClusterName>  集群名称(必填)，指定要查询机器详情的目标集群
    __func__  python3 -m cli.OpenAPI.api tianji Action ListMachines [ProjectName {ProjectName}] ClusterName {ClusterName}

  ListClusterServerRoles  获取指定集群下所有 ServerRole(服务角色) 的列表
    <ProjectName>  产品名称(必填)，指定目标产品
    <ClusterName>  集群名称(必填)，指定要查询 ServerRole 的目标集群
    __func__ python3 -m cli.OpenAPI.api tianji Action ListClusterServerRoles ProjectName {ProjectName} ClusterName {ClusterName}

  ListMachineSRByCluster  根据集群名称和服务角色查询机器与 ServerRole 的对应关系列表
    [ProjectName]  产品名称(可选)，不填时系统自动根据集群名推断
    <ClusterName>  集群名称(必填)，指定要查询的目标集群
    <ServerRole>  服务角色名称(必填)，指定要筛选的 ServerRole，用于查询该角色下的机器列表
    __func__ python3 -m cli.OpenAPI.api tianji Action ListMachineSRByCluster [ProjectName {ProjectName}] ClusterName {ClusterName} ServerRole {ServerRole}

  ListMachineByProjectAndSR  根据产品名和服务角色查询对应的机器列表
    <ProjectName>  产品名称(必填)，指定目标产品
    <ServerRole>  服务角色名称(必填)，指定要筛选的 ServerRole
    __func__ python3 -m cli.OpenAPI.api tianji Action ListMachineByProjectAndSR ProjectName {ProjectName} ServerRole {ServerRole}

  ListMachineByIP  根据 IP 地址查询对应的机器信息
    <ip>  机器的 IP 地址(必填)，用于精确查询该 IP 对应的机器信息
    __func__ python3 -m cli.OpenAPI.api tianji Action ListMachineByIP ip {ip}

  GetClusterNamesOfService  获取指定服务(Service)所关联的所有集群名称列表
    <ServiceName>  服务名称(必填)，用于查询该服务所部署的所有集群
    __func__ python3 -m cli.OpenAPI.api tianji Action GetClusterNamesOfService ServiceName {ServiceName}

  ListInstanceInService  获取指定服务下的所有服务实例列表(包含每个实例的 ClusterName 等信息)
    <ServiceName>  服务名称(必填)，用于查询该服务下的所有实例
    __func__ python3 -m cli.OpenAPI.api tianji Action ListInstanceInService ServiceName {ServiceName}

```

ListMachineByCluster 示例:
```
>>> tianji ListMachineByCluster ProjectName vpc ClusterName slbCluster-A-20260108-0079
[
    "c28d12002.cloud.d12.amtest82",
    "c28d12003.cloud.d12.amtest82",
    "c28d12004.cloud.d12.amtest82",
    "c28d12005.cloud.d12.amtest82",
    "c28d12010.cloud.d12.amtest82",
    "c28d12011.cloud.d12.amtest82",
    "vm010082008063",
    "vm010082013189",
    "vm010082017137"
]
```

ListClusterMachine 示例:
```
>>> tianji ListClusterMachine ProjectName slb ClusterName slbCluster-A-20260108-0079
[
    "c28d12004.cloud.d12.amtest82",
    "c28d12010.cloud.d12.amtest82",
    "vm010082017137",
    "vm010082013189",
    "c28d12002.cloud.d12.amtest82",
    "c28d12003.cloud.d12.amtest82",
    "c28d12005.cloud.d12.amtest82",
    "c28d12011.cloud.d12.amtest82",
    "vm010082008063"
]
```

ListMachines 示例（返回每台机器的详细属性信息，包含 Attributes 和 Topology）:
```
>>> tianji ListMachines ProjectName slb ClusterName slbCluster-A-20260108-0079
{
    "c28d12002.cloud.d12.amtest82": {
        "Attributes": {
            "Bucket": "default",
            "cluster": "slbCluster-A-20260108-0079",
            "cpu_arch": "x86_64",
            "hostname": "c28d12002.cloud.d12.amtest82",
            "hw_cpu": "64",
            "hw_mem": "262144",
            "idc": "amtest82",
            "ip": "10.82.5.10",
            "is_vm": "false",
            "project": "slb",
            ...
        },
        "Topology": { ... }
    },
    ...
}
```

ListClusterServerRoles 示例:
```
>>> tianji ListClusterServerRoles ProjectName slb ClusterName slbCluster-A-20260108-0079
[
    "hids-client.HidsClient#",
    "hids-client",
    "slb-controller.ServiceTest#",
    "slb-controller.SlbAg#",
    "slb-controller.SlbApi#",
    "slb-controller.SlbControlMaster#",
    "slb-controller.SlbMonitorMaster#",
    "slb-controller",
    "slb-keyserver.SlbKeyserver#",
    "slb-keyserver",
    "slb-lvs.SlbLvs#",
    "slb-lvs",
    "slb-monitor.SlbMonitorCms#",
    "slb-monitor.SlbMonitorOms#",
    "slb-monitor",
    "slb-ops-api.SlbOpsApi#",
    "slb-ops-api",
    "slb-proxy.SlbProxy#",
    "slb-proxy",
    "srs-client.srs-client#",
    "srs-client",
    "staragent-client.Staragentd#",
    "staragent-client",
    "tianji-dockerdaemon.DockerDaemon#",
    "tianji-dockerdaemon",
    "tianji-hostservice.host-service#",
    "tianji-hostservice",
    "tianji-sshtunnel-client.SSHTunnelClient#",
    "tianji-sshtunnel-client",
    "tianji.TianjiClient#",
    "tianji"
]
```

ListMachineSRByCluster 示例:
```
>>> tianji ListMachineSRByCluster ProjectName slb ClusterName slb_three_zone-A-20260110-025e ServerRole slb-proxy.SlbProxy#
[
    { "hostname": "c28d12008.cloud.d12.amtest82", "serverrole": "slb-proxy.SlbProxy#" },
    { "hostname": "c28d12009.cloud.d12.amtest82", "serverrole": "slb-proxy.SlbProxy#" }
]
```

ListMachineByProjectAndSR 示例:
```
>>> tianji ListMachineByProjectAndSR ProjectName slb ServerRole slb-proxy.SlbProxy#
[
    "c28b12017.cloud.b12.amtest83",
    "c28b12018.cloud.b12.amtest83",
    "c28d12004.cloud.d12.amtest82",
    "c28d12005.cloud.d12.amtest82",
    "c28d12008.cloud.d12.amtest82",
    "c28d12009.cloud.d12.amtest82",
    "c28i12001.cloud.i12.amtest57",
    "c28i12002.cloud.i12.amtest57"
]
```

ListMachineByIP 示例:
```
>>> tianji ListMachineByIP ip 10.82.5.26
[
    "c28d12009.cloud.d12.amtest82"
]
```

GetClusterNamesOfService 示例:
```
>>> tianji GetClusterNamesOfService ServiceName slb-controller
[
    "slbCluster-B-20260108-0235",
    "slbCluster-A-20260108-0079"
]
```

ListInstanceInService 示例:
```
>>> tianji ListInstanceInService ServiceName slb-controller
[
    {
        "ClusterName": "slbCluster-B-20260108-0235",
        "ClusterState": "normal",
        "ProjectName": "slb",
        "Tags": [],
        "ClusterTags": [],
        "PatchTemplates": [],
        "Templates": [
            { "TemplateName": "TMPL-SLB-CONTROLLER", "Version": "7638ad1815e89b870d017425154e350e913fba81" }
        ],
        "ClusterAttributes": null,
        "InstanceHasConf": true,
        "ReferType": "Base",
        "BaseTemplate": { "TemplateName": "", "Version": "" }
    },
    {
        "ClusterName": "slbCluster-A-20260108-0079",
        "ClusterState": "normal",
        "ProjectName": "slb",
        ...
    }
]
```
### 获取集群基线信息
```
  CheckoutCluster  检出(查看)指定集群的配置信息，返回集群配置(ClusterConf)和版本号(Version)
    [ProjectName]  产品名称(可选)，不填时系统自动根据集群名推断
    <ClusterName>  集群名称(必填)，指定要检出配置的目标集群
    [version]  版本号(可选)，指定检出的配置版本，不填则获取最新版本
    [extend]  是否展开模板继承(可选)，true 表示展开扩展配置，默认 false
    [plain]  是否以纯文本模式返回(可选)，true 表示不做模板解析，默认 false
    __func__ python3 -m cli.OpenAPI.api tianji Action CheckoutCluster [ProjectName {ProjectName}] ClusterName {ClusterName} [ version {version} ] [ extend {extend} ] [ plain {plain} ]

```

示例（返回 [配置字典, 版本号] 的数组，配置字典包含 cluster.conf、kv.conf、machine_group.conf 等）:
```
>>> tianji CheckoutCluster ProjectName slb ClusterName slb_three_zone-A-20260110-025e
[
    {
        "cluster.conf": "CloneTemplateBasis: MachineGroup\nGroup: tianji\nLegacy: \"False\"\nMachines:\n- c28d12006.cloud.d12.amtest82\n- c28d12007.cloud.d12.amtest82\n...\nName: slb_three_zone-A-20260110-025e\nProject: slb\nRegion: region1\n",
        "kv.conf": "{ \"KeyValues\": { \"AccountDispatchMode\": \"1.0\", \"CLUSTER_TYPE\": \"NETFRAME\", ... } }",
        "machine_group.conf": "MachineGroupAttrs:\n  SlbLvs:\n    serverRoleList:\n    - slb-lvs.SlbLvs#\n    ...\nMachineGroups:\n  SlbLvs:\n  - c28d12006.cloud.d12.amtest82\n  ...",
        ...
    },
    "5866c1475208c1f462bb76309e329ec6d9af1601"
]
```

### 获取服务基线信息
```
  CheckoutService  检出(查看)指定服务实例的配置信息，返回集群配置(ClusterConf)、服务配置(ServiceConf)和版本号(Version)
    [ProjectName]  产品名称(可选)，不填时系统自动根据集群名推断
    <ClusterName>  集群名称(必填)，指定服务实例所在的集群
    <ServiceName>  服务名称(必填)，指定要检出配置的目标服务
    [version]  版本号(可选)，指定检出的配置版本，不填则获取最新版本
    [extend]  是否展开模板继承(可选)，true 表示展开扩展配置，默认 false
    [plain]  是否以纯文本模式返回(可选)，true 表示不做模板解析，默认 false
    __func__ python3 -m cli.OpenAPI.api tianji Action CheckoutService [ProjectName {ProjectName}] ClusterName {ClusterName} ServiceName {ServiceName} [ version {version} ] [ extend {extend} ] [ plain {plain} ]

```

示例（返回 [集群配置字典, 服务配置字典, 版本号] 的数组）:
```
>>> tianji CheckoutService ClusterName slbCluster-A-20260108-0079 ServiceName slb-controller
[
    {
        "cluster.conf": "CloneTemplateBasis: MachineGroup\n...\nName: slbCluster-A-20260108-0079\nProject: slb\n",
        "kv.conf": "{ \"KeyValues\": { ... } }",
        "machine_group.conf": "MachineGroupAttrs:\n  SlbControllerGroupOfCount1:\n    serverRoleList:\n    - slb-controller.ServiceTest#\n    - slb-controller.SlbAg#\n    ...\nMachineGroups:\n  SlbLvs:\n  - c28d12002.cloud.d12.amtest82\n  ...",
        ...
    },
    {
        "dependency.conf": "{ \"Dependency\": { \"ServiceTest#\": { \"ServerRole\": [...] } } }",
        "kv.conf": "{ \"KeyValues\": { \"slb_vip_internet\": \"43.82.10.0/23\", ... } }",
        "role.conf": "ServerRoles:\n  ServiceTest#:\n  - vm010082008063\n  SlbAg#:\n  - vm010082008063:10.82.8.64:docker010082008064\n  ...",
        "template.conf": "BaseTemplate:\n  TemplateName: TMPL-SLB-CONTROLLER\n  Version: b76219329be7ef1dfd96142a05b58aa1dcb35f5d\n",
        "version.conf": "Versions:\n  ServiceTest#:\n    Applications:\n      '*': pangu#7460620_slb-controller_ServiceTest_573862fb\n  ...",
        ...
    },
    "f5ccf4577e7dff5fa188912fb699ae315abcf4dc"
]
```

### 获取集群资源信息
```
  GetClusterResources  获取集群的资源信息，包含 app、name、type、status、parameters、result 等属性
    [ProjectName]  产品名称(可选)，不填时系统自动根据集群名推断
    <ClusterName>  集群名称(必填)，指定要查询资源信息的目标集群
    __func__ python3 -m cli.OpenAPI.api tianji Action GetClusterResources [ProjectName {ProjectName}] ClusterName {ClusterName}

```

示例（注意：必须提供 ProjectName，否则会报错）:
```
>>> tianji GetClusterResources ClusterName slbCluster-A-20260108-0079 ProjectName slb
{
    "code": 200,
    "data": [
        {
            "instance_id": "14a9975940ede77286d3d9d337c4f448b0ef835d",
            "app": "slb-ops-api",
            "name": "slbopsapi",
            "parameters": "{\"domain\":\"slbopsapi.intra.env82.shuguang.com\",\"name\":\"slbopsapi\",\"vip\":\"10.82.72.1\"}",
            "result": "{\"dns\":\"slbopsapi.intra.env82.shuguang.com\",\"domain\":\"slbopsapi.intra.env82.shuguang.com\",\"ip\":\"[\\\"10.82.72.1\\\"]\",\"alias\":\"\"}",
            "status": "done",
            "type": "dns"
        },
        {
            "instance_id": "5fa59963badb32fb6bbcaa39dc43f09fb3c5cd3e",
            "app": "slb-control-master",
            "name": "stats",
            "parameters": "{\"db_name\":\"stats\",\"level\":\"rds.mysql.s3.large\",...}",
            "result": "{\"db_host\":\"stats.mysql.minirds.intra.env82.shuguang.com\",\"db_port\":\"3012\",...}",
            "status": "done",
            "type": "db"
        },
        ...
    ],
    "message": ""
}
```

### 获取天基基线配置
```
  FetchFileList  获取指定集群中某个服务实例的应用下，特定路径的文件列表
    <ProjectName>  产品名称(必填)，指定目标产品
    <ClusterName>  集群名称(必填)，指定目标集群
    <HostName>     主机名称(必填)，指定目标机器的 hostname
    <ServiceName>  服务名称(必填)，指定目标服务实例
    <ServerRoleName>  服务角色名称(必填)，指定目标 ServerRole
    <AppName>      应用名称(必填)，指定目标 App
    <Path>         文件路径(必填)，指定要查询的目录路径
    <Pattern>      文件名匹配模式(必填)，用于过滤文件名的通配符或正则表达式
    __func__ python3 -m cli.OpenAPI.api tianji Action FetchFileList ProjectName {ProjectName} ClusterName {ClusterName} HostName {HostName} ServiceName {ServiceName} ServerRoleName {ServerRoleName} AppName {AppName} Path {Path} Pattern {Pattern}

```

### 获取全部天基基线配置
```
  FetchAllDir  获取指定集群中某个服务实例的应用下的所有目录结构
    <ProjectName>  产品名称(必填)，指定目标产品
    <ClusterName>  集群名称(必填)，指定目标集群
    <HostName>     主机名称(必填)，指定目标机器的 hostname
    <ServiceName>  服务名称(必填)，指定目标服务实例
    <ServerRoleName>  服务角色名称(必填)，指定目标 ServerRole
    <AppName>      应用名称(必填)，指定目标 App
    __func__ python3 -m cli.OpenAPI.api tianji Action FetchAllDir ProjectName {ProjectName} ClusterName {ClusterName} HostName {HostName} ServiceName {ServiceName} ServerRoleName {ServerRoleName} AppName {AppName}

```

### 获取docker日志
```
  DockerLogs  获取指定集群中某个服务实例应用的 Docker 容器日志
    <ProjectName>  产品名称(必填)，指定目标产品
    <ClusterName>  集群名称(必填)，指定目标集群
    <HostName>     主机名称(必填)，指定目标机器的 hostname
    <ServiceName>  服务名称(必填)，指定目标服务实例
    <ServerRoleName>  服务角色名称(必填)，指定目标 ServerRole
    <AppName>      应用名称(必填)，指定目标 App
    <Machine>      机器标识(必填)，指定要获取日志的机器
    <Count>        日志行数(必填)，指定返回的日志条数
    __func__ python3 -m cli.OpenAPI.api tianji Action DockerLogs ProjectName {ProjectName} ClusterName {ClusterName} HostName {HostName} ServiceName {ServiceName} ServerRoleName {ServerRoleName} AppName {AppName} Machine {Machine} Count {Count}

```


### 获取实例信息
```
  GetInstanceInfo  获取指定集群中某个服务下特定组件(ServerRole/Component)的实例列表信息(分页查询)
    [ProjectName]    产品名称(可选)，对应接口中的 project 参数
    <ClusterName>    集群名称(必填)，对应接口中的 cluster 参数
    <ServerRoleName> 组件/服务角色名称(必填)，对应接口中的 component 参数
    <ServiceName>    服务名称(必填)，对应接口中的 service 参数
    __func__  python3 -m cli.OpenAPI.api tianji Action GetInstanceInfo [ProjectName {ProjectName}] ClusterName {ClusterName} ServerRoleName {ServerRoleName} ServiceName {ServiceName}

```

""".strip()


# ── 知识召回测试提示词 ───────────────────────────────────────────────

RECALL_PROMPTS = [
    "获取集群资源信息,集群名称为 slbCluster-A-20260108-0079",
    "获取当前天基环境下所有可用的 Region 列表及其编号(region no)",
    "取该产品下所有集群列表，产品名称为 slb",
    "获取指定集群下的服务实例列表,集群： slbCluster-A-20260108-0079",
    "获取指定集群的基线信息，集群名称为 slbCluster-A-20260108-0079",
    "获取指定服务实例的配置信息，集群名称为 slbCluster-A-20260108-0079，服务名称为 slb-controller",
]