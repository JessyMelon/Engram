KNOWLEDGE_ENTRIES = [
  {
    "type": "command_mapping",
    "content": "## create_loadbalancer - 创建SLB负载均衡实例\n[Action:create_loadbalancer]/[Overview]\n接口 create_loadbalancer 用于 create_loadbalancer - 创建SLB负载均衡实例。\n必选参数: region_no, bid, aliyun_idkp。\n成功返回: {\"code\":200,\"msg\":\"successful\",\"data\":{\"lb_id\":\"123\",\"eip\":\"10.250.6.36\",\"site_id\":\"1\"}}\n要创建SLB实例，调用create_loadbalancer接口传入region_no、bid和aliyun_idkp三个必选参数。创建VPC类型实例需要额外传入gw_type=vpc、vpc_instance_id和tunnel_id。如果指定lb_name，同一aliyun_idkp下必须唯一，重复会返回-2625错误码并带回已有实例信息。create_loadbalancer支持通过ha_type指定单机房或双机房灾备模式。\ncreate_loadbalancer接口用于创建SLB负载均衡实例并分配服务IP和lb_id。经典网络实例只需region_no、bid、aliyun_idkp即可创建；VPC网络实例还需传gw_type=vpc、eip、vpc_instance_id、tunnel_id。创建ipv6实例需设ip_version=ipv6且目前仅支持internet类型。默认转发模式为fnat，enable_vpc_vip_flow默认on表示创建后立即引流。",
    "paraphrases": [
      "当需要执行 create_loadbalancer 对应操作时，优先检查必选参数 region_no、bid、aliyun_idkp。",
      "create_loadbalancer 的核心触发锚点是 action=create_loadbalancer。"
    ],
    "recall_prompts": [
      "create_loadbalancer 是做什么的？",
      "什么时候应该调用 create_loadbalancer？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:create_loadbalancer]/[Field:region_no]\n参数 region_no 属于接口 create_loadbalancer，类型 string。必选参数。 LoadBalancer所属的region_no",
    "paraphrases": [
      "create_loadbalancer 的参数 region_no 必须提供。",
      "查询 create_loadbalancer 时，字段 region_no 的含义是：LoadBalancer所属的region_no"
    ],
    "recall_prompts": [
      "create_loadbalancer 的参数 region_no 是什么？",
      "create_loadbalancer 里 region_no 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:create_loadbalancer]/[Field:bid]\n参数 bid 属于接口 create_loadbalancer，类型 string。必选参数。 渠道ID，长度1-80",
    "paraphrases": [
      "create_loadbalancer 的参数 bid 必须提供。",
      "查询 create_loadbalancer 时，字段 bid 的含义是：渠道ID，长度1-80"
    ],
    "recall_prompts": [
      "create_loadbalancer 的参数 bid 是什么？",
      "create_loadbalancer 里 bid 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:create_loadbalancer]/[Field:aliyun_idkp]\n参数 aliyun_idkp 属于接口 create_loadbalancer，类型 string。必选参数。 阿里云云帐号ID",
    "paraphrases": [
      "create_loadbalancer 的参数 aliyun_idkp 必须提供。",
      "查询 create_loadbalancer 时，字段 aliyun_idkp 的含义是：阿里云云帐号ID"
    ],
    "recall_prompts": [
      "create_loadbalancer 的参数 aliyun_idkp 是什么？",
      "create_loadbalancer 里 aliyun_idkp 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Field:eip]\n参数 eip 属于接口 create_loadbalancer，类型 string。可选参数。 关联的IP地址，不传则系统分配，VPC类型必传",
    "paraphrases": [
      "create_loadbalancer 的参数 eip 可以不传。",
      "查询 create_loadbalancer 时，字段 eip 的含义是：关联的IP地址，不传则系统分配，VPC类型必传"
    ],
    "recall_prompts": [
      "create_loadbalancer 的参数 eip 是什么？",
      "create_loadbalancer 里 eip 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Field:eip_type]\n参数 eip_type 属于接口 create_loadbalancer，类型 string。可选参数。 internet或intranet，默认internet",
    "paraphrases": [
      "create_loadbalancer 的参数 eip_type 可以不传。",
      "查询 create_loadbalancer 时，字段 eip_type 的含义是：internet或intranet，默认internet"
    ],
    "recall_prompts": [
      "create_loadbalancer 的参数 eip_type 是什么？",
      "create_loadbalancer 里 eip_type 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Default:eip_type]\n在接口 create_loadbalancer 中，参数 eip_type 的默认值是 internet。",
    "paraphrases": [
      "如果 create_loadbalancer 未显式传入 eip_type，默认使用 internet。",
      "create_loadbalancer 的 eip_type 缺省值为 internet。"
    ],
    "recall_prompts": [
      "create_loadbalancer 的 eip_type 默认值是什么？",
      "不传 eip_type 时 create_loadbalancer 会用什么值？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Field:gw_type]\n参数 gw_type 属于接口 create_loadbalancer，类型 string。可选参数。 vpc表示VPC实例，any_vpc表示any VPC实例，不传为经典实例",
    "paraphrases": [
      "create_loadbalancer 的参数 gw_type 可以不传。",
      "查询 create_loadbalancer 时，字段 gw_type 的含义是：vpc表示VPC实例，any_vpc表示any VPC实例，不传为经典实例"
    ],
    "recall_prompts": [
      "create_loadbalancer 的参数 gw_type 是什么？",
      "create_loadbalancer 里 gw_type 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Field:vpc_instance_id]\n参数 vpc_instance_id 属于接口 create_loadbalancer，类型 string。可选参数。 VPC实例必传",
    "paraphrases": [
      "create_loadbalancer 的参数 vpc_instance_id 可以不传。",
      "查询 create_loadbalancer 时，字段 vpc_instance_id 的含义是：VPC实例必传"
    ],
    "recall_prompts": [
      "create_loadbalancer 的参数 vpc_instance_id 是什么？",
      "create_loadbalancer 里 vpc_instance_id 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Field:tunnel_id]\n参数 tunnel_id 属于接口 create_loadbalancer，类型 int。可选参数。 VPC实例必传，VxLan协议的VNI",
    "paraphrases": [
      "create_loadbalancer 的参数 tunnel_id 可以不传。",
      "查询 create_loadbalancer 时，字段 tunnel_id 的含义是：VPC实例必传，VxLan协议的VNI"
    ],
    "recall_prompts": [
      "create_loadbalancer 的参数 tunnel_id 是什么？",
      "create_loadbalancer 里 tunnel_id 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Field:ha_type]\n参数 ha_type 属于接口 create_loadbalancer，类型 string。可选参数。 single_site(单机房无灾备)/double_site(双机房有灾备)",
    "paraphrases": [
      "create_loadbalancer 的参数 ha_type 可以不传。",
      "查询 create_loadbalancer 时，字段 ha_type 的含义是：single_site(单机房无灾备)/double_site(双机房有灾备)"
    ],
    "recall_prompts": [
      "create_loadbalancer 的参数 ha_type 是什么？",
      "create_loadbalancer 里 ha_type 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Field:ip_version]\n参数 ip_version 属于接口 create_loadbalancer，类型 string。可选参数。 默认ipv4，传入ipv6创建v6实例",
    "paraphrases": [
      "create_loadbalancer 的参数 ip_version 可以不传。",
      "查询 create_loadbalancer 时，字段 ip_version 的含义是：默认ipv4，传入ipv6创建v6实例"
    ],
    "recall_prompts": [
      "create_loadbalancer 的参数 ip_version 是什么？",
      "create_loadbalancer 里 ip_version 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Default:ip_version]\n在接口 create_loadbalancer 中，参数 ip_version 的默认值是 ipv4。",
    "paraphrases": [
      "如果 create_loadbalancer 未显式传入 ip_version，默认使用 ipv4。",
      "create_loadbalancer 的 ip_version 缺省值为 ipv4。"
    ],
    "recall_prompts": [
      "create_loadbalancer 的 ip_version 默认值是什么？",
      "不传 ip_version 时 create_loadbalancer 会用什么值？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Field:mode]\n参数 mode 属于接口 create_loadbalancer，类型 string。可选参数。 nat/fnat/fnat_ext，默认fnat",
    "paraphrases": [
      "create_loadbalancer 的参数 mode 可以不传。",
      "查询 create_loadbalancer 时，字段 mode 的含义是：nat/fnat/fnat_ext，默认fnat"
    ],
    "recall_prompts": [
      "create_loadbalancer 的参数 mode 是什么？",
      "create_loadbalancer 里 mode 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Default:mode]\n在接口 create_loadbalancer 中，参数 mode 的默认值是 fnat。",
    "paraphrases": [
      "如果 create_loadbalancer 未显式传入 mode，默认使用 fnat。",
      "create_loadbalancer 的 mode 缺省值为 fnat。"
    ],
    "recall_prompts": [
      "create_loadbalancer 的 mode 默认值是什么？",
      "不传 mode 时 create_loadbalancer 会用什么值？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Field:lb_name]\n参数 lb_name 属于接口 create_loadbalancer，类型 string。可选参数。 同一aliyun_idkp下必须唯一，长度1-80",
    "paraphrases": [
      "create_loadbalancer 的参数 lb_name 可以不传。",
      "查询 create_loadbalancer 时，字段 lb_name 的含义是：同一aliyun_idkp下必须唯一，长度1-80"
    ],
    "recall_prompts": [
      "create_loadbalancer 的参数 lb_name 是什么？",
      "create_loadbalancer 里 lb_name 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Field:su_name]\n参数 su_name 属于接口 create_loadbalancer，类型 string。可选参数。 指定service_unit，不传则系统分配",
    "paraphrases": [
      "create_loadbalancer 的参数 su_name 可以不传。",
      "查询 create_loadbalancer 时，字段 su_name 的含义是：指定service_unit，不传则系统分配"
    ],
    "recall_prompts": [
      "create_loadbalancer 的参数 su_name 是什么？",
      "create_loadbalancer 里 su_name 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Field:site_id]\n参数 site_id 属于接口 create_loadbalancer，类型 string。可选参数。 指定eip流量走的机房",
    "paraphrases": [
      "create_loadbalancer 的参数 site_id 可以不传。",
      "查询 create_loadbalancer 时，字段 site_id 的含义是：指定eip流量走的机房"
    ],
    "recall_prompts": [
      "create_loadbalancer 的参数 site_id 是什么？",
      "create_loadbalancer 里 site_id 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Field:backup_site_id]\n参数 backup_site_id 属于接口 create_loadbalancer，类型 string。可选参数。 主机房不可用时的备机房",
    "paraphrases": [
      "create_loadbalancer 的参数 backup_site_id 可以不传。",
      "查询 create_loadbalancer 时，字段 backup_site_id 的含义是：主机房不可用时的备机房"
    ],
    "recall_prompts": [
      "create_loadbalancer 的参数 backup_site_id 是什么？",
      "create_loadbalancer 里 backup_site_id 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Field:lb_id]\n参数 lb_id 属于接口 create_loadbalancer，类型 string。可选参数。 指定实例ID，不传则系统生成，同一aliyun_idkp下唯一",
    "paraphrases": [
      "create_loadbalancer 的参数 lb_id 可以不传。",
      "查询 create_loadbalancer 时，字段 lb_id 的含义是：指定实例ID，不传则系统生成，同一aliyun_idkp下唯一"
    ],
    "recall_prompts": [
      "create_loadbalancer 的参数 lb_id 是什么？",
      "create_loadbalancer 里 lb_id 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Field:enable_vpc_vip_flow]\n参数 enable_vpc_vip_flow 属于接口 create_loadbalancer，类型 string。可选参数。 on/off，默认on，控制VPC LB创建时是否立即引流",
    "paraphrases": [
      "create_loadbalancer 的参数 enable_vpc_vip_flow 可以不传。",
      "查询 create_loadbalancer 时，字段 enable_vpc_vip_flow 的含义是：on/off，默认on，控制VPC LB创建时是否立即引流"
    ],
    "recall_prompts": [
      "create_loadbalancer 的参数 enable_vpc_vip_flow 是什么？",
      "create_loadbalancer 里 enable_vpc_vip_flow 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Default:enable_vpc_vip_flow]\n在接口 create_loadbalancer 中，参数 enable_vpc_vip_flow 的默认值是 on。",
    "paraphrases": [
      "如果 create_loadbalancer 未显式传入 enable_vpc_vip_flow，默认使用 on。",
      "create_loadbalancer 的 enable_vpc_vip_flow 缺省值为 on。"
    ],
    "recall_prompts": [
      "create_loadbalancer 的 enable_vpc_vip_flow 默认值是什么？",
      "不传 enable_vpc_vip_flow 时 create_loadbalancer 会用什么值？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Constraint:4]\neip: 关联的IP地址，不传则系统分配，VPC类型必传",
    "paraphrases": [
      "接口 create_loadbalancer 存在约束：eip: 关联的IP地址，不传则系统分配，VPC类型必传",
      "调用 create_loadbalancer 时需要注意：eip: 关联的IP地址，不传则系统分配，VPC类型必传"
    ],
    "recall_prompts": [
      "create_loadbalancer 有哪些限制或条件？",
      "调用 create_loadbalancer 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Constraint:5]\neip_type: internet或intranet，默认internet",
    "paraphrases": [
      "接口 create_loadbalancer 存在约束：eip_type: internet或intranet，默认internet",
      "调用 create_loadbalancer 时需要注意：eip_type: internet或intranet，默认internet"
    ],
    "recall_prompts": [
      "create_loadbalancer 有哪些限制或条件？",
      "调用 create_loadbalancer 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Constraint:7]\nvpc_instance_id: VPC实例必传",
    "paraphrases": [
      "接口 create_loadbalancer 存在约束：vpc_instance_id: VPC实例必传",
      "调用 create_loadbalancer 时需要注意：vpc_instance_id: VPC实例必传"
    ],
    "recall_prompts": [
      "create_loadbalancer 有哪些限制或条件？",
      "调用 create_loadbalancer 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Constraint:8]\ntunnel_id: VPC实例必传，VxLan协议的VNI",
    "paraphrases": [
      "接口 create_loadbalancer 存在约束：tunnel_id: VPC实例必传，VxLan协议的VNI",
      "调用 create_loadbalancer 时需要注意：tunnel_id: VPC实例必传，VxLan协议的VNI"
    ],
    "recall_prompts": [
      "create_loadbalancer 有哪些限制或条件？",
      "调用 create_loadbalancer 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Constraint:10]\nip_version: 默认ipv4，传入ipv6创建v6实例",
    "paraphrases": [
      "接口 create_loadbalancer 存在约束：ip_version: 默认ipv4，传入ipv6创建v6实例",
      "调用 create_loadbalancer 时需要注意：ip_version: 默认ipv4，传入ipv6创建v6实例"
    ],
    "recall_prompts": [
      "create_loadbalancer 有哪些限制或条件？",
      "调用 create_loadbalancer 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Constraint:11]\nmode: nat/fnat/fnat_ext，默认fnat",
    "paraphrases": [
      "接口 create_loadbalancer 存在约束：mode: nat/fnat/fnat_ext，默认fnat",
      "调用 create_loadbalancer 时需要注意：mode: nat/fnat/fnat_ext，默认fnat"
    ],
    "recall_prompts": [
      "create_loadbalancer 有哪些限制或条件？",
      "调用 create_loadbalancer 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Constraint:12]\nlb_name: 同一aliyun_idkp下必须唯一，长度1-80",
    "paraphrases": [
      "接口 create_loadbalancer 存在约束：lb_name: 同一aliyun_idkp下必须唯一，长度1-80",
      "调用 create_loadbalancer 时需要注意：lb_name: 同一aliyun_idkp下必须唯一，长度1-80"
    ],
    "recall_prompts": [
      "create_loadbalancer 有哪些限制或条件？",
      "调用 create_loadbalancer 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Constraint:16]\nlb_id: 指定实例ID，不传则系统生成，同一aliyun_idkp下唯一",
    "paraphrases": [
      "接口 create_loadbalancer 存在约束：lb_id: 指定实例ID，不传则系统生成，同一aliyun_idkp下唯一",
      "调用 create_loadbalancer 时需要注意：lb_id: 指定实例ID，不传则系统生成，同一aliyun_idkp下唯一"
    ],
    "recall_prompts": [
      "create_loadbalancer 有哪些限制或条件？",
      "调用 create_loadbalancer 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Constraint:17]\nenable_vpc_vip_flow: on/off，默认on，控制VPC LB创建时是否立即引流",
    "paraphrases": [
      "接口 create_loadbalancer 存在约束：enable_vpc_vip_flow: on/off，默认on，控制VPC LB创建时是否立即引流",
      "调用 create_loadbalancer 时需要注意：enable_vpc_vip_flow: on/off，默认on，控制VPC LB创建时是否立即引流"
    ],
    "recall_prompts": [
      "create_loadbalancer 有哪些限制或条件？",
      "调用 create_loadbalancer 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Constraint:18]\n要创建SLB实例，调用create_loadbalancer接口传入region_no、bid和aliyun_idkp三个必选参数。创建VPC类型实例需要额外传入gw_type=vpc、vpc_instance_id和tunnel_id。如果指定lb_name，同一aliyun_idkp下必须唯一，重复会返回-2625错误码并带回已有实例信息。create_loadbalancer支持通过ha_type指定单机房或双机房灾备模式。",
    "paraphrases": [
      "接口 create_loadbalancer 存在约束：要创建SLB实例，调用create_loadbalancer接口传入region_no、bid和aliyun_idkp三个必选参数。创建VPC类型实例需要额外传入gw_type=vpc、vpc_instance_id和tunnel_id。如果指定lb_name，同一aliyun_idkp下必须唯一，重复会返回-2625错误码并带回已有实例信息。create_loadbalancer支持通过ha_type指定单机房或双机房灾备模式。",
      "调用 create_loadbalancer 时需要注意：要创建SLB实例，调用create_loadbalancer接口传入region_no、bid和aliyun_idkp三个必选参数。创建VPC类型实例需要额外传入gw_type=vpc、vpc_instance_id和tunnel_id。如果指定lb_name，同一aliyun_idkp下必须唯一，重复会返回-2625错误码并带回已有实例信息。create_loadbalancer支持通过ha_type指定单机房或双机房灾备模式。"
    ],
    "recall_prompts": [
      "create_loadbalancer 有哪些限制或条件？",
      "调用 create_loadbalancer 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_loadbalancer]/[Constraint:19]\ncreate_loadbalancer接口用于创建SLB负载均衡实例并分配服务IP和lb_id。经典网络实例只需region_no、bid、aliyun_idkp即可创建；VPC网络实例还需传gw_type=vpc、eip、vpc_instance_id、tunnel_id。创建ipv6实例需设ip_version=ipv6且目前仅支持internet类型。默认转发模式为fnat，enable_vpc_vip_flow默认on表示创建后立即引流。",
    "paraphrases": [
      "接口 create_loadbalancer 存在约束：create_loadbalancer接口用于创建SLB负载均衡实例并分配服务IP和lb_id。经典网络实例只需region_no、bid、aliyun_idkp即可创建；VPC网络实例还需传gw_type=vpc、eip、vpc_instance_id、tunnel_id。创建ipv6实例需设ip_version=ipv6且目前仅支持internet类型。默认转发模式为fnat，enable_vpc_vip_flow默认on表示创建后立即引流。",
      "调用 create_loadbalancer 时需要注意：create_loadbalancer接口用于创建SLB负载均衡实例并分配服务IP和lb_id。经典网络实例只需region_no、bid、aliyun_idkp即可创建；VPC网络实例还需传gw_type=vpc、eip、vpc_instance_id、tunnel_id。创建ipv6实例需设ip_version=ipv6且目前仅支持internet类型。默认转发模式为fnat，enable_vpc_vip_flow默认on表示创建后立即引流。"
    ],
    "recall_prompts": [
      "create_loadbalancer 有哪些限制或条件？",
      "调用 create_loadbalancer 时要注意什么？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:create_loadbalancer]/[Error:-2620]\n当调用 create_loadbalancer 返回错误码 -2620 NoAvailableIp 时，表示 没有可用IP。",
    "paraphrases": [
      "create_loadbalancer 出现 -2620 时，对应错误名是 NoAvailableIp。",
      "错误 -2620 在 create_loadbalancer 中表示：没有可用IP。"
    ],
    "recall_prompts": [
      "create_loadbalancer 返回 -2620 代表什么？",
      "NoAvailableIp 是什么错误？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:create_loadbalancer]/[Error:-2625]\n当调用 create_loadbalancer 返回错误码 -2625 LbNameExists 时，表示 lb_name已存在，返回已存在lb信息。",
    "paraphrases": [
      "create_loadbalancer 出现 -2625 时，对应错误名是 LbNameExists。",
      "错误 -2625 在 create_loadbalancer 中表示：lb_name已存在，返回已存在lb信息。"
    ],
    "recall_prompts": [
      "create_loadbalancer 返回 -2625 代表什么？",
      "LbNameExists 是什么错误？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:create_loadbalancer]/[Error:-2633]\n当调用 create_loadbalancer 返回错误码 -2633 LbIdExists 时，表示 lb_id已存在。",
    "paraphrases": [
      "create_loadbalancer 出现 -2633 时，对应错误名是 LbIdExists。",
      "错误 -2633 在 create_loadbalancer 中表示：lb_id已存在。"
    ],
    "recall_prompts": [
      "create_loadbalancer 返回 -2633 代表什么？",
      "LbIdExists 是什么错误？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:create_loadbalancer]/[Error:-2619]\n当调用 create_loadbalancer 返回错误码 -2619 RegionIdIsEmpty 时，表示 Region ID为空。",
    "paraphrases": [
      "create_loadbalancer 出现 -2619 时，对应错误名是 RegionIdIsEmpty。",
      "错误 -2619 在 create_loadbalancer 中表示：Region ID为空。"
    ],
    "recall_prompts": [
      "create_loadbalancer 返回 -2619 代表什么？",
      "RegionIdIsEmpty 是什么错误？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:create_loadbalancer]/[Error:-2617]\n当调用 create_loadbalancer 返回错误码 -2617 EipTypeNotSupport 时，表示 IP类型错误。",
    "paraphrases": [
      "create_loadbalancer 出现 -2617 时，对应错误名是 EipTypeNotSupport。",
      "错误 -2617 在 create_loadbalancer 中表示：IP类型错误。"
    ],
    "recall_prompts": [
      "create_loadbalancer 返回 -2617 代表什么？",
      "EipTypeNotSupport 是什么错误？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:create_loadbalancer]/[Error:-2228]\n当调用 create_loadbalancer 返回错误码 -2228 AllocateVpcInstanceFail 时，表示 申请VPC XGW资源失败。",
    "paraphrases": [
      "create_loadbalancer 出现 -2228 时，对应错误名是 AllocateVpcInstanceFail。",
      "错误 -2228 在 create_loadbalancer 中表示：申请VPC XGW资源失败。"
    ],
    "recall_prompts": [
      "create_loadbalancer 返回 -2228 代表什么？",
      "AllocateVpcInstanceFail 是什么错误？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:create_loadbalancer]/[Error:-2229]\n当调用 create_loadbalancer 返回错误码 -2229 AllocatePublicIpFail 时，表示 VPC XGW分配IP失败。",
    "paraphrases": [
      "create_loadbalancer 出现 -2229 时，对应错误名是 AllocatePublicIpFail。",
      "错误 -2229 在 create_loadbalancer 中表示：VPC XGW分配IP失败。"
    ],
    "recall_prompts": [
      "create_loadbalancer 返回 -2229 代表什么？",
      "AllocatePublicIpFail 是什么错误？"
    ]
  },
  {
    "type": "command_mapping",
    "content": "## delete_loadbalancer - delete_loadbalancer + config_loadbalancer - 删除和配置SLB实例\n[Action:delete_loadbalancer]/[Overview]\n接口 delete_loadbalancer 用于 delete_loadbalancer + config_loadbalancer - 删除和配置SLB实例。\n必选参数: region_no, lb_id, aliyun_idkp。\n成功返回: {\"code\":200,\"msg\":\"successful\"}\ndelete_loadbalancer会清除LoadBalancer相关的所有配置，如果LB上还有VIP也会一并被删除。",
    "paraphrases": [
      "当需要执行 delete_loadbalancer 对应操作时，优先检查必选参数 region_no、lb_id、aliyun_idkp。",
      "delete_loadbalancer 的核心触发锚点是 action=delete_loadbalancer。"
    ],
    "recall_prompts": [
      "delete_loadbalancer 是做什么的？",
      "什么时候应该调用 delete_loadbalancer？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:delete_loadbalancer]/[Field:region_no]\n参数 region_no 属于接口 delete_loadbalancer，类型 string。必选参数。 LoadBalancer所属的region_no",
    "paraphrases": [
      "delete_loadbalancer 的参数 region_no 必须提供。",
      "查询 delete_loadbalancer 时，字段 region_no 的含义是：LoadBalancer所属的region_no"
    ],
    "recall_prompts": [
      "delete_loadbalancer 的参数 region_no 是什么？",
      "delete_loadbalancer 里 region_no 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:delete_loadbalancer]/[Field:lb_id]\n参数 lb_id 属于接口 delete_loadbalancer，类型 string。必选参数。 LoadBalancer的唯一标识",
    "paraphrases": [
      "delete_loadbalancer 的参数 lb_id 必须提供。",
      "查询 delete_loadbalancer 时，字段 lb_id 的含义是：LoadBalancer的唯一标识"
    ],
    "recall_prompts": [
      "delete_loadbalancer 的参数 lb_id 是什么？",
      "delete_loadbalancer 里 lb_id 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:delete_loadbalancer]/[Field:aliyun_idkp]\n参数 aliyun_idkp 属于接口 delete_loadbalancer，类型 string。必选参数。 阿里云云帐号ID",
    "paraphrases": [
      "delete_loadbalancer 的参数 aliyun_idkp 必须提供。",
      "查询 delete_loadbalancer 时，字段 aliyun_idkp 的含义是：阿里云云帐号ID"
    ],
    "recall_prompts": [
      "delete_loadbalancer 的参数 aliyun_idkp 是什么？",
      "delete_loadbalancer 里 aliyun_idkp 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:delete_loadbalancer]/[Field:bid]\n参数 bid 属于接口 delete_loadbalancer，类型 string。可选参数。 渠道ID",
    "paraphrases": [
      "delete_loadbalancer 的参数 bid 可以不传。",
      "查询 delete_loadbalancer 时，字段 bid 的含义是：渠道ID"
    ],
    "recall_prompts": [
      "delete_loadbalancer 的参数 bid 是什么？",
      "delete_loadbalancer 里 bid 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:delete_loadbalancer]/[Constraint:2]\nlb_id: LoadBalancer的唯一标识",
    "paraphrases": [
      "接口 delete_loadbalancer 存在约束：lb_id: LoadBalancer的唯一标识",
      "调用 delete_loadbalancer 时需要注意：lb_id: LoadBalancer的唯一标识"
    ],
    "recall_prompts": [
      "delete_loadbalancer 有哪些限制或条件？",
      "调用 delete_loadbalancer 时要注意什么？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:delete_loadbalancer]/[Error:-2601]\n当调用 delete_loadbalancer 返回错误码 -2601 LbIdIsEmpty 时，表示 LB ID为空。",
    "paraphrases": [
      "delete_loadbalancer 出现 -2601 时，对应错误名是 LbIdIsEmpty。",
      "错误 -2601 在 delete_loadbalancer 中表示：LB ID为空。"
    ],
    "recall_prompts": [
      "delete_loadbalancer 返回 -2601 代表什么？",
      "LbIdIsEmpty 是什么错误？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:delete_loadbalancer]/[Error:-2334]\n当调用 delete_loadbalancer 返回错误码 -2334 VipStopFailure 时，表示 停止VIP异常。",
    "paraphrases": [
      "delete_loadbalancer 出现 -2334 时，对应错误名是 VipStopFailure。",
      "错误 -2334 在 delete_loadbalancer 中表示：停止VIP异常。"
    ],
    "recall_prompts": [
      "delete_loadbalancer 返回 -2334 代表什么？",
      "VipStopFailure 是什么错误？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:delete_loadbalancer]/[Error:-2249]\n当调用 delete_loadbalancer 返回错误码 -2249 ServiceIsStarting 时，表示 删除lb时有监听正在启动中。",
    "paraphrases": [
      "delete_loadbalancer 出现 -2249 时，对应错误名是 ServiceIsStarting。",
      "错误 -2249 在 delete_loadbalancer 中表示：删除lb时有监听正在启动中。"
    ],
    "recall_prompts": [
      "delete_loadbalancer 返回 -2249 代表什么？",
      "ServiceIsStarting 是什么错误？"
    ]
  },
  {
    "type": "command_mapping",
    "content": "## config_loadbalancer - delete_loadbalancer + config_loadbalancer - 删除和配置SLB实例\n[Action:config_loadbalancer]/[Overview]\n接口 config_loadbalancer 用于 delete_loadbalancer + config_loadbalancer - 删除和配置SLB实例。\n必选参数: region_no, lb_id, status, aliyun_idkp。\n成功返回: {\"code\":200,\"msg\":\"successful\"}\nconfig_loadbalancer设置status=active时将LB中所有VIP都激活，status=inactive时将所有VIP都停用。",
    "paraphrases": [
      "当需要执行 config_loadbalancer 对应操作时，优先检查必选参数 region_no、lb_id、status、aliyun_idkp。",
      "config_loadbalancer 的核心触发锚点是 action=config_loadbalancer。"
    ],
    "recall_prompts": [
      "config_loadbalancer 是做什么的？",
      "什么时候应该调用 config_loadbalancer？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:config_loadbalancer]/[Field:region_no]\n参数 region_no 属于接口 config_loadbalancer，类型 string。必选参数。 LoadBalancer所属的region_no",
    "paraphrases": [
      "config_loadbalancer 的参数 region_no 必须提供。",
      "查询 config_loadbalancer 时，字段 region_no 的含义是：LoadBalancer所属的region_no"
    ],
    "recall_prompts": [
      "config_loadbalancer 的参数 region_no 是什么？",
      "config_loadbalancer 里 region_no 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:config_loadbalancer]/[Field:lb_id]\n参数 lb_id 属于接口 config_loadbalancer，类型 string。必选参数。 LoadBalancer的唯一标识",
    "paraphrases": [
      "config_loadbalancer 的参数 lb_id 必须提供。",
      "查询 config_loadbalancer 时，字段 lb_id 的含义是：LoadBalancer的唯一标识"
    ],
    "recall_prompts": [
      "config_loadbalancer 的参数 lb_id 是什么？",
      "config_loadbalancer 里 lb_id 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:config_loadbalancer]/[Field:status]\n参数 status 属于接口 config_loadbalancer，类型 string。必选参数。 active或inactive",
    "paraphrases": [
      "config_loadbalancer 的参数 status 必须提供。",
      "查询 config_loadbalancer 时，字段 status 的含义是：active或inactive"
    ],
    "recall_prompts": [
      "config_loadbalancer 的参数 status 是什么？",
      "config_loadbalancer 里 status 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:config_loadbalancer]/[Field:aliyun_idkp]\n参数 aliyun_idkp 属于接口 config_loadbalancer，类型 string。必选参数。 阿里云云帐号ID",
    "paraphrases": [
      "config_loadbalancer 的参数 aliyun_idkp 必须提供。",
      "查询 config_loadbalancer 时，字段 aliyun_idkp 的含义是：阿里云云帐号ID"
    ],
    "recall_prompts": [
      "config_loadbalancer 的参数 aliyun_idkp 是什么？",
      "config_loadbalancer 里 aliyun_idkp 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:config_loadbalancer]/[Field:bid]\n参数 bid 属于接口 config_loadbalancer，类型 string。可选参数。 渠道ID",
    "paraphrases": [
      "config_loadbalancer 的参数 bid 可以不传。",
      "查询 config_loadbalancer 时，字段 bid 的含义是：渠道ID"
    ],
    "recall_prompts": [
      "config_loadbalancer 的参数 bid 是什么？",
      "config_loadbalancer 里 bid 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:config_loadbalancer]/[Field:mode]\n参数 mode 属于接口 config_loadbalancer，类型 string。可选参数。 fnat/fnat_ext/nat",
    "paraphrases": [
      "config_loadbalancer 的参数 mode 可以不传。",
      "查询 config_loadbalancer 时，字段 mode 的含义是：fnat/fnat_ext/nat"
    ],
    "recall_prompts": [
      "config_loadbalancer 的参数 mode 是什么？",
      "config_loadbalancer 里 mode 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:config_loadbalancer]/[Constraint:2]\nlb_id: LoadBalancer的唯一标识",
    "paraphrases": [
      "接口 config_loadbalancer 存在约束：lb_id: LoadBalancer的唯一标识",
      "调用 config_loadbalancer 时需要注意：lb_id: LoadBalancer的唯一标识"
    ],
    "recall_prompts": [
      "config_loadbalancer 有哪些限制或条件？",
      "调用 config_loadbalancer 时要注意什么？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:config_loadbalancer]/[Error:-2622]\n当调用 config_loadbalancer 返回错误码 -2622 LbConfigFailure 时，表示 配置LB失败。",
    "paraphrases": [
      "config_loadbalancer 出现 -2622 时，对应错误名是 LbConfigFailure。",
      "错误 -2622 在 config_loadbalancer 中表示：配置LB失败。"
    ],
    "recall_prompts": [
      "config_loadbalancer 返回 -2622 代表什么？",
      "LbConfigFailure 是什么错误？"
    ]
  },
  {
    "type": "command_mapping",
    "content": "## query_loadbalancer_info - query_loadbalancer_info + list_loadbalancers - 查询SLB实例信息\n[Action:query_loadbalancer_info]/[Overview]\n接口 query_loadbalancer_info 用于 query_loadbalancer_info + list_loadbalancers - 查询SLB实例信息。\n必选参数: region_no, lb_id, aliyun_idkp。\n成功返回: {\"code\":200,\"data\":{\"vm_list\":[],\"su_name\":\"suC\",\"gw_type\":\"classic\",\"lb_id\":\"xxx\",\"eip\":\"10.189.104.141\",\"eip_type\":\"intranet\",\"mode\":\"fnat\",\"frontend_port\":[],\"ha_type\":\"double_site\",\"site_id\":\"t1\",\"backup_site_id\":\"t2\",\"rs_list\":[...]},\"msg\":\"successful\"}\nquery_loadbalancer_info根据lb_id查询LoadBalancer信息，返回信息包括LB属性、前端端口列表frontend_port、VM列表vm_list、RS列表rs_list、ha_type、site_id、backup_site_id、gw_type、eip、eip_type、mode等。\nlist_loadbalancers是高危接口，由于存量数据存在idkp混用，可能返回不属于自己的lb实例。操作前务必用query_loadbalancer_info再次确认idkp、bid、监听、rs等配置是否符合预期。\n注意: list_loadbalancers是高危接口，由于存量数据存在idkp混用，可能返回不属于自己的lb实例。操作前务必用query_loadbalancer_info再次确认idkp、bid、监听、rs等配置是否符合预期。",
    "paraphrases": [
      "当需要执行 query_loadbalancer_info 对应操作时，优先检查必选参数 region_no、lb_id、aliyun_idkp。",
      "query_loadbalancer_info 的核心触发锚点是 action=query_loadbalancer_info。"
    ],
    "recall_prompts": [
      "query_loadbalancer_info 是做什么的？",
      "什么时候应该调用 query_loadbalancer_info？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:query_loadbalancer_info]/[Field:region_no]\n参数 region_no 属于接口 query_loadbalancer_info，类型 string。必选参数。 LoadBalancer所属的region_no",
    "paraphrases": [
      "query_loadbalancer_info 的参数 region_no 必须提供。",
      "查询 query_loadbalancer_info 时，字段 region_no 的含义是：LoadBalancer所属的region_no"
    ],
    "recall_prompts": [
      "query_loadbalancer_info 的参数 region_no 是什么？",
      "query_loadbalancer_info 里 region_no 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:query_loadbalancer_info]/[Field:lb_id]\n参数 lb_id 属于接口 query_loadbalancer_info，类型 string。必选参数。 LoadBalancer的唯一标识",
    "paraphrases": [
      "query_loadbalancer_info 的参数 lb_id 必须提供。",
      "查询 query_loadbalancer_info 时，字段 lb_id 的含义是：LoadBalancer的唯一标识"
    ],
    "recall_prompts": [
      "query_loadbalancer_info 的参数 lb_id 是什么？",
      "query_loadbalancer_info 里 lb_id 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:query_loadbalancer_info]/[Field:aliyun_idkp]\n参数 aliyun_idkp 属于接口 query_loadbalancer_info，类型 string。必选参数。 阿里云云帐号ID",
    "paraphrases": [
      "query_loadbalancer_info 的参数 aliyun_idkp 必须提供。",
      "查询 query_loadbalancer_info 时，字段 aliyun_idkp 的含义是：阿里云云帐号ID"
    ],
    "recall_prompts": [
      "query_loadbalancer_info 的参数 aliyun_idkp 是什么？",
      "query_loadbalancer_info 里 aliyun_idkp 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:query_loadbalancer_info]/[Field:bid]\n参数 bid 属于接口 query_loadbalancer_info，类型 string。可选参数。 渠道ID",
    "paraphrases": [
      "query_loadbalancer_info 的参数 bid 可以不传。",
      "查询 query_loadbalancer_info 时，字段 bid 的含义是：渠道ID"
    ],
    "recall_prompts": [
      "query_loadbalancer_info 的参数 bid 是什么？",
      "query_loadbalancer_info 里 bid 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:query_loadbalancer_info]/[Constraint:2]\nlb_id: LoadBalancer的唯一标识",
    "paraphrases": [
      "接口 query_loadbalancer_info 存在约束：lb_id: LoadBalancer的唯一标识",
      "调用 query_loadbalancer_info 时需要注意：lb_id: LoadBalancer的唯一标识"
    ],
    "recall_prompts": [
      "query_loadbalancer_info 有哪些限制或条件？",
      "调用 query_loadbalancer_info 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:query_loadbalancer_info]/[Constraint:6]\nlist_loadbalancers是高危接口，由于存量数据存在idkp混用，可能返回不属于自己的lb实例。操作前务必用query_loadbalancer_info再次确认idkp、bid、监听、rs等配置是否符合预期。",
    "paraphrases": [
      "接口 query_loadbalancer_info 存在约束：list_loadbalancers是高危接口，由于存量数据存在idkp混用，可能返回不属于自己的lb实例。操作前务必用query_loadbalancer_info再次确认idkp、bid、监听、rs等配置是否符合预期。",
      "调用 query_loadbalancer_info 时需要注意：list_loadbalancers是高危接口，由于存量数据存在idkp混用，可能返回不属于自己的lb实例。操作前务必用query_loadbalancer_info再次确认idkp、bid、监听、rs等配置是否符合预期。"
    ],
    "recall_prompts": [
      "query_loadbalancer_info 有哪些限制或条件？",
      "调用 query_loadbalancer_info 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:query_loadbalancer_info]/[Constraint:7]\n注意: list_loadbalancers是高危接口，由于存量数据存在idkp混用，可能返回不属于自己的lb实例。操作前务必用query_loadbalancer_info再次确认idkp、bid、监听、rs等配置是否符合预期。",
    "paraphrases": [
      "接口 query_loadbalancer_info 存在约束：注意: list_loadbalancers是高危接口，由于存量数据存在idkp混用，可能返回不属于自己的lb实例。操作前务必用query_loadbalancer_info再次确认idkp、bid、监听、rs等配置是否符合预期。",
      "调用 query_loadbalancer_info 时需要注意：注意: list_loadbalancers是高危接口，由于存量数据存在idkp混用，可能返回不属于自己的lb实例。操作前务必用query_loadbalancer_info再次确认idkp、bid、监听、rs等配置是否符合预期。"
    ],
    "recall_prompts": [
      "query_loadbalancer_info 有哪些限制或条件？",
      "调用 query_loadbalancer_info 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:query_loadbalancer_info]/[Constraint:8]\n查询单个SLB实例详情用query_loadbalancer_info，传入region_no和lb_id即可获取包括vm_list、frontend_port列表、rs_list、ha_type、gw_type等完整信息。查询用户所有实例列表用list_loadbalancers，但这是高危接口，因为idkp混用可能返回非自己的实例，需要额外校验。",
    "paraphrases": [
      "接口 query_loadbalancer_info 存在约束：查询单个SLB实例详情用query_loadbalancer_info，传入region_no和lb_id即可获取包括vm_list、frontend_port列表、rs_list、ha_type、gw_type等完整信息。查询用户所有实例列表用list_loadbalancers，但这是高危接口，因为idkp混用可能返回非自己的实例，需要额外校验。",
      "调用 query_loadbalancer_info 时需要注意：查询单个SLB实例详情用query_loadbalancer_info，传入region_no和lb_id即可获取包括vm_list、frontend_port列表、rs_list、ha_type、gw_type等完整信息。查询用户所有实例列表用list_loadbalancers，但这是高危接口，因为idkp混用可能返回非自己的实例，需要额外校验。"
    ],
    "recall_prompts": [
      "query_loadbalancer_info 有哪些限制或条件？",
      "调用 query_loadbalancer_info 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:query_loadbalancer_info]/[Constraint:9]\nquery_loadbalancer_info接口返回SLB实例的完整属性，包含负载均衡的网络类型gw_type、IP地址eip、转发模式mode、灾备类型ha_type、前端端口列表和后端RS列表。list_loadbalancers接口虽然能批量查询用户所有LB，但由于历史数据idkp混用问题被标记为高危接口，建议查询后用query_loadbalancer_info逐一确认。",
    "paraphrases": [
      "接口 query_loadbalancer_info 存在约束：query_loadbalancer_info接口返回SLB实例的完整属性，包含负载均衡的网络类型gw_type、IP地址eip、转发模式mode、灾备类型ha_type、前端端口列表和后端RS列表。list_loadbalancers接口虽然能批量查询用户所有LB，但由于历史数据idkp混用问题被标记为高危接口，建议查询后用query_loadbalancer_info逐一确认。",
      "调用 query_loadbalancer_info 时需要注意：query_loadbalancer_info接口返回SLB实例的完整属性，包含负载均衡的网络类型gw_type、IP地址eip、转发模式mode、灾备类型ha_type、前端端口列表和后端RS列表。list_loadbalancers接口虽然能批量查询用户所有LB，但由于历史数据idkp混用问题被标记为高危接口，建议查询后用query_loadbalancer_info逐一确认。"
    ],
    "recall_prompts": [
      "query_loadbalancer_info 有哪些限制或条件？",
      "调用 query_loadbalancer_info 时要注意什么？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:query_loadbalancer_info]/[Error:-2601]\n当调用 query_loadbalancer_info 返回错误码 -2601 LbIdIsEmpty 时，表示 LB ID为空。",
    "paraphrases": [
      "query_loadbalancer_info 出现 -2601 时，对应错误名是 LbIdIsEmpty。",
      "错误 -2601 在 query_loadbalancer_info 中表示：LB ID为空。"
    ],
    "recall_prompts": [
      "query_loadbalancer_info 返回 -2601 代表什么？",
      "LbIdIsEmpty 是什么错误？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:query_loadbalancer_info]/[Error:-2610]\n当调用 query_loadbalancer_info 返回错误码 -2610 LbIdNotExist 时，表示 LB ID不存在。",
    "paraphrases": [
      "query_loadbalancer_info 出现 -2610 时，对应错误名是 LbIdNotExist。",
      "错误 -2610 在 query_loadbalancer_info 中表示：LB ID不存在。"
    ],
    "recall_prompts": [
      "query_loadbalancer_info 返回 -2610 代表什么？",
      "LbIdNotExist 是什么错误？"
    ]
  },
  {
    "type": "command_mapping",
    "content": "## create_vip - 创建VIP监听\n[Action:create_vip]/[Overview]\n接口 create_vip 用于 create_vip - 创建VIP监听。\n必选参数: region_no, lb_id, listeners, aliyun_idkp。\n成功返回: {\"code\":200,\"msg\":\"successful\"}\ncreate_vip保证事务性，添加的Listener要么全部成功要么全部失败。对于https的VIP必须绑定证书。proxy_protocol_v2_enabled为off时不允许带proxy_fields字段。\n说明: create_vip保证事务性，添加的Listener要么全部成功要么全部失败。对于https的VIP必须绑定证书。proxy_protocol_v2_enabled为off时不允许带proxy_fields字段。\ncreate_vip接口用于在SLB实例中创建VIP监听，支持tcp、http、https、udp、tcps五种协议。传入listeners参数为Listener结构JSON列表，每个Listener需指定protocol和frontend_port，后端可通过backend_port或rs_pool_name指定。create_vip具有事务性，一次最多10个Listener全部成功或全部失败。HTTPS类型VIP必须绑定证书。",
    "paraphrases": [
      "当需要执行 create_vip 对应操作时，优先检查必选参数 region_no、lb_id、listeners、aliyun_idkp。",
      "create_vip 的核心触发锚点是 action=create_vip。"
    ],
    "recall_prompts": [
      "create_vip 是做什么的？",
      "什么时候应该调用 create_vip？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:create_vip]/[Field:region_no]\n参数 region_no 属于接口 create_vip，类型 string。必选参数。 LoadBalancer所属的region_no",
    "paraphrases": [
      "create_vip 的参数 region_no 必须提供。",
      "查询 create_vip 时，字段 region_no 的含义是：LoadBalancer所属的region_no"
    ],
    "recall_prompts": [
      "create_vip 的参数 region_no 是什么？",
      "create_vip 里 region_no 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:create_vip]/[Field:lb_id]\n参数 lb_id 属于接口 create_vip，类型 string。必选参数。 LoadBalancer的唯一标识",
    "paraphrases": [
      "create_vip 的参数 lb_id 必须提供。",
      "查询 create_vip 时，字段 lb_id 的含义是：LoadBalancer的唯一标识"
    ],
    "recall_prompts": [
      "create_vip 的参数 lb_id 是什么？",
      "create_vip 里 lb_id 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:create_vip]/[Field:listeners]\n参数 listeners 属于接口 create_vip，类型 string。必选参数。 Listener结构的JSON列表，一次最多添加10个Listener",
    "paraphrases": [
      "create_vip 的参数 listeners 必须提供。",
      "查询 create_vip 时，字段 listeners 的含义是：Listener结构的JSON列表，一次最多添加10个Listener"
    ],
    "recall_prompts": [
      "create_vip 的参数 listeners 是什么？",
      "create_vip 里 listeners 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:create_vip]/[Field:aliyun_idkp]\n参数 aliyun_idkp 属于接口 create_vip，类型 string。必选参数。 阿里云云帐号ID",
    "paraphrases": [
      "create_vip 的参数 aliyun_idkp 必须提供。",
      "查询 create_vip 时，字段 aliyun_idkp 的含义是：阿里云云帐号ID"
    ],
    "recall_prompts": [
      "create_vip 的参数 aliyun_idkp 是什么？",
      "create_vip 里 aliyun_idkp 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_vip]/[Field:bid]\n参数 bid 属于接口 create_vip，类型 string。可选参数。 渠道ID",
    "paraphrases": [
      "create_vip 的参数 bid 可以不传。",
      "查询 create_vip 时，字段 bid 的含义是：渠道ID"
    ],
    "recall_prompts": [
      "create_vip 的参数 bid 是什么？",
      "create_vip 里 bid 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_vip]/[Constraint:2]\nlb_id: LoadBalancer的唯一标识",
    "paraphrases": [
      "接口 create_vip 存在约束：lb_id: LoadBalancer的唯一标识",
      "调用 create_vip 时需要注意：lb_id: LoadBalancer的唯一标识"
    ],
    "recall_prompts": [
      "create_vip 有哪些限制或条件？",
      "调用 create_vip 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_vip]/[Constraint:3]\nlisteners: Listener结构的JSON列表，一次最多添加10个Listener",
    "paraphrases": [
      "接口 create_vip 存在约束：listeners: Listener结构的JSON列表，一次最多添加10个Listener",
      "调用 create_vip 时需要注意：listeners: Listener结构的JSON列表，一次最多添加10个Listener"
    ],
    "recall_prompts": [
      "create_vip 有哪些限制或条件？",
      "调用 create_vip 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_vip]/[Constraint:6]\ncreate_vip保证事务性，添加的Listener要么全部成功要么全部失败。对于https的VIP必须绑定证书。proxy_protocol_v2_enabled为off时不允许带proxy_fields字段。",
    "paraphrases": [
      "接口 create_vip 存在约束：create_vip保证事务性，添加的Listener要么全部成功要么全部失败。对于https的VIP必须绑定证书。proxy_protocol_v2_enabled为off时不允许带proxy_fields字段。",
      "调用 create_vip 时需要注意：create_vip保证事务性，添加的Listener要么全部成功要么全部失败。对于https的VIP必须绑定证书。proxy_protocol_v2_enabled为off时不允许带proxy_fields字段。"
    ],
    "recall_prompts": [
      "create_vip 有哪些限制或条件？",
      "调用 create_vip 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_vip]/[Constraint:7]\n说明: create_vip保证事务性，添加的Listener要么全部成功要么全部失败。对于https的VIP必须绑定证书。proxy_protocol_v2_enabled为off时不允许带proxy_fields字段。",
    "paraphrases": [
      "接口 create_vip 存在约束：说明: create_vip保证事务性，添加的Listener要么全部成功要么全部失败。对于https的VIP必须绑定证书。proxy_protocol_v2_enabled为off时不允许带proxy_fields字段。",
      "调用 create_vip 时需要注意：说明: create_vip保证事务性，添加的Listener要么全部成功要么全部失败。对于https的VIP必须绑定证书。proxy_protocol_v2_enabled为off时不允许带proxy_fields字段。"
    ],
    "recall_prompts": [
      "create_vip 有哪些限制或条件？",
      "调用 create_vip 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_vip]/[Constraint:8]\ncreate_vip接口用于在SLB实例中创建VIP监听，支持tcp、http、https、udp、tcps五种协议。传入listeners参数为Listener结构JSON列表，每个Listener需指定protocol和frontend_port，后端可通过backend_port或rs_pool_name指定。create_vip具有事务性，一次最多10个Listener全部成功或全部失败。HTTPS类型VIP必须绑定证书。",
    "paraphrases": [
      "接口 create_vip 存在约束：create_vip接口用于在SLB实例中创建VIP监听，支持tcp、http、https、udp、tcps五种协议。传入listeners参数为Listener结构JSON列表，每个Listener需指定protocol和frontend_port，后端可通过backend_port或rs_pool_name指定。create_vip具有事务性，一次最多10个Listener全部成功或全部失败。HTTPS类型VIP必须绑定证书。",
      "调用 create_vip 时需要注意：create_vip接口用于在SLB实例中创建VIP监听，支持tcp、http、https、udp、tcps五种协议。传入listeners参数为Listener结构JSON列表，每个Listener需指定protocol和frontend_port，后端可通过backend_port或rs_pool_name指定。create_vip具有事务性，一次最多10个Listener全部成功或全部失败。HTTPS类型VIP必须绑定证书。"
    ],
    "recall_prompts": [
      "create_vip 有哪些限制或条件？",
      "调用 create_vip 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_vip]/[Constraint:9]\n在SLB中创建VIP监听调用create_vip接口，Listener结构中protocol和frontend_port为必选，后端服务通过backend_port直接指定端口或通过rs_pool_name关联RSPool。支持的协议包括tcp、http、https、udp、tcps。create_vip是事务性操作一次最多10个。创建https监听时必须绑定SSL证书否则报-2904错误。",
    "paraphrases": [
      "接口 create_vip 存在约束：在SLB中创建VIP监听调用create_vip接口，Listener结构中protocol和frontend_port为必选，后端服务通过backend_port直接指定端口或通过rs_pool_name关联RSPool。支持的协议包括tcp、http、https、udp、tcps。create_vip是事务性操作一次最多10个。创建https监听时必须绑定SSL证书否则报-2904错误。",
      "调用 create_vip 时需要注意：在SLB中创建VIP监听调用create_vip接口，Listener结构中protocol和frontend_port为必选，后端服务通过backend_port直接指定端口或通过rs_pool_name关联RSPool。支持的协议包括tcp、http、https、udp、tcps。create_vip是事务性操作一次最多10个。创建https监听时必须绑定SSL证书否则报-2904错误。"
    ],
    "recall_prompts": [
      "create_vip 有哪些限制或条件？",
      "调用 create_vip 时要注意什么？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:create_vip]/[Error:-2301]\n当调用 create_vip 返回错误码 -2301 VipExist 时，表示 VIP已经存在。",
    "paraphrases": [
      "create_vip 出现 -2301 时，对应错误名是 VipExist。",
      "错误 -2301 在 create_vip 中表示：VIP已经存在。"
    ],
    "recall_prompts": [
      "create_vip 返回 -2301 代表什么？",
      "VipExist 是什么错误？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:create_vip]/[Error:-2315]\n当调用 create_vip 返回错误码 -2315 VipTooManyListeners 时，表示 一次传入的Listener数目过多（上限10个）。",
    "paraphrases": [
      "create_vip 出现 -2315 时，对应错误名是 VipTooManyListeners。",
      "错误 -2315 在 create_vip 中表示：一次传入的Listener数目过多（上限10个）。"
    ],
    "recall_prompts": [
      "create_vip 返回 -2315 代表什么？",
      "VipTooManyListeners 是什么错误？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:create_vip]/[Error:-2325]\n当调用 create_vip 返回错误码 -2325 VipProtocolNotSupport 时，表示 VIP协议类型不合法。",
    "paraphrases": [
      "create_vip 出现 -2325 时，对应错误名是 VipProtocolNotSupport。",
      "错误 -2325 在 create_vip 中表示：VIP协议类型不合法。"
    ],
    "recall_prompts": [
      "create_vip 返回 -2325 代表什么？",
      "VipProtocolNotSupport 是什么错误？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:create_vip]/[Error:-2326]\n当调用 create_vip 返回错误码 -2326 VipBackendPortIsEmpty 时，表示 Compact类型LB的VIP后端端口为空。",
    "paraphrases": [
      "create_vip 出现 -2326 时，对应错误名是 VipBackendPortIsEmpty。",
      "错误 -2326 在 create_vip 中表示：Compact类型LB的VIP后端端口为空。"
    ],
    "recall_prompts": [
      "create_vip 返回 -2326 代表什么？",
      "VipBackendPortIsEmpty 是什么错误？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:create_vip]/[Error:-2904]\n当调用 create_vip 返回错误码 -2904 CertKeyNotExisted 时，表示 证书和私钥不存在。",
    "paraphrases": [
      "create_vip 出现 -2904 时，对应错误名是 CertKeyNotExisted。",
      "错误 -2904 在 create_vip 中表示：证书和私钥不存在。"
    ],
    "recall_prompts": [
      "create_vip 返回 -2904 代表什么？",
      "CertKeyNotExisted 是什么错误？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:create_vip]/[Error:-2905]\n当调用 create_vip 返回错误码 -2905 CertKeyIdEmpty 时，表示 证书ID为空。",
    "paraphrases": [
      "create_vip 出现 -2905 时，对应错误名是 CertKeyIdEmpty。",
      "错误 -2905 在 create_vip 中表示：证书ID为空。"
    ],
    "recall_prompts": [
      "create_vip 返回 -2905 代表什么？",
      "CertKeyIdEmpty 是什么错误？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:create_vip]/[Error:-2308]\n当调用 create_vip 返回错误码 -2308 VipNotMatchRspool 时，表示 VIP协议类型和RSPool协议类型不一致。",
    "paraphrases": [
      "create_vip 出现 -2308 时，对应错误名是 VipNotMatchRspool。",
      "错误 -2308 在 create_vip 中表示：VIP协议类型和RSPool协议类型不一致。"
    ],
    "recall_prompts": [
      "create_vip 返回 -2308 代表什么？",
      "VipNotMatchRspool 是什么错误？"
    ]
  },
  {
    "type": "command_mapping",
    "content": "## create_rs_pool - create_rs_pool + delete_rs_pool - 创建和删除RS Pool\n[Action:create_rs_pool]/[Overview]\n接口 create_rs_pool 用于 create_rs_pool + delete_rs_pool - 创建和删除RS Pool。\n必选参数: region_no, rs_pool_name, aliyun_idkp, bid。\n成功返回: {\"code\":200,\"msg\":\"successful\",\"data\":{\"name\":\"testpool\",\"port\":80}}",
    "paraphrases": [
      "当需要执行 create_rs_pool 对应操作时，优先检查必选参数 region_no、rs_pool_name、aliyun_idkp、bid。",
      "create_rs_pool 的核心触发锚点是 action=create_rs_pool。"
    ],
    "recall_prompts": [
      "create_rs_pool 是做什么的？",
      "什么时候应该调用 create_rs_pool？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:create_rs_pool]/[Field:region_no]\n参数 region_no 属于接口 create_rs_pool，类型 string。必选参数。 LoadBalancer所属的region_no",
    "paraphrases": [
      "create_rs_pool 的参数 region_no 必须提供。",
      "查询 create_rs_pool 时，字段 region_no 的含义是：LoadBalancer所属的region_no"
    ],
    "recall_prompts": [
      "create_rs_pool 的参数 region_no 是什么？",
      "create_rs_pool 里 region_no 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:create_rs_pool]/[Field:rs_pool_name]\n参数 rs_pool_name 属于接口 create_rs_pool，类型 string。必选参数。 RS Pool名称，长度1-80，同一用户下必须唯一",
    "paraphrases": [
      "create_rs_pool 的参数 rs_pool_name 必须提供。",
      "查询 create_rs_pool 时，字段 rs_pool_name 的含义是：RS Pool名称，长度1-80，同一用户下必须唯一"
    ],
    "recall_prompts": [
      "create_rs_pool 的参数 rs_pool_name 是什么？",
      "create_rs_pool 里 rs_pool_name 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:create_rs_pool]/[Field:aliyun_idkp]\n参数 aliyun_idkp 属于接口 create_rs_pool，类型 string。必选参数。 阿里云云帐号ID",
    "paraphrases": [
      "create_rs_pool 的参数 aliyun_idkp 必须提供。",
      "查询 create_rs_pool 时，字段 aliyun_idkp 的含义是：阿里云云帐号ID"
    ],
    "recall_prompts": [
      "create_rs_pool 的参数 aliyun_idkp 是什么？",
      "create_rs_pool 里 aliyun_idkp 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:create_rs_pool]/[Field:bid]\n参数 bid 属于接口 create_rs_pool，类型 string。必选参数。 渠道ID",
    "paraphrases": [
      "create_rs_pool 的参数 bid 必须提供。",
      "查询 create_rs_pool 时，字段 bid 的含义是：渠道ID"
    ],
    "recall_prompts": [
      "create_rs_pool 的参数 bid 是什么？",
      "create_rs_pool 里 bid 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_rs_pool]/[Field:port]\n参数 port 属于接口 create_rs_pool，类型 int。可选参数。 端口1-65535。如果在rspool上设置port，则rspool中的rs不要再设port；如果需要每个rs使用不同port，则不要在rspool上设port",
    "paraphrases": [
      "create_rs_pool 的参数 port 可以不传。",
      "查询 create_rs_pool 时，字段 port 的含义是：端口1-65535。如果在rspool上设置port，则rspool中的rs不要再设port；如果需要每个rs使用不同port，则不要在rspool上设port"
    ],
    "recall_prompts": [
      "create_rs_pool 的参数 port 是什么？",
      "create_rs_pool 里 port 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_rs_pool]/[Field:type]\n参数 type 属于接口 create_rs_pool，类型 string。可选参数。 传app则创建应用型rspool，不传默认普通rspool",
    "paraphrases": [
      "create_rs_pool 的参数 type 可以不传。",
      "查询 create_rs_pool 时，字段 type 的含义是：传app则创建应用型rspool，不传默认普通rspool"
    ],
    "recall_prompts": [
      "create_rs_pool 的参数 type 是什么？",
      "create_rs_pool 里 type 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_rs_pool]/[Field:config]\n参数 config 属于接口 create_rs_pool，类型 string。可选参数。 type=app时生效，支持scheduler、sticky_session、check参数",
    "paraphrases": [
      "create_rs_pool 的参数 config 可以不传。",
      "查询 create_rs_pool 时，字段 config 的含义是：type=app时生效，支持scheduler、sticky_session、check参数"
    ],
    "recall_prompts": [
      "create_rs_pool 的参数 config 是什么？",
      "create_rs_pool 里 config 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_rs_pool]/[Constraint:2]\nrs_pool_name: RS Pool名称，长度1-80，同一用户下必须唯一",
    "paraphrases": [
      "接口 create_rs_pool 存在约束：rs_pool_name: RS Pool名称，长度1-80，同一用户下必须唯一",
      "调用 create_rs_pool 时需要注意：rs_pool_name: RS Pool名称，长度1-80，同一用户下必须唯一"
    ],
    "recall_prompts": [
      "create_rs_pool 有哪些限制或条件？",
      "调用 create_rs_pool 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:create_rs_pool]/[Constraint:6]\ntype: 传app则创建应用型rspool，不传默认普通rspool",
    "paraphrases": [
      "接口 create_rs_pool 存在约束：type: 传app则创建应用型rspool，不传默认普通rspool",
      "调用 create_rs_pool 时需要注意：type: 传app则创建应用型rspool，不传默认普通rspool"
    ],
    "recall_prompts": [
      "create_rs_pool 有哪些限制或条件？",
      "调用 create_rs_pool 时要注意什么？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:create_rs_pool]/[Error:-2500]\n当调用 create_rs_pool 返回错误码 -2500 RspoolNameIsEmpty 时，表示 RSPool名称为空。",
    "paraphrases": [
      "create_rs_pool 出现 -2500 时，对应错误名是 RspoolNameIsEmpty。",
      "错误 -2500 在 create_rs_pool 中表示：RSPool名称为空。"
    ],
    "recall_prompts": [
      "create_rs_pool 返回 -2500 代表什么？",
      "RspoolNameIsEmpty 是什么错误？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:create_rs_pool]/[Error:-2503]\n当调用 create_rs_pool 返回错误码 -2503 RspoolNameExist 时，表示 RSPool已经存在。",
    "paraphrases": [
      "create_rs_pool 出现 -2503 时，对应错误名是 RspoolNameExist。",
      "错误 -2503 在 create_rs_pool 中表示：RSPool已经存在。"
    ],
    "recall_prompts": [
      "create_rs_pool 返回 -2503 代表什么？",
      "RspoolNameExist 是什么错误？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:create_rs_pool]/[Error:-2511]\n当调用 create_rs_pool 返回错误码 -2511 RspoolNameNotSupport 时，表示 RSPool名称错误。",
    "paraphrases": [
      "create_rs_pool 出现 -2511 时，对应错误名是 RspoolNameNotSupport。",
      "错误 -2511 在 create_rs_pool 中表示：RSPool名称错误。"
    ],
    "recall_prompts": [
      "create_rs_pool 返回 -2511 代表什么？",
      "RspoolNameNotSupport 是什么错误？"
    ]
  },
  {
    "type": "command_mapping",
    "content": "## delete_rs_pool - create_rs_pool + delete_rs_pool - 创建和删除RS Pool\n[Action:delete_rs_pool]/[Overview]\n接口 delete_rs_pool 用于 create_rs_pool + delete_rs_pool - 创建和删除RS Pool。\n必选参数: region_no, rs_pool_name, aliyun_idkp, bid。\n成功返回: {\"code\":200,\"msg\":\"successful\",\"data\":{\"name\":\"pool1\",\"port\":80,\"protocol\":\"http\",\"realservers\":[...]}}\ndelete_rs_pool在有VIP关联该RSPool时不能删除，需要先解除VIP与RSPool的关联。没有VIP关联时才能成功删除，返回删除前的RS列表。\ncreate_rs_pool接口创建RS Pool后端服务器池，rs_pool_name在同一用户下必须唯一。可以通过port参数设置统一后端端口，也可以不设port让每个RS使用不同端口。type=app时创建应用型rspool支持配置scheduler和check。delete_rs_pool删除RSPool时必须先确保没有VIP关联。\n创建RSPool调用create_rs_pool，核心参数是rs_pool_name（用户下唯一）。RSPool的port设置策略：如果设置了port，则池中所有RS共用此port；如果不设置port，则每个RS在add_rs时单独指定port。应用型RSPool通过type=app创建，支持调度算法和健康检查配置。删除RSPool前需确保无VIP引用。",
    "paraphrases": [
      "当需要执行 delete_rs_pool 对应操作时，优先检查必选参数 region_no、rs_pool_name、aliyun_idkp、bid。",
      "delete_rs_pool 的核心触发锚点是 action=delete_rs_pool。"
    ],
    "recall_prompts": [
      "delete_rs_pool 是做什么的？",
      "什么时候应该调用 delete_rs_pool？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:delete_rs_pool]/[Field:region_no]\n参数 region_no 属于接口 delete_rs_pool，类型 string。必选参数。",
    "paraphrases": [
      "delete_rs_pool 的参数 region_no 必须提供。"
    ],
    "recall_prompts": [
      "delete_rs_pool 的参数 region_no 是什么？",
      "delete_rs_pool 里 region_no 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:delete_rs_pool]/[Field:rs_pool_name]\n参数 rs_pool_name 属于接口 delete_rs_pool，类型 string。必选参数。 RS Pool名称",
    "paraphrases": [
      "delete_rs_pool 的参数 rs_pool_name 必须提供。",
      "查询 delete_rs_pool 时，字段 rs_pool_name 的含义是：RS Pool名称"
    ],
    "recall_prompts": [
      "delete_rs_pool 的参数 rs_pool_name 是什么？",
      "delete_rs_pool 里 rs_pool_name 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:delete_rs_pool]/[Field:aliyun_idkp]\n参数 aliyun_idkp 属于接口 delete_rs_pool，类型 string。必选参数。",
    "paraphrases": [
      "delete_rs_pool 的参数 aliyun_idkp 必须提供。"
    ],
    "recall_prompts": [
      "delete_rs_pool 的参数 aliyun_idkp 是什么？",
      "delete_rs_pool 里 aliyun_idkp 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:delete_rs_pool]/[Field:bid]\n参数 bid 属于接口 delete_rs_pool，类型 string。必选参数。",
    "paraphrases": [
      "delete_rs_pool 的参数 bid 必须提供。"
    ],
    "recall_prompts": [
      "delete_rs_pool 的参数 bid 是什么？",
      "delete_rs_pool 里 bid 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:delete_rs_pool]/[Constraint:3]\ncreate_rs_pool接口创建RS Pool后端服务器池，rs_pool_name在同一用户下必须唯一。可以通过port参数设置统一后端端口，也可以不设port让每个RS使用不同端口。type=app时创建应用型rspool支持配置scheduler和check。delete_rs_pool删除RSPool时必须先确保没有VIP关联。",
    "paraphrases": [
      "接口 delete_rs_pool 存在约束：create_rs_pool接口创建RS Pool后端服务器池，rs_pool_name在同一用户下必须唯一。可以通过port参数设置统一后端端口，也可以不设port让每个RS使用不同端口。type=app时创建应用型rspool支持配置scheduler和check。delete_rs_pool删除RSPool时必须先确保没有VIP关联。",
      "调用 delete_rs_pool 时需要注意：create_rs_pool接口创建RS Pool后端服务器池，rs_pool_name在同一用户下必须唯一。可以通过port参数设置统一后端端口，也可以不设port让每个RS使用不同端口。type=app时创建应用型rspool支持配置scheduler和check。delete_rs_pool删除RSPool时必须先确保没有VIP关联。"
    ],
    "recall_prompts": [
      "delete_rs_pool 有哪些限制或条件？",
      "调用 delete_rs_pool 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:delete_rs_pool]/[Constraint:4]\n创建RSPool调用create_rs_pool，核心参数是rs_pool_name（用户下唯一）。RSPool的port设置策略：如果设置了port，则池中所有RS共用此port；如果不设置port，则每个RS在add_rs时单独指定port。应用型RSPool通过type=app创建，支持调度算法和健康检查配置。删除RSPool前需确保无VIP引用。",
    "paraphrases": [
      "接口 delete_rs_pool 存在约束：创建RSPool调用create_rs_pool，核心参数是rs_pool_name（用户下唯一）。RSPool的port设置策略：如果设置了port，则池中所有RS共用此port；如果不设置port，则每个RS在add_rs时单独指定port。应用型RSPool通过type=app创建，支持调度算法和健康检查配置。删除RSPool前需确保无VIP引用。",
      "调用 delete_rs_pool 时需要注意：创建RSPool调用create_rs_pool，核心参数是rs_pool_name（用户下唯一）。RSPool的port设置策略：如果设置了port，则池中所有RS共用此port；如果不设置port，则每个RS在add_rs时单独指定port。应用型RSPool通过type=app创建，支持调度算法和健康检查配置。删除RSPool前需确保无VIP引用。"
    ],
    "recall_prompts": [
      "delete_rs_pool 有哪些限制或条件？",
      "调用 delete_rs_pool 时要注意什么？"
    ]
  },
  {
    "type": "command_mapping",
    "content": "## add_rs - add_rs + config_rs + delete_rs + switch_rs - RSPool级别RS操作\n[Action:add_rs]/[Overview]\n接口 add_rs 用于 add_rs + config_rs + delete_rs + switch_rs - RSPool级别RS操作。\n必选参数: region_no, rs_pool_name, rs_list, aliyun_idkp, bid。\n成功返回: {\"code\":200,\"msg\":\"successful\",\"data\":{\"rs_pool_name\":\"testpool1\",\"rs_list\":[...]}}\nadd_rs添加一组RS到指定RSPool。RS需指定port（每个RS可使用不同port）。已存在的RS会被忽略不报错。VIP为running状态时进入configuring，stopped状态保持不变。不允许添加的rs本身是lb类型，不支持lb串联。同一rspool中tunnel_id必须相同。\n说明: add_rs添加一组RS到指定RSPool。RS需指定port（每个RS可使用不同port）。已存在的RS会被忽略不报错。VIP为running状态时进入configuring，stopped状态保持不变。不允许添加的rs本身是lb类型，不支持lb串联。同一rspool中tunnel_id必须相同。",
    "paraphrases": [
      "当需要执行 add_rs 对应操作时，优先检查必选参数 region_no、rs_pool_name、rs_list、aliyun_idkp、bid。",
      "add_rs 的核心触发锚点是 action=add_rs。"
    ],
    "recall_prompts": [
      "add_rs 是做什么的？",
      "什么时候应该调用 add_rs？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:add_rs]/[Field:region_no]\n参数 region_no 属于接口 add_rs，类型 string。必选参数。",
    "paraphrases": [
      "add_rs 的参数 region_no 必须提供。"
    ],
    "recall_prompts": [
      "add_rs 的参数 region_no 是什么？",
      "add_rs 里 region_no 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:add_rs]/[Field:rs_pool_name]\n参数 rs_pool_name 属于接口 add_rs，类型 string。必选参数。 RS Pool名称",
    "paraphrases": [
      "add_rs 的参数 rs_pool_name 必须提供。",
      "查询 add_rs 时，字段 rs_pool_name 的含义是：RS Pool名称"
    ],
    "recall_prompts": [
      "add_rs 的参数 rs_pool_name 是什么？",
      "add_rs 里 rs_pool_name 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:add_rs]/[Field:rs_list]\n参数 rs_list 属于接口 add_rs，类型 string。必选参数。 RealServer结构JSON列表，最多20个",
    "paraphrases": [
      "add_rs 的参数 rs_list 必须提供。",
      "查询 add_rs 时，字段 rs_list 的含义是：RealServer结构JSON列表，最多20个"
    ],
    "recall_prompts": [
      "add_rs 的参数 rs_list 是什么？",
      "add_rs 里 rs_list 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:add_rs]/[Field:aliyun_idkp]\n参数 aliyun_idkp 属于接口 add_rs，类型 string。必选参数。",
    "paraphrases": [
      "add_rs 的参数 aliyun_idkp 必须提供。"
    ],
    "recall_prompts": [
      "add_rs 的参数 aliyun_idkp 是什么？",
      "add_rs 里 aliyun_idkp 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:add_rs]/[Field:bid]\n参数 bid 属于接口 add_rs，类型 string。必选参数。",
    "paraphrases": [
      "add_rs 的参数 bid 必须提供。"
    ],
    "recall_prompts": [
      "add_rs 的参数 bid 是什么？",
      "add_rs 里 bid 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:add_rs]/[Constraint:2]\nrs_list: RealServer结构JSON列表，最多20个",
    "paraphrases": [
      "接口 add_rs 存在约束：rs_list: RealServer结构JSON列表，最多20个",
      "调用 add_rs 时需要注意：rs_list: RealServer结构JSON列表，最多20个"
    ],
    "recall_prompts": [
      "add_rs 有哪些限制或条件？",
      "调用 add_rs 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:add_rs]/[Constraint:3]\nadd_rs添加一组RS到指定RSPool。RS需指定port（每个RS可使用不同port）。已存在的RS会被忽略不报错。VIP为running状态时进入configuring，stopped状态保持不变。不允许添加的rs本身是lb类型，不支持lb串联。同一rspool中tunnel_id必须相同。",
    "paraphrases": [
      "接口 add_rs 存在约束：add_rs添加一组RS到指定RSPool。RS需指定port（每个RS可使用不同port）。已存在的RS会被忽略不报错。VIP为running状态时进入configuring，stopped状态保持不变。不允许添加的rs本身是lb类型，不支持lb串联。同一rspool中tunnel_id必须相同。",
      "调用 add_rs 时需要注意：add_rs添加一组RS到指定RSPool。RS需指定port（每个RS可使用不同port）。已存在的RS会被忽略不报错。VIP为running状态时进入configuring，stopped状态保持不变。不允许添加的rs本身是lb类型，不支持lb串联。同一rspool中tunnel_id必须相同。"
    ],
    "recall_prompts": [
      "add_rs 有哪些限制或条件？",
      "调用 add_rs 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:add_rs]/[Constraint:4]\n说明: add_rs添加一组RS到指定RSPool。RS需指定port（每个RS可使用不同port）。已存在的RS会被忽略不报错。VIP为running状态时进入configuring，stopped状态保持不变。不允许添加的rs本身是lb类型，不支持lb串联。同一rspool中tunnel_id必须相同。",
    "paraphrases": [
      "接口 add_rs 存在约束：说明: add_rs添加一组RS到指定RSPool。RS需指定port（每个RS可使用不同port）。已存在的RS会被忽略不报错。VIP为running状态时进入configuring，stopped状态保持不变。不允许添加的rs本身是lb类型，不支持lb串联。同一rspool中tunnel_id必须相同。",
      "调用 add_rs 时需要注意：说明: add_rs添加一组RS到指定RSPool。RS需指定port（每个RS可使用不同port）。已存在的RS会被忽略不报错。VIP为running状态时进入configuring，stopped状态保持不变。不允许添加的rs本身是lb类型，不支持lb串联。同一rspool中tunnel_id必须相同。"
    ],
    "recall_prompts": [
      "add_rs 有哪些限制或条件？",
      "调用 add_rs 时要注意什么？"
    ]
  },
  {
    "type": "command_mapping",
    "content": "## config_rs - add_rs + config_rs + delete_rs + switch_rs - RSPool级别RS操作\n[Action:config_rs]/[Overview]\n接口 config_rs 用于 add_rs + config_rs + delete_rs + switch_rs - RSPool级别RS操作。\n必选参数: 无。\nconfig_rs对指定RSPool中一组RS的weight值进行配置。",
    "paraphrases": [
      "config_rs 的核心触发锚点是 action=config_rs。"
    ],
    "recall_prompts": [
      "config_rs 是做什么的？",
      "什么时候应该调用 config_rs？"
    ]
  },
  {
    "type": "command_mapping",
    "content": "## delete_rs - add_rs + config_rs + delete_rs + switch_rs - RSPool级别RS操作\n[Action:delete_rs]/[Overview]\n接口 delete_rs 用于 add_rs + config_rs + delete_rs + switch_rs - RSPool级别RS操作。\n必选参数: 无。\ndelete_rs从RSPool中去除一组RS。已删除的RS会被忽略不报错。\n说明: delete_rs从RSPool中去除一组RS。已删除的RS会被忽略不报错。",
    "paraphrases": [
      "delete_rs 的核心触发锚点是 action=delete_rs。"
    ],
    "recall_prompts": [
      "delete_rs 是做什么的？",
      "什么时候应该调用 delete_rs？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:delete_rs]/[Constraint:1]\ndelete_rs从RSPool中去除一组RS。已删除的RS会被忽略不报错。",
    "paraphrases": [
      "接口 delete_rs 存在约束：delete_rs从RSPool中去除一组RS。已删除的RS会被忽略不报错。",
      "调用 delete_rs 时需要注意：delete_rs从RSPool中去除一组RS。已删除的RS会被忽略不报错。"
    ],
    "recall_prompts": [
      "delete_rs 有哪些限制或条件？",
      "调用 delete_rs 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:delete_rs]/[Constraint:2]\n说明: delete_rs从RSPool中去除一组RS。已删除的RS会被忽略不报错。",
    "paraphrases": [
      "接口 delete_rs 存在约束：说明: delete_rs从RSPool中去除一组RS。已删除的RS会被忽略不报错。",
      "调用 delete_rs 时需要注意：说明: delete_rs从RSPool中去除一组RS。已删除的RS会被忽略不报错。"
    ],
    "recall_prompts": [
      "delete_rs 有哪些限制或条件？",
      "调用 delete_rs 时要注意什么？"
    ]
  },
  {
    "type": "command_mapping",
    "content": "## switch_rs - add_rs + config_rs + delete_rs + switch_rs - RSPool级别RS操作\n[Action:switch_rs]/[Overview]\n接口 switch_rs 用于 add_rs + config_rs + delete_rs + switch_rs - RSPool级别RS操作。\n必选参数: region_no, rs_pool_name, old_rs, new_rs, aliyun_idkp, bid。\nswitch_rs原子替换一组RS，old_rs中不存在的RS会被忽略。返回更新后的RS列表。\n在RSPool级别操作RS有四个接口：add_rs添加后端服务器，config_rs配置RS权重，delete_rs删除后端服务器，switch_rs原子替换一组RS。RSPool级别的RS操作中每个RS需要单独指定port，同一rspool中所有RS的tunnel_id必须相同。add_rs和delete_rs对已存在或已删除的RS都会忽略不报错。\nRSPool级别RS操作的特点：add_rs向rspool添加RS时每个RS可指定不同port，最多20个；config_rs修改RS的weight值；delete_rs移除RS；switch_rs用new_rs替换old_rs实现原子切换。所有操作中如果关联的VIP处于running状态会自动进入configuring状态。不允许lb串联，即rs不能是lb类型。",
    "paraphrases": [
      "当需要执行 switch_rs 对应操作时，优先检查必选参数 region_no、rs_pool_name、old_rs、new_rs、aliyun_idkp、bid。",
      "switch_rs 的核心触发锚点是 action=switch_rs。"
    ],
    "recall_prompts": [
      "switch_rs 是做什么的？",
      "什么时候应该调用 switch_rs？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:switch_rs]/[Field:region_no]\n参数 region_no 属于接口 switch_rs，类型 string。必选参数。",
    "paraphrases": [
      "switch_rs 的参数 region_no 必须提供。"
    ],
    "recall_prompts": [
      "switch_rs 的参数 region_no 是什么？",
      "switch_rs 里 region_no 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:switch_rs]/[Field:rs_pool_name]\n参数 rs_pool_name 属于接口 switch_rs，类型 string。必选参数。",
    "paraphrases": [
      "switch_rs 的参数 rs_pool_name 必须提供。"
    ],
    "recall_prompts": [
      "switch_rs 的参数 rs_pool_name 是什么？",
      "switch_rs 里 rs_pool_name 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:switch_rs]/[Field:old_rs]\n参数 old_rs 属于接口 switch_rs，类型 string。必选参数。 被删除的RS列表，最多20个",
    "paraphrases": [
      "switch_rs 的参数 old_rs 必须提供。",
      "查询 switch_rs 时，字段 old_rs 的含义是：被删除的RS列表，最多20个"
    ],
    "recall_prompts": [
      "switch_rs 的参数 old_rs 是什么？",
      "switch_rs 里 old_rs 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:switch_rs]/[Field:new_rs]\n参数 new_rs 属于接口 switch_rs，类型 string。必选参数。 新增的RS列表，最多20个",
    "paraphrases": [
      "switch_rs 的参数 new_rs 必须提供。",
      "查询 switch_rs 时，字段 new_rs 的含义是：新增的RS列表，最多20个"
    ],
    "recall_prompts": [
      "switch_rs 的参数 new_rs 是什么？",
      "switch_rs 里 new_rs 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:switch_rs]/[Field:aliyun_idkp]\n参数 aliyun_idkp 属于接口 switch_rs，类型 string。必选参数。",
    "paraphrases": [
      "switch_rs 的参数 aliyun_idkp 必须提供。"
    ],
    "recall_prompts": [
      "switch_rs 的参数 aliyun_idkp 是什么？",
      "switch_rs 里 aliyun_idkp 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:switch_rs]/[Field:bid]\n参数 bid 属于接口 switch_rs，类型 string。必选参数。",
    "paraphrases": [
      "switch_rs 的参数 bid 必须提供。"
    ],
    "recall_prompts": [
      "switch_rs 的参数 bid 是什么？",
      "switch_rs 里 bid 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:switch_rs]/[Constraint:1]\nold_rs: 被删除的RS列表，最多20个",
    "paraphrases": [
      "接口 switch_rs 存在约束：old_rs: 被删除的RS列表，最多20个",
      "调用 switch_rs 时需要注意：old_rs: 被删除的RS列表，最多20个"
    ],
    "recall_prompts": [
      "switch_rs 有哪些限制或条件？",
      "调用 switch_rs 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:switch_rs]/[Constraint:2]\nnew_rs: 新增的RS列表，最多20个",
    "paraphrases": [
      "接口 switch_rs 存在约束：new_rs: 新增的RS列表，最多20个",
      "调用 switch_rs 时需要注意：new_rs: 新增的RS列表，最多20个"
    ],
    "recall_prompts": [
      "switch_rs 有哪些限制或条件？",
      "调用 switch_rs 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:switch_rs]/[Constraint:4]\n在RSPool级别操作RS有四个接口：add_rs添加后端服务器，config_rs配置RS权重，delete_rs删除后端服务器，switch_rs原子替换一组RS。RSPool级别的RS操作中每个RS需要单独指定port，同一rspool中所有RS的tunnel_id必须相同。add_rs和delete_rs对已存在或已删除的RS都会忽略不报错。",
    "paraphrases": [
      "接口 switch_rs 存在约束：在RSPool级别操作RS有四个接口：add_rs添加后端服务器，config_rs配置RS权重，delete_rs删除后端服务器，switch_rs原子替换一组RS。RSPool级别的RS操作中每个RS需要单独指定port，同一rspool中所有RS的tunnel_id必须相同。add_rs和delete_rs对已存在或已删除的RS都会忽略不报错。",
      "调用 switch_rs 时需要注意：在RSPool级别操作RS有四个接口：add_rs添加后端服务器，config_rs配置RS权重，delete_rs删除后端服务器，switch_rs原子替换一组RS。RSPool级别的RS操作中每个RS需要单独指定port，同一rspool中所有RS的tunnel_id必须相同。add_rs和delete_rs对已存在或已删除的RS都会忽略不报错。"
    ],
    "recall_prompts": [
      "switch_rs 有哪些限制或条件？",
      "调用 switch_rs 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:switch_rs]/[Constraint:5]\nRSPool级别RS操作的特点：add_rs向rspool添加RS时每个RS可指定不同port，最多20个；config_rs修改RS的weight值；delete_rs移除RS；switch_rs用new_rs替换old_rs实现原子切换。所有操作中如果关联的VIP处于running状态会自动进入configuring状态。不允许lb串联，即rs不能是lb类型。",
    "paraphrases": [
      "接口 switch_rs 存在约束：RSPool级别RS操作的特点：add_rs向rspool添加RS时每个RS可指定不同port，最多20个；config_rs修改RS的weight值；delete_rs移除RS；switch_rs用new_rs替换old_rs实现原子切换。所有操作中如果关联的VIP处于running状态会自动进入configuring状态。不允许lb串联，即rs不能是lb类型。",
      "调用 switch_rs 时需要注意：RSPool级别RS操作的特点：add_rs向rspool添加RS时每个RS可指定不同port，最多20个；config_rs修改RS的weight值；delete_rs移除RS；switch_rs用new_rs替换old_rs实现原子切换。所有操作中如果关联的VIP处于running状态会自动进入configuring状态。不允许lb串联，即rs不能是lb类型。"
    ],
    "recall_prompts": [
      "switch_rs 有哪些限制或条件？",
      "调用 switch_rs 时要注意什么？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:switch_rs]/[Error:-2500]\n当调用 switch_rs 返回错误码 -2500 RspoolNameIsEmpty 时，表示 RSPool名称为空。",
    "paraphrases": [
      "switch_rs 出现 -2500 时，对应错误名是 RspoolNameIsEmpty。",
      "错误 -2500 在 switch_rs 中表示：RSPool名称为空。"
    ],
    "recall_prompts": [
      "switch_rs 返回 -2500 代表什么？",
      "RspoolNameIsEmpty 是什么错误？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:switch_rs]/[Error:-2501]\n当调用 switch_rs 返回错误码 -2501 LbRsListIsEmpty 时，表示 RS-list为空。",
    "paraphrases": [
      "switch_rs 出现 -2501 时，对应错误名是 LbRsListIsEmpty。",
      "错误 -2501 在 switch_rs 中表示：RS-list为空。"
    ],
    "recall_prompts": [
      "switch_rs 返回 -2501 代表什么？",
      "LbRsListIsEmpty 是什么错误？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:switch_rs]/[Error:-2502]\n当调用 switch_rs 返回错误码 -2502 LB_RS_List_Illegal 时，表示 RS-list格式非法。",
    "paraphrases": [
      "switch_rs 出现 -2502 时，对应错误名是 LB_RS_List_Illegal。",
      "错误 -2502 在 switch_rs 中表示：RS-list格式非法。"
    ],
    "recall_prompts": [
      "switch_rs 返回 -2502 代表什么？",
      "LB_RS_List_Illegal 是什么错误？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:switch_rs]/[Error:-2510]\n当调用 switch_rs 返回错误码 -2510 RealServerWeightNotSupport 时，表示 RS权重错误。",
    "paraphrases": [
      "switch_rs 出现 -2510 时，对应错误名是 RealServerWeightNotSupport。",
      "错误 -2510 在 switch_rs 中表示：RS权重错误。"
    ],
    "recall_prompts": [
      "switch_rs 返回 -2510 代表什么？",
      "RealServerWeightNotSupport 是什么错误？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:switch_rs]/[Error:-2512]\n当调用 switch_rs 返回错误码 -2512 RealServerToMany 时，表示 RS数目过多。",
    "paraphrases": [
      "switch_rs 出现 -2512 时，对应错误名是 RealServerToMany。",
      "错误 -2512 在 switch_rs 中表示：RS数目过多。"
    ],
    "recall_prompts": [
      "switch_rs 返回 -2512 代表什么？",
      "RealServerToMany 是什么错误？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:switch_rs]/[Error:-2514]\n当调用 switch_rs 返回错误码 -2514 RealServerPortNotSupport 时，表示 RS端口不支持。",
    "paraphrases": [
      "switch_rs 出现 -2514 时，对应错误名是 RealServerPortNotSupport。",
      "错误 -2514 在 switch_rs 中表示：RS端口不支持。"
    ],
    "recall_prompts": [
      "switch_rs 返回 -2514 代表什么？",
      "RealServerPortNotSupport 是什么错误？"
    ]
  },
  {
    "type": "command_mapping",
    "content": "## add_lb_rs - add_lb_rs + config_lb_rs + delete_lb_rs + switch_lb_rs - LB实例级别RS操作\n[Action:add_lb_rs]/[Overview]\n接口 add_lb_rs 用于 add_lb_rs + config_lb_rs + delete_lb_rs + switch_lb_rs - LB实例级别RS操作。\n必选参数: region_no, lb_id, rs_list, aliyun_idkp。\n成功返回: {\"code\":200,\"msg\":\"successful\",\"data\":{\"lb_id\":\"12345678\",\"rs_list\":[{\"rs_ip\":\"1.1.1.1\",\"weight\":100}]}}\nadd_lb_rs为LoadBalancer添加一组RS。RS不能指定port（所有RS共用监听的后端port）。添加到LB后端的RS会自动关联到该LB实例的每个监听上。已存在的RS会被忽略不报错。不允许rs本身是lb类型，不允许lb挂lb。\n说明: add_lb_rs为LoadBalancer添加一组RS。RS不能指定port（所有RS共用监听的后端port）。添加到LB后端的RS会自动关联到该LB实例的每个监听上。已存在的RS会被忽略不报错。不允许rs本身是lb类型，不允许lb挂lb。",
    "paraphrases": [
      "当需要执行 add_lb_rs 对应操作时，优先检查必选参数 region_no、lb_id、rs_list、aliyun_idkp。",
      "add_lb_rs 的核心触发锚点是 action=add_lb_rs。"
    ],
    "recall_prompts": [
      "add_lb_rs 是做什么的？",
      "什么时候应该调用 add_lb_rs？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:add_lb_rs]/[Field:region_no]\n参数 region_no 属于接口 add_lb_rs，类型 string。必选参数。",
    "paraphrases": [
      "add_lb_rs 的参数 region_no 必须提供。"
    ],
    "recall_prompts": [
      "add_lb_rs 的参数 region_no 是什么？",
      "add_lb_rs 里 region_no 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:add_lb_rs]/[Field:lb_id]\n参数 lb_id 属于接口 add_lb_rs，类型 string。必选参数。 LoadBalancer的唯一标识",
    "paraphrases": [
      "add_lb_rs 的参数 lb_id 必须提供。",
      "查询 add_lb_rs 时，字段 lb_id 的含义是：LoadBalancer的唯一标识"
    ],
    "recall_prompts": [
      "add_lb_rs 的参数 lb_id 是什么？",
      "add_lb_rs 里 lb_id 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:add_lb_rs]/[Field:rs_list]\n参数 rs_list 属于接口 add_lb_rs，类型 string。必选参数。 RealServer结构JSON列表，最多20个",
    "paraphrases": [
      "add_lb_rs 的参数 rs_list 必须提供。",
      "查询 add_lb_rs 时，字段 rs_list 的含义是：RealServer结构JSON列表，最多20个"
    ],
    "recall_prompts": [
      "add_lb_rs 的参数 rs_list 是什么？",
      "add_lb_rs 里 rs_list 要不要传？"
    ]
  },
  {
    "type": "fact_pair",
    "content": "[Action:add_lb_rs]/[Field:aliyun_idkp]\n参数 aliyun_idkp 属于接口 add_lb_rs，类型 string。必选参数。",
    "paraphrases": [
      "add_lb_rs 的参数 aliyun_idkp 必须提供。"
    ],
    "recall_prompts": [
      "add_lb_rs 的参数 aliyun_idkp 是什么？",
      "add_lb_rs 里 aliyun_idkp 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:add_lb_rs]/[Field:bid]\n参数 bid 属于接口 add_lb_rs，类型 string。可选参数。",
    "paraphrases": [
      "add_lb_rs 的参数 bid 可以不传。"
    ],
    "recall_prompts": [
      "add_lb_rs 的参数 bid 是什么？",
      "add_lb_rs 里 bid 要不要传？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:add_lb_rs]/[Constraint:1]\nlb_id: LoadBalancer的唯一标识",
    "paraphrases": [
      "接口 add_lb_rs 存在约束：lb_id: LoadBalancer的唯一标识",
      "调用 add_lb_rs 时需要注意：lb_id: LoadBalancer的唯一标识"
    ],
    "recall_prompts": [
      "add_lb_rs 有哪些限制或条件？",
      "调用 add_lb_rs 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:add_lb_rs]/[Constraint:2]\nrs_list: RealServer结构JSON列表，最多20个",
    "paraphrases": [
      "接口 add_lb_rs 存在约束：rs_list: RealServer结构JSON列表，最多20个",
      "调用 add_lb_rs 时需要注意：rs_list: RealServer结构JSON列表，最多20个"
    ],
    "recall_prompts": [
      "add_lb_rs 有哪些限制或条件？",
      "调用 add_lb_rs 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:add_lb_rs]/[Constraint:3]\nadd_lb_rs为LoadBalancer添加一组RS。RS不能指定port（所有RS共用监听的后端port）。添加到LB后端的RS会自动关联到该LB实例的每个监听上。已存在的RS会被忽略不报错。不允许rs本身是lb类型，不允许lb挂lb。",
    "paraphrases": [
      "接口 add_lb_rs 存在约束：add_lb_rs为LoadBalancer添加一组RS。RS不能指定port（所有RS共用监听的后端port）。添加到LB后端的RS会自动关联到该LB实例的每个监听上。已存在的RS会被忽略不报错。不允许rs本身是lb类型，不允许lb挂lb。",
      "调用 add_lb_rs 时需要注意：add_lb_rs为LoadBalancer添加一组RS。RS不能指定port（所有RS共用监听的后端port）。添加到LB后端的RS会自动关联到该LB实例的每个监听上。已存在的RS会被忽略不报错。不允许rs本身是lb类型，不允许lb挂lb。"
    ],
    "recall_prompts": [
      "add_lb_rs 有哪些限制或条件？",
      "调用 add_lb_rs 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:add_lb_rs]/[Constraint:4]\n说明: add_lb_rs为LoadBalancer添加一组RS。RS不能指定port（所有RS共用监听的后端port）。添加到LB后端的RS会自动关联到该LB实例的每个监听上。已存在的RS会被忽略不报错。不允许rs本身是lb类型，不允许lb挂lb。",
    "paraphrases": [
      "接口 add_lb_rs 存在约束：说明: add_lb_rs为LoadBalancer添加一组RS。RS不能指定port（所有RS共用监听的后端port）。添加到LB后端的RS会自动关联到该LB实例的每个监听上。已存在的RS会被忽略不报错。不允许rs本身是lb类型，不允许lb挂lb。",
      "调用 add_lb_rs 时需要注意：说明: add_lb_rs为LoadBalancer添加一组RS。RS不能指定port（所有RS共用监听的后端port）。添加到LB后端的RS会自动关联到该LB实例的每个监听上。已存在的RS会被忽略不报错。不允许rs本身是lb类型，不允许lb挂lb。"
    ],
    "recall_prompts": [
      "add_lb_rs 有哪些限制或条件？",
      "调用 add_lb_rs 时要注意什么？"
    ]
  },
  {
    "type": "command_mapping",
    "content": "## config_lb_rs - add_lb_rs + config_lb_rs + delete_lb_rs + switch_lb_rs - LB实例级别RS操作\n[Action:config_lb_rs]/[Overview]\n接口 config_lb_rs 用于 add_lb_rs + config_lb_rs + delete_lb_rs + switch_lb_rs - LB实例级别RS操作。\n必选参数: 无。\nconfig_lb_rs对LB中的一组RS的weight值进行配置。",
    "paraphrases": [
      "config_lb_rs 的核心触发锚点是 action=config_lb_rs。"
    ],
    "recall_prompts": [
      "config_lb_rs 是做什么的？",
      "什么时候应该调用 config_lb_rs？"
    ]
  },
  {
    "type": "command_mapping",
    "content": "## delete_lb_rs - add_lb_rs + config_lb_rs + delete_lb_rs + switch_lb_rs - LB实例级别RS操作\n[Action:delete_lb_rs]/[Overview]\n接口 delete_lb_rs 用于 add_lb_rs + config_lb_rs + delete_lb_rs + switch_lb_rs - LB实例级别RS操作。\n必选参数: 无。\ndelete_lb_rs从LB中删除一组RS。不存在的RS会被忽略不报错。\n说明: delete_lb_rs从LB中删除一组RS。不存在的RS会被忽略不报错。",
    "paraphrases": [
      "delete_lb_rs 的核心触发锚点是 action=delete_lb_rs。"
    ],
    "recall_prompts": [
      "delete_lb_rs 是做什么的？",
      "什么时候应该调用 delete_lb_rs？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:delete_lb_rs]/[Constraint:1]\ndelete_lb_rs从LB中删除一组RS。不存在的RS会被忽略不报错。",
    "paraphrases": [
      "接口 delete_lb_rs 存在约束：delete_lb_rs从LB中删除一组RS。不存在的RS会被忽略不报错。",
      "调用 delete_lb_rs 时需要注意：delete_lb_rs从LB中删除一组RS。不存在的RS会被忽略不报错。"
    ],
    "recall_prompts": [
      "delete_lb_rs 有哪些限制或条件？",
      "调用 delete_lb_rs 时要注意什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:delete_lb_rs]/[Constraint:2]\n说明: delete_lb_rs从LB中删除一组RS。不存在的RS会被忽略不报错。",
    "paraphrases": [
      "接口 delete_lb_rs 存在约束：说明: delete_lb_rs从LB中删除一组RS。不存在的RS会被忽略不报错。",
      "调用 delete_lb_rs 时需要注意：说明: delete_lb_rs从LB中删除一组RS。不存在的RS会被忽略不报错。"
    ],
    "recall_prompts": [
      "delete_lb_rs 有哪些限制或条件？",
      "调用 delete_lb_rs 时要注意什么？"
    ]
  },
  {
    "type": "command_mapping",
    "content": "## switch_lb_rs - add_lb_rs + config_lb_rs + delete_lb_rs + switch_lb_rs - LB实例级别RS操作\n[Action:switch_lb_rs]/[Overview]\n接口 switch_lb_rs 用于 add_lb_rs + config_lb_rs + delete_lb_rs + switch_lb_rs - LB实例级别RS操作。\n必选参数: 无。\nswitch_lb_rs从LB中将一组RS切换成另一组RS，返回更新后的RS列表。\nLB实例级别RS操作有四个接口：add_lb_rs、config_lb_rs、delete_lb_rs、switch_lb_rs。与RSPool级别操作的核心区别是：LB级别RS不能指定port（共用监听后端port），添加到LB后端的RS自动关联到所有监听。不允许LB串联。",
    "paraphrases": [
      "switch_lb_rs 的核心触发锚点是 action=switch_lb_rs。"
    ],
    "recall_prompts": [
      "switch_lb_rs 是做什么的？",
      "什么时候应该调用 switch_lb_rs？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Action:switch_lb_rs]/[Constraint:2]\nLB实例级别RS操作有四个接口：add_lb_rs、config_lb_rs、delete_lb_rs、switch_lb_rs。与RSPool级别操作的核心区别是：LB级别RS不能指定port（共用监听后端port），添加到LB后端的RS自动关联到所有监听。不允许LB串联。",
    "paraphrases": [
      "接口 switch_lb_rs 存在约束：LB实例级别RS操作有四个接口：add_lb_rs、config_lb_rs、delete_lb_rs、switch_lb_rs。与RSPool级别操作的核心区别是：LB级别RS不能指定port（共用监听后端port），添加到LB后端的RS自动关联到所有监听。不允许LB串联。",
      "调用 switch_lb_rs 时需要注意：LB实例级别RS操作有四个接口：add_lb_rs、config_lb_rs、delete_lb_rs、switch_lb_rs。与RSPool级别操作的核心区别是：LB级别RS不能指定port（共用监听后端port），添加到LB后端的RS自动关联到所有监听。不允许LB串联。"
    ],
    "recall_prompts": [
      "switch_lb_rs 有哪些限制或条件？",
      "调用 switch_lb_rs 时要注意什么？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:switch_lb_rs]/[Error:-2103]\n当调用 switch_lb_rs 返回错误码 -2103 APINotSupportForThisTypeOfLb 时，表示 API不支持该类型LB。",
    "paraphrases": [
      "switch_lb_rs 出现 -2103 时，对应错误名是 APINotSupportForThisTypeOfLb。",
      "错误 -2103 在 switch_lb_rs 中表示：API不支持该类型LB。"
    ],
    "recall_prompts": [
      "switch_lb_rs 返回 -2103 代表什么？",
      "APINotSupportForThisTypeOfLb 是什么错误？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:switch_lb_rs]/[Error:-2601]\n当调用 switch_lb_rs 返回错误码 -2601 LbIdIsEmpty 时，表示 LB ID为空。",
    "paraphrases": [
      "switch_lb_rs 出现 -2601 时，对应错误名是 LbIdIsEmpty。",
      "错误 -2601 在 switch_lb_rs 中表示：LB ID为空。"
    ],
    "recall_prompts": [
      "switch_lb_rs 返回 -2601 代表什么？",
      "LbIdIsEmpty 是什么错误？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:switch_lb_rs]/[Error:-2610]\n当调用 switch_lb_rs 返回错误码 -2610 LbIdNotExist 时，表示 LB ID不存在。",
    "paraphrases": [
      "switch_lb_rs 出现 -2610 时，对应错误名是 LbIdNotExist。",
      "错误 -2610 在 switch_lb_rs 中表示：LB ID不存在。"
    ],
    "recall_prompts": [
      "switch_lb_rs 返回 -2610 代表什么？",
      "LbIdNotExist 是什么错误？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:switch_lb_rs]/[Error:-2800]\n当调用 switch_lb_rs 返回错误码 -2800 VmTooMany 时，表示 一次传入的VM过多。",
    "paraphrases": [
      "switch_lb_rs 出现 -2800 时，对应错误名是 VmTooMany。",
      "错误 -2800 在 switch_lb_rs 中表示：一次传入的VM过多。"
    ],
    "recall_prompts": [
      "switch_lb_rs 返回 -2800 代表什么？",
      "VmTooMany 是什么错误？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:switch_lb_rs]/[Error:-2804]\n当调用 switch_lb_rs 返回错误码 -2804 VmWeightNotSupport 时，表示 VM权重不合法。",
    "paraphrases": [
      "switch_lb_rs 出现 -2804 时，对应错误名是 VmWeightNotSupport。",
      "错误 -2804 在 switch_lb_rs 中表示：VM权重不合法。"
    ],
    "recall_prompts": [
      "switch_lb_rs 返回 -2804 代表什么？",
      "VmWeightNotSupport 是什么错误？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Action:switch_lb_rs]/[Error:-2807]\n当调用 switch_lb_rs 返回错误码 -2807 VmTypeNotMatch 时，表示 VM类型不匹配。",
    "paraphrases": [
      "switch_lb_rs 出现 -2807 时，对应错误名是 VmTypeNotMatch。",
      "错误 -2807 在 switch_lb_rs 中表示：VM类型不匹配。"
    ],
    "recall_prompts": [
      "switch_lb_rs 返回 -2807 代表什么？",
      "VmTypeNotMatch 是什么错误？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:Listener]/[Overview]\nListener 结构包含字段: protocol, frontend_port, backend_port, rs_pool_name, bandwidth, status, config",
    "paraphrases": [
      "Listener 的关键字段有 protocol, frontend_port, backend_port, rs_pool_name, bandwidth, status, config。"
    ],
    "recall_prompts": [
      "Listener 结构有哪些字段？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:Listener]/[Field:protocol]\n字段 protocol 在 Listener 中的含义是：string，支持tcp/http/https/udp/tcps",
    "paraphrases": [
      "Listener 里的 protocol: string，支持tcp/http/https/udp/tcps"
    ],
    "recall_prompts": [
      "Listener 的 protocol 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:Listener]/[Field:frontend_port]\n字段 frontend_port 在 Listener 中的含义是：int，前端端口1-65535",
    "paraphrases": [
      "Listener 里的 frontend_port: int，前端端口1-65535"
    ],
    "recall_prompts": [
      "Listener 的 frontend_port 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:Listener]/[Field:backend_port]\n字段 backend_port 在 Listener 中的含义是：int，后端端口1-65535",
    "paraphrases": [
      "Listener 里的 backend_port: int，后端端口1-65535"
    ],
    "recall_prompts": [
      "Listener 的 backend_port 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:Listener]/[Field:rs_pool_name]\n字段 rs_pool_name 在 Listener 中的含义是：string，backend_port和rs_pool_name必传其一，都传以rs_pool_name为准",
    "paraphrases": [
      "Listener 里的 rs_pool_name: string，backend_port和rs_pool_name必传其一，都传以rs_pool_name为准"
    ],
    "recall_prompts": [
      "Listener 的 rs_pool_name 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:Listener]/[Field:bandwidth]\n字段 bandwidth 在 Listener 中的含义是：string，出流量带宽上限，支持k/m/g后缀，最大4GBytes/s",
    "paraphrases": [
      "Listener 里的 bandwidth: string，出流量带宽上限，支持k/m/g后缀，最大4GBytes/s"
    ],
    "recall_prompts": [
      "Listener 的 bandwidth 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:Listener]/[Field:status]\n字段 status 在 Listener 中的含义是：string，active/inactive，默认inactive",
    "paraphrases": [
      "Listener 里的 status: string，active/inactive，默认inactive"
    ],
    "recall_prompts": [
      "Listener 的 status 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:Listener]/[Field:config]\n字段 config 在 Listener 中的含义是：struct，TcpConfig或UdpConfig的JSON字符串",
    "paraphrases": [
      "Listener 里的 config: struct，TcpConfig或UdpConfig的JSON字符串"
    ],
    "recall_prompts": [
      "Listener 的 config 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:TcpConfig]/[Overview]\nTcpConfig 结构包含字段: scheduler, check, syn_proxy, est_timeout, connection_drain, connection_drain_timeout, proxy_protocol_v2_enabled, session_resched",
    "paraphrases": [
      "TcpConfig 的关键字段有 scheduler, check, syn_proxy, est_timeout, connection_drain, connection_drain_timeout, proxy_protocol_v2_enabled, session_resched。"
    ],
    "recall_prompts": [
      "TcpConfig 结构有哪些字段？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:TcpConfig]/[Field:scheduler]\n字段 scheduler 在 TcpConfig 中的含义是：string，wrr/wlc/rr，默认wrr",
    "paraphrases": [
      "TcpConfig 里的 scheduler: string，wrr/wlc/rr，默认wrr"
    ],
    "recall_prompts": [
      "TcpConfig 的 scheduler 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:TcpConfig]/[Field:check]\n字段 check 在 TcpConfig 中的含义是：string，TcpCheck或HttpCheck结构JSON，默认关闭",
    "paraphrases": [
      "TcpConfig 里的 check: string，TcpCheck或HttpCheck结构JSON，默认关闭"
    ],
    "recall_prompts": [
      "TcpConfig 的 check 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:TcpConfig]/[Field:syn_proxy]\n字段 syn_proxy 在 TcpConfig 中的含义是：string，ENABLE/DISABLE",
    "paraphrases": [
      "TcpConfig 里的 syn_proxy: string，ENABLE/DISABLE"
    ],
    "recall_prompts": [
      "TcpConfig 的 syn_proxy 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:TcpConfig]/[Field:est_timeout]\n字段 est_timeout 在 TcpConfig 中的含义是：int，TCP超时时间10-1200秒，默认900",
    "paraphrases": [
      "TcpConfig 里的 est_timeout: int，TCP超时时间10-1200秒，默认900"
    ],
    "recall_prompts": [
      "TcpConfig 的 est_timeout 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:TcpConfig]/[Field:connection_drain]\n字段 connection_drain 在 TcpConfig 中的含义是：string，on/off，连接优雅中断，默认off",
    "paraphrases": [
      "TcpConfig 里的 connection_drain: string，on/off，连接优雅中断，默认off"
    ],
    "recall_prompts": [
      "TcpConfig 的 connection_drain 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:TcpConfig]/[Field:connection_drain_timeout]\n字段 connection_drain_timeout 在 TcpConfig 中的含义是：int，优雅中断超时10-900秒",
    "paraphrases": [
      "TcpConfig 里的 connection_drain_timeout: int，优雅中断超时10-900秒"
    ],
    "recall_prompts": [
      "TcpConfig 的 connection_drain_timeout 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:TcpConfig]/[Field:proxy_protocol_v2_enabled]\n字段 proxy_protocol_v2_enabled 在 TcpConfig 中的含义是：string，on/off",
    "paraphrases": [
      "TcpConfig 里的 proxy_protocol_v2_enabled: string，on/off"
    ],
    "recall_prompts": [
      "TcpConfig 的 proxy_protocol_v2_enabled 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:TcpConfig]/[Field:session_resched]\n字段 session_resched 在 TcpConfig 中的含义是：string，on/off，链接重调度，默认off",
    "paraphrases": [
      "TcpConfig 里的 session_resched: string，on/off，链接重调度，默认off"
    ],
    "recall_prompts": [
      "TcpConfig 的 session_resched 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:TcpCheck]/[Overview]\nTcpCheck 结构包含字段: type, timeout, port, interval, up, down",
    "paraphrases": [
      "TcpCheck 的关键字段有 type, timeout, port, interval, up, down。"
    ],
    "recall_prompts": [
      "TcpCheck 结构有哪些字段？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:TcpCheck]/[Field:type]\n字段 type 在 TcpCheck 中的含义是：string，设为tcp",
    "paraphrases": [
      "TcpCheck 里的 type: string，设为tcp"
    ],
    "recall_prompts": [
      "TcpCheck 的 type 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:TcpCheck]/[Field:timeout]\n字段 timeout 在 TcpCheck 中的含义是：int，连接超时1-1000秒，默认5",
    "paraphrases": [
      "TcpCheck 里的 timeout: int，连接超时1-1000秒，默认5"
    ],
    "recall_prompts": [
      "TcpCheck 的 timeout 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:TcpCheck]/[Field:port]\n字段 port 在 TcpCheck 中的含义是：int，健康检查端口1-65535，默认使用后端服务端口",
    "paraphrases": [
      "TcpCheck 里的 port: int，健康检查端口1-65535，默认使用后端服务端口"
    ],
    "recall_prompts": [
      "TcpCheck 的 port 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:TcpCheck]/[Field:interval]\n字段 interval 在 TcpCheck 中的含义是：int，检查间隔1-1000秒，默认2",
    "paraphrases": [
      "TcpCheck 里的 interval: int，检查间隔1-1000秒，默认2"
    ],
    "recall_prompts": [
      "TcpCheck 的 interval 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:TcpCheck]/[Field:up]\n字段 up 在 TcpCheck 中的含义是：int，fail到success的连续成功次数1-1000，默认3",
    "paraphrases": [
      "TcpCheck 里的 up: int，fail到success的连续成功次数1-1000，默认3"
    ],
    "recall_prompts": [
      "TcpCheck 的 up 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:TcpCheck]/[Field:down]\n字段 down 在 TcpCheck 中的含义是：int，success到fail的连续失败次数1-1000，默认3",
    "paraphrases": [
      "TcpCheck 里的 down: int，success到fail的连续失败次数1-1000，默认3"
    ],
    "recall_prompts": [
      "TcpCheck 的 down 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:HttpCheck]/[Overview]\nHttpCheck 结构包含字段: type, domain, port, uri, up, down, timeout, interval, http_status_code",
    "paraphrases": [
      "HttpCheck 的关键字段有 type, domain, port, uri, up, down, timeout, interval, http_status_code。"
    ],
    "recall_prompts": [
      "HttpCheck 结构有哪些字段？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:HttpCheck]/[Field:type]\n字段 type 在 HttpCheck 中的含义是：string，设为http或ssl（tcp监听支持ssl类型健康检查）",
    "paraphrases": [
      "HttpCheck 里的 type: string，设为http或ssl（tcp监听支持ssl类型健康检查）"
    ],
    "recall_prompts": [
      "HttpCheck 的 type 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:HttpCheck]/[Field:domain]\n字段 domain 在 HttpCheck 中的含义是：string，健康检查域名",
    "paraphrases": [
      "HttpCheck 里的 domain: string，健康检查域名"
    ],
    "recall_prompts": [
      "HttpCheck 的 domain 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:HttpCheck]/[Field:port]\n字段 port 在 HttpCheck 中的含义是：int，健康检查端口",
    "paraphrases": [
      "HttpCheck 里的 port: int，健康检查端口"
    ],
    "recall_prompts": [
      "HttpCheck 的 port 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:HttpCheck]/[Field:uri]\n字段 uri 在 HttpCheck 中的含义是：string，健康检查URI",
    "paraphrases": [
      "HttpCheck 里的 uri: string，健康检查URI"
    ],
    "recall_prompts": [
      "HttpCheck 的 uri 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:HttpCheck]/[Field:up]\n字段 up 在 HttpCheck 中的含义是：int，默认3",
    "paraphrases": [
      "HttpCheck 里的 up: int，默认3"
    ],
    "recall_prompts": [
      "HttpCheck 的 up 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:HttpCheck]/[Field:down]\n字段 down 在 HttpCheck 中的含义是：int，默认3",
    "paraphrases": [
      "HttpCheck 里的 down: int，默认3"
    ],
    "recall_prompts": [
      "HttpCheck 的 down 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:HttpCheck]/[Field:timeout]\n字段 timeout 在 HttpCheck 中的含义是：int，默认5",
    "paraphrases": [
      "HttpCheck 里的 timeout: int，默认5"
    ],
    "recall_prompts": [
      "HttpCheck 的 timeout 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:HttpCheck]/[Field:interval]\n字段 interval 在 HttpCheck 中的含义是：int，默认2",
    "paraphrases": [
      "HttpCheck 里的 interval: int，默认2"
    ],
    "recall_prompts": [
      "HttpCheck 的 interval 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:HttpCheck]/[Field:http_status_code]\n字段 http_status_code 在 HttpCheck 中的含义是：string，如http_2xx,http_3xx，默认http_2xx,http_3xx",
    "paraphrases": [
      "HttpCheck 里的 http_status_code: string，如http_2xx,http_3xx，默认http_2xx,http_3xx"
    ],
    "recall_prompts": [
      "HttpCheck 的 http_status_code 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:UdpConfig]/[Overview]\nUdpConfig 结构包含字段: scheduler, persistence_timeout, check",
    "paraphrases": [
      "UdpConfig 的关键字段有 scheduler, persistence_timeout, check。"
    ],
    "recall_prompts": [
      "UdpConfig 结构有哪些字段？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:UdpConfig]/[Field:scheduler]\n字段 scheduler 在 UdpConfig 中的含义是：string，wrr/wlc/rr，默认wrr",
    "paraphrases": [
      "UdpConfig 里的 scheduler: string，wrr/wlc/rr，默认wrr"
    ],
    "recall_prompts": [
      "UdpConfig 的 scheduler 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:UdpConfig]/[Field:persistence_timeout]\n字段 persistence_timeout 在 UdpConfig 中的含义是：int，0-86400秒，0表示关闭，默认0",
    "paraphrases": [
      "UdpConfig 里的 persistence_timeout: int，0-86400秒，0表示关闭，默认0"
    ],
    "recall_prompts": [
      "UdpConfig 的 persistence_timeout 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:UdpConfig]/[Field:check]\n字段 check 在 UdpConfig 中的含义是：string，UdpCheck结构JSON",
    "paraphrases": [
      "UdpConfig 里的 check: string，UdpCheck结构JSON"
    ],
    "recall_prompts": [
      "UdpConfig 的 check 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:UdpCheck]/[Overview]\nUdpCheck 结构包含字段: type, timeout, port, interval, up, down",
    "paraphrases": [
      "UdpCheck 的关键字段有 type, timeout, port, interval, up, down。"
    ],
    "recall_prompts": [
      "UdpCheck 结构有哪些字段？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:UdpCheck]/[Field:type]\n字段 type 在 UdpCheck 中的含义是：string，设为udp",
    "paraphrases": [
      "UdpCheck 里的 type: string，设为udp"
    ],
    "recall_prompts": [
      "UdpCheck 的 type 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:UdpCheck]/[Field:timeout]\n字段 timeout 在 UdpCheck 中的含义是：int，默认2",
    "paraphrases": [
      "UdpCheck 里的 timeout: int，默认2"
    ],
    "recall_prompts": [
      "UdpCheck 的 timeout 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:UdpCheck]/[Field:port]\n字段 port 在 UdpCheck 中的含义是：int",
    "paraphrases": [
      "UdpCheck 里的 port: int"
    ],
    "recall_prompts": [
      "UdpCheck 的 port 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:UdpCheck]/[Field:interval]\n字段 interval 在 UdpCheck 中的含义是：int，默认2",
    "paraphrases": [
      "UdpCheck 里的 interval: int，默认2"
    ],
    "recall_prompts": [
      "UdpCheck 的 interval 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:UdpCheck]/[Field:up]\n字段 up 在 UdpCheck 中的含义是：int，默认3",
    "paraphrases": [
      "UdpCheck 里的 up: int，默认3"
    ],
    "recall_prompts": [
      "UdpCheck 的 up 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:UdpCheck]/[Field:down]\n字段 down 在 UdpCheck 中的含义是：int，默认3",
    "paraphrases": [
      "UdpCheck 里的 down: int，默认3"
    ],
    "recall_prompts": [
      "UdpCheck 的 down 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:RealServer]/[Overview]\nRealServer 结构包含字段: rs_ip, weight, port, rs_type, proxy_protocol",
    "paraphrases": [
      "RealServer 的关键字段有 rs_ip, weight, port, rs_type, proxy_protocol。"
    ],
    "recall_prompts": [
      "RealServer 结构有哪些字段？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:RealServer]/[Field:rs_ip]\n字段 rs_ip 在 RealServer 中的含义是：string，Real Server的IP",
    "paraphrases": [
      "RealServer 里的 rs_ip: string，Real Server的IP"
    ],
    "recall_prompts": [
      "RealServer 的 rs_ip 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:RealServer]/[Field:weight]\n字段 weight 在 RealServer 中的含义是：int，权重0-1000，默认100",
    "paraphrases": [
      "RealServer 里的 weight: int，权重0-1000，默认100"
    ],
    "recall_prompts": [
      "RealServer 的 weight 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:RealServer]/[Field:port]\n字段 port 在 RealServer 中的含义是：int，add_rs类接口必须指定port，add_lb_rs类接口不指定port",
    "paraphrases": [
      "RealServer 里的 port: int，add_rs类接口必须指定port，add_lb_rs类接口不指定port"
    ],
    "recall_prompts": [
      "RealServer 的 port 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:RealServer]/[Field:rs_type]\n字段 rs_type 在 RealServer 中的含义是：string，classic（经典网络）",
    "paraphrases": [
      "RealServer 里的 rs_type: string，classic（经典网络）"
    ],
    "recall_prompts": [
      "RealServer 的 rs_type 是什么？"
    ]
  },
  {
    "type": "structured_config",
    "content": "[Struct:RealServer]/[Field:proxy_protocol]\n字段 proxy_protocol 在 RealServer 中的含义是：string，on/off",
    "paraphrases": [
      "RealServer 里的 proxy_protocol: string，on/off"
    ],
    "recall_prompts": [
      "RealServer 的 proxy_protocol 是什么？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Flow:slb_online]/[Overview]\n步骤1: 创建LoadBalancer实例\n步骤2: 准备后端服务器（两种方式二选一）\n步骤3: 创建VIP监听\n步骤4: 激活VIP\n步骤5: 验证健康检查",
    "paraphrases": [
      "SLB 上线流程通常包括创建 LB、准备后端、创建 VIP、激活 VIP、验证健康检查。",
      "SLB 的完整操作链可以拆成实例创建、后端准备、监听创建、激活、验证五步。"
    ],
    "recall_prompts": [
      "SLB 从创建到上线的大致流程是什么？",
      "如何把一个新的 SLB 配到可服务状态？"
    ]
  },
  {
    "type": "procedure",
    "content": "[Compare:add_rs|add_lb_rs]\nRSPool方式（add_rs）: 每个RS可指定不同port，通过rs_pool_name关联VIP，适合复杂场景\nLB直接挂载方式（add_lb_rs）: RS共用监听后端port，自动关联到LB所有监听，适合简单场景\n注意: 同一rspool或lb后端的RS的tunnel_id必须相同，不允许LB串联\n完整的SLB上线流程：首先create_loadbalancer获取lb_id和eip，然后选择RSPool方式（create_rs_pool + add_rs）或LB直接挂载方式（add_lb_rs）准备后端服务器，接着create_vip创建监听指定protocol和frontend_port，最后config_vip设active激活VIP开始服务。VIP激活后从stopped变为starting再变为running。",
    "paraphrases": [
      "add_rs 更适合需要每个 RS 使用不同 port 的复杂场景。",
      "add_lb_rs 更适合后端端口统一、希望自动挂到所有监听的简单场景。"
    ],
    "recall_prompts": [
      "add_rs 和 add_lb_rs 有什么区别？",
      "什么时候用 RSPool，什么时候直接给 LB 挂 RS？"
    ]
  }
]