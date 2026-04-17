#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Engram-friendly JSONL samples from a structured SLB API document.

Input:
  Plain UTF-8 text with sections like:
    ## create_loadbalancer - 创建SLB负载均衡实例
    接口路径: action=create_loadbalancer
    必选参数:
    - region_no: string，...
    可选参数:
    - eip_type: string，internet或intranet，默认internet
    常见错误码:
    - -2619 RegionIdIsEmpty: Region ID为空

Output JSONL schema:
  {
    "id": "slb-create_loadbalancer-overview",
    "type": "command_mapping|fact_pair|structured_config|procedure",
    "action": "create_loadbalancer",
    "title": "...",
    "anchor": "[Action:create_loadbalancer]/[Overview]",
    "tags": ["slb", "api", "overview"],
    "content": "Engram training block text",
    "paraphrases": ["..."],
    "recall_prompts": ["..."],
    "must_contain": ["create_loadbalancer", "region_no", "aliyun_idkp"],
    "metadata": {...}
  }

This schema is a superset of the current autoresearch/knowledge_format.py expectations:
`content`, `paraphrases`, and `recall_prompts` can be consumed directly by the
existing build_training_text() pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SECTION_RE = re.compile(r"(?m)^##\s+(.+?)\s*$")
BULLET_RE = re.compile(r"^-\s+(.*)$", re.M)
ACTION_IN_LINE_RE = re.compile(r"action=([a-zA-Z0-9_]+)")
ERROR_RE = re.compile(r"^-\s+(-?\d+)\s+([A-Za-z0-9_]+):\s*(.+)$")
PARAM_RE = re.compile(r"^-\s+([a-zA-Z0-9_]+):\s*([^，,:]+)[，,:]?\s*(.*)$")
KEYWORDS_CONSTRAINT = (
    "必传", "必须", "默认", "至少", "最多", "上限", "不允许", "唯一", "仅支持", "二选一",
    "都传以", "若都传", "同一", "可选", "事务性", "忽略不报错", "高危接口",
)


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "-", text).strip("-").lower()


def clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


@dataclass
class Param:
    name: str
    ptype: str
    desc: str
    required: bool

    @property
    def default(self) -> Optional[str]:
        m = re.search(r"默认\s*([a-zA-Z0-9_\-/]+)", self.desc)
        return m.group(1) if m else None

    @property
    def enum_values(self) -> List[str]:
        vals = []
        # crude but useful for internet/intranet, active/inactive, etc.
        for pat in [r"支持([a-zA-Z0-9_/、,]+)", r"([a-zA-Z0-9_]+(?:/[a-zA-Z0-9_]+)+)"]:
            for m in re.finditer(pat, self.desc):
                raw = m.group(1)
                pieces = [p.strip() for p in re.split(r"[/、,]", raw) if p.strip()]
                vals.extend(pieces)
        uniq = []
        for v in vals:
            if v not in uniq:
                uniq.append(v)
        return uniq


@dataclass
class ApiAction:
    action: str
    section_title: str
    required_params: List[Param] = field(default_factory=list)
    optional_params: List[Param] = field(default_factory=list)
    errors: List[Tuple[str, str, str]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    success_return: Optional[str] = None
    listener_structure: List[Param] = field(default_factory=list)

    def all_params(self) -> List[Param]:
        return self.required_params + self.optional_params


# ---------- parsing ----------

def split_sections(text: str) -> List[Tuple[str, str]]:
    matches = list(SECTION_RE.finditer(text))
    sections = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = clean_line(m.group(1))
        body = text[start:end].strip()
        sections.append((title, body))
    return sections


def extract_bullet_block(body: str, label: str) -> List[str]:
    # find from label to next recognized label or blank double line + keyword
    pattern = re.compile(rf"(?ms)^{re.escape(label)}:\s*\n(.*?)(?=^(?:必选参数|可选参数|常见错误码|成功返回|说明|注意|接口路径|Listener结构说明|Listener结构|TcpConfig结构|TcpCheck结构|HttpCheck结构|UdpConfig结构|UdpCheck结构|RealServer结构|SLB实例从创建到上线的完整流程|两种后端RS管理方式对比|VIP状态流转说明):|\Z)")
    m = pattern.search(body)
    if not m:
        return []
    block = m.group(1)
    return [clean_line(x) for x in BULLET_RE.findall(block)]


def extract_single_line_field(body: str, label: str) -> Optional[str]:
    m = re.search(rf"(?m)^{re.escape(label)}:\s*(.+)$", body)
    return clean_line(m.group(1)) if m else None


def extract_prefixed_lines(body: str, prefix: str) -> List[str]:
    return [clean_line(m.group(1)) for m in re.finditer(rf"(?m)^{re.escape(prefix)}:\s*(.+)$", body)]


def parse_params(bullets: List[str], required: bool) -> List[Param]:
    out = []
    for item in bullets:
        m = PARAM_RE.match("- " + item)
        if not m:
            # compact inline form like "region_no, lb_id, ..."
            for name in [x.strip() for x in re.split(r"[,，]", item) if x.strip()]:
                out.append(Param(name=name, ptype="string", desc="", required=required))
            continue
        out.append(Param(name=m.group(1), ptype=m.group(2).strip(), desc=clean_line(m.group(3)), required=required))
    return out


def parse_errors(bullets: List[str]) -> List[Tuple[str, str, str]]:
    out = []
    for item in bullets:
        m = ERROR_RE.match("- " + item)
        if m:
            out.append((m.group(1), m.group(2), clean_line(m.group(3))))
    return out


def split_actions_from_body(section_title: str, body: str) -> List[ApiAction]:
    # Break a combined section into per-action sub-blocks by “当需要...调用xxx接口” or action path lines.
    action_matches = list(re.finditer(r"(?m)^当需要.*?调用([a-zA-Z0-9_]+)接口。?$", body))
    if not action_matches:
        # fallback: derive from interface path line(s)
        path_actions = ACTION_IN_LINE_RE.findall(body)
        if path_actions:
            action_matches = list(re.finditer(r"(?m)^接口路径:\s*action=([a-zA-Z0-9_]+).*$", body))

    if not action_matches:
        return []

    items: List[ApiAction] = []
    for i, m in enumerate(action_matches):
        action = m.group(1)
        start = m.start()
        end = action_matches[i + 1].start() if i + 1 < len(action_matches) else len(body)
        chunk = body[start:end].strip()
        required = parse_params(extract_bullet_block(chunk, "必选参数"), required=True)
        optional = parse_params(extract_bullet_block(chunk, "可选参数"), required=False)
        errors = parse_errors(extract_bullet_block(chunk, "常见错误码"))
        notes = []
        for label in ["说明", "注意"]:
            val = extract_single_line_field(chunk, label)
            if val:
                notes.append(val)
        success = extract_single_line_field(chunk, "成功返回")
        listener = parse_params(extract_bullet_block(chunk, "Listener结构说明"), required=False)
        items.append(ApiAction(
            action=action,
            section_title=section_title,
            required_params=required,
            optional_params=optional,
            errors=errors,
            notes=notes + extract_freeform_constraint_lines(chunk),
            success_return=success,
            listener_structure=listener,
        ))
    return items


def extract_freeform_constraint_lines(chunk: str) -> List[str]:
    lines = [clean_line(x) for x in chunk.splitlines() if clean_line(x)]
    keep = []
    for line in lines:
        if line.startswith(("当需要", "接口路径:", "必选参数:", "可选参数:", "常见错误码:", "成功返回:", "Listener结构说明:")):
            continue
        if line.startswith("- "):
            continue
        if any(k in line for k in KEYWORDS_CONSTRAINT):
            keep.append(line)
    # dedupe while preserving order
    uniq = []
    for line in keep:
        if line not in uniq:
            uniq.append(line)
    return uniq


# ---------- sample generation ----------

def make_record(
    rec_id: str,
    rec_type: str,
    action: str,
    title: str,
    anchor: str,
    content: str,
    paraphrases: List[str],
    recall_prompts: List[str],
    must_contain: List[str],
    tags: List[str],
    metadata: Dict,
) -> Dict:
    return {
        "id": rec_id,
        "type": rec_type,
        "action": action,
        "title": title,
        "anchor": anchor,
        "tags": tags,
        "content": content.strip(),
        "paraphrases": [p for p in paraphrases if p],
        "recall_prompts": [p for p in recall_prompts if p],
        "must_contain": [m for m in must_contain if m],
        "metadata": metadata,
    }


def overview_record(api: ApiAction) -> Dict:
    req_names = [p.name for p in api.required_params]
    pretty_title = api.section_title
    if pretty_title.lower().startswith(api.action.lower() + " - "):
        pretty_title = pretty_title[len(api.action) + 3:]
    content = (
        f"## {api.action} - {pretty_title}\n"
        f"[Action:{api.action}]/[Overview]\n"
        f"接口 {api.action} 用于 {api.section_title}。\n"
        f"必选参数: {', '.join(req_names) if req_names else '无'}。\n"
        + (f"成功返回: {api.success_return}\n" if api.success_return else "")
        + ("\n".join(api.notes[:3]) if api.notes else "")
    )
    paraphrases = [
        f"当需要执行 {api.action} 对应操作时，优先检查必选参数 {'、'.join(req_names)}。" if req_names else "",
        f"{api.action} 的核心触发锚点是 action={api.action}。",
    ]
    recall = [
        f"{api.action} 是做什么的？",
        f"什么时候应该调用 {api.action}？",
    ]
    must = [api.action] + req_names[:3]
    return make_record(
        rec_id=f"slb-{api.action}-overview",
        rec_type="command_mapping",
        action=api.action,
        title=f"{api.action} 概览",
        anchor=f"[Action:{api.action}]/[Overview]",
        content=content,
        paraphrases=paraphrases,
        recall_prompts=recall,
        must_contain=must,
        tags=["slb", "api", "overview"],
        metadata={"sample_kind": "overview", "required_params": req_names},
    )


def param_records(api: ApiAction) -> List[Dict]:
    records = []
    for p in api.all_params():
        content = (
            f"[Action:{api.action}]/[Field:{p.name}]\n"
            f"参数 {p.name} 属于接口 {api.action}，类型 {p.ptype or 'string'}。"
            f"{'必选参数。' if p.required else '可选参数。'}"
            f" {p.desc}".strip()
        )
        paraphrases = [
            f"{api.action} 的参数 {p.name} {'必须提供' if p.required else '可以不传'}。",
            f"查询 {api.action} 时，字段 {p.name} 的含义是：{p.desc}" if p.desc else "",
        ]
        recall = [
            f"{api.action} 的参数 {p.name} 是什么？",
            f"{api.action} 里 {p.name} 要不要传？",
        ]
        must = [api.action, p.name] + ([p.default] if p.default else [])
        rec_type = "fact_pair" if (p.required and not p.default) else "structured_config"
        tags = ["slb", "api", "param", "required" if p.required else "optional"]
        metadata = {
            "sample_kind": "param",
            "param_name": p.name,
            "required": p.required,
            "default": p.default,
            "enum_values": p.enum_values,
        }
        records.append(make_record(
            rec_id=f"slb-{api.action}-param-{slugify(p.name)}",
            rec_type=rec_type,
            action=api.action,
            title=f"{api.action}.{p.name}",
            anchor=f"[Action:{api.action}]/[Field:{p.name}]",
            content=content,
            paraphrases=paraphrases,
            recall_prompts=recall,
            must_contain=must,
            tags=tags,
            metadata=metadata,
        ))
        if p.default:
            records.append(make_record(
                rec_id=f"slb-{api.action}-default-{slugify(p.name)}",
                rec_type="structured_config",
                action=api.action,
                title=f"{api.action}.{p.name} 默认值",
                anchor=f"[Action:{api.action}]/[Default:{p.name}]",
                content=(
                    f"[Action:{api.action}]/[Default:{p.name}]\n"
                    f"在接口 {api.action} 中，参数 {p.name} 的默认值是 {p.default}。"
                ),
                paraphrases=[
                    f"如果 {api.action} 未显式传入 {p.name}，默认使用 {p.default}。",
                    f"{api.action} 的 {p.name} 缺省值为 {p.default}。",
                ],
                recall_prompts=[
                    f"{api.action} 的 {p.name} 默认值是什么？",
                    f"不传 {p.name} 时 {api.action} 会用什么值？",
                ],
                must_contain=[api.action, p.name, p.default],
                tags=["slb", "api", "default", "param"],
                metadata={"sample_kind": "default", "param_name": p.name, "default": p.default},
            ))
    return records


def constraint_records(api: ApiAction) -> List[Dict]:
    records = []
    seen = set()
    # derive from parameter desc and freeform notes
    lines = [f"{p.name}: {p.desc}" for p in api.all_params() if p.desc] + api.notes
    for idx, line in enumerate(lines, 1):
        if not any(k in line for k in KEYWORDS_CONSTRAINT):
            continue
        norm = line.strip()
        if norm in seen:
            continue
        seen.add(norm)
        content = f"[Action:{api.action}]/[Constraint:{idx}]\n{norm}"
        paraphrases = [
            f"接口 {api.action} 存在约束：{norm}",
            f"调用 {api.action} 时需要注意：{norm}",
        ]
        recall = [
            f"{api.action} 有哪些限制或条件？",
            f"调用 {api.action} 时要注意什么？",
        ]
        must = [api.action]
        # include a few technical anchors
        must.extend(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", norm)[:4])
        records.append(make_record(
            rec_id=f"slb-{api.action}-constraint-{idx}",
            rec_type="structured_config",
            action=api.action,
            title=f"{api.action} 约束 {idx}",
            anchor=f"[Action:{api.action}]/[Constraint:{idx}]",
            content=content,
            paraphrases=paraphrases,
            recall_prompts=recall,
            must_contain=must,
            tags=["slb", "api", "constraint"],
            metadata={"sample_kind": "constraint", "raw": norm},
        ))
    return records


def error_records(api: ApiAction) -> List[Dict]:
    records = []
    for code, name, desc in api.errors:
        content = (
            f"[Action:{api.action}]/[Error:{code}]\n"
            f"当调用 {api.action} 返回错误码 {code} {name} 时，表示 {desc}。"
        )
        paraphrases = [
            f"{api.action} 出现 {code} 时，对应错误名是 {name}。",
            f"错误 {code} 在 {api.action} 中表示：{desc}。",
        ]
        recall = [
            f"{api.action} 返回 {code} 代表什么？",
            f"{name} 是什么错误？",
        ]
        records.append(make_record(
            rec_id=f"slb-{api.action}-error-{code.replace('-', 'neg')}",
            rec_type="procedure",
            action=api.action,
            title=f"{api.action} 错误 {code}",
            anchor=f"[Action:{api.action}]/[Error:{code}]",
            content=content,
            paraphrases=paraphrases,
            recall_prompts=recall,
            must_contain=[api.action, code, name],
            tags=["slb", "api", "error"],
            metadata={"sample_kind": "error", "error_code": code, "error_name": name},
        ))
    return records


def structure_records(body: str) -> List[Dict]:
    records = []
    structure_names = ["Listener结构", "TcpConfig结构", "TcpCheck结构", "HttpCheck结构", "UdpConfig结构", "UdpCheck结构", "RealServer结构"]
    for sname in structure_names:
        params = parse_params(extract_bullet_block(body, sname.replace("结构", "结构")), required=False)
        if not params:
            continue
        struct_id = sname.replace("结构", "")
        fields = ", ".join(p.name for p in params)
        records.append(make_record(
            rec_id=f"slb-struct-{slugify(struct_id)}-overview",
            rec_type="structured_config",
            action=struct_id,
            title=f"{struct_id} 结构概览",
            anchor=f"[Struct:{struct_id}]/[Overview]",
            content=f"[Struct:{struct_id}]/[Overview]\n{struct_id} 结构包含字段: {fields}",
            paraphrases=[f"{struct_id} 的关键字段有 {fields}。"],
            recall_prompts=[f"{struct_id} 结构有哪些字段？"],
            must_contain=[struct_id] + [p.name for p in params[:4]],
            tags=["slb", "struct", "overview"],
            metadata={"sample_kind": "struct_overview", "fields": [p.name for p in params]},
        ))
        for p in params:
            records.append(make_record(
                rec_id=f"slb-struct-{slugify(struct_id)}-field-{slugify(p.name)}",
                rec_type="structured_config",
                action=struct_id,
                title=f"{struct_id}.{p.name}",
                anchor=f"[Struct:{struct_id}]/[Field:{p.name}]",
                content=f"[Struct:{struct_id}]/[Field:{p.name}]\n字段 {p.name} 在 {struct_id} 中的含义是：{p.desc}",
                paraphrases=[f"{struct_id} 里的 {p.name}: {p.desc}"],
                recall_prompts=[f"{struct_id} 的 {p.name} 是什么？"],
                must_contain=[struct_id, p.name],
                tags=["slb", "struct", "field"],
                metadata={"sample_kind": "struct_field", "struct": struct_id, "field": p.name},
            ))
    return records


def workflow_records(body: str) -> List[Dict]:
    records = []
    if "SLB实例从创建到上线的完整流程" not in body:
        return records
    # capture step lines
    steps = [clean_line(m.group(1)) for m in re.finditer(r"(?m)^步骤\d+[:：]\s*(.+)$", body)]
    ops = [clean_line(m.group(1)) for m in re.finditer(r"(?m)^\s*操作[:：]\s*(.+)$", body)]
    if steps:
        content = "[Flow:slb_online]/[Overview]\n" + "\n".join([f"步骤{i+1}: {s}" for i, s in enumerate(steps)])
        records.append(make_record(
            rec_id="slb-flow-online-overview",
            rec_type="procedure",
            action="slb_online_flow",
            title="SLB 从创建到上线流程",
            anchor="[Flow:slb_online]/[Overview]",
            content=content,
            paraphrases=[
                "SLB 上线流程通常包括创建 LB、准备后端、创建 VIP、激活 VIP、验证健康检查。",
                "SLB 的完整操作链可以拆成实例创建、后端准备、监听创建、激活、验证五步。",
            ],
            recall_prompts=[
                "SLB 从创建到上线的大致流程是什么？",
                "如何把一个新的 SLB 配到可服务状态？",
            ],
            must_contain=["create_loadbalancer", "create_vip", "config_vip", "query_lb_healthcheck"],
            tags=["slb", "flow", "procedure"],
            metadata={"sample_kind": "workflow", "steps": steps, "ops": ops},
        ))
    # explicit comparison lines
    cmp_lines = [clean_line(x) for x in body.splitlines() if "方式（add_rs）" in x or "方式（add_lb_rs）" in x or "不允许LB串联" in x]
    if cmp_lines:
        records.append(make_record(
            rec_id="slb-contrast-rspool-vs-lb-rs",
            rec_type="procedure",
            action="add_rs_vs_add_lb_rs",
            title="RSPool 方式与 LB 直挂方式对比",
            anchor="[Compare:add_rs|add_lb_rs]",
            content="[Compare:add_rs|add_lb_rs]\n" + "\n".join(cmp_lines),
            paraphrases=[
                "add_rs 更适合需要每个 RS 使用不同 port 的复杂场景。",
                "add_lb_rs 更适合后端端口统一、希望自动挂到所有监听的简单场景。",
            ],
            recall_prompts=[
                "add_rs 和 add_lb_rs 有什么区别？",
                "什么时候用 RSPool，什么时候直接给 LB 挂 RS？",
            ],
            must_contain=["add_rs", "add_lb_rs", "rs_pool_name", "port"],
            tags=["slb", "compare", "procedure"],
            metadata={"sample_kind": "contrast"},
        ))
    return records


def build_records(text: str) -> List[Dict]:
    records: List[Dict] = []
    sections = split_sections(text)
    for title, body in sections:
        # structure and workflow sections are handled specially
        if title.startswith("SLB数据结构"):
            records.extend(structure_records(body))
            continue
        if title.startswith("SLB完整操作流程"):
            records.extend(workflow_records(body))
            continue

        apis = split_actions_from_body(title, body)
        for api in apis:
            records.append(overview_record(api))
            records.extend(param_records(api))
            records.extend(constraint_records(api))
            records.extend(error_records(api))

    # de-duplicate by id
    dedup: Dict[str, Dict] = {}
    for r in records:
        dedup[r["id"]] = r
    return list(dedup.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Engram JSONL from SLB API document")
    parser.add_argument("input", help="Path to input UTF-8 text file")
    parser.add_argument("output", help="Path to output JSONL file")
    args = parser.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    records = build_records(text)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records -> {out}")
    by_type: Dict[str, int] = {}
    for r in records:
        by_type[r['type']] = by_type.get(r['type'], 0) + 1
    print("By type:", json.dumps(by_type, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
