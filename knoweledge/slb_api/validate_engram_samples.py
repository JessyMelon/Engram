#!/usr/bin/env python3
"""Validate generated Engram sample JSONL against schema + custom quality checks.

Usage:
  python validate_engram_samples.py \
      --input expanded_samples.jsonl \
      --schema engram_sample_schema.json \
      --canonical canonical_facts.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from jsonschema import Draft202012Validator


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def close_pair(text: str, a: str, b: str, max_gap_words: int = 6) -> bool:
    tokens = re.findall(r"[A-Za-z0-9_\-\.]+|[\u4e00-\u9fff]+", text)
    a_positions = [i for i, tok in enumerate(tokens) if tok == a]
    b_positions = [i for i, tok in enumerate(tokens) if tok == b]
    for i in a_positions:
        for j in b_positions:
            if abs(i - j) <= max_gap_words:
                return True
    return False


def validate_sample(
    sample: Dict[str, Any],
    schema_validator: Draft202012Validator,
    canonical: Dict[str, Any] | None,
) -> List[str]:
    errors = [e.message for e in schema_validator.iter_errors(sample)]

    content = str(sample.get("content", ""))
    paraphrases = [str(x) for x in sample.get("paraphrases", [])]
    combined = "\n".join([content] + paraphrases)

    # must_contain check
    for token in sample.get("must_contain", []):
        token = str(token)
        if token and token not in combined:
            errors.append(f"must_contain token missing: {token}")

    # duplicate paraphrase check
    normalized = [normalize(x) for x in paraphrases]
    if len(set(normalized)) != len(normalized):
        errors.append("duplicate paraphrases detected")

    # weak anchor check
    action = str(sample.get("action", "")).strip()
    anchor = str(sample.get("anchor_text", ""))
    if action and action not in anchor:
        errors.append("anchor_text does not contain action")

    # Canonical-grounded checks
    if canonical:
        canonical_action = str(canonical.get("action", "")).strip()
        if canonical_action and action != canonical_action:
            errors.append(f"action mismatch with canonical fact: {action} != {canonical_action}")

        field = str(canonical.get("field", "")).strip()
        error_code = str(canonical.get("error_code", "")).strip()
        value = str(canonical.get("value", "")).strip()

        for token_name, token_value in [("field", field), ("error_code", error_code), ("value", value)]:
            if token_value and token_value not in combined:
                errors.append(f"canonical {token_name} missing in generated text: {token_value}")

        if field and not close_pair(combined, action, field):
            errors.append(f"action and field are not near-adjacent: {action}, {field}")
        if error_code and not close_pair(combined, action, error_code):
            errors.append(f"action and error_code are not near-adjacent: {action}, {error_code}")

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, help="Canonical facts JSONL for grounded checks")
    parser.add_argument("--report", type=Path, default=Path("validation_report.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    schema = load_json(args.schema)
    validator = Draft202012Validator(schema)
    samples = load_jsonl(args.input)

    canonical_map: Dict[str, Dict[str, Any]] = {}
    if args.canonical:
        canonical_rows = load_jsonl(args.canonical)
        for row in canonical_rows:
            fid = str(row.get("fact_id", row.get("id", "")))
            if fid:
                canonical_map[fid] = row

    errors_by_sample: Dict[str, List[str]] = {}
    ok = 0
    failed = 0

    for row in samples:
        sample_id = str(row.get("sample_id", "unknown_sample"))
        # By convention, sample_id often begins with fact_id. Adjust if needed.
        canonical = None
        for fid, fact in canonical_map.items():
            if sample_id.startswith(fid):
                canonical = fact
                break
        errs = validate_sample(row, validator, canonical)
        if errs:
            failed += 1
            errors_by_sample[sample_id] = errs
        else:
            ok += 1

    report = {
        "total": len(samples),
        "ok": ok,
        "failed": failed,
        "errors_by_sample": errors_by_sample,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
