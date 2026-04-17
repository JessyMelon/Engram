#!/usr/bin/env python3
"""Expand canonical facts into Engram-friendly samples using the OpenAI API.

Features:
- Uses the Responses API.
- Uses Structured Outputs with a strict JSON Schema.
- Supports synchronous generation and optional Batch JSONL emission.
- Validates generated samples before writing them.

Usage:
  python expand_engram_samples.py \
      --input canonical_facts.jsonl \
      --output expanded_samples.jsonl \
      --schema engram_sample_schema.json \
      --prompt engram_sample_expansion_prompt.md

  # Generate batch requests instead of calling the API directly
  python expand_engram_samples.py \
      --input canonical_facts.jsonl \
      --schema engram_sample_schema.json \
      --prompt engram_sample_expansion_prompt.md \
      --emit-batch-jsonl batch_requests.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: openai. Install with `pip install openai jsonschema`."
    ) from exc

try:
    from jsonschema import Draft202012Validator
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: jsonschema. Install with `pip install jsonschema`."
    ) from exc


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
DEFAULT_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "low")
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


@dataclass
class GenerationResult:
    fact_id: str
    ok: bool
    sample: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(load_text(path))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {lineno}: {exc}") from exc
    return records


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_messages(system_prompt: str, fact: Dict[str, Any]) -> List[Dict[str, str]]:
    payload = {
        "canonical_fact": fact,
        "generation_rules": {
            "language": "zh-CN",
            "min_paraphrases": 4,
            "max_paraphrases": 4,
            "include_negative_terms": True,
            "prefer_atomic_sentences": True,
            "prefer_action_field_adjacency": True,
        },
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def build_text_format(schema: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": "engram_sample",
            "strict": True,
            "schema": schema,
        }
    }


class SampleValidator:
    def __init__(self, schema: Dict[str, Any]) -> None:
        self.schema = schema
        self.validator = Draft202012Validator(schema)

    def validate(self, sample: Dict[str, Any], fact: Dict[str, Any]) -> List[str]:
        errors = [e.message for e in self.validator.iter_errors(sample)]

        action = str(sample.get("action", "")).strip()
        content = str(sample.get("content", ""))
        paraphrases = [str(x) for x in sample.get("paraphrases", [])]
        joined = "\n".join([content] + paraphrases)
        must_contain = [str(x) for x in sample.get("must_contain", [])]
        negative_terms = [str(x) for x in sample.get("negative_terms", [])]
        sample_type = str(sample.get("sample_type", ""))

        # 1) action must match source fact
        fact_action = str(fact.get("action", "")).strip()
        if fact_action and action != fact_action:
            errors.append(f"action mismatch: generated={action!r} expected={fact_action!r}")

        # 2) must_contain must truly appear
        missing = [token for token in must_contain if token and token not in joined]
        if missing:
            errors.append(f"must_contain tokens missing from content/paraphrases: {missing}")

        # 3) high-value field / error_code should appear if present in source
        for key in ("field", "error_code", "value"):
            value = fact.get(key)
            if value is None:
                continue
            value_str = str(value).strip()
            if value_str and value_str not in joined:
                errors.append(f"source {key}={value_str!r} missing from generated text")

        # 4) encourage adjacency patterns for Engram
        field = str(fact.get("field", "")).strip()
        error_code = str(fact.get("error_code", "")).strip()
        if field:
            if not _contains_close_pair(joined, action, field):
                errors.append(
                    f"action and field are never near-adjacent: {action!r}, {field!r}"
                )
        if error_code:
            if not _contains_close_pair(joined, action, error_code):
                errors.append(
                    f"action and error_code are never near-adjacent: {action!r}, {error_code!r}"
                )

        # 5) content length sanity
        if len(content) < 80:
            errors.append("content too short (<80 chars)")

        # 6) repeated paraphrases / duplicates
        normalized = [_normalize_text(x) for x in paraphrases]
        if len(set(normalized)) != len(normalized):
            errors.append("duplicate paraphrases detected")

        # 7) negative terms should not dominate the content too aggressively
        if negative_terms:
            heavy = [t for t in negative_terms if joined.count(t) > 3]
            if heavy:
                errors.append(f"negative_terms repeated too often: {heavy}")

        # 8) sample_type should remain consistent when provided in source
        fact_sample_type = str(fact.get("sample_type", "")).strip()
        if fact_sample_type and fact_sample_type != sample_type:
            errors.append(
                f"sample_type mismatch: generated={sample_type!r} expected={fact_sample_type!r}"
            )

        return errors


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _contains_close_pair(text: str, a: str, b: str, max_gap_words: int = 6) -> bool:
    if not a or not b:
        return True
    tokens = re.findall(r"[A-Za-z0-9_\-\.]+|[\u4e00-\u9fff]+", text)
    if not tokens:
        return False
    a_positions = [i for i, tok in enumerate(tokens) if tok == a]
    b_positions = [i for i, tok in enumerate(tokens) if tok == b]
    for i in a_positions:
        for j in b_positions:
            if abs(i - j) <= max_gap_words:
                return True
    return False


def call_openai(
    client: OpenAI,
    model: str,
    reasoning_effort: str,
    schema: Dict[str, Any],
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    response = client.responses.create(
        model=model,
        input=messages,
        reasoning={"effort": reasoning_effort},
        text=build_text_format(schema),
    )
    # The Python SDK exposes output_text for text responses.
    raw = getattr(response, "output_text", None)
    if not raw:
        # Safe fallback if SDK shape changes or output_text is empty.
        raw = json.dumps(response.model_dump(), ensure_ascii=False)
        raise RuntimeError(f"No output_text in response. Raw response: {raw}")
    return json.loads(raw)


def generate_sample(
    client: OpenAI,
    model: str,
    reasoning_effort: str,
    schema: Dict[str, Any],
    system_prompt: str,
    fact: Dict[str, Any],
    sample_validator: SampleValidator,
    max_retries: int = 3,
) -> GenerationResult:
    fact_id = str(fact.get("fact_id", fact.get("id", "unknown_fact")))
    messages = build_messages(system_prompt, fact)

    for attempt in range(1, max_retries + 1):
        try:
            sample = call_openai(client, model, reasoning_effort, schema, messages)
            errors = sample_validator.validate(sample, fact)
            if errors:
                return GenerationResult(
                    fact_id=fact_id,
                    ok=False,
                    error="; ".join(errors),
                    sample=sample,
                )
            return GenerationResult(fact_id=fact_id, ok=True, sample=sample)
        except Exception as exc:  # pragma: no cover
            if attempt == max_retries:
                return GenerationResult(fact_id=fact_id, ok=False, error=str(exc))
            sleep_s = 2 ** (attempt - 1)
            time.sleep(sleep_s)

    return GenerationResult(fact_id=fact_id, ok=False, error="unknown failure")


def emit_batch_requests(
    input_records: List[Dict[str, Any]],
    output_path: Path,
    system_prompt: str,
    schema: Dict[str, Any],
    model: str,
    reasoning_effort: str,
) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for idx, fact in enumerate(input_records, start=1):
            custom_id = str(fact.get("fact_id", f"fact_{idx}"))
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": model,
                    "input": build_messages(system_prompt, fact),
                    "reasoning": {"effort": reasoning_effort},
                    "text": build_text_format(schema),
                },
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Canonical facts JSONL")
    parser.add_argument("--output", type=Path, help="Generated samples JSONL")
    parser.add_argument("--schema", type=Path, required=True, help="Output JSON Schema file")
    parser.add_argument("--prompt", type=Path, required=True, help="Prompt markdown file")
    parser.add_argument("--errors", type=Path, default=Path("generation_errors.jsonl"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument(
        "--emit-batch-jsonl",
        type=Path,
        help="Instead of calling the API, emit Batch API request lines to this file.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N records")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    schema = load_json(args.schema)
    prompt_md = load_text(args.prompt)
    # Extract only the system prompt block if present.
    match = re.search(r"## 1\) System prompt\s+```text\n(.*?)\n```", prompt_md, re.S)
    system_prompt = match.group(1).strip() if match else prompt_md.strip()

    facts = load_jsonl(args.input)
    if args.limit > 0:
        facts = facts[: args.limit]

    if args.emit_batch_jsonl:
        emit_batch_requests(
            input_records=facts,
            output_path=args.emit_batch_jsonl,
            system_prompt=system_prompt,
            schema=schema,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
        )
        print(f"Wrote batch request file: {args.emit_batch_jsonl}")
        return 0

    if not args.output:
        raise SystemExit("--output is required unless --emit-batch-jsonl is used")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    sample_validator = SampleValidator(schema)

    good_rows: List[Dict[str, Any]] = []
    bad_rows: List[Dict[str, Any]] = []

    for idx, fact in enumerate(facts, start=1):
        result = generate_sample(
            client=client,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            schema=schema,
            system_prompt=system_prompt,
            fact=fact,
            sample_validator=sample_validator,
        )
        if result.ok and result.sample is not None:
            good_rows.append(result.sample)
            print(f"[{idx}/{len(facts)}] OK   {result.fact_id}")
        else:
            payload = {
                "fact_id": result.fact_id,
                "error": result.error,
                "sample": result.sample,
                "canonical_fact": fact,
            }
            bad_rows.append(payload)
            print(f"[{idx}/{len(facts)}] FAIL {result.fact_id}: {result.error}", file=sys.stderr)

    write_jsonl(args.output, good_rows)
    if bad_rows:
        write_jsonl(args.errors, bad_rows)

    print(f"\nDone. success={len(good_rows)} failed={len(bad_rows)}")
    print(f"Samples written to: {args.output}")
    if bad_rows:
        print(f"Failures written to: {args.errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
