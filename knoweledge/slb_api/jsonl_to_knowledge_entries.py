#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert the generated JSONL schema into the minimal KNOWLEDGE_ENTRIES format
expected by autoresearch/knowledge_format.py.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path


def main(inp: str, out: str) -> None:
    records = [json.loads(line) for line in Path(inp).read_text('utf-8').splitlines() if line.strip()]
    entries = []
    for r in records:
        entries.append({
            'type': r['type'],
            'content': r['content'],
            'paraphrases': r.get('paraphrases', []),
            'recall_prompts': r.get('recall_prompts', []),
        })
    Path(out).write_text(
        'KNOWLEDGE_ENTRIES = ' + json.dumps(entries, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    print(f'Wrote {len(entries)} entries -> {out}')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python jsonl_to_knowledge_entries.py input.jsonl output.py')
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2])
