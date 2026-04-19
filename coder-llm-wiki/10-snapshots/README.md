# Snapshots

Store one checkpoint file per task or batch in this directory.

Suggested shape:

```json
{
  "id": "module-auth",
  "type": "module",
  "status": "review-needed",
  "last_stage": "evidence-linked",
  "outputs": [
    "coder-llm-wiki/03-modules/auth.md",
    "coder-llm-wiki/08-evidence/auth.refs.md"
  ],
  "blockers": [],
  "updated_at": "2026-04-19T00:00:00Z"
}
```
