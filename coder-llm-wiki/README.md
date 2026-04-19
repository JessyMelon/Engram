# LLM Wiki

This directory stores a long-running, evidence-backed knowledge base for the repository.

Rules:
- Write every important conclusion with source evidence.
- Keep facts and interpretation separate.
- Prefer incremental updates over full rewrites.
- Record open questions instead of guessing.

Structure:
- `00-meta/`: project rules, progress, queue, prompts, glossary
- `01-inventory/`: repo map, stack, entrypoints, module candidates
- `02-index/`: routes, models, jobs, symbols, tests
- `03-modules/`: one document per module
- `04-flows/`: one document per important flow
- `05-data/`: schema, cache, event, storage notes
- `06-ops/`: run, build, deploy, monitor, recovery notes
- `07-risks/`: fragile areas, technical debt, hidden constraints
- `08-evidence/`: file and line references backing each doc
- `09-review/`: conflicts, questions, human review backlog
- `10-snapshots/`: per-task checkpoints and resumable state

Recommended workflow:
1. Fill inventory and index first.
2. Generate module docs from the candidate list.
3. Generate flow docs from real entrypoints.
4. Run review passes and track unresolved issues.
5. Update docs incrementally when code changes.
