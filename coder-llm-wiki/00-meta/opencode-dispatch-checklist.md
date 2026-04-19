# OpenCode Dispatch Checklist

Use this as the operator runbook for a long-running repository analysis.

## Phase 0: Initialize
- Create or verify the `coder-llm-wiki/` directory tree.
- Read `coder-llm-wiki/00-meta/project-charter.md` before starting analysis.
- Update `coder-llm-wiki/00-meta/progress.json` to set `phase` to `inventory`.
- Clear or seed `coder-llm-wiki/00-meta/task-queue.json` with the next batch of tasks.
- Record any repository-specific naming rules in `coder-llm-wiki/00-meta/glossary.md`.

## Phase 1: Inventory
- Read root documentation and top-level config files.
- Identify top-level apps, packages, services, scripts, and data directories.
- Identify runtime entrypoints such as app startup, CLI commands, workers, schedulers, or consumers.
- Write `coder-llm-wiki/01-inventory/repo-map.md`.
- Write `coder-llm-wiki/01-inventory/tech-stack.md`.
- Write `coder-llm-wiki/01-inventory/entrypoints.md`.
- Write `coder-llm-wiki/01-inventory/module-candidates.json`.
- Update `progress.json` to mark `inventory` as `done`.

## Phase 2: Index
- Update `progress.json` to set `phase` to `index`.
- Build a route or command index from real entrypoints.
- Build a symbol index for core classes, functions, or schemas.
- Build a job, event, or background task index if the repository has them.
- Build a test map that links important code to tests.
- Write outputs under `coder-llm-wiki/02-index/`.
- Update `progress.json` to mark `index` as `done`.

## Phase 3: Prepare Module Queue
- Review `module-candidates.json`.
- Split the repository into stable module units.
- Create one queue item per module in `task-queue.json`.
- For each module task, include scope, expected outputs, and current status.
- Mark the next module batch as ready.

## Phase 4: Module Analysis Loop
- Update `progress.json` to set `phase` to `modules`.
- Pick the next module task with status `pending`.
- Mark the task as `scanning`.
- Analyze the module directory, related entrypoints, config, and tests.
- Write `coder-llm-wiki/03-modules/<module>.md` from the module template.
- Write `coder-llm-wiki/08-evidence/<module>.refs.md` with file and line references.
- Write `coder-llm-wiki/09-review/<module>.questions.md` for unresolved gaps.
- Mark the task as `review-needed`.
- Repeat until the current batch is complete.

## Phase 5: Lightweight Review Loop
- Review each newly written module doc against the evidence doc.
- Check that the module doc states purpose, boundaries, dependencies, data interactions, risks, tests, and open questions.
- Check that at least five high-value evidence references are present when the module is non-trivial.
- If the module doc is acceptable, mark the task as `done`.
- If the module doc is weak or inconsistent, mark the task as `blocked` or return it to `pending` with a note.

## Phase 6: Flow Planning
- Update `progress.json` to set `phase` to `flows`.
- Identify the highest-value flows from entrypoints, modules, and tests.
- Create one queue item per flow in `task-queue.json`.
- Prefer business-critical, failure-prone, or heavily integrated flows first.

## Phase 7: Flow Analysis Loop
- Pick the next flow task with status `pending`.
- Mark the task as `scanning`.
- Start from real entrypoints and trace the call path.
- Capture the main path, failure paths, state changes, and external calls.
- Write `coder-llm-wiki/04-flows/<flow>.md`.
- Add or update any supporting evidence docs under `coder-llm-wiki/08-evidence/`.
- Mark the flow task as `review-needed`.

## Phase 8: Cross Review
- Update `progress.json` to set `phase` to `review`.
- Compare flow docs against related module docs.
- Record factual conflicts in `coder-llm-wiki/09-review/conflict-log.md`.
- Record unresolved gaps in `coder-llm-wiki/09-review/open-questions.md`.
- Record items that need human judgment in `coder-llm-wiki/09-review/human-review.md`.
- Fix weak docs or queue them for another pass.
- Mark `review` as `done` only when all critical conflicts are either resolved or explicitly tracked.

## Phase 9: Snapshot And Resume
- After each batch, write a checkpoint file under `coder-llm-wiki/10-snapshots/`.
- Include task id, current stage, outputs written, and remaining blockers.
- On restart, read `progress.json`, `task-queue.json`, and the latest snapshot before resuming.

## Phase 10: Incremental Maintenance
- Update `progress.json` to set `phase` to `incremental_updates` when running against new code changes.
- Read the current git diff or changed files.
- Map code changes to affected modules and flows.
- Update only the impacted wiki docs and evidence files.
- Append any new uncertainty to the review docs.
- Keep old evidence only if it still matches the current code.

## Status Vocabulary
- `pending`: not started
- `scanning`: source discovery in progress
- `summarizing`: writing the main document
- `evidence-linked`: evidence file updated
- `review-needed`: ready for review
- `done`: complete for now
- `blocked`: cannot continue without clarification or conflict resolution

## Operator Notes
- Do not treat generated text as final until evidence is linked.
- Do not merge facts and interpretation into the same section when certainty is low.
- Prefer rerunning a small task over rewriting a large batch.
- Keep the queue small enough to review regularly.
