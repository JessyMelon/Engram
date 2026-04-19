# Project Charter

## Goal
Build a repository wiki that helps future development and maintenance through source-grounded, reviewable documentation.

## Non-Goals
- Do not produce prose-only summaries with no evidence.
- Do not guess design intent when the code does not support it.
- Do not rewrite the full wiki on every run.

## Documentation Rules
- Facts must be supported by code, config, tests, or other repository sources.
- Put uncertainty into `Open Questions` instead of presenting it as fact.
- Prefer stable templates so docs can be compared and regenerated.
- Keep file paths and important tokens exact.

## Evidence Rules
- Use file paths and line ranges when possible.
- Prefer primary sources: entrypoints, configs, tests, and core implementation.
- If a conclusion is inferred from multiple files, cite all of them.

## Task Rules
- Work in small, resumable tasks.
- Persist results to disk after each task.
- Update progress and queue state after each task.
- Mark blocked tasks explicitly with a reason.

## Completion Standards
- A module doc is only `done` after evidence is linked and questions are recorded.
- A flow doc is only `done` after the main path and failure paths are described.
- A review pass is only `done` after conflicts and unresolved gaps are logged.
