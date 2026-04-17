# AGENTS.md

Repository instructions for AI coding agents working on the Engram memory-data pipeline.

## Mission

This repository is used to build, expand, validate, and evaluate training data for:
- Engram hot-memory samples
- external expert-memory samples
- API / ops / troubleshooting knowledge compilation

The primary goal is **high-quality, verifiable knowledge injection**, not free-form text generation.

Agents must optimize for:
1. factual correctness
2. stable trigger patterns for Engram
3. structured outputs that can be validated automatically
4. reproducible scripts and experiments
5. minimal manual cleanup after generation

---

## Core principles

### 1) Never invent facts
Only generate or transform samples from explicit source material:
- API docs
- canonical_facts.jsonl
- existing validated JSONL records
- repository scripts and schema files

If a fact is missing from the source, mark it as `needs_review` instead of guessing.

### 2) Preserve hard tokens exactly
These tokens must remain unchanged when present:
- action / API names
- parameter names
- error codes
- enum values
- product names
- protocol names
- field paths

Examples:
- `create_loadbalancer`
- `eip_type`
- `gw_type=vpc`
- `-2249`
- `https`

Do not translate, rename, or normalize these tokens semantically.

### 3) Optimize for Engram trigger quality
When rewriting or expanding samples, prefer:
- action + parameter adjacency
- short high-signal factual sentences
- repeated but meaningful local token patterns
- multiple trigger entrances for the same fact

Do **not** optimize only for prose quality.

### 4) Separate truth from expression
Use this pipeline:
- canonical facts = truth layer
- expanded samples = expression layer
- validation = enforcement layer

Never let expression generation alter truth.

### 5) Prefer deterministic pipelines
If a task can be done by parsing, transforming, filtering, or validating with code, do that first.
Use LLM calls only for:
- multi-angle paraphrases
- scenario phrasing
- contrast samples
- error-recovery phrasing
- controlled gap filling from explicit evidence

---

## Repository workflow model

Agents should follow this order unless the user explicitly requests otherwise:

1. inspect source material
2. update or generate `canonical_facts.jsonl`
3. expand samples through structured generation
4. validate JSON schema and fact consistency
5. compute quality metrics
6. only then modify training/eval datasets
7. summarize changes and risks

Do not skip validation.

---

## Preferred data layers

### Layer A: canonical facts
Minimal, source-grounded records.

Suggested fields:
- `fact_id`
- `domain`
- `action`
- `field`
- `fact_type`
- `value`
- `condition`
- `error_code`
- `evidence`
- `source_ref`
- `confidence`

### Layer B: expanded Engram samples
Derived from canonical facts.

Required fields:
- `sample_id`
- `action`
- `sample_type`
- `content`
- `paraphrases`
- `must_contain`

Recommended fields:
- `anchor_text`
- `negative_terms`
- `contrast_with`
- `source_fact_ids`
- `needs_review`

### Layer C: eval samples
Used only for measurement. Keep separate from training data.

---

## Sample-writing rules

### Good sample shape
Each high-value sample should try to contain:
- 1 main factual sentence
- 1 action+field close-adjacency sentence
- 1 usage/scenario sentence
- 1 reverse or boundary-condition sentence
- 3 to 4 paraphrases with different trigger paths

### Paraphrase requirements
Paraphrases must differ by **entry angle**, not just synonyms.
Prefer these angles:
- definition
- usage scenario
- default / omitted behavior
- reverse constraint
- failure / recovery
- contrast with similar API

### Avoid
- decorative long prose
- unsupported inferences
- changing parameter names to Chinese aliases only
- generic filler like “this parameter is very important”
- one-line samples with no action context

---

## Contrast and negative samples

For confusing APIs or fields, generate explicit contrast samples.

Examples:
- `add_rs` vs `add_lb_rs`
- `backend_port` vs `rs_pool_name`
- `query_loadbalancer_info` vs `list_loadbalancers`

For contrast samples:
- name both sides explicitly
- state one shared trait
- state the key difference
- include one misuse warning if grounded in source

Use `negative_terms` when a sample is likely to be confused with nearby concepts.

---

## Validation rules

Agents must validate generated data before proposing it.

### Required checks
- valid JSON / JSONL
- schema compliance
- required fields present
- `must_contain` strings actually appear in `content` or `paraphrases`
- no invented action / field / error code outside canonical facts
- duplicate and near-duplicate detection
- no train/eval leakage if eval set exists

### Soft checks
- content too short
- paraphrase diversity too low
- shared-template repetition too high
- likely confusion with other APIs

If validation fails, fix the data or mark records `needs_review`.

---

## Metrics agents should track

When changing datasets, report deltas for:
- number of canonical facts
- number of training samples
- average content length
- average paraphrases per sample
- percentage of samples with scenario sentence
- percentage of samples with reverse-constraint sentence
- duplicate rate
- validation failure count
- contrast sample count
- error-recovery sample count

If training or eval is run, also report:
- recall score
- validation perplexity or equivalent loss metric
- tool-call accuracy if available
- constraint satisfaction rate if available

---

## Safe edit policy

Agents may freely edit:
- generation scripts
- validators
- prompts
- schemas
- docs
- dataset files derived from source

Agents must be cautious with:
- deleting large datasets
- overwriting manually curated files
- changing evaluation splits
- changing model/training hyperparameters without explanation

Before destructive changes:
- create a backup file or git commit
- explain what changed
- summarize rollback steps

---

## Command conventions

Prefer commands like these when available:

```bash
python generate_*\.py
python expand_*\.py
python validate_*\.py
python stats_*\.py
```

If exact commands differ, inspect the repo first and then use project-native commands.

When adding new scripts:
- make names explicit
- keep single responsibility
- accept input/output CLI args
- print machine-readable summaries when possible

---

## Pull request expectations

Every dataset or pipeline PR should include:
- what source material was used
- whether facts or only expressions changed
- what scripts were run
- validation results
- key metrics before/after
- known risks
- files requiring human review

Do not submit a “big generated diff” without a concise explanation.

---

## Agent task recipes

### Task: add new API knowledge
1. parse source doc
2. build/update canonical facts
3. generate expanded samples with structured output
4. run validator
5. generate summary stats
6. prepare a small reviewable diff

### Task: improve weak samples
1. identify short or low-diversity samples
2. preserve fact layer
3. regenerate only expression layer
4. compare metrics before/after
5. keep best version

### Task: reduce confusion between similar APIs
1. locate overlapping action/field families
2. add contrast samples
3. add negative terms
4. add boundary-condition phrasing
5. rerun validator and confusion-focused eval

### Task: prepare Engram hot set
1. rank samples by frequency/value/stability
2. prefer defaults, constraints, error recovery, contrasts
3. exclude long-tail weakly validated material
4. export compact high-signal dataset

---

## Definition of done

A task is done only when:
- source-grounded facts are preserved
- generated samples pass validation
- changes are summarized clearly
- outputs are reproducible from scripts
- high-risk records are marked for review

