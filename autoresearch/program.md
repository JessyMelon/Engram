# engram-autoresearch

This is an experiment to have the LLM autonomously optimize Engram knowledge injection.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar26`). The branch `engram-autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b engram-autoresearch/<tag>` from current main/master.
3. **Read the in-scope files**: Read these files for full context:
   - `train.py` — the file you modify. Engram architecture, optimizer, hyperparameters, training loop.
   - `prepare.py` — fixed constants, evaluation functions. Do not modify (except in Phase 4 for BASE_MODEL).
   - `knowledge_format.py` — knowledge data and formatting. Do not modify (except in Phase 3 for format optimization).
4. **Verify dependencies**: Ensure PyTorch, transformers are installed. The base model will be auto-downloaded.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 10 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the primary file you edit. Engram architecture, optimizer, hyperparameters, training loop, batch size, etc.
- In Phase 3: Modify `knowledge_format.py` to optimize training text assembly.
- In Phase 4: Modify `prepare.py` to change BASE_MODEL for scale-up experiments.

**What you CANNOT do:**
- Modify `prepare.py` evaluation functions (`evaluate_recall`, `evaluate_ppl`). They are ground truth metrics.
- Install new packages or add dependencies. Use only what's available (torch, transformers, numpy).
- Modify the evaluation harness logic.

**The goal is simple: get the highest recall_score.** This measures how well the model recalls injected knowledge. Secondary goal: lowest val_ppl (validation perplexity) as a language modeling quality indicator.

**VRAM constraint**: Peak VRAM must stay under ~22GB (A10 has 24GB, keep 2GB safety margin). Experiments exceeding this will crash with OOM.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.01 recall_score improvement that adds 20 lines of hacky code? Probably not worth it. A 0.01 recall_score improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Hardware

- CPU: 16 cores
- RAM: 60GB
- GPU: NVIDIA A10 (Ampere SM 8.6)
- VRAM: 24GB
- Compute: bf16 supported

## Output format

Once the script finishes it prints a summary like this:

```
---
recall_score:     0.750000
val_ppl:          12.345678
training_seconds: 598.5
total_seconds:    650.2
peak_vram_mb:     18500.0
engram_params_M:  15.2
base_model:       Qwen/Qwen2.5-1.5B
total_steps:      1250
final_loss:       2.345678
```

You can extract the key metrics from the log file:

```
grep "^recall_score:\|^val_ppl:\|^peak_vram_mb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	recall_score	val_ppl	peak_vram_gb	status	description
```

1. git commit hash (short, 7 chars)
2. recall_score achieved (e.g. 0.750000) — use 0.000000 for crashes
3. val_ppl achieved (e.g. 12.34) — use 0.00 for crashes
4. peak memory in GB, round to .1f (e.g. 18.1 — divide peak_vram_mb by 1024) — use 0.0 for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	recall_score	val_ppl	peak_vram_gb	status	description
a1b2c3d	0.750000	12.34	18.1	keep	baseline
b2c3d4e	0.820000	11.56	18.3	keep	increase LR to 2e-3
c3d4e5f	0.740000	13.20	18.0	discard	reduce N_EMBED_PER_NGRAM to 64
d4e5f6g	0.000000	0.00	0.0	crash	double vocab_size (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `engram-autoresearch/mar26`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^recall_score:\|^val_ppl:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If recall_score improved (higher), you "advance" the branch, keeping the git commit
9. If recall_score is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~10 minutes total (+ a few minutes for startup, model loading, and eval overhead). If a run exceeds 20 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~15 minutes (including startup overhead) then you can run approx 4/hour, for a total of about 30+ over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

## Search strategy

The search is organized into 4 phases. Progress through phases as you gain confidence.

### Phase 1: Baseline + Quick Scan (~10 rounds)

Goal: Establish baseline and find initial working configurations.

1. **No-Engram baseline**: Run with Engram disabled to see base model performance
2. **Default config baseline**: Run current train.py as-is
3. **Learning rate sweep**: Try [5e-4, 1e-3, 2e-3, 5e-3] — LR often has biggest impact
4. **Layer selection variants**: Try ENGRAM_LAYERS = [4], [8], [4, 8], [4, 12]

Key metrics to watch:
- recall_score should be > 0 with Engram enabled
- val_ppl should be reasonable (< 50)
- peak_vram_mb should be < 20000 (safe margin)

### Phase 2: Architecture Search (~15 rounds)

Goal: Find optimal Engram architecture parameters.

1. **engram_vocab_size sweep**: Try [40000, 40000], [80000, 80000], [120000, 120000]
2. **n_embed_per_ngram sweep**: Try 64, 128, 256 — affects capacity vs. memory tradeoff
3. **n_head_per_ngram sweep**: Try 2, 4, 8 — more heads = more diversity
4. **kernel_size sweep**: Try 2, 4, 8 — affects local context modeling
5. **max_ngram_size**: Try 3 vs 4 — higher captures more context but needs more memory

Combine promising findings. E.g., if LR=2e-3 and n_embed=256 both help, try them together.

### Phase 3: Knowledge Format Optimization (~10 rounds)

Goal: Optimize how knowledge is presented to the model.

At this phase, you CAN modify `knowledge_format.py`:
- Change how training text is assembled in `build_training_text()`
- Experiment with different repetition strategies
- Try different separators or formatting

Also try:
- Different SEQ_LEN values: 128, 256, 384, 512
- Batch size adjustments for memory optimization

### Phase 4: Scale Up to Qwen2.5-3B (~5 rounds)

Goal: Validate findings on larger model.

At this phase, you CAN modify `prepare.py`:
- Change BASE_MODEL to "Qwen/Qwen2.5-3B"
- Adjust BATCH_SIZE in train.py (likely need to reduce due to higher VRAM)
- Re-tune critical hyperparameters (LR may need adjustment)

**Caution**: Qwen2.5-3B will use significantly more VRAM. Monitor peak_vram_mb closely. May need BATCH_SIZE=1 or 2.

---

**Remember**: The goal is to maximize recall_score while keeping val_ppl reasonable and peak_vram under 22GB. Be systematic, track what works, and build on successes. Good luck!
