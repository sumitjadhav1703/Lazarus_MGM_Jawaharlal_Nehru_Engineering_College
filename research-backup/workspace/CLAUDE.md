# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

CardioTwin: Python research prototype that forecasts a **new atrial-fibrillation onset 2–10 min ahead** from the preceding 10 min of ECG (decision every 30 s), plus a FastAPI/SQLite replay service and React dashboard. Research only; no clinical-readiness claim.

`AGENTS.md` (written for the earlier Codex workflow) is the long-form history: dated stage notes, protected-file lists, receipts. Read its "Current stage" section before touching experiments, but trust `reports/ml-investigation/next-cycle/ecgfounder-cycle-20261013/{state.json,progress.md}` over any prose when they disagree. Its model-routing section (Astra/Sol/Terra) is Codex-specific and **superseded by the orchestration below**.

## Orchestration (Opus plans, Sonnet codes)

The main session runs **Opus**. It thinks, plans, makes scientific calls and reviews evidence. It does not write production code itself except for trivial one-line edits.

Subagents (`.claude/agents/`): Sonnet does routine work (coding, review, fact lookup); the Opus `cardio-critic` handles hard critical thinking. Do not use Opus subagents for coding.

| Agent | Use for | Tools |
| --- | --- | --- |
| `cardio-coder` | Implement one scoped unit with tests (RED→GREEN) | read/write/bash |
| `cardio-reviewer` | Read-only technical review of a finished unit | read-only |
| `cardio-researcher` | Verify external facts on the web, return cited findings | search/fetch, read-only |
| `cardio-critic` (**Opus**) | Adversarial gap audit / feasibility / "worth the time?" verdicts, with cited sources | read, bash, search/fetch; no edits |

Rules for Opus:

1. **Plan first.** Before delegating, write down: goal, exact files each agent may touch, interfaces between units, tests that prove it, and what must NOT change. Use plan mode for anything touching more than one module.
2. **Parallelize only on disjoint files.** Several `cardio-coder` agents may run at once when their file sets do not overlap (spawn them in one message). Workspace is **not a git repo** — there is no worktree isolation, so overlapping edits silently clobber each other. Shared files (`config.py`, `evaluation.py`, `alerts.py`, `configs/*.json`) go to exactly one agent per round.
3. **Brief agents fully.** Subagents start cold. Give each: the goal, file paths, the contract it must satisfy, the exact test command, and the "do not touch" list. Tell coders to report changed files, test output and anything skipped.
4. **Verify, don't trust.** After agents return, run the relevant tests/ruff yourself, then send non-trivial changes to `cardio-reviewer`. Only then report done. Report failures with their output.
5. **Scientific interpretation stays with Opus**: what a result means, whether a gate passed, whether to advance. Coders and reviewers report numbers; they do not declare efficacy.

## Council (shared memory)

Agents cannot call each other. They coordinate through `council/BOARD.md` (facts, decisions, per-agent log) under the rules in `council/PROTOCOL.md`; Opus is the hub that reads reports, updates the board, and spawns the next agent. Roles also include `cardio-analyst` (numbers from saved results) and `cardio-writer` (README/video/slide drafts in `council/work/writer/`). Newly added agent files only register after a session restart. Read the board's Decisions before planning.

## Evidence rule: no hallucination

Anything not in this repo — library/API behavior, model checkpoints (e.g. ECGFounder), dataset semantics (IRIDIA-AF, SHDB-AF, LTAF), Kaggle quota/limits, papers — must be **checked on the web in the current session** (`WebSearch`/`WebFetch`, or delegate to `cardio-researcher`) and sourced to a primary source: official docs, PhysioNet pages, the paper or its repo, the vendor's own limits page. Cite the URL. If no strong source is found, say "unverified" and do not build on it. Use `WebSearch` mode `standard` first; escalate to `extended` only for thin or niche results. Repo claims need a file path or command output, not memory.

## Commands

Always run Python from the venv with `OMP_NUM_THREADS=1` set before interpreter start (Torch/XGBoost OpenMP crash on macOS; `tests/conftest.py` also sets it). Add `PYTHONPATH=src` for scripts.

```bash
uv sync --frozen --extra dev --extra research      # add --extra neural only for torch work
OMP_NUM_THREADS=1 .venv/bin/python -m pytest                          # full suite (minutes)
OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_runtime.py::test_alert_persistence_hysteresis_cooldown_and_quality -x   # single test
OMP_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_precursor*.py # one area
.venv/bin/python -m ruff check src tests scripts   # CI lints only src, tests, run_ml_experiments.py, benchmark_beats.py
npm ci --prefix dashboard && npm run build --prefix dashboard   # dev server: npm run dev --prefix dashboard
```

Demo pipeline (synthetic data, software check only): `cardiotwin demo` → `audit` → `train` → `evaluate` → `serve`. See `README.md` for the full command lines. Outputs of `train`, `evaluate` and experiment result JSONs must go to **fresh** paths. `audit` intentionally exits 2 when infeasible; `train` exits 2 on `constraint_not_met`. Check `--help` before using any `scripts/*.py` command; plan docs may name functionality that does not exist yet.

## Architecture

**Online path (runtime, must be causal).** `cli.py` → `service.TwinService` replays each patient's historical ECG on explicit ticks (30 s cadence) → `features.feature_row` + running baseline (updated *after* prediction) → `modeling.Predictor` (joblib; trusted local artifacts only; its config must equal the runtime config) → `alerts.advance_alert` (pure hysteresis/cooldown state machine, dispatches on `config["alert"]["policy_version"]`, e.g. `finite-validity-v1`) → `store.Store` (SQLite, one atomic transaction per commit for cursor+baseline+alert+warning) → `api.create_app` (FastAPI, also serves `dashboard/dist`). Use a single replay worker.

**Offline path (labels and evaluation).** `data.py` (WFDB/metadata, demo generator) → `labels.py` (future rhythm intervals/annotations — **never import from runtime code**) → `audit.py` (cohort feasibility gate) → `train.py` (grouped, patient-level splits; fit/stop/calibration/policy roles disjoint; freezes an artifact dir with manifest + checksums) → `evaluation.py` (replays *complete* chronological streams incl. rejected forecasts, missed events and unknown-outcome warnings; one-to-one warning/event matching; FAR per eligible hour). `config.py`/`configs/*.json` hold the validated contract; `alerts.py` and `evaluation.py` are shared by runtime and offline evaluation, so changes there need the full suite.

**Research layer (large, mostly offline).**
- `investigation.py` + `scripts/run_investigation.py`: IRIDIA-AF cache/ladder/neural/select/freeze/evaluate pipeline over the frozen primary split.
- `precursor.py`, `morphology.py`: the closed linear-precursor and morphology attempts.
- `neural.py`, `neural_architectures.py`: RR-TCN / waveform CNN (optional torch).
- `ecgfounder*.py` (+ `scripts/run_ecgfounder_*.py`, `scripts/kaggle_*.py`): active frozen-encoder study — probe, extraction, tables, runner, compare, study integration; Kaggle entrypoints run in private T4 sessions with bootstrap/time/storage guards.
- Each experiment cycle lives in `reports/ml-investigation/next-cycle/<cycle>/` as hash-bound specs, `state.json`, `progress.md`, receipts. Gates are pre-declared; a failed attempt is preserved, never edited or re-run under the same identity.

## Non-negotiable constraints

- Runtime inputs never use future samples, future annotations, onset/pair IDs. Splits are by patient; all 241 development events stay in continuous evaluation.
- Do not reopen the primary-31 or LTAF cohorts, alter frozen splits/models, or loosen pre-declared gates after seeing results. Protected: the 43 historical evidence bindings, the protected core/probe files listed in `AGENTS.md` and `state.json`, root `README.md`.
- Never claim improvement from synthetic data, telemetry, selected-pair AUC or a Kaggle `COMPLETE` status; require full predictions + review. State what is unmeasured.
- 25,000,000,000-byte cap across `data`, `reports`, `artifacts` (check unique-inode allocated and apparent bytes with `reports/ml-investigation/monitor-storage.py::storage_usage`).
- Kaggle: private only, no automatic retry/resubmit, never print credentials or signed URLs. Reconcile the exact session state before any new job.
- Do not run `/graphify` steps from `AGENTS.md`: no `graphify-out/` exists in this project.

## Skills, plugins, MCPs worth using here

Enabled in `~/.claude/settings.json`; invoke via the Skill tool or name them in briefs.

- **Process (superpowers):** `superpowers:brainstorming` before new features; `writing-plans` → `subagent-driven-development` / `dispatching-parallel-agents` for multi-unit work; `test-driven-development` for coders; `systematic-debugging` for failures; `verification-before-completion` and `requesting-code-review` before declaring done.
- **Review/quality:** `engineering:code-review`, `engineering:testing-strategy`, `engineering:tech-debt`, `/code-review`, `/simplify`; `data:validate-data` and `data:statistical-analysis` to sanity-check evaluation numbers and intervals.
- **Research:** `anthropic-skills:deep-research` for longer literature sweeps; `episodic-memory:remembering-conversations` and `claude-mem:mem-search` to recover earlier decisions instead of re-deriving them.
- **Deliverables (due 18 Oct):** `anthropic-skills:docx` / `pdf` / `pptx` for the submission write-up, slides and the PDFs in `docs/delivery/`; `docs/submission.md` is the checklist.
- **Dashboard (`dashboard/`):** Playwright MCP and `superpowers-chrome` to verify the UI in a browser; `impeccable` / `ui-ux-pro-max` only for UI polish.
- **Token/length style:** `caveman` and `ponytail` plugins shape replies and keep diffs small; they do not change what must be verified.
- **Not useful here, do not call unprompted:** Gmail, Drive, Slack/Asana/Linear and other connectors; cloud/Twilio skills. GitHub MCP needs authorization and there is no git remote.

## Gap audit (17 October 2026; refresh before relying on it)

Evidence is from repo files unless marked. Highest leverage first.

1. **No forecasting gain after the entire effort.** Best reviewed models catch 4–7 of 241 events (≈2–3% sensitivity) at the 0.25 FA/h budget versus the 0.50 goal; waveform CNN AUROC ≈0.458 (`prediction-assessment-20261016.md`). ECGFounder outcome fits = 0; full-run eligibility false; the last Kaggle job failed in pip setup with no data read. Another run cannot be reviewed and written up reliably by the 18 Oct 09:00 IST cutoff. Recommended: treat ECGFounder as a documented engineering/negative result, do not spend the deadline on it, and put the time into the write-up.
2. **Deliverables are stale.** Root `README.md` was last edited 10 Oct and `docs/delivery/*.pdf` predate the precursor, morphology, anchor and ECGFounder cycles; `docs/submission.md` lists team/college/links as pending.
3. **Process outweighs product.** `ecgfounder_study.py` is 2,133 lines and the cycle dir holds hundreds of receipts for zero outcome fits; repeated "quota resume" cycles burn time. Cap any further infrastructure work.
4. **No version control.** Protected-file integrity is enforced with hand-made hash bindings. `git init` would be cheap, but `.gitignore` excludes `data/` and `artifacts/` only; `reports/` is 4.7 GB and would need excluding first.
5. **Storage near the cap:** 23.3 GB allocated of 25 GB (`state.json`), with 719 duplicate-inode paths.
6. **CI lints a subset only.** `checks.yml` lints `src tests scripts/run_ml_experiments.py scripts/benchmark_beats.py`, but `ruff check src tests scripts` passes cleanly today (checked), so CI can widen for free. Dashboard is a single `main.jsx` with no tests; CI only builds it.
7. **`AGENTS.md` is hard to use:** 43 KB, many entries have words fused together (e.g. "schedulechoice"), Codex-specific routing. Keep it as archive; `CLAUDE.md` + `state.json` are the working entry points.
8. **Feature gap worth testing later (new declared question only):** core features are RR/HRV statistics. Earlier work reports atrial premature contraction frequency as a useful paroxysmal-AF predictor (Zong & Mark, PhysioNet/CinC 2001: https://physionet.org/content/challenge-2001/1.0.0/s42-4.htm — **Reported**, snippet only, paper not read). The waveform PAC route failed on detector anchoring, not on the idea; an RR-only ectopy-burden proxy exists in `investigation.py` (~line 198), whether it was ever tested as a precursor is **unverified**. Also check whether the few caught events concentrate in 1–2 patients, and add a sensitivity-vs-FA/h curve to the write-up.
9. **Target may be too hard for this signal.** New-onset 2–10 min ahead at ≤0.25 FA/h is strict. A hospital-scale-hours framing reportedly reaches higher AUROC from Holter HRV (ESC/EHJ 2024 abstract: https://esc365.escardio.org/journal/79408 — **Reported**, different target, not comparable). Frame as a future-work option, not a post-hoc rescue of this study.
10. **External facts to keep unverified until sourced:** Kaggle weekly GPU quota is not stated in docs per search results (account-settings value; ~30 h is community-reported) and the 12 h session limit is third-party-reported. ECGFounder: arXiv 2410.04133 / NEJM AI, single-lead extension claimed, AF-onset forecasting is not a downstream task I found (**Unverified** for this use).

11. **Known failing test:** `tests/test_ecgfounder_probe.py::test_cost_gate_requires_every_overhead_component_and_actual_t4_measurement` fails since 2026-10-17 12:30 UTC because `ecgfounder_probe.py:351` hard-codes that deadline. It is unrelated to other code; everything else passes (984 + 2 new).
