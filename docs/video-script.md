# Twenty-minute demonstration script

Target: one **20:00** screen-and-voice demonstration, uploaded as **unlisted YouTube**. The Unstop listing says "minimum 20 min" and the submission form says "15-20 min" (see [submission checklist](submission.md)); exactly 20:00 satisfies both. Check the exported file's duration, not this schedule.

Production status, 19 October 2026: a separate local export measures exactly 20:00 at 1920×1080 and 30 fps, with locally generated Kokoro am_michael narration, manufactured-data dashboard footage, seven chapters and English captions. The 19 October voice revision uses more conversational wording and measured short speech chunks; the earlier exports are preserved. Four diagrams were added for the forecasting window, runtime architecture, patient roles and measured outcomes. Frame/layout inspection and a full decode check passed. [Watch the Unlisted CardioTwin demo](https://youtu.be/4CJFCPdKEto) was uploaded with English captions and seven chapters on 19 October; anonymous playback was verified. Full subjective playback remains pending. The checklist below is a reusable recording guide, not evidence that every item was performed. See [media details](video/README.md).

## Ground rules for the narrator

- Say "research prototype" in the first minute and the last minute.
- Never say "accurate", "validated", "clinically ready" or "predicts AF". Say "tries to forecast" or "forecasts weakly".
- Show the weak results as the headline, not as a footnote.
- The default dashboard runs on **manufactured demo data** with a model trained on that synthetic data. Say so on camera each time it is on screen. Real-data results come from saved evaluation files, never from the dashboard. If you also show research mode (segment 3), say the record is in-sample and that this shows the model runs, not that it forecasts well.
- Do not present synthetic or SHDB-AF numbers as results. SHDB-AF gets one sentence at most (an earlier cohort that left zero eligible events).
- Do not say anything about Indian performance. We have no Indian data.

## Demo commands (run before recording; fresh paths)

Use a clean clone. Training and result paths must not already exist. On macOS set `OMP_NUM_THREADS=1`.

```bash
uv sync --frozen --extra dev --extra research
npm ci --prefix dashboard
npm run build --prefix dashboard

uv run cardiotwin demo --output data/demo
uv run cardiotwin audit --data-dir data/demo --output data/ml-audit
uv run python scripts/run_ml_experiments.py --data-dir data/demo --audit data/ml-audit \
  --artifacts data/ml-artifacts --output data/ml-experiments/development-ablations.json \
  --synthetic --false-warning-budget 1 --minimum-sensitivity 0.5
# open the JSON and read recommended_artifact (it was patient_class when last run)
uv run cardiotwin evaluate --data-dir data/demo --audit data/ml-audit \
  --model data/ml-artifacts/patient_class --output data/ml-evaluation
```

These `data/` paths are gitignored, unlike the earlier `reports/ml-v2-*` paths. Rehearse them once from a clean clone; they have not been re-run exactly as written.

On camera in segment 3 (start the server before the segment, or type it live):

```bash
uv run cardiotwin serve --data-dir data/demo --model data/ml-artifacts/patient_class \
  --metrics data/ml-evaluation/evaluation.json --db data/state.sqlite
# slower playback: add --playback-seconds 2
```

Optional, only if prepared and rehearsed: the real frozen finalist on one real development record, in research mode. This needs IRIDIA-AF downloaded from Zenodo 8405941 and the frozen model directory, neither of which is shipped in the repository.

```bash
uv run python scripts/prepare_single_record_replay.py --source <IRIDIA data dir> --record <record id> --split-file <primary-split.json> --output data/replay-one
uv run cardiotwin serve --research-model --data-dir data/replay-one --model <frozen model dir> --db data/research-state.sqlite
```

The dashboard then shows a red banner: the model FAILED its acceptance gates and is not for clinical use; records from its development cohort are in-sample. Check `--help` of the helper script first.

Open http://127.0.0.1:8000. Use a fresh `--db` path if you change the model or config. Demo patient 000 has an intentional unreliable segment for the abstention demo.

## Schedule

| Time | Segment |
| --- | --- |
| 00:00-02:30 | 1. Problem and India relevance |
| 02:30-05:30 | 2. Architecture and the two data streams |
| 05:30-10:30 | 3. Live dashboard replay (manufactured data) |
| 10:30-13:30 | 4. Evaluation method |
| 13:30-17:00 | 5. Honest results and what failed |
| 17:00-19:00 | 6. Limitations and future validation |
| 19:00-20:00 | 7. Reproducibility and close |

Total 20:00 (2:30 + 3:00 + 5:00 + 3:00 + 3:30 + 2:00 + 1:00).

### 1. Problem and India relevance (00:00-02:30)

**On screen:** Team Lazarus title, concrete monitoring use case, target interval, then weak real-data headline. India sources remain contextual only.

**Say:**

"Reviewing a long historical ECG recording tells us what happened to a patient's rhythm. CardioTwin asks whether the earlier signal could also support a warning before a new atrial fibrillation episode begins. This is a research prototype, built for the CRP Digital Twin Challenge by Team Lazarus from MGM's Jawaharlal Nehru Engineering College. We combine historical ECG, available patient context, saved patient state and a dashboard for review. The practical aim is to give a human reviewer more than a number: when the warning was issued, which future interval it covers, and what evidence supported it. The dashboard footage you'll see uses manufactured data. We'll keep that software demonstration separate from the real results throughout the video.

Let's put the main result up front. The frozen finalist caught six of two hundred and forty-one development events, one of forty-three primary held-out events, and two of twenty-nine external events. That's far below our own fifty percent sensitivity target. Most events had no advance warning. So we're presenting a working research workflow with a negative forecasting result. The interface makes the workflow inspectable; it doesn't make the predictions useful. We'll also look at why a low assessed false-warning rate can be misleading when many warnings fall outside assessable time.

The intended use case matters in India, but our study contains no Indian recordings and no local validation. We can't infer Indian performance or clinical benefit from these results. Public retrospective recordings let us investigate the pipeline, check causal behavior and measure warnings against saved annotations. They don't tell us how the system would perform in a different population or care setting. Clinical review and prospective evaluation would each need their own evidence before any deployment claim.

Here's the fixed task. At each decision, the system uses the preceding ten minutes of ECG. It makes a new decision every thirty seconds and tries to warn about a new AF onset two to ten minutes ahead. We're forecasting AF onset, not heart attack or stroke. In the video, manufactured dashboard examples show how the software behaves. Separate saved evaluations show what happened on real recordings. That separation matters whenever an attractive demo screen appears beside a research result."

### 2. Architecture and the two data streams (02:30-05:30)

**On screen:** architecture cards and pipeline diagram; show input, frozen scorer, alert policy, database and UI in order.

**Say:**

"Follow the complete online path from left to right. The replay service advances historical ECG in thirty-second ticks. For each decision, it reads the preceding ten minutes, extracts features and checks signal quality. The previous baseline is available as patient context; the frozen finalist scores its own specified inputs. Only after prediction does the baseline update. The warning policy checks persistence, hysteresis and cooldown before starting another warning. SQLite saves the cursor, baseline, alert state and warning together. FastAPI exposes the saved state, and React displays it. After a restart, replay should continue from that cursor without repeating the decision. Each patient has separate state, so inspecting one recording doesn't move another patient's clock.

The separate offline evaluation path checks whether warnings actually arrived before annotated events. It needs future rhythm labels to answer that question, but those labels never belong in runtime inputs. We separate patient groups before fitting, calibration and policy selection. A model package carries its configuration, feature schema and checksums, so the score has an identifiable source. Evaluation then replays complete timelines using the same warning implementation as the service. That connects the reported policy to executable behavior. A test-access ledger records when evaluations were used. It's a record of access, not proof that every historical decision was made blind.

The challenge specification asks us to combine static patient information with a changing signal. In the real IRIDIA-AF cohort, the static information is paired age and sex. We haven't filled the gaps with invented diagnoses or laboratory results. The dynamic stream supplies twenty-three causal ECG features covering heart rate, RR variability and recent changes. The frozen logistic finalist has twenty-seven named inputs: those twenty-three features, age, sex and two availability flags. Missing values and their availability are handled explicitly. The running personal baseline is saved twin state, but it's not one of these predictor inputs. Some exploratory variants did use baseline features; their inputs shouldn't be confused with the finalist's schema. An ECG-only reference gives us a direct check on whether adding age and sex helped.

That's what digital twin means in this project: saved patient-specific history and a replay a reviewer can inspect. We aren't simulating an anatomical heart or continually retraining the model. The code can accept more clinical fields than our real cohort provides. Manufactured profiles let us exercise those interface fields, but they don't establish richer fusion on real patients. In the actual comparison, adding the available patient context showed no fusion benefit. We report that outcome alongside the software capability, because connecting two streams alone doesn't demonstrate better forecasting."

### 3. Live dashboard replay (05:30-10:30)

**On screen:** captured dashboard replay with a persistent Manufactured demo data label. This production does not include the optional real-record replay. Timestamp bullets are editorial anchors; fit narration and captions inside the five-minute segment.

**Say and do:**

- 05:30. "We're now moving into the dashboard. Both the ECG and clinical profiles in this footage are manufactured, and its model was trained on synthetic examples. This shows software behavior, not patient performance. The normal server refuses to serve the real finalist because it failed acceptance gates. There's an explicit research mode with a red warning banner, but this video doesn't replay a real record. We'll show the real-data findings separately from saved reports after the demonstration."
- 06:00. Open the page and select a demo patient. "Start with the patient selector and the clock. This clock belongs to the recording; it isn't the current wall time. The ECG panel shows the most recent ten seconds, while the predictor uses the preceding ten minutes. Heart-rate and RR trends summarize earlier decisions. The patient-context values here are manufactured; real IRIDIA data provide age and sex only. We also downsample the displayed ECG so the trace stays readable. That display isn't the exact array used for feature extraction, and fewer points on screen don't shorten the prediction window. Source sampling rate, signal quality and forecast availability are visible beside the result. Together, they help a reviewer check what the displayed score rests on before interpreting it."
- 06:45. Click **Start**. "Replay runs faster than real time. At the default setting, each wall second advances thirty seconds of historical recording. This is a simulation of saved data, not a wearable connection or a live hospital feed. Early decisions can remain unavailable while the system collects enough history and prior reliable baseline updates. That warm-up has an explicit reason. It isn't a zero-risk estimate. Watch the recording clock and the availability message together as replay progresses." Click **Pause**, then **Step** twice. "Pause holds the replay position. Step moves forward by one decision, so changes are easier to inspect. The forecast panel records an issue time, its future target interval and the frozen model version. The percentage is a calibrated model score, not a clinically validated probability. Passing the quality check is only one requirement. The system also needs enough history, completed warm-up and a loaded model compatible with the runtime configuration before it can provide a forecast."
- 07:45. Point at baseline, trends and evidence. "The baseline summarizes earlier reliable feature values and updates after prediction. When it moves, that doesn't mean the model has learned new weights. For the real finalist, it's contextual patient state rather than a predictor input. The evidence table shows large contributions to the raw model score and compares available values with earlier baseline context. This helps explain how the model reached its score. It doesn't identify causes of AF or recommend a treatment for the patient."
- 08:15. Inspect a warning and its review action. "A warning needs persistently elevated scores under the configured policy. Its saved record includes the issue time, future interval, model identity and feature evidence. The reviewer can mark it reviewed, dismissed or pending, and add a note. Those actions stay in history, making review traceable. They don't change whether the warning predicted an event correctly. A dismissed false warning remains a false warning in evaluation. The software doesn't make a diagnosis or choose treatment."
- 09:00. Show the unreliable demo stretch. "Patient zero has an intentionally unreliable stretch in this manufactured ECG. Look at the quality reason beside the unavailable forecast. The system declines to score this decision; it isn't saying the patient is safe. In real evaluation, rejected time stays visible. An event missed during abstention still counts as missed. Refusing difficult periods must not make missed events disappear."
- 09:30. Inspect no-warning state. "Sometimes the warning list is quiet. The interface preserves that state without creating an alert, but silence doesn't mean the patient is safe. Pause only stops replay from progressing. Reset goes further: it restarts this patient's replay and clears their warning and review history. Those controls have different purposes. Saved state helps review, while this manufactured example tells us nothing about clinical reliability."
- 09:50. Open Evaluation. "This screen also uses made-up data. It shows metrics, calibration plots, misses and warning outcomes. These aren't the project's real results. They're examples that test the screen. Next, we'll leave the demo and look at the weak frozen results."
- 10:00. Optional real-record replay remains omitted from this production. "If a later demonstration uses research mode, retain the failed-gates banner, use a development-only record and disclose its in-sample status. Do not present it as fresh evaluation."
- 10:10. Show the real-prefix receipt. "A saved real development-record prefix has seventy-two replay steps: median twenty-five milliseconds, ninety-fifth percentile twenty-seven milliseconds per step. Changing only future signal or labels left earlier outputs unchanged. These checks support causal behavior and show local run cost. They don't prove useful forecasting or wearable speed."

### 4. Evaluation method (10:30-13:30)

**On screen:** evaluation-flow slide, patient roles, warning/event matching and eligible-hour denominator.

**Say:**

"First, we keep people apart. Overlapping windows from one person's ECG share a lot of information. Randomly mixing them between training and testing could make the task look easier. We split patient groups and keep whole timelines, including records with no eligible events. That measures warnings across monitoring time. Correct software and useful advance warning are separate tests.

Where needed, fitting, early stopping, calibration and warning-policy choice use separate roles. We froze weights, calibration and policy before primary and external tests. We've now seen those cohorts, so a new tuning cycle can't call them untouched. Fresh output folders and source-bound receipts preserve each run. Future model choice needs development evidence, then a fresh cohort to confirm it. An already seen test set can't do that.

We test whole records, not selected windows. A warning can match at most one eligible event, two to ten minutes ahead, including both ends. An event can be matched once. If all chances to warn were unavailable, the event still counts as missed. Warnings during AF, post-AF washout or unknown-outcome time are counted separately. Those parts of the timeline stay in the report.

The denominator matters too. We divide assessed false warnings by labelled, forecast-eligible hours, including quality-rejected eligible windows. Using all recorded hours would let unknown or excluded time dilute the rate. Coverage is separate. Events need enough labelled non-AF history, an episode of at least thirty seconds, the target interval and post-AF washout. We report counts with rates, so readers can see which hours and warnings were assessed.

IRIDIA-AF has one hundred and sixty-seven recordings from one hundred and fifty-two patients: about six thousand six hundred and ninety hours. Development has one hundred and twenty-one patient groups and two hundred and forty-one eligible events. The frozen primary cohort has thirty-one groups and forty-three events. LTAF adds twenty-nine events across eighty-four recording groups. We haven't verified patient identity across its records, so those groups can't simply be called independent people.

Our limits are at most zero point two five assessed false warnings per eligible hour and at least fifty percent event sensitivity. These are research assumptions, not clinical standards or an organizer's required accuracy. Both must pass. Few false warnings can't make up for missing most events."

### 5. Honest results and what failed (13:30-17:00)

**On screen:** finalist results card, chance comparison and closed-attempt summary. Keep counts beside percentages.

**Say:**

"The synthetic demonstration has ended. These are the frozen finalist's saved results on real recordings.

Development caught six of two hundred and forty-one events. The primary held-out cohort caught one of forty-three. LTAF caught two of twenty-nine. That's about two to seven percent sensitivity, well below fifty percent. Acceptance fails despite assessed false-warning rates below budget. Most events had no advance warning.

Primary AUROC was zero point six one one; LTAF AUROC was zero point six three one. This measures score ranking across thresholds, not events caught by the chosen policy within its false-warning budget. The few events caught remain the main result.

We compared saved score timing with random shifts within each record, without refitting. Development caught six events, versus about four point five expected at random. The held-out cohorts together caught three of seventy-two, versus about zero point seven expected. Three events are little evidence. The intervals include zero, and this was a post-hoc check. We claim no demonstrated event-level skill from it.

LTAF had three hundred and ninety-eight warning starts. Three hundred and ninety-two couldn't be assessed: two hundred and twenty-two during AF, seventy-seven in washout and ninety-three in unknown time. Only four were assessed false warnings. The zero point zero zero eight assessed rate needs that full inventory beside it. Randomly placing the same warning time gives about zero point one zero assessed false warnings per hour. A low assessed rate alone doesn't show skill.

Fusion didn't help. ECG alone caught seven development events, versus six for age, sex and ECG. The finalist's development Brier score was zero point zero zero eight three eight four. The observed constant base-rate reference scored zero point zero zero eight three five eight. Lower is better, but the simple reference did as well. The small error doesn't show forecasting value.

Other tests used gradient boosting, personal-baseline features, ectopy-like motifs and RR temporal CNNs. The RR models caught four and seven development events. A waveform CNN caught four, with about thirty-three percent warning-state occupancy. Precursor ranking caught five. ECGFounder had zero outcome fits. Its last cloud job failed in dependency setup, before data reads. Progress on an encoder doesn't show better forecasting.

A development-fold calibration defect reversed ranking from zero point five seven three to zero point four two seven. The frozen finalists have positive slopes and are unaffected. A later exploratory guard rejects non-positive slopes. It doesn't repair history or change these saved results.

Published comparisons use different reporting and omit the operating false-alarm evidence we need here. We keep them as context rather than proof that our target is achievable. Closed attempts retain their failures and limitations, so a later reader can see what was actually tested."

### 6. Limitations and future validation (17:00-19:00)

**On screen:** limitations and prospective validation plan. Label future work as proposed.

**Say:**

"The limits go beyond missed events. These are retrospective public records, not a prospective device study. We have no Indian-population evidence, secured clinical partner or treatment advice. Patient identity across LTAF records is unverified. Development results follow tests of many candidates. Already seen test cohorts can't provide fresh confirmation. Clinical data are limited, and signal preprocessing varies by source. RR variability isn't automatically validated normal-to-normal heart-rate variability. Local replay speed describes this software and workload, not battery use, wireless reliability or continuous wearable performance. The dashboard's made-up profiles must stay separate from real patient evidence. These limits define what the prototype shows and what still needs another study.

A future validation program would start with an ethically approved clinical partner and a clearly specified monitoring setting. Before fitting another model, we'd declare the target, patient roles, missing-data rules, warning budget and advancement gates. Measurement and calibration need diagnosis first. Then we'd compare a candidate with simple references on the same complete timelines, so every method faces the same warning task. Failed attempts would remain saved, with thresholds unchanged after seeing their results. If a model earned further study, the next step would be a fresh prospective silent evaluation: log alerts without showing them to clinicians, then assess events, false warnings, coverage and burden, including uncertainty. A separate later study could ask whether acting on warnings improves care. We haven't completed those stages. For this submission, we preserve the weak forecasting result and make the workflow inspectable, reproducible and clear about what remains unmeasured. The acceptance gates remain research assumptions throughout that process."

Only name a partner or timeline if one really exists.

### 7. Reproducibility and close (19:00-20:00)

**On screen:** canonical repository URL and shipped-versus-omitted checklist. Show the recorded test date.

**Say:**

"The public GitHub repository is Lazarus, MGM, Jawaharlal Nehru Engineering College, under sumitjadhav1703. It includes source, locked dependencies, tests, selected reports and submission documents. Raw ECG, model binaries, local patient state and private receipts are omitted. The recorded seventeenth October public run had eight hundred and forty-one passes and eighty-three skips without PyTorch. That's a historical run, not a fresh test claim. The full repository address stays on screen. Data retain their own licences; code is MIT licensed.

CardioTwin is a research prototype. This video's dashboard footage is synthetic. Its contribution is causal replay, clear evaluation gates and a documented negative forecasting result. A usable interface hasn't made the model clinically useful. Thank you from Team Lazarus."

## Results card (show during segment 5; finalist only)

| Cohort | Detected | Sensitivity [95% CI] | Assessed false warnings / eligible h | All warning starts | AUROC |
| --- | --- | --- | --- | --- | --- |
| Development | 6 / 241 | 2.49% [0.44, 4.96] | 0.171 | 761 | pooled 0.453 |
| Primary held-out | 1 / 43 | 2.33% [0.00, 7.89] | 0.085 | 113 | 0.611 |
| LTAF external | 2 / 29 | 6.90% [0.00, 16.67] | 0.008 | 398 | 0.631 |

Reference (ECG-only): 7/241 development, 0/43 primary, 0/29 LTAF. Source: the investigation report and the saved evaluation JSONs under `reports/ml-investigation/`; the chance comparison is an independent re-analysis of saved predictions (no refit, no new held-out access).

## Pre-recording checklist

Setup
- [ ] Clean clone; demo commands above run to completion with fresh output paths.
- [ ] Screen recorded at 1920x1080 or higher; ECG text readable at 720p; dashboard `dist` built.
- [ ] Notifications off. Hide bookmarks, other tabs, email, chat and the dock.
- [ ] No personal data on screen: no personal names in terminal paths (use a neutral working directory), no tokens, API keys, cookies or signed URLs, no phone numbers or emails.
- [ ] Add team and college on screen only once the real values are supplied.
- [ ] Microphone test; no background noise; steady pace.

Content
- [ ] "Research prototype" said in the first and last minute.
- [ ] "Manufactured demo data" said and shown every time the default dashboard is on screen; the research-mode banner stays visible if shown.
- [ ] Results card and chance slides match the numbers in this file.
- [ ] No claim of clinical readiness, accuracy, Indian validation or fusion benefit.

After recording
- [ ] Exported duration is exactly 20:00 (check the file, then the YouTube player).
- [x] Uploaded as **Unlisted**: [Watch the Unlisted CardioTwin demo](https://youtu.be/4CJFCPdKEto).
- [ ] Anonymous check: open the link in a private window of a different browser, signed out of Google. It must play without login or an access request.
- [ ] Audio audible and UI readable at 720p and 1080p.
- [x] Verified Unlisted link placed in README and delivery documents.
- [ ] Repository link entered and final submission confirmed in Unstop.

Segment 3 shows manufactured data and a synthetic-trained model by default. The optional real-record replay is omitted from this production. Any later use requires the opt-in research mode, a visible red banner and an explicitly in-sample development record. The default server still refuses the failed-gate model. Do not change that to make the demo look better.

## Editorial evidence and timing notes

Narration is grounded in the current local public snapshot: `README.md` (model schema, frozen results, chance re-analysis, limits and recorded 17 October tests), `docs/submission.md` (replay proof scope), `src/cardiotwin/{service,features,modeling,alerts,evaluation}.py` (runtime and evaluation semantics), and `scripts/prepare_single_record_replay.py` (development-only helper with `--split-file`). No fresh external claims or fresh test results are introduced. The canonical repository URL supplied for this revision is https://github.com/sumitjadhav1703/Lazarus_MGM_Jawaharlal_Nehru_Engineering_College; verify final anonymous access separately.

All seven scheduled segments total exactly 1,200 seconds. The optional 10:00 bullet is excluded from narration and visuals. Keep paragraph and mandatory quote-unit counts unchanged for the current visual mapping. Aim for natural 135–155 words per minute, with short pauses to inspect on-screen evidence. Final export duration, intelligibility, caption synchronization and residual silence require measurement by the video builder.
