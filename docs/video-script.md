# Twenty-minute demonstration script

Target: one **20:00** screen-and-voice demonstration, uploaded as **unlisted YouTube**. The Unstop listing says "minimum 20 min" and the submission form says "15-20 min" (see [submission checklist](submission.md)); exactly 20:00 satisfies both. Check the exported file's duration, not this schedule.

Production status, 19 October 2026: a separate local export measures exactly 20:00 at 1920×1080 and 30 fps, with synthetic narration, manufactured-data dashboard footage, seven chapters and English captions. Four diagrams were added for the forecasting window, runtime architecture, patient roles and measured outcomes. Frame/layout inspection and a full decode check passed. Full subjective playback and the unlisted YouTube upload/access check remain pending. The checklist below is a reusable recording guide, not evidence that every item was performed. See [media details](video/README.md).

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

"Imagine a clinician reviewing a long ECG recording. Detecting atrial fibrillation after it begins is one question. Our question is harder: could the preceding signal support a warning before a new episode starts? CardioTwin is a research prototype for that question, built for the CRP Digital Twin Challenge by Team Lazarus from MGM's Jawaharlal Nehru Engineering College. It combines historical ECG, permitted patient context, persistent state and a review dashboard. The practical idea is to make a future warning inspectable rather than display an unexplained number. The intended output is a reviewable warning, with evidence and an explicit availability reason.

Here is the result early: the frozen finalist caught six of two hundred and forty-one development events, one of forty-three primary held-out events, and two of twenty-nine external events. These results miss our own fifty percent sensitivity target by a wide margin. We are presenting a functioning research workflow and an honest negative finding. A working interface does not establish useful forecasting. Later, we will explain why even a low false-warning rate needs careful interpretation.

The intended use case matters in India, but we have no Indian recordings or local validation. We therefore make no claim about Indian performance, clinical benefit or deployment. Public retrospective data can help investigate an engineering question; they cannot substitute for evidence from the population and care setting where a tool would actually be used. Clinical review and prospective evaluation would be separate future work.

Our fixed contract uses the preceding ten minutes of ECG, with one decision every thirty seconds, to try to forecast a new AF onset two to ten minutes ahead. It is not a heart-attack or stroke forecast. During this video, manufactured dashboard examples demonstrate the software. Saved real-data evaluations establish the weak results. Keeping those two forms of evidence separate is central to the project."

### 2. Architecture and the two data streams (02:30-05:30)

**On screen:** architecture cards and pipeline diagram; show input, frozen scorer, alert policy, database and UI in order.

**Say:**

"Follow the online path from left to right. The replay service advances historical ECG in thirty-second ticks. At each decision it reads only the preceding ten minutes, extracts features, and applies shared signal-quality checks. It builds the feature row from the previous patient baseline, scores it with frozen model weights, then updates the baseline. An alert state machine applies persistence, hysteresis and cooldown. Cursor, baseline, alert state and warning are committed together in SQLite. FastAPI exposes that state, and React renders it. Restarting the service should continue from the saved cursor rather than repeat a decision. Each patient keeps separate replay state.

The offline path answers a different question: did warnings arrive before real annotated events? Annotation-derived future labels belong in training and evaluation, never among runtime predictor inputs. Patient groups are separated before fitting, calibration and policy selection. Model packages bind their configuration, feature schema and checksums. Complete chronological evaluation then uses the same alert implementation as replay, so our reported policy is tied to executable behavior. A saved test-access ledger records evaluation use; it does not magically certify historical blinding.

The challenge asks for static and dynamic data fusion. The real static stream contains paired age and sex from IRIDIA-AF metadata. We did not invent diagnoses or laboratory values. The dynamic stream contains twenty-three causal ECG features describing heart rate, RR variability and recent changes. The frozen logistic finalist has twenty-seven named inputs: those ECG features, age, sex and their two availability flags. Missing values are handled explicitly. The running personal baseline is persistent twin state, but it is not an input to this finalist. Some exploratory model variants used baseline features; that does not change the finalist's schema. Our ECG-only reference tests whether adding age and sex helped.

That distinction defines our digital twin: a patient-specific historical state and inspectable replay, not a simulated anatomical heart and not continuous model retraining. The code supports more clinical fields than the real cohort supplies. Manufactured profiles exercise parts of the interface without becoming evidence of richer real-world fusion. The comparison found no fusion benefit. We preserve that conclusion alongside the software capability, rather than infer better prediction from the presence of two input streams."

### 3. Live dashboard replay (05:30-10:30)

**On screen:** captured dashboard replay with a persistent Manufactured demo data label. This production does not include the optional real-record replay. Timestamp bullets are editorial anchors; fit narration and captions inside the five-minute segment.

**Say and do:**

- 05:30. "This dashboard footage uses manufactured ECG and manufactured clinical profiles, with a model trained on synthetic examples. It demonstrates software behavior, not patient performance. The normal server refuses the real finalist because it failed acceptance gates. An explicit research mode exists, with a red warning banner, but no real-record replay is shown in this video. The real-data results appear separately in saved reports."
- 06:00. Open the page and select a demo patient. "Start with the patient selector and record-relative clock. The recent ECG panel displays the last ten seconds, while prediction uses the preceding ten minutes. Heart-rate and RR trends summarize earlier decisions. The patient-context panel contains manufactured values here; the real IRIDIA cohort supplies age and sex only. The displayed ECG is downsampled for readability; it is not the exact array used for feature extraction. Displaying fewer points is a presentation choice, while prediction still uses the complete specified trailing signal window. Source sampling rate, signal quality and forecast availability are visible so a reviewer can understand what the score actually rests on."
- 06:45. Click **Start**. "Replay is accelerated: thirty historical seconds advance per wall second at the default setting. This is historical simulation, not a connected wearable or live hospital feed. Early decisions remain unavailable until enough history and prior reliable baseline updates exist. Warm-up is an explicit state, not a missing zero-risk prediction. Watch the clock and availability reason together as the recording advances." Click **Pause**, then **Step** twice. "Pause holds the current position. Step advances one decision, which makes changes easier to inspect. The forecast panel shows an issue time, the future target interval and the frozen model version. Its percentage is a calibrated model score, not an assurance of clinical probability. Quality acceptance alone does not guarantee a forecast: history, warm-up and a loaded compatible model are also required."
- 07:45. Point at baseline, trends and evidence. "The baseline stores earlier reliable feature statistics and updates after prediction. Its movement does not mean the model is learning new weights. For the real finalist it is contextual state, not a predictor input. The evidence table lists large contributions to the raw model score and compares available values with earlier baseline context. These are model explanations, not causes of AF and not treatment recommendations."
- 08:15. Inspect a warning and its review action. "Warnings require persistent elevated scores under the configured policy. The record captures issue time, future interval, model identity and feature evidence. A reviewer can mark it reviewed, dismissed or pending and add a note. That action becomes part of persistent history. Human review does not turn a false warning into a correct prediction, and the software offers no diagnosis or treatment decision."
- 09:00. Show the unreliable demo stretch. "Manufactured patient zero has an intentional unreliable stretch. Here we inspect the quality reason and unavailable forecast together. Abstention means the system declines to issue a usable score; it does not mean the patient is safe. In real-data evaluation, rejected time and events missed during abstention remain visible. Otherwise a model could look better simply by refusing the difficult periods."
- 09:30. Inspect no-warning state. "When no warning is issued, the interface preserves that state without inventing an alert. A low score or quiet warning list is not reassurance. Reset restarts this patient's replay and clears its warning and review history; pause merely stops progression. The saved state is useful for inspection, but this synthetic example cannot establish clinical reliability."
- 09:50. Open Evaluation. "This evaluation screen is still manufactured data. It exercises metric presentation, calibration plots, misses and warning outcomes. Do not read these attractive demo numbers as the project's real-data results. Those are the weak frozen evaluations we show next."
- 10:00. Optional real-record replay remains omitted from this production. "If a later demonstration uses research mode, retain the failed-gates banner, use a development-only record and disclose its in-sample status. Do not present it as fresh evaluation."
- 10:10. Show the real-prefix receipt. "A separate saved real-development-prefix receipt records seventy-two replay steps, median twenty-five milliseconds and p ninety-five twenty-seven milliseconds per step. Future-only signal or annotation changes left earlier outputs identical. That checks causal behavior and local execution cost, not forecasting usefulness or wearable hardware performance."

### 4. Evaluation method (10:30-13:30)

**On screen:** evaluation-flow slide, patient roles, warning/event matching and eligible-hour denominator.

**Say:**

"Our evaluation starts with the unit of independence. Overlapping ECG windows from one person share substantial physiological information. Randomly mixing those windows between training and testing would answer an easier question than generalization to another person. We separate patient groups and preserve their chronological streams, including negative-only recordings. Software success and model usefulness are assessed separately.

Fitting, early-stopping, calibration and policy roles are disjoint where those stages apply. Model weights, calibration and warning policy were frozen before primary and external evaluation. Those cohorts have now been examined and cannot be described as untouched for another tuning cycle. Fresh output directories and source-bound receipts preserve what was run. Future model selection needs development evidence and a fresh confirmation cohort.

We replay whole recordings. Each warning can match at most one eligible event within the inclusive two-to-ten-minute interval, and each event can be matched once. Missed events stay in sensitivity even when every opportunity was unavailable. Current AF, washout and unknown-outcome warnings receive separate accounting. They must not disappear from the report simply because they do not fit the primary assessment question.

The denominator is crucial. Assessed false warnings are divided by labelled, forecast-eligible monitoring hours, including quality-rejected eligible windows. Using all recorded hours would let unknown or excluded time dilute the rate. Coverage is a separate quantity. Event eligibility requires sufficient labelled non-AF history, a minimum thirty-second episode, the target interval and post-AF washout. Report both counts and rates so a small denominator cannot hide behind a percentage.

IRIDIA-AF contains one hundred and sixty-seven recordings from one hundred and fifty-two patients, about six thousand six hundred and ninety monitoring hours. Our development cohort has one hundred and twenty-one patient groups and two hundred and forty-one eligible events; the frozen primary cohort has thirty-one groups and forty-three events. LTAF contributes twenty-nine events across eighty-four recording groups. Cross-record patient identity is unverified there, so recording groups cannot simply be called independent people.

Our operating assumptions are at most zero point two five assessed false warnings per eligible hour and at least fifty percent event sensitivity. These are our research criteria, not a clinical standard or organizer accuracy requirement. Passing one criterion cannot compensate for failing the other."

### 5. Honest results and what failed (13:30-17:00)

**On screen:** finalist results card, chance comparison and closed-attempt summary. Keep counts beside percentages.

**Say:**

"These are the frozen finalist's results. The dashboard examples have ended.

Development caught six of two hundred and forty-one events. The primary held-out cohort caught one of forty-three. LTAF caught two of twenty-nine. Sensitivity is roughly two to seven percent, far below our fifty percent target. Acceptance fails despite assessed false-warning rates below budget. Most events receive no advance warning.

Primary AUROC is zero point six one one, and LTAF AUROC is zero point six three one. Ranking performance summarizes scores across thresholds; it does not tell us how many events this warning policy catches at the operating budget. The low event counts remain the headline.

We independently compared saved score timing with random within-record shifts, without refitting. Development observed six detections against about four point five expected at random timing. The held-out cohorts together observed three of seventy-two against about zero point seven expected. Only three events, intervals including zero, and a post-hoc check cannot establish skill. We claim no demonstrated event-level skill.

LTAF illustrates the denominator problem. There were three hundred and ninety-eight warning starts. Three hundred and ninety-two were unassessable: two hundred and twenty-two during AF, seventy-seven in washout and ninety-three in unknown time. Only four were assessed false warnings. Its zero point zero zero eight assessed rate therefore needs the whole warning inventory beside it. The same warning time placed randomly gives about zero point one zero assessed false warnings per hour.

Fusion also failed to help. The ECG-only reference caught seven development events, compared with six for age, sex and ECG. Development Brier score was zero point zero zero eight three eight four, versus zero point zero zero eight three five eight for the observed constant base-rate reference. A smaller-looking probability error is not sufficient when the reference performs as well.

Other attempts included gradient boosting, personal-baseline features, ectopy-like motifs, RR temporal CNNs catching four and seven development events, a waveform CNN catching four while occupying warning state about thirty-three percent of the time, and precursor ranking catching five. ECGFounder produced no outcome fit; the last cloud job failed during dependency setup before data reads. Engineering progress is not predictive evidence.

A development-fold calibration defect reversed ranking from zero point five seven three to zero point four two seven. Frozen finalists have positive slopes and are unaffected. A later exploratory guard rejects non-positive slopes; it does not repair history or change these saved results.

Published comparison claims use different reporting and omit the operating false-alarm evidence we need. We therefore keep them as context, not proof that our target is achievable. Closed attempts remain preserved, with their failures and limitations."

### 6. Limitations and future validation (17:00-19:00)

**On screen:** limitations and prospective validation plan. Label future work as proposed.

**Say:**

"The limits extend beyond sensitivity. These are retrospective public recordings, not a prospective device study. There is no Indian-population evidence, no secured clinical partner and no treatment advice. LTAF patient independence across records is unverified. Development estimates follow exploration across many candidates, and already examined test cohorts cannot supply a fresh confirmation claim. Clinical metadata are limited, signal preprocessing is source-specific, and RR variability is not automatically validated normal-to-normal heart-rate variability. Local replay latency describes this software and workload; it does not certify battery use, wireless reliability or continuous wearable hardware performance. The dashboard's manufactured profiles must remain visibly separate from real patient evidence.

A future validation program would begin with an ethically approved clinical partner and a clearly specified monitoring setting. Before fitting a new model, declare the target, patient roles, missing-data rules, warning budget and advancement gates. Diagnose measurement and calibration first, then compare against simple references on the same complete timelines. Preserve failed attempts rather than silently change thresholds after seeing results. If a model earns further study, use a fresh prospective silent evaluation: log alerts without showing them to clinicians, then assess events, false warnings, coverage and burden with uncertainty. Only later could a separate study examine whether acting on warnings improves care. None of those stages has been completed here. For this submission, the appropriate action is to preserve the weak result and make the workflow inspectable, reproducible and clear about what remains unmeasured. The acceptance gates remain research assumptions."

Only name a partner or timeline if one really exists.

### 7. Reproducibility and close (19:00-20:00)

**On screen:** canonical repository URL and shipped-versus-omitted checklist. Show the recorded test date.

**Say:**

"The repository shown here is Lazarus, MGM, Jawaharlal Nehru Engineering College, under sumitjadhav1703 on GitHub. It ships source, locked dependencies, tests, selected evaluation reports and submission documents. Raw ECG, model binaries, local patient state and private research receipts are omitted. The recorded public test run on seventeenth October had eight hundred and forty-one passes and eighty-three skips without PyTorch; this is not a fresh test claim. The full canonical repository address stays visible on screen. Data retain their own licences; code is MIT licensed.

CardioTwin remains a research prototype. Its contribution is a causal replay system, explicit evaluation gates and a documented negative forecasting result. A usable interface has not made the model clinically useful. Thank you from Team Lazarus."

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
- [ ] Uploaded as **Unlisted**, not Private or Public.
- [ ] Anonymous check: open the link in a private window of a different browser, signed out of Google. It must play without login or an access request.
- [ ] Audio audible and UI readable at 720p and 1080p.
- [ ] Verified unlisted link placed in README and in the submission form.

Segment 3 shows manufactured data and a synthetic-trained model by default. The optional real-record replay is omitted from this production. Any later use requires the opt-in research mode, a visible red banner and an explicitly in-sample development record. The default server still refuses the failed-gate model. Do not change that to make the demo look better.

## Editorial evidence and timing notes

Narration is grounded in the current local public snapshot: `README.md` (model schema, frozen results, chance re-analysis, limits and recorded 17 October tests), `docs/submission.md` (replay proof scope), `src/cardiotwin/{service,features,modeling,alerts,evaluation}.py` (runtime and evaluation semantics), and `scripts/prepare_single_record_replay.py` (development-only helper with `--split-file`). No fresh external claims or fresh test results are introduced. The canonical repository URL supplied for this revision is https://github.com/sumitjadhav1703/Lazarus_MGM_Jawaharlal_Nehru_Engineering_College; verify final anonymous access separately.

All seven scheduled segments total exactly 1,200 seconds. The optional 10:00 bullet is excluded from narration and visuals. Keep paragraph and mandatory quote-unit counts unchanged for the current visual mapping. Aim for natural 135–155 words per minute, with short pauses to inspect on-screen evidence. Final export duration, intelligibility, caption synchronization and residual silence require measurement by the video builder.
