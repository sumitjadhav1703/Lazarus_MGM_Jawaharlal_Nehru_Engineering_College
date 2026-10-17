# Twenty-minute demonstration script

Target: one real **20:00** screen-and-voice recording, uploaded as **unlisted YouTube**. The Unstop listing says "minimum 20 min" and the submission form says "15-20 min" (see [submission checklist](submission.md)); exactly 20:00 satisfies both. Check the exported file's duration, not this schedule.

This file is a script, not the recording. The recording, upload and public link are still to be done by the team.

## Ground rules for the narrator

- Say "research prototype" in the first minute and the last minute.
- Never say "accurate", "validated", "clinically ready" or "predicts AF". Say "tries to forecast" or "forecasts weakly".
- Show the weak results as the headline, not as a footnote.
- The dashboard runs on **manufactured demo data**. Say so on camera each time it is on screen. Real-data results come from saved evaluation files, never from the dashboard.
- Do not present synthetic or SHDB-AF numbers as results. SHDB-AF gets one sentence at most (an earlier cohort that left zero eligible events).
- Do not say anything about Indian performance. We have no Indian data.

## Demo commands (run before recording; fresh paths)

Use a clean clone. Training and result paths must not already exist. On macOS set `OMP_NUM_THREADS=1`.

```bash
uv sync --frozen --extra dev --extra research
npm ci --prefix dashboard
npm run build --prefix dashboard

uv run cardiotwin demo --output data/demo
uv run cardiotwin audit --data-dir data/demo --output reports/ml-v2-audit
uv run python scripts/run_ml_experiments.py --data-dir data/demo --audit reports/ml-v2-audit \
  --artifacts artifacts/ml-v2-reviewed --output reports/ml-improvements/development-ablations.json \
  --synthetic --false-warning-budget 1 --minimum-sensitivity 0.5
# open the JSON and read recommended_artifact (it was patient_class when last run)
uv run cardiotwin evaluate --data-dir data/demo --audit reports/ml-v2-audit \
  --model artifacts/ml-v2-reviewed/patient_class --output reports/ml-v2-evaluation
```

On camera in segment 3 (start the server before the segment, or type it live):

```bash
uv run cardiotwin serve --data-dir data/demo --model artifacts/ml-v2-reviewed/patient_class \
  --metrics reports/ml-v2-evaluation/evaluation.json --db data/ml-v2-state.sqlite
# slower playback: add --playback-seconds 2
```

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

**On screen:** title slide with `<<TEAM NAME>>` and `<<COLLEGE NAME>>` (add once supplied), then the problem slide with the four India sources.

**Say:**

"This is CardioTwin, a research prototype for the CRP Digital Twin Challenge. We asked one narrow question. Can a wearable-style ECG stream, plus basic clinical history, warn a clinician a few minutes before a new episode of atrial fibrillation?

The exact target is a new AF onset between two and ten minutes ahead. The input is the last ten minutes of ECG, and we decide every thirty seconds. It is not heart attack, and it is not stroke.

Why India? National AF data are lacking. Community surveys give very different numbers. A door-to-door survey of 4,281 adults in rural Andhra Pradesh found AF in 0.32 percent, rising to 3.5 percent above age 75. A small screening of 354 people over 50 in rural Gujarat found 5.1 percent. India's ischemic stroke burden in the 2019 Global Burden of Disease estimate was about 535,700 new cases. We do not claim how much of that is AF. And in the Kerala-AF registry of 3,421 patients, only about 5 to 6 percent of the rate and rhythm cohort were on newer oral anticoagulants. These sources motivate the question. They do not show that our tool would help.

Everything you see today is a research prototype on public retrospective data. It has not been tested on any Indian patient, and the forecasting results are weak. We will show you that first."

### 2. Architecture and the two data streams (02:30-05:30)

**On screen:** `docs/delivery/architecture.pdf`, then the README architecture diagram. Point at each box as you name it.

**Say:**

"The design has two paths. The online path is what runs at replay time, and it is strictly causal. A replay service feeds each patient's historical ECG in thirty-second ticks. Features are computed from the trailing ECG only. A per-patient running baseline is updated after the prediction, never before. The frozen model scores the row, an alert state machine applies persistence, hysteresis and cooldown, and the result is committed to SQLite in one atomic transaction. FastAPI serves it to the React dashboard.

The offline path builds labels and evaluates. Labels code is never imported by the runtime. Patient-level training, freezing with checksums, then complete chronological replay and event-level scoring.

The challenge asks for fusion of static health records and dynamic sensor data. We have two streams. The static stream is paired age and sex from the IRIDIA-AF record. We did not invent diagnoses or lab values. The dynamic stream is causal ECG features plus the patient baseline. Both enter one logistic regression. That is our frozen finalist. We keep a separate ECG-only reference so we can ask whether the clinical inputs helped. As you will see, they did not.

The alert and evaluation code is shared between the runtime and offline scoring, so the policy we score is the policy that runs."

### 3. Live dashboard replay (05:30-10:30)

**On screen:** terminal with the `serve` command, then http://127.0.0.1:8000. Keep a corner label "Manufactured demo data" visible.

**Say and do** (adjust wording to the actual UI labels during rehearsal):

- 05:30. "First, an important disclosure. This dashboard runs on a manufactured demo cohort that our code generates. It is a software demonstration. It is not patient data and not evidence. The real-data model that failed our acceptance gate is deliberately not served here; the normal serving path refuses it, and we have not changed that."
- 06:00. Open the page. Select a demo patient. "This is the doctor-facing view: ECG trace, trends, the patient's baseline, signal quality and the risk panel."
- 06:45. Click **Start**. "Playback is accelerated, thirty historical seconds per wall second. Each tick is a forecast built only from the previous ten minutes." Click **Pause**, then **Step** twice. "Step advances one decision."
- 07:45. Point at the baseline and trend panels. "The baseline evolves as state. It is updated after each prediction."
- 08:15. When a warning appears, open the warning history and review timeline, and click the clinician review action on one warning. "A clinician can review a warning, and that is stored with the warning history."
- 09:00. Select demo patient 000 and run to its unreliable segment. "This patient has an intentional bad stretch. The system abstains instead of forecasting. Abstention is part of the design."
- 09:30. "When the model does not produce a warning, the UI does not invent one."
- 09:50. Open the evaluation view. "This is the demo cohort's evaluation file. Again, manufactured."
- 10:10. Show the replay-receipt slide. "On real IRIDIA development prefixes, a separate replay receipt records 72 steps, a median of 25 milliseconds and a p95 of 27 milliseconds per step. It also shows that changing only the future ECG or annotations leaves all earlier outputs identical. That is a causality check, not a performance claim."

### 4. Evaluation method (10:30-13:30)

**On screen:** the evaluation-flow slide, then the Methodology section of the README.

**Say:**

"This is where most of our effort went, because easy evaluations flatter weak models.

We split by patient, never by window. Fitting, early-stopping, calibration and policy groups are disjoint. We froze the model, calibration and alert policy before the held-out cohorts were opened. Those cohorts are now examined, so they cannot be used for tuning again.

We replay whole recordings in time order. Quality-rejected stretches, missed events and time with no label stay in. One warning can match at most one event, using the inclusive two-to-ten-minute window. False warnings are counted per eligible hour. Warnings that land during AF, in post-AF washout or in unknown time are reported separately, not hidden.

The data: the IRIDIA-AF database has 167 recordings from 152 patients, about 6,690 monitoring hours. Our development set is 121 patient groups with 241 events. The frozen primary held-out set is 31 groups with 43 events. LTAF is an external set with 29 events across 84 recording groups. For LTAF, patient identity across records is unverified.

Our own criteria are at most 0.25 false warnings per eligible hour and 50 percent sensitivity. We set those ourselves. The organizers published no accuracy threshold."

### 5. Honest results and what failed (13:30-17:00)

**On screen:** the results card (below), then the chance-comparison slide, then the failures slide.

**Say:**

"Here are the results for the frozen finalist.

On development, 6 of 241 events detected. On the primary held-out set, 1 of 43. On LTAF, 2 of 29. That is two to seven percent sensitivity, against our 50 percent target. The gate fails.

Held-out AUROC is 0.611 on the primary set and 0.631 on LTAF. That is a weak ranking signal.

Then we checked against chance. We shifted the saved risk scores to random times within each recording and re-applied the same alert policy. In development we would expect about 4.5 detections at random timing, and we observed 6. That is indistinguishable. On the held-out sets, 3 of 72 events versus about 0.7 expected. That looks better, but it is three events, the intervals include zero, and it was a post-hoc check. We make no claim of skill.

LTAF shows only four assessed false warnings, which looks great. It is not skill. Of 398 warning starts, 392 fell in time that cannot be assessed: 222 during AF, 77 in washout, 93 unknown. Placing the same warning time at random would give about 0.10 assessed false warnings per hour, not 0.008.

The clinical inputs did not help. The ECG-only reference caught 7 of 241, the fused model 6. Probabilities were no better than a constant base rate in development: Brier 0.008384 against 0.008358.

What else did not work. Gradient boosting, a per-patient baseline, ectopy-like motifs, an RR temporal CNN at 4 and 7 of 241, a waveform CNN with a causal temporal head at 4 of 241 but warning 33 percent of the time, and a precursor ranking at 5 of 241. A frozen ECG foundation model, ECGFounder, never produced an outcome fit: our last cloud job failed in dependency setup before reading data.

We also found a calibration defect. On development fold 0 the calibrators learned negative slopes and reversed the ranking, from 0.573 to 0.427. It does not affect the frozen held-out models, and we added a guard in the training code, for exploratory use only.

For context, the closest published design on the same database reports an AUROC of 0.657 to 0.680 in an abstract, without false-alarm rates. We did not find any source showing this kind of warning is achievable at our false-alarm budget."

### 6. Limitations and future validation (17:00-19:00)

**On screen:** limits slide, then validation-plan slide.

**Say:**

"The limits. This is retrospective public data. We have no Indian-population evidence. The LTAF groups are recording IDs, and patient independence is unverified. The held-out cohorts are used up for this model family, and our development numbers come from choosing among many candidates, so they are exploratory. This is not a medical device and gives no treatment advice.

The path to Indian validation is a plan, not something we have started. First, an ethically approved clinical partner. We do not have one secured. Second, a prospective silent evaluation, where alerts are logged but not shown to clinicians, at a pre-registered operating point. Third, fresh cohorts and pre-declared gates for any new model. Until then, we do not claim Indian performance."

Only name a partner or timeline if one really exists.

### 7. Reproducibility and close (19:00-20:00)

**On screen:** the public repository page and the README top box. Repository: `<<GITHUB REPOSITORY URL>>`.

**Say:**

"Everything is in the public repository. Dependencies are locked. The code is MIT licensed, and the datasets keep their own licenses. In the public repository the full test run is 836 passed, 83 skipped and none failed; the skipped tests need private research receipts or the optional PyTorch install.

CardioTwin is a research prototype. Its contribution is a leak-controlled, strictly scored evaluation pipeline and an honest negative result, not a working predictor. Thank you. Team `<<TEAM NAME>>`, `<<COLLEGE NAME>>`."

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
- [ ] "Manufactured demo data" said and shown every time the dashboard is on screen.
- [ ] Results card and chance slides match the numbers in this file.
- [ ] No claim of clinical readiness, accuracy, Indian validation or fusion benefit.

After recording
- [ ] Exported duration is exactly 20:00 (check the file, then the YouTube player).
- [ ] Uploaded as **Unlisted**, not Private or Public.
- [ ] Anonymous check: open the link in a private window of a different browser, signed out of Google. It must play without login or an access request.
- [ ] Audio audible and UI readable at 720p and 1080p.
- [ ] Link placed in README (`<<YOUTUBE UNLISTED URL>>`) and in the submission form.

Segment 3 shows manufactured data only. The repository has a real-data replay receipt, but the failed-gate model is not served. Do not change that to make the demo look better.
