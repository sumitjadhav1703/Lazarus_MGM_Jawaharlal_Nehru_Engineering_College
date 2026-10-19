# Challenge delivery checklist

Status as of 19 Oct 2026. Team information, source, curated evidence and PDFs are supplied; the canonical GitHub repository is verified public. A local 20:00 synthetic-voice export is verified; chapters and captions are in [docs/video](video/README.md). Final video upload/link, registration and submission remain unverified. Official rules and wording conflicts are in the [requirements audit](../reports/ml-investigation/CHALLENGE_REQUIREMENTS.md) (dated 10 Oct 2026; recheck the listing).

## Deadlines (IST)

- **Registration: 18 Oct 2026, 19:00 IST.**
- **Submission: 20 Oct 2026, 19:00 IST.** No changes afterward according to the FAQ.

## Contradictions to confirm with the organizers

The Unstop listing (API fetch, 17 Oct 2026) and form contradict each other. Ask the organizers, and meanwhile satisfy both readings.

- **Shortlist size:** top 5 versus top 10. Nothing to build either way; confirm for expectations.
- **Video length:** listing says minimum 20 min, form says 15-20 min. Record exactly **20:00**.
- The organizers publish **no rubric and no accuracy threshold**; judging is on "technical implementation and real-world healthcare impact". Our 0.25 false-warnings/hour budget and 50% sensitivity target are our own.
- Jury date 30 Oct 2026 (TBA). Teams of 1-4 full-time students. Data must be anonymized, open or synthetic. Folder name `Team Name_College Name`.

## Unstop form items (README must contain these literally)

Mapped in the README [Submission checklist](../README.md#submission-checklist): Team Details, College/Incubator Information, Project Title, Problem Statement, Healthcare Use Case, Technical Stack, AI/ML Model or Framework Details, 15-20 minute Demo Video (unlisted YouTube link), Open-source License Details, Architecture Diagram (PDF/PPT), Presentation (PDF/PPT), All files and links are publicly accessible without additional permissions. The GitHub URL is a separate form field. Canonical public repository: [Lazarus_MGM_Jawaharlal_Nehru_Engineering_College](https://github.com/sumitjadhav1703/Lazarus_MGM_Jawaharlal_Nehru_Engineering_College), verified 19 Oct 2026. Verify the final reviewed revision before submitting. The README table tracks all 12 checklist items. Written materials and PDFs are supplied; the unlisted video link and overall public-access confirmation remain pending.

## Technical items

- [x] Declare the clinical-plus-ECG challenge finalist `logistic_combined` (IRIDIA age/sex and causal ECG); keep the ECG-only scientific reference separately.
- [x] Freeze both artifacts before held-out evaluation; verify input dependence, calibration, policy and cohort contracts. Failed gates stay research-only. See the [challenge model card](../reports/ml-investigation/frozen-challenge-fusion/model_card.md).
- [x] Opt-in research serving: `cardiotwin serve --research-model --model <dir>` replays a gate-failed artifact with a red dashboard banner; the default still refuses it, and `/api/health` reports `research_model` and `deployment_ready`. Local proof 17 Oct 2026 (not a shipped receipt): frozen finalist on IRIDIA development `record_000` (in-sample), forecasts at 3600/7200/10800 s with risks 0.00920/0.00787/0.00804, 0 warnings in the first 10,800 s, about 26-29 ms per step. Shows it runs in the twin, not forecasting quality.
- [x] Verify causal real-data development-prefix replay and step latency (72 steps, median 25 ms, p95 27 ms; future-only ECG/annotation changes leave earlier outputs identical). Not a held-out or UI validation.
- [x] Complete all four primary/external frozen evaluations and predeclared endpoint checks; counts, uncertainty, false warnings, misses, occupancy and abstention are in the [investigation report](../reports/ml-investigation/report.md).
- [x] Independent re-analysis against random timing (17 Oct 2026). Verdict: not meaningfully above chance at event level.
- [x] Calibration slope guard added in `train.py::_calibrate` with tests (exploratory only; frozen models unchanged).
- [x] README rewritten around the honest evidence, India sources and limitations.
- [x] Architecture PDF and presentation PDF in `docs/delivery/` ([architecture](delivery/architecture.pdf), [presentation](delivery/presentation.pdf)).
- [x] Local code/data attribution, reproducible commands and delivery links checked ([DATA_LICENSE.md](../DATA_LICENSE.md), MIT code license).

## Honest findings to keep consistent everywhere

| Finding | Number |
| --- | --- |
| Frozen finalist, development | 6/241 detected, 2.49% sensitivity |
| Primary held-out | 1/43 |
| LTAF external | 2/29 |
| Against random timing | dev 6 observed vs about 4.5 expected; held-out 3/72 vs about 0.7 (post-hoc, no skill claim) |
| LTAF low false-warning rate | not skill: 392 of 398 warning starts in unassessable time |
| Fusion benefit | not shown: ECG-only reference 7/241 vs fused 6/241 |
| Deep learning | RR-TCN16 4/241 and 7/241; waveform CNN 4/241 at 33.13% occupancy; ECGFounder no outcome results |
| Calibration defect (dev fold 0) | AUROC 0.573 raw to 0.427 calibrated; frozen models unaffected |
| Test suite (public repository, no PyTorch) | 842 passed, 83 skipped, 0 failed on 19 Oct 2026; skips need private receipts or PyTorch |
| Static fields | Code supports 16 EHR-style fields; the manufactured demo fills all; real IRIDIA-AF gives age and sex only. Whether that meets "static EHR" is a judgement call |
| Per-patient baseline | Twin state shown to the clinician; not an input to the frozen finalist (27 inputs: age, sex, 2 flags, 23 ECG features) |
| Dashboard default | Manufactured data, synthetic-trained model; real finalist only via opt-in research mode (banner, in-sample record) |

The frozen model fails our development sensitivity gate. Do not claim that clinical fusion improved forecasting, that the system is clinically ready, or that these cohorts establish Indian performance. AF onset is the outcome; heart attack and stroke are different outcomes.

## Delivery and platform actions

- [ ] **Registration confirmed by 18 Oct 2026, 19:00 IST.** Keep the platform receipt. Recheck current organizer terms and the submission form.
- [x] Team details filled in README and presentation PDF: Team Lazarus; MGM's Jawaharlal Nehru Engineering College; Atharva Kulkarni (Team Leader), Pranav Mirkar, Sumit Jadhav. Still verify that every member is a full-time student and confirmed on Unstop. Add the verified unlisted YouTube link to README after upload. Do not rely on post-registration changes.
- [ ] Team leader name, phone and email entered in the submission form (not required in README).
- [x] Canonical **public GitHub repository** verified 19 Oct 2026: [Lazarus_MGM_Jawaharlal_Nehru_Engineering_College](https://github.com/sumitjadhav1703/Lazarus_MGM_Jawaharlal_Nehru_Engineering_College). Its name follows the team/college convention.
- [ ] Publish and verify the final reviewed documentation/PDF/video-support revision. Review the exact revision for raw signals, credentials, private upload files, model binaries or local patient state. Verify anonymous access to every final link.
- [x] Local first draft exists: 19:59, 1920×1080 at 30 fps, synthetic TTS narration and manufactured-data dashboard footage. It is not a published submission video.
- [x] Improved **20:00** local export with captions, seven chapters and four explanatory diagrams built and inspected; full decode passed. Original draft retained. Full subjective playback remains pending.
- [ ] Upload the reviewed export as **unlisted YouTube**. Verify anonymous playback, readable UI, audio and exact duration. Add the link to README and the submission form. A local file is not a completed upload.
- [x] Team, college and leader are filled in the presentation PDF (rebuilt 17 Oct 2026). The demo-video URL is explicitly marked pending (slide 12): after upload, set `VIDEO_URL` in `scripts/build_delivery_pdfs.py` and rebuild with `uv run --with reportlab==4.4.9 python scripts/build_delivery_pdfs.py`.
- [ ] Review the final public contents, including `reports/ml-investigation/primary-split.json` (pseudonymous group IDs), as part of the final publication check.
- [ ] Confirm the two contradictions above with the organizers, or note that both readings were satisfied.
- [ ] Submit **one solution** with the repository link by **20 Oct 2026, 19:00 IST**, leaving time for failures. Verify the platform receipt and the final linked revision. Recheck the [listing](https://unstop.com/hackathons/crp-digital-twin-challenge-2026-happiest-health-1757873) before submitting.
- [x] Optional fresh-cohort check (MIT-BIH AFDB): dropped by the team. The loader would yield no eligible events without new code and the expected event count is small, so it has little statistical value. Not required for submission.

GitHub visibility and the improved local 20:00 export are verified. No registration receipt, unlisted YouTube upload, platform contact details or submission success is claimed here.

## Rebuilding the delivery PDFs

Reproduce with ReportLab 4.4.9. The builder validates saved evaluation/model/split identity and shows missing outcomes as pending.

```bash
uv run --with reportlab==4.4.9 python scripts/build_delivery_pdfs.py --self-check
uv run --with reportlab==4.4.9 python scripts/build_delivery_pdfs.py
```

Running the builder does not complete recording, upload or submission.
