# Challenge delivery checklist

Status as of 17 Oct 2026. Technical evidence is complete; identity, recording, publication and submission are human actions and remain unchecked. Official rules and wording conflicts are in the [requirements audit](../reports/ml-investigation/CHALLENGE_REQUIREMENTS.md) (dated 10 Oct 2026; recheck the listing).

## Deadlines (IST)

- **Registration: 18 Oct 2026, 19:00 IST.**
- **Submission: 20 Oct 2026, 19:00 IST.** No changes afterward according to the FAQ.

## Contradictions to confirm with the organizers

The Unstop listing (API fetch, 17 Oct 2026) and form contradict each other. Ask the organizers, and meanwhile satisfy both readings.

- **Shortlist size:** top 5 versus top 10. Nothing to build either way; confirm for expectations.
- **Video length:** listing says minimum 20 min, form says 15-20 min. Record exactly **20:00**.
- The organizers publish **no rubric and no accuracy threshold**; judging is on "technical implementation and real-world healthcare impact". Our 0.25 false-warnings/hour budget and 50% sensitivity target are our own.
- Jury date 30 Oct 2026 (TBA). Teams of 1-4 full-time students. Data must be anonymized, open or synthetic. Folder name `Team Name_College Name`.

## Technical items

- [x] Declare the clinical-plus-ECG challenge finalist `logistic_combined` (IRIDIA age/sex and causal ECG); keep the ECG-only scientific reference separately.
- [x] Freeze both artifacts before held-out evaluation; verify input dependence, calibration, policy and cohort contracts. Failed gates stay research-only. See the [challenge model card](../reports/ml-investigation/frozen-challenge-fusion/model_card.md).
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
| Test suite (public repository, no PyTorch) | 836 passed, 83 skipped, 0 failed; skips need private receipts or PyTorch |

The frozen model fails our development sensitivity gate. Do not claim that clinical fusion improved forecasting, that the system is clinically ready, or that these cohorts establish Indian performance. AF onset is the outcome; heart attack and stroke are different outcomes.

## Human-only items (not done)

- [ ] **Registration confirmed by 18 Oct 2026, 19:00 IST.** Keep the platform receipt. Recheck current organizer terms and the submission form.
- [ ] Final team: team name, college/incubator, and 1-4 eligible full-time student members verified. Replace the `<<TEAM NAME>>`, `<<COLLEGE NAME>>`, `<<MEMBER NAMES>>` markers in README. Do not rely on post-registration changes.
- [ ] Team leader name, phone and email entered in the submission form (not required in README).
- [ ] Create the authorized **public GitHub repository**; organize final materials as `Team Name_College Name`. Review the exact revision first: no raw signals, credentials, private upload files, model binaries or local patient state. Verify anonymous access to every link.
- [ ] Record a real **20:00** screen-and-voice demonstration with the [video script](video-script.md). Upload as **unlisted YouTube**. Verify anonymous playback, readable UI, audio and exact duration. Put the link in README (`<<YOUTUBE UNLISTED URL>>`).
- [ ] Fill team/college/leader placeholders in docs/delivery PDFs (slides 1 and 12, `scripts/build_delivery_pdfs.py` lines ~36-40) and rebuild with the documented command (`uv run --with reportlab==4.4.9 python scripts/build_delivery_pdfs.py`); re-check every page.
- [ ] Confirm you want `reports/ml-investigation/primary-split.json` (pseudonymous group IDs) published.
- [ ] Confirm the two contradictions above with the organizers, or note that both readings were satisfied.
- [ ] Submit **one solution** with the repository link by **20 Oct 2026, 19:00 IST**, leaving time for failures. Verify the platform receipt and the final linked revision. Recheck the [listing](https://unstop.com/hackathons/crp-digital-twin-challenge-2026-happiest-health-1757873) before submitting.
- [ ] Decide on an optional fresh-cohort check (MIT-BIH AFDB, about 606 MB; eligible events unverified; storage close to the 25 GB cap). Pending a decision; not required for submission.

No registration, publication, contact details, actual video or submission success is claimed here.

## Rebuilding the delivery PDFs

Reproduce with ReportLab 4.4.9. The builder validates saved evaluation/model/split identity and shows missing outcomes as pending.

```bash
uv run --with reportlab==4.4.9 python scripts/build_delivery_pdfs.py --self-check
uv run --with reportlab==4.4.9 python scripts/build_delivery_pdfs.py
```

Running the builder does not complete recording, upload or submission.
