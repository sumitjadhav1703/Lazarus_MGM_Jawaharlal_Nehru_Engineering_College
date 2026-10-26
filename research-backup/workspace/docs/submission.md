# Challenge delivery checklist

Challenge compliance is the priority. The dated [requirements audit](../reports/ml-investigation/CHALLENGE_REQUIREMENTS.md) records official rules and wording conflicts. Complete the technical evidence first; actual team identity and contact details remain deferred to the final delivery stage.

- [x] Declare the genuine paired clinical-plus-ECG challenge finalist: `logistic_combined`, using IRIDIA age/sex and causal ECG features. Preserve the unrestricted ECG-only scientific reference separately. See [challenge selection](../reports/ml-investigation/challenge-selection.json) and [scientific selection](../reports/ml-investigation/unified-selection.json).
- [x] Freeze both declared artifacts before held-out evaluation: [freeze receipt](../reports/ml-investigation/freeze-finalists-receipt.json). Verify actual clinical/ECG schema, fitted input dependence, calibration, policy and cohort contracts; keep failed gates research-only. See the bound [challenge model card](../reports/ml-investigation/frozen-challenge-fusion/model_card.md) and [input-dependence receipt](../reports/ml-investigation/frozen-fusion-dependency-check.json).
- [x] Verify real development-prefix research replay causality and measured step latency: [replay receipt](../reports/ml-investigation/frozen-challenge-replay.json). Future-only ECG/annotation changes leave earlier outputs identical; default serving still refuses the failed-gate artifact. This is not a held-out or restart/UI validation claim.
- [x] Complete all four primary/external frozen evaluations and predeclared endpoint checks; report final counts, uncertainty, false warnings, misses, warning occupancy, abstention and measured replay latency in the [investigation report](../reports/ml-investigation/report.md). Development selection is not held-out validation. Synthetic demonstrations remain software checks.
- [x] Update README, model card, architecture PDF and presentation PDF to the actual IRIDIA/LTAF pipeline, chosen fused artifact and measured limitations. The refreshed two-page architecture and ten-page presentation in `docs/delivery/` were rendered and visually checked; earlier exports are hash-preserved.
- [ ] Record a real **20:00-minute** screen-and-voice demonstration using the [recording script](video-script.md); upload as **unlisted YouTube**, link in README and verify anonymous playback, readable UI, audio and exact duration. The body requires at least twenty minutes while the mandatory form says 15–20; twenty minutes is their common duration. This script is not a recording.
- [x] Verify local code/data attribution, recorded clean installation, reproducible commands, split/feature/model evidence and delivery document links. The [independent final audit](../reports/ml-investigation/independent-final-evidence-review.md) passed; the full 178-test result is recorded for unchanged core diagnostic source, with fresh helper checks.
- [ ] Review the exact public repository revision before publication: exclude raw signals, credentials, private upload files, model binaries and local patient state; verify public installation and anonymous links. Local evidence verification does not certify an uncreated public repository.
- [ ] At the final identity stage, add the true team name, college/incubator and 1–4 eligible full-time student members to README; finalize membership and verify each member. Do not invent details or rely on post-registration changes.
- [ ] Verify actual registration by **18 October 2026, 19:00 IST**; recheck current organizer terms and the submission form.
- [ ] Create the authorized **public GitHub repository** and organize final materials using `Team Name_College Name`. Verify all files and links without extra permissions.
- [ ] Submit **one solution** with the actual team leader's name, phone, email and public repository link by **20 October 2026, 19:00 IST**, allowing time for failures. Verify the platform receipt and final linked revision before the deadline; no changes afterward according to the FAQ.

The fused model currently fails the development sensitivity acceptance gate. Do not claim that clinical fusion improved forecasting, that the system is clinically ready, or that these cohorts establish Indian population performance. AF onset is the outcome; heart attack and stroke are different outcomes.

No registration, publication, contact details, actual video or submission success is claimed here. Identity and those external actions remain pending. Recheck the [official listing](https://unstop.com/hackathons/crp-digital-twin-challenge-2026-happiest-health-1757873) and form before delivery.

All four frozen finalist/reference evaluations and the independent final audit are complete. Reproduce delivery PDFs with ReportLab **4.4.9**. The builder validates saved evaluation/model/split identity and shows missing outcomes as pending; run it only when the final evidence is ready, then render and inspect every page again after any edits.

```bash
uv run --with reportlab==4.4.9 python scripts/build_delivery_pdfs.py --self-check
uv run --with reportlab==4.4.9 python scripts/build_delivery_pdfs.py
```

The recording, upload and actual platform submission are separate final-stage actions; running this builder does not complete them.
