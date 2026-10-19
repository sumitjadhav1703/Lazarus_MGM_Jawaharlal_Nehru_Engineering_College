# Documentation map

CardioTwin is a research prototype for forecasting a new AF onset 2-10 minutes ahead. It is not clinically ready and has no Indian-population evidence.

## Start here

| Document | What it is |
| --- | --- |
| [README](../README.md) | Submission checklist, problem, healthcare use case, technical stack, model details, quick start, results, FAQ, limitations |
| [Architecture PDF](delivery/architecture.pdf) | Two-path design: online causal replay and offline evaluation |
| [Presentation PDF](delivery/presentation.pdf) | Slide deck |
| [Media details, chapters and captions](video/README.md) | [Watch the Unlisted CardioTwin demo](https://youtu.be/4CJFCPdKEto); English captions and seven chapters |
| [Video script](video-script.md) | 20:00 narration, demo commands, recording checklist |
| [Submission checklist](submission.md) | All 12 repository form items supplied; team confirmation and final platform submission remain unverified |

## Reviewer route

Start with the README's result and architecture, follow its synthetic demo commands to inspect replay, signal quality, warnings and clinician review, then read the saved real-data report and model card. The demo shows software behaviour; the real finalist failed its forecasting gates.

The public repository includes source, tests, locked dependencies, curated reports and PDFs. Raw ECG, frozen model binaries, local patient state and complete private research receipts are omitted. The [canonical GitHub repository](https://github.com/sumitjadhav1703/Lazarus_MGM_Jawaharlal_Nehru_Engineering_College) was verified public on 19 Oct 2026. [Watch the Unlisted CardioTwin demo](https://youtu.be/4CJFCPdKEto). Anonymous playback was verified on 19 Oct 2026; see the [delivery status](submission.md) for outstanding platform actions.

## Honest-results summary

Frozen clinical (age/sex) plus causal-ECG logistic regression, at most 0.25 assessed false warnings per eligible hour (our own assumption, not the organizers').

| Cohort | Detected | Sensitivity |
| --- | --- | --- |
| Development (241 events) | 6 / 241 | 2.49% |
| Primary held-out | 1 / 43 | 2.33% |
| LTAF external | 2 / 29 | 6.90% |

- Not clearly above random timing; no fusion benefit shown (ECG-only reference 7/241).
- Per-patient baseline is twin state shown to the clinician, not a model input; the real cohort supplies only age and sex as static fields.
- Default dashboard demo is manufactured data; the real frozen model replays only in opt-in `--research-model` mode (banner, in-sample record).
- Deep-learning arms (RR-TCN16 4/241 and 7/241, waveform CNN 4/241) and the ECGFounder attempt (no outcome results) did not change this.
- Details: [investigation report](../reports/ml-investigation/report.md), [model card](../reports/ml-investigation/frozen-challenge-fusion/model_card.md), [requirements audit](../reports/ml-investigation/CHALLENGE_REQUIREMENTS.md).

## Technical documents

These were written on earlier dates (6-10 Oct 2026). `development-plan.txt` and the SHDB-era documents (SHDB-AF or manufactured data) describe the earlier feasibility stage; they are historical and superseded by the README and the investigation report. Historical reports they reference are not shipped in this repository.

| Document | Content |
| --- | --- |
| [data.md](data.md) | Data sources and forecasting labels |
| [interfaces.md](interfaces.md) | Shared module and API contracts |
| [learning.md](learning.md) | Training and evaluation contracts |
| [ml-findings.md](ml-findings.md) | Measured findings of the earlier SHDB/synthetic iteration (historical) |
| [ml-research.md](ml-research.md) | Research priorities and hypotheses |
| [research.md](research.md) | Source verification notes |
| [ml-code-review.md](ml-code-review.md), [review.md](review.md) | Independent code reviews (6-7 Oct 2026) |
| [progress.md](progress.md), [development-plan.txt](development-plan.txt) | Original execution ledger and plan |

## Licensing

Code: [MIT](../LICENSE). Data: [DATA_LICENSE.md](../DATA_LICENSE.md).
