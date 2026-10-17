# Documentation map

CardioTwin is a research prototype for forecasting a new AF onset 2-10 minutes ahead. It is not clinically ready and has no Indian-population evidence.

## Start here

| Document | What it is |
| --- | --- |
| [README](../README.md) | Overview, India relevance, quick start, results, limitations |
| [Architecture PDF](delivery/architecture.pdf) | Two-path design: online causal replay and offline evaluation |
| [Presentation PDF](delivery/presentation.pdf) | Slide deck |
| [Video script](video-script.md) | 20:00 narration, demo commands, recording checklist |
| [Submission checklist](submission.md) | Deadlines, done and human-only items, contradictions to confirm |

## Honest-results summary

Frozen clinical (age/sex) plus causal-ECG logistic regression, at most 0.25 assessed false warnings per eligible hour (our own assumption, not the organizers').

| Cohort | Detected | Sensitivity |
| --- | --- | --- |
| Development (241 events) | 6 / 241 | 2.49% |
| Primary held-out | 1 / 43 | 2.33% |
| LTAF external | 2 / 29 | 6.90% |

- Not clearly above random timing; no fusion benefit shown (ECG-only reference 7/241).
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
