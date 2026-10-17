# CardioTwin challenge requirements and priorities

Verified 10 October 2026, approximately 01:55 IST. This is a dated requirements audit, not registration, publication or submission confirmation. The team's new priority is challenge compliance. Preserve the earlier unrestricted scientific comparison as a separate reference.

## Official sources and required package

Read the expanded [official listing](https://unstop.com/hackathons/crp-digital-twin-challenge-2026-happiest-health-1757873), its [AMP version](https://unstop.com/hackathons/crp-digital-twin-challenge-2026-happiest-health-1757873/amp), the observed public [competition API](https://unstop.com/api/public/competition/1757873), and [challenge FAQ API](https://unstop.com/api/public/1757873/get-opp-faq-sections). Main-page access required JavaScript; AMP alone omits current submission-form detail.

The listing's mandatory requirements, paraphrased:

- One India-relevant chronic/lifestyle outcome; working adverse-event prediction using historical EHR and dynamic sensor data together; doctor-facing conceptual dashboard.
- Training data must be anonymized, open or synthetic.
- Teams: 1–4 full-time students; cross-college/discipline permitted; members verify details; registrant leads; no post-registration team changes according to the body.
- Public GitHub; README: team/college, title, problem/use case, stack/model, license, demonstration video, architecture PDF/PPT and presentation PDF/PPT. Folder convention: `Team Name_College Name`; all links accessible. Submission identifies leader/contact and repository.
- Registration: 18 October 2026, 19:00 IST (13:30 UTC). Prototype: 20 October 2026, 19:00 IST (13:30 UTC).
- Technical/expert jury; top ten virtual presentations, top five Bengaluru showcase on 20 November; travel/stay for two sponsored. One old body paragraph inconsistently says top five at screening.
- Main page leaves jury date unannounced; AMP's 30 October one-minute slot is not a reliable appointment.

Source: [listing](https://unstop.com/hackathons/crp-digital-twin-challenge-2026-happiest-health-1757873/amp), sections Stages, Team Formation, Sandbox, Submission. No attachment, weighted judging rubric, accuracy threshold, online weight-retraining mandate, treatment/intervention implementation mandate, specific Indian training cohort requirement, or IP-assignment clause was found. This absence is not legal assurance about account terms.

## Submission form conflicts and extra checks

The [competition API](https://unstop.com/api/public/competition/1757873) exposes round `1621855`, `submission_types` IDs `889529` and `892823`: public repository plus a mandatory README checklist. That checklist requires an **unlisted YouTube** demonstration of **15–20 minutes**, while the listing says **minimum twenty minutes**, allowing recorded slides. Our conservative intersection is an actual **20:00-minute unlisted YouTube video**, linked in README; a script or ordinary slide deck is insufficient. Confirm its real playback duration before submission. Do not assume the more permissive wording overrides the form.

API deadline fields contain additional seconds (`end_regn_dt`: `2026-10-18T19:00:43+05:30`; submission `end_date`: `2026-10-20T19:00:45+05:30`). Plan against the displayed 19:00 deadlines. Jury round `1621894` has `to_be_announced=1`, explaining the stale AMP date. No organizer attachment was returned.

The [FAQ](https://unstop.com/api/public/1757873/get-opp-faq-sections) permits **one solution**, editable before the deadline, with no changes afterward. It describes team edits before the first live round/registration closure, conflicting with the body's stricter rule. Finalize membership before registering; confirm any unavoidable exception with the organizer. Registration is free. No registration or contact message was sent during this audit.

## P0: make the submitted predictor actually use both streams

This is our implementation interpretation of the fusion requirement, not a new organizer scoring metric. Showing demographic fields beside an ECG-only prediction does not establish algorithmic fusion. Current scientific reference `current` uses ECG; the RR TCN and waveform CNN likewise lack a static clinical branch. They may remain valuable comparator evidence, but should not be presented as the sole challenge-compliant fused predictor.

IRIDIA provides genuinely paired age/sex. Existing investigation `combined` variants use permitted clinical values and ECG; its `personal` variants exclude clinical values and are not fusion candidates. There is no need to manufacture diagnoses or attach unrelated synthetic EHR to real records. Repository evidence currently shows `logistic_combined` detects 6/241 development events versus 7/241 for the unrestricted reference, with greater warning burden. This does not establish a fusion benefit. See [investigation report](report.md), complete classical pool, and `src/cardiotwin/investigation.py::feature_groups`.

Before opening held-out results:

1. Record a **new, explicitly dated challenge constraint** limiting its finalist pool to genuinely fused, complete, contract-matched candidates already evaluated. Apply the established development ranking within that pool. Keep the earlier unrestricted selection unchanged and labeled as the scientific reference.
2. Verify the chosen schema contains real clinical values and causal ECG features, the fitted predictor consumes them, and a controlled permitted-input perturbation can change its model score. A software sensitivity check is not evidence of improved clinical performance.
3. Freeze each declared artifact, calibration and policy before primary/external evaluation. The same protected held-out cache may serve both predeclared evaluations; neither outcome should cause retuning. Record that this is a constraint-driven change made after seeing development results, rather than pretending it was the original protocol.
4. Keep any failed acceptance gate visible. The challenge's successful-prediction goal is not satisfied merely by passing an API test. If all fused candidates remain weak, show the limited result and feasibility risk honestly.

No additional GPU training, wider search or neural fusion rewrite is justified by this audit. The team reports Kaggle completion; artifact validation remains owned by the main investigation workflow.

## Remaining delivery work in priority order

| Priority | Actual repository evidence | Required next work |
| --- | --- | --- |
| P0 scientific evidence | Real development comparison exists; primary/external frozen evaluation still pending in the latest report. | Validate completed Kaggle artifacts, declare compliant finalist selection, freeze, evaluate and report misses/false warnings/abstention alongside sensitivity. |
| P0 working twin | `TwinService` already stores permitted clinical fields, prior-only baseline, risk, quality, warnings and review timeline; React dashboard exists. | Run the actual chosen research artifact through held-out historical replay, show timestamped changing state and both streams. Keep weights frozen and historical replay visibly labeled. |
| P1 delivery consistency | Architecture/presentation PDFs exist under `docs/delivery/`; `docs/video-script.md` still describes SHDB and previous synthetic results. | Rewrite materials around final IRIDIA/LTAF evidence and the actual fused artifact. Do not leave SHDB or synthetic numbers presented as the final result. |
| P1 reproducibility | Python lockfile, tests, build and installation evidence exist; diagnostics verification records 178 passing tests and a base-only wheel check. | Recheck changed code, accessible README artifact links, licenses, public clean-install instructions and final reproducible demo. No raw ECG, credentials or local state in the public repository. |
| Human-dependent delivery | README explicitly marks real team/college, registration, public repository/video and submission pending. | Supply true eligible member/college and leader contact details, finalize membership, verify registration, publish authorized materials, record the real video and submit one solution. Do not invent identities or claim completion from generated files. |

## Optional choices versus requirements

A physical wearable, Synthea, MIMIC, glucose/sleep features, genetic markers, deep learning, both GPUs, an LLM, a whole-body simulator and automatic medication recommendations are not demanded by the inspected challenge text. Examples of EHR/sensor fields are examples, not a requirement to implement every field.

Our patient baseline updates and persistent state are a practical digital-twin interpretation. Online model-weight updates and treatment-effect simulation require separate data and validation; do not add them merely because the introductory vision discusses personalized treatment. Existing clinician review and clear uncertainty are suitable scope choices. A conceptual UI is the published minimum; our working replay dashboard can provide stronger engineering evidence if verified.

AF onset is our localized cardiovascular outcome. The Indian relevance argument must explain clinical workflow and future local validation; the current datasets do not establish Indian population performance. Do not rename AF prediction as heart-attack or stroke prediction. The 50% sensitivity gate and 0.25 false-warning/hour budget are our research engineering criteria, not official challenge thresholds.

Recheck the official listing and form before submission. This audit changed only this document; it did not register, publish, submit, freeze or retune a model.
