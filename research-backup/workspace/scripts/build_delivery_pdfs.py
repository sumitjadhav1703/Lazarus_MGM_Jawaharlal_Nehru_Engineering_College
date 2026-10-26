"""Generate architecture/presentation PDFs from saved evidence; no fabricated results."""

import json
import math
import sys
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/delivery"
OUT.mkdir(parents=True, exist_ok=True)
W, H = 960, 540
NAVY = HexColor("#122C3B")
TEAL = HexColor("#007C83")
GRAY = HexColor("#50646F")
BG = HexColor("#F5F7F5")


def frame(c, title, subtitle, page):
    c.setFillColor(BG)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(TEAL)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(48, 494, "CARDIOTWIN  /  RESEARCH PROTOTYPE")
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 29)
    c.drawString(48, 442, title)
    c.setFillColor(GRAY)
    c.setFont("Helvetica", 13)
    c.drawString(48, 414, subtitle)
    c.setFont("Helvetica", 10)
    c.drawString(48, 25, "Historical ECG simulation. No clinical readiness or Indian-generalization claim.")
    c.drawRightString(W - 48, 25, str(page))


def prose(c, items, y=365):
    for heading, text in items:
        c.setFillColor(TEAL)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(48, y, heading)
        y -= 24
        c.setFillColor(NAVY)
        c.setFont("Helvetica", 15)
        for line in simpleSplit(text, "Helvetica", 15, 850):
            c.drawString(48, y, line)
            y -= 22
        y -= 20


def box(c, x, y, w, h, title, lines):
    c.setFillColor(HexColor("#FFFFFF"))
    c.setStrokeColor(HexColor("#CCD8DA"))
    c.roundRect(x, y, w, h, 8, fill=1, stroke=1)
    c.setFillColor(TEAL)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x + 16, y + h - 26, title)
    c.setFillColor(NAVY)
    c.setFont("Helvetica", 12)
    yy = y + h - 50
    for line in lines:
        for wrapped in simpleSplit(line, "Helvetica", 12, w - 32):
            c.drawString(x + 16, yy, wrapped)
            yy -= 17


def arrow(c, x1, y1, x2, y2):
    c.setStrokeColor(TEAL)
    c.setFillColor(TEAL)
    c.setLineWidth(2)
    c.line(x1, y1, x2, y2)
    p = c.beginPath()
    p.moveTo(x2, y2)
    p.lineTo(x2 - 7, y2 + 4)
    p.lineTo(x2 - 7, y2 - 4)
    p.close()
    c.drawPath(p, fill=1, stroke=0)


def architecture():
    c = canvas.Canvas(str(OUT / "architecture.pdf"), pagesize=(W, H))
    c.setTitle("CardioTwin Architecture")
    frame(
        c,
        "Clinical history plus causal ECG replay",
        "Frozen model: offline research replay. Conceptual UI: manufactured-data demonstration.",
        1,
    )
    box(
        c,
        48,
        248,
        250,
        122,
        "Audited input",
        [
            "IRIDIA ECG; external LTAF WFDB",
            "Paired IRIDIA baseline age / sex",
            "Audited patient / record groups",
        ],
    )
    box(
        c,
        350,
        248,
        250,
        122,
        "Causal feature service",
        ["Trailing 10-minute ECG", "RR variability and quality", "Baseline state (not model input)"],
    )
    box(
        c,
        652,
        248,
        260,
        122,
        "Frozen predictor and policy",
        [
            "Clinical + ECG logistic finalist",
            "Development-only calibration",
            "Persistence / hysteresis / cooldown",
        ],
    )
    arrow(c, 298, 310, 350, 310)
    arrow(c, 600, 310, 652, 310)
    box(
        c,
        48,
        74,
        250,
        118,
        "Offline evaluator only",
        ["Rhythm intervals / future labels", "One-to-one warning matching", "Untouched patients / bootstrap"],
    )
    box(
        c,
        350,
        74,
        250,
        118,
        "SQLite patient twin",
        [
            "Atomic cursor, state and warnings",
            "Bounded buffers and baseline",
            "Replay identity / restart guard",
        ],
    )
    box(
        c,
        652,
        74,
        260,
        118,
        "Conceptual clinician UI",
        [
            "ECG / trends / quality / future risk",
            "Warning evidence and review",
            "Manufactured-data UI demo",
        ],
    )
    arrow(c, 600, 130, 652, 130)
    c.setStrokeColor(TEAL)
    c.line(782, 248, 782, 219)
    c.line(782, 219, 475, 219)
    c.line(475, 219, 475, 192)
    c.showPage()
    frame(
        c,
        "Release and evaluation boundaries",
        "Frozen research artifacts retain failed clinical acceptance gates.",
        2,
    )
    prose(
        c,
        [
            (
                "Development only",
                "121 IRIDIA development groups; independent fitting, calibration and policy groups. The challenge requires clinical + ECG fusion; the unrestricted ECG reference remains separate.",
            ),
            (
                "Freeze before opening test",
                "Both declared finalists are frozen before primary/external evaluation. Preserve weights, calibration, schema, policy, split, seed and input identities; no held-out retuning.",
            ),
            (
                "Complete chronological evaluation",
                "Report sensitivity and misses, false warnings, occupied time, calibration, coverage and latency. Primary intervals use patient groups; LTAF record groups do not prove independent people.",
            ),
            (
                "Local deployment",
                "Localhost FastAPI/React with manufactured-data UI demo and SQLite. Default serving rejects failed-gate research models; actual fused replay is a trusted offline harness.",
            ),
        ],
    )
    c.showPage()
    c.save()


def number(value, spec=".1f"):
    if value is None:
        return "NA"
    if not math.isfinite(value):
        raise ValueError("Non-finite evaluation value")
    return format(value, spec)


def result_text(report):
    if report is None:
        return "Pending saved frozen-cohort evaluation; no development or synthetic result substituted."
    sensitivity = report["event_sensitivity"]
    bounds = report["confidence_intervals"]["intervals"].get("event_sensitivity")
    ci = (
        f"95% CI {number(100 * bounds['lower'])}-{number(100 * bounds['upper'])}%"
        if bounds
        else "95% CI unavailable"
    )
    occupancy = (
        100 * report["warning_hours"] / report["monitored_hours"] if report["monitored_hours"] else None
    )
    return (
        f"{report['matched_events']}/{report['eligible_events']} events; sensitivity "
        f"{number(100 * sensitivity if sensitivity is not None else None)}% ({ci}); "
        f"FAR {number(report['false_warnings_per_hour'], '.4f')}/eligible h; "
        f"{report['warnings']} starts; occupancy {number(occupancy)}%; "
        f"coverage {number(100 * report['forecast_coverage'] if report['forecast_coverage'] is not None else None)}%; "
        f"group Brier {number(report['probability']['group_macro_brier'], '.6f')}."
    )


def saved_evaluation(folder, cohort, artifact):
    path = ROOT / "reports/ml-investigation" / folder / f"{cohort}-evaluation/evaluation.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    manifest = json.loads((ROOT / "reports/ml-investigation" / artifact / "manifest.json").read_text())
    if (
        report.get("synthetic") is True
        or report.get("cohort") != cohort
        or report.get("model_sha256") != manifest["model_sha256"]
        or report.get("primary_split_sha256") != manifest["primary_split"]["sha256"]
        or not report.get("predictions_sha256")
    ):
        raise ValueError(f"Evaluation is not bound to the declared frozen artifact: {path}")
    return report


def self_check():
    assert "Pending" in result_text(None)
    report = {
        "matched_events": 2,
        "eligible_events": 10,
        "event_sensitivity": 0.2,
        "confidence_intervals": {"intervals": {"event_sensitivity": {"lower": 0.1, "upper": 0.3}}},
        "warning_hours": 2,
        "monitored_hours": 10,
        "false_warnings_per_hour": 0.1,
        "warnings": 4,
        "forecast_coverage": 0.8,
        "probability": {"group_macro_brier": 0.01},
    }
    text = result_text(report)
    assert all(
        part in text
        for part in ("2/10", "20.0%", "10.0-30.0%", "0.1000/eligible h", "occupancy 20.0%", "coverage 80.0%")
    )
    report.update(event_sensitivity=None, monitored_hours=0, forecast_coverage=None)
    assert "sensitivity NA%" in result_text(report)
    try:
        number(float("nan"))
    except ValueError:
        pass
    else:
        raise AssertionError("Non-finite result must fail")
    print("Result formatter self-check passed; no PDFs generated.")


def presentation():
    primary = saved_evaluation("frozen-cohorts", "primary", "frozen-challenge-fusion")
    external = saved_evaluation("frozen-cohorts", "external", "frozen-challenge-fusion")
    primary_reference = saved_evaluation("reference-evaluations", "primary", "frozen-reference")
    external_reference = saved_evaluation("reference-evaluations", "external", "frozen-reference")
    pages = [
        (
            "A defined AF forecasting prototype",
            "Clinical history plus dynamic ECG; clinician review; research only.",
            [
                (
                    "Forecast target",
                    "Previous 600 seconds; new AF onset 120-600 seconds ahead; update every 30 seconds. AF is different from heart attack or stroke.",
                ),
                (
                    "Challenge fusion",
                    "The declared challenge finalist is logistic_combined: paired IRIDIA age/sex and causal ECG. An ECG-only model displayed beside demographic fields would not establish fusion.",
                ),
                (
                    "Scope",
                    "A persistent virtual patient shows changing baseline, signal quality, risk and warning history. Historical replay is labeled; model weights stay frozen. No automated treatment decisions.",
                ),
            ],
        ),
        (
            "Audited real data and limitations",
            "IRIDIA-AF primary cohort; LTAF reserved external transfer.",
            [
                (
                    "Paired primary cohort",
                    "IRIDIA: 167 recordings / 152 people, 200 Hz two-lead ECG and baseline age/sex. Development: 132 recordings / 121 groups, 5,243.98 monitored hours and 241 eligible AF onsets.",
                ),
                (
                    "External transfer",
                    "LTAF: 84 two-lead recordings, 128 Hz, reviewed .atr rhythm references. Missing clinical fields remain missing. Recording IDs do not establish unique people or cross-dataset subject separation.",
                ),
                (
                    "Timing safeguards",
                    "No invented diagnoses or unrelated synthetic EHR joins. Prior-only features; minimum AF duration 30 s; washout 600 s. IRIDIA daily calibration intervals excluded; upstream measurement limitations remain.",
                ),
            ],
        ),
        (
            "Measured development comparisons",
            "Same grouped development cohort; these are not held-out test results.",
            [
                (
                    "Fused challenge finalist",
                    "Combined logistic: 6/241 events (2.49%), FAR 0.1706/eligible h, 761 starts, 40.78% warning occupancy. It fails the 50% engineering sensitivity gate; fusion benefit is not established.",
                ),
                (
                    "Scientific reference preserved",
                    "ECG current recipe: 7/241 events (2.90%), FAR 0.1186/eligible h, 624 starts, 21.84% occupancy. The new challenge constraint was declared after development and before held-out access.",
                ),
                (
                    "Completed CUDA challenger",
                    "RR TCN16: 4/241 events, FAR 0.0300/eligible h, 154 starts, 0.088% occupancy; 9,092.06 s full candidate cost. Waveform completed only 2/3 folds and is excluded; RR32 was not reached.",
                ),
            ],
        ),
        (
            "Persistent patient twin and replay",
            "State adapts to prior observations; the chosen classifier omits baseline-deviation inputs.",
            [
                (
                    "Two input streams",
                    "Paired clinical values and causal ECG feed the fused predictor. Verify the actual frozen schema, coefficients and controlled input sensitivity; displaying fields alone is insufficient.",
                ),
                (
                    "Changing state",
                    "Trailing ECG/RR features, warm-up, quality, baseline and forecast evolve along historical time. SQLite commits cursor, baseline, warning and review state atomically with restart identity guards.",
                ),
                (
                    "Working demonstration",
                    "Show artifact-bound real-prefix research replay logs. Label the manufactured-data dashboard as the conceptual doctor UI; it does not serve this failed-gate artifact. Held-out warnings and misses come from offline reports.",
                ),
            ],
        ),
        (
            "Frozen primary patient test",
            "Saved evaluation only; no test-set threshold or model retuning.",
            [
                ("Clinical + ECG finalist", result_text(primary)),
                ("Preserved ECG scientific reference", result_text(primary_reference)),
                (
                    "Interpretation and provenance",
                    "300 patient-group bootstrap draws; the reference [0,0] interval is degenerate, not precise safety evidence. Frozen model/split/prediction hashes bind these results. Full denominators, misses and lead time are in the report.",
                ),
            ],
        ),
        (
            "Frozen external LTAF transfer",
            "External inputs cannot change preprocessing, calibration, weights or policy.",
            [
                ("Clinical + ECG finalist", result_text(external)),
                ("Preserved ECG scientific reference", result_text(external_reference)),
                (
                    "Transfer limits",
                    "392/398 fusion starts are unassessable: 299 known-excluded, 93 unknown. At 60/300 s minimum AF duration, fusion detects 0/19 and 0/8. Clinical fields are missing; record-group intervals do not prove independent people or Indian generalization.",
                ),
            ],
        ),
        (
            "Warnings, uncertainty and failure",
            "Small start-rate budgets do not guarantee a low warning-time burden.",
            [
                (
                    "Policy",
                    "Require usable history/quality and persistent elevated calibrated risk. Hysteresis and cooldown suppress duplicate starts. Count every eligible onset, including misses during abstention.",
                ),
                (
                    "Honest measurement",
                    "Report false starts per eligible hour, total starts, occupied time, coverage, excluded/unknown warnings, calibration and lead time together. Low Brier in rare events does not establish useful forecasting.",
                ),
                (
                    "Acceptance",
                    "Both declared finalists fail development acceptance. Default serving refuses failed-gate artifacts. Research evaluation can still quantify limitations; neither clinician readiness nor improved outcomes is claimed.",
                ),
            ],
        ),
        (
            "Compute cost and debugging evidence",
            "Allocated GPU hardware and useful compute are separate measurements.",
            [
                (
                    "GPU use verified",
                    "Saved fit receipts and placement code confirm CUDA use on one device despite two T4s being available. The neural screen consumed 20,884.37 s; its candidate/fold costs are nested and must not be added twice.",
                ),
                (
                    "Why the run was slow",
                    "The 10,000-window cap applies to fitting. Normalization, source IO, calibration, policy search and 604,231 available outer decisions also cost time; much preprocessing runs on CPU.",
                ),
                (
                    "Diagnostics",
                    "Current source emits timestamped stage start/done/failure, bounded progress counters, actual model device and NVIDIA utilization samples. These changes were absent from the immutable completed Kaggle v1 run; no T4 speedup benchmark is claimed.",
                ),
            ],
        ),
        (
            "Software checks and reproducibility",
            "Engineering evidence is separate from clinical forecasting evidence.",
            [
                (
                    "178 passing tests",
                    "Recorded diagnostics suite: 178 tests passed. Coverage includes leakage/group guards, target boundaries, alert accounting, causal state, restart and progress failures. Synthetic fixtures validate software, not clinical sensitivity.",
                ),
                (
                    "Reproduce",
                    "Locked Python/npm dependencies, clean base-only wheel verification, optional neural dependencies, saved source/data/split/model hashes and recorded runtime versions. CUDA versus MPS is confounded by runtime differences.",
                ),
                (
                    "Practical limits",
                    "Local API/React dashboard and SQLite; bind to localhost. No hospital authentication, PHI governance or prospective wearable validation. Raw data and credentials stay outside the public repository.",
                ),
            ],
        ),
        (
            "Sources, delivery and next validation",
            "One honest challenge solution; real identity and recording supplied at final delivery.",
            [
                (
                    "Data attribution",
                    "Gilon et al. (2023), IRIDIA-AF v1.0.1: zenodo.org/records/8405941, CC BY 4.0. Petrutiu et al. (2007), LTAF v1.0.0: physionet.org/content/ltafdb/1.0.0/, ODC Attribution 1.0. Code: MIT; full citations in DATA_LICENSE.md.",
                ),
                (
                    "Required delivery",
                    "Public GitHub, actual team/college, architecture/presentation PDFs and a real 20:00-minute unlisted YouTube demo linked in README. Identity, recording and submission remain pending. Register 18 Oct; submit 20 Oct 2026, both 19:00 IST.",
                ),
                (
                    "Clinical next step",
                    "Ethically approved Indian clinical partner data, device/patient-group calibration and prospective silent evaluation before care use. Official rules: unstop.com/hackathons/crp-digital-twin-challenge-2026-happiest-health-1757873.",
                ),
            ],
        ),
    ]
    c = canvas.Canvas(str(OUT / "presentation.pdf"), pagesize=(W, H))
    c.setTitle("CardioTwin Research Presentation")
    for i, (title, subtitle, items) in enumerate(pages, 1):
        frame(c, title, subtitle, i)
        prose(c, items)
        c.showPage()
    c.save()


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        self_check()
    else:
        architecture()
        presentation()
        print(OUT)
