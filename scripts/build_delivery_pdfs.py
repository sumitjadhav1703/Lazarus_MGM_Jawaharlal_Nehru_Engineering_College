"""Generate architecture/presentation PDFs from saved evidence; no fabricated results.

Cohort result rows for the primary and LTAF cohorts are read from the saved,
hash-bound evaluation files under reports/ml-investigation/. Development and
chance-reference figures come from the saved chance-analysis results (see the
source line on each slide); they are constants here and are labelled as such.
"""

import json
import math
import sys
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/delivery"
W, H = 960, 540
NAVY = HexColor("#122C3B")
TEAL = HexColor("#007C83")
GRAY = HexColor("#50646F")
BG = HexColor("#F5F7F5")
WHITE = HexColor("#FFFFFF")
LINE = HexColor("#CCD8DA")
AMBER = HexColor("#9A5B00")
RED = HexColor("#A33A2A")
LEFT = 48
RIGHT = W - 48
FLOOR = 52  # content must stay above the source/footer band

PLACEHOLDER = '<font color="#9A5B00"><b>[%s]</b></font>'
TEAM = "Lazarus"
COLLEGE = "MGM's Jawaharlal Nehru Engineering College"
LEADER = "Atharva Kulkarni; members Pranav Mirkar, Sumit Jadhav"
REPO_URL = "https://github.com/sumitjadhav1703/lazarus_ECG_traces"
VIDEO_URL = PLACEHOLDER % "UNLISTED YOUTUBE URL - to be supplied"


def style(size=13, color=NAVY, bold=False, leading=None, bullet=False):
    return ParagraphStyle(
        "s",
        fontName="Helvetica-Bold" if bold else "Helvetica",
        fontSize=size,
        leading=leading or size * 1.28,
        textColor=color,
        leftIndent=12 if bullet else 0,
        bulletIndent=0,
        bulletFontName="Helvetica-Bold",
        bulletColor=TEAL,
    )


def draw_para(c, html, x, ytop, w, st):
    p = Paragraph(html, st)
    _, h = p.wrap(w, 1000)
    if ytop - h < FLOOR - 4:
        raise ValueError(f"Text overflows page floor: {html[:60]!r}")
    p.drawOn(c, x, ytop - h)
    return ytop - h


def bullets(c, items, x, ytop, w, size=13, gap=5, color=NAVY):
    for item in items:
        ytop = _bullet(c, item, x, ytop, w, size, color) - gap
    return ytop


def _bullet(c, html, x, ytop, w, size, color):
    st = style(size, color, bullet=True)
    p = Paragraph(html, st, bulletText="•")
    _, h = p.wrap(w, 1000)
    if ytop - h < FLOOR - 4:
        raise ValueError(f"Bullet overflows page floor: {html[:60]!r}")
    p.drawOn(c, x, ytop - h)
    return ytop - h


def card(c, x, ytop, w, h, title, items, size=11.5, accent=TEAL):
    """White card; raises if its content does not fit inside h."""
    c.setFillColor(WHITE)
    c.setStrokeColor(LINE)
    c.roundRect(x, ytop - h, w, h, 8, fill=1, stroke=1)
    c.setFillColor(accent)
    c.setFont("Helvetica-Bold", 13.5)
    c.drawString(x + 14, ytop - 24, title)
    y = ytop - 34
    for item in items:
        y = _bullet(c, item, x + 14, y, w - 28, size, NAVY) - 4
        if y < ytop - h + 4:
            raise ValueError(f"Card {title!r} content overflows its box")
    return ytop - h


def table(c, x, ytop, col_widths, rows, size=11, header_bg=NAVY, zebra=True, align_left_cols=(0,)):
    st_head = style(size, WHITE, bold=True)
    st_cell = style(size, NAVY)
    data = []
    for r, row in enumerate(rows):
        data.append([Paragraph(str(cell), st_head if r == 0 else st_cell) for cell in row])
    t = Table(data, colWidths=col_widths)
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, LINE),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    if zebra:
        for r in range(1, len(rows)):
            cmds.append(("BACKGROUND", (0, r), (-1, r), WHITE if r % 2 else HexColor("#EAF1F1")))
    t.setStyle(TableStyle(cmds))
    tw, th = t.wrap(sum(col_widths), 1000)
    if ytop - th < FLOOR - 4:
        raise ValueError("Table overflows page floor")
    t.drawOn(c, x, ytop - th)
    return ytop - th


def frame(c, title, subtitle, page, source=None):
    if stringWidth(title, "Helvetica-Bold", 27) > RIGHT - LEFT:
        raise ValueError(f"Title too wide: {title}")
    if stringWidth(subtitle, "Helvetica", 13) > RIGHT - LEFT:
        raise ValueError(f"Subtitle too wide: {subtitle}")
    c.setFillColor(BG)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(TEAL)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(LEFT, 508, "CARDIOTWIN  /  RESEARCH PROTOTYPE")
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 27)
    c.drawString(LEFT, 468, title)
    c.setFillColor(GRAY)
    c.setFont("Helvetica", 13)
    c.drawString(LEFT, 444, subtitle)
    if source:
        sp = Paragraph(f"Source: {source}", style(8, GRAY, leading=9.5))
        _, sh = sp.wrap(RIGHT - LEFT, 100)
        sp.drawOn(c, LEFT, 40)
    c.setFillColor(GRAY)
    c.setFont("Helvetica", 9)
    c.drawString(LEFT, 20, "Retrospective public data. Research prototype; no clinical-readiness claim; no Indian-population evidence.")
    c.drawRightString(RIGHT, 20, str(page))


def arrow(c, x1, y1, x2, y2):
    c.setStrokeColor(TEAL)
    c.setFillColor(TEAL)
    c.setLineWidth(2)
    c.line(x1, y1, x2 - 6 if x2 > x1 else x2, y2)
    p = c.beginPath()
    p.moveTo(x2, y2)
    p.lineTo(x2 - 8, y2 + 4)
    p.lineTo(x2 - 8, y2 - 4)
    p.close()
    c.drawPath(p, fill=1, stroke=0)
    c.setLineWidth(1)


def node(c, x, ytop, w, h, title, sub, fill=WHITE, tcolor=TEAL, dashed=False, size=9.5):
    c.setFillColor(fill)
    c.setStrokeColor(LINE if not dashed else GRAY)
    if dashed:
        c.setDash(4, 3)
    c.roundRect(x, ytop - h, w, h, 8, fill=1, stroke=1)
    c.setDash()
    c.setFillColor(tcolor)
    c.setFont("Helvetica-Bold", 11.5)
    c.drawString(x + 10, ytop - 20, title)
    y = ytop - 34
    for line in sub:
        y = _bullet(c, line, x + 10, y, w - 20, size, NAVY) - 2
        if y < ytop - h + 2:
            raise ValueError(f"Node {title!r} overflows")


# --------------------------------------------------------------------------- numbers

DEV = {
    "events": "6 / 241", "sens": "2.49% [0.44, 4.96]", "far": "0.171", "starts": 761,
    "occ": "40.8%", "auroc": "0.45 pooled (0.40 / 0.49 / 0.57 by fold)",
}  # reports/ml-investigation/chance-analysis/dev_summary.json, brier_results.json
DEV_REF_EVENTS = "7 / 241"  # reports/ml-investigation/report.md (ECG-only reference)
HELD_AUROC = {"primary": "0.611 [0.542, 0.681]", "external": "0.631 [0.513, 0.761]"}  # reports/ml-investigation/chance-analysis/heldout_results.json


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


def result_row(label, report, auroc="n/a"):
    """One results-table row from a saved, hash-bound evaluation."""
    if report is None:
        return [label, "pending", "pending", "pending", "pending", "pending", auroc]
    bounds = report["confidence_intervals"]["intervals"].get("event_sensitivity")
    ci = f"{number(100 * bounds['lower'], '.2f')}, {number(100 * bounds['upper'], '.2f')}" if bounds else "NA"
    sens = report["event_sensitivity"]
    occ = 100 * report["warning_hours"] / report["monitored_hours"] if report["monitored_hours"] else None
    return [
        label,
        f"{report['matched_events']} / {report['eligible_events']}",
        f"{number(100 * sens if sens is not None else None, '.2f')}% [{ci}]",
        number(report["false_warnings_per_hour"], ".3f"),
        str(report["warnings"]),
        f"{number(occ)}%",
        auroc,
    ]


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
    row = result_row("x", report)
    assert row[1] == "2 / 10" and row[3] == "0.100" and row[5] == "20.0%", row
    report.update(event_sensitivity=None, monitored_hours=0, forecast_coverage=None)
    assert "sensitivity NA%" in result_text(report)
    try:
        number(float("nan"))
    except ValueError:
        pass
    else:
        raise AssertionError("Non-finite result must fail")
    # layout guard: an overlong paragraph must be refused, not silently clipped
    import io

    probe = canvas.Canvas(io.BytesIO(), pagesize=(W, H))
    try:
        draw_para(probe, "word " * 5000, LEFT, 300, 200, style(13))
    except ValueError:
        pass
    else:
        raise AssertionError("Overflow guard must fail")
    # the saved frozen results this deck states must match the files
    for folder, cohort, artifact, expected in (
        ("frozen-cohorts", "primary", "frozen-challenge-fusion", (1, 43)),
        ("frozen-cohorts", "external", "frozen-challenge-fusion", (2, 29)),
        ("reference-evaluations", "primary", "frozen-reference", (0, 43)),
        ("reference-evaluations", "external", "frozen-reference", (0, 29)),
    ):
        saved = saved_evaluation(folder, cohort, artifact)
        if saved is not None:
            assert (saved["matched_events"], saved["eligible_events"]) == expected, (cohort, expected)
    print("Result formatter self-check passed; no PDFs generated.")


# --------------------------------------------------------------------------- architecture


def architecture():
    c = canvas.Canvas(str(OUT / "architecture.pdf"), pagesize=(W, H))
    c.setTitle("CardioTwin Architecture")
    frame(
        c,
        "Causal ECG replay with a persistent patient twin",
        "Runtime sees only the past; labels and scoring live offline. Manufactured-data UI demo.",
        1,
        "README.md Architecture; reports/ml-investigation/report.md; frozen-challenge-fusion/config.json (persistence 60 s, cooldown 600 s).",
    )
    # runtime row
    y0, bh, bw, gap = 408, 104, 127, 20.4
    xs = [LEFT + i * (bw + gap) for i in range(6)]
    runtime = [
        ("Replay service", ["Historical ECG on a 30 s tick", "Single worker, explicit ticks"]),
        ("Causal features", ["Trailing 10 min of ECG", "RR variability, signal quality"]),
        ("Frozen predictor", ["Logistic: age/sex + ECG features", "Dev-only calibration; slope guard"]),
        ("Alert state machine", ["Persistence 60 s, hysteresis", "Cooldown 600 s; abstains on poor quality"]),
        ("SQLite twin store", ["One atomic commit per step", "Cursor, baseline, warning"]),
        ("FastAPI + React UI", ["ECG, trends, risk, quality", "Warning history, clinician review"]),
    ]
    for x, (t, sub) in zip(xs, runtime):
        node(c, x, y0, bw, bh, t, sub)
    for i in range(5):
        arrow(c, xs[i] + bw, y0 - bh / 2, xs[i + 1], y0 - bh / 2)
    c.setFillColor(GRAY)
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(LEFT, y0 + 10, "ONLINE PATH (causal): no future samples, annotations, onset or pair identifiers")
    # baseline loop
    c.setStrokeColor(TEAL)
    c.setDash(3, 3)
    c.line(xs[1] + bw / 2, y0 - bh, xs[1] + bw / 2, y0 - bh - 16)
    c.line(xs[1] + bw / 2, y0 - bh - 16, xs[4] + bw / 2, y0 - bh - 16)
    c.line(xs[4] + bw / 2, y0 - bh - 16, xs[4] + bw / 2, y0 - bh)
    c.setDash()
    c.setFillColor(TEAL)
    c.setFont("Helvetica", 9.5)
    c.drawString(xs[1] + 60, y0 - bh - 29, "per-patient baseline updated AFTER each prediction")

    # offline row
    y1 = 250
    c.setFillColor(GRAY)
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(LEFT, y1 + 10, "OFFLINE PATH (labels and scoring): never imported by runtime code")
    ow = 262
    off = [
        ("Data audit + labels", ["IRIDIA-AF and LTAF WFDB; future rhythm intervals", "Cohort feasibility gate"]),
        ("Training + freeze", ["Patient-level splits; fit / stop / calibrate / policy groups disjoint", "Hashed artifact + manifest"]),
        ("Complete-stream evaluation", ["Whole recordings in time order; misses kept", "One-to-one matching; FAR per eligible hour"]),
    ]
    for i, (t, sub) in enumerate(off):
        x = LEFT + i * (ow + 30)
        node(c, x, y1, ow, 88, t, sub, fill=HexColor("#EEF3F3"), tcolor=NAVY, dashed=True)
        if i:
            arrow(c, x - 30, y1 - 42, x, y1 - 42)

    # shared + models
    y2 = 154
    card(
        c, LEFT, y2, 420, 102, "Shared by runtime and evaluation",
        ["Alert state machine and evaluation code are one code path, so scored behaviour equals served behaviour.",
         "Default demo: manufactured data and a synthetic-trained model. Opt-in <i>serve --research-model</i> runs the gate-failed frozen finalist behind a red FAILED banner."],
        size=10,
    )
    card(
        c, LEFT + 440, y2, 424, 102, "Model families tried (all offline, all development-only)",
        ["Logistic and boosted classical models; RR-TCN16 (RR-interval CNN); waveform CNN with causal TCN head.",
         "Frozen ECG foundation-model (ECGFounder) attempt: no outcome results."],
        size=10.5,
    )
    c.showPage()

    frame(
        c,
        "Evidence and boundaries, at a glance",
        "What was frozen, what was measured, and what is not claimed.",
        2,
        "reports/ml-investigation/{report.md, frozen-cohorts, reference-evaluations}; reports/ml-investigation/chance-analysis/*.json.",
    )
    card(
        c, LEFT, 430, 420, 238, "Release and evaluation boundaries",
        [
            "<b>Development only:</b> 121 IRIDIA-AF patient groups; fit, stop, calibration and policy groups disjoint.",
            "<b>Freeze before opening test:</b> model, calibration, schema, policy, split and seed hashed before primary and LTAF were scored. No held-out retuning; both cohorts are now used up.",
            "<b>Complete chronological evaluation:</b> misses, false warnings, occupied time, coverage reported together. LTAF record groups do not prove independent people.",
            "<b>Local deployment:</b> localhost FastAPI + React + SQLite; manufactured-data demo; no hospital authentication or PHI governance.",
        ],
        size=12,
    )
    card(
        c, LEFT + 440, 430, 424, 238, "Measured results (event level)",
        [
            "Frozen age/sex + ECG logistic finalist detects <b>6/241</b> development, <b>1/43</b> primary held-out, <b>2/29</b> LTAF events (sensitivity 2-7%).",
            "Against random timing the development result is within chance (6 observed vs about 4.5 expected; P(at least 6) about 0.33).",
            "LTAF false-warning rate 0.008/h is not skill: 392 of 398 warning starts fall in unassessable time.",
            "The age/sex inputs show no benefit: ECG-only reference 7/241 vs fused 6/241.",
        ],
        size=12,
    )
    card(
        c, LEFT, 180, 420, 128, "Defect found and guarded",
        [
            "Fold-0 calibrators learned negative slopes and reversed ranking (AUROC 0.573 raw to 0.427).",
            "A slope guard now raises in train.py::_calibrate; test in tests/test_calibration_slope.py. Frozen models had positive slopes, so held-out results are unaffected.",
        ],
        size=12,
        accent=RED,
    )
    card(
        c, LEFT + 440, 180, 424, 128, "Not claimed",
        [
            "No clinical readiness, no fusion benefit, no Indian-population evidence.",
            "Research criteria (0.25 false warnings/h, 50% sensitivity) are ours, not the organizers'; the acceptance gate fails.",
        ],
        size=12,
        accent=RED,
    )
    c.showPage()
    c.save()


# --------------------------------------------------------------------------- presentation


def slide_title(c):
    frame(c, "CardioTwin", "Causal forecasting of new atrial-fibrillation onset 2-10 minutes ahead", 1,
          "reports/ml-investigation/CHALLENGE_REQUIREMENTS.md.")
    bullets(
        c,
        [
            "CRP Digital Twin Challenge 2026 (Happiest Health) submission",
            f"Team: {TEAM}",
            f"College / incubator: {COLLEGE}",
            f"Team leader: {LEADER}",
        ],
        LEFT, 400, 700, size=16, gap=8,
    )
    card(
        c, LEFT, 270, 864, 150, "Scope of this submission",
        [
            "Decision every 30 s from the previous 10 min of ECG; target is a <b>new AF onset 120-600 s ahead</b>.",
            "<b>Research prototype</b> on retrospective public data. Not clinical, not a medical device, no automated treatment decisions.",
            "Honest result up front: the forecasting signal is weak. The rest of this deck shows how weak, and how we know.",
        ],
        size=13,
    )


def slide_problem(c):
    frame(c, "Problem and India relevance", "Early warning to support clinician review, not diagnosis.", 2,
          "India: pmc.ncbi.nlm.nih.gov/articles/PMC9039683 (Saggu 2022); /PMC5083844 (Soni 2016); /PMC12609587 (Goyal 2025, GBD 2019); "
          "/PMC11317704 (Calvert 2024). Challenge: CHALLENGE_REQUIREMENTS.md.")
    card(
        c, LEFT, 425, 380, 340, "Why this problem, and our scope",
        [
            "The challenge asks for adverse-event prediction from EHR plus sensor data and a doctor-facing dashboard.",
            "We chose <b>one event: new AF onset</b>. Heart attack and stroke prediction are out of scope.",
            "Use case: flag a short window of higher risk so a clinician can review the ECG. Not diagnosis.",
            "Community AF estimates in India range from about <b>0.1% to 5%</b> depending on method and age, and national data are lacking.",
        ],
        size=14.5,
    )
    card(
        c, LEFT + 400, 425, 464, 340, "India context (cited; no causal claim)",
        [
            "Rural Andhra Pradesh survey, 4,281 adults: AF <b>0.32%</b> (95% CI 0.17-0.54); 1.5% above age 60, 3.5% above 75. Saggu et al., Indian Heart J 2022.",
            "Small rural Gujarat screened sample (354 people aged 50+): 5.1% AF. Soni et al., JMIR Public Health Surveill 2016.",
            "India GBD 2019, <b>ischemic stroke only</b>: 535,700 incident cases and 271,200 deaths. Goyal et al., J Clin Med 2025.",
            "Kerala-AF Registry, 3,421 AF patients in 53 hospitals: NOAC use only about 5-6% in the 2,901-patient rate/rhythm cohort. Calvert et al., J Arrhythmia 2024.",
        ],
        size=13,
    )


def slide_principles(c):
    frame(c, "Design principles", "Causal inputs, patient-level splits, and every miss and false warning counted.", 3,
          "README.md; reports/ml-investigation/report.md; frozen-challenge-fusion/config.json (history 600 s, gap 120 s, horizon 600 s, cadence 30 s).")
    # timeline
    x0, y = LEFT, 372
    c.setFillColor(HexColor("#DCEBEC"))
    c.rect(x0, y, 400, 34, fill=1, stroke=0)
    c.setFillColor(HexColor("#F3D9A8"))
    c.rect(x0 + 440, y, 240, 34, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 11)
    c.drawCentredString(x0 + 200, y + 12, "Input: previous 600 s of ECG (causal)")
    c.drawCentredString(x0 + 560, y + 12, "Target: new AF onset in 120-600 s")
    c.setStrokeColor(TEAL)
    c.setLineWidth(2)
    c.line(x0 + 400, y - 6, x0 + 400, y + 44)
    c.line(x0 + 440, y - 6, x0 + 440, y + 44)
    c.line(x0 + 680, y - 6, x0 + 680, y + 44)
    c.setLineWidth(1)
    c.setFillColor(GRAY)
    c.setFont("Helvetica", 10)
    c.drawRightString(x0 + 396, y - 20, "decision time")
    c.drawString(x0 + 404, y - 20, "120 s gap")
    c.drawString(x0 + 700, y + 12, "A new decision every 30 s")
    bullets(
        c,
        [
            "<b>Causal:</b> only the previous 10 min of ECG; no future samples, annotations, onset or pair identifiers at runtime. The per-patient baseline is updated after each prediction.",
            "<b>Patient-level splits:</b> fit, early-stop, calibration and policy groups are disjoint. Model, calibration and alert policy were frozen before the held-out cohorts were opened.",
            "<b>Complete accounting:</b> report misses, false warnings per eligible hour, warning time and unassessable warnings, not just sensitivity.",
            "<b>Failed attempts stay on record:</b> gates were declared in advance and not loosened after results; held-out cohorts are not reused for tuning.",
        ],
        LEFT, 325, 864, size=15.5, gap=12,
    )


def slide_architecture(c):
    frame(c, "Architecture", "One shared alert and evaluation code path for runtime and offline scoring.", 4,
          "README.md Architecture; docs/delivery/architecture.pdf.")
    y0, bh, bw, gap = 400, 100, 128, 19
    xs = [LEFT + i * (bw + gap) for i in range(6)]
    names = [
        ("Replay", ["30 s ticks over", "historical ECG"]),
        ("Causal features", ["10 min trailing", "RR + quality"]),
        ("Logistic model", ["age/sex + ECG,", "frozen artifact"]),
        ("Alert machine", ["hysteresis,", "cooldown"]),
        ("SQLite store", ["atomic commit", "per step"]),
        ("API + dashboard", ["FastAPI, React", "clinician review"]),
    ]
    for x, (t, sub) in zip(xs, names):
        node(c, x, y0, bw, bh, t, sub, size=11.5)
    for i in range(5):
        arrow(c, xs[i] + bw, y0 - bh / 2, xs[i + 1], y0 - bh / 2)
    node(c, LEFT, 270, 864, 40, "Offline only: labels, audit, training, complete-stream evaluation", [], fill=HexColor("#EEF3F3"), tcolor=NAVY, dashed=True)
    bullets(
        c,
        [
            "The labels module (future rhythm intervals) is <b>never imported</b> by runtime code.",
            "Alert and evaluation modules are shared, so the behaviour that is scored is the behaviour that is served.",
            "The default demo uses manufactured data and a synthetic-trained model. Opt-in <font face=\"Courier\">cardiotwin serve --research-model</font> runs the gate-failed frozen finalist in the twin under a red \u2018RESEARCH MODEL ... FAILED its acceptance gates\u2019 banner.",
        ],
        LEFT, 215, 864, size=15, gap=10,
    )


def slide_streams(c):
    frame(c, "Two data streams, one model", "Static clinical fields plus dynamic ECG, fused in one logistic regression.", 5,
          "CHALLENGE_REQUIREMENTS.md; README.md; frozen-challenge-fusion/feature_schema.json (27 inputs); src/cardiotwin/features.py (CLINICAL_FIELDS); report.md.")
    card(c, LEFT, 420, 380, 150, "Static stream", [
        "Code supports <b>16 static EHR-style fields</b>; the manufactured demo fills them.",
        "The real cohort provides <b>age and sex</b> only; nothing else is invented.",
    ], size=13.5)
    card(c, LEFT, 252, 380, 150, "Dynamic stream", [
        "Causal ECG features: RR variability, beats, signal quality.",
        "Per-patient baseline is twin state shown to the clinician, <b>not a model input</b>.",
    ], size=13.5)
    card(c, LEFT + 440, 420, 424, 150, "Fused predictor", [
        "One logistic regression, <b>27 inputs</b>: age, sex, their availability flags and 23 ECG features. Development-only calibration, frozen before test.",
        "ECG-only reference is kept as a separate model.",
    ], size=13.5)
    card(c, LEFT + 440, 252, 424, 150, "Did fusion help? No benefit shown", [
        "Development: ECG-only reference <b>7/241</b>, fused <b>6/241</b>; fusion showed no benefit.",
        "Held-out: fused 1/43 and 2/29; the reference issued no warnings (policy threshold 1.0).",
    ], size=13.5, accent=RED)
    bullets(c, ["Baseline deviation is computed for display; no baseline feature is among the 27 model inputs."], LEFT, 82, 864, size=13.5)


def slide_data(c):
    frame(c, "Data", "Public, retrospective, anonymized recordings only.", 6,
          "README.md Evidence-driven investigation; report.md; docs/video-script.md results card. IRIDIA-AF: Gilon et al. 2023, CC BY 4.0. LTAF: Petrutiu et al. 2007.")
    table(
        c, LEFT, 410, [200, 170, 150, 344],
        [
            ["Cohort", "Groups", "Eligible AF onsets", "Role"],
            ["IRIDIA-AF development", "121 patient groups (132 recordings)", "241", "Grouped 3-fold development: fit, stop, calibration, policy"],
            ["IRIDIA-AF primary held-out", "31 patient groups", "43", "Opened once, after the model was frozen"],
            ["LTAF external", "84 recording groups", "29", "External transfer; patient independence unverified"],
        ],
        size=13,
    )
    bullets(
        c,
        [
            "IRIDIA-AF in total: <b>167 recordings, 152 patients, 6,690.476 monitoring hours</b>; 284 eligible events across 143 patients (241 + 43).",
            "Development monitored time: 5,243.98 h; 233 of 241 events have at least one available decision.",
            "Missing clinical fields stay missing (LTAF has no age/sex). Synthetic fixtures validate software only and are not evidence.",
            "Raw data and credentials stay outside the public repository.",
        ],
        LEFT, 235, 864, size=14.5, gap=9,
    )


def slide_eval(c):
    frame(c, "Evaluation method", "Whole recordings replayed in time order, nothing filtered out afterwards.", 7,
          "README.md Model and evaluation protocol; CHALLENGE_REQUIREMENTS.md (organizers published no accuracy threshold).")
    card(c, LEFT, 420, 420, 300, "How we score", [
        "Recordings replayed chronologically; quality-rejected stretches and missed events stay in.",
        "One warning matches at most one event; false warnings per <b>eligible hour</b>.",
        "Unknown-outcome and unassessable warnings (during AF, post-AF washout) are reported separately.",
        "Also reported: warning occupancy, coverage, calibration, lead time, patient-group bootstrap intervals.",
    ], size=15)
    card(c, LEFT + 444, 420, 420, 300, "Our own criteria (not the organizers')", [
        "Research assumption: at most <b>0.25 assessed false warnings per eligible hour</b>.",
        "Research assumption: <b>50% event sensitivity</b> as a provisional engineering gate.",
        "The organizers published no accuracy threshold or rubric; these are ours and chosen before the held-out look.",
        "Gate result: sensitivity 2-7% against 50%, so the gate <b>fails</b>.",
    ], size=15, accent=AMBER)


def slide_dashboard(c):
    frame(c, "Dashboard demo (manufactured data)", "Software demonstration of the doctor-facing view; synthetic data is not evidence.", 8,
          "reports/ml-investigation/frozen-challenge-replay.json (72 steps; passed: true); src/cardiotwin/cli.py and tests/test_research_serving.py (--research-model); research-model run was a local check, no saved receipt; full suite 841 passed, 83 skipped, 0 failed.")
    card(c, LEFT, 420, 280, 210, "What the UI shows", [
        "ECG, trends, per-patient baseline",
        "Signal quality and abstention",
        "Risk, warning history",
        "Clinician review of each warning",
    ], size=14.5)
    card(c, LEFT + 292, 420, 280, 210, "What it is not", [
        "Default demo data are manufactured, not real patients.",
        "Default model is synthetic-trained; the real finalist runs only via opt-in <font face=\"Courier\">--research-model</font>, red FAILED banner.",
        "No hospital login, PHI governance or wearable validation.",
    ], size=14.5, accent=RED)
    card(c, LEFT + 584, 420, 280, 210, "Causality and speed checks", [
        "Replay step latency on real prefixes: median <b>25 ms</b>, p95 <b>27 ms</b> (72 steps).",
        "Changing only future ECG or annotations left earlier outputs identical.",
    ], size=14.5)
    bullets(c, ["Local proof, <b>in-sample</b>: frozen finalist on one IRIDIA development record gave forecasts at 3600, 7200 and 10800 s (risks 0.0092, 0.0079, 0.0080), 0 warnings, about 26-29 ms per step. This proves the model runs in the twin, not forecasting quality.",
                "Tests: full suite in this public repo on 17 Oct 2026: 841 passed, 83 skipped (private receipts or optional PyTorch), 0 failed."],
            LEFT, 190, 864, size=13, gap=6)


def slide_results(c):
    primary = saved_evaluation("frozen-cohorts", "primary", "frozen-challenge-fusion")
    external = saved_evaluation("frozen-cohorts", "external", "frozen-challenge-fusion")
    ref_primary = saved_evaluation("reference-evaluations", "primary", "frozen-reference")
    ref_external = saved_evaluation("reference-evaluations", "external", "frozen-reference")
    frame(c, "Results: weak", "Frozen age/sex + ECG logistic finalist; events detected out of eligible events.", 9,
          "Primary/LTAF: saved hash-bound evaluation.json files. Development and AUROC: reports/ml-investigation/chance-analysis/{dev_summary,brier_results,heldout_results}.json; report.md.")
    rows = [
        ["Cohort", "Detected", "Sensitivity [95% CI]", "False warnings / eligible h", "Warning starts", "Occupancy", "Window AUROC"],
        ["Development (121 groups)", DEV["events"], DEV["sens"], DEV["far"], str(DEV["starts"]), DEV["occ"], "0.45 pooled"],
        result_row("Primary held-out (31)", primary, HELD_AUROC["primary"]),
        result_row("LTAF external (29 events)", external, HELD_AUROC["external"]),
        ["ECG-only reference", f"dev {DEV_REF_EVENTS.replace(' ', '')}; primary {ref_primary['matched_events'] if ref_primary else '?'}/43; LTAF {ref_external['matched_events'] if ref_external else '?'}/29", "reference primary and LTAF issued no warnings", "-", "-", "-", "-"],
    ]
    y = table(c, LEFT, 415, [190, 110, 170, 120, 90, 90, 94], rows, size=12)
    bullets(
        c,
        [
            "Sensitivity is <b>2-7%</b> against our 50% target; assessed false warnings are under our 0.25/h budget, but sensitivity is far below target.",
            "Dev AUROC by fold 0.40 / 0.49 / 0.57: no reliable ranking. Held-out AUROC 0.61-0.63 is modest.",
            "Probabilities add nothing over base rate in development: Brier 0.008384 vs 0.008358 for a constant-prevalence predictor.",
        ],
        LEFT, y - 16, 864, size=14, gap=8,
    )


def slide_chance(c):
    frame(c, "Against chance", "Same alert policy re-applied to risk shifted to random times inside each recording.", 10,
          "reports/ml-investigation/chance-analysis/dev_summary.json (60 draws), heldout_results.json (200 draws); report.md LTAF burden.")
    # dot plot: observed vs random-timing expectation
    ox, oy, ow = LEFT + 150, 395, 330

    def sx(v):
        return ox + ow * v / 8.0

    c.setStrokeColor(LINE)
    c.setFillColor(GRAY)
    c.setFont("Helvetica", 9)
    for v in range(0, 9):
        c.line(sx(v), oy - 120, sx(v), oy + 6)
        c.drawCentredString(sx(v), oy - 132, str(v))
    c.drawString(ox, oy - 146, "events detected")
    rows = [("Development (n=241)", 6, 4.45, (1.5, 7.0), 355), ("Held-out, primary + LTAF (n=72)", 3, 0.70, None, 305)]
    for label, obs, exp, rng, yy in rows:
        c.setFillColor(NAVY)
        c.setFont("Helvetica", 10.5)
        c.drawRightString(ox - 10, yy - 12, label)
        if rng:
            c.setStrokeColor(GRAY)
            c.setLineWidth(5)
            c.line(sx(rng[0]), yy - 8, sx(rng[1]), yy - 8)
            c.setLineWidth(1)
        c.setFillColor(GRAY)
        c.circle(sx(exp), yy - 8, 5, fill=1, stroke=0)
        c.setFillColor(TEAL)
        c.circle(sx(obs), yy - 8, 7, fill=1, stroke=0)
    c.setFillColor(TEAL)
    c.circle(ox + 20, 238, 6, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.setFont("Helvetica", 10)
    c.drawString(ox + 32, 234, "observed")
    c.setFillColor(GRAY)
    c.circle(ox + 120, 238, 5, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.drawString(ox + 132, 234, "mean at random timing (bar: 2.5-97.5%, dev only)")
    bullets(
        c,
        [
            "<b>Development:</b> 6 observed vs about 4.5 expected; chance of at least 6 is 0.33. Indistinguishable from random timing.",
            "<b>Held-out:</b> 3 of 72 vs about 0.7 expected (uncorrected p about 0.035). Three events, intervals include zero, post-hoc: <b>no skill claim</b>.",
        ],
        LEFT, 215, 500, size=12.5, gap=8,
    )
    card(
        c, 560, 420, 352, 170, "LTAF low false-warning rate is not skill", [
            "392 of 398 warning starts fall in unassessable time (222 during AF, 77 washout, 93 unknown).",
            "Random placement of the same warning time would give about 0.10 assessed false warnings/h vs our 0.008.",
        ], size=12.5, accent=RED,
    )
    card(
        c, 560, 238, 352, 118, "One exception, not our finalist", [
            "RR-TCN16 patient/class: 7/241 vs 1.5 expected, fold 0 only, 1 of 8 arms. Bonferroni-adjusted p about 0.13.",
        ], size=12.5, accent=AMBER,
    )


def slide_failed(c):
    frame(c, "What failed", "Many model families, all within 1-10 of 241 development events.", 11,
          "README.md; reports/ml-investigation/{report.md, classical-calibration-diagnostic.json, chance-analysis/*.json}; src/cardiotwin/train.py; tests/test_calibration_slope.py.")
    rows = [
        ["Attempt", "Dev events", "Note"],
        ["Age/sex + ECG logistic (frozen finalist)", "6 / 241", "ECG-only reference 7 / 241: clinical fields gave no benefit"],
        ["Gradient boosting, per-patient baseline, ectopy-like motifs", "1-10 / 241", "Classical sweeps, all within chance range"],
        ["RR-TCN16 (RR-interval CNN), two arms", "4 / 241 and 7 / 241", "Only 7/241 beat random timing, fold 0 only"],
        ["Waveform CNN with causal TCN head", "4 / 241", "33% warning occupancy; window AUROC about 0.458"],
        ["Precursor ranking", "5 / 241", "Gains in fold 1 only"],
        ["ECGFounder foundation-model attempt", "no results", "0 outcome fits; last Kaggle job failed in pip setup before reading any data"],
    ]
    y = table(c, LEFT, 420, [330, 140, 394], rows, size=12)
    card(
        c, LEFT, y - 14, 864, 128, "Calibration defect found, fixed in code, not used to re-select", [
            "Fold-0 calibrators learned <b>negative slopes</b> and reversed ranking (AUROC 0.573 raw to 0.427). New slope guard in train.py::_calibrate rejects such fits; test added.",
            "Frozen finalist and reference had positive slopes, so held-out results are unaffected. No sweeps after this: 241 dev events invite winner's curse.",
        ], size=13, accent=RED,
    )


def slide_limits(c):
    frame(c, "Limitations and validation plan", "Public retrospective data; the next step needs new data and pre-declared gates.", 12,
          "Limits: README.md, reports/ml-investigation/report.md. Attribution: Gilon et al. 2023, IRIDIA-AF v1.0.1 (zenodo.org/records/8405941, CC BY 4.0); Petrutiu et al. 2007, LTAF v1.0.0 "
          "(physionet.org/content/ltafdb/1.0.0/); full text in DATA_LICENSE.md.")
    card(c, LEFT, 420, 420, 280, "Limitations", [
        "Public retrospective data; no Indian-population evidence; not a medical device.",
        "Primary and LTAF cohorts are used up: no tuning on them.",
        "Development numbers come from choosing among many candidates and are exploratory.",
        "LTAF groups are recording IDs; patient independence unverified.",
        "Target is AF onset only, not heart attack, stroke or treatment advice.",
    ], size=14, accent=RED)
    card(c, LEFT + 444, 420, 420, 280, "Future validation plan", [
        "Ethically approved Indian clinical partner and data (none secured yet).",
        "Fresh cohort with model, metrics and gates declared before looking at it.",
        "Device and patient-group calibration, then prospective silent (non-interventional) evaluation.",
        "No care use before that evidence exists.",
    ], size=14)
    bullets(
        c,
        [f"Repository: {REPO_URL}", f"Demo video: {VIDEO_URL}"],
        LEFT, 125, 864, size=13.5, gap=4,
    )


def presentation():
    c = canvas.Canvas(str(OUT / "presentation.pdf"), pagesize=(W, H))
    c.setTitle("CardioTwin Research Presentation")
    for slide in (
        slide_title, slide_problem, slide_principles, slide_architecture, slide_streams,
        slide_data, slide_eval, slide_dashboard, slide_results, slide_chance, slide_failed, slide_limits,
    ):
        slide(c)
        c.showPage()
    c.save()


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        self_check()
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        architecture()
        presentation()
        print(OUT)
