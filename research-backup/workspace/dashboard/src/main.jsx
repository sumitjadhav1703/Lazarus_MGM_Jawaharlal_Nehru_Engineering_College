import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const API = (import.meta.env.VITE_API_BASE || '').replace(/\/$/, '');
async function request(path, options = {}) {
  const response = await fetch(`${API}/api${path}`, { ...options, headers: { 'Content-Type': 'application/json', ...options.headers } });
  if (!response.ok) {
    let detail;
    try { detail = (await response.json()).detail; } catch { /* HTTP status remains useful for non-JSON failures. */ }
    throw new Error(typeof detail === 'string' ? detail : `Request failed (${response.status}). Check the API service and retry.`);
  }
  return response.json();
}
const human = value => String(value ?? '').replace(/^(ecg_|clinical_|baseline_)/, '').replaceAll('_', ' ');
const number = (value, digits = 1) => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
const percent = value => typeof value !== 'number' || !Number.isFinite(value) ? 'Unavailable' : value > 0 && value < .001 ? '<0.1%' : value < 1 && value > .999 ? '>99.9%' : `${(value * 100).toFixed(1)}%`;
const clock = seconds => {
  if (typeof seconds !== 'number' || !Number.isFinite(seconds)) return '—';
  const total = Math.floor(Math.max(0, seconds));
  return `${String(Math.floor(total / 3600)).padStart(2, '0')}:${String(Math.floor(total / 60) % 60).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
};
const featureValue = (key, value) => {
  if (value == null) return 'Unavailable';
  if (typeof value !== 'number') return String(value);
  return `${number(value, key.includes('count') ? 0 : 2)}${key === 'ecg_hr' ? ' bpm' : key === 'ecg_rr_mean' ? ' s' : ''}`;
};

function Icon({ name, size = 20 }) {
  const paths = { pulse: 'M2 12h4l3-8 5 16 3-8h5', overview: 'M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z', chart: 'M4 3v17h17 M8 15l4-6 4 3 5-7', play: 'M7 4l14 8-14 8z', pause: 'M8 4v16 M16 4v16', step: 'M5 4l12 8-12 8z M20 4v16', reset: 'M3 10a9 9 0 1 1 1 9 M3 4v6h6', shield: 'M12 2l8 4v6c0 5-8 10-8 10S4 17 4 12V6z M8 12l3 3 5-6', arrow: 'M5 12h14 M13 6l6 6-6 6', check: 'M5 12l4 4L19 6', history: 'M3 10a9 9 0 1 1 1 9 M3 4v6h6 M12 7v5l3 2' };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name] || paths.pulse} /></svg>;
}
function Badge({ children, tone = '' }) { return <span className={`badge ${tone}`}>{children}</span>; }
function Empty({ children }) { return <div className="empty">{children}</div>; }
function SectionTitle({ eyebrow, children, aside }) { return <div className="section-title"><div>{eyebrow && <span className="eyebrow">{eyebrow}</span>}<h2>{children}</h2></div>{aside}</div>; }

function Trace({ series, times, title, unit, color = 'teal', ecg = false }) {
  const values = (series || []).map(value => typeof value === 'number' && Number.isFinite(value) ? value : null);
  const valid = values.filter(value => value != null);
  if (valid.length < 2) return <Empty>{title} will appear as reliable samples become available.</Empty>;
  const low = Math.min(...valid), high = Math.max(...valid);
  const margin = Math.max((high - low) * 0.18, ecg ? 0.1 : 0.01);
  const minimum = low - margin, maximum = high + margin;
  let path = '', open = false;
  values.forEach((value, index) => {
    if (value == null) { open = false; return; }
    const x = 48 + index / Math.max(values.length - 1, 1) * 710;
    const y = 14 + (maximum - value) / (maximum - minimum) * 98;
    path += `${open ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)} `;
    open = true;
  });
  return <svg className={`trace ${ecg ? 'ecg-trace' : ''} ${color}`} viewBox="0 0 780 140" role="img" aria-label={`${title}. ${valid.length} samples; minimum ${number(low, 2)}, maximum ${number(high, 2)} ${unit}.`}>
    <title>{title}</title>
    {ecg && <defs><pattern id={`grid-${title.replaceAll(' ', '-')}`} width="14" height="14" patternUnits="userSpaceOnUse"><path d="M 14 0 L 0 0 0 14" fill="none" stroke="currentColor" strokeWidth=".5" /></pattern></defs>}
    {ecg && <rect x="48" y="7" width="710" height="109" fill={`url(#grid-${title.replaceAll(' ', '-')})`} className="ecg-grid" />}
    {[14, 63, 112].map((y, index) => <g key={y}><line x1="48" x2="758" y1={y} y2={y} className="chart-grid" /><text x="40" y={y + 4} textAnchor="end">{number(maximum - index / 2 * (maximum - minimum), ecg ? 1 : high < 10 ? 2 : 0)}</text></g>)}
    <path d={path} className="chart-line" />
    <text x="48" y="134">{clock(times?.[0])}</text><text x="758" y="134" textAnchor="end">{clock(times?.at(-1))}</text>
  </svg>;
}

function Clinical({ clinical = {} }) {
  const entries = Object.entries(clinical).filter(([key]) => key !== 'availability');
  return <dl className="clinical-grid">{entries.length ? entries.map(([key, field]) => {
    const structured = field !== null && typeof field === 'object';
    const value = structured ? field.value : field;
    const available = structured ? field.available !== false && value != null : value != null;
    return <div key={key}><dt>{human(key)}</dt><dd>{available ? typeof value === 'number' ? number(value, Number.isInteger(value) ? 0 : 1) : String(value) : 'Unavailable'}</dd>{structured && field.reason && <small>{human(field.reason)}</small>}</div>;
  }) : <div className="muted">No permitted clinical fields supplied.</div>}</dl>;
}

function Evidence({ evidence = [], baseline = {} }) {
  if (!evidence.length) return <p className="muted">Feature evidence is unavailable until a model forecast is supplied.</p>;
  return <div className="table-wrap"><table><thead><tr><th>Contributing feature</th><th>Current value</th><th>Prior baseline</th><th>Deviation</th><th>Score contribution</th></tr></thead><tbody>{evidence.slice(0, 6).map((item, index) => {
    const mean = item.baseline ?? baseline[item.feature];
    const deviation = item.deviation ?? (typeof item.value === 'number' && typeof mean === 'number' ? item.value - mean : null);
    return <tr key={`${item.feature}-${index}`}><th scope="row">{human(item.feature)}</th><td>{featureValue(item.feature, item.value)}</td><td>{featureValue(item.feature, mean)}</td><td>{number(deviation, 2)}</td><td>{typeof item.contribution === 'number' ? `${item.contribution >= 0 ? '+' : ''}${number(item.contribution, 3)}` : '—'}</td></tr>;
  })}</tbody></table></div>;
}

function WarningDetails({ warnings = [], patientId, refresh, disabled, onError }) {
  const [selected, setSelected] = useState('');
  const [note, setNote] = useState('');
  const [reviewStatus, setReviewStatus] = useState('reviewed');
  const [saving, setSaving] = useState(false);
  const warning = warnings.find(item => String(item.id ?? item.warning_id) === selected) || warnings.at(-1);
  useEffect(() => { setSelected(''); setNote(''); }, [patientId]);
  useEffect(() => { setNote(''); }, [warning?.id ?? warning?.warning_id]);
  const review = async event => {
    event.preventDefault();
    if (!warning || saving) return;
    setSaving(true);
    try {
      await request(`/patients/${encodeURIComponent(patientId)}/warnings/${encodeURIComponent(warning.id ?? warning.warning_id)}/review`, { method: 'POST', body: JSON.stringify({ status: reviewStatus, note }) });
      setNote(''); await refresh();
    } catch (error) { onError(error.message); } finally { setSaving(false); }
  };
  return <section className="card" id="warnings"><SectionTitle eyebrow="Human review" aside={<Badge tone={warnings.length ? 'amber' : ''}>{warnings.length} issued</Badge>}>Warning details</SectionTitle>
    {!warning ? <Empty>No warnings issued in this replay. Absence of a warning does not establish absence of AF risk.</Empty> : <>
      {warnings.length > 1 && <label className="field">Warning<select value={String(warning.id ?? warning.warning_id)} onChange={event => { setSelected(event.target.value); setNote(''); }}>{warnings.map(item => <option key={item.id ?? item.warning_id} value={item.id ?? item.warning_id}>{clock(item.timestamp ?? item.issued_at)} · {human(item.status ?? item.review_status ?? 'pending')}</option>)}</select></label>}
      <div className="warning-callout"><Icon name="shield" /><div><strong>Forecast warning · clinician review</strong><p>Issued at {clock(warning.timestamp ?? warning.issued_at)} · {percent(warning.risk)} score</p></div><Badge tone="amber">{human(warning.review_status ?? warning.status ?? 'pending')}</Badge></div>
      <dl className="detail-row"><div><dt>Persistence at issue</dt><dd>{warning.persistence_seconds != null ? `${number(warning.persistence_seconds, 0)} s` : 'Unavailable'}</dd></div><div><dt>Future interval</dt><dd>{clock(warning.target_start)} – {clock(warning.target_end)}</dd></div><div><dt>Warning state</dt><dd>{warning.end_time == null ? 'Active' : `Ended ${clock(warning.end_time)}`}</dd></div><div><dt>Model</dt><dd>{warning.model_version || 'Unavailable'}</dd></div></dl>
      <Evidence evidence={warning.evidence} baseline={warning.baseline?.means || warning.baseline || {}} />
      <p className="fine-print">Contributions explain the model score; they do not establish causation. Values are captured at warning issue.</p>
      {(warning.note || warning.review_note) && <p className="review-note"><strong>Review note:</strong> {warning.note || warning.review_note}</p>}
      <form className="review-form" onSubmit={review}><label className="field">Review outcome<select value={reviewStatus} onChange={event => setReviewStatus(event.target.value)}><option value="reviewed">Reviewed</option><option value="dismissed">Dismissed</option><option value="pending">Pending</option></select></label><label className="field note-field">Review note<textarea value={note} onChange={event => setNote(event.target.value)} maxLength={2000} rows={2} placeholder="Add an observation for the replay record" /></label><button className="button primary" disabled={saving || disabled}>{saving ? 'Saving…' : 'Save review'}<Icon name="check" size={16} /></button></form>
    </>}
  </section>;
}

function Timeline({ events = [] }) {
  return <section className="card" id="timeline"><SectionTitle eyebrow="Persistent patient state" aside={<Icon name="history" />}>Replay timeline</SectionTitle>{!events.length ? <Empty>Baseline updates, rejected forecasts, warnings and reviews will appear here.</Empty> : <ol className="timeline">{events.slice(-12).reverse().map((item, index) => <li key={item.id || `${item.timestamp}-${index}`}><span className={`timeline-dot ${String(item.type).includes('warning') ? 'amber' : ''}`} /><span className="mono">{clock(item.timestamp)}</span><div><strong>{human(item.type || item.event || 'State update')}</strong><p>{item.message || item.reason && human(item.reason) || item.note || (item.count != null ? `${item.count} prior reliable baseline updates` : 'Patient state updated.')}</p></div></li>)}</ol>}</section>;
}

function Evaluation({ metrics, reload, loading }) {
  if (!metrics) return <section className="card"><SectionTitle eyebrow="Untouched patient test">Evaluation</SectionTitle><Empty>Evaluation metrics could not be loaded.</Empty><button className="button" onClick={reload} disabled={loading}>Retry evaluation</button></section>;
  const unavailable = metrics.status === 'not_evaluated';
  const measured = metrics.metrics || metrics;
  const probability = measured.probability || {};
  const leadTimes = [...(measured.lead_seconds || [])].sort((a, b) => a - b);
  const medianLead = leadTimes.length ? (leadTimes[Math.floor((leadTimes.length - 1) / 2)] + leadTimes[Math.ceil((leadTimes.length - 1) / 2)]) / 2 : null;
  const intervals = measured.confidence_intervals?.intervals || {};
  const cards = [['Test patients', measured.patients, ''], ['Eligible AF onsets', measured.eligible_events, ''], ['Monitoring hours', measured.monitored_hours, 'h'], ['Event sensitivity', measured.event_sensitivity, '%'], ['False warnings / hour', measured.false_warnings_per_hour, ''], ['Warning precision', measured.warning_precision, '%'], ['Median lead time', medianLead, 's'], ['Time under warning', measured.warning_hours, 'h'], ['Missed events', measured.missed_events, ''], ['False warnings', measured.false_warnings, ''], ['Rejected monitoring', measured.rejected_hours, 'h'], ['Events missed during abstention', measured.events_missed_during_abstention, ''], ['Forecast coverage', measured.forecast_coverage, '%'], ['False warnings / patient-day', measured.false_warnings_per_patient_day, ''], ['Cooldown updates suppressed', measured.cooldown_suppressed_updates, ''], ['Brier score', probability.brier, ''], ['PR-AUC', probability.pr_auc, ''], ['Calibration error', probability.expected_calibration_error, ''], ['Inference latency · median', measured.inference_latency_ms?.median, 'ms'], ['Inference latency · p95', measured.inference_latency_ms?.p95, 'ms'], ['Warnings with unavailable outcome', measured.unassessable_warnings, '']];
  return <><section className="card evaluation-intro"><SectionTitle eyebrow="Evidence before claims" aside={<Badge tone={unavailable || metrics.synthetic ? 'amber' : 'teal'}>{unavailable ? 'Not evaluated' : metrics.synthetic ? 'Software check only' : human(metrics.status || 'Reported evaluation')}</Badge>}>Held-out evaluation</SectionTitle><p>Complete-record replay on untouched patient groups. Counts, misses, warning burden and uncertainty belong beside the headline metrics.</p>{metrics.provenance === 'synthetic' || metrics.synthetic ? <div className="notice amber"><strong>SOFTWARE CHECK ONLY</strong> · Synthetic evaluation exercises the pipeline. It does not establish clinical or Indian population performance.</div> : null}{unavailable && <div className="notice amber">No evaluation report is available. Missing results are shown as unavailable.</div>}{metrics.model_version && <p className="fine-print">Frozen model {metrics.model_version} · {human(metrics.partition)}</p>}</section>
    <div className="evaluation-grid">{cards.map(([label, value, unit]) => <div className="card metric-card" key={label}><span>{label}</span><strong>{unavailable || value == null ? 'Unavailable' : unit === '%' ? percent(value) : `${number(value, Number.isInteger(value) ? 0 : ['Brier score', 'Calibration error'].includes(label) ? 4 : 2)}${unit ? ` ${unit}` : ''}`}</strong></div>)}</div>
    {metrics.comparisons && <section className="card"><SectionTitle eyebrow="Same held-out patients">Model comparisons</SectionTitle><div className="table-wrap"><table><thead><tr><th>Model & feature group</th><th>Event sensitivity</th><th>False warnings / hour</th><th>Warning precision</th><th>Brier score</th></tr></thead><tbody>{Object.entries(metrics.comparisons).map(([name, result]) => <tr key={name}><th scope="row">{human(name)}</th><td>{percent(result.event_sensitivity)}</td><td>{number(result.false_warnings_per_hour, 3)}</td><td>{percent(result.warning_precision)}</td><td>{number(result.probability?.brier, 4)}</td></tr>)}</tbody></table></div><p className="fine-print">Clinical, ECG, combined and personal-baseline variants use the same untouched test cohort with each model’s frozen alert policy.</p></section>}
    {!unavailable && <section className="card"><SectionTitle eyebrow="Continuous replay context">Lead time & rejected monitoring</SectionTitle><p className="muted">{leadTimes.length ? `${leadTimes.length} matched events · observed lead time ${number(leadTimes[0], 0)}–${number(leadTimes.at(-1), 0)} seconds.` : 'No matched-event lead times are available.'}</p><p className="fine-print">{measured.denominator || 'Monitoring includes intervals where a forecast was unavailable.'}</p><div className="table-wrap"><table><thead><tr><th>Rejected forecast reason</th><th>Hours</th></tr></thead><tbody>{Object.entries(measured.rejection_hours || {}).map(([key, hours]) => <tr key={key}><th scope="row">{human(key)}</th><td>{number(hours, 3)}</td></tr>)}</tbody></table></div>{measured.missed_event_details?.length > 0 && <details className="evaluation-details"><summary>Inspect {measured.missed_event_details.length} missed events</summary><div className="table-wrap"><table><thead><tr><th>Patient</th><th>Record</th><th>AF onset time</th><th>All warning opportunities unavailable</th></tr></thead><tbody>{measured.missed_event_details.map((event, index) => <tr key={index}><th scope="row">{event.patient_id}</th><td>{event.record_id}</td><td>{clock(event.onset)}</td><td>{event.all_warning_opportunities_unavailable ? 'Yes' : 'No'}</td></tr>)}</tbody></table></div></details>}{measured.warning_details?.length > 0 && <details className="evaluation-details"><summary>Inspect {measured.warning_details.length} warning outcomes</summary><div className="table-wrap"><table><thead><tr><th>Patient</th><th>Issue time</th><th>End time</th><th>Outcome</th></tr></thead><tbody>{measured.warning_details.map((warning, index) => <tr key={index}><th scope="row">{warning.patient_id}</th><td>{clock(warning.timestamp)}</td><td>{clock(warning.end_time)}</td><td>{human(warning.outcome)}</td></tr>)}</tbody></table></div></details>}</section>}
    {probability.calibration?.length > 0 && <section className="card"><SectionTitle eyebrow="Probability quality">Calibration</SectionTitle><svg className="calibration-chart" viewBox="0 0 420 265" role="img" aria-label="Calibration plot comparing mean forecast probability to observed event rate. Dashed diagonal represents ideal calibration."><line x1="45" y1="220" x2="375" y2="20" stroke="#a6b8ad" strokeDasharray="5 5" />{[0, .25, .5, .75, 1].map(value => <g key={value}><line x1="45" x2="375" y1={220 - value * 200} y2={220 - value * 200} className="chart-grid" /><text x="35" y={224 - value * 200} textAnchor="end">{value.toFixed(2)}</text><text x={45 + value * 330} y="237" textAnchor="middle">{value.toFixed(2)}</text></g>)}{probability.calibration.map((bin, index) => <circle key={index} cx={45 + bin.mean_probability * 330} cy={220 - bin.event_rate * 200} r="5" fill="#12665b"><title>{`${bin.count} windows · mean score ${number(bin.mean_probability, 3)} · observed event rate ${number(bin.event_rate, 3)}`}</title></circle>)}<text x="210" y="260" textAnchor="middle">Mean forecast probability</text><text transform="translate(12 130) rotate(-90)" textAnchor="middle">Observed event rate</text></svg><p className="fine-print">Dots represent occupied probability bins; the dashed diagonal is ideal calibration. {probability.labeled_windows} labeled windows. Overlapping windows are not independent patients.</p></section>}
    <section className="card"><SectionTitle eyebrow="Scope of evidence">Uncertainty & limitations</SectionTitle>{Object.keys(intervals).length > 0 && <div className="table-wrap"><table><thead><tr><th>Patient-bootstrap interval</th><th>Lower</th><th>Upper</th></tr></thead><tbody>{Object.entries(intervals).map(([key, interval]) => <tr key={key}><th scope="row">{human(key)}</th><td>{number(interval.lower, key === 'brier' || key === 'expected_calibration_error' ? 4 : 3)}</td><td>{number(interval.upper, key === 'brier' || key === 'expected_calibration_error' ? 4 : 3)}</td></tr>)}</tbody></table></div>}{measured.confidence_intervals && <p className="fine-print">{measured.confidence_intervals.method} · {measured.confidence_intervals.samples} resamples. {measured.confidence_intervals.limitation}</p>}<ul className="limitations">{([...(metrics.uncertainty_flags || []), ...(metrics.limitations?.length ? metrics.limitations : ['Forecasting performance and clinical usefulness have not been established by this interface.', 'Historical simulation is not prospective validation.', 'Generalization to Indian patients requires a separately approved local evaluation.'])]).map((item, index) => <li key={index}>{item}</li>)}</ul>{measured.hardware && <p className="fine-print">Evaluation hardware: {typeof measured.hardware === 'string' ? measured.hardware : JSON.stringify(measured.hardware)}{measured.inference_latency_ms?.scope ? ` · Latency scope: ${measured.inference_latency_ms.scope}` : ''}</p>}</section>
  </>;
}

function App() {
  const [patients, setPatients] = useState([]), [patientId, setPatientId] = useState(''), [state, setState] = useState(null);
  const [view, setView] = useState('overview'), [metrics, setMetrics] = useState(null), [error, setError] = useState(''), [connectionError, setConnectionError] = useState('');
  const [loading, setLoading] = useState(true), [busy, setBusy] = useState(false), [connected, setConnected] = useState(false), [updatedAt, setUpdatedAt] = useState(null);
  const currentPatient = useRef(patientId);
  currentPatient.current = patientId;
  const loadMetrics = async () => { try { setMetrics(await request('/metrics')); } catch (err) { setError(err.message); } };
  const loadPatients = async () => {
    setLoading(true);
    try { const list = await request('/patients'); setPatients(list); setPatientId(current => list.some(item => item.patient_id === current) ? current : list[0]?.patient_id || ''); setError(''); }
    catch (err) { setError(err.message); } finally { setLoading(false); }
  };
  useEffect(() => { loadPatients(); loadMetrics(); }, []);
  useEffect(() => {
    setState(null); setConnected(false); setUpdatedAt(null); setConnectionError('');
    if (!patientId) return;
    const controller = new AbortController(); let timer;
    const poll = async () => {
      if (document.hidden) { timer = setTimeout(poll, 1500); return; }
      try {
        const next = await request(`/patients/${encodeURIComponent(patientId)}/state`, { signal: controller.signal });
        if (!controller.signal.aborted) { setState(next); setConnected(true); setUpdatedAt(new Date()); setConnectionError(''); }
      } catch (err) { if (!controller.signal.aborted) { setConnected(false); setConnectionError(err.message); } }
      if (!controller.signal.aborted) timer = setTimeout(poll, 1500);
    };
    poll(); return () => { controller.abort(); clearTimeout(timer); };
  }, [patientId]);
  const refresh = async () => { const next = await request(`/patients/${encodeURIComponent(patientId)}/state`); if (next.patient_id === currentPatient.current) { setState(next); setUpdatedAt(new Date()); } };
  const control = async action => {
    setBusy(true); setError('');
    try { await request(`/patients/${encodeURIComponent(patientId)}/replay`, { method: 'POST', body: JSON.stringify({ action, steps: 1 }) }); await refresh(); }
    catch (err) { setError(err.message); } finally { setBusy(false); }
  };
  const forecast = state?.forecast || {}, features = state?.features || {}, history = state?.feature_history || [];
  const accepted = state?.quality?.accepted;
  const leads = Array.isArray(state?.ecg?.leads) ? state.ecg.leads.map((values, index) => [`Lead ${index + 1}`, values]) : Object.entries(state?.ecg?.leads || {});
  const trendTimes = history.map(row => row.timestamp);
  const trend = key => history.map(row => (row.features || row)[key]);
  const progress = state?.duration_seconds ? Math.min(100, (state.timestamp || 0) / state.duration_seconds * 100) : 0;
  const disabled = busy || !state || !connected;
  return <div className="app-shell"><a className="skip-link" href="#main">Skip to main content</a>
    <aside className="sidebar"><a className="brand" href="#main"><span className="brand-mark"><Icon name="pulse" size={26} /></span><span>CardioTwin<small>CLINICIAN WORKSPACE</small></span></a><div className="workspace-label">RESEARCH WORKSPACE</div><nav aria-label="Main navigation"><button className={view === 'overview' ? 'nav-item active' : 'nav-item'} aria-current={view === 'overview' ? 'page' : undefined} onClick={() => setView('overview')}><Icon name="overview" />Patient overview<Icon name="arrow" size={16} /></button><button className={view === 'evaluation' ? 'nav-item active' : 'nav-item'} aria-current={view === 'evaluation' ? 'page' : undefined} onClick={() => setView('evaluation')}><Icon name="chart" />Evaluation<Icon name="arrow" size={16} /></button></nav><div className="sidebar-note"><Icon name="shield" size={22} /><strong>Research prototype</strong><p>Patient-specific state.<br />Frozen model weights.<br />Clinician-reviewed warnings.</p><span>AF onset forecasting</span></div><div className="sidebar-footer">CARDIOTWIN / 01<br /><span>Historical ECG simulation</span></div></aside>
    <div className="workspace"><header className="topbar"><span className="breadcrumb">Workspace <span>/</span> {view === 'overview' ? 'Patient overview' : 'Evaluation'}</span><div className="topbar-right"><span className={`connection ${connected ? 'connected' : ''}`}>{connected ? 'API connected' : loading ? 'Connecting' : 'API unavailable'}</span><span className="avatar" aria-hidden="true">CT</span></div></header>
      <main id="main"><div className="page-heading"><div><span className="eyebrow">PATIENT-SPECIFIC DIGITAL TWIN</span><h1>{view === 'overview' ? 'A clearer view of the next interval.' : 'Evidence behind the forecast.'}</h1><p>{view === 'overview' ? 'Review the ECG, evolving baseline and future AF onset forecast in one place.' : 'Patient-separated evaluation, with the complete monitoring context.'}</p></div><Badge tone="teal">Historical simulation</Badge></div>
      <div className="simulation-banner"><Icon name="history" size={19} /><p><strong>Historical replay</strong> · {state?.provenance === 'synthetic' ? 'Synthetic UI example — generated ECG and clinical profiles; no clinical performance claim.' : state?.provenance === 'SHDB-AF' ? 'SHDB-AF recordings — research simulation, not live monitoring.' : 'Research simulation — data provenance will appear when the API supplies a patient.'}</p></div>
      {(error || connectionError) && <div className="notice error" role="alert"><span>{error || connectionError} {state && !connected ? 'Displayed data may be stale.' : ''}</span><button className="button small" onClick={() => { loadPatients(); loadMetrics(); }}>Retry</button></div>}
      {view === 'evaluation' ? <Evaluation metrics={metrics} reload={loadMetrics} /> : <>
        <section className="patient-strip"><div className="patient-identity"><span className="patient-symbol"><Icon name="pulse" size={24} /></span><div><label htmlFor="patient-select" className="eyebrow">SELECT PATIENT</label><select id="patient-select" value={patientId} onChange={event => setPatientId(event.target.value)} disabled={!patients.length || busy}>{!patients.length && <option value="">{loading ? 'Loading patients…' : 'No patients available'}</option>}{patients.map(item => <option key={item.patient_id} value={item.patient_id}>{item.patient_id}</option>)}</select><small>Record {state?.record_id || '—'} · pseudonymous identity</small></div></div><div className="replay-position"><span className="eyebrow">RECORD TIME</span><strong className="mono">{clock(state?.timestamp)}<span> / {clock(state?.duration_seconds)}</span></strong><div className="progress-track" role="progressbar" aria-label="Replay position" aria-valuenow={Math.round(progress)} aria-valuemin={0} aria-valuemax={100}><span style={{ width: `${progress}%` }} /></div></div><div className="replay-controls"><Badge tone={state?.replay_status === 'running' ? 'teal' : ''}>{human(state?.replay_status || 'Awaiting data')}</Badge><div><button className="button primary" onClick={() => control(state?.replay_status === 'running' ? 'pause' : 'start')} disabled={disabled || state?.replay_status === 'complete'}><Icon name={state?.replay_status === 'running' ? 'pause' : 'play'} size={16} />{busy ? 'Updating…' : state?.replay_status === 'running' ? 'Pause' : 'Start replay'}</button><button className="button icon-button" aria-label="Advance one replay step" title="Advance one step" onClick={() => control('step')} disabled={disabled || state?.replay_status === 'running' || state?.replay_status === 'complete'}><Icon name="step" size={17} /></button><button className="button icon-button" aria-label="Reset patient replay" title="Reset replay" onClick={() => control('reset')} disabled={disabled}><Icon name="reset" size={17} /></button></div><small>Playback cadence set by the API</small></div></section>
        {!patients.length && !loading && <section className="card"><Empty>No replay records are available. Generate the synthetic demo or configure the API data directory.</Empty></section>}
        <div className="overview-grid"><section className="card signal-card"><SectionTitle eyebrow="Signal & physiology" aside={<Badge tone={accepted === true ? 'teal' : 'amber'}>{accepted === true ? 'Quality accepted' : accepted === false ? 'Quality rejected' : 'Quality unavailable'}</Badge>}>Recent ECG</SectionTitle><div className="signal-meta"><span>{state?.ecg?.fs ? `${number(state.ecg.fs, 0)} Hz source` : 'Sampling rate unavailable'} · last 10 seconds</span><span>{leads.length} leads · record-relative time</span></div>{leads.length ? leads.map(([name, values]) => <div className="lead" key={name}><span className="lead-label">{name}<small>amplitude</small></span><Trace title={name} series={values} times={state.ecg.times} unit="source units" ecg /></div>) : <Empty>Start or step the historical replay to inspect the ECG signal.</Empty>}<div className="quality-details"><span>Missing samples <strong>{percent(state?.quality?.missing_fraction ?? features.quality_missing)}</strong></span><span>Lead disagreement <strong>{number(state?.quality?.lead_disagreement ?? features.quality_lead_disagreement, 2)}</strong></span>{state?.quality?.reason && <span className="quality-reason">{human(state.quality.reason)}</span>}</div><div className="trend-grid"><div><div className="trend-heading"><span>Heart rate</span><strong>{number(features.ecg_hr, 0)}<small> bpm</small></strong></div><Trace series={trend('ecg_hr')} times={trendTimes} title="Heart rate trend" unit="bpm" /></div><div><div className="trend-heading"><span>Mean RR interval</span><strong>{number(features.ecg_rr_mean, 2)}<small> s</small></strong></div><Trace series={trend('ecg_rr_mean')} times={trendTimes} title="RR interval trend" unit="seconds" color="navy" /></div></div></section>
          <section className="card forecast-card"><SectionTitle eyebrow="Future AF onset" aside={<Icon name="arrow" />}>Forecast</SectionTitle><Badge tone={forecast.available ? 'teal' : 'amber'}>{forecast.available ? 'Forecast available' : 'Forecast unavailable'}</Badge><div className="risk-number">{forecast.available ? <>{typeof forecast.risk === 'number' ? percent(forecast.risk).replace('%', '') : '—'}<span>%</span></> : '—'}</div><p className="risk-caption">Calibrated model score</p><div className="score-track"><span style={{ width: `${forecast.available && typeof forecast.risk === 'number' ? Math.max(0, Math.min(100, forecast.risk * 100)) : 0}%` }} /></div><dl className="forecast-details"><div><dt>Future target interval</dt><dd>{clock(forecast.target_start)} – {clock(forecast.target_end)}</dd></div><div><dt>Forecast issued</dt><dd>{clock(forecast.timestamp)}</dd></div><div><dt>Model version</dt><dd>{forecast.model_version || 'Not loaded'}</dd></div></dl><div className="forecast-context"><strong>{forecast.available ? 'Uncertainty context' : human(forecast.reason || 'Awaiting replay data')}</strong><p>{forecast.uncertainty || 'A forecast is unavailable until history, signal quality and model requirements are met.'}</p></div><p className="fine-print">A low score is not reassurance. This model estimates a new AF onset in a future interval; it does not forecast heart attack or stroke.</p></section>
          <section className="card clinical-card"><SectionTitle eyebrow="Permitted prior history" aside={<Badge>Baseline profile</Badge>}>Patient context</SectionTitle><Clinical clinical={state?.clinical} /><div className="baseline-strip"><Icon name="pulse" /><div><strong>Evolving personal baseline</strong><p>{state?.baseline?.count ?? 0} prior reliable updates · {forecast.reason?.includes('history') || forecast.reason?.includes('warm') ? 'warming up' : state ? 'availability governed by forecast checks' : 'awaiting data'}</p></div></div>{Object.keys(state?.baseline?.means || {}).length > 0 && <dl className="baseline-values">{Object.entries(state.baseline.means).slice(0, 4).map(([key, value]) => <div key={key}><dt>{human(key)}</dt><dd>{featureValue(key, value)}</dd></div>)}</dl>}</section>
          <section className="card evidence-card"><SectionTitle eyebrow="Transparent model output">Forecast evidence</SectionTitle><Evidence evidence={forecast.evidence} baseline={forecast.baseline?.means || state?.baseline?.means} /><p className="fine-print">Personal deviations use earlier reliable samples. Model weights remain frozen throughout replay.</p></section>
        </div>
        <WarningDetails warnings={state?.warnings} baseline={state?.baseline?.means} patientId={patientId} refresh={refresh} disabled={disabled} onError={setError} /><Timeline events={state?.timeline} />
      </>}
      <footer className="page-footer"><span>CardioTwin · Research and demonstration only</span><span>{updatedAt ? `Last API update ${updatedAt.toLocaleTimeString()}` : 'Awaiting API data'}</span></footer>
      <span className="sr-only" aria-live="polite">{busy ? 'Updating replay' : connected ? 'Patient data connected' : 'Patient data disconnected'}</span>
      </main></div></div>;
}

createRoot(document.getElementById('root')).render(<App />);
