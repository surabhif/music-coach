"""VocalMatch — Gradio UI for Hugging Face Spaces (ZeroGPU).

Polished to match the Flask VocalMatch visual identity (Inter, indigo/violet on slate).
Demucs runs under @spaces.GPU on ZeroGPU; CPU fallback locally. Flask `app.py` stays for Docker.
"""
from __future__ import annotations

import html
import json
import os
import shutil
import traceback
import uuid

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import analysis
from paths import EXPERTS, RESULTS, ensure_dirs

ensure_dirs()

try:
    import spaces
except ImportError:
    class _SpacesStub:
        @staticmethod
        def GPU(duration=60, **_kwargs):
            def deco(fn):
                return fn
            return deco
    spaces = _SpacesStub()

if os.environ.get("SPACES_ZERO_GPU") == "1" or os.environ.get("VOCALMATCH_PRELOAD_DEMUCS") == "1":
    try:
        analysis.get_demucs_model()
        print("[vocalmatch] demucs preloaded", flush=True)
    except Exception as exc:  # pragma: no cover
        print(f"[vocalmatch] demucs preload skipped: {exc}", flush=True)

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm", ".opus")
METRICS = [
    ("pitch_score", "Pitch accuracy", "How close your notes are to the expert's. 50 cents = half a semitone.", 35),
    ("timing_score", "Timing & rhythm", "Whether you come in early or late compared with the expert's phrasing.", 15),
    ("steadiness_score", "Sustained notes", "How steady you hold long notes, compared with how steady the expert holds them.", 10),
    ("vibrato_score", "Vibrato", "How closely your vibrato speed and width match the expert's.", 5),
    ("phrasing_score", "Breath & phrasing", "Whether you breathe and pause in the same places as the expert.", 10),
    ("dynamics_score", "Dynamics", "Whether you get louder and softer in the same places as the expert.", 10),
    ("voicing_score", "Coverage", "Whether you sing where the expert sings and rest where they rest.", 10),
    ("stability_score", "Micro-stability", "Frame-to-frame pitch wobble compared with the expert's.", 5),
]
GRADE_COLORS = {
    "great": ("Excellent", "#10B981"),
    "good": ("Good", "#6366F1"),
    "fair": ("Fair", "#F59E0B"),
    "poor": ("Needs work", "#F43F5E"),
}

HEAD_HTML = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
"""

CUSTOM_CSS = """
:root {
  --vm-bg: #F8FAFC; --vm-card: #ffffff; --vm-ink: #0F172A; --vm-muted: #64748B;
  --vm-line: #E2E8F0; --vm-brand: #6366F1; --vm-brand2: #8B5CF6; --vm-soft: #EEF2FF;
  --vm-radius: 16px;
}
.gradio-container {
  max-width: 1120px !important;
  margin: 0 auto !important;
  font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif !important;
  background: var(--vm-bg) !important;
  color: var(--vm-ink) !important;
}
.gradio-container, .gradio-container * { font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif !important; }
footer, .svelte-1ipelgc, .built-with, a[href*="gradio.app"] { display: none !important; }
.main > .wrap { background: var(--vm-bg) !important; }

#vm-header {
  background: rgba(255,255,255,.92);
  border: 1px solid var(--vm-line);
  border-radius: var(--vm-radius);
  padding: 14px 20px;
  margin-bottom: 8px;
  box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px rgba(15,23,42,.04);
}
#vm-header .logo-row { display:flex; align-items:center; gap:12px; }
#vm-header .mark {
  width: 36px; height: 36px; border-radius: 10px;
  background: linear-gradient(135deg, var(--vm-brand), var(--vm-brand2));
  display: grid; place-items: center;
  box-shadow: 0 4px 14px rgba(99,102,241,.35);
}
#vm-header .wordmark { font-weight: 800; font-size: 20px; letter-spacing: -.02em; color: var(--vm-ink); }
#vm-header .wordmark b { color: var(--vm-brand); }
#vm-header .tag { color: var(--vm-muted); font-size: 13px; margin: 6px 0 0 48px; }

#vm-hero { padding: 8px 4px 4px; }
#vm-hero .pill {
  display: inline-block; background: var(--vm-soft); color: var(--vm-brand);
  font-weight: 600; font-size: 12px; padding: 6px 12px; border-radius: 99px;
  letter-spacing: .02em; margin-bottom: 10px;
}
#vm-hero h1 {
  font-size: 34px; line-height: 1.15; letter-spacing: -.03em; margin: 0 0 8px;
  font-weight: 800; color: var(--vm-ink);
}
#vm-hero p { color: var(--vm-muted); font-size: 16px; max-width: 640px; margin: 0; }

.vm-card {
  background: var(--vm-card) !important;
  border: 1px solid var(--vm-line) !important;
  border-radius: var(--vm-radius) !important;
  box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px rgba(15,23,42,.04) !important;
  padding: 8px !important;
}
.vm-step {
  display: inline-grid; place-items: center; width: 24px; height: 24px;
  border-radius: 50%; background: var(--vm-soft); color: var(--vm-brand);
  font-size: 12px; font-weight: 700; margin-right: 8px; vertical-align: middle;
}
.vm-section-title { font-size: 17px; font-weight: 700; letter-spacing: -.01em; margin: 4px 0; color: var(--vm-ink); }
.vm-section-sub { color: var(--vm-muted); font-size: 13px; margin: 0 0 10px; }

#compare-btn {
  background: linear-gradient(135deg, var(--vm-brand), var(--vm-brand2)) !important;
  border: 0 !important; color: #fff !important; font-weight: 600 !important;
  border-radius: 12px !important; padding: 14px 22px !important;
  box-shadow: 0 6px 18px rgba(99,102,241,.35) !important;
  min-height: 48px !important;
}
#compare-btn:hover { filter: brightness(1.03); transform: translateY(-1px); }

#vm-footer {
  border-top: 1px solid var(--vm-line); margin-top: 28px; padding: 20px 4px 8px;
  color: var(--vm-muted); font-size: 13px;
  display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;
}
#vm-footer b { color: var(--vm-ink); }

.vm-score-wrap { padding: 8px 4px 4px; }
.vm-gauge {
  display: grid; grid-template-columns: 200px 1fr; gap: 24px; align-items: center;
  background: #fff; border: 1px solid var(--vm-line); border-radius: var(--vm-radius);
  padding: 22px; box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 8px 24px rgba(15,23,42,.04);
}
@media (max-width: 820px) {
  .vm-gauge { grid-template-columns: 1fr; justify-items: center; text-align: center; }
  #vm-hero h1 { font-size: 28px; }
}
.vm-ring { position: relative; width: 180px; height: 180px; }
.vm-ring svg { transform: rotate(-90deg); }
.vm-ring .center { position: absolute; inset: 0; display: grid; place-items: center; align-content: center; }
.vm-ring .num { font-size: 48px; font-weight: 800; letter-spacing: -.04em; line-height: 1; color: var(--vm-ink); }
.vm-ring .of { color: var(--vm-muted); font-size: 12px; margin-top: 2px; }
.vm-badge {
  display: inline-block; font-size: 12px; font-weight: 700; padding: 4px 10px;
  border-radius: 99px; margin-bottom: 8px;
}
.vm-badge.great { background: #ECFDF5; color: #059669; }
.vm-badge.good { background: var(--vm-soft); color: var(--vm-brand); }
.vm-badge.fair { background: #FFFBEB; color: #B45309; }
.vm-badge.poor { background: #FFF1F2; color: #E11D48; }
.vm-summary h3 { margin: 0 0 8px; font-size: 16px; }
.vm-summary p { color: var(--vm-muted); font-size: 14px; line-height: 1.5; margin: 0 0 12px; }
.vm-meta {
  display: grid; grid-template-columns: 1fr 1fr; gap: 8px 16px; font-size: 13px;
}
.vm-meta div span { display: block; color: var(--vm-muted); font-size: 11px; font-weight: 500; }
.vm-meta div b { color: var(--vm-ink); font-weight: 600; }
.vm-keynote {
  margin-top: 12px; background: #EFF6FF; border: 1px solid #BFDBFE; color: #1E3A8A;
  border-radius: 12px; padding: 12px 14px; font-size: 13px; line-height: 1.5;
}
.vm-metrics {
  display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-top: 14px;
}
@media (max-width: 820px) { .vm-metrics { grid-template-columns: 1fr; } }
.vm-metric {
  background: #fff; border: 1px solid var(--vm-line); border-radius: 14px; padding: 16px 18px;
  box-shadow: 0 1px 2px rgba(15,23,42,.03);
}
.vm-metric .lbl { display:flex; justify-content:space-between; gap:8px; font-size:13px; color:var(--vm-muted); font-weight:500; }
.vm-metric .val { font-size: 26px; font-weight: 800; letter-spacing: -.03em; margin: 6px 0 8px; color: var(--vm-ink); }
.vm-metric .val small { font-size: 13px; color: var(--vm-muted); font-weight: 500; }
.vm-mbar { height: 6px; border-radius: 99px; background: #F1F5F9; overflow: hidden; margin-bottom: 10px; }
.vm-mbar i { display: block; height: 100%; border-radius: 99px; }
.vm-metric p { color: var(--vm-muted); font-size: 12.5px; line-height: 1.45; margin: 0; }
.vm-metric .d { font-size: 12px; color: var(--vm-ink); margin-top: 8px; font-weight: 500; }
.vm-metric .w { font-size: 11px; color: var(--vm-muted); margin-top: 4px; }
.vm-empty {
  color: var(--vm-muted); font-size: 14px; padding: 28px 12px; text-align: center;
  border: 1px dashed var(--vm-line); border-radius: 14px; background: #FAFAFF;
}
.vm-worst table { width: 100%; border-collapse: collapse; font-size: 13px; }
.vm-worst th { text-align: left; color: var(--vm-muted); font-weight: 600; padding: 8px 6px; border-bottom: 1px solid var(--vm-line); }
.vm-worst td { padding: 10px 6px; border-bottom: 1px solid #F1F5F9; color: var(--vm-ink); }
.vm-pill {
  display: inline-block; font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 99px; text-transform: uppercase;
}
.vm-pill.good { background:#ECFDF5; color:#059669; }
.vm-pill.sharp { background:#FFF1F2; color:#E11D48; }
.vm-pill.flat { background:#FFFBEB; color:#B45309; }
.vm-pill.early { background:#F0FDFA; color:#0F766E; }
.vm-pill.late { background:#F5F3FF; color:#6D28D9; }
.vm-pill.missed { background:#F1F5F9; color:#64748B; }
.vm-status { color: var(--vm-muted); font-size: 13px; min-height: 1.2em; margin: 4px 0 8px; }
.vm-status.busy { color: var(--vm-brand); font-weight: 600; }
.vm-status.err { color: #E11D48; font-weight: 600; }
"""


def list_experts():
    ensure_dirs()
    files = sorted(f for f in os.listdir(EXPERTS) if f.lower().endswith(AUDIO_EXTS))
    return files or ["(add an expert track)"]


def _expert_label(name: str) -> str:
    title = os.path.splitext(name)[0].replace("_", " ").replace("-", " ").title()
    try:
        size = os.path.getsize(os.path.join(EXPERTS, name)) / 1e6
        return f"{title}  ·  {name}  ·  {size:.1f} MB"
    except OSError:
        return title


def expert_choices():
    files = list_experts()
    return [(_expert_label(f), f) for f in files]


def grade(v: float):
    if v >= 90: return "great"
    if v >= 75: return "good"
    if v >= 60: return "fair"
    return "poor"


def _save_audio(audio, dest_dir, stem="user"):
    os.makedirs(dest_dir, exist_ok=True)
    if audio is None:
        raise ValueError("Please upload or record your singing.")
    if isinstance(audio, str):
        ext = os.path.splitext(audio)[1].lower() or ".wav"
        path = os.path.join(dest_dir, f"{stem}{ext}")
        shutil.copy(audio, path)
        return path
    if isinstance(audio, tuple) and len(audio) == 2:
        import soundfile as sf
        sr, data = audio
        data = np.asarray(data)
        if data.dtype in (np.int16, np.int32):
            data = data.astype(np.float32) / (np.iinfo(data.dtype).max or 1)
        else:
            data = data.astype(np.float32)
        path = os.path.join(dest_dir, f"{stem}.wav")
        sf.write(path, data, int(sr))
        return path
    raise ValueError("Unsupported audio input from Gradio.")


def _resolve_expert(expert_name, expert_upload, dest_dir):
    if expert_upload is not None:
        return _save_audio(expert_upload, dest_dir, stem="expert")
    if not expert_name or str(expert_name).startswith("("):
        raise ValueError("Choose an expert from the library or upload one.")
    path = os.path.join(EXPERTS, expert_name)
    if not os.path.isfile(path):
        raise ValueError(f"Expert track not found: {expert_name}")
    return path


def _section(start, end):
    try:
        a, b = float(start), float(end)
    except (TypeError, ValueError):
        return None
    if b - a > 1:
        return (round(a, 2), round(b, 2))
    return None


def _plotly(series, scores):
    t = series.get("t") or []
    user = series.get("user") or []
    expert = series.get("expert") or []
    lag = series.get("lag") or []
    err = series.get("err") or []
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
        vertical_spacing=0.07, subplot_titles=("Pitch contour (MIDI)", "Timing drift (s)"),
    )
    fig.add_trace(go.Scatter(x=t, y=expert, name="Expert", line=dict(color="#0F172A", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=user, name="You (aligned)", line=dict(color="#6366F1", width=1.8)), row=1, col=1)
    # Highlight sharp / flat frames when err series is present
    try:
        te = np.array(t, dtype=float)
        ue = np.array([np.nan if v is None else float(v) for v in user], dtype=float)
        ee = np.array([np.nan if v is None else float(v) for v in err], dtype=float)
        sharp = ee > 50
        flat = ee < -50
        if sharp.any():
            fig.add_trace(go.Scatter(x=te[sharp], y=ue[sharp], mode="markers", name="Sharp >50¢",
                                     marker=dict(color="#F43F5E", size=5)), row=1, col=1)
        if flat.any():
            fig.add_trace(go.Scatter(x=te[flat], y=ue[flat], mode="markers", name="Flat >50¢",
                                     marker=dict(color="#F59E0B", size=5)), row=1, col=1)
    except Exception:
        pass
    fig.add_trace(go.Scatter(x=t, y=lag, name="Lag", line=dict(color="#94A3B8", width=1.2), showlegend=False), row=2, col=1)
    fig.update_layout(
        template="plotly_white",
        height=480,
        margin=dict(l=48, r=20, t=48, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0),
        font=dict(family="Inter, system-ui, sans-serif", color="#334155", size=12),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        title=dict(text=f"Overall {scores.get('overall', '—')}/100", font=dict(size=14, color="#0F172A")),
    )
    fig.update_xaxes(title_text="Expert time (s)", row=2, col=1, gridcolor="#E2E8F0", zeroline=False)
    fig.update_yaxes(gridcolor="#E2E8F0", zeroline=False)
    return fig


def _metric_detail(scores, key):
    s = scores
    if key == "pitch_score":
        return f"{s.get('pct_within_50c', '—')}% within 50¢ · avg error {s.get('mean_abs_cents', '—')}¢"
    if key == "timing_score":
        drift = s.get("mean_timing_drift_s")
        return f"avg drift {int(round((drift or 0) * 1000))} ms" if drift is not None else ""
    if key == "steadiness_score":
        return f"spread {s.get('steadiness_user_c', '—')}¢ vs expert {s.get('steadiness_expert_c', '—')}¢"
    if key == "vibrato_score":
        vu, ve = s.get("vibrato_user") or {}, s.get("vibrato_expert") or {}
        you = f"you {vu.get('rate_hz', 0)} Hz / {vu.get('extent_c', 0)}¢" if vu.get("extent_c") else "you: no vibrato"
        exp = f"expert {ve.get('rate_hz', 0)} Hz / {ve.get('extent_c', 0)}¢" if ve.get("extent_c") else "expert: none"
        return f"{you} · {exp}"
    if key == "phrasing_score":
        extra = s.get("extra_pauses_at") or []
        base = f"{s.get('pauses_matched', 0)}/{s.get('pauses_expert', 0)} breaths matched"
        if extra:
            base += " · extra pause at " + ", ".join(f"{x}s" for x in extra)
        return base
    if key == "dynamics_score":
        return f"loudness correlation {s.get('dynamics_corr', '—')}"
    if key == "voicing_score":
        return "voiced-frame agreement"
    if key == "stability_score":
        return f"jitter {s.get('user_jitter_c', '—')}¢ vs {s.get('expert_jitter_c', '—')}¢"
    return ""


def _summary_blurb(overall: float) -> str:
    if overall >= 90:
        return "Outstanding: you're closely tracking the expert in pitch and phrasing."
    if overall >= 75:
        return "Solid performance. A few passages drift from the reference; the note map below shows exactly where."
    if overall >= 60:
        return "You've got the shape of the melody. Work through the worst moments list to tighten it up."
    return "There's a noticeable gap from the reference. Practise phrase by phrase to close the gap."


def _score_html(scores) -> str:
    overall = float(scores.get("overall") or 0)
    gkey = grade(overall)
    glabel, gcolor = GRADE_COLORS[gkey]
    # Circumference for r=78
    c = 2 * 3.14159265 * 78
    dash = c * min(1.0, max(0.0, overall / 100.0))
    notes_good = scores.get("notes_good", 0)
    notes_total = scores.get("notes_total", 0)
    meta = [
        ("Your take", html.escape(str(scores.get("user_file") or "recording"))),
        ("Expert", html.escape(str(scores.get("expert_file") or "—"))),
        ("Duration", f"{scores.get('user_dur_s', '—')}s / {scores.get('expert_dur_s', '—')}s"),
        ("Expert key", html.escape(str(scores.get("key_expert") or "—"))),
        ("Your key", html.escape(str(scores.get("key_user") or "—"))),
        ("Transposition", f"{scores.get('semitone_shift', 0):+d} st"),
        ("Processed in", f"{scores.get('processing_s', '—')}s" if scores.get("processing_s") is not None else "—"),
        ("Notes on target", f"{notes_good} of {notes_total}"),
    ]
    meta_html = "".join(f"<div><span>{k}</span><b>{v}</b></div>" for k, v in meta)
    key_note = ""
    if scores.get("key_tolerant"):
        key_note = (
            f'<div class="vm-keynote"><b>Key:</b> Expert {html.escape(str(scores.get("key_expert")))}, '
            f'You {html.escape(str(scores.get("key_user")))}. '
            f'Shift {scores.get("semitone_shift", 0):+d} semitone(s). '
            f'Scored after matching keys; individual wrong notes are still counted.</div>'
        )
    if scores.get("mode") == "indian":
        key_note = (
            f'<div class="vm-keynote"><b>Sa = {html.escape(str(scores.get("sa")))}'
            f'{" (auto-detected)" if scores.get("sa_auto") else ""}'
            f'{" · Raga " + html.escape(str(scores.get("raga"))) if scores.get("raga") else ""}'
            f': {scores.get("raga_violations", 0)} note(s) outside the raga.</b> '
            f'Glides (meend/gamak) between notes aren\'t penalised; held notes are scored.</div>'
        )
    cards = []
    for key, name, desc, weight in METRICS:
        if key not in scores:
            continue
        val = float(scores[key])
        mk = grade(val)
        ml, mc = GRADE_COLORS[mk]
        detail = html.escape(_metric_detail(scores, key))
        cards.append(
            f'<div class="vm-metric"><div class="lbl">{html.escape(name)}'
            f'<span class="vm-badge {mk}" style="margin:0;padding:2px 8px;font-size:11px">{ml}</span></div>'
            f'<div class="val">{int(round(val))}<small> /100</small></div>'
            f'<div class="vm-mbar"><i style="width:{val}%;background:{mc}"></i></div>'
            f'<p>{html.escape(desc)}</p><div class="d">{detail}</div>'
            f'<div class="w">{weight}% of overall score</div></div>'
        )
    warn = ""
    if scores.get("warnings"):
        warn = "<div class='vm-keynote' style='background:#FFFBEB;border-color:#FDE68A;color:#92400E'>" + \
               html.escape("; ".join(scores["warnings"])) + "</div>"
    return f"""
<div class="vm-score-wrap">
  <div class="vm-gauge">
    <div class="vm-ring">
      <svg width="180" height="180" viewBox="0 0 180 180">
        <circle cx="90" cy="90" r="78" fill="none" stroke="#EEF2FF" stroke-width="14"/>
        <circle cx="90" cy="90" r="78" fill="none" stroke="{gcolor}" stroke-width="14"
                stroke-linecap="round" stroke-dasharray="{dash:.2f} {c:.2f}"/>
      </svg>
      <div class="center"><div class="num">{int(round(overall))}</div><div class="of">out of 100</div></div>
    </div>
    <div class="vm-summary">
      <span class="vm-badge {gkey}">{glabel} match</span>
      <h3>Summary</h3>
      <p>{html.escape(_summary_blurb(overall))}</p>
      <div class="vm-meta">{meta_html}</div>
    </div>
  </div>
  {key_note}{warn}
  <div class="vm-metrics">{''.join(cards)}</div>
</div>
"""


def _worst_html(series, scores) -> str:
    worst = series.get("worst") or []
    notes = series.get("notes") or []
    if not worst and not notes:
        return '<div class="vm-empty">Note-by-note feedback will appear after a comparison.</div>'
    rows = worst or [n for n in notes if n.get("status") != "good"][:8]
    if not rows:
        rows = notes[:8]
    body = []
    for n in rows:
        status = str(n.get("status") or "missed")
        t = float(n.get("t") or 0)
        mins, secs = int(t // 60), t % 60
        cents = n.get("cents", "")
        cents_s = f"{cents:+.0f}¢" if isinstance(cents, (int, float)) else html.escape(str(cents or "—"))
        body.append(
            f"<tr><td>{mins}:{secs:04.1f}</td>"
            f"<td><span class='vm-pill {html.escape(status)}'>{html.escape(status)}</span></td>"
            f"<td>{html.escape(str(n.get('name') or '—'))}</td>"
            f"<td>{cents_s}</td>"
            f"<td>{html.escape(str(n.get('lag_ms', '—')))}</td></tr>"
        )
    summary = (
        f"<p style='color:#64748B;font-size:13px;margin:0 0 10px'>"
        f"<b style='color:#0F172A'>{scores.get('notes_good', 0)}</b>/{scores.get('notes_total', 0)} notes good · "
        f"key {html.escape(str(scores.get('key_user')))} → {html.escape(str(scores.get('key_expert')))} · "
        f"shift {scores.get('semitone_shift', 0)} st</p>"
    )
    return f"""
<div class="vm-worst">
  {summary}
  <table>
    <thead><tr><th>Time</th><th>Status</th><th>Note</th><th>Cents</th><th>Lag (ms)</th></tr></thead>
    <tbody>{''.join(body)}</tbody>
  </table>
</div>
"""


EMPTY_SCORE = '<div class="vm-empty">Run a comparison to see your overall score, grade, and metric breakdown.</div>'
EMPTY_WORST = '<div class="vm-empty">Worst moments and note-by-note feedback appear here after analysis.</div>'


@spaces.GPU(duration=180)
def _compare_gpu(user_path, expert_path, out_png, key_tolerant, mode, sa, raga, section):
    return analysis.compare(
        user_path, expert_path, out_png,
        key_tolerant=key_tolerant, mode=mode, sa=sa, raga=raga, section=section,
        device="cuda",
    )


def _compare_cpu(user_path, expert_path, out_png, key_tolerant, mode, sa, raga, section):
    return analysis.compare(
        user_path, expert_path, out_png,
        key_tolerant=key_tolerant, mode=mode, sa=sa, raga=raga, section=section,
        device="cpu",
    )


def run_compare(user_audio, expert_name, expert_upload, key_tolerant, style, sa_choice, raga_choice,
                sec_start, sec_end, progress=gr.Progress(track_tqdm=False)):
    rid = uuid.uuid4().hex[:10]
    work = os.path.join(RESULTS, rid)
    os.makedirs(work, exist_ok=True)
    empty_fig = go.Figure()
    empty_fig.update_layout(
        template="plotly_white", height=320, paper_bgcolor="#FFFFFF", plot_bgcolor="#F8FAFC",
        font=dict(family="Inter, system-ui, sans-serif", color="#64748B"),
        title="Pitch chart appears after analysis",
    )
    try:
        progress(0.05, desc="Preparing audio…")
        user_path = _save_audio(user_audio, work, stem="user")
        expert_path = _resolve_expert(expert_name, expert_upload, work)
        mode = "indian" if style == "Indian classical" else "western"
        sa = None
        if mode == "indian" and sa_choice and sa_choice != "Auto-detect":
            sa = analysis.NOTE.index(sa_choice) if sa_choice in analysis.NOTE else None
        raga = raga_choice if mode == "indian" and raga_choice and raga_choice != "None" else None
        section = _section(sec_start, sec_end)
        out_png = os.path.join(work, "plot.png")
        progress(0.15, desc="Isolating vocals (Demucs)…")
        use_gpu = analysis.resolve_device() == "cuda" or os.environ.get("SPACES_ZERO_GPU") == "1"
        runner = _compare_gpu if use_gpu else _compare_cpu
        import time
        t0 = time.time()
        scores, uv, ev = runner(
            user_path, expert_path, out_png,
            bool(key_tolerant), mode, sa, raga, section,
        )
        scores["processing_s"] = round(time.time() - t0, 1)
        scores["user_file"] = os.path.basename(user_path)
        scores["expert_file"] = os.path.basename(expert_path)
        progress(0.85, desc="Building report…")
        series = json.load(open(out_png[:-4] + ".json"))
        fig = _plotly(series, scores)
        score_html = _score_html(scores)
        worst_html = _worst_html(series, scores)
        user_voc = os.path.join(work, "user_vocals.wav")
        exp_voc = os.path.join(work, "expert_vocals.wav")
        shutil.copy(uv, user_voc)
        shutil.copy(ev, exp_voc)
        progress(1.0, desc="Done")
        status = f'<div class="vm-status">Analysis complete · overall <b>{scores.get("overall")}</b>/100</div>'
        return score_html, fig, worst_html, user_voc, exp_voc, status
    except Exception as exc:
        traceback.print_exc()
        status = f'<div class="vm-status err">Error: {html.escape(str(exc))}</div>'
        return EMPTY_SCORE, empty_fig, EMPTY_WORST, None, None, status


THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
    radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill="#F8FAFC",
    body_background_fill_dark="#F8FAFC",
    block_background_fill="#FFFFFF",
    block_border_color="#E2E8F0",
    block_shadow="0 1px 2px rgba(15,23,42,.04), 0 8px 24px rgba(15,23,42,.04)",
    button_primary_background_fill="linear-gradient(135deg, #6366F1, #8B5CF6)",
    button_primary_background_fill_hover="linear-gradient(135deg, #4F46E5, #7C3AED)",
    button_primary_text_color="#FFFFFF",
)

with gr.Blocks(title="VocalMatch", theme=THEME, css=CUSTOM_CSS, head=HEAD_HTML) as demo:
    gr.HTML(
        """
<div id="vm-header">
  <div class="logo-row">
    <div class="mark">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round">
        <path d="M3 12h2M7 8v8M11 4v16M15 7v10M19 10v4M21 12h0"/>
      </svg>
    </div>
    <div class="wordmark">Vocal<b>Match</b></div>
  </div>
  <div class="tag">Sing it like the pros · AI vocal coach</div>
</div>
<div id="vm-hero">
  <div class="pill">AI VOCAL COACH</div>
  <h1>How close is your voice to the pros?</h1>
  <p>Upload or record your take. VocalMatch isolates the vocals, traces every note, and scores pitch, timing, and stability against an expert recording.</p>
</div>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, elem_classes=["vm-card"]):
            gr.HTML('<div class="vm-section-title"><span class="vm-step">1</span>Upload or record</div>'
                    '<p class="vm-section-sub">Your full song with backing music is fine — we separate the vocals automatically.</p>')
            user_audio = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Your singing",
                show_label=True,
            )

            gr.HTML('<div class="vm-section-title" style="margin-top:12px"><span class="vm-step">2</span>Choose an expert</div>'
                    '<p class="vm-section-sub">The reference performance to compare against — or upload your own.</p>')
            choices = expert_choices()
            expert_name = gr.Radio(
                choices=choices,
                value=choices[0][1] if choices else None,
                label="Expert library",
            )
            expert_upload = gr.Audio(sources=["upload"], type="filepath", label="Or upload an expert track")

            gr.HTML('<div class="vm-section-title" style="margin-top:12px"><span class="vm-step">3</span>Options</div>'
                    '<p class="vm-section-sub">Style and scoring preferences.</p>')
            style = gr.Radio(
                choices=["Western", "Indian classical"],
                value="Western",
                label="Style",
            )
            key_tolerant = gr.Checkbox(value=True, label="Key-tolerant (allow singing in a different key or octave)")
            with gr.Accordion("Advanced options", open=False):
                sa_choice = gr.Dropdown(
                    choices=["Auto-detect"] + analysis.NOTE, value="Auto-detect", label="Sa (tonic)"
                )
                raga_choice = gr.Dropdown(
                    choices=["None"] + list(analysis.RAGAS), value="None", label="Raga (optional)"
                )
                with gr.Row():
                    sec_start = gr.Number(label="Section start (s)", value=None, precision=2)
                    sec_end = gr.Number(label="Section end (s)", value=None, precision=2)
                gr.Markdown(
                    "<span style='color:#64748B;font-size:12px'>Section bounds crop the expert. "
                    "Leave blank to score the full track. First run downloads Demucs weights "
                    "(30–90s on CPU / faster on ZeroGPU).</span>"
                )

            run_btn = gr.Button("Analyze my singing →", variant="primary", elem_id="compare-btn")
            status = gr.HTML('<div class="vm-status">Ready when you are.</div>')

            gr.HTML(
                """
<div class="vm-card" style="margin-top:12px;padding:16px 18px !important">
  <div class="vm-section-title" style="font-size:15px;margin-bottom:6px">How it works</div>
  <ol style="margin:0;padding-left:18px;color:#64748B;font-size:13px;line-height:1.55">
    <li><b style="color:#0F172A">Vocal isolation</b>: Demucs separates the voice from the instruments.</li>
    <li><b style="color:#0F172A">Pitch tracking</b>: every frame of the voice is mapped to a musical note.</li>
    <li><b style="color:#0F172A">Alignment</b>: dynamic time warping lines up your phrasing with the expert's.</li>
    <li><b style="color:#0F172A">Scoring</b>: pitch, timing, stability and coverage into one overall score.</li>
  </ol>
</div>
                """
            )

        with gr.Column(scale=6, elem_classes=["vm-card"]):
            gr.HTML('<div class="vm-section-title">Results</div>'
                    '<p class="vm-section-sub">Overall score, metrics, pitch chart, notes, and isolated vocals.</p>')
            score_html = gr.HTML(EMPTY_SCORE)
            with gr.Tabs():
                with gr.Tab("Pitch chart"):
                    chart = gr.Plot(label="Pitch alignment")
                with gr.Tab("Notes"):
                    worst_html = gr.HTML(EMPTY_WORST)
                with gr.Tab("Vocals"):
                    with gr.Row():
                        user_vocals = gr.Audio(label="Your isolated vocals", type="filepath")
                        expert_vocals = gr.Audio(label="Expert isolated vocals", type="filepath")

    run_btn.click(
        fn=run_compare,
        inputs=[user_audio, expert_name, expert_upload, key_tolerant, style, sa_choice, raga_choice,
                sec_start, sec_end],
        outputs=[score_html, chart, worst_html, user_vocals, expert_vocals, status],
    )

    gr.HTML(
        """
<div id="vm-footer">
  <div><b>VocalMatch</b> · Sing it like the pros. Prototype build.</div>
  <div>Powered by open-source Demucs · librosa · Plotly</div>
</div>
        """
    )


if __name__ == "__main__":
    demo.queue(max_size=8).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        show_error=True,
    )
