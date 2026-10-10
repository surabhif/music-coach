"""VocalMatch — Gradio UI for Hugging Face Spaces (ZeroGPU).

Heavy Demucs separation runs under @spaces.GPU on ZeroGPU; on local/CPU the
decorator is a no-op (or spaces is stubbed) and analysis falls back to CPU.
Flask `app.py` remains the local/Docker server entrypoint.
"""
from __future__ import annotations

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

# --------------------------------------------------------------------------- spaces / ZeroGPU
try:
    import spaces
except ImportError:  # local without the `spaces` package
    class _SpacesStub:
        @staticmethod
        def GPU(duration=60, **_kwargs):
            def deco(fn):
                return fn
            return deco
    spaces = _SpacesStub()

# On ZeroGPU, place Demucs on cuda at module scope (emulation outside @spaces.GPU).
# Skip eager preload on local CPU so `python gradio_app.py` starts quickly.
if os.environ.get("SPACES_ZERO_GPU") == "1" or os.environ.get("VOCALMATCH_PRELOAD_DEMUCS") == "1":
    try:
        analysis.get_demucs_model()
        print("[vocalmatch] demucs preloaded", flush=True)
    except Exception as exc:  # pragma: no cover - download may fail offline
        print(f"[vocalmatch] demucs preload skipped: {exc}", flush=True)

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm", ".opus")
METRIC_ROWS = [
    ("pitch_score", "Pitch accuracy"),
    ("timing_score", "Timing & rhythm"),
    ("steadiness_score", "Sustained notes"),
    ("vibrato_score", "Vibrato"),
    ("phrasing_score", "Breath & phrasing"),
    ("dynamics_score", "Dynamics"),
    ("voicing_score", "Coverage"),
    ("stability_score", "Micro-stability"),
]


def list_experts():
    ensure_dirs()
    files = sorted(f for f in os.listdir(EXPERTS) if f.lower().endswith(AUDIO_EXTS))
    return files or ["(add an expert track)"]


def _save_audio(audio, dest_dir, stem="user"):
    """gr.Audio may return a filepath (str) or (sample_rate, ndarray)."""
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
    if not expert_name or expert_name.startswith("("):
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
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                        vertical_spacing=0.06, subplot_titles=("Pitch (MIDI)", "Timing drift (s)"))
    fig.add_trace(go.Scatter(x=t, y=expert, name="Expert", line=dict(color="#0F172A", width=2.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=user, name="You (aligned)", line=dict(color="#0D9488", width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=lag, name="Lag", line=dict(color="#94A3B8", width=1.2), showlegend=False), row=2, col=1)
    fig.update_layout(
        title=f"Overall {scores.get('overall', '—')}/100",
        template="plotly_white",
        height=520,
        margin=dict(l=48, r=24, t=56, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", color="#334155"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
    )
    fig.update_xaxes(title_text="Expert time (s)", row=2, col=1, gridcolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#E2E8F0")
    return fig


def _notes_table(series, scores):
    notes = series.get("notes") or []
    worst = series.get("worst") or []
    worst_keys = {(w.get("t"), w.get("status"), w.get("name")) for w in worst}
    rows = []
    for n in notes:
        flag = "★" if (n.get("t"), n.get("status"), n.get("name")) in worst_keys else ""
        rows.append([
            flag,
            n.get("name", ""),
            n.get("status", ""),
            round(float(n.get("t", 0)), 2),
            round(float(n.get("end", n.get("t", 0))), 2),
            n.get("cents", ""),
            n.get("lag_ms", ""),
        ])
    headers = ["Worst", "Note", "Status", "Start (s)", "End (s)", "Cents", "Lag (ms)"]
    summary = (
        f"**Notes:** {scores.get('notes_good', 0)}/{scores.get('notes_total', 0)} good · "
        f"key {scores.get('key_user')}→{scores.get('key_expert')} · shift {scores.get('semitone_shift')} st"
    )
    if scores.get("mode") == "indian":
        summary += f" · Sa {scores.get('sa')} · raga {scores.get('raga') or 'auto'} · off-raga {scores.get('raga_violations', 0)}"
    return summary, rows, headers


def _breakdown_md(scores):
    lines = [f"### Overall **{scores.get('overall', '—')}**/100", ""]
    for key, label in METRIC_ROWS:
        if key in scores:
            lines.append(f"- **{label}:** {scores[key]}")
    if scores.get("warnings"):
        lines.append("")
        lines.append("**Warnings:** " + "; ".join(scores["warnings"]))
    return "\n".join(lines)


@spaces.GPU(duration=180)
def _compare_gpu(user_path, expert_path, out_png, key_tolerant, mode, sa, raga, section):
    """Runs on ZeroGPU when hosted there; no-op decorator locally."""
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


def run_compare(user_audio, expert_name, expert_upload, key_tolerant, indian_mode, sa_choice, raga_choice,
                sec_start, sec_end):
    rid = uuid.uuid4().hex[:10]
    work = os.path.join(RESULTS, rid)
    os.makedirs(work, exist_ok=True)
    try:
        user_path = _save_audio(user_audio, work, stem="user")
        expert_path = _resolve_expert(expert_name, expert_upload, work)
        mode = "indian" if indian_mode else "western"
        sa = None
        if mode == "indian" and sa_choice and sa_choice != "auto":
            sa = analysis.NOTE.index(sa_choice) if sa_choice in analysis.NOTE else None
        raga = raga_choice if mode == "indian" and raga_choice and raga_choice != "(none)" else None
        section = _section(sec_start, sec_end)
        out_png = os.path.join(work, "plot.png")
        use_gpu = analysis.resolve_device() == "cuda" or os.environ.get("SPACES_ZERO_GPU") == "1"
        runner = _compare_gpu if use_gpu else _compare_cpu
        scores, uv, ev = runner(
            user_path, expert_path, out_png,
            bool(key_tolerant), mode, sa, raga, section,
        )
        series = json.load(open(out_png[:-4] + ".json"))
        fig = _plotly(series, scores)
        summary, rows, _headers = _notes_table(series, scores)
        breakdown = _breakdown_md(scores)
        user_voc = os.path.join(work, "user_vocals.wav")
        exp_voc = os.path.join(work, "expert_vocals.wav")
        shutil.copy(uv, user_voc)
        shutil.copy(ev, exp_voc)
        return breakdown, fig, summary, rows, user_voc, exp_voc, ""
    except Exception as exc:
        traceback.print_exc()
        empty = go.Figure()
        empty.update_layout(title="Comparison failed", template="plotly_white", height=320)
        return f"**Error:** {exc}", empty, "", [], None, None, str(exc)


CUSTOM_CSS = """
.gradio-container { max-width: 1100px !important; }
footer { display: none !important; }
#title-block h1 {
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  font-weight: 650;
  letter-spacing: -0.02em;
  color: #0F172A;
}
#title-block p { color: #64748B; font-size: 1.05rem; }
.score-panel { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 0.5rem 0.75rem; }
"""

with gr.Blocks(
    title="VocalMatch",
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.teal,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("IBM Plex Sans"),
        font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
    ),
    css=CUSTOM_CSS,
) as demo:
    with gr.Row(elem_id="title-block"):
        gr.Markdown(
            """
# VocalMatch
Score how closely your singing matches an expert recording — pitch, timing, and voice metrics with note-by-note feedback.
            """
        )

    with gr.Row():
        with gr.Column(scale=5):
            user_audio = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Your singing (upload or record)",
            )
            expert_name = gr.Dropdown(choices=list_experts(), value=(list_experts()[0] if list_experts() else None),
                                      label="Expert library", interactive=True)
            expert_upload = gr.Audio(sources=["upload"], type="filepath", label="Or upload an expert track")
            with gr.Accordion("Options", open=True):
                key_tolerant = gr.Checkbox(value=True, label="Key-tolerant (allow transposition)")
                indian_mode = gr.Checkbox(value=False, label="Indian classical / raga mode")
                with gr.Row():
                    sa_choice = gr.Dropdown(choices=["auto"] + analysis.NOTE, value="auto", label="Sa")
                    raga_choice = gr.Dropdown(choices=["(none)"] + list(analysis.RAGAS), value="(none)", label="Raga")
                with gr.Row():
                    sec_start = gr.Number(label="Section start (s)", value=None, precision=2)
                    sec_end = gr.Number(label="Section end (s)", value=None, precision=2)
            run_btn = gr.Button("Compare", variant="primary")
            err = gr.Markdown(visible=True)

        with gr.Column(scale=6):
            breakdown = gr.Markdown(elem_classes=["score-panel"])
            chart = gr.Plot(label="Pitch alignment")
            notes_md = gr.Markdown()
            notes_df = gr.Dataframe(
                headers=["Worst", "Note", "Status", "Start (s)", "End (s)", "Cents", "Lag (ms)"],
                interactive=False,
                label="Note-by-note feedback",
            )
            with gr.Row():
                user_vocals = gr.Audio(label="Your isolated vocals", type="filepath")
                expert_vocals = gr.Audio(label="Expert isolated vocals", type="filepath")

    run_btn.click(
        fn=run_compare,
        inputs=[user_audio, expert_name, expert_upload, key_tolerant, indian_mode, sa_choice, raga_choice,
                sec_start, sec_end],
        outputs=[breakdown, chart, notes_md, notes_df, user_vocals, expert_vocals, err],
    )

    gr.Markdown(
        """
<div style="color:#94A3B8;font-size:0.9rem;margin-top:1rem">
First compare downloads Demucs weights and may take 1–3 minutes on CPU / ZeroGPU.
Flask app (<code>app.py</code>) remains available for local Docker hosting.
</div>
        """
    )


if __name__ == "__main__":
    demo.queue(max_size=8).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        show_error=True,
    )
