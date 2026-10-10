import os, uuid, shutil, time, json, subprocess
from flask import Flask, request, render_template, send_from_directory, redirect, abort, url_for, jsonify, send_file
from werkzeug.utils import secure_filename
import numpy as np, librosa
import analysis, db
from paths import EXPERTS as EXP, UPLOADS as UP, RESULTS as RES, ensure_dirs

ensure_dirs()
AUDIO = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".webm", ".opus", ".mp4")
app = Flask(__name__); app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
METRICS = [  # key, name, plain-English explanation
    ("pitch_score", "Pitch accuracy", "How close your notes are to the expert's. 50 cents = half a semitone."),
    ("timing_score", "Timing & rhythm", "Whether you come in early or late compared with the expert's phrasing."),
    ("steadiness_score", "Sustained notes", "How steady you hold long notes, compared with how steady the expert holds them."),
    ("vibrato_score", "Vibrato", "How closely your vibrato speed and width match the expert's."),
    ("phrasing_score", "Breath & phrasing", "Whether you breathe and pause in the same places as the expert."),
    ("dynamics_score", "Dynamics", "Whether you get louder and softer in the same places as the expert."),
    ("voicing_score", "Coverage", "Whether you sing where the expert sings and rest where they rest."),
    ("stability_score", "Micro-stability", "Frame-to-frame pitch wobble compared with the expert's."),
]

def experts():
    out = []
    for f in sorted(os.listdir(EXP)):
        if f.lower().endswith(AUDIO):
            out.append(dict(file=f, title=os.path.splitext(f)[0].replace("_", " ").replace("-", " ").title(),
                            size=f"{os.path.getsize(os.path.join(EXP, f)) / 1e6:.1f} MB"))
    return out

def grade(v):
    return ("Excellent", "great") if v >= 90 else ("Good", "good") if v >= 75 else ("Fair", "fair") if v >= 60 else ("Needs work", "poor")

def to_wav(path):
    """Browser recordings (webm/opus) and compressed formats -> 44.1 kHz WAV via ffmpeg."""
    if path.lower().endswith(".wav"): return path
    out = os.path.splitext(path)[0] + ".wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-ac", "2", "-ar", "44100", out], check=True)
    return out

def expert_path(name):
    if name not in [e["file"] for e in experts()]: abort(404)
    return os.path.join(EXP, name)

ctx = lambda **k: dict(experts=experts(), ragas=list(analysis.RAGAS), notes=analysis.NOTE, singers=db.singers(), **k)

@app.get("/")
def index(): return render_template("index.html", **ctx(msg=request.args.get("msg"), page="compare"))

@app.get("/studio")
def studio(): return render_template("studio.html", **ctx(sel=request.args.get("expert"), page="studio"))

@app.post("/expert")
def add_expert():
    f = request.files.get("file"); n = secure_filename(f.filename if f else "")
    if not n.lower().endswith(AUDIO): abort(400, "Unsupported file type")
    p = os.path.join(EXP, n); f.save(p)
    if not n.lower().endswith(".wav"):
        w = to_wav(p); os.remove(p)
    return redirect(url_for("index", msg=f"Added expert track “{n}”"))

@app.get("/api/expert/<name>")
def expert_info(name):
    """Waveform peaks + duration + auto sections (sections need the vocal stem; cached after first call)."""
    p = expert_path(name)
    y, sr = librosa.load(p, sr=8000, mono=True); n = 1200
    pk = np.abs(y[: len(y) // n * n]).reshape(n, -1).max(1)
    out = dict(duration=round(len(y) / sr, 2), peaks=[round(float(v), 3) for v in pk / (pk.max() + 1e-9)])
    if request.args.get("sections") != "0":
        out["sections"] = analysis.expert_sections(p)
        out["tempo"] = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr)[0])[0])
    return jsonify(out)

@app.get("/expert_audio/<name>/<kind>")
def expert_audio(name, kind):
    p = expert_path(name)
    if kind == "mix": return send_file(p)
    if kind == "accomp": return send_file(analysis.separate_vocals(p, want_accomp=True)[1])
    abort(404)

@app.post("/compare")
def compare():
    f = request.files.get("user"); ex = request.form.get("expert")
    if not f or not f.filename: abort(400, "Please upload or record your singing")
    if ex not in [e["file"] for e in experts()]: abort(400, "Unknown expert track")
    n = secure_filename(f.filename) or "user.wav"
    rid = uuid.uuid4().hex[:10]; d = os.path.join(RES, rid); os.makedirs(d)
    up = os.path.join(UP, rid + "_" + n); f.save(up); up = to_wav(up); t = time.time()
    sec = None
    try:
        a, b = float(request.form.get("sec_start", "")), float(request.form.get("sec_end", ""))
        if b - a > 1: sec = (round(a, 2), round(b, 2))
    except ValueError: pass
    mode = "indian" if request.form.get("mode") == "indian" else "western"
    sa = request.form.get("sa") or None; raga = request.form.get("raga") or None
    s, uv, ev = analysis.compare(up, os.path.join(EXP, ex), os.path.join(d, "plot.png"),
                                 key_tolerant=bool(request.form.get("key_tolerant")), mode=mode,
                                 sa=int(sa) if sa not in (None, "auto") else None, raga=raga if mode == "indian" else None, section=sec)
    shutil.copy(uv, os.path.join(d, "user_vocals.wav")); shutil.copy(ev, os.path.join(d, "expert_vocals.wav"))
    singer = (request.form.get("singer") or "Guest").strip()[:60]
    s.update(processing_s=round(time.time() - t, 1), user_file=n, expert_file=ex, singer=singer,
             source="recording" if request.form.get("recorded") else "upload")
    json.dump(s, open(os.path.join(d, "scores.json"), "w"))
    db.add(rid, singer, ex, mode, sec, s)
    return redirect(url_for("result", rid=rid))

@app.get("/result/<rid>")
def result(rid):
    d = os.path.join(RES, secure_filename(rid))
    if not os.path.exists(os.path.join(d, "scores.json")): abort(404)
    s = json.load(open(os.path.join(d, "scores.json"))); series = json.load(open(os.path.join(d, "plot.json")))
    return render_template("result.html", **ctx(s=s, rid=rid, series=series, grade=grade, metrics=METRICS, page="result"))

@app.get("/history")
def history():
    singer = request.args.get("singer") or (db.singers() or [None])[0]; song = request.args.get("song") or None
    rows = db.attempts(singer, song)
    return render_template("history.html", **ctx(rows=rows, singer=singer, song=song, song_list=db.songs(singer),
                                                 metrics=METRICS, grade=grade, page="history"))

@app.get("/r/<rid>/<name>")
def res(rid, name): return send_from_directory(os.path.join(RES, secure_filename(rid)), secure_filename(name))

@app.get("/healthz")
def healthz():
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), threaded=True)
