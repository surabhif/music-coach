"""VocalMatch core pipeline.
separate vocals (demucs) -> pitch contour (pyin) -> key/Sa detection -> shift-invariant DTW with global
transposition search -> frame + note level scoring -> extra voice metrics -> plot + JSON details."""
import os, json, hashlib, numpy as np, soundfile as sf, librosa, torch
from scipy.signal import medfilt
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from paths import CACHE, ensure_dirs

ensure_dirs()
SR = 22050; HOP = 256; STEP = 4
FPS_FULL = SR / HOP; FPS = FPS_FULL / STEP          # ~86 fps contour, ~21.5 fps for alignment
_model = None

NOTE = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
SARGAM = ["Sa", "re", "Re", "ga", "Ga", "Ma", "Ma'", "Pa", "dha", "Dha", "ni", "Ni"]   # lower = komal, ' = tivra
RAGAS = {  # pitch classes relative to Sa
    "Yaman": [0, 2, 4, 6, 7, 9, 11], "Bhairav": [0, 1, 4, 5, 7, 8, 11], "Bhupali": [0, 2, 4, 7, 9],
    "Desh": [0, 2, 4, 5, 7, 9, 10, 11], "Kafi": [0, 2, 3, 5, 7, 9, 10], "Bilawal": [0, 2, 4, 5, 7, 9, 11]}
WEIGHTS = dict(pitch_score=.35, timing_score=.15, voicing_score=.10, steadiness_score=.10,
               phrasing_score=.10, stability_score=.05, vibrato_score=.05, dynamics_score=.10)

def _hash(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()[:16]

# ---------------------------------------------------------------- separation & features
def separate_vocals(path, want_accomp=False):
    """htdemucs; writes cache/<hash>_vocals.wav and _accomp.wav (sum of non-vocal stems)."""
    global _model
    h = _hash(path); out = os.path.join(CACHE, h + "_vocals.wav"); acc = os.path.join(CACHE, h + "_accomp.wav")
    if os.path.exists(out) and (os.path.exists(acc) or not want_accomp):
        return (out, acc) if want_accomp else out
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    if _model is None:
        _model = get_model("htdemucs"); _model.eval()
    y, _ = librosa.load(path, sr=_model.samplerate, mono=False)
    if y.ndim == 1: y = np.stack([y, y])
    wav = torch.tensor(y[:2], dtype=torch.float32)
    ref = wav.mean(0); wav = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        src = apply_model(_model, wav[None], device="cpu", split=True, overlap=0.25, progress=False)[0]
    src = src * ref.std() + ref.mean()
    vi = _model.sources.index("vocals")
    sf.write(out, src[vi].numpy().T, _model.samplerate)
    sf.write(acc, sum(src[i] for i in range(len(_model.sources)) if i != vi).numpy().T, _model.samplerate)
    return (out, acc) if want_accomp else out

def pitch_contour(vocals_path):
    """Returns (t, midi with NaN=unvoiced, rms_db) at HOP; cached."""
    c = os.path.join(CACHE, _hash(vocals_path) + "_f0.npz")
    if os.path.exists(c):
        z = np.load(c); return z["t"], z["midi"], z["rms"]
    y, _ = librosa.load(vocals_path, sr=SR, mono=True)
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=1050, sr=SR, hop_length=HOP, frame_length=2048)
    midi = librosa.hz_to_midi(f0); midi[~voiced] = np.nan
    rms = librosa.amplitude_to_db(librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0], ref=1.0)
    n = min(len(midi), len(rms)); midi, rms = midi[:n], rms[:n]
    t = librosa.times_like(midi, sr=SR, hop_length=HOP)
    np.savez(c, t=t, midi=midi, rms=rms)
    return t, midi, rms

# ---------------------------------------------------------------- music helpers
_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

def pc_hist(midi):
    m = midi[~np.isnan(midi)]
    h = np.bincount(np.round(m).astype(int) % 12, minlength=12).astype(float) if len(m) else np.zeros(12)
    return h / (h.sum() + 1e-9)

def _key(midi):
    h = pc_hist(midi)
    if h.sum() == 0: return None
    return max(((np.corrcoef(h, np.roll(p, k))[0, 1], k, q) for k in range(12) for p, q in ((_MAJ, "major"), (_MIN, "minor"))))

def detect_key(midi):
    k = _key(midi); return f"{NOTE[k[1]]} {k[2]}" if k else "unknown"

def detect_sa(midi):
    """Tonic (Sa) estimate for Indian mode. Ragas are modal, so major/minor profiles mislead (Yaman reads as
    a Lydian / relative minor). Heuristic: Sa and Pa (+7) are fixed, strongly-sounded swaras and phrases
    tend to resolve on Sa -> score(pc) = h[pc] + 0.7*h[pc+7] + 0.15 if the last held note is pc
    + 0.1 if the lowest frequent note sits on pc; penalise pc whose +1/+6 neighbours are both strong (unlikely Sa)."""
    h = pc_hist(midi)
    if h.sum() == 0: return 0
    notes = segment_notes(midi, FPS_FULL, min_dur=0.15)
    last = int(round(notes[-1][2])) % 12 if notes else None
    sc = [h[p] + 0.7 * h[(p + 7) % 12] + (0.15 if p == last else 0) - 0.3 * min(h[(p + 1) % 12], h[(p + 11) % 12])
          for p in range(12)]
    return int(np.argmax(sc))

def sargam(midi_val, sa_pc):
    """Bhatkhande-style: dot above = taar saptak, dot below = mandra; lower-case = komal, Ma' = tivra.
    Madhya saptak starts at the Sa nearest above C3 for male / C4 range: we anchor at Sa of MIDI octave 4."""
    r = int(np.round(midi_val)) - 60 - sa_pc; pc = r % 12; octv = r // 12
    name = SARGAM[pc]
    if octv > 0: return name[0] + "\u0307" * octv + name[1:]
    if octv < 0: return name[0] + "\u0323" * (-octv) + name[1:]
    return name

def segment_notes(m, fps, min_dur=0.12, thresh=0.6):
    """Pitch-stable segments. Returns list of (start, end, median_midi) in frame indices of m."""
    notes = []; n = len(m); i = 0
    k = max(3, int(fps * 0.2) | 1)
    while i < n:
        if np.isnan(m[i]): i += 1; continue
        j = i
        while j < n and not np.isnan(m[j]): j += 1
        run = m[i:j]; sm = medfilt(run, k) if len(run) >= k else run
        s = 0
        for q in range(1, len(sm) + 1):
            if q == len(sm) or abs(sm[q] - np.median(sm[s:q])) > thresh:
                if (q - s) / fps >= min_dur: notes.append((i + s, i + q, float(np.median(run[s:q]))))
                s = q
        i = j
    return notes

def runs(mask):
    out = []; i = 0; n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]: j += 1
            out.append((i, j)); i = j
        else: i += 1
    return out

def vibrato(m, fps=FPS_FULL):
    """Median vibrato rate (Hz) and extent (cents peak-to-peak) over sustained notes (>=0.35 s)."""
    rates, exts = [], []
    for s, e, _ in segment_notes(m, fps, min_dur=0.35, thresh=0.8):
        x = m[s:e] * 100; x = x[int(.1 * len(x)):int(.9 * len(x))]
        if len(x) < 20: continue
        tt = np.arange(len(x)); r = x - np.polyval(np.polyfit(tt, x, 1), tt)
        spec = np.abs(np.fft.rfft(r * np.hanning(len(r)), 1024)); fr = np.fft.rfftfreq(1024, 1 / fps)
        band = (fr >= 3) & (fr <= 9)
        if spec[band].max() < 0.35 * spec[1:].max(): continue
        ext = 2 * np.sqrt(2) * np.std(r)
        if ext < 15: continue
        rates.append(fr[band][np.argmax(spec[band])]); exts.append(ext)
    return (float(np.median(rates)), float(np.median(exts))) if rates else (0.0, 0.0)

def vocal_ratio_db(mix_path, voc_path):
    y, _ = librosa.load(mix_path, sr=SR, mono=True, duration=120); v, _ = librosa.load(voc_path, sr=SR, mono=True, duration=120)
    return float(10 * np.log10((np.mean(v ** 2) + 1e-12) / (np.mean(y ** 2) + 1e-12)))

def _features(midi):
    F = np.zeros((13, len(midi)))
    for i, m in enumerate(midi):
        if np.isnan(m): F[12, i] = 1.0
        else:
            pc = m % 12; lo = int(np.floor(pc)) % 12; w = pc - np.floor(pc)
            F[lo, i] += 1 - w; F[(lo + 1) % 12, i] += w
    return F

def _cost(u, e):
    """Pitch-class distance cost (octave-invariant, graded: 1 semitone = 0.25, >=4 semitones = 1).
    Graded cost makes it cheaper to keep a slightly wrong note on the diagonal than to warp around it."""
    d = np.abs(((u[:, None] - e[None, :]) + 6) % 12 - 6)
    C = np.clip(d / 4, 0, 1)
    nu_, ne_ = np.isnan(u)[:, None], np.isnan(e)[None, :]
    C = np.where(nu_ & ne_, 0.0, np.where(nu_ | ne_, 1.0, C))
    return C

def expert_sections(expert_path):
    """Auto-detected practice sections from expert vocal phrases (pauses >= 0.4 s)."""
    t, m, _ = pitch_contour(separate_vocals(expert_path)); dur = float(t[-1])
    ph = [(s / FPS_FULL, e / FPS_FULL) for s, e in runs(~np.isnan(m)) if (e - s) / FPS_FULL > 0.3]
    gaps_ok = []
    for p in ph:
        if gaps_ok and p[0] - gaps_ok[-1][1] < 0.4: gaps_ok[-1] = (gaps_ok[-1][0], p[1])
        else: gaps_ok.append(p)
    secs = []
    for p in gaps_ok:
        if secs and secs[-1][1] - secs[-1][0] < 4: secs[-1] = (secs[-1][0], p[1])
        else: secs.append(p)
    if len(secs) <= 1 and dur > 8:  # no clear phrase breaks -> equal chunks of ~6 s
        k = int(np.ceil(dur / 6)); secs = [(dur * i / k, dur * (i + 1) / k) for i in range(k)]
    return [dict(label=f"Section {i + 1}", start=round(max(0, s - .15), 2), end=round(min(dur, e + .15), 2))
            for i, (s, e) in enumerate(secs)]

# ---------------------------------------------------------------- comparison
def compare(user_path, expert_path, out_png, key_tolerant=True, separate=True, mode="western",
            sa=None, raga=None, section=None):
    """section=(start_s, end_s) on the expert; the user recording is assumed to cover just that section."""
    uv = separate_vocals(user_path) if separate else user_path
    ev = separate_vocals(expert_path) if separate else expert_path
    tu, mu, ru = pitch_contour(uv); te, me, re_ = pitch_contour(ev)
    warnings = []
    if separate:
        for lbl, src, voc in (("your recording", user_path, uv), ("the expert track", expert_path, ev)):
            if vocal_ratio_db(src, voc) < -25:
                warnings.append(f"Very little vocal energy was separated from {lbl}; results may be unreliable.")
    t0 = 0.0
    if section:
        i0, i1 = int(section[0] * FPS_FULL), int(section[1] * FPS_FULL)
        me, re_ = me[i0:i1], re_[i0:i1]; t0 = i0 / FPS_FULL
    indian = mode == "indian"
    mu_d, me_d, ru_d, re_d = mu[::STEP], me[::STEP], ru[::STEP], re_[::STEP]
    key_u, key_e = detect_key(mu), detect_key(me)

    def align(sh):
        _, wp = librosa.sequence.dtw(C=_cost(mu_d - sh, me_d), global_constraints=True, band_rad=0.25,
                                     weights_add=np.array([0.0, 0.15, 0.15]))
        wp = wp[::-1]; a, b = mu_d[wp[:, 0]], me_d[wp[:, 1]]
        both = ~np.isnan(a) & ~np.isnan(b)
        return wp, a, b, both, a[both] - b[both]
    if key_tolerant:
        # One global integer shift (pitch class -5..+6 x octaves). Choose min median |error|, but a
        # non-octave transposition is only accepted when the singer is genuinely in tune in the new key
        # (|median residual| < 25c); otherwise a uniformly sharp/flat singer would be "forgiven".
        cands = []
        for pcs in range(-5, 7):
            wp, a, b, both, diff = align(pcs)
            if not both.any(): continue
            lg = wp[:, 0] / FPS - wp[:, 1] / FPS * len(mu_d) / len(me_d); drift_ms = float(np.mean(np.abs(lg - np.median(lg)))) * 1000
            for c in [pcs + 12 * k for k in range(-3, 4)]:
                # cost = median |cents| + 0.1 x warp drift (ms): a wrong shift forces DTW to pair neighbouring notes
                cands.append((float(np.median(np.abs(diff - c))) + 0.1 * drift_ms, abs(c), c, abs(float(np.median(diff - c))), wp, a, b, both, diff))
        ok = [x for x in cands if x[2] % 12 == 0 or x[3] < 25]
        best = min(ok or cands, key=lambda x: (x[0], x[1]))
        shift = float(best[2]); wp, a, b, both, diff = best[4:]
    else:
        shift = 0.0; wp, a, b, both, diff = align(0)
    iu, ie = wp[:, 0], wp[:, 1]
    err_full = np.full(len(a), np.nan)
    e = (a - shift - b) * 100
    if key_tolerant: e = (e + 600) % 1200 - 600
    err_full[both] = e[both]

    # expert notes (pitch-stable segments) and held-frame mask
    notes_e = segment_notes(me_d, FPS)
    held = np.zeros(len(me_d), bool)
    for s, en, _ in notes_e:
        pad = int(round((en - s) * 0.15)); held[s + pad:max(s + pad + 1, en - pad)] = True
    score_mask = both.copy()
    if indian:   # meend/gamak leniency: score held notes only, skip user glide frames too
        ash = a - shift; slope = np.abs(np.gradient(np.nan_to_num(ash, nan=0.0))) * 100
        score_mask &= held[ie] & (slope < 40)
        if score_mask.sum() < 10: score_mask = both.copy()
    err = err_full[score_mask]; abs_err = np.abs(err)
    pct50 = float(np.mean(abs_err <= 50) * 100) if len(err) else 0.0
    mae = float(np.mean(abs_err)) if len(err) else 999.0
    voicing = float(np.mean(~np.isnan(b) == ~np.isnan(a)) * 100)
    nu, ne = len(mu_d), len(me_d)
    tu_p, te_p = iu / FPS, ie / FPS
    lag = tu_p - te_p * (nu / ne); lag = lag - np.median(lag)
    timing_dev = float(np.mean(np.abs(lag)))
    def jitter(m):
        d = np.abs(np.diff(m)) * 100; d = d[~np.isnan(d) & (d < 100)]
        return float(np.median(d)) if len(d) else 0.0
    ju, je = jitter(mu), jitter(me)

    # ---- per-note feedback
    sa_pc = (int(sa) % 12) if sa not in (None, "", "auto") else detect_sa(me)
    allowed = set(RAGAS.get(raga, [])) if raga else None
    notes, mads_u, mads_e, raga_viol, loud = [], [], [], [], []
    for s, en, med in notes_e:
        k = np.where((ie >= s) & (ie < en))[0]
        if not len(k): continue
        uvals = a[k] - shift; frac = float(np.mean(~np.isnan(uvals)))
        ce = err_full[k]; ce = ce[~np.isnan(ce)]
        if indian and len(ce) > 4: ce = ce[int(len(ce) * .15):max(int(len(ce) * .85), int(len(ce) * .15) + 1)]
        cents = float(np.median(ce)) if len(ce) else None
        lg = float(np.median(lag[k]))
        st = "missed" if frac < 0.3 or cents is None else "sharp" if cents > 50 else "flat" if cents < -50 \
            else "late" if lg > 0.15 else "early" if lg < -0.15 else "good"
        sung = float(np.nanmedian(uvals)) if frac > 0 else None
        nd = dict(t=round(t0 + s / FPS, 3), end=round(t0 + en / FPS, 3), ut=round(float(tu_p[k[0]]), 3),
                  midi=round(med, 2), name=sargam(med, sa_pc) if indian else NOTE[int(round(med)) % 12] + str(int(round(med)) // 12 - 1),
                  status=st, cents=None if cents is None else round(cents), lag_ms=round(lg * 1000))
        if allowed is not None and sung is not None and (en - s) / FPS >= 0.2 and frac > 0.5 and int(round(sung - sa_pc)) % 12 not in allowed:
            nd["off_raga"] = True; nd["sung"] = SARGAM[int(round(sung - sa_pc)) % 12]
            raga_viol.append(nd)
        notes.append(nd)
        if frac > 0.5: loud.append((float(np.mean(ru_d[iu[k]][~np.isnan(uvals)])), float(np.mean(re_d[s:en]))))
        if (en - s) / FPS >= 0.3 and frac > 0.5:
            uu = uvals[~np.isnan(uvals)] * 100; ee = me_d[s:en][~np.isnan(me_d[s:en])] * 100
            mads_u.append(np.median(np.abs(uu - np.median(uu)))); mads_e.append(np.median(np.abs(ee - np.median(ee))))
    worst = []
    for nd in notes:
        if nd["status"] == "missed": worst.append((3.0, nd, "Missed note", "not sung"))
        elif nd["status"] in ("sharp", "flat"): worst.append((abs(nd["cents"]) / 50, nd, nd["status"].title(), f"{nd['cents']:+d}¢"))
        elif nd["status"] in ("early", "late"): worst.append((abs(nd["lag_ms"]) / 150, nd, nd["status"].title(), f"{nd['lag_ms']:+d} ms"))
        if nd.get("off_raga"): worst.append((2.5, nd, "Outside raga", f"sang {nd['sung']}"))
    worst = [dict(t=n["t"], ut=n["ut"], note=n["name"], issue=i, value=v, sev=round(sv, 2))
             for sv, n, i, v in sorted(worst, key=lambda x: -x[0])[:10]]

    # ---- extra voice metrics
    stead_u = float(np.median(mads_u)) if mads_u else 0.0; stead_e = float(np.median(mads_e)) if mads_e else 0.0
    vr_u, vx_u = vibrato(mu); vr_e, vx_e = vibrato(me)
    s_vib = 100.0 if (vx_u == 0 and vx_e == 0) else max(0.0, 100 - 10 * abs(vr_u - vr_e) - 0.5 * abs(vx_u - vx_e)) \
        if (vx_u and vx_e) else max(0.0, 100 - 0.8 * max(vx_u, vx_e))
    # phrasing: expert interior pauses >=0.25 s matched by user silence; user extra pauses where expert sings
    minp = int(0.25 * FPS)
    ep = [(s, e) for s, e in runs(np.isnan(me_d)) if e - s >= minp and s > 0 and e < ne]
    matched = sum(1 for s, e in ep if np.mean(np.isnan(a[(ie >= s) & (ie < e)])) >= 0.5) if ep else 0
    extra = []
    for s, e in runs(np.isnan(mu_d)):
        if e - s >= minp and s > 0 and e < nu:
            k = (iu >= s) & (iu < e)
            if k.any() and np.mean(~np.isnan(b[k])) > 0.7: extra.append(round(t0 + float(np.median(te_p[k])), 2))
    tot = len(ep) + len(extra); s_phr = 100.0 * matched / tot if tot else 100.0
    # dynamics: note-level loudness contour (mean dB per expert note vs user's aligned note) correlation
    if len(loud) >= 4:
        L = np.array(loud); lu, le = L[:, 0] - L[:, 0].mean(), L[:, 1] - L[:, 1].mean()
        if le.std() < 1.0 and lu.std() < 2.0: corr = 1.0          # expert sings flat; so does the user
        else: corr = float(np.corrcoef(lu, le)[0, 1]) if lu.std() > 1e-6 else 0.0
    else: corr = 0.0
    if not np.isfinite(corr): corr = 0.0

    sub = dict(pitch_score=0.6 * pct50 + 0.4 * max(0, 100 - mae), timing_score=max(0.0, 100 - timing_dev * 100),
               stability_score=max(0.0, 100 - abs(ju - je) * 10), voicing_score=voicing,
               steadiness_score=max(0.0, 100 - max(0.0, stead_u - stead_e) * 2), vibrato_score=s_vib,
               phrasing_score=s_phr, dynamics_score=100 * np.sqrt(max(0.0, corr)))
    overall = sum(WEIGHTS[k] * v for k, v in sub.items())
    scores = dict(overall=round(overall, 1), **{k: round(float(v), 1) for k, v in sub.items()},
                  pct_within_50c=round(pct50, 1), mean_abs_cents=round(mae, 1),
                  mean_timing_drift_s=round(timing_dev, 3), user_jitter_c=round(ju, 1), expert_jitter_c=round(je, 1),
                  steadiness_user_c=round(stead_u, 1), steadiness_expert_c=round(stead_e, 1),
                  vibrato_user=dict(rate_hz=round(vr_u, 1), extent_c=round(vx_u)), vibrato_expert=dict(rate_hz=round(vr_e, 1), extent_c=round(vx_e)),
                  pauses_expert=len(ep), pauses_matched=matched, extra_pauses_at=extra, dynamics_corr=round(corr, 2),
                  semitone_shift=int(shift), octave_shift_applied=int(np.round(shift / 12)),
                  key_user=key_u, key_expert=key_e, key_tolerant=bool(key_tolerant), mode=mode,
                  sa=NOTE[sa_pc], sa_auto=sa in (None, "", "auto"), raga=raga or None, raga_violations=len(raga_viol),
                  section=list(section) if section else None, notes_total=len(notes),
                  notes_good=sum(n["status"] == "good" for n in notes),
                  user_dur_s=round(float(tu[-1]), 1), expert_dur_s=round(len(me) / FPS_FULL, 1))
    scores["warnings"] = warnings
    details = dict(notes=notes, worst=worst, sa_pc=sa_pc, t0=t0)
    _plot(t0 + te_p, a - shift, b, err_full, lag, out_png, scores, details)
    return scores, uv, ev

def _plot(t, a, b, err, lag, out_png, s, details):
    J = lambda x: [None if not np.isfinite(v) else round(float(v), 3) for v in x]
    json.dump(dict(t=J(t), user=J(a), expert=J(b), err=J(err), lag=J(lag), **details), open(out_png[:-4] + ".json", "w"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#E2E8F0", "axes.labelcolor": "#475569",
                         "xtick.color": "#64748B", "ytick.color": "#64748B", "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    col = dict(good="#10B981", sharp="#F43F5E", flat="#F59E0B", early="#14B8A6", late="#8B5CF6", missed="#94A3B8")
    for n in details["notes"]:
        ax.add_patch(plt.Rectangle((n["t"], n["midi"] - .4), n["end"] - n["t"], .8, color=col[n["status"]], alpha=.18, lw=0))
    ax.plot(t, b, color="#0F172A", lw=2.2, label="Expert")
    ax.plot(t, a, color="#6366F1", lw=1.6, label="You (aligned)")
    sharp = err > 50; flat = err < -50
    ax.scatter(t[sharp], a[sharp], s=8, color="#F43F5E", label="Sharp >50c", zorder=3)
    ax.scatter(t[flat], a[flat], s=8, color="#F59E0B", label="Flat >50c", zorder=3)
    ax.set_ylabel("Pitch (MIDI note)"); ax.legend(loc="upper right", fontsize=8); ax.grid(color="#F1F5F9")
    ax.set_title(f"VocalMatch  ·  Overall {s['overall']}/100 | pitch {s['pitch_score']} | timing {s['timing_score']} | "
                 f"steadiness {s['steadiness_score']} | phrasing {s['phrasing_score']} | dynamics {s['dynamics_score']}")
    ax2.plot(t, lag, color="#94A3B8")
    ax2.fill_between(t, lag, 0, where=lag > 0.15, color="#8B5CF6", alpha=.35, label="Late >150ms")
    ax2.fill_between(t, lag, 0, where=lag < -0.15, color="#10B981", alpha=.35, label="Early >150ms")
    ax2.set_ylabel("Timing (s)"); ax2.set_xlabel("Expert time (s)"); ax2.legend(loc="upper right", fontsize=8); ax2.grid(color="#F1F5F9")
    ax.title.set_color("#0F172A"); fig.patch.set_facecolor("white"); fig.tight_layout(); fig.savefig(out_png, dpi=100); plt.close(fig)
