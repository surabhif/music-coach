"""Formant ('ah'/'oh' vowel) singing-voice synthesizer driven by a per-sample MIDI trajectory."""
import numpy as np
from scipy.signal import lfilter
SR = 44100
FORM = {"a": [(730, 90), (1090, 110), (2440, 170), (3400, 250)], "o": [(570, 80), (840, 100), (2410, 170), (3300, 250)],
        "e": [(530, 80), (1840, 120), (2480, 170), (3500, 250)]}
def _reson(x, f, bw):
    r = np.exp(-np.pi * bw / SR); th = 2 * np.pi * f / SR
    return lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], x)
def sing(midi_traj, amp_traj, vowel="a", vib_rate=5.5, vib_cents=35, seed=0):
    rng = np.random.default_rng(seed); n = len(midi_traj); t = np.arange(n) / SR
    m = midi_traj + vib_cents / 100 * np.sin(2 * np.pi * vib_rate * t) + 0.03 * np.cumsum(rng.normal(size=n)) / np.sqrt(SR) * 0
    f = 440 * 2 ** ((np.nan_to_num(m, nan=60) - 69) / 12)
    ph = np.cumsum(f) / SR; saw = 2 * (ph % 1) - 1                # glottal-ish source
    src = np.diff(np.concatenate([[0], saw])) * -1 + 0.02 * rng.normal(size=n)  # spectral tilt + breath
    y = sum(g * _reson(src, fr, bw) for (fr, bw), g in zip(FORM[vowel], (1, .6, .3, .15)))
    y = y * np.nan_to_num(amp_traj) * (~np.isnan(midi_traj))
    return y / (np.abs(y).max() + 1e-9)
def melody_traj(notes, dur, gap=0.05, glide=0.0, tempo=1.0, jitter=0.0, detune=0.0, seed=0, amps=None):
    """notes: list of midi (None = rest). glide: seconds of portamento into each note (meend)."""
    rng = np.random.default_rng(seed); mt, at = [], []; prev = None
    for i, m in enumerate(notes):
        d = dur * tempo * (1 + jitter * rng.uniform(-1, 1)); n = int(d * SR)
        if m is None:
            mt.append(np.full(n, np.nan)); at.append(np.zeros(n)); prev = None; continue
        x = np.full(n, m + detune / 100.0)
        if glide and prev is not None:
            g = min(n // 2, int(glide * SR)); x[:g] = np.linspace(prev, m, g) + detune / 100.0
        tt = np.arange(n) / SR; env = np.minimum(1, np.minimum(tt / .04, (d - tt) / .06)).clip(0)
        a = (amps[i] if amps else 1.0) * env
        if glide and prev is not None: a = np.maximum(a, 0.7 * (amps[i] if amps else 1.0) * (tt < glide))
        mt.append(x); at.append(a); prev = m + detune / 100.0
        if gap and not glide: mt.append(np.full(int(gap * SR), np.nan)); at.append(np.zeros(int(gap * SR)))
    return np.concatenate(mt), np.concatenate(at)
