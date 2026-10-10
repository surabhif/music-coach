"""Synthesize an expert 'song' (formant voice + chords + drums) and user variants."""
import os
import numpy as np, soundfile as sf
from voice_synth import sing, melody_traj, SR
mel = [60,62,64,65,67,67,69,67,65,64,62,60, None, 64,65,67,72,71,69,67,65,64,62,60,60]
amps = [0.6,0.7,0.8,0.9,1,1,1,.9,.8,.7,.6,.5, 1, .5,.6,.8,1,1,.9,.8,.7,.6,.5,.4,.4]
chords = [[48,52,55],[53,57,60],[55,59,62],[48,52,55]]
def backing(n, seed=0):
    t = np.arange(n) / SR; y = np.zeros(n); seg = n // 4
    for i, c in enumerate(chords):
        for m in c:
            f = 440 * 2 ** ((m - 69) / 12); tt = t[i*seg:(i+1)*seg]
            y[i*seg:(i+1)*seg] += 0.05 * np.sign(np.sin(2*np.pi*f*tt)) * np.exp(-((tt - tt[0]) % 1.0) * 2)
    for k in range(0, n - 2000, int(0.5 * SR)):
        y[k:k+2000] += 0.3 * np.random.default_rng(k).normal(size=2000) * np.exp(-np.arange(2000) / 300)
    return y
def song(notes, a=amps, **kw):
    mt, at = melody_traj(notes, 0.45, amps=a, **{k: v for k, v in kw.items() if k not in ("vib_rate","vib_cents")})
    v = 0.6 * sing(mt, at, vib_rate=kw.get("vib_rate", 5.5), vib_cents=kw.get("vib_cents", 35), seed=kw.get("seed", 0))
    y = v + backing(len(v)); return np.stack([y, y], 1) / np.abs(y).max() * 0.9
flat = [0.8] * len(amps)
if __name__ == "__main__":
    os.makedirs("experts", exist_ok=True)
    sf.write("experts/demo_expert.wav", song(mel), SR)
    sf.write("test_user_identical.wav", song(mel), SR)
    sf.write("test_user_slightly_off.wav", song(mel, detune=25, tempo=1.05, jitter=.1, seed=1), SR)
    bad = list(mel); bad[3] += 2; bad[8] -= 2; bad[15] += 1; bad[20] -= 3
    sf.write("test_user_detuned.wav", song(bad, a=flat, detune=70, tempo=1.15, jitter=.3, seed=2, vib_cents=10), SR)
    sf.write("test_user_octave_down.wav", song([m - 12 if m else None for m in mel]), SR)
    sf.write("test_user_transposed_down2.wav", song([m - 2 if m else None for m in mel], detune=10, tempo=1.03, jitter=.05, seed=3), SR)
    sf.write("test_user_transposed_up3.wav", song([m + 3 if m else None for m in mel], detune=-10, tempo=0.97, jitter=.05, seed=4), SR)
    # voice-metric probes
    sf.write("test_user_flat_dynamics.wav", song(mel, a=flat), SR)                                  # same notes, no dynamics
    sf.write("test_user_wobbly.wav", song(mel, vib_rate=8.0, vib_cents=110, seed=5), SR)           # fast wide vibrato
    sf.write("test_user_extra_breath.wav", song(mel[:5] + [None] + mel[5:12] + mel[13:], a=amps[:5] + [1] + amps[5:12] + amps[13:]), SR)  # breath mid-phrase, skips the real one
    inv = [(1.05 - a) ** 2 * 1.5 for a in amps]
    sf.write("test_user_reversed_dynamics.wav", song(mel, a=inv), SR)
    # Indian classical: Raga Yaman, Sa = D (62); notes N. R G M' P D N S' ... with meend (glides)
    SA = 62; Y = [-1, 2, 4, 6, 7, 9, 11, 12, 11, 9, 7, 6, 4, 2, 0, None, 4, 6, 7, 9, 7, 6, 4, 2, 0]
    ym = [SA + d if d is not None else None for d in Y]; ya = [0.8] * len(ym)
    sf.write("experts/yaman_expert.wav", song(ym, a=ya, glide=0.12), SR)
    sf.write("test_user_yaman_meend.wav", song(ym, a=ya, glide=0.26, tempo=1.04, jitter=.05, seed=7), SR)   # same notes, longer slides
    bad = [m - 1 if (m is not None and (m - SA) % 12 == 6) else m for m in ym]                               # Ma tivra -> shuddha Ma
    sf.write("test_user_yaman_offraga.wav", song(bad, a=ya, glide=0.12, seed=8), SR)
    print("ok")
