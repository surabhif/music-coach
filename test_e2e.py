import json
from analysis import compare
files=["identical","slightly_off","detuned","octave_down","transposed_down2","transposed_up3","flat_dynamics","wobbly","extra_breath"]
print(f"{'case':18}{'tol':>4}{'overall':>8}{'pitch':>7}{'%50c':>7}{'MAE':>7}{'shift':>6}{'stead':>7}{'vib':>6}{'phr':>6}{'dyn':>6}{'notes ok':>9}")
for f in files:
    for tol in (True,False):
        s,_,_=compare(f"test_user_{f}.wav","experts/demo_expert.wav",f"results/test_{f}_{int(tol)}.png",key_tolerant=tol)
        print(f"{f:18}{'on' if tol else 'off':>4}{s['overall']:8}{s['pitch_score']:7}{s['pct_within_50c']:7}{s['mean_abs_cents']:7}{s['semitone_shift']:6}"
              f"{s['steadiness_score']:7}{s['vibrato_score']:6}{s['phrasing_score']:6}{s['dynamics_score']:6}{str(s['notes_good'])+'/'+str(s['notes_total']):>9}")
