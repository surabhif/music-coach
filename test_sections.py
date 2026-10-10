import soundfile as sf, numpy as np
import analysis as A
secs = A.expert_sections("experts/demo_expert.wav"); print("sections:", secs)
y, sr = sf.read("experts/demo_expert.wav"); s = secs[-1]
sf.write("/tmp/sec_identical.wav", y[int(s["start"]*sr):int(s["end"]*sr)], sr)
yu, sru = sf.read("test_user_transposed_down2.wav"); sc = 1.03   # user take was 3% slower
sf.write("/tmp/sec_trans.wav", yu[int(s["start"]*sc*sru):int(s["end"]*sc*sru)], sru)
y1 = secs[0]; sf.write("/tmp/sec_wrong.wav", y[int(y1["start"]*sr):int(y1["end"]*sr)], sr)  # sang the wrong section
for name, f in [("identical section", "/tmp/sec_identical.wav"), ("same section, user -2 st", "/tmp/sec_trans.wav"), ("wrong section sung", "/tmp/sec_wrong.wav")]:
    r, _, _ = A.compare(f, "experts/demo_expert.wav", "results/x_sec.png", section=(s["start"], s["end"]))
    print(f"{name:26} overall {r['overall']:5}  pitch {r['pitch_score']:5}  shift {r['semitone_shift']:3}  notes {r['notes_good']}/{r['notes_total']}  expert_dur {r['expert_dur_s']}")
