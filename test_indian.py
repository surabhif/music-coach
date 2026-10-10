from analysis import compare
print(f"{'case':22}{'mode':>9}{'raga':>8}{'overall':>8}{'pitch':>7}{'%50c':>7}{'MAE':>7}{'Sa':>4}{'offraga':>8}  first note names")
for f in ["yaman_meend","yaman_offraga"]:
    for mode,raga in [("western",None),("indian",None),("indian","Yaman")]:
        s,_,_=compare(f"test_user_{f}.wav","experts/yaman_expert.wav",f"results/test_{f}_{mode}_{raga}.png",mode=mode,raga=raga)
        import json; d=json.load(open(f"results/test_{f}_{mode}_{raga}.json"))
        print(f"{f:22}{mode:>9}{str(raga):>8}{s['overall']:8}{s['pitch_score']:7}{s['pct_within_50c']:7}{s['mean_abs_cents']:7}{s['sa']:>4}{s['raga_violations']:8}  {' '.join(n['name'] for n in d['notes'][:8])}")
