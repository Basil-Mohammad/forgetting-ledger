import json, glob, numpy as np
from scipy import stats

R = {}
for f in sorted(glob.glob("/home/claude/pilot/results/*.json")):
    r = json.load(open(f)); R.setdefault(r["bench"], []).append(r)

ms = lambda a: f"{np.mean(a):.3f}±{np.std(a):.3f}"
out = {}
for b, rs in R.items():
    print(f"\n===== {b}  (n={len(rs)} seeds)")
    F0 = np.array([r["accA0"] - r["accA1"] for r in rs])
    print("accA0", ms([r["accA0"] for r in rs]), "accA1", ms([r["accA1"] for r in rs]),
          "accB", ms([r["accB1"] for r in rs]), "forgetting", ms(F0))
    tr = np.array([sum(r["E1"]["true_dL"]) for r in rs])
    print("E1 completeness rel err", ms([r["E1"]["rel_err_total"] for r in rs]),
          "| one-shot@thA / true", ms(np.array([r["E1"]["first_order_start"] for r in rs]) / tr),
          "| one-shot@thB / true", ms(np.array([r["E1"]["first_order_end"] for r in rs]) / tr))
    print("layer share", {k: round(np.mean([r["layer_share"][k] for r in rs]), 3) for k in rs[0]["layer_share"]})
    print("layer rollback gain", {k: round(np.mean([r["layer_rollback_gain"][k] for r in rs]), 3) for k in rs[0]["layer_rollback_gain"]})
    print("E2 rollback: recovered fraction of forgetting / B acc")
    for m in rs[0]["E2"]:
        row = []
        for i, fr in enumerate([0.005, 0.01, 0.05]):
            j = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20].index(fr)
            rec = np.array([(r["E2"][m][j]["accA"] - r["accA1"]) / (r["accA0"] - r["accA1"]) for r in rs])
            bb = np.array([r["E2"][m][j]["accB"] for r in rs])
            row.append(f"{fr*100:.1f}%: rec {np.mean(rec):.2f} B {np.mean(bb):.3f}")
        print(f"  {m:14s}", " | ".join(row))
    print("E3 data removal (retrain without top-k new samples): forgetting reduction / B acc")
    red = {}
    for m in rs[0]["E3"]:
        row = []
        for j, fr in enumerate([0.05, 0.10, 0.20]):
            Fn = np.array([r["accA0"] - r["E3"][m][j]["accA"] for r in rs])
            rr = (F0 - Fn) / F0; red[(m, fr)] = rr
            bb = np.array([r["E3"][m][j]["accB"] for r in rs])
            row.append(f"{int(fr*100)}%: red {np.mean(rr):+.2f}±{np.std(rr):.2f} B {np.mean(bb):.3f}")
        print(f"  {m:15s}", " | ".join(row))
    for base in ["random", "static_tracin", "high_loss", "tracin_cp3"]:
        t = stats.ttest_rel(red[("ledger_harmful", 0.10)], red[(base, 0.10)])
        print(f"   paired t (10%) ledger vs {base}: diff {np.mean(red[('ledger_harmful',0.10)]-red[(base,0.10)]):+.3f} p={t.pvalue:.4f}")
    print("rank corr early20 vs full", ms([r["rank_corr_early_full"] for r in rs]))
    print("E4 class-targeted removal (10%): gain on most-forgotten class vs other classes")
    for m in rs[0]["E4"]["runs"]:
        tg = [r["E4"]["runs"][m]["target_gain"] for r in rs]; og = [r["E4"]["runs"][m]["other_gain"] for r in rs]
        print(f"  {m:17s} target {ms(tg)}  other {ms(og)}  B {ms([r['E4']['runs'][m]['accB'] for r in rs])}")
    print("E5 online (probe set = 1000-sample memory for all methods)")
    for m in rs[0]["E5"]:
        print(f"  {m:10s} A {ms([r['E5'][m]['accA'] for r in rs])}  B {ms([r['E5'][m]['accB'] for r in rs])}")
    M = np.mean([np.array(r["interference_BxA"]) for r in rs], 0)
    print("interference matrix (rows: new-task class, cols: old-task class), mean over seeds:")
    print(np.array2string(M, precision=3, suppress_small=True))
