"""twolayer_a6_probe.py の出力（runs_a6/*.npz）の集計。PREDICTIONS.md の PB5-a〜e を採点する。
割合はすべて「上流 ÷（自分 + 上流）」を、切替時に開いた入力を持つ第 2 層の unit で和の比にしたもの。"""
import glob, sys
import numpy as np
TAUS = (1, 3, 10, 30, 50, 100, 200)
def spear(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])
files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "runs_a6/*.npz"))
score = {"a": [], "b": [], "c": [], "d": [], "e": []}
for fn in files:
    d = np.load(fn); print(f"==== {fn.split('/')[-1]}")
    tasks = sorted({int(k.split("_")[0][1:]) for k in d.files if k[0] == "t" and k[1].isdigit()})
    for t in tasks:
        p = f"t{t}_"; live = d[p + "k2"] > 0
        sh = lambda u, s: float(u[live].sum() / (u[live].sum() + s[live].sum()))
        print(f" 切替 {t}（開いた入力を持つ unit {live.sum()}・p_old {float(d[p+'p_old']):.3f}・ρ の中央値 第2層 {float(d[p+'L2_rho_med']):.3f} 第1層 {float(d[p+'L1_rho_med']):.3f}・|ḡ/G| の中央値 {float(d[p+'L2_snr_absmed']):.3f}/{float(d[p+'L1_snr_absmed']):.3f}）")
        cs = d[p + "cum_s1"].astype(float); cu = d[p + "cum_u1"].astype(float); cul = d[p + "cum_u1lin"].astype(float)
        chs = d[p + "cum_hs"].astype(float); chu = d[p + "cum_hu"].astype(float)
        Es, Eu, Eul, Ehs, Ehu = cs.mean(0), cu.mean(0), cul.mean(0), chs.mean(0), chu.mean(0)   # [len(TAUS), H]
        rows = []
        for ti, tau in enumerate(TAUS):
            pS = d[p + f"L2_pred_{tau}"] + d[p + f"L2_hist_{tau}"]; pU = d[p + f"L1_pred_{tau}"] + d[p + f"L1_hist_{tau}"]
            pSn = d[p + f"L2_pred_{tau}"]; pUn = d[p + f"L1_pred_{tau}"]
            fS = d[p + f"L2_frz_{tau}"]; fU = d[p + f"L1_frz_{tau}"]
            real = sh(Eu[ti], Es[ti]); real_new = sh(Eul[ti] - Ehu[ti], Es[ti] - Ehs[ti])
            per_draw = [sh(cu[r, ti], cs[r, ti]) for r in range(cs.shape[0])]
            mag = float((pS[live] + pU[live]).sum() / (Es[ti][live] + Eu[ti][live]).sum())
            rows.append((tau, sh(pU, pS), sh(pUn, pSn), sh(fU, fS), real, real_new, np.std(per_draw), mag,
                         float(Ehs[ti][live].sum() + Ehu[ti][live].sum()) / float((Es[ti][live] + Eu[ti][live]).sum())))
        sS, sU = d[p + "L2_snr"], d[p + "L1_snr"]; gS, gU = d[p + "L2_sgd"], d[p + "L1_sgd"]
        print(f"   SGD の 1 歩の割合 {sh(gU, gS):.2f}   信号対雑音の極限（τ→∞）の割合 {sh(sU, sS):.2f}")
        print("   τ    A6（履歴込み） A6（新しい勾配） D0凍結×期待  実(ラベル平均) 実(旧モーメント抜き) 引きの SD  予測÷実の大きさ  旧モーメント÷実")
        for r in rows:
            print(f"   {r[0]:4d}   {r[1]:+.2f}          {r[2]:+.2f}           {r[3]:+.2f}          {r[4]:+.2f}            {r[5]:+.2f}             {r[6]:.2f}       {r[7]:.2f}           {r[8]:+.3f}")
        # 毎更新の割合（ラベル平均）
        rs, ru, rul, rhs, rhu = (d[p + "rd_" + k].astype(float) for k in ("s1", "u1", "u1lin", "hs", "hu"))
        line = "   毎更新（ラベル平均・区間）: "
        for a_, b_ in ((0, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 140), (140, 200)):
            line += f"{a_+1}-{b_} {sh(ru[a_:b_].sum(0), rs[a_:b_].sum(0)):+.2f} "
        print(line)
        line = "   毎更新・旧モーメント抜き: "
        for a_, b_ in ((0, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 140), (140, 200)):
            line += f"{a_+1}-{b_} {sh((rul-rhu)[a_:b_].sum(0), (rs-rhs)[a_:b_].sum(0)):+.2f} "
        print(line)
        ti = TAUS.index(200)
        pS200 = d[p + "L2_pred_200"] + d[p + "L2_hist_200"]; pU200 = d[p + "L1_pred_200"] + d[p + "L1_hist_200"]
        print(f"   unit 間の順位相関（τ200・予測 対 ラベル平均の実）: 自分 {spear(pS200[live], Es[ti][live]):+.2f}  上流 {spear(pU200[live], Eu[ti][live]):+.2f}  全体 {spear((pS200+pU200)[live], (Es[ti]+Eu[ti])[live]):+.2f}"
              f"   上流の線形化 ÷ 実 {float(Eul[ti][live].sum()/Eu[ti][live].sum()):.3f}   下向き（ラベル平均の Δm 1 次）{float(np.mean((Es[ti]+Eu[ti])[live] < 0)):.2f}")
        r200 = rows[TAUS.index(200)]; r1 = rows[0]
        if t in (2, 3):
            score["a"].append(abs(r200[1] - r200[4]) <= 0.10); score["b"].append(r200[1] < r1[1]); score["c"].append(0.25 <= r200[4] <= 0.45)
            score["e"].append(1.2 <= r200[7] <= 2.5)
        if t == 10:
            score["d"].append(abs(r200[1] - r200[4]) <= 0.10)
print("\n採点（PREDICTIONS.md）:", {k: f"{sum(v)}/{len(v)}" for k, v in score.items()})
