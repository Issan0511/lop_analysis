"""spec §4.3 の表: α ごとの連鎖の性質（seed 0〜9 × 課題 1〜3 = 30 連鎖）。chain_stats.py の出力（full_a<α>_s<seed>_t<task>.json）を集める。
使い方: python -I chain_table.py <chain_stats の出力のフォルダ>"""
import glob, json, sys
import numpy as np
D = sys.argv[1]
rows = {}
for a in ("0.01", "0.001", "0.0001"):
    R = [json.load(open(f)) for f in sorted(glob.glob(f"{D}/full_a{a}_s*_t*.json"))]
    rows[a] = R
def mr(v, f="{:.3g}"):
    v = np.asarray(v, float); return f"{f.format(np.median(v))} [{f.format(v.min())}, {f.format(v.max())}]"
items = [
    ("連鎖の数", lambda R: str(len(R))),
    ("H（π で重み、nats）", lambda R: mr([r["H"] for r in R], "{:.4f}")),
    ("H（文脈を一様に平均）", lambda R: mr([r["H_uniform_ctx"] for r in R], "{:.4f}")),
    ("最頻遷移の確率 max p の分位（文脈一様、10/25/50/75/90%）", lambda R: "/".join(f"{x:.4f}" for x in np.median([r["maxp_uniform_q"] for r in R], 0))),
    ("E_π[max p]", lambda R: mr([r["maxp_pi_mean"] for r in R], "{:.4f}")),
    ("ラベル雑音 E_π[1 − max p]", lambda R: mr([r["nonmode_pi"] for r in R], "{:.4f}")),
    ("当てはめ後の誤差の大きさ E_π[1 − Σp²]", lambda R: mr([r["resid_pi"] for r in R], "{:.4f}")),
    ("float64 でちょうど one-hot の文脈の割合", lambda R: mr([r["frac_rows_exact_onehot"] for r in R], "{:.3f}")),
    ("π の有効な状態数 exp(H(π))", lambda R: mr([r["pi_eff_states"] for r in R], "{:.0f}")),
    ("π の 99% を持つ状態の数", lambda R: mr([r["pi_n99"] for r in R], "{:.0f}")),
    ("最初の 500 更新の異なる文脈の数（4,225 のうち）", lambda R: mr([r["distinct_ctx_500upd"] for r in R], "{:.0f}")),
    ("1 本の系列の異なる文脈の数（127 目標のうち）", lambda R: mr([r["distinct_ctx_per_seq_mean"] for r in R], "{:.1f}")),
    ("文脈の中の写し取りで当たる目標の割合（全体）", lambda R: mr([r["copy_correct_frac"] for r in R], "{:.3f}")),
    ("同（系列の後半 64 目標）", lambda R: mr([r["copy_correct_frac_second_half"] for r in R], "{:.3f}")),
    ("π のうち最頻の写像の閉路の上の質量", lambda R: mr([r["pi_mass_on_argmax_cycles"] for r in R], "{:.3f}")),
    ("系列で測った最頻でないラベルの割合", lambda R: mr([r["label_nonmode_frac_data"] for r in R], "{:.4f}")),
    ("系列で測った真の連鎖の CE − H", lambda R: mr([r["true_ce_data"] - r["H"] for r in R], "{:.2e}")),
    ("定常分布の反復が上限 1e6 に当たった数", lambda R: f"{sum(r['pi_iters'] >= 1000000 for r in R)}/{len(R)}"),
    ("最後の 1 歩の L1 の変化 d（最大）", lambda R: f"{max(r['pi_last_step_L1'] for r in R):.1e}"),
    ("連鎖の準備の時間（秒）", lambda R: mr([r["chain_sec"] for r in R], "{:.1f}")),
    ("異なる文脈が 100 以下の課題（縮退）", lambda R: f"{sum(r['distinct_ctx_500upd'] <= 100 for r in R)}/{len(R)}"),
]
print("| 量 | α = 0.01 | α = 1e−3 | α = 1e−4 |")
print("| --- | --- | --- | --- |")
for name, f in items:
    print(f"| {name} | " + " | ".join(f(rows[a]) for a in ("0.01", "0.001", "0.0001")) + " |")
print()
for a in ("0.001", "0.0001"):
    deg = [(r["seed"], r["task"], r["distinct_ctx_500upd"], round(r["H"], 6), r["cycles"][:3]) for r in rows[a] if r["distinct_ctx_500upd"] <= 100]
    print(a, "縮退:", deg)
    print(a, "上限に当たった連鎖の d:", sorted(f"{r['pi_last_step_L1']:.1e}" for r in rows[a] if r["pi_iters"] >= 1000000))
    print(a, "S-entropy |CE_true − H|（最大）:", max(abs(r["true_ce_data"] - r["H"]) for r in rows[a]))
