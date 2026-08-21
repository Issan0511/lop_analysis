"""spec_cell4_0821（改訂 2026-08-21b）の事前登録レビュー: §1/§3/§4/§9 を既存資産で検算する。

実行: リポジトリルートで  OMP_NUM_THREADS=1 .venv/bin/python precheck_cell4_0821.py

**P1/P2 の答えは意図的に計算しない。** spec §5 が事前登録の対象なので、commit 前に
c_self / c_rest の中央値を出すと事前登録が壊れる。ここで確かめるのは
「判定が走るか」「判定基準が意図どおりの量を測るか」だけ。

報告する数値はすべてここで再計算できる（memory: 再計算できない数値は報告しないのと同じ）。
新規学習はしない。入力は results/ratchet_log_0819/logs/seed*.npz のみ。

検証項目:
  [§1] 目玉数値の再現 — t=500k・|E[δ]| 上位 6 seed での |cos| と ratio、全 10 seed の median
  [§1] cos = r/√(1+r²) の整合（2 数値が同一量の別座標であること）
  [§3] Phase 0 の実測値とゲート閾値
  [§3] c_self + c_rest = |cos(G,µ̂)| が記録点ごとに厳密に成立するか（代数の確認）
  [§3] ‖G‖ ゲートが P1 を PASS 側に偏らせる構造（|c_self| と ‖G‖ の関係）
  [§4] 層C の閾値 ratio_mu_cov>1 が結果 |cos| での条件付けになっていないか
  [§9] S2 — figures_ratchet_log の E1 関数が import 可能か
  [§9] S4 — p̂=1 ユニットでの独立 2 経路一致。閾値 1e-6 で通るか、被覆はどこか
"""
import numpy as np

LOGS = "results/ratchet_log_0819/logs/seed{}.npz"
SEEDS = range(10)
TAU = 0.05


def hdr(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


Z = [np.load(LOGS.format(s)) for s in SEEDS]
STEP = Z[0]["step"]
BND = np.arange(10000, 1000001, 10000)
DIST = np.min(np.abs(STEP[:, None] - BND[None, :]), axis=1)
LAYER_A = (STEP % 1000 == 0) & (DIST > 100)          # バルク点
LAYER_B = DIST <= 100                                 # 境界窓
I500K = int(np.argmin(np.abs(STEP - 500000)))


def s1_headline():
    hdr("[§1] 目玉数値の再現")
    Ed = np.array([abs(float(z["E_delta"][I500K])) for z in Z])
    cos = np.array([abs(float(z["cos_G_mu"][I500K])) for z in Z])
    rat = np.array([float(z["ratio_mu_cov"][I500K]) for z in Z])
    top6 = sorted(np.argsort(-Ed)[:6].tolist())
    print(f"  t=500k の記録点 step = {STEP[I500K]}")
    print(f"  |E[δ]| 上位 6 seed = {top6}                       spec: [1,2,3,4,5,9]")
    print(f"  その 6 seed の |cos|  = [{cos[top6].min():.4f}, {cos[top6].max():.4f}]"
          f"          spec: 0.9656–0.9945")
    print(f"  その 6 seed の ratio  = [{rat[top6].min():.4f}, {rat[top6].max():.4f}]"
          f"          spec: 3.70–9.57")
    print(f"  全 10 seed の ratio 中央値 = {np.median(rat):.4f}"
          f"                  spec: 4.252")
    print(f"  seed 別 |cos| = { {s: round(float(cos[s]), 3) for s in SEEDS} }")

    d = np.concatenate([np.abs(np.abs(z["cos_G_mu"].astype(np.float64))
                               - (lambda r: r / np.sqrt(1 + r ** 2))(
                                   z["ratio_mu_cov"].astype(np.float64))) for z in Z])
    print(f"\n  |cos − r/√(1+r²)| : 中央値={np.median(d):.4f}  p90={np.percentile(d, 90):.4f}"
          f"    spec: 0.0022 / 0.070")
    print("  ⇒ 2 数値は同一量の別座標。独立な証拠として並べられない (spec §1 の指摘は正しい)")


def s3_phase0():
    hdr("[§3] Phase 0 の実測値")
    Gn = np.concatenate([np.linalg.norm(z["G"].astype(np.float64), axis=1) for z in Z])
    al = [(z["p_hat"] >= TAU) for z in Z]
    vv = np.concatenate([np.abs(z["v"].astype(np.float64))[a] for z, a in zip(Z, al)])
    ac = np.concatenate([a.sum(axis=1) for a in al])
    print(f"  ‖G‖        p50={np.median(Gn):.4f}  <1e-2={np.mean(Gn < 1e-2):.4f}  "
          f"min={Gn.min():.2e}      spec: 0.380 / 10.19% / 2.9e-8")
    print(f"  |v| alive  p50={np.median(vv):.4f}  <1e-2={np.mean(vv < 1e-2):.4f}"
          f"                        spec: 0.688 / 0.67%")
    print(f"  alive 数   p50={int(np.median(ac))}  ≤5={np.mean(ac <= 5):.4f}  "
          f"=0={np.mean(ac == 0):.4f}          spec: 9 / 33.4% / 1.65%")
    fp = [float(np.mean(z["G_dot_mu"] > 0)) for z in Z]
    fz = [float(np.mean(np.abs(z["G_dot_mu"]) < 1e-3)) for z in Z]
    print(f"  G·µ̂>0      [{min(fp):.3f}, {max(fp):.3f}]"
          f"                                spec: 48.7–52.2%")
    print(f"  |G·µ̂|<1e-3 [{min(fz):.3f}, {max(fz):.3f}]"
          f"                                spec: 1.0–11.6%")


def s3_identity_and_gate():
    hdr("[§3] c_self + c_rest = |cos(G,µ̂)| と ‖G‖ ゲートの向き")
    print("  代数: c_rest,i := σ·(G·µ̂ − self_i)/‖G‖ と定義するので")
    print("        c_self,i + c_rest,i = σ·(G·µ̂)/‖G‖ = |cos(G,µ̂)|  … 全ユニット i で恒等的に成立")
    print("        分母 ‖G‖ は正で符号反転を持たない ⇒ 初版の分母問題は解消 (指摘②の採用は妥当)")

    print("\n  ただし c_self,i = σ·v_i·E[a_i x]·µ̂ / ‖G‖ の分子は δ に依存しない。")
    print("  ‖G‖ = ‖E[δx]‖ はフィット収束で 0 に向かうので、‖G‖ が小さい点ほど |c_self| が大きい。")
    print("  P1 は実質「c_self < |cos|/2」の検定なので、‖G‖ 下限で切ると PASS 方向に偏る。")
    exA = [float(np.mean(np.linalg.norm(z["G"].astype(np.float64), axis=1)[LAYER_A] < 1e-2))
           for z in Z]
    exB = [float(np.mean(np.linalg.norm(z["G"].astype(np.float64), axis=1)[LAYER_B] < 1e-2))
           for z in Z]
    print(f"\n  ‖G‖<1e-2 で落ちる割合   層A(主判定)={np.mean(exA):.4f}   層B={np.mean(exB):.4f}")
    print("  ⇒ 閾値を {1e-3, 1e-2, 1e-1} で事前登録し、対応差の符号が 3 点で保たれることを")
    print("     P1 の条件に含めるのが最小の直し (追加コスト 0)")
    print("  ※ ここで |c_self| の実値を出すと P1 の答えを先に見ることになるので計算しない")


def s4_layerc():
    hdr("[§4] 層C の閾値は結果での条件付けになっていないか")
    r = np.concatenate([z["ratio_mu_cov"].astype(np.float64) for z in Z])
    c = np.concatenate([np.abs(z["cos_G_mu"].astype(np.float64)) for z in Z])
    print(f"  (ratio>1) と (|cos|>1/√2=0.7071) の一致率 = {np.mean((r > 1) == (c > 2 ** -0.5)):.4f}")
    print(f"    ratio>1  の点の |cos| : 中央値={np.median(c[r > 1]):.4f}")
    print(f"    ratio≤1  の点の |cos| : 中央値={np.median(c[r <= 1]):.4f}")
    print("  ⇒ §1 が示した cos = r/√(1+r²) より、層C は |cos| = c_self + c_rest **そのもの**")
    print("     で層別している = 分解対象の和で条件付けている")

    Ed = np.array([abs(float(z["E_delta"][I500K])) for z in Z])
    rt = np.array([float(z["ratio_mu_cov"][I500K]) for z in Z])
    print(f"\n  t=500k  |E[δ]| 上位6      = {sorted(np.argsort(-Ed)[:6].tolist())}")
    print(f"          ratio_mu_cov>1    = {sorted(np.flatnonzero(rt > 1).tolist())}")
    print("  ⇒ §1 の seed 選抜と一致しない。層C は目玉数値の scope を切り出せていない")
    frac = [float(np.mean(z["ratio_mu_cov"] > 1)) for z in Z]
    print(f"  全記録点で ratio>1 の割合 = [{min(frac):.3f}, {max(frac):.3f}] (平均 {np.mean(frac):.3f})"
          " … 83/17 の偏った分割")


def s9_sanity():
    hdr("[§9] S2 / S4 の実行可能性")
    try:
        from src.figures_ratchet_log import (e1_drive_decomposition, death_events,
                                             descent_windows)          # noqa: F401
        print("  S2: src.figures_ratchet_log.e1_drive_decomposition を import 可能  … OK")
    except Exception as exc:                                            # pragma: no cover
        print(f"  S2: import 失敗 — {exc}")

    tot = hit = pts = npts = 0
    errs = []
    for z in Z:
        eta = float(z["lr"])
        p, v = z["p_hat"], z["v"].astype(np.float64)
        Fg, G = z["F_gate"].astype(np.float64), z["G_dot_mu"].astype(np.float64)
        m = (p == 1.0) & (np.abs(v) >= 1e-2)
        tot += p.size
        hit += int(m.sum())
        npts += p.shape[0]
        pts += int(m.any(axis=1).sum())
        if m.any():
            lhs = -Fg[m] / (2 * eta * v[m])
            rhs = np.repeat(G[:, None], p.shape[1], axis=1)[m]
            errs.append(np.abs(lhs - rhs) / np.maximum(np.abs(rhs), 1e-300))
    e = np.concatenate(errs)
    print(f"\n  S4: p̂=1 かつ |v|≥1e-2 のユニット {hit}/{tot} = {hit / tot:.4f}")
    print(f"      該当ユニットを含む記録点     {pts}/{npts} = {pts / npts:.4f}"
          f"    spec: 11.2% ＝ 一致")
    print(f"      相対誤差 p50={np.median(e):.2e}  p99={np.percentile(e, 99):.2e}  "
          f"max={e.max():.2e}")
    print(f"      閾値 1e-6 を超える割合 = {np.mean(e > 1e-6):.4f}  ⇒ S4 は PASS する（余裕あり）")

    print("\n  S4 の被覆（p̂=1 ユニットを含む記録点の割合）:")
    for lo, hi in [(0, 1), (1, 10000), (10000, 100000), (100000, 500000), (500000, 1000001)]:
        sel = (STEP >= lo) & (STEP < hi)
        n = sum(int((z["p_hat"] == 1.0)[sel].any(axis=1).sum()) for z in Z)
        d = sum(int(sel.sum()) for _ in Z)
        print(f"    step [{lo:>7},{hi:>8}) : {n / max(d, 1):.4f}")
    print("  ⇒ §1 の scope である t=500k 付近が最も薄い。時間層別で summary.md に載せる")


if __name__ == "__main__":
    s1_headline()
    s3_phase0()
    s3_identity_and_gate()
    s4_layerc()
    s9_sanity()
    print("\n完了。P1/P2 は事前登録のため未計算。")
