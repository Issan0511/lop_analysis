"""Independent numerical/symbolic audit, report, and scientific figures."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nonsinking_ce_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def exact_identities():
    a, b, k, h, t, c, w, u, eta, q, s = sp.symbols("A B K H t C w u eta q s")
    r = (a + b) / (a - b)
    r1 = (a + b + t * (k + h)) / (a - b + t * (k - h))
    expected = 2 * t * (a * h - b * k) / ((a - b) * (a - b + t * (k - h)))
    checks = {
        "binary_ratio_finite_step": sp.simplify(r1 - r - expected) == 0,
        "binary_invariant": sp.expand((a + t*k)*h - (b + t*h)*k - (a*h - b*k)) == 0,
        "readout_separatrix": sp.expand((u + eta*s*q/2)**2 - (s + eta*k*u*q/2)**2/k
                                       - (1 - k*eta**2*q**2/4)*(u*u - s*s/k)) == 0,
    }
    r = (a + (c-1)*b) / (sp.sqrt(c-1)*(a-b))
    r1 = (a+t*k+(c-1)*(b+t*h)) / (sp.sqrt(c-1)*(a-b+t*(k-h)))
    expected = c*t*(a*h-b*k) / (sp.sqrt(c-1)*(a-b)*(a-b+t*(k-h)))
    checks["multiclass_ratio_finite_step"] = sp.simplify(r1-r-expected) == 0
    checks["simplex_invariant_value"] = (
        sp.Rational(31,20)*(-1) - sp.Rational(-9,20)*19 == 7)
    assert all(checks.values()), checks
    return checks


def calculate(data, mc):
    w, b, v, c = [data[k] for k in ("w", "b", "v", "c")]
    z = np.einsum("rnd,srhd->srnh", data["x"], w) + b[:, :, None, :]
    h = np.maximum(z, 0)
    if mc:
        f = np.einsum("srnh,srhc->srnc", h, v) + c[:, :, None, :]
        ymax = f.max(-1)
        logz = ymax + np.log(np.exp(f - ymax[..., None]).sum(-1))
        true = np.take_along_axis(f, np.broadcast_to(data["y"][None, ..., None], f.shape[:-1] + (1,)), -1)[..., 0]
        loss = (logz - true).mean(-1)
        pred = f.argmax(-1)
    else:
        f = (h * v[:, :, None, :]).sum(-1) + c[:, :, None]
        # Stable and independent of Torch's fused BCE implementation.
        loss = ((1-data["y"])[None] * np.logaddexp(0, f)
                + data["y"][None] * np.logaddexp(0, -f)).mean(-1)
        pred = (f >= 0)
    mu = z.mean(-2)
    sigma = np.sqrt(((z - mu[..., None, :])**2).mean(-2))
    return dict(loss=loss, mu=mu, sigma=sigma, ratio=mu/sigma,
                active=(z>0).mean(-2), accuracy=(pred==data["y"][None]).mean(-1))


def audit(rows, raw):
    errors = {}
    for name in dict.fromkeys(r["group"] for r in rows):
        with np.load(raw / (name + ".npz")) as f:
            data = dict(f)
        calc = calculate(data, name == "softmax10")
        err = {k: float(np.max(np.abs(v-data[k]))) for k, v in calc.items()}
        assert max(err.values()) < 1e-10, (name, err)
        group = [r for r in rows if r["group"] == name]
        assert len(group) == data["mu"].shape[1]
        for r in group:
            seed = int(r["seed"])
            assert int(r["steps"]) == int(data["step"][-1])
            for key in ("mu", "sigma", "ratio", "active"):
                for index, endpoint in ((0, "initial"), (-1, "final")):
                    assert np.allclose(calc[key][index, seed], json.loads(r[f"{key}_{endpoint}_units"]), atol=1e-12, rtol=0)
            for key in ("loss", "accuracy"):
                for index, endpoint in ((0, "initial"), (-1, "final")):
                    assert abs(calc[key][index, seed] - float(r[f"{key}_{endpoint}"])) < 1e-12
            assert float(r["gradient_max_error"]) < 1e-10
            assert float(r["recurrence_max_error"]) < 1e-10
        errors[name] = err
    required = ["primary", "initial_errors", "zero_cross_gram", "open_set", "within_class_spread",
                "width32", "minibatch_sgd", "adam", "softmax10"]
    counts = {}
    for name in required:
        group = [r for r in rows if r["group"] == name]
        assert len(group) == {"open_set": 32, "within_class_spread": 8, "width32": 8}.get(name, 1)
        for r in group:
            for key in ("mu_down_unit_steps", "ratio_down_unit_steps", "gate_changed_observations", "loss_increasing_steps"):
                assert int(r[key]) == 0, (name, r["seed"], key)
            assert float(r["loss_final"]) < float(r["loss_initial"]) / 100
            for key in ("w", "b", "v"):
                assert float(r[key + "_total_displacement"]) > 1e-4
        counts[name] = len(group)
    # Actual breaking controls, not simply states outside sufficient conditions.
    lookup = {r["group"]: r for r in rows}
    assert int(lookup["high_initial_ratio"]["mu_down_unit_steps"]) == 0
    assert int(lookup["high_initial_ratio"]["ratio_down_unit_steps"]) == 40000
    assert int(lookup["biased_geometry"]["mu_down_unit_steps"]) == 20000
    assert int(lookup["biased_geometry"]["ratio_down_unit_steps"]) == 20000
    assert float(lookup["initial_errors"]["accuracy_initial"]) == .5
    assert float(lookup["initial_errors"]["accuracy_final"]) == 1
    assert float(lookup["wrong_readout_small"]["accuracy_final"]) == 1
    assert abs(float(lookup["wrong_readout_large"]["loss_final"]) - np.log(2)) < 1e-12
    timings = {}
    for name, u in [("wrong_readout_small", -.5), ("wrong_readout_large", -.75)]:
        w, b = .75, -.25
        down, flip, close = [], None, None
        for step in range(1, 101):
            s = 2*w + b
            if s <= 0:
                continue
            q = np.exp(-np.logaddexp(0, u*s))
            t = .05*u*q/2
            w, b, u = w + 2*t, b + t, u + .05*s*q/2
            if t < -1e-12:
                down.append(step)
            if u >= 0 and flip is None:
                flip = step
            if 2*w+b <= 0 and close is None:
                close = step
        assert len(down)*2 == int(lookup[name]["mu_down_unit_steps"])
        assert down == list(range(1, len(down)+1))
        timings[name] = dict(down_steps=down, readout_changes_sign_at=flip, gate_closes_at=close)
    return dict(exact_identities=exact_identities(), independent_snapshot_errors=errors,
                independently_reconstructed_boundary_timings=timings,
                certified_numerical_groups=counts, all_passed=True,
                numerical_tolerance=1e-12, scope="Finite trajectories validate implementation; proof.md establishes permanence.")


def plot(raw, dest):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    colors = {"primary": "#2074ae", "softmax10": "#df8535"}
    for name, label, limit in [("primary", "Binary CE / 2 units", .25), ("softmax10", "Softmax CE / 10 units", 1/6)]:
        d = np.load(raw / (name + ".npz"))
        steps = d["step"]
        for a, key in [(ax[0,0], "loss"), (ax[0,1], "mu"), (ax[1,0], "ratio")]:
            values = d[key].reshape(len(steps), -1).mean(-1)
            a.plot(steps, values, label=label, color=colors[name], linewidth=2)
        ax[1,0].axhline(limit, color=colors[name], linestyle="--", linewidth=1, label=f"Proved limit: {limit:.3f}")
    ax[0,0].set(yscale="log", ylabel="Mean cross-entropy", title="Loss keeps decreasing")
    ax[0,1].set(ylabel="Mean preactivation", title="All units move upward")
    ax[1,0].set(ylabel="Mean / population standard deviation", title="Standardized position also rises")
    for name, label, color in [
        ("primary", "Correct readout", "#2074ae"),
        ("wrong_readout_small", "Wrong sign, magnitude 0.50", "#9371b4"),
        ("wrong_readout_large", "Wrong sign, magnitude 0.75", "#c54848")]:
        d = np.load(raw / (name + ".npz"))
        ax[1,1].plot(d["step"], d["ratio"][:,0,0], label=label, color=color, linewidth=2)
    ax[1,1].set(ylabel="Mean / population standard deviation", title="Reverse the readout: recover or shut down")
    for a in ax.flat:
        a.set_xscale("symlog", linthresh=10)
        a.set_xlim(0, 100000)
        a.set_xlabel("Optimizer steps")
        a.grid(alpha=.2)
        a.legend(fontsize=8)
    fig.suptitle("Persistent non-sinking under ordinary cross-entropy and backpropagation", fontsize=14)
    fig.savefig(dest / "counterexample.png", dpi=180)
    fig.savefig(dest / "counterexample.pdf")
    plt.close(fig)


def report(rows, dest, checks):
    lookup = {r["group"]: r for r in rows}
    p, m, e = [lookup[k] for k in ("primary", "softmax10", "initial_errors")]
    lines = [
        "# 通常の CE・逆伝播で沈まない反例：結果（2026-10-26）", "",
        "**達成したこと：通常の損失・全パラメータの逆伝播のまま、損失が 0 へ近づいても全 unit が沈まない反例を、二値分類と10クラス分類で構成した。**",
        "永久性は [proof.md](proof.md) の離散 GD の証明で保証し、下の有限走は PyTorch 実装と成立条件を検算する。",
        "初期状態を解析的に選んだ構成であり、ViT の典型性の証明ではない。入力・初期値を変えた開集合でも非沈降は保たれ、gradient flow では CE→0 まで示した。", "",
        "## 主反例", "",
        "| 走 | 更新数 | CE 初期 → 末尾 | μ 初期 → 末尾 | μ/σ 初期 → 末尾 | 下降した unit×step |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in (p, m):
        lines.append(f"| {r['group']} | {int(r['steps']):,} | {float(r['loss_initial']):.6g} → {float(r['loss_final']):.6g} | "
                     f"{float(r['mu_initial_mean']):.6g} → {float(r['mu_final_mean']):.6g} | "
                     f"{float(r['ratio_initial_mean']):.6g} → {float(r['ratio_final_mean']):.6g} | "
                     f"μ: {r['mu_down_unit_steps']}, r: {r['ratio_down_unit_steps']} |")
    lines += ["", "主反例は初期 accuracy=100%。ここでの学習は CE と margin の改善を意味する。",
              f"初期出力 bias を 2 にして誤分類を作った別の走では、accuracy は 50%→100%、CE は {float(e['loss_initial']):.6g}→{float(e['loss_final']):.6g}。"
              "その間も両 unit の μ と r に下降はない。全 W,b,v,c を学習し、初期の対称性で c の勾配が 0 の走でも c は固定していない。",
              "", "![通常学習の反例と読み出しの向きの対照](counterexample.png)", "",
              "## 一点の微調整ではない", "",
              "| 事前指定した比較 | 本数 | 各走の全 unit で μ と r が非減少 | gate の変化 | 最終 CE の範囲 |",
              "|---|---:|---:|---:|---:|"]
    labels = {"open_set": "入力・初期値を非対称に摂動", "within_class_spread": "各クラス32点のばらつき",
              "width32": "32 unit", "minibatch_sgd": "minibatch 1 の SGD", "adam": "標準 Adam"}
    for name, label in labels.items():
        group = [r for r in rows if r["group"] == name]
        good = sum(int(r["mu_down_unit_steps"]) == 0 and int(r["ratio_down_unit_steps"]) == 0 for r in group)
        vals = [float(r["loss_final"]) for r in group]
        lines.append(f"| {label} | {len(group)} | {good}/{len(group)} | "
                     f"{sum(int(r['gate_changed_observations']) for r in group)} | {min(vals):.4g}–{max(vals):.4g} |")
    lines += ["", "二点データでの非沈降は不等式が定める開集合の保証がある。多点の μ/σ と Adam の無限時間の性質は、ここでは有限走の確認に留める。",
              "32 unit の走では平均位置は末尾でも負の unit があるが、全 unit が初期位置から上がる。正側化と非沈降は異なる。", "",
              "## 成立条件を壊すとどうなるか", "",
              "| 比較（全て同じ通常学習） | 結果 |",
              "|---|---|",
              "| 読み出しの符号だけ反転、絶対値0.5 | 各 unit が最初の48更新で沈み、その後読み出しの向きが戻って上昇。accuracy 0%→100%。 |",
              "| 読み出しの符号を反転、絶対値0.75 | 各 unit が30更新で閉じて学習が止まる。最終 CE=log(2)、accuracy=50%。学習が進む反例には数えない。 |",
              "| 初期 b=1（μ/σが高い）、読み出しは正しい | μは1→約1.495へ上がる一方、rは2/3→約0.430へ全step下がる。σの成長による。 |",
              "| 入力の平均・幾何を変更 | 正しいクラスだけで開く unit でも、対象 unit の μ は0.45→約−0.718、rは約0.818→−0.151へ全step下がる。 |",
              "| 反対クラス間の拡張入力内積を正にする | gate が開閉して r の単調性は失われたが、μ は下がらなかった。十分条件を外すだけで沈降が必ず起きるわけではない。 |", "",
              "上の48/30更新は登録 CSV の下降 unit×step 数を2 unitで割った値。独立なスカラー更新式でも最初の48/30更新であること、読み出しの反転・gate 閉鎖の時刻を再構成して照合した（checks.json）。", "",
              "## 何が絞れたか", "",
              "反例の核は、**活動する入力で読み出しが正しいクラスを支え、その対応が通常の更新によって補強されること**。"
              "その場合、CE の正味の信号は活動を上げる。下向きの自己成分の存在だけでは合計の沈降を説明できない。",
              "この構成の十分条件は、入力の配置が gate の分離を維持し、平均へ運ぶ更新の向きが正で、幅の増加が r を逆転させないこと。",
              "クラス純度だけでは十分でないことも、入力の幾何を変えた対照で確認した。",
              "「常に沈む」を主張するなら、この領域を除外する条件が要る。"
              "「平均的・典型的に沈みやすい」を主張するなら、沈まない領域も含めて出現頻度と学習での到達性を説明する必要がある。"
              "候補は誤差で重み付けした活動の競合、入力幾何、幅の成長であり、どれが通常の学習状態で優勢になるかは未証明。"
              "本走を ViT の相2の原因同定や T5 の検証として運ばない。", "",
              "## 検証と再現", "",
              f"- 全59走の各stepをオンライン集計し、保存状態を独立な NumPy 実装で再計算。検査結果: {checks['all_passed']}。",
              f"- 通常 CE 全パラメータ勾配と autograd の最大誤差: {max(float(r['gradient_max_error']) for r in rows):.3g}。",
              f"- 主例・10クラスの厳密な再帰との最大誤差: {max(float(r['recurrence_max_error']) for r in rows):.3g}。",
              f"- 開集合の不変量 I の最大誤差: {max(float(r['invariant_max_error']) for r in rows):.3g}。",
              "- 有限差分の r の式、I の不変性、読み出し反転の境界、多クラスの式を SymPy で恒等式として検算。",
              "- 数値の単調性閾値は −1e−12。永続性の根拠には、この許容差や有限走を使っていない。",
              "- 数値の正本: verdict.csv。環境・全初期値: provenance.json。検算: checks.json。証明: proof.md。",
              "- 事前登録: specs/spec_nonsinking_ce_1026.md（d930de0a）、10クラス追補（6a4dbf56）。訓練ソース: 5a5baa91。",
              "", "再現（repository root）:", "",
              "    python analysis/nonsinking_ce_1026/run.py",
              "    python analysis/nonsinking_ce_1026/analyze.py",
              "", "依存: PyTorch、NumPy、SymPy、Matplotlib。GPU不要。生データの退避後は analyze.py に --raw で archive の raw フォルダを指定する。", ""]
    (dest / "summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("raw/nonsinking_ce_1026"))
    parser.add_argument("--results", type=Path, default=Path("results/nonsinking_ce_1026"))
    args = parser.parse_args()
    rows = list(csv.DictReader((args.results / "verdict.csv").open()))
    assert len(rows) == 59
    checks = audit(rows, args.raw)
    (args.results / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    plot(args.raw, args.results)
    report(rows, args.results, checks)
    print(json.dumps(dict(all_passed=checks["all_passed"], rows=len(rows),
                          exact_identities=checks["exact_identities"],
                          snapshot_max_error=max(max(e.values()) for e in checks["independent_snapshot_errors"].values()))))


if __name__ == "__main__":
    main()
