#!/usr/bin/env python3
"""Rebuild the descriptive CNN response report from saved outputs; no torch/forward.

Run from the repository root:
  .venv/bin/python analysis/cnn_response_center_1021/result_report.py
Python+NumPy only. All inference is recomputed from stored per-batch values.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ARMS=("LR_self","LR_swap","LRc_self","LRc_swap")
SEEDS=list(range(10,20))
UPDATES=(0,100,750,3000,15000,30000)
METRICS=("first3000","all","last3000")
LABELS={"first3000":"更新 1–3000","all":"全 30000 更新","last3000":"最後 3000 更新"}


def read_csv(path):
    with path.open() as f:return list(csv.DictReader(f))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False)


def sign_test(d):
    pos=sum(bool(x>0) for x in d);neg=sum(bool(x<0) for x in d);n=pos+neg
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(pos,neg)+1))/2**n) if n else 1.
    return {"n_positive":pos,"n_negative":neg,"n_tie":len(d)-n,"p_two_sided":p}


def paired(a,b,boots):
    d=np.asarray(a)-np.asarray(b)
    return {"mean_difference":float(d.mean()),"ci95_seed_bootstrap":np.percentile(d[boots].mean(1),[2.5,97.5]).tolist(),
            "seed_differences":dict(zip(map(str,SEEDS),map(float,d))),**sign_test(d)}


def table(headers,rows):
    esc=lambda x:str(x).replace("|", "\\|")
    return "\n".join(["| "+" | ".join(map(esc,headers))+" |","|"+"|".join(["---"]*len(headers))+"|",
                      *["| "+" | ".join(map(esc,r))+" |" for r in rows]])


def run(results,stem):
    source_files=[ROOT/"specs/spec_cnn_response_center_1021.md",results/"checks/checks.json",results/"verdict.json"]
    verdict=json.loads((results/"verdict.json").read_text());checks=json.loads((results/"checks/checks.json").read_text())
    summaries={};diags={};provs={};means={};curves={};unit_agreement={};failures={}
    seeds=np.array(SEEDS);boots=np.random.default_rng(1021).integers(0,10,size=(20000,10))
    for arm in ARMS:
        p=results/arm
        source_files += [p/name for name in ("summary.csv","diagnostics.csv","unit_diagnostics.csv","provenance.json","done.json","anchor.json","batch_curves.npz","epochs.csv")]
        ss=read_csv(p/"summary.csv");assert [int(r["seed"]) for r in ss]==SEEDS
        summaries[arm]=[{k:(int(v) if k=="seed" else v=="True" if k=="failed" else v if k=="arm" else float(v)) for k,v in r.items()} for r in ss]
        ds=read_csv(p/"diagnostics.csv")
        assert len(ds)==60 and {(int(r["seed"]),int(r["update"])) for r in ds}=={(s,t) for s in SEEDS for t in UPDATES}
        diags[arm]=[{k:(int(v) if k in ("seed","update") else v=="True" if k=="failed" else v if k=="arm" else float(v)) for k,v in r.items()} for r in ds]
        prov=json.loads((p/"provenance.json").read_text());done=json.loads((p/"done.json").read_text());anchor=json.loads((p/"anchor.json").read_text())
        assert prov["identity"]==done["identity"] and done["summary_sha256"]==digest(p/"summary.csv") and anchor["pass"]
        assert prov["identity"]["steps"]==30000 and prov["identity"]["data"]["seeds"]==SEEDS
        provs[arm]=prov
        c=np.load(p/"batch_curves.npz")
        assert np.array_equal(c["seeds"],seeds) and c["accuracy"].shape==(10,30000) and c["ce"].shape==(10,30000)
        curves[arm]={"accuracy":c["accuracy"],"ce":c["ce"]}
        for metric,sl in (("first3000",slice(0,3000)),("all",slice(None)),("last3000",slice(-3000,None))):
            for field,col in (("accuracy",metric),("ce","ce_"+metric)):
                expected=curves[arm][field][:,sl].mean(1,dtype=np.float64)
                got=np.array([r[col] for r in summaries[arm]])
                assert np.array_equal(expected,got),(arm,col,float(np.max(np.abs(expected-got))))
        ep=read_csv(p/"epochs.csv")
        assert len(ep)==4000 and {(int(r["seed"]),int(r["epoch"])) for r in ep}=={(s,e) for s in SEEDS for e in range(1,401)}
        failures[arm]={"summary_failed":sum(r["failed"] for r in summaries[arm]),
                       "diagnostic_failed":sum(r["failed"] for r in diags[arm]),
                       "epoch_failed":sum(r["failed"]=="True" for r in ep),
                       "nonfinite_saved_batch_values":int((~np.isfinite(c["accuracy"])).sum()+(~np.isfinite(c["ce"])).sum())}
        # Verify channel aggregation against the saved per-channel records.
        units=read_csv(p/"unit_diagnostics.csv");maxerr=0.
        for d in diags[arm]:
            rows=[r for r in units if r["site"]=="c1_effective" and int(r["seed"])==d["seed"] and int(r["update"])==d["update"]]
            assert len(rows)==16
            for key in ("mean_d","pool_mean_d","var_d","pool_var_d","p","pool_p"):
                maxerr=max(maxerr,abs(np.mean([float(r[key]) for r in rows])-d[f"c1_effective_{key}_unit_mean"]))
        assert maxerr<1e-12
        unit_agreement[arm]=float(maxerr)
        means[arm]={k:float(np.mean([r[k] for r in summaries[arm]])) for k in (*METRICS,*("ce_"+m for m in METRICS))}
        means[arm].update(final_memo_acc=float(np.mean([r["memo_acc"] for r in diags[arm] if r["update"]==30000])),
                          final_memo_ce=float(np.mean([r["memo_ce"] for r in diags[arm] if r["update"]==30000])))
    # Exact configuration equality across all arms except their arm label.
    bases={a:{k:v for k,v in provs[a]["identity"].items() if k!="arm"} for a in ARMS}
    assert all(v==bases[ARMS[0]] for v in bases.values())
    contract=checks["preflight_contract"]["value"]
    assert contract["identity"]==bases[ARMS[0]] and checks["all_pass"]
    githeads={provs[a]["git_hash"] for a in ARMS};assert len(githeads)==1
    run_commit=next(iter(githeads));assert run_commit==contract["implementation_commit"]
    # Compare recorded implementation digests to immutable Git objects, not the
    # currently edited statistics serializer in the working tree.
    code_verified={}
    for rel,expected in bases[ARMS[0]]["code_sha256"].items():
        content=subprocess.run(["git","-C",str(ROOT),"show",run_commit+":"+rel],capture_output=True,check=True).stdout
        got=hashlib.sha256(content).hexdigest();assert got==expected,rel
        code_verified[rel]=got
    pairs={}
    for metric in METRICS:
        pairs[metric]={name:paired([r[metric] for r in summaries[a]],[r[metric] for r in summaries[b]],boots)
                       for name,a,b in (("rescue","LR_swap","LR_self"),("reverse","LRc_self","LRc_swap"))}
    ordered=sorted(pairs["first3000"],key=lambda k:pairs["first3000"][k]["p_two_sided"]);prev=0.
    for i,k in enumerate(ordered):
        q=pairs["first3000"][k];q["p_holm"]=min(1.,max(prev,(2-i)*q["p_two_sided"]));prev=q["p_holm"]
        saved=verdict["tests"][k]
        for key in ("mean_difference","p_two_sided","p_holm","n_positive","n_negative","n_tie"):
            assert abs(q[key]-saved[key])<1e-14,(k,key)
        assert np.allclose(q["ci95_seed_bootstrap"],saved["ci95_seed_bootstrap"],rtol=0,atol=1e-14)
    response_keys=("c1_effective_pool_mean_d_unit_mean","c1_effective_pool_var_d_unit_mean","c1_effective_pool_p_unit_mean",
                   "c1_effective_mean_d_unit_mean","c1_effective_var_d_unit_mean","c1_effective_p_unit_mean",
                   "c1_m_unit_median","c1_sd_unit_median","c1_r_unit_median",
                   "c1_effective_m_unit_median","c1_effective_sd_unit_median","c1_effective_r_unit_median",
                   "c1_mean_d_unit_mean","c1_pool_mean_d_unit_mean","c1_effective_rank","memo_acc","memo_ce")
    dynamics={a:{str(t):{k:float(np.mean([r[k] for r in diags[a] if r["update"]==t])) for k in response_keys} for t in UPDATES} for a in ARMS}
    rounding={}
    for a in ("LR","LRc"):
        item=checks["natural_self_short_window_"+a]
        rounding[a]={**item,"online_acc_mean_difference_self_minus_natural":float(np.mean(item["self_online_acc_by_seed"])-np.mean(item["natural_online_acc_by_seed"])),
                     "online_ce_mean_difference_self_minus_natural":float(np.mean(item["self_online_ce_by_seed"])-np.mean(item["natural_online_ce_by_seed"]))}
    audit={"all_arm_identities_equal_except_arm":True,"preflight_identity_equal":True,"main_run_commit":run_commit,
           "registered_spec_commit":contract["registered_spec_commit"],"recorded_code_hashes_verified_against_run_commit":code_verified,
           "configuration_sha256":hashlib.sha256(canonical(bases[ARMS[0]]).encode()).hexdigest(),
           "summary_matches_saved_batch_curves_exactly":True,"done_summary_hashes_match":True,
           "unit_mean_aggregation_max_abs_error":unit_agreement,"failure_counts":failures,
           "all_failures_zero":all(all(v==0 for v in f.values()) for f in failures.values()),
           "runtime":bases[ARMS[0]]["numerical_context"],
           "all_initial_anchors_exact":all(checks["anchor_"+a]["pass"] for a in ARMS)}
    assert audit["all_failures_zero"]
    report={"experiment":"cnn_response_center_1021","primary":verdict["primary"],"audit":audit,
            "primary_inference":pairs["first3000"],"secondary_descriptive_pairs":{k:v for k,v in pairs.items() if k!="first3000"},
            "arm_means":means,"per_seed_summary":summaries,"c1_dynamics":dynamics,
            "diagnostic_aggregation":"Response = per-channel observation mean, then mean over 16 channels, then mean over 10 seeds; variances = within-channel variance averaged over channels/seeds. m/s/r columns = channel medians per seed, then seed mean.",
            "natural_self_rounding_preflight":rounding,
            "reference_gap":verdict["self_control_gap_secondary_reference"],
            "report_script_sha256":digest(Path(__file__)),
            "input_sha256":{str(p.relative_to(ROOT)):digest(p) for p in source_files},
            "no_new_forward_or_training":True,
            "post_run_serializer_change":"Root converted NumPy booleans to native bool in sign_test counts after terminal JSON printing failed. Stored primary verdict/statistics were independently reproduced here from saved batch curves; training code is pinned to the run commit above."}
    stem.parent.mkdir(parents=True,exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    lines=["# CNN 中心化が保った c1 応答場の交換 — 登録走の結果", "",
           "**主判定は `ONE_DIRECTION`。非中心化 LR へ中心化 LRc の場を移すと初期適応が改善し、逆移植による悪化は今回の主判定では確認できなかった。**", "",
           "RL-CIFAR、LeakyReLU(0.1)、課題30末の保存状態、seed10–19。課題31を全4腕で30,000更新した。4腕×10seedの全結果を含み、除外・非有限損失は0件。中心化あり/なしの受け手の重み・入力処理・Adam状態は維持し、c1の応答場だけを自己場対移植場で比較した。", "",
           "![初期対応差と750更新binの平均学習曲線。最後3000更新の悪化も拡大表示する。](result_figure.png)", "",
           "図の [PDF](result_figure.pdf)／[再生成スクリプト](result_figure.py)。点はseed、誤差棒はseed bootstrap CI、曲線は750更新ごとに区切った10seed平均。末尾の比較は副の記述統計。", "",
           "## 主判定：最初3,000更新のonline accuracy", "",
           table(["比較","対照 → 移植の精度 (%)","判定用差 (point)","seed bootstrap 95% CI (point)","予測方向のseed","Holm補正p"],[
               ["LRへの救済",f"{100*means['LR_self']['first3000']:.4f} → {100*means['LR_swap']['first3000']:.4f}",f"+{100*pairs['first3000']['rescue']['mean_difference']:.4f}","["+", ".join(f"{100*x:.4f}" for x in pairs['first3000']['rescue']['ci95_seed_bootstrap'])+"]","10/10",f"{pairs['first3000']['rescue']['p_holm']:.8f}"],
               ["LRcへの逆移植",f"{100*means['LRc_self']['first3000']:.4f} → {100*means['LRc_swap']['first3000']:.4f}",f"悪化 +{100*pairs['first3000']['reverse']['mean_difference']:.4f}","["+", ".join(f"{100*x:.4f}" for x in pairs['first3000']['reverse']['ci95_seed_bootstrap'])+"]","7/10",f"{pairs['first3000']['reverse']['p_holm']:.5f}"]]),"",
           "検定の差は救済=LR_swap−LR_self、逆移植=LRc_self−LRc_swap。両側exact符号検定2本をHolm補正。CIはseed bootstrap percentile 20,000回、乱数seed1021。逆移植のCIは『悪化量』の符号で示した。多数派符号と平均差の方向は両比較で整合する。", "",
           "主指標の全seed（精度は%、差はpoint）：", "",
           table(["seed","LR_self","LR_swap","救済差","LRc_self","LRc_swap","逆移植の悪化量"],[
               [s,*[f"{100*summaries[a][i]['first3000']:.4f}" for a in ('LR_self','LR_swap')],f"{100*pairs['first3000']['rescue']['seed_differences'][str(s)]:+.4f}",*[f"{100*summaries[a][i]['first3000']:.4f}" for a in ('LRc_self','LRc_swap')],f"{100*pairs['first3000']['reverse']['seed_differences'][str(s)]:+.4f}"] for i,s in enumerate(SEEDS)]),"",
           "## 副指標：改善の時間範囲", "",
           table(["腕","最初3000 (%)","全30000 (%)","最後3000 (%)","最終memo (%)","最終memo CE"],[
               [a,*[f"{100*means[a][k]:.4f}" for k in METRICS],f"{100*means[a]['final_memo_acc']:.4f}",f"{means[a]['final_memo_ce']:.6g}"] for a in ARMS]),"",
           "**LRへの移植の改善は初期適応と全更新平均に見られるが、最後3,000更新では自己場を下回った。** この後半の悪化を省いて『全期間で救済した』とは書かない。固定集合の最終memoは最後の一点、onlineは更新前のbatch予測の窓平均であり、両者は一致を要しない。副指標で主判定を置き換えない。", ""]
    for metric in ("all","last3000"):
        lines += [f"### {LABELS[metric]}の全seed", "",
                  table(["seed",*ARMS,"救済差 (pt)","逆移植悪化量 (pt)"],[
                      [s,*[f"{100*summaries[a][i][metric]:.4f}" for a in ARMS],f"{100*pairs[metric]['rescue']['seed_differences'][str(s)]:+.4f}",f"{100*pairs[metric]['reverse']['seed_differences'][str(s)]:+.4f}"] for i,s in enumerate(SEEDS)]),"",
                  "対応平均差（記述的）："+"、".join(f"{name} {100*pairs[metric][key]['mean_difference']:+.4f} point [95% CI {100*pairs[metric][key]['ci95_seed_bootstrap'][0]:+.4f}, {100*pairs[metric][key]['ci95_seed_bootstrap'][1]:+.4f}]" for key,name in (("rescue","救済"),("reverse","逆移植の悪化量")))+"。追加の確認的検定として扱わない。",""]
    lines += ["平均online CE：", "",table(["腕","最初3000","全30000","最後3000"],[[a,*[f"{means[a]['ce_'+k]:.6g}" for k in METRICS]] for a in ARMS]),"",
              "## c1で実際に渡した応答", "",
              "次の平均微分は、各チャネルの画像×位置での平均→16チャネル平均→10seed平均。微分分散は**チャネル内分散**をチャネル・seedで平均したもの。全ユニットを混ぜた総分散、勾配ノルム、学習速度そのものではない。Pool列は、その腕の実際の活動が選んだMaxPool winner上を集約した。開始時winnerは受け手自然網と完全一致している。", "",
              table(["開始時の腕","全位置 平均φ′","受け手winner上 平均φ′","winner上 φ′分散","winner上 p(z_eff>0)"],[
                  [a,*[f"{dynamics[a]['0'][k]:.6f}" for k in ('c1_effective_mean_d_unit_mean','c1_effective_pool_mean_d_unit_mean','c1_effective_pool_var_d_unit_mean','c1_effective_pool_p_unit_mean')]] for a in ARMS]),"",
              "LRへの移植はwinner上の平均微分を0.190540→0.300939へ上げ、LRcへの逆移植は0.385770→0.172172へ下げた。したがって逆移植の非有意を『応答を下げられなかった』とは読めない。ただし平均・分散・配置が一緒に変わり、平均微分だけの因果効果は分離していない。gain対照は走らせていない。", "",
              "全位置の供与者場は交換できても、receiver winnerが違うため、winner上の値は供与者自身のwinner上の値と同一にはならない。例えばLR_swapの開始値0.300939はLRc_selfの0.385770とは異なる。", "",
              "受け手winner上の実効平均微分の推移：", "",
              table(["更新",*ARMS],[[t,*[f"{dynamics[a][str(t)]['c1_effective_pool_mean_d_unit_mean']:.6f}" for a in ARMS]] for t in UPDATES]),"",
              "両方向の応答差は3,000更新以後も残る。後半のLR移植の成績悪化や、LRc逆移植の主指標が非有意だったことを、平均応答の値だけでは説明し切れない。", "",
              "## raw前活性と人工場の座標を分ける", "",
              "rawは学習中の受け手c1が作るz。effectiveは交換場の `z_D0+(z_Rθ−z_R0)` であり、受け手重みをその位置へ直接移した量ではない。m/s/rの表はチャネル中央値をseedで平均した値（rは各チャネルm/sの中央値）。", "",
              table(["更新","腕","raw m","raw s","raw r","effective m","effective s","effective r"],[
                  [t,a,*[f"{dynamics[a][str(t)][k]:.5f}" for k in ('c1_m_unit_median','c1_sd_unit_median','c1_r_unit_median','c1_effective_m_unit_median','c1_effective_sd_unit_median','c1_effective_r_unit_median')]] for t in (0,30000) for a in ARMS]),"",
              "開始時には各受け手のraw前活性・活動・pool winner・全後段特徴・logits・CEが対照と移植で完全一致した。活動のeffective rankも受け手内で一致（LR系5.86233、LRc系12.95001）。以後のrank差は介入後の観察であり、この走からrankを原因として同定しない。LeakyReLUは負側でも微分0.1を持つので、負側割合を厳密なゼロ勾配死と呼ばない。", "",
              "## 自然網と自己場の丸め差", "",
              "自己場は自然forwardに置き換えず、移植と同じanchor演算を通した。開始時の全特徴と、最初の1更新後のP/m/vは両受け手で完全一致。2更新目以後には差が生じた。次は同じ課題31・同じ最初75更新を使ったpreflightの記録であり、主走の腕を追加した結果ではない。", "",
              table(["受け手","2更新後 max|ΔP|","75更新後 max|ΔP|","75更新online差 self−natural (pt)","75更新CE差 self−natural"],[
                  [a,f"{rounding[a]['rows'][1]['P_max_abs']:.8g}",f"{rounding[a]['rows'][-1]['P_max_abs']:.8g}",f"{100*rounding[a]['online_acc_mean_difference_self_minus_natural']:+.6f}",f"{rounding[a]['online_ce_mean_difference_self_minus_natural']:+.6g}"] for a in ('LR','LRc')]),"",
              "このため主比較は同じ計算経路の自己場対移植場とした。今回の30,000更新の自己場軌跡を、自然網の30,000更新軌跡とbit同一だとは主張しない。保存された課題30の自然網評価との差は記録されたCSV精度程度で、最大絶対差は5×10⁻¹⁰未満だった。", "",
              "## 読める結論と残った接続", "",
              "このCNNでは、中心化で保たれたc1応答場を、開始時の特徴・予測を保って非中心化受け手へ渡すと、課題31の初期適応を10/10seedで改善した。長期の中心化履歴と、c1応答場に対する一方向の因果介入を結べた。", "",
              "逆移植の悪化は主指標で未確認であり、双方向因果介入が閉じたとは言わない。自己場対照群差は最初3,000更新で17.458125pointあるが、この差で救済量を割って『媒介率』とはしない。中心化は入力・重み・下流特徴・最適化状態にも履歴を残す。c1の全Jacobian交換、中心化効果の完全媒介、実ラベルへの一般化は未同定。", "",
              "同indexチャネルは初期化を共有するが、課題30で同じ意味の特徴である保証はない。非有意を『c1が原因でない』、後のrank差を『rankが原因』とは読まない。", "",
              "## 再現・検査記録", "",
              f"登録spec commit: `{audit['registered_spec_commit']}`。本走実装 commit: `{run_commit}`。4腕のidentityはarm名以外完全一致し、preflightのidentityとも一致。各依存コードhashは本走commitのGit内容と照合済み。", "",
              "RTX5060Ti、torch2.13.0+cu130、CUDA13.0、cuDNN92000。CUDA matmul TF32=false、cuDNN TF32=true、matmul precision=highest、deterministic=true。同順序seed10–19の10-run bundle、batch16、lr0.001、30,000更新。", "",
              "保存batch曲線から全seedの精度・CE要約と主検定を再計算して一致。全400epoch×10seed×4腕の記録、doneのsummary hash、unit別記録からの応答集約も照合した。除外0、失敗0、非有限値0。graph/eagerと状態復元・反復の検査も合格。", "",
              "本走後、集計の端末JSON表示がNumPyの型で失敗したため、親が符号カウントをnative boolへ変更した。学習更新や統計定義は変わらず、本報告は保存batch値から主判定を独立再計算して一致を確認している。", "",
              "再生成：`python analysis/cnn_response_center_1021/result_report.py`。新たなforward・学習・GPU計算を行わず、保存CSV/NPZ/JSONのみを読む。入力hash、全seed要約、副の対応差、全時点のc1集約、丸め差の全記録は [result_report.json](result_report.json)。登録specは [spec](../../specs/spec_cnn_response_center_1021.md)、元の主判定は [verdict](../../results/cnn_response_center_1021/verdict.json)。", ""]
    stem.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps({"primary":verdict["primary"],"all_failures_zero":audit["all_failures_zero"],"identities_match":True,
                      "report_md":str(stem.with_suffix('.md')),"report_json":str(stem.with_suffix('.json'))},ensure_ascii=False,indent=2))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results",type=Path,default=ROOT/"results/cnn_response_center_1021")
    ap.add_argument("--out-stem",type=Path,default=Path(__file__).with_name("result_report"))
    args=ap.parse_args();run(args.results.resolve(),args.out_stem.resolve())
