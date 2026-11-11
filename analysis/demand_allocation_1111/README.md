# demand_allocation_1111 の再現手順

問いは、活動への増加需要が逆伝播されるのに飽和が残ることと、読み出し縮小後に活動が増えることの両立。仕様は `specs/spec_demand_allocation_1111.md`、式は `results/demand_allocation_1111/theory.md`、数値の正本は同ディレクトリの `summary.md` と `verdict.csv`。

## 固定した実装と登録順

- 親: `14402c91ce78a58162a1e4ef8bd658199166d54c` の `stage1_ctrl_1111.py`。
- 仕様登録: `2acc729860c4ddfd9a87958fcbe39dad51f5ee06`。
- 本走コード・起動時版: `37c9ea82bb968bb0f09a892fa6451bc96e0bbe76`。
- 実行中の追加コミットは式・既存データ監査・集計・独立した代数検算。学習コードは変更していない。各jobの provenance に記録された git hash が異なる場合も、入力とソースの SHA256 を照合できる。
- CPU float32、各worker 1 thread、2 workers、seed 0,1,2。初期化と環境の乱数ストリームは旧走と同じ。基準軌道の全375課題末を旧保存状態とbit単位で照合する。

## 実行

唯一のcloneを使い、`CLAUDE.md` に従って別のworktreeで行う。Python環境は `/home/issan/Projects/claude/proj_004_drift/.venv/bin/python`。MNISTキャッシュは同cloneの `data/mnist`、旧状態は `/home/issan/Projects/obsidian-research-data/stage1_ctrl_1111/results/stage1_ctrl_1111/raw`。旧監査はさらに `push_lift_ladder_1layer_0928/vfreeze/runs_vf` の保存NPZを使う。両方とも実測入力であり、無ければ再現不能として止まる。

```bash
python src/demand_allocation_1111.py --smoke --out /tmp/demand_sanity
python analysis/demand_allocation_1111/verify_algebra.py /tmp/demand_algebra.json
python analysis/demand_allocation_1111/audit_existing.py --out results/demand_allocation_1111
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python src/demand_allocation_1111.py \
  --out results/demand_allocation_1111/raw --workers 2
python analysis/demand_allocation_1111/summarize.py \
  --raw results/demand_allocation_1111/raw --out results/demand_allocation_1111
```

`--out` のrawディレクトリは新規の場所を使う。既存jobに上書き・途中再開せず、欠損は欠損として扱う。集計は `complete.json` と15全cellを要求する。退避後の既存rawを再集計する場合は、`--raw /home/issan/Projects/obsidian-research-data/demand_allocation_1111/results/demand_allocation_1111/raw` を指定し、`--out` は新しいscratchディレクトリにする。

## 保存物と読み方

- `state.csv`: 課題・腕・stepごとのCE、精度、開口、幅、位置、読み出しノルム。3seed平均前の行を保存。
- `demand.csv`: 入力×unitの需要と実装微分。現在群と介入前に固定した群の両方。空の群はNA。
- `instant.csv`: 同じ状態からS/Q/scalar縮小、旧ラベル・新ラベル・一様ラベル期待を比較。新しい予測を用いて需要を再計算。
- `step.csv`: 次の予定ミニバッチによる実際の一歩。clone上のprobeなので本走に副作用なし。開始時のみfresh Adam/SGDも比較。
- `source.csv`: 実際のAdam分母を固定した条件付き会計。旧一次moment、現在の開口・境界帯・深部の勾配、丸め残差。source削除の因果効果とは異なる。
- `response.csv`: 需要を受けた群の実際の前活性・活動変化。需要と実変化が逆向きになりうる。
- `allocation.csv`: logit変化をbody/V/交差項/output biasに厳密分解。一歩の内積は局所の一次寄与で、長期性能の因果的な割合ではない。
- `noise.csv`: 固定した64ミニバッチでのV勾配の平均・分散・共分散。ELUの定数−1成分を分けるが、分散比だけでAdamの抑制を結論しない。
- rawの `.pt`: 介入前と腕末の全重み・Adam状態。`.npz`: unit単位の時間診断。`provenance.json`: 入力・依存ソースのhashと基準軌道照合。
- `existing_*.csv`: 既存走の事後再集計。新しい登録判定と混ぜない。
- `checks.json`、`sanity.json`、`algebra_checks.json`: 完全性・更新会計・有限差分等の検算。
- `backup_manifest.json`: git外ファイルの元パス、退避先、byte数、SHA256。rawとログはこの記録に従って復元できる。

実験単位はseedであり、120,000個の入力×unitや100unitを独立反復としてp値を作らない。プロットの帯は3seedの範囲で信頼区間ではない。今回のQは一度の介入。旧stage1のQallは毎切替の予防、旧vfreezeは課題151–200の長期窓であり、時間尺度を混同しない。
