# 実系検証の再実行と実行版の区別

結論は[統合報告](interpretation.md)。登録した実系介入は2seed・task51–53で終了した。ここにあるコマンドは再実行方法であり、追加のseedや期間を調べた記録ではない。

## 保存版

- A–Eの学習実行版はcommit `008302c` の `real_interventions.py`。実行時SHAはseed別provenanceと一致する。
- 後で同ファイルの報告処理を訂正・追加した。凍結REFの保存データ比較先をEEQからVF1へ訂正したもので、学習済みstateや主対比は変えていない。
- Fのdriver実行版は `f11cf4d` の `real_direction_control.py`。その実行時にimportした `real_interventions.py` は報告処理訂正後のworking copyだった。正確な依存先SHAは `5cc409b730e5631fd7a72d7c5e23774ff3993a70601335c377de4b3b8ae2c250` で、最終保存版に含まれる。`f11cf4d` だけで依存先まで含む全実行版と主張しない。
- F後の変更は `--summarize` とread-only集計の追加。元のdriver版もGitに残る。
- float64精度対照と実状態の一歩予測は、それぞれのcode SHA・入力SHAをprovenanceへ記録した。学習をやり直さず確認する `validate.py` が、これらの対応を検算する。
- CSVは表示・Git管理のためCRLFをLFへ変換した。全フィールドの同一性、変換前後のSHAを `line_endings.json` に保存した。

## 必要な手元の元ファイル

元のMNISTファイルは `/home/issan/Projects/claude/proj_004_drift/data/mnist/train-images-idx3-ubyte.gz`、元のhelperは `/home/issan/Projects/obsidian-research-data/push_lift_ladder_1layer_0928/alpha_ladder_1layer.py`。両方のSHAは `real_s*_provenance.json` にある。保存値との再比較には同じ外部ディレクトリ内の `vfreeze` 記録を使う。外部の元ファイルは変更していない。

Python 3.12.3、torch 2.13.0+cu130、numpy 2.5.1、CPU単一thread。元実装のfloat32 `expm1(z)+1` の負側微分、denormal処理、乱数列を保持した。数学的な `exp(z)` へ差し替えていない。

## コマンド

リポジトリrootで、手元の仮想環境を使用する。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/logistic-v-height-mpl
PYTHON=/home/issan/Projects/claude/proj_004_drift/.venv/bin/python
```

保存成果物の代数・出所チェックと図の生成（新しい学習なし）：

```bash
"$PYTHON" analysis/logistic_v_height_0929/validate.py
"$PYTHON" analysis/logistic_v_height_0929/real_plot.py
```

学習の再実行は既存成果物を上書きするため、別のcheckoutで行う。A–Eの各seedはtask50までの再構成から始まり、task51–53へ進む。

```bash
"$PYTHON" analysis/logistic_v_height_0929/real_interventions.py --seed 0
"$PYTHON" analysis/logistic_v_height_0929/real_interventions.py --seed 1
"$PYTHON" analysis/logistic_v_height_0929/real_interventions.py --summarize
"$PYTHON" analysis/logistic_v_height_0929/real_state_predictions.py
"$PYTHON" analysis/logistic_v_height_0929/real_null_precision.py
"$PYTHON" analysis/logistic_v_height_0929/real_direction_control.py --seed 0
"$PYTHON" analysis/logistic_v_height_0929/real_direction_control.py --seed 1
"$PYTHON" analysis/logistic_v_height_0929/real_direction_control.py --summarize
```

task50状態・入力・task51–53ラベルとバッチ計画、各対照の全parameters/moments、Fの全step長さ監査をNPZへ保存した。生ログも保存した。Fの長さ監査は総量が大きいが、ゼロ例外と丸めを省略せず検算するため残している。

この実行環境では外部データ退避先へ書き込めないため、今回の新規生データもGitへ含めて保持した。初期のGitHub接続は名前解決に失敗したが、最終pushは成功し、実験commit `216e2f4` をremote mainへ反映できた。Git外の生データが残っていないこととremoteへの取り込みを確認したうえで、実験worktreeとbranchを削除する。
