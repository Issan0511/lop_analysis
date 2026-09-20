# erosion_race_0919

**登録した第1層の一時保護による救済は不支持。** Leak/Snakeの自然軌道は、GELUより大きな下向き移動を、さらに大きく増えた上向き移動で相殺していた。第1層を保護する介入は、第2層の著しい低応答化を伴う失敗経路へ変わった。

- [登録結果・全seedの対応差](summary.md) / [判定JSON](verdict.json)
- [層別の解釈とELUの中央値の限界（事後解析）](interpretation.md) / [事後集計JSON](interpretation.json)
- [登録図](figure.png) / [層別の補助図](figure_layers.png)
- [実装検査](checks.json) / [完走後の独立照合](qc.json)
- [生データの退避manifest](backup_manifest.json)

事前登録初版 `4a69757`、本走の実行commit `8856bab42b3aad4b012566624c3825cdfa0f172b`。
raw RL-CIFAR、9条件×5 seed、2課題×30000更新。初期重み・画像・ラベル・batch列はseedごとに全条件で同じ。

自然軌道の寄与帰属、全網の因果、介入後の事後解釈は区別する。2課題から50課題の結論は出さない。登録判定と予測の採点は事後解釈によって変更していない。

## 再集計

登録結果はcommitされたCSVのみから再集計できる。

```bash
python analysis/erosion_race_0919/report.py results/erosion_race_0919
```

unit配列・launcher記録を使う事後解析と独立照合は、退避先のraw result mirrorを指定する。

```bash
python analysis/erosion_race_0919/interpretation.py results/erosion_race_0919 --raw-root /home/issan/Projects/obsidian-research-data/erosion_race_0919/results
python analysis/erosion_race_0919/qc.py results/erosion_race_0919 --raw-root /home/issan/Projects/obsidian-research-data/erosion_race_0919/results
```

各走の `provenance.json` に入力・初期重み・ラベル/batch stream・source hashesを保存。`final.pt`、`unit_metrics.npz`、launcherログ、検査pilotはGit外の退避先にあり、bytesとSHA256をmanifestで検証した。
