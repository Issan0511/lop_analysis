# V11 図表統合・校正（0923）

実験は実行しない。入力は committed CSV/JSON と Vault の原稿、成果物は日本語の確認稿。

- `results/v11_completion_0923/V11_integrated_0923.md`: Vault 通し稿の統合時点のスナップショット。
- `output/pdf/V11_review_0923.pdf`: 本文・図表・付録・文献を含む 50 ページの確認稿。保留の【 】は維持。
- `results/v11_completion_0923/verification.json`: 数値・構造・出所の照合結果。独立査読ではない。
- `results/v11_completion_0923/render_checks.json`: 画像・数式・表・文献数のレンダー検査。

## 再生成

図 6 は次で生成する。Q1 の対応関係と Q4 の後期窓を原データと照合する。

```bash
python3 analysis/v11_figures_0921/fig5s_switch_push.py
```

PDF は Chrome と Node、KaTeX 0.16.22、markdown-it 14.1.0、Playwright を使う。依存先は引数で指定する。描画はローカルの画像・フォントだけを使用し、HTML に埋め込む。HTML は再生成可能な中間出力（退避先は backup_manifest.json）。

```bash
npm install --prefix /tmp/v11-render-deps katex@0.16.22 markdown-it@14.1.0
node analysis/v11_completion_0923/render.mjs /tmp/v11-render-deps/node_modules /path/to/node_modules
python3 analysis/v11_completion_0923/verify.py
pdftoppm -scale-to 1300 -png output/pdf/V11_review_0923.pdf /tmp/v11-page
```

`verify.py` は pandas、numpy、pypdf を使用する。本文の保留文の SHA-256 を記録し、数値は seed 内の課題窓から再計算する。Figure 6 と結果の対応、表 2 の比較名、表 3/4 の水準、G/H の commit と予測を点検する。全実験の再実行や統計的な独立再監査ではない。

## 編集上の訂正

表 2 D1 は C−ref、S2 は H−CH。図 4 の KKT1−調整 L2 Init は 12/20。図 5 は本文用 1 パネル、図 7 の ref は ELU。図 8 の Q は全画像×ユニットの対の割合。付録 F の共通診断帯と KKT1 の継ぎ目を区別した。方法節は dphi/autograd、seed 中央値/平均、符号検定/t 区間/順位区間を実験別に明記した。

5+1 の Claude の予測は元記録 3/10 を保存し、確率 p≥.50 の向きで統一した事後採点 5/10 を併記した。Swish の CNN は中断を明記し、未実行の比較の結果を作っていない。
