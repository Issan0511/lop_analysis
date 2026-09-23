# V11 監査訂正の反映（0923）

Issa「訂正反映して」に基づく編集。実験は実行しない。
正本は Vault `可塑性喪失/論文作成/V11通し稿_0922.md`。

- `results/v11_corrections_0923/V11_corrected_0923.md`: 訂正後の正本のスナップショット。
- `corrections.json`: 79編集の訂正前・訂正後、元の行番号、27監査項目への対応。
- `reflection.md`: Vaultの反映ノートの写し。各項目の根拠リンク付き。
- `verification.json`: 数値、成立条件、登録予測、構造、キャプションの照合。
- `output/pdf/V11_review_0923.pdf`: 訂正後の50ページPDF。旧統合稿と同じ出力パス。

T3の条件付き導出は維持。B2/B8は条件を戻した保留文案で、採用は未確定。
決定9の【 】は原文を保持。ViTは主要証拠外・出所確認待ちとした。
構成ボードの元原稿・カードID・保存形式は変更しない。

## 再生成

```bash
python3 analysis/v11_corrections_0923/apply_corrections.py
python3 analysis/v11_corrections_0923/sync_captions.py
python3 analysis/v11_figures_0921/fig02_doors.py
python3 analysis/v11_figures_0921/fig08_response.py
python3 analysis/v11_figures_0921/figA_mnist_chain.py
python3 analysis/v11_figures_0921/figB_5p1_budget.py
python3 analysis/v11_figures_0921/figC_initgeom_rest.py
python3 analysis/v11_figures_0921/figJ_layer_chimera.py
npm install --prefix /tmp/v11-corrections-render-deps katex@0.16.22 markdown-it@14.1.0
node analysis/v11_completion_0923/render.mjs /tmp/v11-corrections-render-deps/node_modules /path/to/playwright/node_modules --corrected
python3 analysis/v11_corrections_0923/verify.py
pdftoppm -scale-to 1000 -png output/pdf/V11_review_0923.pdf /tmp/v11-corrected-page
```

Pythonはpandas/numpy/matplotlib/pypdfを使用。描画はローカルのChromeとフォントを使う。
`--corrected`なしの旧レンダーは履歴の統合稿を生成するため、現在のPDFの再生成には使わない。
全ページを画像化して一覧確認し、訂正図と代表ページを拡大点検した。
中間HTMLと確認画像は`backup_manifest.json`の退避先に保存する。

Vaultの変更はObsidian MCPでハッシュ照合して書き込む。上のスクリプトはVaultへ直接書かない。
