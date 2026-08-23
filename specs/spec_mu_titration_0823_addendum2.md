# spec_mu_titration_0823 addendum 2: S3 を決定論的 support 再重み付け検査へ変更

proj_004 / 2026-08-23 / **初回本走の sanity 結果を見た後の完全事後追補 2**

親仕様: `specs/spec_mu_titration_0823.md`
第1追補: `specs/spec_mu_titration_0823_addendum.md`

## 1. 第1追補だけでは解けなかったこと

第1追補は、S3実装を初回走前のdocstringに書かれていた `frac(|z|>3)<=1%` へ一致させる、と固定した。その後、コードを変える前に退避済み初回走の**sanity metadataだけ**へ同じ規則を機械的に当てたところ、`center_alpha=0` は次だった。

```text
median |z|       = 0.6466
|z| > 3          = 6 / 469 = 1.279%
max |z|          = 3.1486
degenerate error = 0
```

したがって第1追補の1%規則でも、再走すればこのarmが確定的にFAILする。ここまでにも `p_hat`、`cos_u_mu`、theta、用量反応、strict_dead、lossの科学集計は見ていない。

1%を1.5%などへ結果適合的に緩めることは禁止する。問題は閾値の数値でなく、同じ固定eval batchを共有する相関したunit集合に、周辺binomial zのfamily集計閾値を置いた設計そのものにある。

## 2. S3の目的を決定論的に検査する

条件Aのeval batchは、固定された `N=2000` 本の5-bit自由patternである。全supportは32 patternで厳密列挙できる。各S3記録点で次を独立に計算する。

```text
g_p,i              = 32 support pattern p 上の unit i のgate（float64）
p_uniform,i        = (1/32) sum_p g_p,i
n_p                = 固定eval batch中のpattern pの実現個数
p_reweighted,i     = sum_p (n_p/N) g_p,i
p_empirical,i      = 固定eval 2000本を直接forwardしたgate率（float64）
```

必須S3は次の決定論的恒等式とshapeを検査する。

1. logger本体のexact `p_hat` と、S3内で独立に再列挙した `p_uniform` の最大絶対誤差 `<=1e-12`
2. 直接 `p_empirical` と、同じ固定evalの実現pattern頻度でsupport gateを再重み付けした `p_reweighted` の最大絶対誤差 `<=1e-12`
3. eval各行が32 supportのちょうど1つに対応し、`sum_p n_p=N=2000`
4. `p_uniform in {0,1}` の退化unitで直接経験率が厳密に同値

これは「経験率がbinomialゆらぎの何sigma以内か」という確率的主張を使わない。実際に観測したpattern頻度を条件付けして、exact-support列挙・centered前処理・直接eval forwardのshape/定義が同じかを全unitで厳密に照合する。probeが乱数を消費しないことは従来どおりS2で別に検査する。

## 3. z値の扱い

従来の `median|z|`、`max|z|`、`frac(|z|>3)`、独立二項を仮定したtail pは、過去との互換診断として `meta.json` に残す。ただし相関したfamilyに正当なPASS閾値を与えないため、いずれもS3のPASS/FAILには使わない。

第1追補 §3 の1%必須条件は、本追補で置き換える。第1追補 §1--2の「独立二項tailを必須化したのは誤り」という診断と、全arm再走・科学定義不変・provenanceの規律は維持する。

## 4. 再実行と不変部分

- 第1追補と本追補を別commitで先に固定してから、runner/loggerを変更する。
- configは両追補を順序つき `addenda` listで指し、各armは両path/SHA256を保存する。
- 解析は8arm間のaddenda list完全一致、現在の両ファイルSHA一致、sweep commitがclean analysis commitのGit祖先であることを必須検査する。
- 新しい単一commitのclean worktreeから8armすべてを再走する。
- S3以外の学習条件、alpha grid、保存統計、scope、theta、bootstrap、C0/C1/W1/C2/P1、総合判定は変更しない。
- 初回commit `39986e2` の退避データは正式結果へ混ぜない。
