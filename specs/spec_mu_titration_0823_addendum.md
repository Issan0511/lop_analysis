# spec_mu_titration_0823 addendum: S3 判定実装の訂正

proj_004 / 2026-08-23 / **初回本走の sanity 結果を見た後の完全事後追補**

親仕様: `specs/spec_mu_titration_0823.md`

## 1. 起きたこと

事前登録 commit `39986e271cc55963c4f9e41558c6fbbb023f70de` から8 armを4本ずつ並列実行した。第1群4 armは必須 sanity を通過した。第2群では `center_alpha=3e-5` が S3 のみ FAIL し、runner は設計どおり `failed_sanity`、非0終了にした。並列中だった `center_alpha=.01` はこの時点で停止した。

この判断までに見たのは `meta.json` の S1--S7 診断だけであり、`p_hat`、`cos_u_mu`、`theta`、用量反応、strict_dead、loss の科学集計は見ていない。

失敗 arm の S3 診断は次のとおりだった。

```text
median |z|       = 0.6466
|z| > 3          = 19 / 2468 = 0.770%
max |z|          = 3.1486
degenerate error = 0
binomial-tail p  = 0.00007
```

S2、S4、S5、S7は同 armでも PASSし、S5の壁恒等式の生不一致は0だった。

## 2. 原因

`src/ratchet_log.py::check_s3` の事前実装には、docstring と実コードの不一致があった。

- docstring が固定していた判定: `median |z| <= 1.0`、`frac(|z|>3) <= 1%`、退化ユニット厳密一致
- 実コードが使った判定: `median |z| <= 1.0`、`Binomial(n, P(|Z|>3))` の上側確率 `>= .001`、退化ユニット厳密一致

後者は全 unit・3時点の exceedance を独立 Bernoulli とみなす。しかし S3 は全unit・全時点で同じ固定 eval batchを使う。異なるunitのゲート集合は同じ32パターン上で重なり、時点をまたいで同じゲート集合も現れるため、exceedance は強く依存する。実装中の `s3_note` 自体もこの非独立性を記録していた。したがって独立二項の個数検定を必須 PASS 条件にしたのが実装バグであり、docstring に固定済みだった1% family criterion と食い違っていた。

失敗 arm は `frac(|z|>3)=0.770%`、`median|z|=0.6466`、退化誤差0なので、事前に文章化されていた criterion では PASS する。

## 3. 訂正

S3 の必須条件を、初回走前の docstring に書かれていた次の形へ一致させる。

```text
median |z| <= 1.0
frac(|z| > 3) <= 0.01
max error at p_exact in {0,1} == 0
```

独立二項を仮定した上側確率は診断値として残すが、PASS/FAILには使わない。`max|z|` も診断値のまま残す。S3 の目的は exact-support probe と固定N=2000経験率の取り違え・shapeずれ・退化点不一致を検出することであり、相関したunit群に対する多重検定を新しい科学アウトカムとして導入することではない。

## 4. 再実行規律

- 初回 commit `39986e2` の全 arm は、PASSしたものも含めて正式結果から除外する。
- 初回結果directoryは `results/mu_titration_0823_invalid_s3_39986e2/` へ退避し、上書きしない。
- 本追補を先にcommitし、その後にS3訂正、configのaddendum pointer、runner/analysis provenance検査をcommitする。
- その新しい単一commitのclean worktreeから8 armすべてを再走する。
- S3以外の学習条件、alpha grid、記録列、scope、theta、bootstrap、C0/C1/W1/C2/P1、総合判定は一切変えない。
- 新しい8 armがS1--S7をすべて通るまで科学集計を実行しない。

## 5. provenance

再走では原specだけでなく本addendumのpathとSHA256も各 arm の `provenance.json` / `arm_meta.json` に保存し、解析前に8 arm間の一致と解析側ファイルとの一致を必須検査する。解析commitは再走commitをGit祖先として持つことも確認する。
