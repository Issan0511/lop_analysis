# 実装の依頼: 逆伝播の微分だけを入れ替える腕（bwdswap_1111）

## 目的

論文 V13 の段 ④「活性化の微分を失ったユニットが新しい課題への適応を遅らせる」は、付録 E.1 の介入で確かめた。第 2 隠れ層の活動を

  ã(x) = φ(z⁰(x)) + [ φ(z(x) + d(x)) − φ(z⁰(x) + d(x)) ]

に置き換え、同じ軌道の別の時点（供与側）の前活性とのずれ d = zᴰ − z⁰（画像ごとに固定）を渡す。分岐の時点では活動・予測が一致し、学習に使う局所微分は φ′(z + d) になる。ただし分岐の後は、順伝播の応答も対照から離れる。査読者 3 人は全員、「逆伝播の微分だけを替え、順伝播の応答は揃えたままにする対照」を求めた。

今回足す腕（B = backward only）は、順伝播は自然な網そのもので、逆伝播の局所微分だけを φ′(z + d) にする:

  a(x) = φ(z(x))（順伝播の値）, ∂a/∂z = φ′(z(x) + d(x))（逆伝播）

torch では `a = phi(z).detach() + (phi(z + d) - phi(z + d).detach())` の形で書ける（順伝播の値は φ(z) と厳密に一致し、勾配は φ′(z + d)）。ただし、各箱の既存の実装の数え方（演算の順・bit 一致の検査・CUDA graph）に合わせること。d の作り方（供与側と受け手の凍結した網の前活性の差、画像ごとの表）は E.1 と同じにする。

## 三つの設定（付録 E.1 の表）

| 設定 | 活性化 | 受け手 ← 供与側（課題の末） | 次の課題・予算 | 既存のコード（推定。確かめること） |
|---|---|---|---|---|
| RC（乱数ラベル CIFAR・MLP） | ELU・std | 10 ← 1（戻す）、逆 1 ← 10（渡す） | 11 / 2、各 30,000 更新 | `src/resp_cifar_ee_0920.py`（`src/cifar_interventions_0920.py` を使う・CUDA） |
| RM（乱数ラベル MNIST・2 隠れ層 MLP） | ELU | 10 ← 2、逆 2 ← 10 | 11 / 3、各 6,000 更新 × 2 課題 | `src/resp_ee_0917.py`（CPU） |
| 5+1（実ラベル CIFAR-100 の 5+1・MLP） | ReLU・std | 28 ← 2、逆 2 ← 28 | hard 29 / 3、各 780 更新 | `src/resp_cifar5p1_1007.py`（CUDA・CUDA graph） |

Adam の扱いは E.1 と同じ（戻す側は継続、渡す側は両方の条件で初期化）。主の集計は E.1 と同じ（次の 1 課題の online 精度の 10 seed 平均・対応差の 97.5% t 区間）。E.1 の数は付録の「E.1 への追記」の表にある。戻す向き +21.3〜+76.0 ポイント、渡す向き −67.9〜−6.0 ポイント（10/10）。

E.1 の作りの元の spec は `specs/spec_resp_cifar_ee_0920.md`・`specs/spec_resp_ee_0917.md`・`specs/spec_resp_cifar5p1_1007.md`。三つの設定をまとめて回し直した `src/resp_step_controls_1018.py`（付録 E.3。戻す向きだけ）も参考になる。付録の原稿は `/tmp/claude-1000/-home-issan-Projects-claude/71433115-54e9-4cf1-bae9-6c9b92eb9b22/scratchpad/kune/zip/base_appendix_v6.md` の E.1〜E.3（352 行から）と、同じ folder の `appendix_ja_v6.md` の「E.1 への追記」。

## 腕（各設定・各向き・10 seed）

- N: 対照（自然な網）
- X: 既存の入れ替え（E.1 と同じ。順伝播も替わる）
- B: 新しい腕（逆伝播の微分だけ φ′(z + d)、順伝播は自然）

N と X は、E.1 と同じ数を出すはずである。CPU の RM は bit 一致を確かめる。GPU の RC・5+1 は、再起動をまたぐと結果が変わりうる（white-san は 11/05 に kernel が替わった）ので、N・X・B を同じ起動の中で全部回し直す。E.1 の数との差は記録するが、判定は今回の N・X・B の間で行う。

## やってほしいこと

1. 三つの設定の既存のコードを読み、E.1 の N と X を作った正確な道（spec・腕の名・分岐の課題・seed・Adam の扱い・online の集計）を特定する。不明な所は推測と書く。
2. このworktree（`/home/issan/Projects/claude/wt/bwdswap_1111`、ブランチ `claude/bwdswap_1111`）の中に、B の腕を足した走の script を書く。既存の file は変えず、新しい file（例 `src/bwdswap_1111.py`、必要なら設定ごとに分ける）を作り、既存の module は import して使う。
3. 検査を書く:
   - B の順伝播の値が、自然な網（N）と毎歩 bit で一致する（同じ重みの状態に対して）。少なくとも分岐の後の最初の数歩で、B の網の logit が、同じ重みに自然な順伝播を当てた logit と一致することを確かめる。
   - B の逆伝播の局所微分が φ′(z + d) と一致する（数値で）。
   - 分岐の時点（更新 0）で、N・X・B の logit・損失が一致する。
   - N と X が、短い走で既存のコードと同じ数を出す（RM は bit 一致、GPU は同じ起動の中で再現）。
4. 短い走（smoke: 各設定 1〜2 seed・予算を短く）で全部の検査を通す。所要時間と最大メモリ（RSS と GPU メモリ）を測る。
5. 本走の起動 script（例 `analysis/bwdswap_1111/launch.sh`）を書く。並列の本数は、CPU 側は同時 10 本まで、GPU は 1 枚（RTX 5060 Ti 16 GB）を見て決める。**本走は起動しない。** 起動は依頼主（親）が予測を登録してから行う。
6. 集計の script（例 `analysis/bwdswap_1111/judge.py`）の骨組みを書く: 各設定・各向きで、N・X・B の次の課題の online 精度の 10 seed 平均と、対応差 X − N・B − N・B − X の 97.5% t 区間、seed ごとの向き。

## 守ること

- 書いてよいのは `/home/issan/Projects/claude/wt/bwdswap_1111` の中だけ（`src/`・`analysis/bwdswap_1111/`・`specs/` の新しい file と、`results/bwdswap_1111/` の下）。ほかの worktree・`proj_004_drift`・データの folder は読むだけ。
- git の commit・push・branch の操作はしない（親が見てから commit する）。
- Python は `/home/issan/Projects/claude/proj_004_drift/.venv/bin/python` を使う。
- 計算機は共有である。smoke でも CPU は同時 4 本まで、GPU は 1 本ずつ。走らせる前に `free -h` と `nvidia-smi` を見る。swap はほぼ埋まっているので、メモリを多く使う並列はしない。
- 既存の file（特に `src/` の既存の script と `results/` の既存の結果）を変えない。

## 報告

日本語で、次を返す。作った file の一覧、三つの設定それぞれの E.1 の作りの特定（どの script のどの腕が N と X か）、検査の結果（数）、smoke の所要時間とメモリ、本走の見積もり（本数 × 1 本の時間 × 並列）、決められなかった点。
