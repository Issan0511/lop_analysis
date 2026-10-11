# cnn_center_relu_1011 本走前監査（Codex、独立の第二の目）

日付: 2026-10-11  
監査対象: 依頼時の `fdb97c2b`〜`3f635551`。監査中に HEAD が別作業者の `653dc304`、さらに `8f6dd5a2` へ進んだため、§12、新しい `launch.sh`、報告用三分解も追って確認した。訓練・本走・`checks.py` 全実行・`launch.sh`・`cost.py` は実行していない。

## 判定: **直してから**

C/H の訓練実装、C の LRc との bit 同一性、ref の再現、主窓・閾値・符号検定の算術は通っている。一方、要約 `DOORS_TRANSFER` / `DOORS_FAIL` は、あり得る腕ラベルの組合せで登録した日本語と矛盾する。これは結果を誤って要約する high であり、本走前に spec 追補・`verdict.py`・合成検査・L6 の確率を揃える必要がある。

| # | 観点 | severity | 一言 | 根拠 |
|---|---:|---|---|---|
| 1 | 1 | **high** | H 単独が RESCUED（`N=NEED_H`）でも `DOORS_TRANSFER` になり、「MLP と同じ並び」ではない結果を移植成功と呼ぶ。 | `specs/spec_cnn_center_relu_1011.md:143-149`; `analysis/cnn_center_relu_1011/verdict.py:205-215`; `analysis/cnn_center_relu_1011/checks.py:677-682` |
| 2 | 1 | **high** | C または H が RESCUED でも CH が FAIL なら `DOORS_FAIL` になり、「中心化は崩壊を防がない」という読みが偽になる。 | `specs/spec_cnn_center_relu_1011.md:148-150,182-186`; `analysis/cnn_center_relu_1011/verdict.py:205-215` |
| 3 | 1, 9 | **medium** | verdict は腕名・seed 集合・30 行の一意性・provenance・実走した C の入力を検証せず、古い `checks.json` の boolean だけを信じる。 | `analysis/cnn_center_relu_1011/verdict.py:109-147,265-289,550-566`; `src/cnn_center_relu_1011.py:468-479,535-547` |
| 4 | 2, 3, 9 | **medium** | ここでの C/H は MLP の演算の厳密移植ではなく CNN 用の適応版。とくに C は seed ごとの特徴平均でなく訓練集合のチャネル平均、H は位置ごとでなくチャネル平均である。 | `specs/spec_relu_doors_0919.md:30-33`; `specs/spec_cnn_center_relu_1011.md:61-69` |
| 5 | 4 | **medium** | Cr の完走・有限性を確認せずラベル化し、`EVALOFF_NO_DEPENDENCE` は「差を検出しない」を依存なしと言い過ぎる。 | `analysis/cnn_center_relu_1011/verdict.py:234-262,311-315`; `specs/spec_cnn_center_relu_1011.md:169-172` |
| 6 | 9 | **medium** | 監査対象 `3f635551` の verdict は、spec が約束した µ 経路の自分・上流・交差の三分解を出さない。後続 `8f6dd5a2` に候補実装が入ったが未検査。 | `specs/spec_cnn_center_relu_1011.md:106-110`; `analysis/cnn_center_relu_1011/verdict.py@3f635551:332-365`; `analysis/cnn_center_relu_1011/verdict.py@8f6dd5a2:379-416` |
| 7 | 7 | **low** | B2 は「9 本以上」と登録したのに、欠損時は `kstar(n)`（n=9 なら8本）で採点する。 | `specs/spec_cnn_center_relu_1011.md:234-247`; `analysis/cnn_center_relu_1011/verdict.py:513-520` |
| 8 | 2 | **low** | spec の訓練微分 `1[z>0]` は、実装した `torch.clamp` の z=0 における微分1と一致しない（診断ゲートは z>0 のまま）。 | `specs/spec_cnn_center_relu_1011.md:64`; `src/cnn_center_relu_1011.py:137-141`; `specs/spec_cnn_center_relu_1011.md:286` |
| 9 | 9 | **low** | `A(1)≤0.1` のとき `T_f=1` を返し、「最初の t≥2」という登録定義から外れる。 | `specs/spec_cnn_center_relu_1011.md:108`; `analysis/cnn_center_relu_1011/verdict.py:91-99` |

## 1. 判定規則 — **指摘 high ×2、medium ×1**

### 通っていること

- 主窓は inclusive な t21–30 の10課題平均で、`W >= 0.5` が KEPT、10 seed では9本以上が RESCUED、1本以下が FAIL、AT_FLOOR 9本以上が `FAIL_FLOOR` である。spec の根拠は、0.5 が MLP の登録線であり、CNN の A(1)=0.954 なら q=(0.5−0.1)/(0.954−0.1)=0.47 と V12 の q<1/2 にほぼ一致する、まで書かれている（`specs/spec_cnn_center_relu_1011.md:114-125`; `analysis/cnn_center_relu_1011/verdict.py:29-38,86-99,109-129,167-186`）。
- 正確な両側符号検定を CPU で確認した。10/0 は p=0.001953125、9/1 は p=0.021484375 で HELPS/HURTS、8/2 は p=0.109375 で TIE。`kstar(8,9,10)=(8,8,9)` も正しい（`analysis/cnn_center_relu_1011/verdict.py:48-79`）。
- ref/C/H/CH の各3ラベル、D\_{H|C} の3ラベル、C の t1 学習の真偽、適用条件の真偽を全972組合せで `ladder()` と `doors_summary()` に入れた。コード上は出力なし・二重 return はなく、出力集合は N の6値、DOORS の5値を全て含んだ。作業スクリプトは `/tmp/codex_relu_audit_1011/audit_cpu.py`。

### high 1: `DOORS_TRANSFER` が H の結果を見ない

実例は `ref=FAIL, C=FAIL, H=RESCUED, CH=RESCUED, D_H|C=HELPS, C_t1=True` で、現コードは `N=NEED_H` と `DOORS_TRANSFER` を同時に返す。検査もこの矛盾を期待値として固定している（`analysis/cnn_center_relu_1011/checks.py:677-682`）。しかし既知の MLP は H が FAIL で `N=NEED_CH` だった（`specs/spec_cnn_center_relu_1011.md:41-43`）。

修正案:

- `3. CH が RESCUED（CH 保つ）。` → `3. H が FAIL（H 単独は救わない）。4. CH が RESCUED（CH 保つ）。`
- `4. D_{H|C} ... HELPS` → `5. D_{H|C} ... HELPS`
- コードの `labels["CH"] == "RESCUED" and labels["C"] != "RESCUED"` → `labels["CH"] == "RESCUED" and labels["C"] != "RESCUED" and labels["H"] == "FAIL"`

より簡潔には `N == NEED_CH` を必要条件にしてよい。H=PARTIAL も「MLP と同じ」ではないので `DOORS_PARTIAL` に送るのが明瞭である。

### high 2: `DOORS_FAIL` が他腕の救命を無視する

`ref=FAIL, C=RESCUED, H=FAIL, CH=FAIL` は現コードで `N=NEED_C` かつ `DOORS_FAIL`、`ref=FAIL, C=FAIL, H=RESCUED, CH=FAIL` は `N=NEED_H` かつ `DOORS_FAIL` になる。このとき「中心化は ReLU の CNN の崩壊を防がない」（§5.12）は偽で、正しくは「組合せ CH は失敗し、相互作用が有害」である。

修正案:

- ``DOORS_FAIL: CH が FAIL`` → ``DOORS_FAIL: C・H・CH がすべて FAIL``
- コードの `if labels["CH"] == "FAIL":` → `if all(labels[a] == "FAIL" for a in ("C", "H", "CH")):`
- CH が FAIL でも C/H のどちらかが RESCUED または PARTIAL なら `DOORS_PARTIAL` とし、どの腕が救った／部分的だったかを本文に出す。

この修正は L6 の事象を変える。元の L6 を `DOORS_v1` として保存し、修正版を主判定にするか、修正版に対する Claude/Fable/Codex の確率を結果を見る前に再登録する必要がある。

### medium: 入力ファイルと provenance の検証がない

`load_frames()` はディレクトリ名だけでCSVを読み、`seed_table()` は未知の seed も数える。`checks_state()` は `checks.json` の `all_pass` と `K-center.pass` だけを読む。したがって、腕CSVの取り違え、seed 20の混入、重複 task、別 epochs/lr、dirty engine、検査後に変わった engine を拒否しない。実走側は十分な provenance を保存しているのに verdict が使っていない。

修正案:

- `frames = {a: pd.read_csv(...) for a in MAIN_ARMS}` → `frames = load_and_validate_frames(...)`（各腕の `arm` 列、seed=10–19、各 seed の task=1–30 が一意、必須列 finite を検証）。
- `return bool(c.get("all_pass")), ...` → engine の sha256 `a7517aa5…`、検査時の engine sha256、各腕 `provenance.json` の arm/slots/tasks/epochs/lr/beta/data hash/dirty を照合してから True を返す。
- C/CH の適用条件は静的 K-center だけでなく、実走の `input_shift` と `units.npz` t0 の `m_in_c1` を ref と比較して `<0.1` を確認する。

## 2. H の実装 — **実装 PASS、解釈に medium、文言 low**

- c1/c2 は pool 後の ReLU 出力を画像×位置で平均したチャネル別 m、f1/f2 は画像で平均したユニット別 m（`src/cnn_center_relu_1011.py:149-167`）。β=0.01、初期0、`no_grad`、Adam後に更新前重みの forward 出力で更新する（同:290-309）。
- forward は `ReLU -> pool -> -m -> 次層`、fc も `ReLU -> -m -> 次層` で、減算後に ReLU を掛け直していない（同:189-218）。pool 前にチャネル定数を引く場合と値は可換で、追跡する平均を pool 後に置く判断は論理的である。
- online/memo/switch/診断は同じ `act.m` を使い、m=0 は明示 override の読み取り評価だけ（同:225-244,406-420）。`checks.json` の K-H-ema/grad/eval/first/evaloff は pass し、対応変異を検出している。
- 初回 forward は m=0 なので H=ref、最初の Adam 更新後に m₁=0.01·batch_mean、2回目から差が出る。有効時定数は約100更新で、最初の数十更新は中心化が弱い。これはバグではないが、task1 の低下は `T1_LOWER` と EMA の立上りを併記して読む。

ただし MLP の H は各特徴（100ユニット）の平均を引くのに対し、CNN の c2→f1 は1024位置特徴を16チャネル平均だけで処理し、位置ごとの DC を残す（`specs/spec_cnn_center_relu_1011.md:66-69`）。失敗時に「MLP の H は CNN へ移らない」とは言えず、「pool後・チャネル平均版 H は移らない」が限界である。

修正案:

- `MLP の H ... の移植。` → `MLP の H を CNN の畳み込み対称性に合わせてチャネル平均へ適応した版。特徴（位置）ごとの厳密移植ではない。`
- `∂a/∂z = 1[z>0]` → `m の減算は微分を変えない。訓練実装 torch.clamp の微分は z>0 と z=0 で1、z<0で0、診断ゲートは z>0。`

## 3. C の実装 — **PASS（ただし MLP との同一性は限定）**

`train_channel_mean()` の式は `cnn_center_1010` と同一で、float64 で5万枚の平均を取り float32 に戻し、seed 10–19 の画像テンソル構築後に一度だけ引く（`src/cnn_center_1010.py:53-70`; `src/cnn_center_relu_1011.py:82-110,251-279`）。K-center は全 seed の入力テンソルが LRc と bit 一致し、平均パッチノルム 3.758–3.842 → 0.0149–0.0496 を記録している（`results/cnn_center_relu_1011/checks.json:61-90`）。seed、subset、label、batch stream も継承元と同じである。

ただし MLP の C は「その seed の1200枚の特徴ごとの平均」を引く（`specs/spec_relu_doors_0919.md:30-33`）。本実験は「cnn_center_1010 の全訓練集合チャネル平均 C」の移植であり、MLP 演算の bit/数学的同一移植ではない。主張の限定は観点9に記す。

## 4. Cr と EVALOFF — **指摘 medium**

Cr の幾何は式どおりである。r=(0.2562,0.0905,1.1282)、残る DC のノルム比は raw の0.989–1.012、raw 方向との cos は−0.009–0.016で、同じ大きさをほぼ直交方向へ残している（`results/cnn_center_relu_1011/checks.json:91-139`）。`cr_label()` の4分岐は、有限な完走1 seedを前提にすれば排他的・網羅的である。

問題は Cr seed10 の `valid` を見ずに `.loc[10]` を渡す点である。欠損窓や NaN を `CR_UNRESOLVED` / `CR_UNINFORMATIVE` と誤読し得る。

修正案:

- `cst.loc[CR_SEED] if CR_SEED in cst.index else None` → `cst.loc[CR_SEED] if CR_SEED in cst.index and bool(cst.loc[CR_SEED,"valid"]) else None`
- Cr は seed10・task1–30 が一意で、online/memo/4窓が有限でなければ `INAPPLICABLE` と理由を出す。

EVALOFF は「訓練済み関数から t30 評価時だけ m を外した分布シフト」である。`EVALOFF_DEPENDS` は t30 の関数が m の存在に依存する証拠だが、m が訓練中に救命を因果的に生んだ証拠ではない。逆に TIE は依存なしの証明ではない。

- `EVALOFF_NO_DEPENDENCE` → `EVALOFF_NO_DETECTED_DEPENDENCE`
- `網が扉の m に依存している` → `t30 の memo 関数が評価時の m に依存している（訓練効果の因果判定ではない）`

## 5. ref の bit 再現 — **PASS**

K-repro は新しいファイルを `REF_R` と別々に読み、旧CSVの全列が新CSVにあること、arm を除く共通列の文字列列全体、`curves.npy` の shape と int32 view を比較している（`analysis/cnn_center_relu_1011/checks.py:210-229`）。比較対象は別 inode で自己比較ではない。

独立に読み比べた結果:

- 新CSV `(3,91)`、旧CSV `(3,58)`。arm を除く旧57列は欠損0・差0。新側の追加33列は今回の診断列。
- curves は両方 `(1,3,400)`, float32、bit 一致。ファイル sha256 も両方 `2d4fcdf…`。
- CSVファイル全体の sha256 が違うのは追加列と arm 名 (`ref` 対 `R`) のためで、検査の比較対象選択は正しい。

## 6. 予算と同時本数 — **PASS（最新 §12 を正とする）**

`checks.json` は16機能検査+K-cost=17項目、変異27/27を検出、検査時 hash `0a5c0873`、dirty false。engine の現在の sha256 は検査時と同じ `a7517aa5…` だった。

依頼記載の現況を旧 §8 の式に代入すると、M_free=13,593 MiB、既存の別 agent 888 MiB、余裕2,048 MiBなので M_keep=2,936 MiB、分母1.25×1,170=1,462.5 MiB、

`n = floor((13,593 − 2,936) / 1,462.5) = floor(7.287) = 7`。

5本の安全係数込み需要は7,312.5 MiBで、M_free−M_keep=10,657 MiBより小さいため、GPUメモリだけなら5本同時は可能である。なお M_free は既存888 MiBを既に差し引いた値なので、それを M_keep に再度入れる旧式は保守的な二重控除だが、nは888を再控除しなくても7で変わらない。

ただし監査中に追加された §12 は RAM の実測（定常2,345 MiB、load峰3,395 MiB）を入れ、既定 `MAX_CONC=2` の逐次キューへ変更した（`specs/spec_cnn_center_relu_1011.md:305-321`; `analysis/cnn_center_relu_1011/launch.sh:18-50`）。総資源で見れば**5本同時にはせず2本まで**が正しい。新 launch は各起動直前に GPU/RAM/本数を再計算し、依頼時の5本一斉方式より安全である。`bash -n` は pass。安全係数 `1170*125/100` は整数切捨てで1462 MiBになるため、厳密には `(1170*125+99)/100=1463` と切上げるのが望ましいが、差0.5 MiBは low 未満である。

seed 10–19 は `sna_cnn_cause_1009`、`cnn_drive_verify_1009`、`cnn_center_1010` と整合し、init/subset/label/batch の流れも同じ（`src/cnn_center_relu_1011.py:265-280`）。

## 7. spec の予測節 — **概ね PASS、B2 のみ low**

§7 は観点8の表をこのファイルに固定した後に読んだ。Claude L1–L8、Fable L2–L7 は各行の和が1で、実装定数も全行1だった。Claude の「他」は名前のないラベルを1桶にし、L6 では NOT_REPRODUCED と INAPPLICABLE の和である。Fable L6 は NOT_REPRODUCED .08 を明記し、INAPPLICABLE は0、L7 の OTHER は非 CORRESPONDENCE 全部を受ける。`score_labels()` の Brier も未列挙ラベルを確率0の追加カテゴリとして扱う（`analysis/cnn_center_relu_1011/verdict.py:434-461`）。B1 は ref FAIL でなければ未採点になっている。

B2 だけ、登録文は「c2 が入る seed が9本以上」なのに、コードは有効 n の `kstar(n)` を使う。n=9なら8本で真になる。

- `>= (kstar(len(vr)) or 99)` → `None if len(vr) != 10 else count >= 9`
- または spec を「有効 n の k*(n) 本以上、n<10ならその旨を採点に併記」に直す。元の予測文を尊重するなら前者。

## 8. Codex 自身の予測 — **PASS（事前記録済み）**

> **独立性の開示**: 下表は `specs/spec_cnn_center_relu_1011.md` §7（215–253行）を開く前に固定した。ただし判定実装を通読した際、`analysis/cnn_center_relu_1011/verdict.py:389-410`（当時の行番号）に複製された Claude/Fable の確率が視界に入ったため、完全な非誘導ではない。下の値は §0–6、V12 §3.3/§10.2、MLP の既存結果、`cnn_center_1010` の leaky CNN の結果から自分で割り当てた。

| 行 | 項目 | §7 前に固定した確率（合計1） | 根拠（1行） |
|---|---|---|---|
| L1 | ref | FAIL .97 / PARTIAL .01 / RESCUED .00 / 他 .02 | 宿主 seed 0–9 の主窓が全て0.5未満で、seed10 の短走も t3=0.123。 |
| L2 | C | FAIL .72 / PARTIAL .20 / RESCUED .06 / 他 .02 | leaky CNN では低下を減らしたが c2 は深く沈み、MLP の C も深層崩壊を防げなかった。 |
| L3 | H | FAIL .28 / PARTIAL .25 / RESCUED .45 / 他 .02 | MLP は c1 死だが CNN ref は c2 が先に閉じ c1 死亡率が低いため、H 単独を MLP より有望視。 |
| L4 | CH | FAIL .11 / PARTIAL .17 / RESCUED .70 / 他 .02 | MLP で完全救命し、CNN の c1 DC と c2以降の平均入力経路を同時に塞ぐ。 |
| L5 | N | NEED_C .06 / NEED_H .40 / NEED_CH .32 / NONE_RESCUED .18 / 他 .04 | C 単独は低く、CNN では H 単独を本命、H が足りない場合の CH を次点とした。 |
| L6 | 要約 | DOORS_TRANSFER .55 / DOORS_PARTIAL .30 / DOORS_FAIL .11 / NOT_REPRODUCED .02 / INAPPLICABLE .02 | ref再現、Cのt1学習、CH救命、CH−C差が揃う公算を過半とし、部分救命・差不成立を残した。 |
| L7 | LAYER | LAYER_CORRESPONDENCE .40 / LAYER_CORRESPONDENCE_NOT_SHOWN .56 / NOT_REPRODUCED .04 | Cの深層先行は有力だが、Hのc1が30課題内に9/10で mob<.01へ先着する条件は厳しい。 |
| L8 | Cr（任意） | CR_UNINFORMATIVE .55 / CONST_SHIFT_NO_EFFECT .15 / CONST_SHIFT_ACTS_LIKE_C .05 / CR_UNRESOLVED .23 / 他 .02 | ref/Cとも後期FAILなら分ける窓がなく、分かれても1 seedが全窓で一方の範囲に入る保証は弱い。 |

§7 を読んだ後に変えた行: **なし**。主な相違は H の RESCUED を Codex .45、Claude .28 とした点である。ただし観点1の修正で L6 の定義を変えるなら、上の L6 は旧定義への予測として保存し、新定義の確率を結果を見る前に別途固定する。

## 9. 結果が出ても解釈できなくなる落とし穴 — **指摘 medium**

| 落とし穴 | 起きる条件 | 起きたときの読み方 | 事前に直す案 |
|---|---|---|---|
| ref 非再現 | ref が FAIL でない（Codex予測: PARTIAL/RESCUED 1%、INAPPLICABLE 2%は別） | N/DOORS/LAYER は NOT_REPRODUCED。扉の成否ではなく箱の再現失敗。 | 現規則を維持し、ref の W・有効seed・宿主との差を最初に報告。 |
| C が task1 を学ばない | C の t1 online≥.5 が k*未満 | CHが救っても `DOORS_PARTIAL`。これは「CH無効」でなく MLP の C一時という順序が再現しない。 | summary に条件2の真偽と本数を必ず出す。 |
| H の死層免除 | raw ReLU 出力が全要素0で dead_frac=1、mが0へ減衰し下位bitで停止 | h_upd不足は装置故障でなく入力信号消失。ただし次層は減衰中の −m を一時受ける。 | 免除を維持し、m_norm の減衰列と exact-zero の頻度を併記。 |
| AT_FLOOR | W≤F_s+3s_s/√10。既知較正では約.113–.120 | FAILの内訳だけで、RESCUED/FAIL本体を変えない。床より少し上でも FAIL_ABOVE_FLOOR。 | verdict に seed別 F,s,閾値,W を保存。 |
| 主窓 | t21–30だけで前半が高く後半が落ちる、またはt30直後に落ちる | 30課題内の後期10課題だけの救命。無限時間の非崩壊ではない。 | 現窓を維持し、副窓・T_f・t30を併記。 |
| f1 の位置DC | チャネル平均は0に近いが8×8位置ごとの平均マップが残る | H/CHのFAILは特徴ごとのMLP Hの反証でなく、チャネル版Hの失敗かもしれない。 | §5.12の主張を限定し、`r_in_f1` を必須表にする。 |
| C の定義差 | train全体の3チャネル平均は引くがseed別3072特徴平均は引かない | 「MLPのCの厳密移植」ではなく、CNNで自然な色DC除去の移植。 | 主張を「CNN用C/H」に限定。将来は seed別位置特徴Cを別腕にする。 |
| EMA立上り | task1最初の約100更新で m がまだ追いつかない | task1低下は定常Hの害と即断できない。 | T1_LOWER、最初のepoch列、m_normを一緒に読む。 |
| 要約の矛盾 | H救命+CH救命、またはC/H救命+CH失敗 | 現コードでは `NEED_H+TRANSFER`、`NEED_C/H+FAIL` が同時に出る。 | 観点1の high 2件を本走前に修正。 |
| provenance未検証 | CSV取り違え、wrong seed、重複task、dirty/別engine | 数字は有限でも登録箱の判定ではない。 | `load_and_validate_frames` と engine sha/provenance gate を追加。 |
| Cr 1 seed/range | Cr が範囲外、またはref/C範囲が重なる | UNRESOLVED/UNINFORMATIVE は機構反証でない。`NO_EFFECT` も一般の定数shift全体でなく、この直交1方向だけ。 | ラベルの本文を `TRACKS_REF_THIS_DIRECTION` と限定し、valid gateを追加。 |
| EVALOFF TIE | t30 memo差の符号が揃わない | 「依存なし」ではなく「t30で依存を検出しない」。訓練中の因果はどちらのラベルでも不明。 | 名称と読みを観点4の案へ変更。 |
| T_f の分母 | A(1)≤0.1 | 現コードは T_f=1を返すが、登録定義は最初の t≥2であり不整合。 | `return 1` → `return NaN/INAPPLICABLE`（または2と明記）。この実験ではref/Cの既知値から低確率。 |
| µ経路三分解 | 結果後に自分・上流・交差を分けたい | `3f635551` の report は総量しか出さないが、保存済みunits+ckptから後算可能で再走不要。 | `8f6dd5a2` の `three_way_table()` を合成checkpointで検査する。 |

## 本走に進んでよいか: **直してから**

go に必要な最小条件:

1. `DOORS_TRANSFER` に H=FAIL（または N=NEED_CH）を加え、`DOORS_FAIL` を C/H/CH 全FAILに限定する。spec追補、verdict、合成検査、L6予測の扱いを同時に固定する。
2. verdict に腕/seed/task/provenance/engine sha の gate と、実走 C/CH の条件(C)確認を加える。
3. Cr の valid gate、EVALOFF の限定した読み、B2の9本固定を直す。
4. `3f635551` で欠け、後続 `8f6dd5a2` に追加されたµ経路三分解を、小さな合成checkpointで検査する。
5. 変更後に CPU の全組合せ検査（少なくとも上記の矛盾3例、972組合せ、wrong-arm/wrong-seed/duplicate-task/provenance不一致）を通し、作業木の `src/analysis` を clean にする。GPU本走はその後。最新 §12 の既定2本キューを使い、5本同時にはしない。
