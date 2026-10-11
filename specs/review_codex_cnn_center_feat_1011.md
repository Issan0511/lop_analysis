# cnn_center_feat_1011 本走前 Codex 監査（2026-10-11）

## 判定: **直してから**

H′ の本体、主判定、P181 再利用、2 本キューは通る。一方、C の実走確認に使う `units.npz` が無いとき条件を成立扱いする fail-open は登録判定を誤り得るため、本走前に塞ぐ。

| # | 観点 | severity | 一言 | 根拠 |
|---|---:|---|---|---|
| 1 | 2, 8 | **high** | Hf/CHf の `units.npz` が欠けると C の扉の実走確認を `ok=None` のまま通し、FEATURE_DOOR を適用できる。 | `specs/spec_cnn_center_feat_1011.md:96-99`; `analysis/cnn_center_feat_1011/verdict.py:696-703,856-858` |
| 2 | 5, 8 | **medium** | 採用済み Hs/CHs が未実行でも provenance gate が通り、`NOT_RUN` のまま最終報告できる。 | `specs/spec_cnn_center_feat_1011.md:258-266`; `analysis/cnn_center_feat_1011/verdict.py:323-331,657-674` |
| 3 | 8 | **medium** | `units.npz` / checkpoint の欠損を機構表が黙って飛ばすため、Q5 の説明が空でも主判定だけ出る。 | `specs/spec_cnn_center_feat_1011.md:132-139,225-227`; `analysis/cnn_center_feat_1011/verdict.py:345-368,397-417` |

## 1. H′ の定義と実装 — **PASS**

`m` は c1 後 4,096、c2 後 1,024、fc は100単位で、`ReLU → pool → -m → 次層`。画像だけを平均した中心化前出力を Adam 更新後に、更新前重みの forward から β=.01 で追う（`src/cnn_center_feat_1011.py:115-147,165-194,266-285`）。これは V12 の「隠れ出力の各スカラー特徴からユニット別 EMA を引く」の CNN 隠れ層での厳密対応（V12:698-700）。c1 入力の C は特徴ごとでないという限定も登録済み（spec:54）。

初回は m=0 なので ref/C と同じ、1 更新後は `m=.01 bm`、時定数は約100更新。死んだ特徴で m が約7e-44に止まれば次層へ渡る値は `0-m`≈-7e-44で、float32 実装として実質0かつ仕様どおり。K-Hf-pos/ema/grad/eval/first は pass（`results/cnn_center_feat_1011/checks.json`）。

## 2. 適用可能性 (C′) — **指摘 high**

`h_upd` は「その step で m のどれか1要素が bit 上変化した」を数え（`src/cnn_center_feat_1011.py:130-139`）、課題ごとに0へ戻る（同:501-505）。課題1で1以上なら P181 の死後の非正規数固定点を理由に落とさず、誤実装は K-Hf-ema/loop と engine sha が塞ぐので H 部分は妥当。

ただし C の実走条件は `units.npz` 欠損時 `{"ok": None}` なのに、`is not False` で True 扱いされる。

- 元: `c_door_ok=cdoor.get("ok") is not False`
- 新: `c_door_ok=cdoor.get("ok") is True`
- K-gate/K-verdict に「Hf または CHf の units 欠損 ⇒ FEATURE_DOOR INAPPLICABLE」を追加する。

## 3. FEATURE_DOOR — **PASS**

優先順は `(A/C′) NG → INAPPLICABLE`, `(B) NG → NOT_REPRODUCED`, `CHf RESCUED → SUFFICES`, `CHf=Hf=FAIL → NOT_A_DOOR`, その他 PARTIAL で、spec と一致（`spec:101-110`; `verdict.py:235-245,284-308`）。3⁴×2×2=324 合成入力を CPU で総当たりし不一致0。内訳は INAPPLICABLE 162、NOT_REPRODUCED 81、SUFFICES 27、PARTIAL 45、NOT_A_DOOR 9。P181 監査の high 2件は塞がった。

算術も `k*(8,9,10)=(8,8,9)`、p(8/8)=.0078125、p(8/9)=.0390625、p(9/10)=.021484375（`verdict.py:63-74,202-219`）。

## 4. P181 再利用 — **PASS**

比較対象は新エンジンの再走と P181 退避物で自己比較ではない。課題1の per_task/t0 は文字列、curves/40 checkpoint は bit 比較し、checkpoint は manifest sha256 後に読む（`checks.py:1201-1202`; `spec:239-243`）。独立に、4本の per_task の git blob が b9ae5a26 と一致し、manifest の正規の40個の t01 checkpoint を含む対象 sha に不一致0を確認した。

課題2–30を新エンジンで再計算した照合ではない。ただし全状態が課題1末で同一、ref/C/H/CH の演算経路が K-same181 で同一、旧30課題CSV自体が登録 commit の blob なので、決定論的な帰納と provenance による再利用である。この限界は解釈を妨げない（`spec:64-66,154-155`; `verdict.py:627-640`）。

## 5. 対照 Hs / CHs — **指摘 medium**

Hs は同じ EMA 地図を更新し、減算時だけ全チャネル共通の固定 derangement を当てるため、「その位置との対応」を壊す対照になっている（`src/cnn_center_feat_1011.py:91-103,121-157`）。ただし誤位置平均を引いて残差を増やす介入でもあり、1 seed×1置換なので一般的な必要性は言えない。この限定と `SHUF_UNINFORMATIVE/UNRESOLVED` は適切（`spec:126-130`）。

Hf と H が分かれない可能性は十分あり、かつ CHf が救った場合に C なしの Hs だけでは Q4 に答えられないので CHs 追加には価値がある。実装の比較腕も Hs=(Hf,H)、CHs=(CHf,CH) で正しい（`verdict.py:36-38,248-273`）。しかし§12で両対照を回す決定後も欠損を許す。

- 元: `if a in NEW_ARMS: bad.append(f"{a}: provenance.json missing")`
- 新: `if a in NEW_ARMS + tuple(CTRL): ...`
- §12.1 に「採用後は Hs/CHs の `NOT_RUN` を最終結果で許さない」を追記する。

## 6. 資源の算術 — **PASS**

GPU 閾値は `2048+ceil(1.25×1180)=3523 MiB`、RAM は `6144+1040+ceil(1.2×2310)=9956 MiB` で spec と shell が一致（`spec:268-278`; `launch.sh:19-23,35-51`）。依頼時の free 12,231 MiB なら GPU余裕8,708、MemAvailable 14,538 MiB なら RAM余裕4,582 MiB。90秒ずらすので2本の読み込み峰は重ならず、定常RSS約4.6 GiB後にも6 GiB余白を残す。

swap残0.65 GiBは圧迫の警告だが、MemAvailableを各起動直前に再評価し6 GiBを予約する規則なら、それ自体を停止条件にしない判断は妥当。`[s]rc/... --arm Hf` は自己一致も CHf 取り違えも避け、キューは CHf,Hf,Hs,CHs（`launch.sh:22-28`）。`bash -n` は pass。別 agent が続けば所要が延びるだけで、メモリ算術にはその時点の free/available が入る。

## 7. Codex 自身の予測（spec §7 を読む前に固定）

| 行 | 対象 | 確率（和 = 1） | 根拠 |
|---|---|---|---|
| L1 | CHf | RESCUED .45 / PARTIAL .28 / FAIL .24 / 他 .03 | H′は CH の f1 を沈めた位置DCのµ経路を塞ぐが、5/10 seedのc1を先に落としたbias経路には触らない。 |
| L2 | Hf | RESCUED .15 / PARTIAL .25 / FAIL .57 / 他 .03 | f1の位置DCは消えるが raw 入力平均はc1に残り、MLPでもH単独は第1層死でFAIL。 |
| L3 | FEATURE_DOOR | SUFFICES .45 / PARTIAL .32 / NOT_A_DOOR .20 / 他 .03 | CHf救済を最多としつつ、残るbias経路によるCHf非救済とHf非救済を相関させた。 |
| L4 | T_f: CHf 対 CH | LATER .70 / NO_TIMING_DIFF .23 / EARLIER .07 | CHの早いf1位置DC経路を除くので遅延が本命だが、c1 bias経路が同じ時刻を支配し得る。 |

この表を保存後に spec §7（177–214行）を読んだ。変更した行は**なし**。Claude は CHf FAIL .37 を最多とするが、同じ二経路の不確実性についての別配分であり、新情報ではない。

## 8. 結果が出ても解釈できなくなる落とし穴 — **指摘 high 1 / medium 2**

| 落とし穴 | 起きる条件 | 起きたときの読み方 | 事前に直す案 |
|---|---|---|---|
| 有効seed不足 | diverged/非有限で主腕の有効seedが8未満（表のNaNは(A)も落とす） | 効果なしでなく INAPPLICABLE。 | seed別の除外理由をsummaryへ出す。 |
| (B)不成立 | 再利用refまたはCHがこの規則でFAILでない | H′の数値は報告できるがP181対比は NOT_REPRODUCED。 | 現規則を維持し腕ラベルを併記。 |
| 分ける窓なし | Hf/H、またはCHf/CHの10-seed範囲が全4窓で重なる | SHUF_UNINFORMATIVEは位置対応の反証でない。 | 現ラベルと範囲を必ず報告。 |
| EMA追従遅れ | 切替後約100更新、または平均が速く漂流 | `pdc_post`残りを定義ミスと即断しない。 | `pdc_pre/post`課題列とβ由来床.0177を併記。 |
| C確認fail-open | Hf/CHf unitsが欠損 | 現コードは誤って適用可能になり得る。 | 観点2の `is True` 修正。 |
| 機構物欠損 | units/checkpointが欠損・退避破損 | FEATURE判定は残るがQ5の経路説明は不能。 | 新腕・再利用腕の必須task/shaを検証し、欠損時は黙ってskipせず報告生成を停止。 |
| 対照未完 | Hs/CHsが未起動・途中終了 | 主Q1は残るが、救済時のQ4が未回答。 | 採用済み2対照を完了関門にする。 |
| 時刻の食違い | spec時刻とgit commit時刻が異なる | 内容の先後は本文時刻でなくcommit DAG/hashで読む。 | 最終summaryはhashとcommit時刻を自動出力。 |

## 本走に進んでよいか: **直してから**

go の最小条件は、(1) C 扉の units 欠損を fail-closed (`ok is True`) にし合成検査を足す、(2) 親が採用した Hs/CHs を完了関門にする、の2点。機構物の必須一覧と sha/欠損検査も本走前に入れるのが安全で、いずれも走の設計や予測を変えない。
