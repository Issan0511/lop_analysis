# chargpt_detlabel_1023 の報告（byakko、2026-10-24）

spec: `specs/spec_chargpt_detlabel_1023.md`（登録 db44d165、追補 1 a078e2b4）。判定の全文は `judge.md`・`judge.json`。どれも箱 M・この Adam・lr 1e−3・2 ブロック・seed 0〜9 の範囲の結果で、その外へ一般化しない（spec §9）。

## 1. 結論（A4 と S2 の判定、§7）

| 判定 | A4（α 1e−4、主） | S2（α 0.01・目標を真の分布、判定） |
| --- | --- | --- |
| J1 喪失 | **LOSS_GONE**（LOSS 0/10） | **LOSS_PERSISTS**（LOSS 10/10） |
| J2 L0 の伸び | **L0_GROWTH_SIMILAR**（ρ2 = 0.77。g 11.7 [9.75, 15.9] 対 A2 15.1） | **L0_GROWTH_SIMILAR**（ρ2 = 1.24。g 18.8 [14.9, 25.5]） |
| J3 L0 の死 | **L0_DEATH_REDUCED**（課題 100 末の死の中央値 0、d ≥ 0.5 は 0/10） | **L0_DEATH_REDUCED**（中央値 0.034 < 0.433。ただし d ≥ 0.5 は 4/10、対の符号 5 対 5・p = 1） |
| J4 後半の動き（net5k） | **MOTION_SIMILAR**（ρ4 = 0.60。M 0.361 対 A2 の byakko での再生 0.601、10/10 対で A4 が小さい、p = 0.002） | **MOTION_SIMILAR**（ρ4 = 1.04） |
| J5a L1 の早期の死 | **NO_EARLY_DEATH**（0/10） | **MIXED**（5/10） |
| J5b 後期の戻り | **INSUFFICIENT**（早期に死んだ seed 0） | **INSUFFICIENT**（早期に死んだ seed 5 < 6。5 seed とも戻った） |
| J6 後期の他 unit 項 | **OTHER_TERM_FLIPS**（≥ 0.75 が 0/10。後期の中央値 0.564、A4 > A2 の対 10/10） | **OTHER_TERM_FLIPS**（1/10） |
| J6-LN | **LN_HALVES**（10/10） | **LN_HALVES**（10/10） |

- J6・J6-LN のラベルには「S-J6 の unit ごとの確認待ち」の印が付く（追補 1 §13.2）。判定に使う量は A2 で loadsign と完全に一致しているが、符号が食い違った unit が近零に限られることの unit ごとの確認には loadsign の unit の配列（white-san にある 30 ファイル）が要る。
- J1 の相ごとの喪失の seed 数（T015 / E1000 / ELast）は、A4 で 1 / 3 / 1、S2 で 10 / 10 / 10。ΔE は A4 0.0013 [0.0005, 0.0040]、S2 1.49 [1.12, 1.93]（A2 1.075）。J1b（追補 6 の条件 1）は A4 0/10、S2 10/10。
- spec §11 の表に当てると、A4 は「消える」の行の条件（J1 `LOSS_GONE`）に当たる。S2 では喪失が残った（J1 `LOSS_PERSISTS`）ので、§11 の続きの文は「S2 では残った」の側になる。ここでは本文の文を書かない（親と Issa の判断）。

## 2. 予測の採点（§3、A4 について。数は書き換えていない。Issa の欄は空欄で登録）

| 予測 | 採点の規則 | 結果 | Fable p（Brier） | Opus 5.5 p（Brier） |
| --- | --- | --- | --- | --- |
| P1 | J1 が LOSS_PERSISTS かつ ΔE が A4 < A2 の seed 8/10 以上 | 偽（J1 は LOSS_GONE。ΔE は 10/10 で A4 < A2） | 0.55（0.303） | 0.35（0.122） |
| P2 | J2 が L0_GROWTH_SMALLER | 偽（SIMILAR、ρ2 0.77） | 0.60（0.360） | 0.35（0.122） |
| P3 | ρ4 ≤ 1/3（net5k） | 偽（ρ4 0.60） | 0.80（0.640） | 0.15（0.023） |
| P6 | J6 が OTHER_TERM_KEPT | 偽（FLIPS） | 0.40（0.160） | — |
| P6-LN | J6-LN が LN_HALVES | 真 | 0.70（0.090） | — |

## 3. 検査・台・版・A2 との差

- 台: byakko、NVIDIA GeForce RTX 5070 Ti 16 GB（driver 580.178.04、CUDA 13.0）、Python 3.12.7、torch 2.13.0+cu130、numpy 2.5.1。A2 は white-san の RTX 5060 Ti・Python 3.12.3（torch・CUDA・numpy は同じ版）。
- 走る前の検査（commit 8f7ec958、`checks/`）:
  - S-code PASS（blob cc524f85・272e7554、RUN_ID 不変）。
  - S-stream PASS（α 0.01・seed 0〜9・課題 1〜3 の rule/data/probe_sha 30/30、init_sha 10/10 が A2 と一致）。
  - S-recorder（§12.3 の読み替え）: (a) byakko で記録器あり・なしがビット一致（online と checkpoint 9/9）PASS。(b) A2 seed 0 との差は課題 1 が更新 2 から、課題 2 が更新 0 から離れ、更新ごとの差の最大 0.021・0.081、平均 0.0033・0.0045。
  - S-pi PASS 3/3。S-soft PASS（真の分布の CE の手計算との相対差 1.4e−7、online のサンプル CE は 20/20 ビット一致）。
  - S-J6: 恒等式と autograd は ≤ 3.5e−15 で PASS、states.csv の再現は符号のずれ 0。components の符号の割合が 1 個ずれ（判定に使わない列）、追補 1 §13.2 の条件（判定に使う量の完全一致と既存の数値の検証）で通した。全 350 状態では 12,950 の状態 × 列のうち 86 がずれ、数え上げの必要条件の破れは 0、後期のずれは 0。本走の 3 腕でも gate（恒等式・autograd・π のビット一致）は 3/3 通った。
  - judge.py は A2 の記録だけで spec §2.1〜§2.3 の登録前の値を差 0 で出し直した（check_judge 66/66）。
- 走りながらの検査:
  - S-chain（止める条件）: 3,000 連鎖すべて通った（外れ 0）。
  - S-entropy（追補 1 §13.1 で記録だけ）: 旧規則で外れた連鎖は A4 442/1000、A3 205/1000、S2 6/1000。系列ごとと真の分散の大きい方の標準誤差では、A4 0、A3 1、S2 0。外れた連鎖は除かず、引き直していない（一覧は judge.md）。
- A2 の byakko での再生（§6.4、290 課題、10 seed とも完走）: online のビット一致 0/290、online の差の最大は seed ごとに 0.033〜0.084（課題平均の差の中央値 0.0035〜0.0057）、終わりの状態の重みの相対差の最大 0.86〜1.28。この値は「元の軌道の値」ではなく「byakko での再生」の値である。
- net100（記録）の A2 の値: (i) 登録前の参照（white-san の保存状態 u100 から、`prereg/a2_j4_fallback.txt` sha256 5eb9d37e…）L0 1.66 [1.37, 1.78]、(ii) byakko での再生 L0 1.66 [1.35, 1.78]。A4 は 0.481 [0.339, 0.567]、S2 は 1.60 [0.986, 1.89]、A3 は judge.md。

## 4. spec からずれた所と理由

1. 台が white-san と違う（RTX 5070 Ti）。S-recorder と S-replay を §12.3（Issa の決定）のとおり読み替えた。
2. 追補 1（a078e2b4、検査の結果を見てから・本走の前、Issa の判断）: S-entropy を止める条件から記録に替えた。S-J6 を、判定に使う量の完全一致と既存の数値の検証で通した。§7 の判定・閾値・予測は変えていない。
3. S-J6 の unit ごとの確認は未了。loadsign の unit の配列が byakko に無いため。J6 のラベルに印を付けた。
4. A2 の checkpoint は、研究データのサーバからは落とさず、chargpt_seed10_1022 で byakko に受け取った同じ束（tar の sha256 5729ec02…）を使った。manifest で 1,459 ファイルを照合し、不一致 0。
5. 束に更新 100 の状態（u100_*）が無いので、net100 の登録前の参照値は byakko では出し直していない（byakko での再生の値を並べた）。
6. 生データの置き場は §10.1 の `~/Projects/obsidian-research-data/` ではなく、byakko の作業ツリーの `results/chargpt_detlabel_1023/` の下（gitignore。§12.4）。
7. J6 の L0 の記録（§6.5・§7.6、判定に使わない）は出していない。第二・第三段階の L0 版は、block 1 を通る経路の扱いが spec の文から決まらないため。`rotation_report.py` は写したが回していない（写す一覧に無い `rotation_measure.py` の出力を読むため）。
8. §7.7 の追補 6 の条件 3 は出していない（別の解析の出力 decomp_dmu.csv が要るため）。
9. GPU の並べ方は、π を先に計算し（16 プロセス）、走は GPU 1 本ずつにした（4 本並べても処理量が増えないことを測った）。provenance に書いた。
10. push の認証がこのマシンに無いので、commit は Kubo 先生の手で push するか、git bundle で渡す。

## 5. 掛かった時間

- GPU（RTX 5070 Ti、1 本ずつ）: 本走 13.56 時間（S2 4.57・A4 4.49・A3 4.50、1 単位 約 27 分）、A2 の byakko での再生 1.39 時間（290 課題）、検査の試走は別に数十分。
- CPU: π の先計算 2,000 連鎖（α 1e−4・1e−3）が壁時計 2.17 時間 × 16 プロセス（α 0.01 の 1,000 連鎖は 25 秒）。J6 は 3 腕で壁時計 17 分（8 threads）。

## 6. A3（α 1e−3、記録）

| 判定 | A3 |
| --- | --- |
| J1 | LOSS_GONE（0/10。相ごとの喪失 0 / 7 / 0、ΔE 0.0070 [−0.0009, 0.0117]） |
| J2 | L0_GROWTH_SIMILAR（ρ2 1.01、g 15.3 [14.1, 19.8]） |
| J3 | L0_DEATH_REDUCED（中央値 0.017、d ≥ 0.5 は 0/10） |
| J4 | MOTION_SIMILAR（ρ4 0.67） |
| J5a / J5b | NO_EARLY_DEATH（0/10）/ INSUFFICIENT |
| J6 / J6-LN | OTHER_TERM_FLIPS（0/10）/ LN_HALVES（10/10）（S-J6 の unit ごとの確認待ち） |

## 置き場

- commit: ブランチ `claude/chargpt_detlabel_1023`（登録 db44d165、実装と検査 8f7ec958、追補 1 a078e2b4、結果と判定はその後の commit）。
- 生データ: byakko の `/home/kubo/project/Nakatsuka/lop_detlabel/results/chargpt_detlabel_1023/` の下（7,259 ファイル・10.31 GB。checkpoint・π・J6 の unit の配列など）。一覧は `backup_manifest.json`（path・bytes・sha256）。受け渡しは Issa と決める。
