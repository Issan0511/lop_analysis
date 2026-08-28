# ratchet_log_0829c（無効・seed 衝突）

天井T0b spec 初版 §8.2 のコマンド

    OMP_NUM_THREADS=1 python -m src.ratchet_log --config configs/ratchet_log_0819.yaml \
        --seeds 10 11 12 13 14 15 16 17 18 19 --outdir results/ratchet_log_0829c

で 2026-08-28 に生成した走。**新しい seed 群になっていない。**

`src/train.py` の `make_gens` は乱数生成器を `SEED_BASE[exp] + width +
generator_offset` からのみ作り、config の seed 値は `run_id` と保存列にしか入らない。
その結果、本走の seed10..19 は `results/ratchet_log_0819/` の seed0..9 と
`cos_u_mu`・`w_norm`・`p_hat`・`F_gate`・`F_self`・`F_rest`・`flip_state`・`mu_norm`
まで bit 一致した。

`stream_fingerprints.csv` に 10 対すべての一致を記録してある（軌道列のみの sha256。
`seed` と `run_id` は除く）。**`logs/*.npz` 自体は commit していない。**中身は
`results/ratchet_log_0819/logs/seed{0..9}.npz` と同一であり、83MB の重複になるため。
再現するには上のコマンドをそのまま実行し、fingerprint を照合すればよい。

正しい腕C は `configs/ratchet_log_0829c.yaml`（`generator_offset: 20260831`）で
生成した `results/ratchet_log_0829c/`。経緯は spec §12 の 4 行目。
