# sink_roots_0930 R7 — GPU への移し替え（再起動の後、2026-09-30 13:1x、走る前に記録）

R7（spec_postfit_elu_cifar_0924 を CPU で回す、spec_sink_roots_0930 §4）は、GPU が使えなかった（ドライバ 580.173 とライブラリ 580.178 の不一致）ため CPU・1 seed 1 プロセスで回していた。Issa の判断で再起動し GPU が使えるようになったので、登録元の 0924 spec の本来の形に戻す。

- 形: `src/postfit_elu_cifar_0924.py --device cuda`、R = 10（seed 0〜9 を 1 本で同時）、CUDA graph、`--no-keep-ckpts`（登録の解析は snapshot だけ読む）。置き場 `obsidian-research-data/sink_roots_0930/r7gpu/<arm>/`
- 順: K1 の前半（A の課題 1〜2）→ S の η の試走（0924 §2 の規則）→ A・F・F99 の本走 → η_S が決まれば S
- K1 の前半は済み: A の課題 1〜2 の online_acc・memo_acc が S5 ref（cap_cifar_ee_0920 の ref 腕、GPU）と、CSV に印字された 10 桁まで全 (seed, 課題) で一致（最大差 4.8e−11、書式の差だけ）。1 更新 0.83 ms（10 seed 同時）
- CPU の走（A 10/10 済み、F 3/10 済み・7/10 途中）は S5 ref と課題 1 から一致しない（CPU と GPU の数値の差、課題 1 の online_acc で 5e−4〜7e−4）。止めたまま、別機材の再現の記録として残す（CPU の F の続きは回さない）
- 登録の解析（analyze_r7.py の Q1〜Q6）は GPU の走に当てる（置き場の読み方だけ変える）
