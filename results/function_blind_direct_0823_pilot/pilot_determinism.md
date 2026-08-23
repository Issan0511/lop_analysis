# Pilot集計の決定性確認

- 実装 commit: `a975d6a`
- 集計コマンドを同じNPZ入力に対して2回連続実行し、直下CSV 5本の sha256 が全一致した
- 実行: `OMP_NUM_THREADS=1 .venv/bin/python -m analysis.function_blind_direct.pilot --logs results/function_blind_direct_0823_pilot/logs --outdir results/function_blind_direct_0823_pilot --bootstrap-n 10000`

| file | sha256 |
|---|---|
| `exposures.csv` | `937ff2f5cf1d3ebce6af7e5127d192abbb49f1a0d532d7fd2b61c0f3e8aef1ca` |
| `pilot_diagnostics.csv` | `6bc9d8cbdee2d3ff43f648829229db3b38733cd1ca6d224337ca93521eb0afa4` |
| `pilot_rates.csv` | `5ccfdb53b80120b4c1ce94f248cd7486facea5391cdf669e8bfce6f2a2c18640` |
| `pilot_sanity.csv` | `ad9f8074e48d7b08edf62410c215c999f23cdd454ca0b2c715917c0a32a3e6c9` |
| `pilot_verdict_candidates.csv` | `693791438f0218a047946bc1039b8f8b128b6e5c264ef18c72e515b7fb9c92f8` |

これは集計決定性の確認であり、pilotの効果量をconfirmation結果へ昇格させるものではない。
