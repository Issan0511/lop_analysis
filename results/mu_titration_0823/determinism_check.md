# Determinism manifest

- Result: **PASS**
- Implementation / analysis commit: `dcabc43a75323370b529c38dca767395df6e7bc3`
- Replay output: `/tmp/mu-titration-analysis-rerun-EBqB36/derived/`
- Method: the canonical analysis was independently re-run from the same clean commit and the same raw inputs, then every relative output path was compared byte-for-byte with `cmp`.
- Compared outputs: 16 files (8 CSV, `analysis_meta.json`, `summary.md`, this generated manifest, and 5 PNG figures); mismatches: **0**.
- Raw-input audit: all 132 entries in `raw_sha256.csv` independently matched both recorded byte length and SHA-256.

All scientific outputs omit timestamps and elapsed time. The hashes below are the canonical output hashes from the verified run.

| file | sha256 |
|---|---|
| arm_manifest.csv | `4f46a16fb0c110829d82383a5f1a2ad3f770cd391a38bb97e96f9963dcb199c1` |
| raw_sha256.csv | `514d090f0c7972d4c5612c5a1efc2cf79c88f4dbca79892e105d02bd39824728` |
| gate_curve.csv | `8bee4bb2321f9fc6951acd9ce47dd22e6a34d0f5b4dbb8bd3c17ddd97b9d4fda` |
| theta_estimates.csv | `080855868f2e8b69f264128dfb972c1d0b39db54f9b24d6443e6372f867e5cb5` |
| dose_response.csv | `eaf168eba3972681a189ef809b092d46451c99b76b6687c7db248fd8e98cbed8` |
| path_decomposition.csv | `3626d7fbc2e5b5ed26d32964c83cbcbac51568d9da5b2e64953fdb164793bd02` |
| per_seed_metrics.csv | `ca82c74a6fedd3d4fd045dc2a72f79f46f177af26412ab985ad0e2008883cf5f` |
| verdict.csv | `1d1c8bd126ac58a23c40a5e901497cdcb3398efc8a576bfd82dd8b47833796c7` |
| summary.md | `7827eed55704da99f532f3274fe8852e49e2452ed918858789c4dc6672f7b67a` |
| analysis_meta.json | `1c83a5f7cc81503237eef1c78b50c0bee38c357d3ea0dfc109da05de330f4bf2` |
| figures/fig_bias_escape.png | `efa9b6469e5eafaf3e405c27fa80c62dabdf2a2ef5001e748ffcc26198f739c3` |
| figures/fig_final_phenotype.png | `404128079b21077ec6c32510e59b5eebf52ea0125a472204bf63f1dfa90017dc` |
| figures/fig_gate_curves_bulk.png | `8e7cbb9123b947fb99fe655f9c3948b3e6722e3220ffdbc2c245409a5ddfbb96` |
| figures/fig_theta_dose.png | `665a4934ebad1353a91eafb925a880c27b46ddd19f11649252f45ad4ab9cf669` |
| figures/fig_wall_decomposition.png | `a27e6be5b9b66d4dd26c745c4afb7ada6f28f19cca5312bae86e5d3b7358cc9c` |
