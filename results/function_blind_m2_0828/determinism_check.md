# determinism check

- result: **PASS**
- method: 同じ replay S / exposure / RNG から解析全体を2回再構成
- bootstrap: B=10000, seed=20260901

| CSV | SHA-256 |
|---|---|
| joined_exposures.csv | `af8a27475a54faa839fd169cbe56f565e08b8f2a4a7b01a94595ebf628d737cd` |
| group_distributions.csv | `119d719fec12c5ac0305d1148f4ee7f286814394dce01b17ec5c4f1782c72742` |
| per_seed_distributions.csv | `55d2b7e469c5aa0b719ee435064b918ae4bdb37278c0a854fb47456460ca4f9f` |
| cell_effects.csv | `0b5a279439a0f2dfe695b8687cdd077c83bf5ae8fc3320b04b0454e7b4204cfe` |
| estimates.csv | `31606257a4fb65766919c0ccc2983382a64902a7863f3b86b0298a964b2cd846` |
| bootstrap.csv | `ddb0268225d604abdf2e4afbfaee7885353d3520caf88c59cf35ca3f86db5222` |
| sanity.csv | `ade53ab150fbfadf79fe5f4fa6a12cb8fef29e4f7c86da97d55bbed05df2503c` |
| verdict.csv | `071095f6f8fdb2eea5d332a4d3ace1c21addcd5f8236bceadf1dcd3c75578909` |
