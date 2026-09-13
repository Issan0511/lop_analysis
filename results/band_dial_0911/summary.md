# band_dial_0911 summary

spec: `specs/spec_band_dial_0911.md`（事前登録 commit `1388d10`・判定値は未読で起動）

## 0. G1 / G2

| arm | g1 maxabs (s0/s1/s2) | cnorm | φ′ vs autograd | ELUF continuity |
|---|---|---|---|---|
| CELU03 | None / None / None | None / None / None | 1.1102230246251565e-16 | None |
| CELU1 | 1.7763568394002505e-15 / 1.7763568394002505e-15 / 1.4210854715202004e-14 | 0.0 / 0.0 / 0.0 | 1.1102230246251565e-16 | None |
| CELU3 | None / None / None | None / None / None | 2.220446049250313e-16 | None |
| ELUF | None / None / None | None / None / None | 1.1102230246251565e-16 | 2.000000165480742e-10 |

## 1. 腕ごとの量（seed 別 L と late 窓の中央値）

| arm | L s0/s1/s2 [pt] | L med | Cov | Cov θ=0.5 | NL_x | NL_u | Ḡ | N | z̄ | σ | p⁺ | dead |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CELU03 | 3.00 / 1.96 / 2.68 | 2.68 | 0.782 | 0.793 | 0.0687 | 0.0888 | 0.212 | 7.00 | -4.99 | 2.97 | 0.197 | 14 |
| CELU1 | 1.77 / 1.86 / 2.02 | 1.86 | 0.712 | 0.763 | 0.0798 | 0.0761 | 0.263 | 7.97 | -6.60 | 4.36 | 0.194 | 0 |
| CELU3 | 0.58 / 0.92 / 0.65 | 0.65 | 0.282 | 0.493 | 0.0704 | 0.0589 | 0.558 | 8.06 | -2.38 | 2.51 | 0.285 | 0 |
| ELUF | 1.06 / 0.87 / 1.53 | 1.06 | 0.666 | 0.734 | 0.0879 | 0.0489 | 0.347 | 8.06 | -3.08 | 2.37 | 0.209 | 0 |

- **D0**: Cov(CELU3) < Cov(CELU1) < Cov(CELU03) in 3/3 seeds: True → `MANIPULATION_OK`
- **D1**: `WIDER_BAND_LESS_LOSS`、効果量 |L(CELU3) − L(CELU03)| 中央値 = 2.02 pt → `BAND_STRONG`
- **D3**: N の CELU1 に対する相対差 {"CELU03": -0.137, "CELU3": 0.012, "ELUF": 0.007} → `WIDTH_MOVES_TOO`
- **D2**: ΔL(ELUF − CELU1) = [-0.71, -0.99, -0.49] pt → `FLOOR_NEUTRAL`
