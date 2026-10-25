"""Post-hoc unit-level magnitudes; cancellation must not masquerade as small T5.

These are components of the fixed-metric expected next raw-mu update, never
causal shares of finite Adam or of standardized position. No new GPU work.
"""
import csv
import json
from pathlib import Path
import numpy as np
import common as T
from t5_launch import completed


def divide(a, b):
    return float(a / b) if b > 0 else None


def describe(total, mean_field, covariance, half0, half1, gate_variance):
    arrays = (total, mean_field, covariance, half0, half1, gate_variance)
    assert all(x.shape == total.shape and np.isfinite(x).all() for x in arrays)
    assert total.ndim == 1
    identity_error = float(np.abs(total - mean_field - covariance).max())
    assert identity_error < 1e-11
    # Global magnitudes retain every unit and have no sign/size exclusion.
    abs_c, abs_m = np.abs(covariance), np.abs(mean_field)
    mass_c, mass_m = abs_c.sum(), abs_m.sum()
    down_c, down_m = np.maximum(-covariance, 0).sum(), np.maximum(-mean_field, 0).sum()
    # A bounded per-unit composition avoids divisions by a cancelled total.
    mass = abs_c + abs_m
    threshold = max(1e-12, 1e-6 * float(mass.max()))
    active = mass > threshold
    composition = np.divide(abs_c, mass, out=np.zeros_like(mass), where=mass > 0)
    tol_c = max(1e-14, 1e-6 * float(abs_c.max()))
    tol_m = max(1e-12, 1e-6 * float(abs_m.max()))
    signed = (abs_c > tol_c) & (abs_m > tol_m)
    half_tol = max(1e-14, 1e-6 * max(float(np.abs(half0).max()), float(np.abs(half1).max())))
    half_valid = (np.abs(half0) > half_tol) & (np.abs(half1) > half_tol)
    sign_agree = (half0[half_valid] < 0) == (half1[half_valid] < 0)
    row = dict(units=len(total), identity_error=identity_error,
        total_mean=float(total.mean()), mean_field_mean=float(mean_field.mean()), covariance_mean=float(covariance.mean()),
        covariance_abs_sum=float(mass_c), mean_field_abs_sum=float(mass_m),
        covariance_abs_composition=divide(mass_c, mass_c + mass_m),
        covariance_down_composition=divide(down_c, down_c + down_m),
        covariance_net_over_abs=divide(abs(covariance.sum()), mass_c),
        covariance_negative_units=int((covariance < -tol_c).sum()),
        covariance_positive_units=int((covariance > tol_c).sum()),
        covariance_zero_units=int((abs_c <= tol_c).sum()), covariance_sign_tolerance=tol_c,
        paired_sign_units=int(signed.sum()),
        covariance_mean_field_opposite= float(((covariance[signed] < 0) != (mean_field[signed] < 0)).mean()) if signed.any() else None,
        covariance_mean_field_both_down=int(((covariance < -tol_c) & (mean_field < -tol_m)).sum()),
        any_replica_gate_variation_units=int((gate_variance > 0).sum()),
        active_units=int(active.sum()), active_threshold=threshold,
        per_unit_covariance_composition_median=float(np.median(composition[active])) if active.any() else None,
        per_unit_covariance_composition_p90=float(np.quantile(composition[active], .9)) if active.any() else None,
        per_unit_covariance_composition_p99=float(np.quantile(composition[active], .99)) if active.any() else None,
        per_unit_covariance_over_10pct=int(((composition > .1) & active).sum()),
        per_unit_covariance_over_50pct=int(((composition > .5) & active).sum()),
        half_sign_units=int(half_valid.sum()),
        half_sign_agreement=float(sign_agree.mean()) if len(sign_agree) else None)
    return row


def self_check():
    # Equal upward/downward covariance has zero mean but a large absolute mass.
    c, m = np.array([-1., 1., 0.]), np.array([-1., 1., 0.])
    r = describe(c + m, m, c, c, c, np.array([.1, .1, 0.]))
    assert r['covariance_mean'] == 0 and r['covariance_abs_composition'] == .5
    assert r['covariance_down_composition'] == .5 and r['covariance_net_over_abs'] == 0
    assert r['active_units'] == 2 and r['covariance_zero_units'] == 1
    # Opposing terms can cancel the total without corrupting the composition.
    r = describe(np.zeros(3), -c, c, c, c, np.array([.1, .1, 0.]))
    assert r['covariance_abs_composition'] == .5 and r['covariance_mean_field_opposite'] == 1
    z = np.zeros(3)
    r = describe(z, z, z, z, z, z)
    assert r['covariance_abs_composition'] is None and r['active_units'] == 0
    return True


def main():
    assert self_check()
    rows, consumed = [], []
    for marker in sorted((T.ARCHIVE / 't5').glob('*/complete.json')):
        root = marker.parent
        assert completed(root), root
        cfg = json.loads(marker.read_text())['config']
        with np.load(root / 'measurements.npz') as d:
            for step in cfg['points'][1:]:
                fields = [d[f's{step}_{name}'] for name in ('total', 'mean_field', 'gate_covariance', 'covariance_half0', 'covariance_half1', 'gate_variance')]
                rows.append(dict(**{k: cfg[k] for k in ('seed', 'task', 'start', 'layer', 'mode')}, step=step,
                    directory=str(root), **describe(*fields)))
        consumed.append(dict(path=str(root / 'measurements.npz'), sha256=T.sha256(root / 'measurements.npz')))
    output = T.ROOT / 'results/vit_two_block_1025'
    if rows:
        with (output / 't5_units.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)
    T.atomic_json(output / 't5_units_inputs.json', dict(source_sha256=T.sha256(Path(__file__)),
        self_check_passed=True, completed_conditions=len(consumed), inputs=consumed))
    text = ['# T5対応項をunit別の大きさで確認', '',
        f'完了済み{len(consumed)}/80条件。初期のunit平均結果を見た後の探索的集計。新しい学習条件は追加していない。', '',
        'C_iをゲート共分散項、M_iを平均場項とする。両者は固定計量での「次の期待Δμ」の加法成分。実Adam条件でも計算する台帳は固定計量のもの。',
        '絶対量の構成比 = Σ|C_i| / Σ(|C_i|+|M_i|)。下向き成分の構成比 = Σ(-C_i)₊ / Σ[(-C_i)₊+(-M_i)₊]。',
        'いずれも有限更新の因果割合やΔ(μ/σ)の割合ではない。絶対量の比は、unit間や成分間で符号が相殺しても小さくならない。',
        'unitごとの比も|C_i|/(|C_i|+|M_i|)を使う。母数は|C_i|+|M_i| > max(1e−12, 1e−6×最大値)のunit。閾値・分母・全時点はCSVに保存した。', '',
        '以下は複製50更新後。P99はunit別構成比の99パーセンタイル。符号の判定はmax(1e−14, 1e−6×max|C|)を使う。', '',
        '| seed | 起点 | block | mode | 絶対量の比 | 下向きの比 | unit P99 | unit比>10% | C下向き/上向き/ほぼ0 |',
        '|---:|---:|---:|---|---:|---:|---:|---:|---|']
    def percent(x):
        return '—' if x is None else f'{100*x:.4g}%'
    for r in rows:
        if r['step'] == 50:
            text.append(f"| {r['seed']} (課題{r['task']}) | {r['start']} | {r['layer']} | {r['mode']} | "
                f"{percent(r['covariance_abs_composition'])} | {percent(r['covariance_down_composition'])} | "
                f"{percent(r['per_unit_covariance_composition_p99'])} | {r['per_unit_covariance_over_10pct']}/{r['active_units']} | "
                f"{r['covariance_negative_units']}/{r['covariance_positive_units']}/{r['covariance_zero_units']} |")
    text += ['', '半分ずつの複製での符号一致もCSVに記録したが、これは分割への安定性であり有意差検定ではない。',
             'gate_varianceの対象は全token。CLSだけでの閉口率・開口率とは異なる。', '']
    (output / 't5_units_partial.md').write_text('\n'.join(text))
    print(json.dumps(dict(completed_conditions=len(consumed), self_check_passed=True, output=str(output / 't5_units_partial.md'))))


if __name__ == '__main__':
    main()
