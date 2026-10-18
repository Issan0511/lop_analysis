"""Descriptive, post hoc width/position trajectories requested by Issa."""
import csv
from pathlib import Path
import numpy as np
from analysis.cap_act_box_fresh_followup_1018.run import M, OUT, put, read, sha, rows

def main():
    records, inputs = [], {}
    configs = [('F51','ELU','ref'), ('F51','ELU','cap12'),
               ('F51','ReLU','ref'), ('RC','ELU','ref')]
    for box, act, arm in configs:
        base = M.OUT_ROOT/'evaluation'/box/act/arm
        manifest = read(base/'output_manifest.json')
        assert sha(base/'output_manifest.json') == read(base/'status.json')['output_manifest_sha256']
        for task in range(1,31 if box=='F51' else 51):
            p = base/f'units_t{task:02d}.npz'
            digest = sha(p)
            assert digest == manifest['files'][p.name]['sha256']
            inputs[str(p)] = digest
            with np.load(p) as z:
                for layer in (1,2):
                    for seed in range(10):
                        record = dict(box=box, activation=act, arm=arm, task=task,
                                      layer=layer, seed=seed)
                        for key in ('s','m','r','Wnorm','dead','native_dphi_mean','native_dphi_var'):
                            value=z[f'{key}_l{layer}'][seed]
                            record[key+'_unit_mean']=float(np.nanmean(value))
                        records.append(record)
    OUT.mkdir(parents=True,exist_ok=True)
    M.H.write_csv(OUT/'width_per_seed.csv',records)
    selected=[]
    for box, act, arm in configs:
        for task in ([1,3,11,21,29] if box=='F51' else [1,10,30,50]):
            for layer in (1,2):
                rr=[r for r in records if (r['box'],r['activation'],r['arm'],r['task'],r['layer']) == (box,act,arm,task,layer)]
                selected.append(dict(box=box,activation=act,arm=arm,task=task,layer=layer,
                    **{k:float(np.mean([r[k] for r in rr])) for k in rr[0] if k.endswith('_unit_mean')}))
    put(OUT/'width_summary.json',dict(kind='descriptive post hoc; no causal identification',
        statistic='equal-weight mean of 100 units within each of 10 seeds, then mean over seeds',
        source_sha256=inputs,selected=selected))
    lines=['# Width and position in the original cap experiment','',
        'Post hoc descriptive readout requested by Issa. Width s is the population standard deviation of preactivation over the task images, calculated separately per unit. These are means over units and seeds; r is mean(m_i/s_i), not mean(m_i)/mean(s_i).', '',
        '| Box | Activation | Arm | Task | Layer | Width s | Position m/s | Mean m | Dead fraction |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in selected:
        lines.append(f"| {r['box']} | {r['activation']} | {r['arm']} | {r['task']} | {r['layer']} | {r['s_unit_mean']:.3f} | {r['r_unit_mean']:.3f} | {r['m_unit_mean']:.3f} | {r['dead_unit_mean']:.3f} |")
    lines += ['', 'F51 ELU width does grow (roughly 1.8x/1.7x from hard task 1 to 29), but much less than RC ELU (14x/39x from task 1 to 50), and is nearly flat from F51 task 21 to 29. F51 ELU also stays much shallower in relative position. This is consistent with the absence of large loss in F51 ELU, but does not identify width alone as its cause. RC and F51 differ in task distribution, learning rate, update budget and endpoint. The evaluated image classes also change across F51 tasks.', '',
        'F51 ReLU has comparable late width to F51 ELU, yet has a more negative second-layer relative position and substantial death. Width alone therefore does not describe the whole contrast.']
    (OUT/'width_summary.md').write_text('\n'.join(lines)+'\n')
    print('Saved descriptive width trajectories and verified original input hashes.')

if __name__=='__main__':main()
