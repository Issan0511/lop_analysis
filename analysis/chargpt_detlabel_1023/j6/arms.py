#!/usr/bin/env python3
"""J6 の腕の表（spec_chargpt_detlabel_1023 §6.5 の変更 4: 腕を模型の腕とデータの α の組で持つ）。

torch を読まない（report.py からも読む）。腕は環境変数 J6_ARM で選び、出力先は J6_OUT で替えられる
（既定は results/chargpt_detlabel_1023/j6/<腕>/）。
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]          # $DL
CG = Path('/home/kubo/project/Nakatsuka/lop_analysis_chargpt_cl_1019')

# label: (模型の腕, データの α, 走の置き場（seed<k>/ckpts/task<t>.pt と seed<k>/pi.npz の親）)
ARM_TABLE = {
    'A4': dict(model_arm='C3', alpha=1e-4, run_dir=ROOT/'results/chargpt_detlabel_1023/M2/A4'),
    'A3': dict(model_arm='C3', alpha=1e-3, run_dir=ROOT/'results/chargpt_detlabel_1023/M2/A3'),
    'S2': dict(model_arm='C3', alpha=0.01, run_dir=ROOT/'results/chargpt_detlabel_1023/M2/S2'),
    # 対照（S-J6 の再現の検査に使う。既存の C3-2、走らせ直さない）
    'A2': dict(model_arm='C3', alpha=0.01, run_dir=CG/'results/chargpt_cl_1019/M2/C3'),
}


def current():
    label = os.environ.get('J6_ARM')
    assert label in ARM_TABLE, f'set J6_ARM to one of {list(ARM_TABLE)} (got {label!r})'
    out = Path(os.environ['J6_OUT']) if os.environ.get('J6_OUT') else ROOT/'results/chargpt_detlabel_1023/j6'/label
    return label, ARM_TABLE[label], out
