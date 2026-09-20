"""Descriptive figure after the registered primary report; no inferential changes."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src import drive_cifar_c_0920 as D
root=D.ROOT/'results/drive_cifar_c_0920/report'
rows=list(csv.DictReader((root/'transport_by_window.csv').open()))
lookup={(int(r['seed']),r['window']):r for r in rows if r['task']=='all_tasks2to5'}
seeds=np.arange(5);width=.36
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for offset,key,label,color in [(-width/2,'cert_down_frequency','Down certificate','#397bb7'),(width/2,'cert_up_frequency','Up certificate','#da8b32')]:
 axes[0].bar(seeds+offset,[float(lookup[s,'full'][key])*100 for s in seeds],width,label=label,color=color)
axes[0].set(title='All updates: tasks 2–5',ylabel='Certified unit-updates (%)')
for offset,win,label,color in [(-width/2,'boundary','Boundary: updates 1–75','#397bb7'),(width/2,'later','Later: updates 76–30,000','#7698b5')]:
 axes[1].bar(seeds+offset,[float(lookup[s,win]['cert_down_frequency'])*100 for s in seeds],width,label=label,color=color)
axes[1].set(title='Down certificates by registered window',ylabel='Certified unit-updates (%)')
for offset,key,label,color in [(-width/2,'S_sum','Self movement S','#397bb7'),(width/2,'U_sum','Upstream movement U','#da8b32')]:
 axes[2].bar(seeds+offset,[float(lookup[s,'full'][key])/100 for s in seeds],width,label=label,color=color)
axes[2].set(title='Net transport, summed over tasks 2–5',ylabel='Mean per unit (sum of updates)')
for ax in axes:ax.set_xticks(seeds);ax.set_xlabel('Primary seed');ax.legend(fontsize=8);ax.axhline(0,color='black',lw=.7)
fig.savefig(root/'drive_summary.png',dpi=160);fig.savefig(root/'drive_summary.pdf');plt.close(fig)
