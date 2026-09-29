"""Static figures from the bounded, completed two-group calculation."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('results/two_group_logistic_0929')
COLORS={.1:'#2467a3',.3:'#31a18a',1.:'#ca892d',2.:'#bc4b52'}


def read(name):
    with (OUT/name).open(newline='') as f:return list(csv.DictReader(f))


def main():
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    rows=[r for r in read('flow_endpoints.csv') if r['mode']=='saturated']
    fig,ax=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for i,r in enumerate(rows):
        a=float(r['gain']);b=float(r['b']);h=float(r['h'])
        ax[0].plot([b-2,b+2],[i,i],color=COLORS[a],linewidth=6,solid_capstyle='round')
        ax[0].scatter([b],[i],color='white',s=22,zorder=4)
        ax[0].scatter([h],[i],color=COLORS[a],marker='^',s=75,zorder=4)
        ax[0].annotate(f'{h:.2f}',(h,i),xytext=(0,9),textcoords='offset points',ha='center',color=COLORS[a])
    ax[0].set_yticks(range(4),[f'v={r["gain"]}' for r in rows])
    ax[0].set_ylim(-.4,3.6)
    ax[0].set_xlabel('Preactivation z')
    ax[0].axvline(0,color='.6',lw=.8)
    ax[0].set_title('A  The bulk keeps its width\nThe rare open input moves',fontweight='bold')
    a=np.geomspace(.07,2.6,240)
    h=np.log(16)/a-1;b=-12+(h-1)/17;gap=h-b
    tail=gap/np.sqrt(.98*2+.02*.98*gap**2)
    initial=13/np.sqrt(.98*2+.02*.98*13**2)
    ax[1].plot(a,tail,color='#2467a3',lw=2,label='Solved gradient-flow equilibrium')
    ax[1].axhline(1/np.sqrt(.02*.98),color='.45',ls='--',label='Zero-width bulk: constant shape')
    ax[1].axhline(initial,color='#b579a5',ls=':',label='Uniform rescaling: constant shape')
    for r in rows:
        ax[1].scatter(float(r['gain']),float(r['T']),color=COLORS[float(r['gain'])],s=35,zorder=5)
    ax[1].set_xscale('log');ax[1].set_xlabel('Readout v');ax[1].set_ylabel('T = (maximum - median) / SD')
    ax[1].set_xticks([.1,.3,1,2],['0.1','0.3','1','2']);ax[1].minorticks_off()
    ax[1].set_title('B  The sign follows from conservation\nNo noise or Adam is required',fontweight='bold')
    ax[1].legend(loc='upper right',bbox_to_anchor=(1,.85),fontsize=8,frameon=True,facecolor='white',framealpha=.95)
    trace=read('flow_trace.csv')
    for gain,color in COLORS.items():
        rr=[r for r in trace if r['mode']=='saturated' and float(r['gain'])==gain]
        ax[2].plot([float(r['time']) for r in rr],[float(r['h']) for r in rr],color=color,label=f'v={gain:g}')
    ax[2].set_xscale('symlog',linthresh=1)
    ax[2].set_xlabel('Gradient-flow time')
    ax[2].set_ylabel('Rare-input height h')
    ax[2].set_title('C  Finite-time ordering is different\nSmall v also relaxes more slowly',fontweight='bold')
    ax[2].legend(frameon=False,fontsize=9)
    for aax in ax:
        aax.spines[['top','right']].set_visible(False);aax.grid(alpha=.15)
    fig.suptitle('A shared-neuron logistic mechanism: fit changes, stored bulk variation remains',fontsize=15,fontweight='bold')
    fig.savefig(OUT/'mechanism.png',dpi=180,bbox_inches='tight')
    fig.savefig(OUT/'mechanism.pdf',bbox_inches='tight');plt.close(fig)

    if (OUT/'exact_elu_endpoints.csv').exists():
        exact=read('exact_elu_endpoints.csv');curve=read('exact_elu_curve.csv')
        fig,axes=plt.subplots(1,2,figsize=(10.5,4.3),layout='constrained')
        palette=['#2467a3','#31a18a','#ca892d','#bc4b52']
        for i,r in enumerate(exact):
            b=float(r['b']);h=float(r['h']);color=palette[i]
            axes[0].plot([b-.5,b+.5],[i,i],color=color,lw=6,solid_capstyle='round')
            axes[0].scatter(b,i,c='white',s=20,zorder=4)
            axes[0].scatter(h,i,c=color,marker='^',s=70,zorder=4)
            axes[0].annotate(f'h={h:g}',(h,i),xytext=(0,9),textcoords='offset points',ha='center',color=color)
            axes[1].scatter(float(r['gain']),float(r['T']),c=color,s=40,zorder=5)
        axes[0].set_yticks(range(4),[f'v={r["gain"]}' for r in exact])
        axes[0].set_ylim(-.4,3.6);axes[0].set_xlabel('Preactivation z')
        axes[0].axvline(0,color='.6',lw=.8)
        axes[0].set_title('Negative bulk shifts; its width stays fixed\nThe positive height follows a different law',fontweight='bold')
        axes[1].plot([float(r['gain']) for r in curve],[float(r['T']) for r in curve],color='#2467a3',lw=2)
        axes[1].set_xscale('log');axes[1].set_xlabel('Readout v');axes[1].set_ylabel('T = (maximum - median) / SD')
        axes[1].set_xticks([.2,.5,1,2],['0.2','0.5','1','2']);axes[1].minorticks_off()
        axes[1].set_title('An exact sign at the selected optimum\ndT / d log(v) < 0 while the upper input is open',fontweight='bold')
        for aax in axes:
            aax.spines[['top','right']].set_visible(False);aax.grid(alpha=.18)
        fig.suptitle('True ELU + logistic loss: an exact shared-neuron solution\nAll four target probabilities are identical across v',fontweight='bold',fontsize=14)
        fig.savefig(OUT/'exact_mechanism.png',dpi=180,bbox_inches='tight')
        fig.savefig(OUT/'exact_mechanism.pdf',bbox_inches='tight');plt.close(fig)

    path=OUT/'adam_endpoints.csv'
    if path.exists():
        rows=read(path.name)
        fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
        for col,b0 in enumerate([-12.,-24.]):
            for activation in ['saturated','exact_elu']:
                for eps in [0.,1e-8]:
                    rr=[r for r in rows if r['activation']==activation and float(r['b0'])==b0 and float(r['epsilon'])==eps]
                    rr.sort(key=lambda r:float(r['gain']))
                    label=f'{activation}, eps={eps:g}'
                    color='#2467a3' if activation=='saturated' else '#c66a36'
                    style='-' if eps==0 else '--'
                    for row,key in enumerate(['body_sd','T']):
                        axes[row,col].plot([float(r['gain']) for r in rr],[float(r[key]) for r in rr],
                            marker='o',color=color,ls=style,label=label)
            axes[0,col].set_title(f'Initial bulk center b = {b0:g}',fontweight='bold')
            axes[1,col].set_xlabel('Readout v')
            for aax in axes[:,col]:
                aax.set_xscale('log');aax.grid(alpha=.2);aax.spines[['top','right']].set_visible(False)
                aax.set_xticks([.1,.3,1,2],['0.1','0.3','1','2']);aax.minorticks_off()
        axes[0,0].set_ylabel('Bulk SD after 20,000 Adam updates')
        axes[1,0].set_ylabel('Centered tail T')
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
        fig.suptitle('The saturation assumption can fail under Adam\nSmall raw gradients alone do not preserve the bulk',fontweight='bold')
        fig.savefig(OUT/'adam_limits.png',dpi=180,bbox_inches='tight')
        fig.savefig(OUT/'adam_limits.pdf',bbox_inches='tight');plt.close(fig)


if __name__=='__main__':main()
