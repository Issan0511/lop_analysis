"""Static figures for the preregistered input-scope analysis CSVs."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _grid(n: int, cols: int = 4):
    cols=min(cols,max(1,n))
    rows=math.ceil(n/cols)
    fig,axes=plt.subplots(rows,cols,figsize=(4.1*cols,3.2*rows),squeeze=False,
                          layout="constrained")
    axes=axes.ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig,axes


def growth_all_arms(transitions: pd.DataFrame, out: Path) -> list[Path]:
    """Separate scales for measured-covariance width and raw weight norm."""
    paths=[]
    for env,block in transitions.groupby("environment"):
        arms=sorted(block.arm.unique())
        for view in ("effective","raw"):
            fig,axes=_grid(len(arms))
            for ax,arm in zip(axes,arms):
                part=block[block.arm==arm]
                metrics=(("reference","#2367ad"),("actual","#ce6d17")) if view=="effective" else (("raw","#555555"),)
                for metric,color in metrics:
                    by_task=part[part.metric==metric].groupby("task").V_next
                    if not by_task.ngroups:
                        continue
                    median=by_task.median()
                    lower=by_task.min().reindex(median.index)
                    upper=by_task.max().reindex(median.index)
                    ax.fill_between(median.index.to_numpy(dtype=float),lower.to_numpy(dtype=float),
                                    upper.to_numpy(dtype=float),color=color,alpha=.12,lw=0)
                    ax.plot(median.index,median.to_numpy(),color=color,lw=1.0,label=metric)
                ax.set_title(arm,fontsize=9)
                ax.set_xlabel("endpoint task")
                ax.set_ylabel((r"$V_\Sigma=\mathrm{tr}(W_1\Sigma W_1^\top)$"
                               if view=="effective" else r"$R=\|W_1\|_F^2$"))
                ax.grid(alpha=.2)
            axes[0].legend(fontsize=7)
            fig.suptitle(f"{env}: {view} first-layer width, seed median and range (all arms)")
            path=out/(f"growth_all_{env}.png" if view=="effective" else f"growth_raw_{env}.png")
            fig.savefig(path,dpi=145)
            plt.close(fig)
            paths.append(path)
    return paths


def balance_all_arms(summary: pd.DataFrame, out: Path) -> list[Path]:
    paths=[]
    for env,block in summary[summary.metric.isin(["reference","actual"])].groupby("environment"):
        arms=sorted(block.arm.unique())
        fig,axes=_grid(len(arms))
        for ax,arm in zip(axes,arms):
            part=block[(block.arm==arm)&(block.window.str.startswith(("primary","long")))]
            for metric,stat,color in (("reference","late_nearbalance","#2367ad"),
                                      ("actual","late_A_balance","#ce6d17")):
                sub=part[part.metric==metric]
                for window,win in sub.groupby("window"):
                    marker="s" if window=="long150" else "o"
                    ax.scatter(win.late_activity if metric=="reference" else win.late_A_activity,
                               win[stat],s=20,color=color,marker=marker,alpha=.8,
                               label=f"{metric}/{window}")
            ax.axhspan(.9,1.1,color="#999999",alpha=.15)
            ax.axvline(.1,color="#555555",ls=":",lw=.7)
            ax.set_title(arm,fontsize=9)
            ax.set_xlabel("sum Q / mean V")
            ax.set_ylabel("balance")
            finite=part[["late_nearbalance","late_A_balance"]].to_numpy(dtype=float)
            if np.isfinite(finite).any() and np.nanmax(np.abs(finite))>10:
                ax.set_ylim(-2,2)
                n=int(np.count_nonzero(np.isfinite(finite)&(np.abs(finite)>2)))
                ax.text(.02,.98,f"{n} off-scale; CSV has exact values",transform=ax.transAxes,
                        va="top",fontsize=6)
            ax.grid(alpha=.2)
            if part.window.nunique()>1 or ax is axes[0]:
                ax.legend(fontsize=6,ncol=2)
        fig.suptitle(f"{env}: update-only reference and current-bank actual balance")
        path=out/f"balance_all_{env}.png"
        fig.savefig(path,dpi=145)
        plt.close(fig)
        paths.append(path)
    return paths


def paired_energy_all_arms(summary: pd.DataFrame, out: Path) -> list[Path]:
    paths=[]
    fields=("late_sum_A_energy","late_sum_E_energy","late_sum_AE_cross")
    if not set(fields).issubset(summary.columns):
        return paths
    for env,block in summary[(summary.metric=="actual")&
                             (summary.window.isin(["primary400","primary50","long150"]))].groupby("environment"):
        arms=sorted(block.arm.unique())
        fig,axes=_grid(len(arms))
        for ax,arm in zip(axes,arms):
            part=block[block.arm==arm]
            for window,win in part.groupby("window"):
                med=np.array([win[field].median() for field in fields],dtype=float)
                if np.isfinite(med).any():
                    ax.plot(range(3),med,marker="o",lw=1,label=window)
            ax.axhline(0,color="black",lw=.5)
            ax.set_xticks(range(3),["A update","E input","2<A,E>"],rotation=35,ha="right")
            ax.set_title(arm,fontsize=9)
            ax.set_ylabel("sum per-unit energy")
            ax.grid(alpha=.2)
            if len(part.window.unique())>1:
                ax.legend(fontsize=6)
        fig.suptitle(f"{env}: same-ID preactivation change, all arms")
        path=out/f"paired_energy_all_{env}.png"
        fig.savefig(path,dpi=145)
        plt.close(fig)
        paths.append(path)
    return paths


def closure_all_arms(closure: pd.DataFrame, out: Path) -> list[Path]:
    paths=[]
    if closure.empty:
        return paths
    for (env,window),block in closure.groupby(["environment","window"]):
        arms=sorted(block.arm.unique())
        fig,axes=_grid(len(arms))
        for ax,arm in zip(axes,arms):
            part=block[block.arm==arm]
            for column,color,label in (("observed_V","#222222","observed"),
                                       ("model_V","#2367ad","closure"),
                                       ("constant_V","#999999","constant")):
                med=part.groupby("task")[column].median()
                ax.plot(med.index,med.to_numpy(),lw=1,color=color,label=label)
            ax.set_title(arm,fontsize=9)
            ax.set_xlabel("held-out endpoint task")
            ax.set_ylabel("fixed-reference V")
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=7)
        fig.suptitle(f"{env}/{window}: calibration-only closure, all arms")
        path=out/f"closure_all_{env}_{window}.png"
        fig.savefig(path,dpi=145)
        plt.close(fig)
        paths.append(path)
    return paths


def factorial_contrasts(paired_group: pd.DataFrame, out: Path) -> list[Path]:
    paths=[]
    if paired_group.empty:
        return paths
    selected=paired_group[(paired_group.metric=="reference")&
                          (paired_group.field.isin(["late_nearbalance","late_activity","late_loglog_slope"]))&
                          (paired_group.contrast=="difference")&
                          (paired_group.design.isin(["input_X1_minus_X0_given_Y",
                                                          "target_Y1_minus_Y0_given_X"]))]
    for env,block in selected.groupby("environment"):
        fig,axes=_grid(3,cols=3)
        for ax,field in zip(axes,("late_nearbalance","late_activity","late_loglog_slope")):
            part=block[(block.field==field)&(block.window.isin(["primary400","primary50"]))]
            part=part[(part.n_seed==5)&np.isfinite(part.seed_median)&
                      np.isfinite(part.ci95_low)&np.isfinite(part.ci95_high)]
            part=part.sort_values(["design","reference_arm","comparison_arm"])
            y=np.arange(len(part))
            ax.errorbar(part.seed_median,y,
                        xerr=np.vstack((part.seed_median-part.ci95_low,
                                        part.ci95_high-part.seed_median)),
                        fmt="o",markersize=3,lw=.7,capsize=2)
            ax.set_yticks(y,[f"{r.design}: {r.comparison_arm}" for r in part.itertuples()],fontsize=6)
            ax.axvline(0,color="black",lw=.6)
            ax.set_title(field)
            ax.grid(alpha=.2)
        fig.suptitle(f"{env}: paired seed factorial differences (bootstrap95% CI)")
        path=out/f"factorial_{env}.png"
        fig.savefig(path,dpi=145)
        plt.close(fig)
        paths.append(path)
    return paths


def make_plots(transitions: pd.DataFrame, summary: pd.DataFrame, closure: pd.DataFrame,
               paired_group: pd.DataFrame, out: Path) -> list[Path]:
    out=Path(out)
    return (growth_all_arms(transitions,out)+balance_all_arms(summary,out)+
            paired_energy_all_arms(summary,out)+closure_all_arms(closure,out)+
            factorial_contrasts(paired_group,out))
