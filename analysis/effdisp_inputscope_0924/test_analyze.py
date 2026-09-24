"""Small synthetic checks of time alignment and registered verdict separation."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from analysis.effdisp_inputscope_0924.analyze import (
    actual_terms, expected_series, low_response_onset, paired_terms,
    paired_comparisons, summarize_series, group_verdicts, group_statistics,
    summarize_window, preflight, Series, _sum_terms)


def test_expected_series_and_windows():
    cases=expected_series(__import__("pathlib").Path("/tmp/raw"))
    assert len(cases)==130
    assert sum(s.environment=="conda" for s in cases)==80
    assert sum(s.environment=="mnist" for s in cases)==50
    assert sum(s.tasks==150 for s in cases)==10


def test_preflight_missing_status_with_existing_metadata(tmp_path):
    arm=tmp_path/"conda"/"LR_k1_X0Y0"
    arm.mkdir(parents=True)
    (tmp_path/"conda"/"metadata.json").write_text("{}")
    s=Series("conda","LR_k1_X0Y0","LR","X0Y0",200,400,arm/"seed200")
    row=preflight(s)
    assert row["status"]!="COMPLETE"
    assert row["source_git_hash"]==""


def test_current_covariance_and_paired_mean_are_exact():
    xp=np.array([[0.,0.],[0.,1.],[0.,2.],[0.,3.]])
    xc=np.array([[1.,0.],[1.,2.],[1.,1.],[1.,3.]])
    w0=np.array([[2.,.5],[-1.,1.]])
    w1=np.array([[3.,1.],[-.5,2.]])
    b0=np.array([.1,-.2]); b1=np.array([.4,.1])
    actual=actual_terms(w0,w1,xp,xc)
    pair=paired_terms(w0,w1,b0,b1,xp,xc)
    np.testing.assert_allclose(actual["Q"],pair["Q_current"],atol=1e-14)
    np.testing.assert_allclose(actual["identity_residual"],0,atol=1e-14)
    np.testing.assert_allclose(pair["A_identity_residual"],0,atol=1e-14)
    np.testing.assert_allclose(pair["total_identity_residual"],0,atol=1e-14)
    # Input mean changes while centered covariance changes independently.
    assert np.any(pair["E_energy"]>0)
    np.testing.assert_allclose(pair["A_energy"],pair["Q_current"]+pair["mean_A_sq"])
    assert not np.allclose(pair["E_energy"],actual["G_pre"])


def test_no_update_can_still_have_input_switch_response():
    xp=np.array([[0.,0.],[0.,1.]])
    xc=np.array([[1.,0.],[1.,1.]])
    w=np.array([[4.,2.]])
    pair=paired_terms(w,w,np.zeros(1),np.zeros(1),xp,xc)
    actual=actual_terms(w,w,xp,xc)
    np.testing.assert_allclose(pair["A_energy"],0)
    np.testing.assert_allclose(actual["Q"],0)
    np.testing.assert_allclose(pair["E_energy"],16)


def test_actual_c_and_rho_use_current_covariance():
    xp=np.array([[-.1,0.],[.1,0.]])
    xc=np.array([[-1.,0.],[1.,0.]])
    w0=np.array([[1.,0.]])
    w1=np.array([[0.,0.]])
    unit=actual_terms(w0,w1,xp,xc)
    summed=_sum_terms(unit)
    np.testing.assert_allclose(summed["V"],.01)
    np.testing.assert_allclose(summed["Vprev_current"],1.)
    np.testing.assert_allclose(summed["c"],-1.)
    np.testing.assert_allclose(summed["rho"],1.)
    assert abs(summed["c"])<=1


def test_mixed_frame_uses_environment_specific_response_columns():
    rows=[]
    for env in ("conda","mnist"):
        for task in range(1,6):
            rows.append({"environment":env,"task":task,
                         "derivative_absmean":1.0 if env=="conda" else math.nan,
                         "activeunit_frac":1.0 if env=="conda" else math.nan,
                         "derivative_abs_mean_l1":0. if env=="mnist" and task>=2 else 1.,
                         "derivative_abs_mean_l2":1.,"active_unit_frac_l1":1.,
                         "active_unit_frac_l2":1.})
    mixed=pd.DataFrame(rows)
    assert low_response_onset(mixed[mixed.environment=="conda"],"conda") is None
    assert low_response_onset(mixed[mixed.environment=="mnist"],"mnist")==2


def test_reference_closure_heldout_qx_does_not_refit():
    task=np.arange(1,51)
    v=np.full(50,10.,dtype=float)
    table=pd.DataFrame({"task":task,"V":v,"V_next":v,"Q":np.ones(50),
                        "X":np.full(50,-.5),"covariance_term":np.zeros(50),
                        "derivative_absmean":np.ones(50),"activeunit_frac":np.ones(50)})
    original,_=summarize_window(table,environment="conda",metric="reference",
                                window="primary50",endpoint=50)
    changed=table.copy()
    changed.loc[changed.task>25,"Q"]=100.
    changed.loc[changed.task>25,"X"]=-50.
    second,_=summarize_window(changed,environment="conda",metric="reference",
                              window="primary50",endpoint=50)
    assert original["qbar"]==second["qbar"]
    assert original["gamma"]==second["gamma"]
    assert original["heldout_mape"]==second["heldout_mape"]


def test_actual_balance_kept_separate_from_fixed_p():
    task=np.arange(1,51)
    # Current-bank update grows V by 1, compensated by G_pre=-1 each task.
    v=10+np.arange(50,dtype=float)*0
    actual=pd.DataFrame({"task":task,"V":v,"V_next":v,"Vprev_current":v-1,
                         "Q":np.ones(50),"X":np.zeros(50),"G_pre":-np.ones(50),
                         "covariance_term":-np.ones(50),"derivative_absmean":np.ones(50),
                         "activeunit_frac":np.ones(50)})
    fit,_=summarize_window(actual,environment="conda",metric="actual",
                           window="primary50",endpoint=50)
    assert fit["A_BALANCE"]=="PASS"
    assert fit["A_PLATEAU"]=="PASS"
    assert fit["P1"]=="NOT_APPLICABLE"
    assert fit["late_update_only_balance"]==0


def test_small_end_to_end_summary_and_partial_pairing(tmp_path):
    rows=[]
    for task in range(1,51):
        for metric in ("reference","actual","raw"):
            rows.append({"environment":"mnist","arm":"LR_X0Y1","activation":"LR",
                         "cell":"X0Y1","seed":200,"task":task,"metric":metric,
                         "V":10.,"V_next":10.,"Vprev_current":10.,"Q":1.,"X":-.5,
                         "A_energy":1.,"E_energy":0.,"AE_cross":0.,
                         "total_energy":1.,"mean_A_sq":0.,"mean_E_sq":0.,
                         "G_pre":0.,"covariance_term":0.,"source_prev":"prev",
                         "source_next":"next","reference_input_source":"bank",
                         "input_current_source":"task_input","derivative_abs_mean_l1":1.,
                         "derivative_abs_mean_l2":1.,"active_unit_frac_l1":1.,
                         "active_unit_frac_l2":1.})
    transition=pd.DataFrame(rows)
    summary,closure,sources=summarize_series(transition)
    assert len(summary)==3
    assert len(closure)==25
    assert len(sources)==3
    assert summary.loc[summary.metric.eq("reference"),"P1"].iloc[0]=="PASS"
    assert summary.loc[summary.metric.eq("actual"),"A_BALANCE"].iloc[0]=="PASS"
    verdict=group_verdicts(summary)
    assert verdict.verdict.eq("INCOMPLETE").all()
    stats=group_statistics(summary)
    assert not stats.empty and stats.n_seed.le(1).all()
    pair,group=paired_comparisons(summary)
    assert pair.status.eq("INCOMPLETE").any()
    assert group.n_seed.le(1).all()
    from analysis.effdisp_inputscope_0924.plots import make_plots
    files=make_plots(transition,summary,closure,group,tmp_path)
    assert files and all(file.is_file() for file in files)
