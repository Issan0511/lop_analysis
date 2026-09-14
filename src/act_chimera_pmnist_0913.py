"""act_chimera_pmnist_0913: 箱 B（Permuted MNIST）のランナー（spec_act_chimera_0913 §2.2）。

プロトコルは elu_growth_0909 / gate_shape_0911 と同一（784→100→100→10・batch 16 × 625・
手書き Adam 1e−3・CPU 1 thread・deterministic）。学習ループは ``GS.train``
（`src/gate_shape_0911.py:184-230`）を下の ``train`` へ**逐語で写し**、登録ブロック P1–P6
だけを足した（§2.2 の表・S14 が AST で照合する）。既存モジュール（H・GS・C・EG・T）は
import するだけで 1 byte も変えない。

  P1  挿入  GS:189 の直後   null 双子の摂動（``twin`` が None でないとき b1[k] に +1e−6・追補 2-2。旧 W1[k, 0] の 1 ulp）
  P2  置換  GS:190          ``act = make_act(arm)`` → ``act = act_obj``（make_act は act_chimera_0913 の側）
  P3  挿入  GS:191 の直後   読み出しの状態の初期化（W̃・層 2 の前回値・init の sha256）
  P4  挿入  GS:199 の直後   タスク開始時の読み出し（層 1・2 の zstart_i と帯分率・§3.4）
  P5  置換  GS:210          ``.001`` → ``lr``（lr = 0.001 では同じ double。G1-PM がそれを証明する）
  P6  挿入  GS:216 の直後   タスク終端の読み出し（§3.3–3.4・E 恒等式・params の sha256）

GS.run は写さない（自前の ``run``）。GS.run の検査の扱いは §2.2 のとおり: TIME_CAP は置かない、
G2（check_dphi）と ELUF の連続性は S1/S3（test_act_chimera_0913）に置き換え、G3（gate_block の
分散の恒等式 ≤ 1e−9 と seed 0 の対照）はそのまま残す。``T.finite_guard`` は try/except で囲み、
落ちたら部分行と ``{ARM}_s{seed}_divergence.json`` を書いて ``DIVERGED`` にする（S20 が検査する）。

腕（段 1・17 本）: 8 関数 + 分離腕 GN/FN/GD/FD + LR02 + null 双子 LRtw0–3（§2.1・§3.2）。
腕ラベルは ``AC.twin_of`` で (活性化名, 双子番号) に分け、活性化は ``AC.make_act``（未知名は
KeyError）で作る。EG.make_act / EG.hdefect / EG.measure は使わない（S8）。

出力（§8）: ``{ARM}_s{seed}_rows.csv`` / ``_units.npz``（GS と同じ 16 キー・将来の bit 錨）/
``_readout.npz``（凍結スキーマ）/ ``_provenance.json`` / ``_divergence.json``（DIVERGED のみ）。

起動: /usr/bin/python3 -m src.act_chimera_pmnist_0913 --arm SMINH --seed 0 [--tasks 120 --outdir ...]
      /usr/bin/python3 -m src.act_chimera_pmnist_0913 --smoke   （17 腕 × seed 0–2 × t1–3・§6）
"""
from pathlib import Path
import argparse, ast, hashlib, json, os, platform, resource, subprocess, sys, time
import numpy as np
import torch
from src import gate_shape_0911 as GS
from src import width_sink_clamp_0909 as C
from src import elu_growth_0909 as EG
from src import transport_common_0910 as T
from src import act_chimera_0913 as AC

H = GS.H
ROOT = GS.ROOT
RUN_ID = 'act_chimera_0913'
OUT = ROOT / 'results/act_chimera_0913/pmnist'
SMOKE = ROOT / 'results/_smoke_act_chimera_0913/pmnist'
SPEC = ROOT / 'specs/spec_act_chimera_0913.md'
TASKS = 120
LR = 0.001                                  # §2.1: 主走は全腕でプロトコルの lr（段 2 で 5e−4 / 2e−3）
E_TASKS = (20, 40, 60, 80, 100, 120)        # EG:15 E_TASKS（欠損項の恒等式を計るタスク）
NGRAD = 4096                                # EG:15（訓練 4096 枚）
REF_SEED = 20260909                         # EG:15/121（manual_seed(REF_SEED + seed)・学習系列と独立）
N_PROBE = 512
EXPECTED_ENV = dict(executable='/usr/bin/python3', torch='2.13.0+cu130', numpy='2.5.2')   # §2.1
SMOKE_TASKS = 3
SMOKE_E_TASKS = (1, 2, 3)                   # スモークでは E 恒等式を t1–3 で計る（主走は E_TASKS）

# 段 1 の 17 腕（§2.1）。ラベル → (活性化名, 双子番号) は AC.twin_of。
ARMS_STAGE1 = AC.FUNCS8 + AC.SPLIT_NAMES + ('LR02',) + tuple(f'LRtw{k}' for k in range(4))
# G1 の錨（§2.2）: gate_shape_0911 の committed 行・units（SMAXH ≡ ELUF）。副次の錨は elu_growth_0909。
ANCHOR = {'LR': 'LR', 'ELU1': 'ELU1', 'SMAXH': 'ELUF'}
ANCHOR_EG = ('LR', 'ELU1')
# gate_shape_0911 の units の 16 キー（C.measure の 6 + gate_block の 10）
UNIT_KEYS16 = T.UNIT_KEYS + ('gbar_i', 'gvar_i', 'off_i', 'offabs_i', 'off10_i', 'off50_i',
                             'q10_i', 'zcur_i', 'sdcur_i', 'hard_i')
# §3.4: 誤った h（変異対照 E_pred_wrong）は腕ごとに固定する（EG:66 のクラス分岐はキメラに意味を持たない）
WRONG_H = {**{a: 'ELU1' for a in ('LR', 'LR02', 'VMAX', 'SMINH', 'SMINS', 'GN', 'FN')},
           **{a: 'LR' for a in ('ELU1', 'SMAXH', 'VMIN', 'SMAXS', 'GD', 'FD')}}
LEAKY_ZERO = ('LR', 'LR02')                 # E ≈ 0 を assert する腕（EG:139-141・双子は LR）
G3_TOL = 1e-9                               # GS:315
EG_ANCHOR_TOL = 1e-10                       # GS:331（累積順の違いで 1e−15 の丸め）
IDENTITY_TOL, IDENTITY_CTL = 1e-6, 1e-4     # EG:137-138
READOUT_BUDGET = 4 * 1024 * 1024            # §8: 箱 B の readout ≤ 4 MB/走（非圧縮）
EPS64 = 2. ** -52

gate_block = GS.gate_block                  # 写した本体が宿主と同じ名前で呼べるように束縛する
gcorr = GS.gcorr


# ------------------------------------------------------------------ 環境（§2.1・G0）
def env_info():
    return dict(executable=sys.executable, torch=torch.__version__, numpy=np.__version__,
                hostname=platform.node(), cpu=_cpu_model(), gpu=_gpu_name(), device='cpu',
                python=platform.python_version())


def assert_env():
    got = env_info()
    bad = {k: (got[k], v) for k, v in EXPECTED_ENV.items() if got[k] != v}
    assert not bad, ('interpreter / torch / numpy differ from the reference (§2.1)', bad)
    return got


def _cpu_model():
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return None


def _nvidia_smi(args):
    try:
        return subprocess.run(['nvidia-smi'] + args, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _gpu_name():
    return _nvidia_smi(['--query-gpu=name', '--format=csv,noheader'])


def process_snapshot():
    """他セッションの学習プロセス（§7）: python を含むコマンド行の PID・RSS と nvidia-smi の compute プロセス。"""
    me, procs = os.getpid(), []
    for d in Path('/proc').iterdir():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            cmd = (d / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
            if 'python' not in cmd:
                continue
            rss = [l for l in (d / 'status').read_text().splitlines() if l.startswith('VmRSS')]
        except OSError:
            continue
        procs.append(dict(pid=int(d.name), rss_kib=int(rss[0].split()[1]) if rss else None, cmd=cmd[:300]))
    mem = {}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            k, v = line.split(':', 1)
            if k in ('MemAvailable', 'SwapFree'):
                mem[k] = int(v.split()[0])
    except OSError:
        pass
    return dict(python_processes=procs, nvidia_smi_compute=_nvidia_smi(['--query-compute-apps=pid,used_memory',
                                                                         '--format=csv,noheader']),
                load1=os.getloadavg()[0], meminfo_kib=mem, time=time.strftime('%Y-%m-%dT%H:%M:%S'))


def _git(*a):
    try:
        return subprocess.check_output(['git'] + list(a), cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# §6: 本走は push した 1 つの commit から回す。この 3 つが commit に入っていない走は git_hash がコードを指さない。
COMMITTED_FOR_MAIN = ('src/act_chimera_0913.py', 'src/act_chimera_pmnist_0913.py', 'specs/spec_act_chimera_0913.md')


def git_info():
    """git_hash と汚れ。``git_dirty_tracked`` は追跡ファイルの変更、``git_dirty_src`` は src/ configs/ specs/ の
    未追跡も含む（``--untracked-files=all``。新しいモジュールが commit 前だと git_hash がコードを指さないのを見る）。"""
    dirty = _git('status', '--porcelain', '--untracked-files=no')
    dirty_src = _git('status', '--porcelain', '--untracked-files=all', '--', 'src', 'configs', 'specs')
    return dict(git_hash=_git('rev-parse', 'HEAD'), git_branch=_git('rev-parse', '--abbrev-ref', 'HEAD'),
                git_dirty_tracked=(None if dirty is None else dirty.splitlines()),
                git_dirty_src=(None if dirty_src is None else
                               [l for l in dirty_src.splitlines() if '__pycache__' not in l]))


def require_committed(paths=COMMITTED_FOR_MAIN):
    """登録の地平線の走（本走）の前: paths が追跡されていて、作業木で変更も未追跡もないこと（§6）。"""
    bad = {}
    for p in paths:
        if _git('ls-files', '--error-unmatch', p) is None:
            bad[p] = 'untracked'
            continue
        st = _git('status', '--porcelain', '--', p)
        if st:
            bad[p] = st
    assert not bad, ('main run from uncommitted code: commit and push first (§6)', bad)
    return dict(committed=list(paths), pass_=True)


# ------------------------------------------------------------------ 双子（P1・S18・S18b・追補 2-2）
TWIN_KS = tuple(AC.twin_of(a)[1] for a in ARMS_STAGE1 if AC.twin_of(a)[1] is not None)   # (0, 1, 2, 3)


def twin_report(seed, k, loop_init_sha=None, delta=AC.TWIN_DELTA, perturb=None):
    """S18（追補 2-2）: 双子の init が LR と b1[k] の 1 要素だけ違い、その値が float32 で b1[k] + 1e−6 に最も近い表現値で
    あること（``AC.twin_init_report``。判定はつねに登録値 1e−6 と比べる）。``delta`` と ``perturb(params, k)`` は変異対照
    用: delta = 0 は「差 0 要素」で、delta = 2e−6 は値で、旧来の W1[k, 0] の nextafter は位置で落ちる。
    ``loop_init_sha`` を渡すと、ここで作った摂動済みの init の sha256 が、学習ループの中で P1 の後に記録した
    init（``Readout.rng_sha['init']``）と一致することも pass に要る（走そのものの init を検査する）。"""
    base = H.init_params(seed, torch.device('cpu'))
    tw = H.init_params(seed, torch.device('cpu'))
    if perturb is None:
        AC.perturb_twin_b1(tw, k, delta)
    else:
        perturb(tw, k)
    rep = AC.twin_init_report(base, tw, k)
    sha = _sha_params(tw)
    tied = None if loop_init_sha is None else bool(sha == loop_init_sha)
    rep.update(perturbed_init_sha=sha, loop_init_sha=loop_init_sha, tied_to_run=tied,
               pass_=bool(rep['pass_'] and tied is not False))
    return rep


# ------------------------------------------------------------------ 乱数の指紋（S9）
def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _sha_tensor(t):
    return _sha_bytes(t.detach().contiguous().numpy().tobytes())


def _sha_params(p):
    return _sha_bytes(b''.join(q.detach().contiguous().numpy().tobytes() for q in p))


def rng_fingerprint(seed, tasks=3, mnist=None):
    """ループの外で同じ系列から同じ順に引いた perm・層化 idx・バッチ順の sha256（t1..tasks）と init・プローブ idx。
    ループの中の記録（Readout.rng_sha）と一致すれば、ループはこれ以外に系列を消費していない（S9）。"""
    gp, gd, gb = H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)
    out = dict(init=_sha_params(H.init_params(seed, torch.device('cpu'))))
    if mnist is not None:
        pidx = torch.randperm(len(mnist.test_x), generator=H.stream('boundary_probe', seed))[:N_PROBE]
        out['probe_idx'] = _sha_tensor(pidx)
    for task in range(1, tasks + 1):
        perm = torch.randperm(784, generator=gp)
        idx = H.stratified_draw(mnist, gd) if mnist is not None else None
        order = torch.randperm(H.TASK_EXAMPLES, generator=gb)
        out[f't{task}'] = dict(perm=_sha_tensor(perm), idx=(None if idx is None else _sha_tensor(idx)),
                               order=_sha_tensor(order))
    return out


# ------------------------------------------------------------------ 読み出し（P3/P4/P6 の呼び先・§3.3–3.4）
class Readout:
    """新しい読み出し（§8 の凍結スキーマ）。学習には触れない（S10）。すべて no_grad・float64、保存時に §8 の dtype。

    層 l ∈ {1, 2}（配列の軸 1）、U = 100、T = tasks。
    - zbar_start/occ_start: タスク開始時（P4）、プローブ 512 枚・現置換。zbar_end/occ_end/pabs1/mob_band/fn_mom:
      タスク終端（P6）。帯の述語はすべて float64 に写した z の上で評価し、ゲートは float32 の z の上の自腕の
      ``dphi``（分離腕は φ′_bwd・gate_block と同じ z）。fn_mom は 8 関数を自腕の状態の上で評価する（§1.5）。
    - depth_vel: 層 1 は v_i = S̄·Δm_i + Δb_i（C:218 の恒等式・置換不変の深さ成分）、層 2 は Δ(W2_i·μ_φ1 + b2_i)
      （μ_φ1 = 終端のプローブ平均の層 1 出力で固定）。Δ は前タスク終端（t = 1 は初期値）からの差。
    - step_num/wt_norm: 層 1 は中心化行 W̃ = W − 行平均、層 2 は W̃2 = W2 − (W2·μ̂)μ̂（μ̂ = μ_φ1/‖μ_φ1‖・終端）。
      mu_comp = W2·μ̂（層 2 の「深さ」）。
    - adam_ratio/adam_epsfrac: 重み行ごとの mean |m̂|/(√v̂ + ε) と √v̂ < 10ε の要素の割合（タスク終端）。
    - E_ident: t ∈ e_tasks で E = r_W1b1 − r_W2、E_pred = Σ(∂L/∂a₁)h、E_pred_wrong（§3.4 の WRONG_H）。
      式は EG:56-68 の写しで h だけ AC.h の表（EG.measure は呼ばない）。
    - param_sha: タスク終端の params の sha256（キメラが親から離れる最初のタスク・§3.4）。
    - sha_excl[k]: タスク終端の params の sha256 を、b1[k] を 0 にした写しで計算した列（S18b・追補 2-2。LR は k = 0..3、
      双子 k は自分の k だけ。provenance の ``checks.s18b.sha_excl_b1`` に書き、判定器が ``twin_live_task`` を出す）。
    mut='add1e-9' は S10 の変異対照（測定の後に W1 += 1e−9）、inject_inf_after=t は S20 の注入
    （task t の読み出しの後に W1[0, 0] = inf）。どちらも主走では None。
    """

    def __init__(self, act_name, tasks, xg, yg, e_tasks=E_TASKS, mut=None, inject_inf_after=None, sha_excl_ks=()):
        self.act_name, self.T = act_name, int(tasks)
        self.sha_excl = {int(k): [] for k in sha_excl_ks}
        self.xg, self.yg = xg, yg
        self.e_tasks = tuple(int(t) for t in e_tasks)
        self.mut, self.inject = mut, inject_inf_after
        T_, U, nE = self.T, H.DIMS[1], len(self.e_tasks)
        f64 = lambda *s: np.full(s, np.nan, np.float64)
        f32 = lambda *s: np.full(s, np.nan, np.float32)
        self.a = dict(fn_mom=f64(T_, 2, 8, 4, 3), occ_start=f32(T_, 2, U, 4), occ_end=f32(T_, 2, U, 4),
                      mob_band=f32(T_, 2, U, 4), pabs1=f32(T_, 2, U), zbar_start=f64(T_, 2, U),
                      zbar_end=f64(T_, 2, U), depth_vel=f64(T_, 2, U), step_num=f64(T_, 2, U),
                      wt_norm=f64(T_, 2, U), mu_comp=f64(T_, U), adam_ratio=f32(T_, 2, U),
                      adam_epsfrac=f32(T_, 2, U), gbar_l2=f64(T_, U), mu_phi1=f64(T_, U),
                      E_ident=f64(nE, 3), param_sha=np.full(T_, '', dtype='<U64'))
        self.E_scale = f64(nE)
        self.rng_sha = {}
        self.prev = None
        # 走の中の恒等式（S12 を実状態で・C.measure との整合）
        self.ck = dict(s12_mob_sum_maxdiff=0., s12_band_sum_maxdiff=0., s12_band_sum_maxratio=0.,
                       s12_ctl_dropB2_maxdiff=0., s12_ctl_dropB2_demonstrable=False, wtnorm_vs_cnorm_maxdiff=0.,
                       mob_sum_vs_gbar_i_maxdiff=0.)
        self.n_inputs = None

    # ---- 状態
    @staticmethod
    def _state(p):
        W1 = p[0].detach().double()
        m1 = W1.mean(1)
        return dict(m1=m1, b1=p[1].detach().double().clone(), Wt1=W1 - m1[:, None],
                    W2=p[2].detach().double().clone(), b2=p[3].detach().double().clone())

    def begin(self, p, act):                                          # P3
        self.prev = self._state(p)
        self.rng_sha['init'] = _sha_params(p)

    def task_start(self, p, act, probe, perm, idx, order, task):     # P4
        t = task - 1
        with torch.no_grad():
            z1, _, z2, _, _ = H.forward([q.detach() for q in p], probe.px[:, perm], act)
            for l, z in ((0, z1), (1, z2)):
                zd = z.double()
                self.a['zbar_start'][t, l] = zd.mean(0).numpy()
                self.a['occ_start'][t, l] = AC.occupancy(zd).numpy()
        if task <= 3:
            self.rng_sha[f't{task}'] = dict(perm=_sha_tensor(perm), idx=_sha_tensor(idx), order=_sha_tensor(order))

    def task_end(self, p, act, adam, probe, perm, task, r, u, gu):    # P6
        t = task - 1
        self.a['param_sha'][t] = _sha_params(p)
        for k, lst in self.sha_excl.items():                          # S18b（追補 2-2）: b1[k] を除いた sha256
            lst.append(AC.params_sha256(p, exclude=(AC.TWIN_PARAM, (k,))))
        with torch.no_grad():
            pd = [q.detach() for q in p]
            z1, a1, z2, _, _ = H.forward(pd, probe.px[:, perm], act)
            for l, z in ((0, z1), (1, z2)):
                zd = z.double()
                g = act.dphi(z).double()                              # 自腕の φ′（分離腕は φ′_bwd）・float32 の z
                self.a['zbar_end'][t, l] = zd.mean(0).numpy()
                self.a['occ_end'][t, l] = AC.occupancy(zd).numpy()
                self.a['pabs1'][t, l] = AC.pabs1(zd).numpy()
                mb = AC.mob_band(g, zd)
                self.a['mob_band'][t, l] = mb.numpy()
                fm = AC.fn_mom(zd)
                self.a['fn_mom'][t, l] = fm.numpy()
                self._s12(fm, mb, g, zd)
                if l == 0:
                    # 層 1 の Σ_k mob_k は gate_block の gbar_i（同じ float32 の z・同じ dphi）と一致する
                    self._up('mob_sum_vs_gbar_i_maxdiff', (mb.sum(1) - torch.from_numpy(gu['gbar_i'])).abs().max())
                    # §3.4 の報告: EG の sat（φ′ < 0.05）と dead_units（EG:50-51 と同じ式・自腕の dphi = φ′_bwd・
                    # gate_block と同じ float32 の z）。腕ごとに意味が変わる（§1.2: SMINS では P[z < z_c]、SMINH・ELU1 では
                    # P[z < ln 0.05]・LR は常に 0）。rows.csv の列だけ（G1 は acc と units しか比べない）。
                    r['sat'] = float((g < EG.SAT).double().mean())
                    r['dead_units'] = int((g.max(0).values < EG.SAT).sum())
                else:
                    self.a['gbar_l2'][t] = g.mean(0).numpy()
            mu = a1.double().mean(0)
            self.a['mu_phi1'][t] = mu.numpy()
            cur, prev = self._state(pd), self.prev
            # 層 1（§3.4）
            self.a['depth_vel'][t, 0] = (probe.Sbar * (cur['m1'] - prev['m1']) + (cur['b1'] - prev['b1'])).numpy()
            self.a['step_num'][t, 0] = (cur['Wt1'] - prev['Wt1']).norm(dim=1).numpy()
            self.a['wt_norm'][t, 0] = cur['Wt1'].norm(dim=1).numpy()
            self._up('wtnorm_vs_cnorm_maxdiff', float(np.abs(self.a['wt_norm'][t, 0] - u['cnorm_i']).max()))
            # 層 2
            nrm = float(mu.norm())
            muh = mu / nrm if nrm > 0. else torch.zeros_like(mu)
            dW2 = cur['W2'] - prev['W2']
            proj = lambda W: W - torch.outer(W @ muh, muh)
            self.a['depth_vel'][t, 1] = (dW2 @ mu + (cur['b2'] - prev['b2'])).numpy()
            self.a['step_num'][t, 1] = proj(dW2).norm(dim=1).numpy()
            self.a['wt_norm'][t, 1] = proj(cur['W2']).norm(dim=1).numpy()
            self.a['mu_comp'][t] = (cur['W2'] @ muh).numpy()
            # Adam の状態（重み行ごと）
            m_, v_, tc = adam
            c1 = 1 - .9 ** tc[0]; c2 = 1 - .999 ** tc[0]
            for l, j in ((0, 0), (1, 2)):
                mh = m_[j].double() / c1
                vh = (v_[j].double() / c2).sqrt()
                self.a['adam_ratio'][t, l] = (mh.abs() / (vh + 1e-8)).mean(1).numpy()
                self.a['adam_epsfrac'][t, l] = (vh < 10 * 1e-8).double().mean(1).numpy()
            self.prev = cur
        if task in self.e_tasks:
            self._e_identity(p, act, perm, self.e_tasks.index(task))
        if self.mut == 'add1e-9':                                     # S10 (i) の変異対照
            with torch.no_grad():
                p[0].data.add_(1e-9)
        if self.inject is not None and task == int(self.inject):     # S20 の注入
            with torch.no_grad():
                p[0].data[0, 0] = float('inf')

    def _up(self, k, v):
        self.ck[k] = max(self.ck[k], float(v))

    def _s12(self, fm, mb, g, zd):
        """S12 を実状態で: Σ_k fn_mom = 帯なしの平均、Σ_k mob_k = mob。対照は B2 を落とした和（B2 の質量 > 0 のとき）。
        帯の和の許容値は n_inputs·eps64·max(1, max|v|)（§6 の n_inputs·eps64 は |v| ≤ 1 の量の値。φ = 0.1z や
        分離腕の h は深部で |v| > 1 になるので、同じ算術を値の大きさで伸ばす）。比を ``s12_band_sum_maxratio`` に残す。"""
        worst, ratio = 0., 0.
        self.n_inputs = int(zd.shape[0])
        for i, f in enumerate(AC.FUNCS8):
            for q, v in enumerate((AC.DPHI[f](zd), AC.PHI[f](zd), AC.h(f, zd))):
                d = abs(float(fm[i, :, q].sum()) - float(v.mean()))
                worst = max(worst, d)
                ratio = max(ratio, d / (zd.shape[0] * EPS64 * max(1., float(v.abs().max()))))
        self._up('s12_band_sum_maxdiff', worst)
        self._up('s12_band_sum_maxratio', ratio)
        self._up('s12_mob_sum_maxdiff', float((mb.sum(1) - g.mean(0)).abs().max()))
        mass2 = float((AC.band_index(zd) == 2).double().mean())
        if mass2 > 0.:
            ix = AC.FUNCS8.index('SMAXH')
            drop = abs(float(fm[ix, [0, 1, 3], 0].sum()) - float(AC.DPHI['SMAXH'](zd).mean()))
            self._up('s12_ctl_dropB2_maxdiff', drop)
            self.ck['s12_ctl_dropB2_demonstrable'] = True

    def _e_identity(self, p, act, perm, i):
        """EG:56-68 の写し。h だけ腕別の表（§1.2・§1.5）にし、誤った h は WRONG_H（§3.4）。
        EG:59 との差は W1/b1/W2 を detach してから double にすること（値は同じ・autograd の警告を出さない）。"""
        q = [t.detach().clone().requires_grad_(True) for t in p]
        o = H.forward(q, self.xg[:, perm], act)
        loss = torch.nn.functional.cross_entropy(o[4], self.yg)
        g = torch.autograd.grad(loss, q + [o[1]])
        W1, b1, W2 = q[0].detach().double(), q[1].detach().double(), q[2].detach().double()
        r1 = float((g[0].double() * W1).sum() + (g[1].double() * b1).sum())
        r2 = float((g[2].double() * W2).sum())
        z1 = o[0].detach().double()
        hh = AC.h(self.act_name, z1)
        hw = AC.h(WRONG_H[self.act_name], z1)
        scale = float(g[0].double().norm() * W1.norm() + g[1].double().norm() * b1.norm() + g[2].double().norm() * W2.norm())
        self.a['E_ident'][i] = (r1 - r2, float((g[6].double() * hh).sum()), float((g[6].double() * hw).sum()))
        self.E_scale[i] = scale

    def s12_in_run(self):
        """走の中の S12 の判定（§6 S12）: 帯の和・mob の和・gbar_i との一致がすべて許容値以内、対照（B2 を落とした和）は
        B2 に質量があった走でだけ実証でき、そのとき許容値を超えること。B2 が一度も埋まらなければ ctl_demonstrated =
        False と記録する（その走の S12 の対照は実証できない。pass は恒等式だけで決め、対照は pytest の合成状態の S12 が担う）。"""
        n = self.n_inputs or N_PROBE
        tol = n * EPS64
        c = self.ck
        ident = (c['s12_band_sum_maxratio'] <= 1. and c['s12_mob_sum_maxdiff'] <= tol
                 and c['mob_sum_vs_gbar_i_maxdiff'] <= tol)
        demo = bool(c['s12_ctl_dropB2_demonstrable'])
        ctl_fails = bool(c['s12_ctl_dropB2_maxdiff'] > tol) if demo else None
        return dict(tol=tol, tol_band='n_inputs*eps64*max(1,max|v|)', identities_pass=bool(ident),
                    ctl_demonstrated=demo, ctl_fails=ctl_fails, pass_=bool(ident and ctl_fails is not False))

    def arrays(self, n_tasks=None):
        """保存用（§8 のスキーマ）。n_tasks で切り詰める（DIVERGED の走）。"""
        n = self.T if n_tasks is None else int(n_tasks)
        out = {}
        for k, v in self.a.items():
            if k == 'E_ident':
                v = v.copy()
                v[[j for j, t in enumerate(self.e_tasks) if t > n]] = np.nan
                out[k] = v
            else:
                out[k] = v[:n]
        return out

    def identity_checks(self, n_tasks=None):
        """EG:135-141 と同じ比（走ごとの assert に使う）。評価できた e_tasks が無ければ evaluated=False。"""
        n = self.T if n_tasks is None else int(n_tasks)
        js = [j for j, t in enumerate(self.e_tasks) if t <= n and np.isfinite(self.E_scale[j])]
        if not js:
            return dict(evaluated=False, e_tasks=list(self.e_tasks))
        E = self.a['E_ident'][js]
        s = self.E_scale[js]
        out = dict(evaluated=True, e_tasks=[self.e_tasks[j] for j in js],
                   identity_rel=float(np.max(np.abs(E[:, 0] - E[:, 1]) / s)),
                   identity_mutctl=float(np.max(np.abs(E[:, 0] - E[:, 2]) / s)),
                   wrong_h=WRONG_H[self.act_name])
        if self.act_name in LEAKY_ZERO:
            out['leaky_zero'] = float(np.max(np.abs(E[:, 0]) / s))
        return out


# ------------------------------------------------------------------ 学習ループ（GS.train の逐語の写し + P1–P6）
def train(arm, seed, mnist, probe, tasks, mutate=None, rows=None, units=None, ck=None,
          act_obj=None, twin=None, lr=LR, readout=None):
    """GS.train（GS:184-230）の写し。差は登録ブロック P1–P6 だけ（S14）。署名は比較しない。"""
    p = H.init_params(seed, torch.device('cpu'))
    if mutate == 'init':
        with torch.no_grad():
            p[0].data[0, 0] += 1e-3
    # ↓ P1: null 双子の摂動（b1[k] に +1e−6・追補 2-2・S18）
    if twin is not None:
        AC.perturb_twin_b1(p, twin)
    act = act_obj                                   # P2: make_act(arm) → act_obj（AC.make_act は呼び手）
    adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])
    # ↓ P3: 読み出しの状態の初期化（W̃・層 2 の前回値・init の sha256）
    if readout is not None:
        readout.begin(p, act)
    gp, gd, gb = H.stream('perm', seed), H.stream('data', seed), H.stream('batch', seed)
    for task in range(1, tasks + 1):
        perm = torch.randperm(784, generator=gp)
        idx = H.stratified_draw(mnist, gd)
        order = torch.randperm(H.TASK_EXAMPLES, generator=gb)
        xs = mnist.train_x[idx][:, perm][order]
        ys = mnist.train_y[idx][order]
        _, _, g_start = gate_block(p, act, probe, perm)
        # ↓ P4: タスク開始時の読み出し（層 1・2 の zstart_i と帯分率・乱数の指紋）
        if readout is not None:
            readout.task_start(p, act, probe, perm, idx, order, task)
        for step in range(1, 626):
            out = H.forward(p, xs[(step - 1) * 16:step * 16], act)
            gr = torch.autograd.grad(torch.nn.functional.cross_entropy(out[4], ys[(step - 1) * 16:step * 16]), p)
            with torch.no_grad():
                m_, v_, tc = adam
                tc[0] += 1
                c1 = 1 - .9 ** tc[0]; c2 = 1 - .999 ** tc[0]
                for q, g, mi, vi in zip(p, gr, m_, v_):
                    mi.mul_(.9).add_(g, alpha=1 - .9)
                    vi.mul_(.999).addcmul_(g, g, value=1 - .999)
                    q -= lr * (mi / c1) / ((vi / c2).sqrt() + 1e-8)      # P5: .001 → lr
                if isinstance(act, H.AdaptiveSnake):
                    act.update(out[0], out[2])
        # ---- task end: committed measurement + gate block (read-outs only)
        r, u = C.measure(p, act, probe, perm, True, mnist, ck)
        gr_, gu, g_end = gate_block(p, act, probe, perm)
        r.update(gr_)
        # ↓ P6: タスク終端の読み出し（§3.3–3.4・E 恒等式・params の sha256）
        if readout is not None:
            readout.task_end(p, act, adam, probe, perm, task, r, u, gu)
        r['gcorr'], r['gcorr_degenerate'] = gcorr(g_start, g_end)
        r.update(task=task, step=625, clamp='ref')
        if isinstance(act, H.AdaptiveSnake):
            r['alpha_med'] = float(act.alpha(0).median())
        if mutate == 'measure':
            with torch.no_grad():
                p[0].data.add_(1e-9)
        if rows is not None:
            rows.append(r)
            for k, v in u.items():
                units[f'ref_{k}_t{task}'] = v
            for k, v in gu.items():
                units[f'ref_{k}_t{task}'] = v
    return p, act


# ------------------------------------------------------------------ S14: ループの写しの AST 照合
HOST_BODY_STMTS = 45      # GS.train の本体の文の数（docstring を除く・実装前に GS:186-230 から数えて固定）
# 登録ブロック（§2.2 の表）。行は body_stmts の正規化形 '<深さ>:<文>'。
REGISTERED = (
    ('P1', 'insert_after', '2:p[0].data[0, 0] += 1e-3',
     ('0:if twin is not None:', '1:AC.perturb_twin_b1(p, twin)')),
    ('P2', 'replace', '0:act = make_act(arm)', ('0:act = act_obj',)),
    ('P3', 'insert_after', '0:adam = ([torch.zeros_like(q) for q in p], [torch.zeros_like(q) for q in p], [0])',
     ('0:if readout is not None:', '1:readout.begin(p, act)')),
    ('P4', 'insert_after', '1:_, _, g_start = gate_block(p, act, probe, perm)',
     ('1:if readout is not None:', '2:readout.task_start(p, act, probe, perm, idx, order, task)')),
    ('P5', 'replace', '4:q -= .001 * (mi / c1) / ((vi / c2).sqrt() + 1e-8)',
     ('4:q -= lr * (mi / c1) / ((vi / c2).sqrt() + 1e-8)',)),
    ('P6', 'insert_after', '1:r.update(gr_)',
     ('1:if readout is not None:', '2:readout.task_end(p, act, adam, probe, perm, task, r, u, gu)')),
)
body_stmts = AC.ast_body_stmts             # AST の文の取り出し（深さつき・3 ランナー共通・act_chimera_0913）
apply_registered = AC.apply_registered


def s14_check(mine_src=None, host_src=None, expected_host=HOST_BODY_STMTS, registered=REGISTERED):
    """S14: 写した train の本体 = 宿主の本体 + 登録ブロック（差分 0）、宿主の文の数 = 固定した期待値。"""
    host_src = Path(GS.__file__).read_text(encoding='utf-8') if host_src is None else host_src
    mine_src = Path(__file__).read_text(encoding='utf-8') if mine_src is None else mine_src
    return AC.ast_copy_check(host_src, 'train', mine_src, 'train', registered, expected_host)


# ------------------------------------------------------------------ G1-PM（§2.2）
def g1_pm(act_name, seed, rows, units, n_tasks, n_registered=None):
    """gate_shape_0911 の committed units（16 配列 × t）と rows の acc を np.array_equal（形・dtype も）で照合。
    件数ガード: 登録した地平線 n_registered（既定 n_tasks）× 17 件（1 走 120 タスクで 2040）。完了したタスクが
    地平線より少ない走（錨の腕の DIVERGED）は、比べた分が一致しても pass にしない（§4.x「切り詰めで早死にが通る」）。
    maxabs は数値配列だけに併記する（hard_i は bool）。"""
    ref_arm = ANCHOR.get(act_name)
    if ref_arm is None:
        return dict(anchor=None, note='no committed anchor (new activation or twin)')
    n_registered = int(n_tasks if n_registered is None else n_registered)
    R = ROOT / 'results/gate_shape_0911'
    ref = np.load(R / f'{ref_arm}_s{seed}_units.npz')
    import csv
    with (R / f'{ref_arm}_s{seed}_rows.csv').open() as fh:
        ref_rows = list(csv.DictReader(fh))
    n_cmp, n_eq, maxabs, first_bad = 0, 0, 0., None
    for t in range(1, n_tasks + 1):
        for k in UNIT_KEYS16:
            key = f'ref_{k}_t{t}'
            a, b = units[key], ref[key]
            n_cmp += 1
            ok = a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
            n_eq += int(ok)
            if a.dtype != np.bool_ and b.dtype != np.bool_ and a.shape == b.shape:
                maxabs = max(maxabs, float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()))
            if not ok and first_bad is None:
                first_bad = key
        n_cmp += 1
        ok = float(ref_rows[t - 1]['acc']) == float(rows[t - 1]['acc'])
        n_eq += int(ok)
        if not ok and first_bad is None:
            first_bad = f'acc_t{t}'
    note = '' if n_tasks == n_registered else f'anchor arm completed {n_tasks}/{n_registered} tasks (diverged)'
    return dict(anchor=f'gate_shape_0911/{ref_arm}_s{seed}_{{units.npz,rows.csv}}', n_compared=n_cmp,
                n_equal=n_eq, n_expected=17 * n_registered, maxabs_numeric=maxabs, first_mismatch=first_bad,
                note=note, pass_=bool(n_cmp == 17 * n_registered and n_eq == n_cmp))


def g1_eg(act_name, seed, units, n_tasks, n_registered=None):
    """副次の錨（§2.2）: LR・ELU1 の zbar_i・sd_i・cnorm_i を elu_growth_0909 と ≤ 1e−10 で照合（GS:239-249 と同じ）。
    件数ガードは g1_pm と同じく登録した地平線で数える。"""
    if act_name not in ANCHOR_EG:
        return dict(anchor=None)
    n_registered = int(n_tasks if n_registered is None else n_registered)
    eg = np.load(ROOT / 'results/elu_growth_0909' / f'{act_name}_none_s{seed}_units.npz')
    w, n = 0., 0
    for t in range(1, min(n_tasks, 120) + 1):
        for k in ('zbar_i', 'sd_i', 'cnorm_i'):
            w = max(w, float(np.abs(units[f'ref_{k}_t{t}'] - eg[k][t - 1]).max())); n += 1
    want = 3 * min(n_registered, 120)
    return dict(anchor=f'elu_growth_0909/{act_name}_none_s{seed}_units.npz', maxabs=w, n_compared=n,
                n_expected=want, tol=EG_ANCHOR_TOL, pass_=bool(w <= EG_ANCHOR_TOL and n == want))


# ------------------------------------------------------------------ 発散（§2.2・§4.0）
def _first_nonfinite_task(rows):
    for r in rows:
        if any(isinstance(v, float) and not np.isfinite(v) for v in r.values()):
            return int(r['task'])
    return None


def _task_of(key):
    return int(key.rsplit('_t', 1)[1])


def _maxabs_params(pa, pb):
    return max(float((a.detach() - b.detach()).abs().max()) for a, b in zip(pa, pb))


# ------------------------------------------------------------------ run
def load_mnist():
    torch.set_num_threads(1); H.setup('cpu'); T.data_dir()
    return H.Mnist(torch.device('cpu'))


def data_check(mnist):
    """S11: MNIST 4 ファイルの sha256 = gate_shape_0911 と 0906 の provenance の data_sha256。"""
    gs = json.loads((ROOT / 'results/gate_shape_0911/LR_s0_provenance.json').read_text())['data_sha256']
    rl = json.loads((ROOT / 'results/pmnist_rlmnist_0906/LR/provenance.json').read_text())['data_sha256']
    ok = mnist.sha256 == gs == rl == T.DATA_SHA
    return dict(pass_=bool(ok), data_sha256=mnist.sha256, data_dir=str(H.DATA_DIR))


def make_probe(mnist, seed):
    pidx = torch.randperm(len(mnist.test_x), generator=H.stream('boundary_probe', seed))[:N_PROBE]
    probe = C.Probe(mnist.test_x[pidx], mnist.test_y[pidx], float(mnist.train_x.mean()))
    gi = torch.Generator().manual_seed(REF_SEED + seed)              # EG:121-123
    gidx = torch.randperm(len(mnist.train_x), generator=gi)[:NGRAD]
    return probe, pidx, mnist.train_x[gidx], mnist.train_y[gidx]


def is_anchored(act_name, twin, lr):
    """G1-PM と副次の錨を当てる走か（§2.2・P5）: 錨の活性化（LR・ELU1・SMAXH）で、双子でなく、lr がプロトコルの
    1e−3 と同じ double のとき。双子は init の b1[k] が +1e−6 違うので錨と一致しない（追補 2-2）（一致したら S18 の方が落ちる）。
    段 2 の lr {5e−4, 2e−3} は committed 軌道と別の走なので錨を当てない。"""
    return act_name in ANCHOR and twin is None and float(lr) == LR


def run(arm, seed, tasks=TASKS, out=OUT, controls=True, lr=LR, e_tasks=E_TASKS, inject_inf_after=None,
        mnist=None, check_env=True):
    """1 ジョブ = 1 (腕, seed)（§7）。戻り値は状態（'COMPLETE' / 'DIVERGED'）。

    走ごとの検査（G1-PM・副次の錨・G3・S9・S18・E 恒等式・S12）は出力をすべて書いてから判定し、落ちた検査の名前を
    provenance の ``failed_checks`` に、全体を ``checks_passed`` に書いてから AssertionError を投げる（launcher は
    checks_passed で完了を判定し、判定器は CHECK_FAILED として打ち切る）。錨の腕が DIVERGED なら G1-PM は落ちる。"""
    env = assert_env() if check_env else env_info()
    out = Path(out)
    lr = float(lr)
    registered_horizon = (int(tasks) == TASKS and inject_inf_after is None)
    committed = require_committed() if registered_horizon else dict(pass_=None, note='shortened / check run')
    mnist = load_mnist() if mnist is None else mnist
    out.mkdir(parents=True, exist_ok=True)
    tag = f'{arm}_s{seed}'
    act_name, twin = AC.twin_of(arm)
    act = AC.make_act(act_name)                                       # 未知名は KeyError（S8）
    anchored = is_anchored(act_name, twin, lr)
    probe, pidx, xg, yg = make_probe(mnist, seed)
    t0 = time.monotonic()
    snapshot_before = process_snapshot()
    rows, units, ck = [], {}, {}
    ck['s11'] = data_check(mnist)
    assert ck['s11']['pass_'], ('S11: MNIST bytes differ from gate_shape_0911 / 0906 provenance', ck['s11'])
    # S18b（追補 2-2）: LR は b1[k]（k = 0..3）を除いた sha256 を、双子 k は自分の k だけを、タスク終端ごとに記録する
    ks = (twin,) if twin is not None else (TWIN_KS if act_name == 'LR' else ())
    ro = Readout(act_name, tasks, xg, yg, e_tasks=e_tasks, inject_inf_after=inject_inf_after, sha_excl_ks=ks)
    train(arm, seed, mnist, probe, tasks, rows=rows, units=units, ck=ck, act_obj=act, twin=twin, lr=lr, readout=ro)
    # ---- 発散の記録（§2.2）: finite_guard は走り切った後に落ちるので try/except で囲む
    status, first_bad = 'COMPLETE', None
    try:
        T.finite_guard(rows, tag)
    except AssertionError as exc:
        first_bad = _first_nonfinite_task(rows)
        status = 'DIVERGED'
        ck['finite_guard_error'] = str(exc)[:500]
    done = tasks if first_bad is None else first_bad - 1
    if status == 'DIVERGED':
        rows = rows[:done]
        units = {k: v for k, v in units.items() if _task_of(k) <= done}
        T.dump(out / f'{tag}_divergence.json',
               dict(run_id=RUN_ID, arm=arm, seed=seed, status=status, first_nonfinite_task=first_bad,
                    tasks_completed=done, tasks_registered=tasks, injected_inf_after=inject_inf_after))
    ck.update(status=status, first_nonfinite_task=first_bad, tasks_completed=done, anchored=anchored)
    # ---- G3（GS:313-315）: gate_block の分散の恒等式
    ck['g3_ident_max'] = max([r['g3_ident'] for r in rows], default=0.)
    ck['g3_pass'] = bool(ck['g3_ident_max'] <= G3_TOL)
    # ---- G1-PM（§2.2）と副次の錨: 錨の腕・双子でない・lr = 1e−3 の走だけ。件数は登録の地平線（tasks）で数える
    if not anchored:
        why = ('lr differs from anchored protocol' if act_name in ANCHOR and twin is None else
               'twin (init b1[k] differs by +1e-6)' if act_name in ANCHOR else 'no committed anchor (new activation)')
        ck['g1_pm'] = dict(anchor=None, note=why)
        ck['g1_eg'] = dict(anchor=None, note=why)
    elif done > 0:
        ck['g1_pm'] = g1_pm(act_name, seed, rows, units, done, n_registered=tasks)
        ck['g1_eg'] = g1_eg(act_name, seed, units, done, n_registered=tasks)
    else:
        note = 'anchor arm diverged before completing task 1'
        ck['g1_pm'] = dict(anchor=ANCHOR[act_name], n_compared=0, n_expected=17 * tasks, pass_=False, note=note)
        ck['g1_eg'] = (dict(anchor=act_name, n_compared=0, pass_=False, note=note) if act_name in ANCHOR_EG
                       else dict(anchor=None))
    if anchored and status == 'DIVERGED':
        ck['g1_pm']['note'] = 'anchor arm diverged (committed trajectory completed 120 tasks)'
    # ---- 走の中の恒等式（S12・C.measure との整合）と E 恒等式（EG:135-141）
    ck.update(ro.ck)
    ck['s12_in_run'] = ro.s12_in_run()
    ck['identity'] = ro.identity_checks(done)
    if ck['identity']['evaluated']:
        # 誤った h でも ≤ 1e−4 になる走は pass にせず「対照を実証できない」と記録する（§3.4）
        ck['identity']['ctl_demonstrated'] = bool(ck['identity']['identity_mutctl'] > IDENTITY_CTL)
    # ---- 乱数の指紋（S9）: ループの中の記録 = ループの外で引いた系列
    fp = rng_fingerprint(seed, min(3, done), mnist)
    fp_loop = dict(ro.rng_sha)
    fp_loop['probe_idx'] = _sha_tensor(pidx)
    ck['s9'] = dict(pass_=bool(all(fp[k] == fp_loop.get(k) for k in fp if k != 'init')
                               and (fp['init'] == fp_loop['init']) == (twin is None)),
                    twin=twin, loop=fp_loop, standalone_init=fp['init'])
    if twin is not None:
        ck['s18'] = twin_report(seed, twin, loop_init_sha=fp_loop['init'])     # 走そのものの init と結ぶ
        c0 = twin_report(seed, twin, delta=0.)                                  # 対照: 摂動 0 の双子は「差 0 要素」で落ちる
        ck['s18']['ctl_zero_perturbation'] = dict(n_diff=c0['n_diff'], pass_=c0['pass_'],
                                                  fails_as_required=bool(c0['n_diff'] == 0 and not c0['pass_']))
    if ks:
        # S18b（追補 2-2）: 生存は判定器が同じ seed の LR の列と比べて twin_live_task を出す（LR と双子は並列のジョブ）
        ck['s18b'] = dict(excluded='params[1][k] = b1[k] set to 0 in a CPU copy before sha256', ks=list(ks),
                          sha_excl_b1={str(k): list(v[:done]) for k, v in ro.sha_excl.items()},
                          n_tasks=done, rule='twin_live_task = first task where the twin differs from same-seed LR; '
                                             'a twin not live by the window start (t16) is excluded from sigma_traj')
    # ---- 対照（seed 0）
    if controls and seed == 0 and done > 0 and anchored:
        # G1 の変異対照: 初期値 W1[0,0] += 1e−3 で t1 まで学習し、不一致になること（GS:341-344）
        r1, u1 = [], {}
        train(arm, seed, mnist, probe, 1, mutate='init', rows=r1, units=u1, ck={}, act_obj=act, twin=twin, lr=lr)
        c = g1_pm(act_name, seed, r1, u1, 1)
        ck['ctl_g1_init'] = dict(n_compared=c['n_compared'], n_mismatch=c['n_compared'] - c['n_equal'],
                                 maxabs_numeric=c['maxabs_numeric'])
        assert ck['ctl_g1_init']['n_mismatch'] > 0, ('vacuous G1 init control', ck['ctl_g1_init'])
    if controls and seed == 0 and status == 'COMPLETE' and tasks >= 2:
        # S10: 新読み出しの有無で 2 タスク後の params が maxabs 0.0、対照は測定の後に W1 += 1e−9 する読み出し
        pa, _ = train(arm, seed, mnist, probe, 2, act_obj=act, twin=twin, lr=lr, readout=None)
        pb, _ = train(arm, seed, mnist, probe, 2, act_obj=act, twin=twin, lr=lr,
                      readout=Readout(act_name, 2, xg, yg, e_tasks=(1, 2)))
        pc, _ = train(arm, seed, mnist, probe, 2, act_obj=act, twin=twin, lr=lr,
                      readout=Readout(act_name, 2, xg, yg, e_tasks=(1, 2), mut='add1e-9'))
        ck['s10_noninvasive'] = _maxabs_params(pa, pb)
        ck['ctl_s10_add1e-9'] = _maxabs_params(pa, pc)
        assert ck['s10_noninvasive'] == 0., ('readout changed the trajectory', ck['s10_noninvasive'])
        assert ck['ctl_s10_add1e-9'] > 0., 'vacuous S10 control'
    if controls and seed == 0:
        # G3 の対照（GS:355-360）
        pr, _, _ = gate_block([q.detach() for q in H.init_params(seed, torch.device('cpu'))], act, probe,
                              probe.refs[0], mut_var=True)
        ck['ctl_g3_ident'] = pr['g3_ident']
        ck['ctl_g3_applicable'] = pr['pooled_var'] > 1e-6
        if ck['ctl_g3_applicable']:
            assert pr['g3_ident'] > 100 * G3_TOL, ('vacuous G3 control', pr['g3_ident'])
    # ---- 走ごとの判定（書き出しの前に名前を集め、書き出しの後に落とす: 失敗した走も監査できるように）
    failed = []
    if not ck['g3_pass']:
        failed.append('G3')
    if ck['g1_pm'].get('anchor') and not ck['g1_pm'].get('pass_'):
        failed.append('G1_PM')
    if ck['g1_eg'].get('anchor') and not ck['g1_eg'].get('pass_'):
        failed.append('G1_EG')
    if not ck['s9']['pass_']:
        failed.append('S9')
    if twin is not None and not (ck['s18']['pass_'] and ck['s18']['ctl_zero_perturbation']['fails_as_required']):
        failed.append('S18')
    idc = ck['identity']
    if idc['evaluated'] and not idc['identity_rel'] < IDENTITY_TOL:
        failed.append('E_IDENTITY')
    if idc['evaluated'] and 'leaky_zero' in idc and not idc['leaky_zero'] < IDENTITY_TOL:
        failed.append('E_LEAKY_ZERO')
    if not ck['s12_in_run']['pass_']:
        failed.append('S12_IN_RUN')
    ck['failed_checks'] = failed
    # ---- 書き出し
    for r in rows:
        r.update(arm=arm, seed=seed, act=act_name, twin=twin, lr=lr)
    if rows:
        T.write_rows(out / f'{tag}_rows.csv', rows)
    np.savez_compressed(out / f'{tag}_units.npz', **units)
    np.savez(out / f'{tag}_readout.npz', **ro.arrays(done))
    wall = time.monotonic() - t0
    sizes = dict(units_bytes=(out / f'{tag}_units.npz').stat().st_size,
                 readout_bytes=(out / f'{tag}_readout.npz').stat().st_size)
    ck['readout_within_budget'] = bool(sizes['readout_bytes'] <= READOUT_BUDGET)
    prov = dict(run_id=RUN_ID, env='pmnist', arm=arm, act=act_name, twin=twin, seed=seed, tasks=tasks,
                tasks_completed=done, status=status, lr=lr, e_tasks=list(e_tasks), theta=GS.THETA,
                n_probe=N_PROBE, ngrad=NGRAD, ref_seed=REF_SEED, inject_inf_after=inject_inf_after,
                registered_horizon=registered_horizon, committed_code=committed,
                checks_passed=not failed, failed_checks=failed, checks=ck, **git_info(),
                band_predicate_dtype=AC.BAND_PREDICATE_DTYPE,
                fn_mom_axes=dict(funcs=list(AC.FUNCS8), bands=list(AC.BANDS), moments=list(AC.MOM),
                                 shape='(T, layer, func, band, moment)'),
                spec_sha256=T.sha(SPEC), code_sha256=T.sha(Path(__file__)), act_sha256=T.sha(Path(AC.__file__)),
                host_sha256=T.sha(Path(H.__file__)), gs_sha256=T.sha(Path(GS.__file__)),
                base_sha256=T.sha(Path(C.__file__)), elu_sha256=T.sha(Path(EG.__file__)),
                common_sha256=T.sha(Path(T.__file__)), data_sha256=mnist.sha256, data_dir=str(H.DATA_DIR),
                **{f'env_{k}': v for k, v in env.items()},
                wall_seconds=wall, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                other_sessions_before=snapshot_before, other_sessions_after=process_snapshot(), **sizes,
                scope='CPU training from init (GS.train copied verbatim + P1-P6); LR/ELU1/SMAXH reproduce gate_shape_0911 bit for bit')
    T.dump(out / f'{tag}_provenance.json', prov)
    g1 = ck['g1_pm']
    print('FINISHED', tag, status, 'G1', g1.get('n_equal'), '/', g1.get('n_expected'), 'identity',
          ck['identity'].get('identity_rel'), 'failed', failed, round(wall, 1), 's', flush=True)
    assert not failed, ('per-run checks failed (outputs and provenance written)', tag, failed)
    return status


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--arm'); ap.add_argument('--seed', type=int)
    ap.add_argument('--tasks', type=int, default=None); ap.add_argument('--outdir', default=None)
    ap.add_argument('--lr', type=float, default=LR)
    ap.add_argument('--smoke', action='store_true', help='17 腕 × seed 0–2 × t1–3（§6・出力は results/_smoke_...）')
    ap.add_argument('--arms', default=None); ap.add_argument('--seeds', default=None)
    ap.add_argument('--no-controls', action='store_true')
    ap.add_argument('--e-tasks', default=None, help='E 恒等式を計るタスク（カンマ区切り・既定は 20,40,...,120）')
    ap.add_argument('--inject-inf-after', type=int, default=None, help='S20 の検査だけ: task t の後に W1[0,0]=inf')
    a = ap.parse_args()
    assert_env()
    e_tasks = tuple(int(t) for t in a.e_tasks.split(',')) if a.e_tasks else None
    if a.smoke:
        out = Path(a.outdir) if a.outdir else SMOKE
        tasks = a.tasks or SMOKE_TASKS
        arms = a.arms.split(',') if a.arms else list(ARMS_STAGE1)
        seeds = [int(s) for s in a.seeds.split(',')] if a.seeds else [0, 1, 2]
        mnist = load_mnist()
        summary = []
        for arm in arms:
            for s in seeds:
                t0 = time.monotonic()
                err = None
                try:
                    st = run(arm, s, tasks=tasks, out=out, controls=not a.no_controls, lr=a.lr,
                             e_tasks=e_tasks or SMOKE_E_TASKS, mnist=mnist)
                except AssertionError as exc:                 # 走ごとの検査の失敗: 記録して次へ（最後に非 0 で終わる）
                    st, err = 'CHECK_FAILED', repr(exc)[:2000]
                pp = out / f'{arm}_s{s}_provenance.json'
                prov = json.loads(pp.read_text()) if pp.exists() else dict(checks={}, peak_rss_kib=None)
                summary.append(dict(arm=arm, seed=s, status=st, error=err, wall=time.monotonic() - t0,
                                    checks_passed=prov.get('checks_passed', False),
                                    failed_checks=prov.get('failed_checks'),
                                    g1=prov['checks'].get('g1_pm'), identity=prov['checks'].get('identity'),
                                    peak_rss_kib=prov.get('peak_rss_kib')))
        ok = all(r['status'] == 'COMPLETE' and r['checks_passed'] for r in summary)
        T.dump(out / 'smoke_summary.json', dict(pass_=ok, tasks=tasks, arms=arms, seeds=seeds, runs=summary,
                                                 env=env_info(), **git_info()))
        if not ok:
            raise SystemExit(f'box B smoke: {sum(1 for r in summary if not r["checks_passed"])} runs failed their checks')
        return
    assert a.arm is not None and a.seed is not None, '--arm and --seed are required (or --smoke)'
    tasks = a.tasks or TASKS
    if a.outdir is None and (float(a.lr) != LR or tasks != TASKS or a.inject_inf_after is not None):
        # §8: 段 2（lr ≠ 1e−3）は stage2/ に、短縮・注入の走は本走の出力先に書かない（同じ {ARM}_s{seed} で上書きする）
        ap.error('--outdir is required for --lr != 0.001, --tasks != 120 or --inject-inf-after '
                 '(stage 2 goes to results/act_chimera_0913/stage2/pmnist_lr{lr}, checks go to scratch)')
    st = run(a.arm, a.seed, tasks=tasks, out=Path(a.outdir) if a.outdir else OUT,
             controls=not a.no_controls, lr=a.lr, e_tasks=e_tasks or E_TASKS, inject_inf_after=a.inject_inf_after)
    print('STATUS', st, flush=True)


if __name__ == '__main__':
    main()
