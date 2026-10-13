"""TASK V (cobirth_seeds_1013, verifier): run the ORIGINAL p172c_channels.py source text, unmodified in its computation, at one state.
The source is read from push_direction_proof_0930/.../agent_bound/p172c_channels.py (read-only) and executed with exactly four textual
substitutions, each asserted to hit exactly once:
  (1) thread count "2" -> "1"                       (ground rules)
  (2) HERE = <script folder>  -> HERE = OUTDIR      (output goes to the verify folder; S10 is used only in 'helpers' mode)
  (3) the seed-0 src/a dict lookup -> the given npz and slope
  (4) nothing else; sys.argv is set to [tag, t, 'real'] in the ORIGINAL convention t (state end0[t-2]; M1 T = t - 1).
usage: verify_p172c_orig.py <npz rel. to ROOT> <a> <tag> <t (original convention)> [outdir] [slim]
  slim: after the original script has written its json, the per-input 'rows' and 'top10' lists of each target record are dropped
        (n_b, need_b, max_L_b, A, self, ... are kept) to save space for the all-state sweep."""
import os, sys
ORIG = "/home/issan/Projects/obsidian-research-data/push_direction_proof_0930/review/stage4_1002/block_1003/stage14_1007_h4/agent_bound/p172c_channels.py"
OUTDIR = "/home/issan/Projects/obsidian-research-data/cobirth_seeds_1013/verify/p172c_orig"
npz, a, tag, t = sys.argv[1], float(sys.argv[2]), sys.argv[3], int(sys.argv[4])
if len(sys.argv) > 5: OUTDIR = sys.argv[5]
SLIM = len(sys.argv) > 6 and sys.argv[6] == "slim"
os.makedirs(OUTDIR, exist_ok=True)
src = open(ORIG).read()
subs = [
    ('for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "2"',
     'for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "1"'),
    ('HERE = os.path.dirname(os.path.abspath(__file__))', f'HERE = {OUTDIR!r}'),
    ('src, a = {"a01": ("review/opus_bwdfloor/out/LR_ref_s0.npz", 0.1), "a03": ("review/opus_lrslope/out/LRa0.3_s0.npz", 0.3), "a05": ("review/opus_lrslope/out/LRa0.5_s0.npz", 0.5)}[tag]',
     f'src, a = {npz!r}, {a!r}'),
]
for old, new in subs:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
_argv_keep = list(sys.argv)
sys.argv = ["p172c_channels.py", tag, str(t), "real"]
g = {"__name__": "__main__", "__file__": ORIG}
exec(compile(src, ORIG, "exec"), g)
if SLIM:
    import json
    fp = os.path.join(OUTDIR, f"p172c_{tag}_t{t}_real.json"); j = json.load(open(fp))
    for r in j["real"]: r.pop("rows", None); r.pop("top10", None)
    json.dump(j, open(fp, "w"))
