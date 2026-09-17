#!/usr/bin/env bash
# neffdir_ee_0918 -- spec section 8: 10 jobs (seeds 30-39, one per seed: the prefix + the 12 registered carrier arms), 1 thread each, at most
# MAX_SLOTS at once.
# Refuses to start unless checks.json says all_pass, PREREG_COMMIT is set in the runner, the code and the
# spec are committed and HEAD is pushed.  This is a shared desktop: every start goes through one flock; the
# holder starts a job only if MemAvailable leaves 6 GiB for the desktop after 1.2x the S-cost peak RSS, then
# waits 25 s (so the next reading sees the new RSS) before releasing the lock; otherwise it releases and
# retries after 60 s.  A job with provenance.json is skipped (relaunch-safe).  A STOP file in $OUT keeps
# jobs that have not started from starting.  Log names are flat.
#   setsid nohup bash analysis/neffdir_ee_0918/launch.sh > results/neffdir_ee_0918/logs/launch.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=results/neffdir_ee_0918
MOD=src/neffdir_ee_0918.py
PY=/usr/bin/python3
MAX_SLOTS=${MAX_SLOTS:-3}
[[ "$MAX_SLOTS" =~ ^[1-3]$ ]] || { echo "ABORT: MAX_SLOTS must be 1, 2, or 3"; exit 1; }
$PY - <<'GATE'
import ast, hashlib, json, re, subprocess
from pathlib import Path
c = json.loads(Path("results/neffdir_ee_0918/checks.json").read_text())
required = ("S1a_S-records", "S1b_S-carriers", "S1c_S-derivatives", "S8_S-verdict", "S9_S-cli", "S-cost")
assert c.get("all_pass") and c.get("finished_at"), "checks incomplete or failed"
tree = ast.parse(Path("src/neffdir_ee_0918.py").read_text())
prereg = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "PREREG_COMMIT" for t in n.targets))
assert c["prereg_commit"] == prereg and len(prereg) == 40, "registration differs from checks"
subprocess.run(["git", "merge-base", "--is-ancestor", prereg, "HEAD"], check=True)
predictions = Path("specs/spec_neffdir_ee_0918.md").read_text().split("### 7.3", 1)[1].split("## 8.", 1)[0]
for quantity in ("primary", "prop_add_h", "compensation", "felu"):
    assert re.search(r"^\| " + quantity + r" \| [A-Z_]+ \|$", predictions, re.M), f"Issa prediction missing: {quantity}"
for name in required:
    entry = c[name]
    assert entry["pass"] and entry.get("all_mutations_detected", True), name
for path, expected in c["code_sha256"].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, f"checks stale: {path}"
GATE
for f in "$MOD"; do
  grep -q '^PREREG_COMMIT = "[0-9a-f]\{40\}"' "$f" || { echo "ABORT: PREREG_COMMIT not set in $f"; exit 1; }
done
[ -z "$(git status --porcelain -- src analysis/neffdir_ee_0918 specs/spec_neffdir_ee_0918.md)" ] \
  || { echo "ABORT: uncommitted code or spec"; exit 1; }
git fetch -q origin
[ "$(git rev-list --count '@{u}..HEAD')" = 0 ] || { echo "ABORT: HEAD not pushed"; exit 1; }
NEED_KB=$($PY -c "
import json
c = json.load(open('$OUT/checks.json'))['S-cost']
print(int(1.2 * c['peak_rss_gib'] * 2**20))")
RESERVE_KB=$((6 * 1024 * 1024))
mkdir -p "$OUT/runs" "$OUT/logs"
LOCK="$OUT/logs/.start.lock"
exec 8>"$OUT/logs/.launch.lock"
flock -n 8 || { echo "ABORT: this experiment already has a launcher"; exit 1; }

run_job() {
  local kind=$1 name=$2 dir=$3; shift 3
  if [ -f "$dir/provenance.json" ]; then echo "$(date -Is) skip $name (done)"; return 0; fi
  while true; do
    [ -f "$OUT/STOP" ] && { echo "$(date -Is) STOP: not starting $name"; return 0; }
    exec 9>"$LOCK"
    flock 9
    local avail
    avail=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo)
    if [ "$avail" -ge $((RESERVE_KB + NEED_KB)) ] && [ ! -f "$OUT/STOP" ]; then
      echo "$(date -Is) start $name  avail $((avail / 1024)) MiB"
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 "$PY" "$@" --threads 1 --out "$dir" \
        > "$OUT/logs/$name.log" 2>&1 9>&- &
      local pid=$!
      sleep 25
      flock -u 9
      exec 9>&-
      if wait "$pid"; then
        echo "$(date -Is) done  $name"
      else
        local rc=$?
        echo "$(date -Is) FAIL  $name (exit $rc)"
        touch "$OUT/STOP"
        return "$rc"
      fi
      return 0
    fi
    flock -u 9
    exec 9>&-
    sleep 60
  done
}
export -f run_job
export OUT PY NEED_KB RESERVE_KB LOCK

JOBS=()
for s in 30 31 32 33 34 35 36 37 38 39; do JOBS+=("main s$s $OUT/runs/s$s $MOD --seed $s"); done

echo "$(date -Is) launch: git $(git rev-parse HEAD)  slots $MAX_SLOTS  need $((NEED_KB / 1024)) MiB/job  $(free -g | awk '/^Mem:/{print "avail "$7" GiB"}')"
printf '%s\n' "${JOBS[@]}" | xargs -P "$MAX_SLOTS" -I{} bash -c 'run_job $1' _ {}
echo "$(date -Is) all jobs returned"
