"""Sequential replay launcher; does not poll the child or use extra GPU processes."""
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[2]
dest=root/'results/redo_positive_lineage_1019/logs'
dest.mkdir(parents=True,exist_ok=True)
rows=[]
for method in ('redo','none'):
    for group in (0,10):
        name=f'{method}_s{group}-{group+9}'
        print(f'START {name}',flush=True)
        start=time.time()
        with (dest/f'{name}.log').open('w') as out:
            code=subprocess.call([sys.executable,str(root/'analysis/redo_positive_lineage_1019/run.py'),
                                  '--mode','run','--method',method,'--group',str(group)],
                                 cwd=root,stdout=out,stderr=subprocess.STDOUT)
        rows.append(dict(name=name,code=code,seconds=time.time()-start))
        (dest/'status.json').write_text(json.dumps(rows,indent=2))
        print(f'FINISH {name}: code={code}, {time.time()-start:.1f}s',flush=True)
        if code:
            print((dest/f'{name}.log').read_text()[-8000:],flush=True)
            raise SystemExit(code)
