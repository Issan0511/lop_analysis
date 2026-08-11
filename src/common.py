"""共通ユーティリティ: config ロード、run テーブル構築、デバイス選択。"""
import os
import yaml
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def config_title(config_path):
    """config のファイル名 (拡張子なし) が実験のタイトル兼出力ディレクトリ名。"""
    return os.path.splitext(os.path.basename(config_path))[0]


def resolve_outdir(config_path, smoke=False, outdir=None):
    """出力先: --outdir 明示 > --smoke なら results/_smoke > config 名から results/<title>。"""
    if outdir:
        return outdir
    if smoke:
        return os.path.join(ROOT, "results", "_smoke")
    return os.path.join(ROOT, "results", config_title(config_path))


def pick_device(cfg):
    dev = cfg["common"].get("device", "auto")
    if dev == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return dev


def build_runs(cfg):
    """全 run (条件×シード) の平坦リストを返す。各 run は dict。

    batch_values (default [1]): ミニバッチサイズ。1=従来のオンライン SGD、
      整数 B=iid B サンプル平均勾配、"full"=フルバッチ GD
      (条件A: 2^(m-f) パターン厳密列挙 / 条件B: full_batch_B サンプル近似)。
    kappa_values (条件B, default [1]): スパイク共分散 Sigma = I + (kappa-1)uu^T。"""
    runs = []
    A, B, C = cfg["condA"], cfg["condB"], cfg["common"]
    lr0 = C["lr_main"]
    for width in A.get("widths", []):
        for T in A.get("T_values", []):
            for enc in A["encodings"]:
                for batch in A.get("batch_values", [1]):
                    for seed in C["seeds"]:
                        runs.append(dict(exp="A", width=width, period=T, enc=enc,
                                         c=None, kappa=1, lr=lr0, batch=batch, seed=seed))
    gc_ = A.get("lr_grid_condition")
    for lr in A.get("lr_grid", []):
        for seed in C["seeds"]:
            runs.append(dict(exp="A", width=gc_["width"], period=gc_["T"],
                             enc=gc_["encoding"], c=None, kappa=1, lr=lr, batch=1, seed=seed))
    for width in B.get("widths", []):
        for K in B.get("K_values", []):
            for c in B.get("c_values", []):
                for kappa in B.get("kappa_values", [1]):
                    for batch in B.get("batch_values", [1]):
                        for lr in B.get("lr_values", [lr0]):
                            for seed in C["seeds"]:
                                runs.append(dict(exp="B", width=width, period=K, enc="std",
                                                 c=c, kappa=kappa, lr=lr, batch=batch, seed=seed))
    for r in runs:
        r["run_id"] = run_id(r)
    return runs


def run_id(r):
    b = f"_b{r.get('batch', 1)}" if r.get("batch", 1) != 1 else ""
    if r["exp"] == "A":
        return f"A_w{r['width']}_T{r['period']}_{r['enc']}_lr{r['lr']}{b}_s{r['seed']}"
    k = f"_k{r.get('kappa', 1)}" if r.get("kappa", 1) != 1 else ""
    return f"B_w{r['width']}_K{r['period']}_c{r['c']}{k}_lr{r['lr']}{b}_s{r['seed']}"


def group_runs(runs):
    """(exp, width, batch) ごとにグループ化。グループ内は R 次元でベクトル化して学習する。"""
    groups = {}
    for r in runs:
        groups.setdefault((r["exp"], r["width"], r.get("batch", 1)), []).append(r)
    return groups


def group_name(gkey):
    """グループ名。batch=1 は従来どおり A_w5 等 (drift_0809 の成果物と互換)。"""
    exp, width, batch = gkey
    return f"{exp}_w{width}" + (f"_b{batch}" if batch != 1 else "")
