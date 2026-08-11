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
    """全 run (条件×シード) の平坦リストを返す。各 run は dict。"""
    runs = []
    A, B, C = cfg["condA"], cfg["condB"], cfg["common"]
    lr0 = C["lr_main"]
    for width in A["widths"]:
        for T in A["T_values"]:
            for enc in A["encodings"]:
                for seed in C["seeds"]:
                    runs.append(dict(exp="A", width=width, period=T, enc=enc,
                                     c=None, lr=lr0, seed=seed, norm="free"))
    gc_ = A["lr_grid_condition"]
    for lr in A["lr_grid"]:
        for seed in C["seeds"]:
            runs.append(dict(exp="A", width=gc_["width"], period=gc_["T"],
                             enc=gc_["encoding"], c=None, lr=lr, seed=seed, norm="free"))
    for width in B["widths"]:
        for K in B["K_values"]:
            for c in B["c_values"]:
                for nm in B.get("norm_modes", ["free"]):
                    for seed in C["seeds"]:
                        runs.append(dict(exp="B", width=width, period=K, enc="std",
                                         c=c, lr=lr0, seed=seed, norm=nm))
    for r in runs:
        r["run_id"] = run_id(r)
    return runs


def run_id(r):
    if r["exp"] == "A":
        return f"A_w{r['width']}_T{r['period']}_{r['enc']}_lr{r['lr']}_s{r['seed']}"
    tag = "" if r.get("norm", "free") == "free" else "_nfix"
    return f"B_w{r['width']}_K{r['period']}_c{r['c']}_lr{r['lr']}_s{r['seed']}{tag}"


def group_runs(runs):
    """(exp, width) ごとにグループ化。グループ内は R 次元でベクトル化して学習する。"""
    groups = {}
    for r in runs:
        groups.setdefault((r["exp"], r["width"]), []).append(r)
    return groups
