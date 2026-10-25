"""Shared pieces of vit_long_1024 (specs/spec_vit_long_1024.md §4, §6).

Everything here either imports the frozen 0919 code unchanged or mirrors it so that
cycle 0 (tasks 1-40) is bit-identical to the factorial N cell's task stream:
images from `randlabel_control_joudaki_vit_0919.make_task_dataset`, labels from its
`random_labels`, the model from `factorial_joudaki_vit_0919.cell_config(4.0, 'none')`.
Nothing is added to `analysis/joudaki_vit_battle_0919/`, which is frozen by sha256.
"""
import hashlib
import json
import math
import os
import random
import subprocess
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from analysis.factorial_joudaki_vit_0919 import cell_config
from analysis.joudaki_vit_battle_0919.data import task_classes, transform
from analysis.joudaki_vit_battle_0919.model import make_model, stream_seed
from analysis.joudaki_vit_battle_0919.run import DATA, sha256
from analysis.randlabel_control_joudaki_vit_0919 import CLASSES, IMAGES_PER_CLASS, random_labels

REPO = Path(__file__).resolve().parents[2]


def _production_file():
    """production.json (6 blocks, the main run) unless VIT_LONG_PRODUCTION names another file, e.g.
    production_2blk.json for addendum A (spec 追補 A); a bare name is looked up next to this file."""
    path = Path(os.environ.get('VIT_LONG_PRODUCTION', 'production.json'))
    return path if path.is_absolute() else Path(__file__).resolve().parent / path


PRODUCTION_FILE = _production_file()
_PRODUCTION = json.loads(PRODUCTION_FILE.read_text())
# Paths can be moved for the rented GPU (spec §10.3); the defaults are white-san's. A production
# file may name its own results and raw folders so that addendum A does not overwrite the main run.
RAW = Path(os.environ.get('VIT_LONG_RAW', str(Path(_PRODUCTION.get(
    'raw_dir', '~/Projects/obsidian-research-data/vit_long_1024')).expanduser())))
DATA_ROOT = Path(os.environ.get('VIT_LONG_DATA', str(DATA)))
RESULTS = REPO / _PRODUCTION.get('results_dir', 'results/vit_long_1024')

CONFIG = cell_config(4.0, 'none')        # the factorial N cell: 13 norms -> Identity, FFN 1536
CONFIG_LN = cell_config(4.0, 'layer')    # the randlabel cell: 13 affine LayerNorms (the original ViT)
ACT = 'R'                                # ReLU in the six FFNs
TASKS, STEPS, SEEDS = 60, 500, (0, 1, 2, 3, 4)
# 2 x 2 (Issa 1024): head reset (R0) or shared (S), without LN or with LN (suffix L).
ARMS = ('R0', 'S', 'R0L', 'SL')
FRESH_ARMS = ('fresh', 'freshL')         # one fresh net per seed and LN condition, R0's protocol
SHARED_ARMS = ('S', 'SL')
LN_ARMS = ('R0L', 'SL', 'freshL')
PAIRS = (('R0', 'S'), ('R0L', 'SL'))
FRESH_FOR = {'R0': 'fresh', 'S': 'fresh', 'R0L': 'freshL', 'SL': 'freshL'}
FRESH_TASKS = (25, 50, 60)
BATCH, LR = 128, 1e-4
EVAL_EVERY = 25
UNREACHED = STEPS + EVAL_EVERY           # T90 of a task that never reaches .9 (spec §1)
def production():
    return json.loads(PRODUCTION_FILE.read_text())


# Saves (spec §6.4): task-end weights, Adam state (+RNG), update-100 weights, from production.json
# (reduced on 1024 for transfer and for the rented machine's 80 GB disk).
_SAVES = production()
END_TASKS = tuple(_SAVES['end_tasks'])
ADAM_TASKS = tuple(_SAVES['adam_tasks'])
MID_TASKS = tuple(_SAVES['mid_tasks'])
MID_FRESH = bool(_SAVES['mid_fresh'])


def free_floor():
    """S-disk floor: 25 GiB as in the battle; VIT_LONG_FREE_FLOOR_GIB lowers it on a small rented disk."""
    return int(float(os.environ.get('VIT_LONG_FREE_FLOOR_GIB', '25')) * 1024**3)
MID_STEP = 100
PROBE_IMAGES = 64
READOUT_IMAGES = 16
DEPTH = CONFIG['depth']                  # 6: the default (main run); addendum A uses 2 by argument
MLP_RATIO = CONFIG['mlp_ratio']          # 4.0 (FFN 1536); addendum A uses 8.0 (FFN 3072)
WIDTH = int(CONFIG['embed_dim'] * CONFIG['mlp_ratio'])


# --------------------------------------------------------------------------- task plan (§4.2)

# The default layout is the N cell's: 5 labels x 50 images per class, bit-identical to the
# factorial/randlabel task stream (S-plan, S-stream). Production uses production.json
# (Issa 1024: 10 labels x 25 images per class), passed explicitly by the launcher.
N_LABELS, PER_CLASS = CLASSES, IMAGES_PER_CLASS


def production_layout():
    p = production()
    return dict(arms=p['arms'], n_labels=p['n_labels'], images_per_class=p['images_per_class'], steps=p['steps'],
                tasks=p['tasks'], augment=p['augment'], dropout=p['dropout'], depth=p.get('depth', DEPTH),
                mlp_ratio=p.get('mlp_ratio', MLP_RATIO))


def cycle_classes(seed, cycle, n_labels=N_LABELS):
    """Disjoint groups of n_labels classes; with 5 labels, cycle 0 is the battle's task_classes(seed)."""
    if cycle == 0 and n_labels == CLASSES:
        return task_classes(seed)
    order = list(range(200))
    random.Random(seed if cycle == 0 else stream_seed('cycle', seed, cycle)).shuffle(order)
    return [order[i:i + n_labels] for i in range(0, 200 - 200 % n_labels, n_labels)]


def task_plan(seed, task, n_labels=N_LABELS, per_class=PER_CLASS):
    """Task t's classes and image range. With the production key fixed_images (addendum D) every
    task uses the classes task classes_of_task would have had and images first.. of each class;
    only the labels change with the task (task_labels)."""
    per_cycle = 200 // n_labels
    fixed = _PRODUCTION.get('fixed_images')
    if fixed:
        cycle, position = divmod(fixed['classes_of_task'] - 1, per_cycle)
        first = fixed['first']
        if first + per_class > 500:
            raise ValueError(f'fixed images need {first + per_class} images per class, beyond 500')
        return dict(task=task, cycle=cycle, position=position, n_labels=n_labels, fixed=True,
                    classes=list(cycle_classes(seed, cycle, n_labels)[position]), first=first, count=per_class)
    cycle, position = divmod(task - 1, per_cycle)
    first = per_class * cycle
    if first + per_class > 500:
        raise ValueError(f'task {task} needs images beyond 500 per class')
    return dict(task=task, cycle=cycle, position=position, n_labels=n_labels,
                classes=list(cycle_classes(seed, cycle, n_labels)[position]), first=first, count=per_class)


def is_fresh(arm):
    return arm in FRESH_ARMS


def loss_classes(arm, seed, task, n_labels=N_LABELS, per_class=PER_CLASS):
    """The head rows used by the loss and accuracy (spec §1, §4.3): S and SL keep task 1's rows."""
    if arm in SHARED_ARMS:
        return task_plan(seed, 1, n_labels, per_class)['classes']
    return task_plan(seed, task, n_labels, per_class)['classes']           # R0 and fresh


def task_labels(n, seed, task, n_labels=N_LABELS):
    """iid uniform labels on the ('randlabel', seed, task) stream; 5 labels is randlabel's own function."""
    if n_labels == CLASSES:
        return random_labels(n, seed, task)
    generator = torch.Generator().manual_seed(stream_seed('randlabel', seed, task))
    return torch.randint(0, n_labels, (n,), generator=generator)


def zero_head_at(arm, task, first_task):
    """R0, R0L and fresh zero all 200 head rows at every task start; S and SL only at task 1."""
    return arm not in SHARED_ARMS or task == first_task == 1


# --------------------------------------------------------------------------- images

class RealImages:
    kind = 'real'

    def __init__(self, root=None):
        self.root = Path(DATA_ROOT if root is None else root)
        self.synsets = sorted((self.root / 'wnids.txt').read_text().split())
        assert len(self.synsets) == 200, len(self.synsets)

    def train_paths(self, cls):
        paths = sorted((self.root / 'train' / self.synsets[cls] / 'images').glob('*.JPEG'))
        assert len(paths) == 500, (self.synsets[cls], len(paths))
        return paths

    def train(self, cls, first, count):
        chosen = self.train_paths(cls)[first:first + count]
        images = np.stack([np.array(Image.open(p).convert('RGB')) for p in chosen])
        return images, [str(p.relative_to(self.root)) for p in chosen]

    def val_names(self):
        return sorted(line.split('\t')[0] for line in
                      (self.root / 'val/val_annotations.txt').read_text().splitlines())

    def val(self, names):
        return np.stack([np.array(Image.open(self.root / 'val/images' / n).convert('RGB')) for n in names])


class SyntheticImages:
    """Deterministic uint8 noise with the dataset's shapes, for the data-free checks."""
    kind = 'synthetic'

    @staticmethod
    def _image(*key):
        return np.random.default_rng(stream_seed('synthetic', *key)).integers(0, 256, (64, 64, 3), dtype=np.uint8)

    def train(self, cls, first, count):
        idx = range(first, first + count)
        return np.stack([self._image('train', cls, j) for j in idx]), [f'synthetic/train/{cls}/{j}' for j in idx]

    def val_names(self):
        return sorted(f'val_{i}.JPEG' for i in range(10000))

    def val(self, names):
        return np.stack([self._image('val', n) for n in names])


def image_source(kind):
    return RealImages() if kind == 'real' else SyntheticImages()


class TaskData(Dataset):
    """Same item protocol as the battle's TaskImages: (index, augment seed) -> (image, label).

    augment=False feeds training the evaluation transform (no pad/crop/flip/jitter); the batch
    order is unchanged because it comes from the sampler, not from here.
    """

    def __init__(self, images, labels, paths, augment=True):
        self.images, self.labels, self.paths, self.augment = images, labels, paths, augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        idx, aug_seed = item if isinstance(item, tuple) else (item, None)
        return transform(self.images[idx], aug_seed if self.augment else None), self.labels[idx]


def task_dataset(source, seed, task, n_labels=N_LABELS, per_class=PER_CLASS, augment=True):
    """Task t's images (per_class per class, path order) and its iid uniform labels."""
    plan = task_plan(seed, task, n_labels, per_class)
    images, paths = [], []
    for cls in plan['classes']:
        array, names = source.train(cls, plan['first'], plan['count'])
        images.append(array)
        paths.extend(names)
    images = np.concatenate(images)
    return TaskData(images, task_labels(len(images), seed, task, n_labels), paths, augment), plan


def first_epoch_sizes(n):
    """Distinct batch sizes in the order task 1 first meets them (compile replay, spec §4.1)."""
    sizes = [min(BATCH, n)]
    if n > BATCH and n % BATCH:
        sizes.append(n % BATCH)
    return sizes


def eval_tensor(images, device):
    return torch.stack([transform(a, None) for a in images]).to(device)


def probe_names(source, seed):
    names = source.val_names()
    assert len(names) == 10000, len(names)
    order = torch.randperm(len(names), generator=torch.Generator().manual_seed(stream_seed('probe', seed)))
    return [names[i] for i in order[:PROBE_IMAGES].tolist()]


def readout_indices(n):
    return np.linspace(0, n - 1, READOUT_IMAGES, dtype=int).tolist()


# --------------------------------------------------------------------------- model

DROPOUT = CONFIG['dropout_p']           # 0.1, also the attention dropout (attn_drop_rate)


def model_config(dropout=DROPOUT, ln=False, depth=DEPTH, mlp_ratio=MLP_RATIO):
    """The N cell's config (ln=True: the randlabel cell's, affine LayerNorm); --dropout overrides
    both dropout_p and attn_drop_rate (no RNG at build, so initial weights do not change);
    --depth and --mlp-ratio change the number of blocks and the FFN width (addendum A)."""
    base = CONFIG_LN if ln else CONFIG
    if dropout == base['dropout_p'] == base['attn_drop_rate'] and depth == base['depth'] \
            and mlp_ratio == base['mlp_ratio']:
        return base
    return dict(base, dropout_p=dropout, attn_drop_rate=dropout, depth=depth, mlp_ratio=mlp_ratio)


def build_model(seed, device, dropout=DROPOUT, ln=False, depth=DEPTH, mlp_ratio=MLP_RATIO):
    return make_model(ACT, seed, device, model_config(dropout, ln, depth, mlp_ratio))


def n_params_expected(depth, mlp_ratio, ln, embed=384, patch=8, tokens=65, classes=200):
    """Parameter count of the upstream ViT for a depth and FFN ratio (S-model)."""
    width = int(embed * mlp_ratio)
    block = (embed * 3 * embed + 3 * embed) + (embed * embed + embed) + (embed * width + width) + (width * embed + embed)
    total = (3 * patch * patch * embed + embed) + embed + tokens * embed + depth * block + (embed * classes + classes)
    return total + ((2 * depth + 1) * 2 * embed if ln else 0)


def build_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=LR, betas=(.9, .999), eps=1e-8,
                            weight_decay=0., fused=True)


def fc1(model, i):
    return model.layers[f'block_{i}'].layers['mlp'].layers['fc1']


def set_determinism(threads=4):
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


@torch.no_grad()
def trace(model, x):
    """Eval-mode forward in the upstream order, block by block, without hooks.

    Hooks or attribute changes would alter the compiled forward's guards; calling the
    submodules in the order of `VisionTransformer.forward` and `TransformerBlock.forward`
    returns the same logits as `model(x)` bit for bit (checked in S-model).
    Returns logits and, per block, (FFN input, fc1 pre-activation).
    """
    if model.training:
        raise RuntimeError('trace is the eval-mode forward; call model.eval() first')
    layers = model.layers
    h = layers['patch_embed'](x)
    h = torch.cat((model.cls_token.expand(h.shape[0], -1, -1), h), dim=1)
    h = h + model.pos_embed
    h = layers['pos_drop'](h)
    blocks = []
    for i in range(model.depth):
        block = layers[f'block_{i}'].layers
        h = h + block['attn'](block['norm1'](h))
        u = block['norm2'](h)
        mlp = block['mlp'].layers
        z = mlp['fc1'](u)
        f = mlp['drop2'](mlp['fc2'](mlp['drop1'](mlp['act'](z))))
        h = h + f
        blocks.append((u, z))
    logits = layers['out'](layers['norm'](h)[:, 0])
    return logits, blocks


def _median(x):
    return float(torch.quantile(x.double(), .5)) if x.numel() else math.nan


@torch.no_grad()
def layer_summary(model, x_probe):
    """Per-FFN summary on the fixed probe (spec §6.3). Leaves the train/eval mode as found."""
    was_training = model.training
    model.eval()
    _, blocks = trace(model, x_probe)
    rows = []
    for i, (u, z) in enumerate(blocks):
        Z = z.reshape(-1, z.shape[-1]).double()
        p = (Z > 0).double().mean(0)
        mu, sd = Z.mean(0), Z.std(0, unbiased=True)
        cls = z[:, 0, :].double()
        pc = (cls > 0).double().mean(0)
        muc, sdc = cls.mean(0), cls.std(0, unbiased=True)
        layer = fc1(model, i)
        norms, bias = layer.weight.double().norm(dim=1), layer.bias.double()
        X = u.reshape(-1, u.shape[-1]).double()
        m = X.mean(0)
        C = torch.cov(X.T)
        rows.append(dict(
            layer=i + 1, dead005=float((p < .005).double().mean()), dead0=float((p == 0).double().mean()),
            open99=float((p >= .99).double().mean()), neg_mass=float((Z < 0).double().mean()),
            mu_med=_median(mu), sigma_med=_median(sd), r_med=_median(mu / (sd + 1e-12)),
            cls_dead0=float((pc == 0).double().mean()), cls_mu_med=_median(muc),
            cls_sigma_med=_median(sdc), cls_r_med=_median(muc / (sdc + 1e-12)),
            up_norm_med=_median(norms), up_norm_med_live=_median(norms[p > 0]),
            bias_med=_median(bias), bias_neg=float((bias < 0).double().mean()),
            in_mean_norm=float(m.norm()), in_mean_norm_per_dim=float(m.norm() / math.sqrt(X.shape[1])),
            in_mean_frac=float(m.pow(2).sum() / X.pow(2).sum(1).mean()),
            in_pr=float(C.trace() ** 2 / C.pow(2).sum())))
    model.train(was_training)
    return rows


@torch.no_grad()
def factorial_readout(model, x16):
    """`randlabel_control_joudaki_vit_0919.readout`'s two numbers, per layer, on 16 task images."""
    was_training = model.training
    model.eval()
    _, blocks = trace(model, x16)
    rows = []
    for i, (_, z) in enumerate(blocks):
        z = z.reshape(-1, z.shape[-1]).float()
        rows.append(dict(layer=i + 1, readout_neg_mass=float((z < 0).float().mean()),
                         readout_dead=float((z.max(0).values <= 0).float().mean())))
    model.train(was_training)
    return rows


# --------------------------------------------------------------------------- provenance

def source_files():
    paths = sorted((REPO / 'analysis/vit_long_1024').glob('*.py')) + [PRODUCTION_FILE]
    paths += sorted((REPO / 'analysis/joudaki_vit_battle_0919').rglob('*.py'))
    paths += [REPO / 'analysis/randlabel_control_joudaki_vit_0919.py',
              REPO / 'analysis/factorial_joudaki_vit_0919.py',
              REPO / 'src/rlcifar_mlp_battle_0918.py', REPO / 'src/pmnist_0905.py',
              REPO / 'src/pmnist_rlcifar_0907.py', REPO / 'src/pmnist_rlmnist_0906.py']
    return paths


def source_hashes():
    return {str(p.relative_to(REPO)): sha256(p) for p in source_files()}


def git_state():
    """(commit, dirty). A copy sent to the rented GPU has no .git: it carries SOURCE_COMMIT and
    SOURCE_HASHES.json written by remote/pack_for_remote.sh, and is clean iff the hashes match."""
    if (REPO / '.git').exists():
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'src', 'analysis', 'specs'],
                                        cwd=REPO, text=True)
        return head, dirty
    head = (REPO / 'SOURCE_COMMIT').read_text().strip()
    packed = json.loads((REPO / 'SOURCE_HASHES.json').read_text())
    now = source_hashes()
    changed = sorted(k for k in set(packed) | set(now) if packed.get(k) != now.get(k))
    return head, ''.join(f'changed since pack: {k}\n' for k in changed)


def state_sha256(state_dict):
    """One digest over a state_dict's tensors in key order (bit-level identity)."""
    h = hashlib.sha256()
    for key, value in state_dict.items():
        h.update(key.encode())
        if isinstance(value, torch.Tensor):
            t = value.detach().cpu().contiguous()
            h.update(str(t.dtype).encode())
            h.update(str(tuple(t.shape)).encode())
            h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b'')
        else:
            h.update(repr(value).encode())
    return h.hexdigest()


def array_sha256(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.tobytes() + str(array.dtype).encode() + str(array.shape).encode()).hexdigest()
