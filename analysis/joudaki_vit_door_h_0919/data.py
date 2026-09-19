"""Tiny ImageNet with paired class order, batches and augmentation across arms."""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset, DataLoader

from .model import stream_seed

MEAN = torch.tensor([.485, .456, .406])[:, None, None]
STD = torch.tensor([.229, .224, .225])[:, None, None]


def task_classes(seed):
    classes = list(range(200))
    random.Random(seed).shuffle(classes)
    return [classes[i:i+5] for i in range(0, 200, 5)]


def transform(array, seed=None):
    image = Image.fromarray(array)
    if seed is not None:
        rng = random.Random(seed)
        image = ImageOps.expand(image, border=8, fill=0)
        top, left = rng.randrange(17), rng.randrange(17)
        image = image.crop((left, top, left + 64, top + 64))
        if rng.random() < .5:
            image = ImageOps.mirror(image)
        factors = [rng.uniform(.9, 1.1) for _ in range(3)]
        operations = [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color]
        order = list(range(3))
        rng.shuffle(order)
        for j in order:
            image = operations[j](image).enhance(factors[j])
    x = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float().div_(255)
    return (x - MEAN) / STD


class TaskImages(Dataset):
    def __init__(self, root, classes, split):
        root = Path(root)
        synsets = sorted((root / 'wnids.txt').read_text().split())
        assert len(synsets) == 200
        self.paths, labels = [], []
        if split == 'train':
            for local, cls in enumerate(classes):
                paths = sorted((root / 'train' / synsets[cls] / 'images').glob('*.JPEG'))
                assert len(paths) == 500, (synsets[cls], len(paths))
                self.paths.extend(paths)
                labels.extend([local] * len(paths))
        else:
            mapping = {synsets[c]: j for j, c in enumerate(classes)}
            annotations = sorted(line.split('\t')[:2] for line in
                                 (root / 'val/val_annotations.txt').read_text().splitlines())
            for file, cls in annotations:
                if cls in mapping:
                    self.paths.append(root / 'val/images' / file)
                    labels.append(mapping[cls])
            assert len(labels) == 250
        self.images = np.stack([np.array(Image.open(p).convert('RGB')) for p in self.paths])
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        idx, aug_seed = item if isinstance(item, tuple) else (item, None)
        return transform(self.images[idx], aug_seed), self.labels[idx]


class PairedBatches:
    def __init__(self, n, batch, steps, seed, task):
        self.n, self.batch, self.steps, self.seed, self.task = n, batch, steps, seed, task

    def __len__(self):
        return self.steps

    def __iter__(self):
        gen = torch.Generator().manual_seed(stream_seed('batch', self.seed, self.task))
        step = 0
        while step < self.steps:
            order = torch.randperm(self.n, generator=gen).tolist()
            for start in range(0, self.n, self.batch):
                if step >= self.steps:
                    return
                yield [(idx, stream_seed('augment', self.seed, self.task, step, idx))
                       for idx in order[start:start + self.batch]]
                step += 1


def train_loader(dataset, seed, task, steps=500, workers=2):
    # A private loader RNG prevents worker setup from consuming the dropout RNG.
    generator = torch.Generator().manual_seed(stream_seed('loader', seed, task))
    return DataLoader(dataset, batch_sampler=PairedBatches(len(dataset), 128, steps, seed, task),
                      num_workers=workers, pin_memory=True, generator=generator)


def eval_loader(dataset, workers=2):
    return DataLoader(dataset, batch_size=128, num_workers=workers, pin_memory=True,
                      generator=torch.Generator().manual_seed(123456))
