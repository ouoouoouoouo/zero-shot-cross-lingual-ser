"""Batch samplers over the TRAINING rows only (they are handed plan.train)."""
from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class HierarchicalBatchSampler(Sampler):
    """Sec. 2.2: pick N_lang languages, then N_cls emotion classes, then N_sam
    utterances per (language, class); pairs with < N_sam items draw with
    replacement. Batch size is N_lang * N_cls * N_sam."""

    def __init__(self, rows, n_lang, n_cls, n_sam, num_batches, seed=0):
        self.pools = defaultdict(list)
        for i, r in enumerate(rows):
            self.pools[(r["lang"], r["label"])].append(i)
        self.langs = sorted({l for l, _ in self.pools})
        self.labels = sorted({c for _, c in self.pools})
        if len(self.langs) < 2:
            raise ValueError("hierarchical sampling needs >= 2 training languages")
        self.n_lang, self.n_cls, self.n_sam = min(n_lang, len(self.langs)), min(n_cls, len(self.labels)), n_sam
        self.num_batches, self.seed, self.epoch = num_batches, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        for _ in range(self.num_batches):
            batch = []
            for lang in rng.choice(self.langs, self.n_lang, replace=False):
                for cls in rng.choice(self.labels, self.n_cls, replace=False):
                    pool = self.pools.get((lang, cls), [])
                    if pool:
                        batch += rng.choice(pool, self.n_sam, replace=len(pool) < self.n_sam).tolist()
            yield batch


class RandomBatchSampler(Sampler):
    """Plain shuffled mini-batches (baselines / upper bound)."""

    def __init__(self, n_rows, batch_size, seed=0):
        self.n, self.bs, self.seed, self.epoch = n_rows, batch_size, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return max(1, self.n // self.bs)

    def __iter__(self):
        perm = np.random.default_rng((self.seed, self.epoch)).permutation(self.n)
        for b in range(len(self)):
            yield perm[b * self.bs:(b + 1) * self.bs].tolist()
