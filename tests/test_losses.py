import math

import torch
import torch.nn.functional as F

from xlser.losses import language_aware_supcon


def naive_eq8(h, y, g, tau, lam):
    z = F.normalize(h, dim=-1)
    n, total, anchors = len(z), 0.0, 0
    for i in range(n):
        P = [p for p in range(n) if p != i and y[p] == y[i]]
        if not P:
            continue
        denom = sum(math.exp(float(z[i] @ z[a]) / tau) for a in range(n) if a != i)
        w = {p: (lam if g[p] != g[i] else 1.0) for p in P}
        total += -sum(w[p] * math.log(math.exp(float(z[i] @ z[p]) / tau) / denom) for p in P) / sum(w.values())
        anchors += 1
    return total / anchors


def test_matches_equation_8():
    torch.manual_seed(0)
    h = torch.randn(12, 8)
    y = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1, 2, 3, 3, 0])
    g = torch.tensor([0, 1, 0, 1, 0, 1, 2, 2, 2, 0, 1, 1])
    for lam in (1.0, 2.5):
        got = language_aware_supcon(h, y, g, temperature=0.5, lam=lam).item()
        assert abs(got - naive_eq8(h, y, g, 0.5, lam)) < 1e-4


def test_cross_lingual_pairs_weigh_more():
    # anchor 0 (lang 0) is close to its same-language positive, far from its cross-lingual one:
    # a larger lambda must increase the loss.
    h = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]])
    y = torch.tensor([0, 0, 0, 1])
    g = torch.tensor([0, 0, 1, 0])
    assert language_aware_supcon(h, y, g, 0.1, 2.5) > language_aware_supcon(h, y, g, 0.1, 1.0)
