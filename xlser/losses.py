import torch
import torch.nn.functional as F


def language_aware_supcon(h, labels, langs, temperature=0.07, lam=2.5):
    """Eq. (8). h: (B, d); labels: (B,); langs: (B,) integer language ids.
    Same-emotion pairs from a different language get weight `lam`, same-language
    pairs weight 1. Anchors without any positive are skipped."""
    z = F.normalize(h, dim=-1)
    sim = z @ z.t() / temperature
    self_mask = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)          # denominator over A(i)
    pos = (labels[:, None] == labels[None, :]) & ~self_mask               # P(i)
    w = torch.where(langs[:, None] != langs[None, :], lam, 1.0) * pos
    has_pos = pos.any(dim=1)
    if not has_pos.any():
        return h.sum() * 0.0
    per_anchor = -(w * log_prob.masked_fill(~pos, 0.0)).sum(1) / w.sum(1).clamp_min(1e-12)
    return per_anchor[has_pos].mean()
