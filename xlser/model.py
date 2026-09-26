"""wav2vec 2.0 + parameter-efficient fine-tuning + emotion / adversarial speaker heads.

The paper fine-tunes the encoder with LoRA [29], bottleneck adapters [30] and
weight gating [31] without further detail. Our reading: the backbone is frozen;
LoRA is added to the attention q/v projections, a Houlsby bottleneck adapter
follows every transformer feed-forward block, and each LoRA / adapter branch is
scaled by a learnable sigmoid gate.
"""
import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2Model


class GatedLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank, alpha):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.base(x) + torch.sigmoid(self.gate) * self.scale * self.up(self.down(x))


class GatedAdapterFFN(nn.Module):
    def __init__(self, ffn: nn.Module, dim, bottleneck):
        super().__init__()
        self.ffn = ffn
        self.adapter = nn.Sequential(nn.Linear(dim, bottleneck), nn.GELU(), nn.Linear(bottleneck, dim))
        nn.init.zeros_(self.adapter[2].weight)
        nn.init.zeros_(self.adapter[2].bias)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        y = self.ffn(x)
        return y + torch.sigmoid(self.gate) * self.adapter(y)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.coeff * g, None


def load_backbone(name):
    """`name` is a HF hub id / local dir, or "tiny-random" (tests / smoke runs)."""
    if name == "tiny-random":
        cfg = Wav2Vec2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                             conv_dim=(32, 32), conv_stride=(5, 4), conv_kernel=(10, 4),
                             num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4)
        model = Wav2Vec2Model(cfg)
    else:
        model = Wav2Vec2Model.from_pretrained(name)
    # identical treatment for every backbone: no SpecAugment, no LayerDrop
    model.config.apply_spec_augment = False
    model.config.layerdrop = 0.0
    return model


class SERModel(nn.Module):
    def __init__(self, backbone, n_labels, n_speakers=0, lora_rank=8, lora_alpha=16, adapter_dim=64,
                 spk_hidden=256, spk_dropout=0.1, grl_coeff=1.0):
        super().__init__()
        self.backbone = backbone if isinstance(backbone, nn.Module) else load_backbone(backbone)
        for p in self.backbone.parameters():
            p.requires_grad = False
        d = self.backbone.config.hidden_size
        for layer in self.backbone.encoder.layers:
            att = layer.attention
            att.q_proj = GatedLoRALinear(att.q_proj, lora_rank, lora_alpha)
            att.v_proj = GatedLoRALinear(att.v_proj, lora_rank, lora_alpha)
            layer.feed_forward = GatedAdapterFFN(layer.feed_forward, d, adapter_dim)
        self.emotion_head = nn.Linear(d, n_labels)
        self.speaker_head = None
        if n_speakers:
            self.speaker_head = nn.Sequential(nn.Linear(d, spk_hidden), nn.ReLU(), nn.Dropout(spk_dropout),
                                              nn.Linear(spk_hidden, n_speakers))
        self.grl_coeff = grl_coeff

    def trainable_state_dict(self):
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if k in names}

    def forward(self, wav, lengths):
        cfg = self.backbone.config
        attn = None
        if cfg.feat_extract_norm == "layer":  # group-norm checkpoints (e.g. base-960h) expect no mask
            attn = (torch.arange(wav.shape[1], device=wav.device)[None] < lengths[:, None]).long()
        H = self.backbone(wav, attention_mask=attn).last_hidden_state               # (B, n, d)
        n_frames = self.backbone._get_feat_extract_output_lengths(lengths).clamp(1, H.shape[1])
        mask = (torch.arange(H.shape[1], device=H.device)[None] < n_frames[:, None]).unsqueeze(-1)
        h = (H * mask).sum(1) / mask.sum(1)                                         # Eq. (4)
        out = {"h": h, "emotion": self.emotion_head(h)}
        if self.speaker_head is not None:
            out["speaker"] = self.speaker_head(GradReverse.apply(h, self.grl_coeff))  # Eq. (9)
        return out
