"""wav2vec 2.0 + parameter-efficient fine-tuning + emotion / adversarial speaker heads.

The paper fine-tunes the encoder with LoRA [29], bottleneck adapters [30] and
weight gating [31] and builds its baselines "as in [14]" (Han et al., ICASSP
2025), whose code is public. The PEFT below follows that code
(github.com/zhan7721/Crosslingual_SER, utils.py / runner_various_peft.py):

- the encoder is truncated after `feature_layer` transformer layers and the
  mean-pooled output of that layer is the utterance embedding h;
- LoRA (r 8, alpha 16, dropout 0.1) on q / k / v / out projections and both
  feed-forward linears of every kept layer;
- a bottleneck adapter (768 -> 48 -> 768, ReLU) in parallel with each
  feed-forward block: h + BA(FF(h)) + FF(h);
- weight gating: each layer's input is scaled by sigmoid(g), g in R^768, g = 1 at init.

The emotion head is a single linear layer, as in the paper's Eq. (11).
"""
import torch
import torch.nn as nn
from transformers import Wav2Vec2Config, Wav2Vec2Model
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2EncoderLayer


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank, alpha, dropout):
        super().__init__()
        self.base = base
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + self.scale * self.lora_B(self.lora_A(self.dropout(x)))


class PEFTEncoderLayer(nn.Module):
    """Wav2Vec2EncoderLayer (post-LN) with weight gating and a parallel bottleneck adapter."""

    def __init__(self, layer: Wav2Vec2EncoderLayer, hidden, adapter_dim, lora_rank, lora_alpha, lora_dropout):
        super().__init__()
        if type(layer) is not Wav2Vec2EncoderLayer:
            raise TypeError(f"expected a post-LN Wav2Vec2EncoderLayer, got {type(layer).__name__}")
        att, ff = layer.attention, layer.feed_forward
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(att, name, LoRALinear(getattr(att, name), lora_rank, lora_alpha, lora_dropout))
        for name in ("intermediate_dense", "output_dense"):
            setattr(ff, name, LoRALinear(getattr(ff, name), lora_rank, lora_alpha, lora_dropout))
        self.layer = layer
        self.weighted_gate = nn.Parameter(torch.ones(hidden))
        self.bottleneck = nn.Sequential(nn.Linear(hidden, adapter_dim), nn.ReLU(), nn.Linear(adapter_dim, hidden))

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        l = self.layer
        hidden_states = torch.sigmoid(self.weighted_gate) * hidden_states
        residual = hidden_states
        hidden_states, _ = l.attention(hidden_states, attention_mask=attention_mask, **kwargs)
        hidden_states = residual + l.dropout(hidden_states)
        hidden_states = l.layer_norm(hidden_states)
        ff_out = l.feed_forward(hidden_states)
        hidden_states = hidden_states + self.bottleneck(ff_out) + ff_out
        return l.final_layer_norm(hidden_states)


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
    def __init__(self, backbone, n_labels, n_speakers=0, feature_layer=12, lora_rank=8, lora_alpha=16,
                 lora_dropout=0.1, adapter_dim=48, spk_hidden=256, spk_dropout=0.1, grl_coeff=1.0):
        super().__init__()
        self.backbone = backbone if isinstance(backbone, nn.Module) else load_backbone(backbone)
        for p in self.backbone.parameters():
            p.requires_grad = False
        n_layers = len(self.backbone.encoder.layers)
        if not 1 <= feature_layer <= n_layers:
            raise ValueError(f"feature_layer {feature_layer} outside 1..{n_layers}")
        self.feature_layer = feature_layer
        d = self.backbone.config.hidden_size
        self.backbone.encoder.layers = nn.ModuleList(
            PEFTEncoderLayer(layer, d, adapter_dim, lora_rank, lora_alpha, lora_dropout)
            for layer in self.backbone.encoder.layers[:feature_layer])
        self.backbone.config.num_hidden_layers = feature_layer
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
        H = self.backbone(wav, attention_mask=attn).last_hidden_state               # (B, n, d), layer `feature_layer`
        n_frames = self.backbone._get_feat_extract_output_lengths(lengths).clamp(1, H.shape[1])
        mask = (torch.arange(H.shape[1], device=H.device)[None] < n_frames[:, None]).unsqueeze(-1)
        h = (H * mask).sum(1) / mask.sum(1)                                         # Eq. (4)
        out = {"h": h, "emotion": self.emotion_head(h)}
        if self.speaker_head is not None:
            out["speaker"] = self.speaker_head(GradReverse.apply(h, self.grl_coeff))  # Eq. (9)
        return out
