# Foundation depth model (Depth Anything V2 via HuggingFace) wrapped with LoRA
# adapters and a learnable global scale/shift that maps its affine-invariant
# relative output to metric inverse depth.
#
# lora_rank = 0 degenerates to the "global affine rescaling" baseline: the
# backbone stays fully frozen and only scale/shift are trained.
from __future__ import absolute_import, division, print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# the relative-depth median of the first batch is anchored to this depth (m),
# so training starts from a roughly correct metric scale
ANCHOR_DEPTH = 15.0


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank residual."""

    def __init__(self, base, rank, alpha=None):
        super(LoRALinear, self).__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = (alpha if alpha is not None else rank) / rank

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling


def inject_lora(module, rank, alpha=None, target_names=("query", "value")):
    """Recursively replace attention q/v Linears with LoRA-wrapped versions.
    Returns the number of layers replaced."""
    replaced = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name in target_names:
            setattr(module, name, LoRALinear(child, rank, alpha))
            replaced += 1
        else:
            replaced += inject_lora(child, rank, alpha, target_names)
    return replaced


class FoundationDepthAdapter(nn.Module):
    """Depth Anything V2 + LoRA + learnable metric scale/shift.

    forward(x): x is (B,3,H,W) in [0,1]; returns metric depth (B,1,H,W).
    """

    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf",
                 lora_rank=8, lora_alpha=None, train_head=False,
                 min_depth=0.1, max_depth=80.0):
        super(FoundationDepthAdapter, self).__init__()
        from transformers import AutoModelForDepthEstimation
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.min_depth = min_depth
        self.max_depth = max_depth

        for p in self.model.parameters():
            p.requires_grad = False

        self.num_lora_layers = 0
        if lora_rank > 0:
            self.num_lora_layers = inject_lora(self.model.backbone, lora_rank, lora_alpha)
            assert self.num_lora_layers > 0, \
                "no query/value Linear layers found to wrap with LoRA"

        if train_head:
            for p in self.model.head.parameters():
                p.requires_grad = True

        # metric inverse depth = exp(log_scale) * rel + shift
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.shift = nn.Parameter(torch.zeros(1))
        self.register_buffer("calibrated", torch.tensor(False))

        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("pix_mean", mean)
        self.register_buffer("pix_std", std)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def adapter_state_dict(self):
        """Only what adaptation changed: LoRA, scale/shift (+ head if trained)."""
        state = {n: p for n, p in self.named_parameters() if p.requires_grad}
        state["calibrated"] = self.calibrated
        return state

    def load_adapter_state_dict(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        assert not unexpected, "unexpected keys in adapter checkpoint: {}".format(unexpected)

    def _calibrate(self, rel):
        med = rel.detach().median().clamp(min=1e-6)
        self.log_scale.data.fill_(math.log(1.0 / (ANCHOR_DEPTH * float(med))))
        self.calibrated.fill_(True)
        print("-> calibrated foundation scale: median rel {:.4f} anchored to {:.1f} m".format(
            float(med), ANCHOR_DEPTH))

    def forward(self, x):
        b, _, h, w = x.shape
        # DINOv2 needs multiples of the 14px patch size
        h14 = max(14, int(round(h / 14.)) * 14)
        w14 = max(14, int(round(w / 14.)) * 14)
        inp = (x - self.pix_mean) / self.pix_std
        if (h14, w14) != (h, w):
            inp = F.interpolate(inp, (h14, w14), mode="bilinear", align_corners=False)

        rel = self.model(pixel_values=inp).predicted_depth  # (B, h', w'), bigger = closer
        rel = rel.unsqueeze(1)

        if not bool(self.calibrated):
            self._calibrate(rel)

        idepth = torch.exp(self.log_scale) * rel + self.shift
        idepth = idepth.clamp(1.0 / self.max_depth, 1.0 / self.min_depth)
        depth = 1.0 / idepth

        if depth.shape[-2:] != (h, w):
            depth = F.interpolate(depth, (h, w), mode="bilinear", align_corners=False)
        return depth
