from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class PoseCNNAttn(nn.Module):
    """Strengthened relative-pose network.

    Drop-in replacement for the timm-based ``PoseCNN`` used in SPIdepth, with
    three additions that target the pose bottleneck of self-supervised depth:

    C1 (attention pooling): instead of collapsing the spatial pose feature map
        with a global average (``out.mean(3).mean(2)``), we predict a *dense*
        per-location 6-DoF field and a per-source-frame spatial attention map,
        then pool the field with attention weights. This lets the network
        suppress independently-moving regions (cars, pedestrians) before
        committing to a single ego-motion estimate.

    C2 (heteroscedastic uncertainty): a small head emits a per-source-frame
        log-variance ``log_var``. The trainer uses it to down-weight the
        photometric loss where pose is unreliable (uncertainty-weighted
        reprojection), and the ``+ log_var`` regulariser stops it collapsing.

    C3 (iterative refinement, optional): when ``n_iters > 1`` the pose estimate
        is refined for a few residual steps via a lightweight GRU-style update
        conditioned on the pooled context, loosely echoing RAFT/DROID iterative
        updates but kept tiny so it is nearly free.

    The interface mirrors ``PoseCNN``: it consumes channel-wise concatenated
    frames ``(b, 3*n_imgs, h, w)`` and returns ``(axisangle, translation)`` of
    shape ``(b, n_imgs-1, 1, 3)``. When ``return_logvar`` is requested it also
    returns ``log_var`` of shape ``(b, n_imgs-1, 1, 1)``.

    :param num_input_frames: number of frames concatenated on the channel dim.
    :param enc_name: ``timm`` encoder key.
    :param pretrained: load ImageNet weights for the encoder.
    :param uncertainty: if ``True`` build the uncertainty head and return log_var.
    :param n_iters: number of iterative refinement steps (1 == single pass).
    """

    def __init__(self, num_input_frames, enc_name: str = 'resnet18',
                 pretrained: bool = True, uncertainty: bool = True,
                 n_iters: int = 1):
        super().__init__()
        self.enc_name = enc_name
        self.pretrained = pretrained
        self.n_imgs = num_input_frames
        self.n_preds = self.n_imgs - 1          # poses predicted per forward
        self.uncertainty = uncertainty
        self.n_iters = max(1, int(n_iters))

        self.encoder = timm.create_model(
            enc_name, in_chans=3 * self.n_imgs, features_only=True,
            pretrained=pretrained)
        self.n_chenc = self.encoder.feature_info.channels()
        ctx_ch = 256

        self.squeeze = nn.Sequential(
            nn.Conv2d(self.n_chenc[-1], ctx_ch, 1), nn.ReLU(inplace=True))

        # Shared trunk over the spatial feature map.
        self.trunk = nn.Sequential(
            self._block(ctx_ch, ctx_ch, 3, 1, 1),
            self._block(ctx_ch, ctx_ch, 3, 1, 1),
        )

        # C1: dense per-location 6-DoF field, one 6-vector per predicted pose.
        self.pose_field = nn.Conv2d(ctx_ch, 6 * self.n_preds, 1)
        # C1: spatial attention logits, one map per predicted pose.
        self.attn_logits = nn.Conv2d(ctx_ch, self.n_preds, 1)

        # C2: per-source-frame log-variance from the globally pooled context.
        if self.uncertainty:
            self.unc_head = nn.Sequential(
                nn.Linear(ctx_ch, ctx_ch // 2), nn.ReLU(inplace=True),
                nn.Linear(ctx_ch // 2, self.n_preds))

        # C3: GRU-style residual refinement on the pooled context -> pose delta.
        if self.n_iters > 1:
            self.gru = nn.GRUCell(6 * self.n_preds, ctx_ch)
            self.delta_head = nn.Linear(ctx_ch, 6 * self.n_preds)

    @staticmethod
    def _block(in_ch, out_ch, k, s=1, p=0):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor, return_logvar: bool = False):
        b = x.shape[0]
        feat = self.encoder(x)[-1]
        ctx = self.squeeze(feat)                 # (b, C, h, w)
        h = self.trunk(ctx)

        # --- C1: attention-pooled pose ---------------------------------------
        field = self.pose_field(h)               # (b, 6*n_preds, h, w)
        field = field.view(b, self.n_preds, 6, -1)            # (b, n, 6, hw)
        attn = self.attn_logits(h).view(b, self.n_preds, -1)  # (b, n, hw)
        attn = torch.softmax(attn, dim=-1).unsqueeze(2)       # (b, n, 1, hw)
        pose = (field * attn).sum(-1)            # (b, n, 6) weighted pooling

        # Pooled global context for uncertainty / refinement heads.
        gctx = h.mean(dim=(2, 3))                 # (b, C)

        # --- C3: iterative residual refinement -------------------------------
        if self.n_iters > 1:
            state = gctx
            for _ in range(self.n_iters - 1):
                state = self.gru(pose.reshape(b, -1), state)
                delta = self.delta_head(state).view(b, self.n_preds, 6)
                pose = pose + 0.01 * delta

        pose = 0.01 * pose                       # match PoseCNN output scaling
        pose = pose.view(b, self.n_preds, 1, 6)
        axisangle = pose[..., :3]
        translation = pose[..., 3:]

        if self.uncertainty:
            log_var = self.unc_head(gctx).view(b, self.n_preds, 1, 1)
            if return_logvar:
                return axisangle, translation, log_var
            return axisangle, translation, log_var
        # Keep a stable 3-tuple contract; log_var is None when disabled.
        if return_logvar:
            return axisangle, translation, None
        return axisangle, translation, None
