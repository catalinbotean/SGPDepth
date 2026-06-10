"""Smoke test for the strengthened pose network (PoseCNNAttn).

Run on the training VM (needs torch + timm):

    python tools/smoke_pose_attn.py

Verifies forward-pass shapes for the pairs setting and the
uncertainty-weighted loss broadcast, plus the iterative-refinement path.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from networks import PoseCNNAttn

B, H, W = 2, 192, 640


def check(name, t, shape):
    assert tuple(t.shape) == shape, f"{name}: got {tuple(t.shape)}, want {shape}"
    print(f"  ok  {name:18s} {tuple(t.shape)}")


def run(n_iters, uncertainty):
    print(f"\n== n_iters={n_iters} uncertainty={uncertainty} ==")
    net = PoseCNNAttn(num_input_frames=2, enc_name="resnet18",
                      pretrained=False, uncertainty=uncertainty,
                      n_iters=n_iters)
    x = torch.randn(B, 6, H, W)  # two RGB frames concatenated
    axisangle, translation, log_var = net(x, return_logvar=True)
    check("axisangle", axisangle, (B, 1, 1, 3))
    check("translation", translation, (B, 1, 1, 3))
    if uncertainty:
        check("log_var", log_var, (B, 1, 1, 1))
        # broadcast against a (B,1,H,W) reprojection map
        rep = torch.rand(B, 1, H, W)
        weighted = torch.exp(-log_var) * rep
        check("weighted rep", weighted, (B, 1, H, W))
    else:
        assert log_var is None, "log_var should be None when uncertainty=False"
        print("  ok  log_var            None")

    # gradients flow
    loss = axisangle.abs().mean() + translation.abs().mean()
    if uncertainty:
        loss = loss + log_var.mean()
    loss.backward()
    print("  ok  backward()")


if __name__ == "__main__":
    run(n_iters=1, uncertainty=True)
    run(n_iters=3, uncertainty=True)
    run(n_iters=1, uncertainty=False)
    print("\nALL SMOKE TESTS PASSED")
