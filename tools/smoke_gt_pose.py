# Sanity checks for the oracle GT-pose pipeline (run on the VM, needs KITTI raw).
#
#   python tools/smoke_gt_pose.py --data_path /path/to/kitti_raw
#
# Checks, per sampled training frame:
#   - rotation blocks are orthonormal
#   - ||translation|| of the +1/-1 relative poses matches the oxts forward
#     speed * 0.1s (KITTI runs at 10 Hz)
#   - the flipped pose is the mirror conjugation of the unflipped one
from __future__ import absolute_import, division, print_function

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import KITTIRAWDataset
from kitti_utils import read_oxts_packet
from utils import readlines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--split", default="eigen_zhou")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args()

    fpath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "splits", args.split, "train_files.txt")
    filenames = readlines(fpath)[::max(1, len(readlines(fpath)) // args.num_samples)]
    filenames = filenames[:args.num_samples]

    img_ext = '.png' if args.png else '.jpg'
    dataset = KITTIRAWDataset(args.data_path, filenames, 192, 640, [0, -1, 1], 1,
                              is_train=False, img_ext=img_ext, load_gt_pose=True)

    max_speed_err = 0.0
    for idx, line in enumerate(filenames):
        folder, frame_index, side = line.split()
        frame_index = int(frame_index)

        for offset in (-1, 1):
            T = dataset.get_pose(folder, frame_index, offset, side, do_flip=False)
            R, t = T[:3, :3], T[:3, 3]

            ortho_err = np.abs(R.dot(R.T) - np.eye(3)).max()
            assert ortho_err < 1e-5, f"{line}: rotation not orthonormal ({ortho_err})"

            packet = read_oxts_packet(os.path.join(
                args.data_path, folder, "oxts/data/{:010d}.txt".format(frame_index)))
            vf = packet[8]  # forward velocity, m/s
            expected = abs(vf) * 0.1
            err = abs(np.linalg.norm(t) - expected)
            max_speed_err = max(max_speed_err, err)

            T_flip = dataset.get_pose(folder, frame_index, offset, side, do_flip=True)
            F = np.diag([-1., 1., 1., 1.]).astype(np.float32)
            assert np.abs(F.dot(T).dot(F) - T_flip).max() < 1e-5, f"{line}: flip mismatch"

            if idx < 3:
                print(f"{line} offset {offset:+d}: ||t|| = {np.linalg.norm(t):.3f} m, "
                      f"speed*0.1s = {expected:.3f} m, t = {np.round(t, 3)}")

    print(f"\nChecked {len(filenames)} frames x 2 offsets.")
    print(f"Max | ||t|| - speed*dt | = {max_speed_err:.3f} m "
          "(should be small, ~cm; lateral/vertical motion adds a little)")

    # full dataloader item, makes sure ("gt_pose", f) tensors come through
    item = dataset[0]
    for f in (-1, 1):
        assert ("gt_pose", f) in item, "gt_pose missing from dataset item"
        print(f"dataset[0] gt_pose {f:+d}:\n{item[('gt_pose', f)].numpy().round(4)}")
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
