# Eigen-split evaluation for the adapted foundation depth model.
# Reports metrics BOTH raw-metric (no scaling -- the headline number) and
# median-scaled (for comparison against the self-supervised literature).
#
#   python tools/evaluate_adapt.py args_files/hisfog/kitti/eval_adapt_dav2_kitti.txt
#
# Pass --load_weights_folder pointing at a weights_<n> folder with adapt.pth,
# or omit adapt.pth to evaluate the raw (calibration-only) foundation model.
from __future__ import absolute_import, division, print_function

import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets
from networks.foundation_depth import FoundationDepthAdapter
from options import MonodepthOptions
from utils import readlines

cv2.setNumThreads(0)

splits_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "splits")

MIN_DEPTH = 1e-3
MAX_DEPTH = 80


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def evaluate(opt):
    device = torch.device("cpu" if opt.no_cuda else "cuda")

    model = FoundationDepthAdapter(
        model_name=opt.foundation_model,
        lora_rank=opt.lora_rank,
        lora_alpha=opt.lora_alpha,
        train_head=opt.train_head,
        min_depth=opt.min_depth,
        max_depth=opt.max_depth)

    if opt.load_weights_folder is not None:
        adapt_path = os.path.join(os.path.expanduser(opt.load_weights_folder), "adapt.pth")
        print("-> Loading adapter from", adapt_path)
        model.load_adapter_state_dict(torch.load(adapt_path, map_location="cpu"))
    else:
        print("-> No adapter weights given; evaluating the raw foundation model "
              "(scale will be uncalibrated unless adapt.pth is loaded)")

    model.to(device)
    model.eval()

    filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    img_ext = '.png' if opt.png else '.jpg'
    dataset = datasets.KITTIRAWDataset(opt.data_path, filenames, opt.height, opt.width,
                                       [0], 1, is_train=False, img_ext=img_ext)
    dataloader = DataLoader(dataset, 8, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)

    pred_depths = []
    print("-> Computing predictions at {}x{}".format(opt.width, opt.height))
    with torch.no_grad():
        for data in dataloader:
            input_color = data[("color", 0, 0)].to(device)
            depth = model(input_color)
            pred_depths.append(depth.cpu()[:, 0].numpy())
    pred_depths = np.concatenate(pred_depths)

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    errors_raw, errors_scaled, ratios = [], [], []

    for i in range(pred_depths.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_depth = cv2.resize(pred_depths[i], (gt_width, gt_height))

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
        else:
            mask = gt_depth > 0

        pred = pred_depth[mask]
        gt = gt_depth[mask]

        ratio = np.median(gt) / np.median(pred)
        ratios.append(ratio)

        pred_raw = np.clip(pred, MIN_DEPTH, MAX_DEPTH)
        errors_raw.append(compute_errors(gt, pred_raw))

        pred_scaled = np.clip(pred * ratio, MIN_DEPTH, MAX_DEPTH)
        errors_scaled.append(compute_errors(gt, pred_scaled))

    ratios = np.array(ratios)
    print("\n Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(
        np.median(ratios), np.std(ratios / np.median(ratios))))

    header = ("{:>10} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")
    for name, errs in (("RAW METRIC (no scaling)", errors_raw),
                       ("median-scaled", errors_scaled)):
        mean_errors = np.array(errs).mean(0)
        print("\n {}:".format(name))
        print(header)
        print(("{:10.3f} | " * 7).format(*mean_errors.tolist()))

    print("\n-> Done!")


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == "__main__":
    options = MonodepthOptions()
    options.parser.convert_arg_line_to_args = convert_arg_line_to_args
    if sys.argv.__len__() == 2:
        opt = options.parser.parse_args(['@' + sys.argv[1]])
    else:
        opt = options.parser.parse_args()
    evaluate(opt)
