# Zero-label metric adaptation trainer: a frozen depth foundation model with
# LoRA adapters + global scale/shift is tuned with the photometric loss, where
# the warp pose is metric GT ego-motion (KITTI oxts). No pose network.
#
# --lora_rank 0 trains only scale/shift -> the "global affine rescaling"
# baseline that adaptation must beat.
from __future__ import absolute_import, division, print_function

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from utils import readlines, sec_to_hm_str, normalize_image
from layers import BackprojectDepth, Project3D, SSIM, get_smooth_loss

import datasets
from networks.foundation_depth import FoundationDepthAdapter


class AdaptTrainer:
    def __init__(self, options):
        self.opt = options
        assert not self.opt.use_stereo, "adaptation is mono-only"
        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.model = FoundationDepthAdapter(
            model_name=self.opt.foundation_model,
            lora_rank=self.opt.lora_rank,
            lora_alpha=self.opt.lora_alpha,
            train_head=self.opt.train_head,
            min_depth=self.opt.min_depth,
            max_depth=self.opt.max_depth).to(self.device)

        self.parameters_to_train = self.model.trainable_parameters()
        num_trainable = sum(p.numel() for p in self.parameters_to_train)
        print("Adapting {} | LoRA rank {} ({} layers) | trainable params: {}".format(
            self.opt.foundation_model, self.opt.lora_rank,
            self.model.num_lora_layers, num_trainable))

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        assert self.opt.dataset == "kitti", "metric GT pose is implemented for KITTI raw"
        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        self.num_total_steps = len(train_filenames) // self.opt.batch_size * self.opt.num_epochs

        train_dataset = datasets.KITTIRAWDataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 1, is_train=True, img_ext=img_ext, load_gt_pose=True)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = datasets.KITTIRAWDataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 1, is_train=False, img_ext=img_ext, load_gt_pose=True)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.ssim = SSIM().to(self.device)
        self.backproject_depth = BackprojectDepth(
            self.opt.batch_size, self.opt.height, self.opt.width).to(self.device)
        self.project_3d = Project3D(
            self.opt.batch_size, self.opt.height, self.opt.width).to(self.device)

        print("Training model named:\n  ", self.opt.model_name)
        print("Using split:\n  ", self.opt.split)
        self.save_opts()

    def train(self):
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            self.model_lr_scheduler.step()
            self.save_model()

    def run_epoch(self):
        print("Training")
        self.model.train()

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            if self.opt.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters_to_train, self.opt.grad_clip)
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 1000 == 0
            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        outputs = {}
        depth = self.model(inputs["color_aug", 0, 0])
        outputs[("depth", 0, 0)] = depth

        for frame_id in self.opt.frame_ids[1:]:
            T = inputs[("gt_pose", frame_id)]
            cam_points = self.backproject_depth(depth, inputs[("inv_K", 0)])
            pix_coords = self.project_3d(cam_points, inputs[("K", 0)], T)
            outputs[("color", frame_id, 0)] = F.grid_sample(
                inputs[("color", frame_id, 0)], pix_coords,
                padding_mode="border", align_corners=True)

        losses = self.compute_losses(inputs, outputs)
        return outputs, losses

    def compute_reprojection_loss(self, pred, target):
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        return 0.85 * ssim_loss + 0.15 * l1_loss

    def compute_losses(self, inputs, outputs):
        losses = {}
        target = inputs[("color", 0, 0)]

        reprojection_losses = []
        identity_reprojection_losses = []
        for frame_id in self.opt.frame_ids[1:]:
            reprojection_losses.append(
                self.compute_reprojection_loss(outputs[("color", frame_id, 0)], target))
            identity_reprojection_losses.append(
                self.compute_reprojection_loss(inputs[("color", frame_id, 0)], target))

        reprojection_loss = torch.cat(reprojection_losses, 1)
        identity_reprojection_loss = torch.cat(identity_reprojection_losses, 1)
        # break ties so static pixels pick the identity branch (automasking)
        identity_reprojection_loss += torch.randn(
            identity_reprojection_loss.shape, device=self.device) * 0.00001

        combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
        to_optimise, idxs = torch.min(combined, dim=1)
        outputs["identity_selection/0"] = (
            idxs > identity_reprojection_loss.shape[1] - 1).float()

        loss = to_optimise.mean()

        disp = 1.0 / outputs[("depth", 0, 0)]
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        smooth_loss = get_smooth_loss(norm_disp, target)
        loss += self.opt.disparity_smoothness * smooth_loss

        losses["loss"] = loss
        losses["loss/reprojection"] = to_optimise.mean()
        losses["loss/smooth"] = smooth_loss
        return losses

    def val(self):
        self.model.eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)
            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)
            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.model.train()

    def compute_depth_losses(self, inputs, outputs, losses):
        """Metric depth metrics on the (sparse) velodyne GT — NO median scaling,
        the whole point is that the adapted model is metric."""
        depth_pred = outputs[("depth", 0, 0)].detach()
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False),
            1e-3, 80)

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt_m = depth_gt[mask]
        depth_pred_m = torch.clamp(depth_pred[mask], min=1e-3, max=80)

        losses["da/scale_ratio"] = torch.median(depth_gt_m) / torch.median(depth_pred_m)
        losses["de/abs_rel"] = torch.mean(torch.abs(depth_gt_m - depth_pred_m) / depth_gt_m)
        thresh = torch.max(depth_gt_m / depth_pred_m, depth_pred_m / depth_gt_m)
        losses["da/a1"] = (thresh < 1.25).float().mean()

    def log_time(self, batch_idx, duration, loss):
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar \
            if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)
        writer.add_scalar("scale/log_scale", self.model.log_scale.item(), self.step)
        writer.add_scalar("scale/shift", self.model.shift.item(), self.step)

        for j in range(min(2, self.opt.batch_size)):
            writer.add_image("color_0/{}".format(j), inputs[("color", 0, 0)][j].data, self.step)
            disp = 1.0 / outputs[("depth", 0, 0)][j]
            writer.add_image("disp/{}".format(j), normalize_image(disp.data), self.step)
            for frame_id in self.opt.frame_ids[1:]:
                writer.add_image("color_pred_{}/{}".format(frame_id, j),
                                 outputs[("color", frame_id, 0)][j].data, self.step)
            writer.add_image("automask/{}".format(j),
                             outputs["identity_selection/0"][j][None, ...], self.step)

    def save_opts(self):
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()
        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        torch.save(self.model.adapter_state_dict(), os.path.join(save_folder, "adapt.pth"))
        torch.save(self.model_optimizer.state_dict(), os.path.join(save_folder, "adam.pth"))
        print("-> saved adapter to {}".format(save_folder))
