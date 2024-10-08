import numpy as np
import torch
from torch import nn
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import os
import pandas as df
import seaborn as sns
import sys
import gc
import codes_new.utils_data as utils_data
import torch
from torch import Tensor
from codes_new.Z_unet_model import UNet
from codes_new.Z_Swin_UNet import SwinTransformerSys
import segmentation_models_pytorch as smp


def count_trainable_parameters(model: nn.Module):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params}")


class SuppressPrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._devnull = open(os.devnull, "w")
        sys.stdout = self._devnull

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        if not self._devnull.closed:
            self._devnull.close()


def free_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()


def print_time(t_i, t_f):
    elapsed_time_seconds = t_f - t_i
    hours = int(elapsed_time_seconds // 3600)
    minutes = int((elapsed_time_seconds % 3600) // 60)
    seconds = int(elapsed_time_seconds % 60)
    print("Elapsed time: {:02d}:{:02d}:{:02d}".format(hours, minutes, seconds))


def f1_score_from_confusion_matrix(confusion_matrix):
    f1_scores = []
    for i in range(confusion_matrix.shape[0]):
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0
        )
        f1_scores.append(f1)
    return f1_scores


def accuracy_from_confusion_matrix(conf_matrix):
    correct_predictions = np.trace(conf_matrix)
    total_predictions = np.sum(conf_matrix)
    accuracy = correct_predictions / total_predictions
    return accuracy


def dice_loss_2D(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
        pred: tensor with first dimension as batch
        target: tensor with first dimension as batch
    """
    smooth = 1.0
    iflat = pred.contiguous().view(-1)
    iflat = iflat / iflat.max()
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    A_sum = torch.sum(iflat)
    B_sum = torch.sum(tflat)
    return 1 - ((2.0 * intersection + smooth) / (A_sum + B_sum + smooth))


def dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
):
    assert input.size() == target.size()
    assert input.dim() == 2 or not reduce_batch_first
    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(
    input: Tensor,
    target: Tensor,
    reduce_batch_first: bool = False,
    epsilon: float = 1e-6,
):
    return dice_coeff(
        input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon
    )


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def model_selector(arch, N=256):
    device = "cuda" if torch.cuda.is_available() == True else "cpu"
    if arch == "UNet":
        model = UNet(in_channels=1, out_channels=6, init_features=32).to(device)
    if arch == "UNetPlusPlus":
        model = smp.UnetPlusPlus(
            in_channels=1, classes=6, encoder_weights=None, activation="softmax2d"
        ).to(device)
    if arch == "DeepLabV3":
        model = smp.DeepLabV3Plus(
            in_channels=1, classes=6, encoder_weights=None, activation="softmax2d"
        ).to(device)
    if arch == "SwinUNet":
        model = SwinTransformerSys(
            img_size=N,
            patch_size=4,
            in_chans=1,
            num_classes=6,
            embed_dim=96,
            depths=[2, 2, 2, 2],
            depths_decoder=[1, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            window_size=8,
        ).to(device)
    return model
