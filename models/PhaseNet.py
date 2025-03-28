import json
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version


class PhaseNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        classes=3,
        depth=5,
        kernel_size=7,
        stride=4,
        filters_root=8,
        norm="std",
    ):

        super().__init__()

        self.in_channels = in_channels
        self.classes = classes
        self.norm = norm
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.filters_root = filters_root
        self.activation = torch.relu

        self.inc = nn.Conv1d(
            self.in_channels, self.filters_root, self.kernel_size, padding="same"
        )
        self.in_bn = nn.BatchNorm1d(self.filters_root, eps=1e-3)
        self.down_branch = nn.ModuleList()
        self.up_branch = nn.ModuleList()

        last_filters = self.filters_root

        for i in range(self.depth):
            filters = int(2**i * self.filters_root)
            conv_same = nn.Conv1d(
                last_filters, filters, self.kernel_size, padding="same", bias=False
            )
            last_filters = filters
            bn1 = nn.BatchNorm1d(filters, eps=1e-3)
            if i == self.depth - 1:
                conv_down = None
                bn2 = None
            else:
                if i in [1, 2, 3]:
                    padding = 0  # Pad manually
                else:
                    padding = self.kernel_size // 2
                conv_down = nn.Conv1d(
                    filters,
                    filters,
                    self.kernel_size,
                    self.stride,
                    padding=padding,
                    bias=False,
                )
                bn2 = nn.BatchNorm1d(filters, eps=1e-3)

            self.down_branch.append(nn.ModuleList([conv_same, bn1, conv_down, bn2]))

        for i in range(self.depth - 1):
            filters = int(2 ** (self.depth - 2 - i) * self.filters_root)
            conv_up = nn.ConvTranspose1d(
                last_filters, filters, self.kernel_size, self.stride, bias=False
            )
            last_filters = filters
            bn1 = nn.BatchNorm1d(filters, eps=1e-3)
            conv_same = nn.Conv1d(
                2 * filters, filters, self.kernel_size, padding="same", bias=False
            )
            bn2 = nn.BatchNorm1d(filters, eps=1e-3)

            self.up_branch.append(nn.ModuleList([conv_up, bn1, conv_same, bn2]))

        # for i in range(self.depth - 1):
        #     filters = int(2 ** (3 - i) * self.filters_root)
        #     conv_up = nn.ConvTranspose1d(
        #         last_filters, filters, self.kernel_size, self.stride, bias=False
        #     )
        #     last_filters = filters
        #     bn1 = nn.BatchNorm1d(filters, eps=1e-3)
        #     conv_same = nn.Conv1d(
        #         2 * filters, filters, self.kernel_size, padding="same", bias=False
        #     )
        #     bn2 = nn.BatchNorm1d(filters, eps=1e-3)

        #     self.up_branch.append(nn.ModuleList([conv_up, bn1, conv_same, bn2]))

        self.out = nn.Conv1d(last_filters, self.classes, 1, padding="same")
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, logits=False):
        x = self.activation(self.in_bn(self.inc(x)))

        skips = []
        for i, (conv_same, bn1, conv_down, bn2) in enumerate(self.down_branch):
            x = self.activation(bn1(conv_same(x)))

            if conv_down is not None:
                skips.append(x)
                if i == 1:
                    x = F.pad(x, (2, 3), "constant", 0)
                elif i == 2:
                    x = F.pad(x, (1, 3), "constant", 0)
                elif i == 3:
                    x = F.pad(x, (2, 3), "constant", 0)

                x = self.activation(bn2(conv_down(x)))

        for i, ((conv_up, bn1, conv_same, bn2), skip) in enumerate(
            zip(self.up_branch, skips[::-1])
        ):
            x = self.activation(bn1(conv_up(x)))
            x = x[:, :, 1:-2]

            x = self._merge_skip(skip, x)
            x = self.activation(bn2(conv_same(x)))

        x = self.out(x)
        if logits:
            return x
        else:
            return self.softmax(x)

    @staticmethod
    def _merge_skip(skip, x):
        offset = (x.shape[-1] - skip.shape[-1]) // 2
        x_resize = x[:, :, offset : offset + skip.shape[-1]]

        return torch.cat([skip, x_resize], dim=1)
