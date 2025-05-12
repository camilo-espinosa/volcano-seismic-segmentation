from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn

# This script uses a model implementation from the following GitHub repository:
# U-Net for brain segmentation: https://github.com/mateuszbuda/brain-segmentation-pytorch

# The model is utilized as-is from the repository and follows the
# original documentation and usage guidelines.


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, depth=4):
        super(UNet, self).__init__()

        features = init_features
        self.encoder_list = nn.ModuleList()
        self.pool_list = nn.ModuleList()
        self.decoder_list = nn.ModuleList()
        self.upconv_list = nn.ModuleList()

        feat_in = in_channels
        for idx in range(depth):
            feat_out = features * 2 ** (idx)
            encoder = UNet._block(feat_in, feat_out, name=f"enc{idx}")
            feat_in = feat_out
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.encoder_list.append(encoder)
            self.pool_list.append(pool)

        self.bottleneck = UNet._block(
            features * 2 ** (depth - 1), features * 2**depth, name="bottleneck"
        )

        for idx in range(depth):
            feat_in = features * 2 ** (depth - idx)
            feat_out = features * 2 ** (depth - 1 - idx)

            upconv = nn.ConvTranspose2d(feat_in, feat_out, kernel_size=2, stride=2)
            self.upconv_list.append(upconv)

            decoder = UNet._block(feat_in, feat_out, name=f"dec{depth-idx}")
            self.decoder_list.append(decoder)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        encodings = []
        for i in range(len(self.encoder_list)):
            x = self.encoder_list[i](x)
            encodings.append(x)
            if i < len(self.pool_list):  # Avoid index error
                x = self.pool_list[i](x)

        x = self.bottleneck(x)

        for i in range(len(self.decoder_list)):
            x = self.upconv_list[i](x)
            x = torch.cat((x, encodings[-(i + 1)]), dim=1)
            x = self.decoder_list[i](x)
        return F.softmax(self.conv(x), dim=1)
        # return torch.sigmoid()

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
