from torch import cuda
from segmentation_models_pytorch import UnetPlusPlus, DeepLabV3Plus
from SwinUNet import SwinTransformerSys
from UNet import UNet


def model_selector(arch, N=256):
    device = "cuda" if cuda.is_available() == True else "cpu"
    if arch == "UNet":
        model = UNet(in_channels=1, out_channels=6, init_features=32).to(device)
    if arch == "UNetPlusPlus":
        model = UnetPlusPlus(
            in_channels=1, classes=6, encoder_weights=None, activation="softmax2d"
        ).to(device)
    if arch == "DeepLabV3":
        model = DeepLabV3Plus(
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
