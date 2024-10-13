from torch import cuda, load
from segmentation_models_pytorch import UnetPlusPlus, DeepLabV3Plus
from SwinUNet import SwinTransformerSys
from UNet import UNet
import requests
import os

def model_selector(arch, N=256, pretrained=True):
    device = "cuda" if cuda.is_available() == True else "cpu"
    if arch == "UNet":
        model = UNet(in_channels=1, out_channels=6, init_features=32)
    elif arch == "DeepLabV3":
        model = DeepLabV3Plus(
            in_channels=1, classes=6, encoder_weights=None, activation="softmax2d"
        )
    elif arch == "UNetPlusPlus":
        model = UnetPlusPlus(
            in_channels=1, classes=6, encoder_weights=None, activation="softmax2d"
        )
    elif arch == "SwinUNet":
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
        )
    if pretrained:
        doi = '10.5281/zenodo.13902232'
        record_id = doi.split('.')[-1]  
        metadata_url = f"https://zenodo.org/api/records/{record_id}"
        response = requests.get(metadata_url)
        metadata = response.json()
        files = metadata['files']
        if arch == "UNet":
            file_to_download = files[0] # 1: UNet.pt - https://zenodo.org/api/records/13902232/files/UNet.pt/content
        elif arch == "DeepLabV3":
            file_to_download = files[1] # 2: DeepLabV3Plus.pt - https://zenodo.org/api/records/13902232/files/DeepLabV3Plus.pt/content
        elif arch == "UNetPlusPlus":
            file_to_download = files[2] # 3: UNetPlusPlus.pt - https://zenodo.org/api/records/13902232/files/UNetPlusPlus.pt/content
        elif arch == "SwinUNet":
            file_to_download = files[3] # 4: SwinUNet.pt - https://zenodo.org/api/records/13902232/files/SwinUNet.pt/content
        file_url = file_to_download['links']['self']
        weights_path = file_to_download['key']
        if os.path.exists(weights_path):
            print(f"{weights_path} Already downloaded")   
        else:
            file_response = requests.get(file_url)
            with open(weights_path, 'wb') as f:
                f.write(file_response.content)
            print(f"Downloaded {weights_path}")   
        print(f"Loading weights...")
        model.load_state_dict(load(weights_path, map_location=device)["model_state_dict"])
    return model




