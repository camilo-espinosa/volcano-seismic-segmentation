
# Real-Time Seismic Event Recognition with Semantic Segmentation

Code Implementation of the article "A Framework for Real-Time Volcano-Seismic Event Recognition Based on Multi-Station Seismograms and Semantic Segmentation Models" (under revision).

## Installation

Install libraries in requirements.txt

```bash
  pip install -r requirements.txt
```
A version of PyTorch, with CUDA compatibility is also necessary: https://pytorch.org/get-started/locally/.

Datasets are freely available at: 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg)](10.5281/zenodo.13901244)

Pre-trained weights for the four evaluated models (UNet, UNet++, DeepLabV3+ and SwinUNet) are also freely available at: 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg)](10.5281/zenodo.13902232).

UNet and SwinUNet Implementations are used as is from the codes at https://github.com/mateuszbuda/brain-segmentation-pytorch and https://github.com/HuCaoFighting/Swin-Unet, respectively.
