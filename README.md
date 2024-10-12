
# Real-Time Seismic Event Recognition with Semantic Segmentation


This repository contains the code for reproducing the results in the paper: "[Paper Title](link-to-paper)".

## Structure
- **`utils/`**: Contains utility functions for data preprocessing and model setup.
- **`models/`**: Includes model architectures. Pre-trained weights are available on Zenodo at [link].
- **`examples/`**: Example scripts for data exploration, running the model, and segmentation tasks.



Code Implementation of the article "A Framework for Real-Time Volcano-Seismic Event Recognition Based on Multi-Station Seismograms and Semantic Segmentation Models" (under revision).

## Setup
Clone this repository:
```bash
git clone https://github.com/camilo-espinosa/volcano-seismic-segmentation.git
cd repository-name
```
Install dependencies in requirements.txt

```bash
pip install -r requirements.txt
```
A version of PyTorch, with CUDA compatibility is also necessary: https://pytorch.org/get-started/locally/.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Usage examples (Notebooks):

from the examples folder:
```bash
cd examples
```
### Explore the data: 
explore_data.ipynb

### Folding/Unfolding Demonstration: 
explore_data.ipynb

Volcano Seismic Event Recognition
Datasets are freely available at: 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg)](10.5281/zenodo.13901244)

Pre-trained weights for the four evaluated models (UNet, UNet++, DeepLabV3+ and SwinUNet) are also freely available at: 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg)](10.5281/zenodo.13902232).

UNet and SwinUNet Implementations are used as is from the codes at https://github.com/mateuszbuda/brain-segmentation-pytorch and https://github.com/HuCaoFighting/Swin-Unet, respectively.
