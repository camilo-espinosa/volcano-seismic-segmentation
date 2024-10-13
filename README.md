
# Real-Time Seismic Event Recognition with Semantic Segmentation


Code Implementation of the article "[A Framework for Real-Time Volcano-Seismic Event Recognition Based on Multi-Station Seismograms and Semantic Segmentation Models](link-to-paper)" (under review).

## Structure
- **`utils/`**: Contains utility functions to perform the proposed framework.
- **`models/`**: Includes model architectures. UNet and SwinUNet Implementations are used out-of-the-box from the codes at [https://github.com/mateuszbuda/brain-segmentation-pytorch](https://github.com/mateuszbuda/brain-segmentation-pytorch) and [https://github.com/HuCaoFighting/Swin-Unet](https://github.com/HuCaoFighting/Swin-Unet), respectively. Pre-trained weights are available at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13902232.svg)](https://doi.org/10.5281/zenodo.13902232).
- **`examples/`**: Example scripts for data exploration and running the model.


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
A version of PyTorch, with CUDA compatibility is also necessary: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

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

### Folding Demonstration: 
folding_example.ipynb

### Segmentation Demo:
segmentation_demo.ipynb

## Data and weights availability:
Datasets are freely available at: 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13901244.svg)](https://doi.org/10.5281/zenodo.13901244)

Pre-trained weights for the four evaluated models (UNet, UNet++, DeepLabV3+ and SwinUNet) are also freely available at: 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13902232.svg)](https://doi.org/10.5281/zenodo.13902232).

