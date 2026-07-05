## Preparation
CUDA Version: 11.7

Pytorch: 1.12.0

Numpy: 1.23.1 

Python: 3.9.7

GPU: NVIDIA 3090

Dataset: 
1. Download the THUMOS14 video dataset.
2. Extract two-stream video features using the I3D model.
3. Extract video features and text features using the VideoCLIP-XL model.

We provide the pre-extracted video features at: https://drive.google.com/file/d/17wShClfJ3w-srKm78BBPT8Ljz8yFTz0_/view?usp=drive_link


## Training
Set the correct path to the data.
```
    bash ./scripts/train.sh
```

## Inference
We have placed the trained model parameters in the directory "outputs/best_model".
```
    bash ./scripts/inference.sh
```
