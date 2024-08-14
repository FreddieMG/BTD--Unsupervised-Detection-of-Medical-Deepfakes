# Back-in-Time Diffusion (BTD): Unsupervised Detection of Medical Deepfakes

## Overview
Back-in-Time Diffusion (BTD) is a novel approach for detecting medical deepfakes using Denoising Diffusion Probabilistic Models (DDPMs). This repository contains the implementation of BTD, which is designed to detect tampered CT and MRI medical images, specifically focusing on the injection and removal of tumors.

BTD leverages the generative abilities of DDPMs in a reverse diffusion process to reveal synthetic content by analyzing residuals. This approach allows for robust detection of deepfakes in medical imagery, outperforming other state-of-the-art unsupervised detection methods.

For more details, please see our paper: [Back-in-Time Diffusion: Unsupervised Detection of Medical Deepfakes](https://arxiv.org/pdf/2407.15169)


## Introduction
Recent advancements in generative AI have made it possible to create highly realistic deepfakes, raising significant concerns in the medical field. The proliferation of these technologies allows for the manipulation of medical images, posing threats such as false diagnoses and insurance fraud.

*An illustration of the danger: slices from real medical scans (left) have tumors rejected into them or removed from (center).*
![AI_tampering_showcase-1](https://github.com/user-attachments/assets/95613067-76b0-4c98-b77d-a211acb53e62)


BTD introduces a new paradigm in unsupervised anomaly detection for medical images. By reversing the diffusion process, BTD can identify subtle forensics left behind by generative models, even when these artifacts are not apparent to the human eye or conventional AI-based detectors.


## Methodology
BTD operates by applying a backward diffusion process to a given image and measuring the residuals between the original and partially denoised image. The model is trained unsupervised on genuine medical images, making it adaptable to a variety of unseen deepfake technologies.

*Overview of the Back-in-Time Diffusion framework: On the left, the model architecture is depicted, where the U-Net predicts the noise added to the images during training. The center part shows the training process, where noise is progressively added to an image (x0) to create noisy versions (xt), and the model learns to predict the noise at each step. On the right, the detection process is outlined, where the model calculates the difference between the original image (x0) and the partially denoised image (x-1) after one reverse step, with the error serving as the detection signal.*
![image](https://github.com/user-attachments/assets/ac42887b-27d7-401f-aed3-f3d4cc2f52d6)

### Key Features:
- **Unsupervised Learning:** BTD does not require labeled datasets, making it practical for real-world applications.
- **Robust Detection:** The model outperforms existing methods in detecting both injection and removal deepfakes in CT and MRI scans.


## Dataset

The test sets used in this repository are specifically curated to evaluate the performance of the Back-in-Time Diffusion (BTD) model in detecting medical deepfakes in CT and MRI scans. These datasets are available for download on Kaggle:

- **Kaggle Dataset:** [BTD MRI and CT Deepfake Test Sets](https://www.kaggle.com/datasets/freddiegraboski/btd-mri-and-ct-deepfake-test-sets)

### Provenance

The  MRI scans used in this work were sourced from **Duke Breast Cancer MRI Dataset** [found here](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/). 
The CT scans were sourced from **LIDC-IDRI CT Dataset** [found here](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254).

### Dataset Description

The test sets consist of both original and tampered images, where fake tumors were either injected into or removed from the scans. These manipulations were performed using advanced generative models such as CT-GAN and fine-tuned Stable Diffusion.

- **CT Test Set:** Contains slices of lung CT scans with injected and removed tumors. The tampered images were generated using CT-GAN and fine-tuned Stable Diffusion models.
- **MRI Test Set:** Contains breast MRI slices with injected and removed tumors. Similar to the CT set, the tampered images were generated using CT-GAN and fine-tuned Stable Diffusion models.


## Model Weights

Pretrained model weights for both CT and MRI models are available for download. 

### Download Links:
- **CT Model Weights:** [CT_model.pt](https://github.com/FreddieMG/BTD--Unsupervised-Detection-of-Medical-Deepfakes/releases/download/v1.0-weights/CT_model.pt)
- **MRI Model Weights:** [MRI_model.pt](https://github.com/FreddieMG/BTD--Unsupervised-Detection-of-Medical-Deepfakes/releases/download/v1.0-weights/MRI_model.pt)

### Usage

To load the pretrained weights in your project, you can use the following code:

```python
import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

# Load model weights from the release
CT_weights = "path_to_downloaded/CT_model.pt"
MRI_weights = "path_to_downloaded/MRI_model.pt"

# Initialize the UNet model
unet = Unet(
    dim = 32,
    dim_mults = (1, 2, 4, 8),
    channels = 1
)

# Load CT model
CT_model = GaussianDiffusion(
    unet,
    objective = "pred_noise",
    image_size = 96,
    timesteps = 1000,
    sampling_timesteps = 250 
).to('cuda' if torch.cuda.is_available() else 'cpu')

CT_model.load_state_dict(torch.load(CT_weights)['model'])
CT_model.eval()

# Load MRI model
MRI_model = GaussianDiffusion(
    unet,
    objective = "pred_noise",
    image_size = 128,
    timesteps = 1000,
    sampling_timesteps = 250   
).to('cuda' if torch.cuda.is_available() else 'cpu')

MRI_model.load_state_dict(torch.load(MRI_weights)['model'])
MRI_model.eval()
```

### BTD Evaluation Notebook
For a ready-to-use evaluation of the Back-in-Time Diffusion (BTD) models on the provided test sets, please refer to the [BTD Evaluation Notebook](https://colab.research.google.com/drive/1Dj2U8LjzkW0J4gs-E7HojjRF65O8_UkG).

This notebook demonstrates how to:

1. Load the test sets from the [Kaggle dataset](https://www.kaggle.com/datasets/freddiegraboski/btd-mri-and-ct-deepfake-test-sets\).
2. Load the pretrained models.
3. Evaluate the models on the test sets.

This notebook is a comprehensive example for users looking to quickly assess the performance of BTD on medical deepfake detection tasks.

## Citation
If you use this code, the pretrained models, or any of the provided datasets in your research, please cite our paper:

```bibtex
@article{grabovski2024back,
  title={Back-in-Time Diffusion: Unsupervised Detection of Medical Deepfakes},
  author={Grabovski, Fred and Yasur, Lior and Amit, Guy and Elovici, Yuval and Mirsky, Yisroel},
  journal={arXiv preprint arXiv:2407.15169},
  year={2024}
}
```
