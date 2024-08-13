# Back-in-Time Diffusion (BTD): Unsupervised Detection of Medical Deepfakes

## Overview
Back-in-Time Diffusion (BTD) is a novel approach for detecting medical deepfakes using Denoising Diffusion Probabilistic Models (DDPMs). This repository contains the implementation of BTD, which is designed to detect tampered CT and MRI medical images, specifically focusing on the injection and removal of tumors.

BTD leverages the generative abilities of DDPMs in a reverse diffusion process to reveal synthetic content by analyzing residuals. This approach allows for robust detection of deepfakes in medical imagery, outperforming other state-of-the-art unsupervised detection methods.


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

The  MRI scans used in this dataset were sourced from **Duke Breast Cancer MRI Dataset** [found here](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/). 
The CT scans were sourced from **LIDC-IDRI CT Dataset** [found here](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254).

### Dataset Description

The test sets consist of both original and tampered images, where fake tumors were either injected into or removed from the scans. These manipulations were performed using advanced generative models such as CT-GAN and fine-tuned Stable Diffusion.

- **CT Test Set:** Contains slices of lung CT scans with injected and removed tumors. The tampered images were generated using CT-GAN and fine-tuned Stable Diffusion models.
- **MRI Test Set:** Contains breast MRI slices with injected and removed tumors. Similar to the CT set, the tampered images were generated using CT-GAN and fine-tuned Stable Diffusion models.

