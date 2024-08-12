# Back-in-Time Diffusion (BTD): Unsupervised Detection of Medical Deepfakes

## Overview
Back-in-Time Diffusion (BTD) is a novel approach for detecting medical deepfakes using Denoising Diffusion Probabilistic Models (DDPMs). This repository contains the implementation of BTD, which is designed to detect tampered CT and MRI medical images, specifically focusing on the injection and removal of tumors.

BTD leverages the generative abilities of DDPMs in a reverse diffusion process to reveal synthetic content by analyzing residuals. This approach allows for robust detection of deepfakes in medical imagery, outperforming other state-of-the-art unsupervised detection methods.
