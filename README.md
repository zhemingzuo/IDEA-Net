# IDEA-Net: Adaptive Dual Self-Attention Network for Single Image Denoising
...

## _Introduction_
This is an official implementation of our adapt**I**ve **D**ual s**E**lf-**A**ttention **Net**work (**IDEA-Net**).

IDEA-Net is an unsupervised single image denoiser, which performs on superiorly on AWGN and real-world image noises with a single end-to-end deep neural network, and contributes to the downstream task of face detection in low-light conditions.

<img src = "./fig/IDEA-Net.jpg" width = "100%" alt = "Architecture of our IDEA-Net">

For more details, please refer our [paper](https://arxiv.org/pdf/2101.03581.pdf).

## _Contents_
1. [Requirement](#requirement)
2. [Preparation](#preparation)
3. [Run](#run)
4. [Performance](#performance)

### _Requirement_
- TensorFlow == 1.14.0
- Python == 3.6
- CUDA == 10.0
- keras
- scikit-image
- scipy
- cv2

### _Preparation_
Clone the github repository. We will call the directory `$IDEA-Net_ROOT`
```Shell
  git https://github.com/zhemingzuo/IDEA-Net
  cd $IDEA-Net_ROOT
```

### _Run_
Run our IDEA-Net
```Shell
  cd $IDEA-Net_ROOT/src
```
and then run `denoise.py`.

### _Performance_
1. Removing AWGN Image Noise

Comparisons of denoising results with respect to PSNR in the case of AWGN with $\sigma$ valued as 25, 50, and 75. \textcolor{green}{$\square$} denotes the selected image region for comparison and \textcolor{blue}{$\square$} indicates the attention \textcolor{blue}{$\mathcal{A}$} drawn by IDEA-Net. Best viewed in zoomed mode.
<img src = "./fig/Example_AWGN.jpg" width = "100%" alt = "">

2. Removing Real-World Image Noise

Comparisons of denoising results in terms of PSNR on a real-world noisy image.  ̋ denotes the selected image region for
comparison and  ̋ indicates the attention A drawn by IDEA-Net. Best viewed in zoomed mode.
<img src = "./fig/Example_real_noise.jpg" width = "100%" alt = "">

3. Downstream task on dark face detection