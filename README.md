# IDEA-Net
IDEA-Net: Adaptive Dual Self-Attention Network for Single Image Denoising

## _Introduction_
This is an official implementation of our adaptIve Dual sElf-Attention Network (IDEA-Net).

IDEA-Net is an unsupervised single image denoiser, which performs on superiorly on AWGN and real-world image noises with a single end-to-end deep neural network, and contributes to the downstream task of face detection in low-light conditions.

## _Contents_
1. [Requirement](#requirement)
2. [Preparation](#preparation)
3. [Run](#run)
4. [Performance](#performance)

### _Requirement_
- TensorFlow == 1.14.0
- Python == 3.6
- CUDA == 10.0

### _Requirement_
- MATLAB >= 2016a

### _Preparation_
Clone the github repository. We will call the directory `$IDEA-Net_ROOT`
```Shell
  git https://github.com/zhemingzuo/IDEA-Net
  cd $IDEA-Net
```

### _Run_
Run our IDEA-Net
```Shell
  cd $IDEA-Net_ROOT/src
```
and then run `denoise.py`.

### _Performance_
..........