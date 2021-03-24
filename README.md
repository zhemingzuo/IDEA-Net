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

Comparisons of AWGN denoising results in terms of PSNR on the [(C)BSD68 dataset](https://github.com/clausmichele/CBSD68-dataset) with <img src="https://latex.codecogs.com/gif.latex?\sigma"/>  valued as 25 and 50. <img src = "./fig/green_box.png" width = "100%" alt = ""> denotes the selected image region for comparison and<img src = "./fig/blue_box.png" width = "100%" alt = ""> indicates the dual self-attention region drawn by IDEA-Net. Best viewed in zoomed mode.
<img src = "./fig/Example_AWGN.jpg" width = "100%" alt = "">

2. Removing Real-World Image Noise

Comparisons of real-world image noise removal results with respect to PSNR on the [PolyU dataset](https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset). \textcolor{green}{$\square$} denotes the selected image region for comparison and \textcolor{blue}{$\square$} indicates the dual self-attention region drawn by the proposed IDEA-Net. Best viewed in zoomed mode..
<img src = "./fig/Example_real_noise.jpg" width = "100%" alt = "">

3. Downstream task on dark face detection

Performance comparisons of real-world dark/noisy face detection on the [DARK FACE dataset](https://flyywh.github.io/CVPRW2019LowLight/). Light-Enhanced Noisy Image (LENI) is yielded by [MSRCR](https://ieeexplore.ieee.org/document/597272). Detection results are generated by a [RetinaNet](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf) that pre-trained on the [WIDER FACE dataset](http://shuoyang1213.me/WIDERFACE/). \textcolor{green}{$\square$} and \textcolor{red}{$\square$} respectively represents the correct and erroneous detections. \textcolor{blue}{$\square$} indicates the dual self-attention region drawn by IDEA-Net. Best viewed in zoomed mode.
<img src = "./fig/Example_dark_face.jpg" width = "100%" alt = "">