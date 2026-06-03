# ERRNet-DIP26: Single Image Reflection Removal

This repository is a course project for Digital Image Processing. It is based on
ERRNet, the CVPR 2019 single image reflection removal baseline, and explores
several lightweight improvements for reflection removal:

- replacing the MSE pixel loss with Charbonnier loss;
- adding FFT amplitude loss for frequency-domain supervision;
- using dataset-adaptive test-time augmentation (TTA).

The final model improves the weighted average PSNR on five standard benchmarks
from **24.954 dB** to **25.253 dB**.

## Project Overview

Single image reflection removal aims to recover the clean transmission layer
from a single image degraded by glass reflection. Given an input image `I`, the
task is usually modeled as:

```text
I = T + R * h
```

where `T` is the desired transmission layer, `R` is the reflection layer, and
`h` is a blur kernel modeling defocused reflection. Since only one mixed image
is given, the problem is inherently ill-posed.

This project uses ERRNet as the baseline and focuses on loss-level and
inference-level improvements without introducing a heavier network.

## Main Contributions

1. **Charbonnier pixel loss**

   The original MSE pixel component is replaced by Charbonnier loss:

   ```text
   L_pixel(ours) = 0.2 L_Charb + 0.4 L_gradient
   ```

   Charbonnier loss is more robust to large residuals and helps reduce
   over-smoothing in strong-reflection regions.

2. **FFT amplitude loss**

   A frequency-domain amplitude loss is added:

   ```text
   L_G = L_pixel(ours) + 0.1 L_feat + 0.01 L_adv + 0.05 L_FFT
   ```

   It complements spatial-domain supervision and encourages the model to
   preserve high-frequency texture details.

3. **Dataset-adaptive TTA**

   During inference, predictions from transformed inputs are averaged. For most
   datasets, horizontal and vertical flips are used. For `real20`, only
   horizontal flip is used to avoid artifacts caused by non-square resized
   inputs.

4. **Negative-result analysis**

   The project also reports unsuccessful attempts, including SSIM loss,
   reflection-layer supervision, and color jitter augmentation.

## Repository Structure

```text
ERRNet/
  data/                         # data loading utilities
  datasets/                     # dataset preparation scripts
  imgs/                         # original ERRNet visual assets
  models/                       # ERRNet model, losses, networks
  options/                      # training and testing options
  processed_pic/                # five self-collected qualitative examples
  util/                         # utility functions
  engine.py                     # training/evaluation engine
  train_errnet.py               # aligned-data training
  train_errnet_unaligned.py     # unaligned-data finetuning
  test_errnet.py                # benchmark/custom testing
  README_DIP26.md               # short running guide
  requirements.txt
```

## Environment

The project was tested with Python 3.10 and PyTorch. A recommended setup is:

```bash
conda create -n errnet python=3.10 -y
conda activate errnet

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -U pip wheel "setuptools<82"
pip install visdom==0.2.4 --no-build-isolation
```

If your CUDA version is different, install the corresponding PyTorch build from
the official PyTorch website.

## Data and Checkpoints

Download the prepared data and checkpoints:

- BaiduYun: <https://pan.baidu.com/s/1MWb4eT18ySjogKVlcfPozg?pwd=egv2>
- Google Drive: <https://drive.google.com/drive/folders/1_tN6JDlAmKZTgaqniQep1YJXmbFwGav7?usp=drive_link>

After downloading, organize files as follows:

```text
ERRNet/
  checkpoints/
    errnet/
      errnet_060_00463920.pt
      ...
  datasets/
    raw_data/
      VOCdevkit/
      CEILNet/
      real89/
      robustsirr_test_dataset/
      Dataset/
```

Then prepare processed datasets:

```bash
python datasets/prepare_test_data.py
python datasets/prepare_train_data.py
```

## Testing

The testing script is `test_errnet.py`.

Supported benchmark names:

```text
ceilnet_table2
real20
objects
postcard
wild
sir2_withgt
```

Run benchmark evaluation:

```bash
python test_errnet.py \
  --name errnet \
  --dataset ceilnet_table2 \
  -r \
  --icnn_path checkpoints/errnet/errnet_060_00463920.pt \
  --hyper
```

For CPU testing:

```bash
python test_errnet.py \
  --name errnet_cpu \
  --dataset ceilnet_table2 \
  -r \
  --gpu_ids -1 \
  --icnn_path checkpoints/errnet/errnet_060_00463920.pt \
  --hyper
```

Replace `ceilnet_table2` with `real20`, `objects`, `postcard`, or `wild` to
evaluate other datasets.

## Testing on Custom Images

For real images without ground truth, put images into a folder, for example:

```text
datasets/raw_data/my_test_images/
  img1.jpg
  img2.jpg
```

Run:

```bash
python test_errnet.py \
  --name errnet \
  --dataset custom \
  --input_dir ./datasets/raw_data/my_test_images \
  -r \
  --icnn_path checkpoints/errnet/errnet_060_00463920.pt \
  --hyper
```

Outputs are saved under the result directory. For custom images, PSNR, SSIM,
NCC, and LMSE are not reported because these full-reference metrics require a
clean ground-truth transmission image.

## Training

Baseline aligned-data training:

```bash
python train_errnet.py --name errnet --hyper
```

Improved aligned-data training:

```bash
python train_errnet.py \
  --name errnet_charb_fft \
  --hyper \
  --use_charbonnier \
  --lambda_fft 0.05
```

The proposed Charbonnier loss and FFT amplitude loss are controlled by command-line options and are not enabled by default. `--use_charbonnier` replaces the MSE term in the pixel loss with Charbonnier loss, and `--lambda_fft 0.05` enables the FFT amplitude loss with weight 0.05.

CPU debug run:

```bash
python train_errnet.py --name errnet_cpu --hyper --gpu_ids -1 --debug
```

Unaligned-data finetuning:

```bash
python train_errnet_unaligned.py \
  --name errnet_unaligned_ft \
  --hyper \
  -r \
  --icnn_path checkpoints/errnet/errnet_060_00463920.pt \
  --unaligned_loss vgg
```

## Testing

Baseline testing:

```bash
python test_errnet.py --name errnet --hyper
```

Improved testing with TTA:

```bash
python test_errnet.py \
  --name errnet_charb_fft \
  --hyper \
  --tta_mode full
```

Test-time augmentation is controlled by `--tta_mode` and is not enabled by default. `--tta_mode full` averages predictions from the original image, horizontal flip, and vertical flip.

## Main Results
The final method, denoted as **Ours: Charbonnier + FFT + TTA**, uses `--use_charbonnier`, `--lambda_fft 0.05`, and `--tta_mode full`.

The best checkpoints are available at <https://pan.baidu.com/s/1Gq7d8ij17qyyucdO9Sugfw?pwd=yqag>.

### PSNR / SSIM

| Method | CEILNet | real20 | objects | postcard | wild | Weighted Avg |
| --- | --- | --- | --- | --- | --- | --- |
| ERRNet Baseline | 27.88 / 0.9407 | 23.55 / 0.8285 | 24.85 / 0.8980 | 22.07 / 0.8773 | 25.18 / 0.8860 | 24.954 / 0.8953 |
| Baseline + TTA | 28.18 / 0.9461 | 23.54 / 0.8288 | 24.91 / 0.9003 | 22.13 / 0.8820 | 25.31 / 0.8898 | 25.075 / 0.8990 |
| Charbonnier + FFT v1 | 27.98 / 0.9418 | 23.60 / 0.8231 | 24.62 / 0.8958 | 22.73 / 0.8867 | 25.49 / 0.9062 | 25.118 / 0.9030 |
| Charbonnier + FFT v2 | 27.96 / 0.9420 | 23.75 / 0.8282 | 24.68 / 0.8973 | 22.64 / 0.8835 | 25.22 / 0.8954 | 25.032 / 0.8995 |
| **Ours: Charbonnier + FFT + TTA** | **28.35 / 0.9467** | 23.64 / 0.8233 | 24.70 / 0.8977 | **22.80 / 0.8903** | **25.60 / 0.9091** | **25.253 / 0.9059** |

### NCC / LMSE

| Method | Weighted NCC | Weighted LMSE |
| --- | --- | --- |
| ERRNet Baseline | 0.9572 | 0.00584 |
| Baseline + TTA | 0.9585 | 0.00545 |
| Charbonnier + FFT v1 | 0.9621 | 0.00485 |
| Charbonnier + FFT v2 | 0.9601 | 0.00515 |
| **Ours: Charbonnier + FFT + TTA** | **0.9628** | **0.00464** |

## Ablation Summary

| Method | CEILNet PSNR | postcard PSNR | Weighted Avg PSNR |
| --- | ---: | ---: | ---: |
| Baseline | 27.88 | 22.07 | 24.954 |
| + Charbonnier | 28.08 | 22.79 | 25.031 |
| + FFT only | 27.85 | 22.31 | 24.922 |
| + Charbonnier + TTA | 28.41 | 22.90 | 25.163 |
| + Charbonnier + FFT v1 | 27.98 | 22.73 | 25.120 |
| **+ Charbonnier + FFT v1 + TTA** | **28.35** | 22.80 | **25.253** |

## Negative Results

| Attempt | Weighted Avg PSNR | Observation |
| --- | ---: | --- |
| SSIM loss | 24.713 | May preserve reflection structures such as text and patterns. |
| Reflection-layer supervision | 24.427 | `I - T_hat` is not a correct reflection proxy under blur/gamma/clip synthesis. |
| Color jitter | 24.937 | May destroy color consistency between transmission and reflection layers. |

## Self-Collected Real Images

The folder `processed_pic/` contains five self-collected real-world reflection
examples. Each subfolder includes:

```text
m_input.png          # input image
errnet_baseline.png  # baseline output
ours.png             # improved method output
```

These images do not have clean ground-truth transmission layers, so they are
used only for qualitative visualization.


## Acknowledgements

This project is based on the official ERRNet implementation:

> Wei, K., Yang, J., Fu, Y., Wipf, D., & Huang, H.  
> Single Image Reflection Removal Exploiting Misaligned Training Data and Network Enhancements.  
> CVPR 2019.

Original repository:

<https://github.com/Vandermode/ERRNet>

