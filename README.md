<h1 align="center">
  <b>Image Generation and Reconstruction with Convolutional Variational Autoencoder (VAE) in PyTorch</b><br>
</h1>

<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.9-ff69b4.svg" /></a>
       <a href= "https://pytorch.org/">
        <img src="https://img.shields.io/badge/PyTorch-1.9-2BAF2B.svg" /></a>
       <a href= "https://github.com/julian-8897/Vanilla-VAE-PyTorch/blob/master/LICENSE.md">
        <img src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
         
</p>

## Implementation Details

A PyTorch implementation of the standard Variational Autoencoder (VAE). The amortized inference model (encoder) is parameterized by a convolutional network, while the generative model (decoder) is parameterized by a transposed convolutional network. The choice of the approximate posterior is the usual multivariate Gaussian distribution with diagonal covariance.

This implementation supports model training on the [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). For ease of access, the zip file which contains the dataset can be downloaded from: https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip.

The VAE model was evaluated on several downstream tasks, such as image reconstruction and image generation. Some sample results can be found in the [Results](https://github.com/julian-8897/Vanilla-VAE-PyTorch/blob/master/README.md#--Results) section.

## Requirements

- Python >= 3.9
- PyTorch >= 1.9

## Installation Guide

```
$ git clone https://github.com/julian-8897/Vanilla-VAE-PyTorch.git
$ cd Vanilla-VAE-PyTorch
$ pip install -r requirements.txt
```

## Usage

### Training

To train the model, please modify the 'config.json' configurations file, and run:

```
python train.py --config config.json
```

### Testing

To test the model, you can run the following:

```
python test.py --resume path/to/checkpoint
```

---

<h2 align="center">
  <b>Results</b><br>
</h2>

## 128 Latent Dimensions

| Reconstructed Samples | Generated Samples |
| --------------------- | ----------------- |
| ![][1]                | ![][2]            |

## 256 Latent Dimensions

| Reconstructed Samples | Generated Samples |
| --------------------- | ----------------- |
| ![][3]                | ![][4]            |

[1]: https://github.com/julian-8897/Vanilla-VAE-PyTorch/blob/master/Reconstructions/recons_epoch_20_128dims.png
[2]: https://github.com/julian-8897/Vanilla-VAE-PyTorch/blob/master/Samples/generated_samples_epoch_20_128dims.png
[3]: https://github.com/julian-8897/Vanilla-VAE-PyTorch/blob/master/Reconstructions/recons_epoch_20_256dims.png
[4]: https://github.com/julian-8897/Vanilla-VAE-PyTorch/blob/master/Samples/generated_samples_epoch_20_256dims.png

## References

1. Original VAE paper "Auto-Encoding Variational Bayes" by Kingma & Welling:
   https://arxiv.org/abs/1312.6114

2. Various implementations of VAEs in PyTorch:
   https://github.com/AntixK/PyTorch-VAE

3. PyTorch template used in this project:
   https://github.com/victoresque/pytorch-template

4. A comprehensive introduction to VAEs:
   https://arxiv.org/pdf/1906.02691.pdf
