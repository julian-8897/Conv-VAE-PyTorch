<h1 align="center">
  <b>Image Generation and Reconstruction with Variational Autoencoder (VAE) in PyTorch</b><br>
</h1>

<p align="center">
      <a href="https://www.python.org/">
        <img src="https://img.shields.io/badge/Python-3.9-ff69b4.svg" /></a>
       <a href= "https://pytorch.org/">
        <img src="https://img.shields.io/badge/PyTorch-1.9-2BAF2B.svg" /></a>
       <a href= "https://github.com/julian-8897/Vanilla-VAE-PyTorch/blob/master/LICENSE.md">
        <img src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
         
</p>

## Description of Project

Vanilla Variational Autoencoder (VAE) implemented in PyTorch using convolutional/transposed convolutional in the encoder/decoder architecture.

The model was trained on the CelebA dataset, which can be downloaded as a zip file from https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip.

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
