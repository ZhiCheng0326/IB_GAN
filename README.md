# IB_GAN

## Description
This repository is modified from [artemyk/ibsgd](https://github.com/artemyk/ibsgd) to fit the experiment on GAN.

## Prerequisites
* python 3.5.6
* six 1.15.0
* keras 2.3.1
* matplotlib 2.0.2
* numpy 1.16.6
* seaborn 0.9.1


## Usage
* `save_activity_gan.py` serves two purposes:
  * train GAN and **save generator model**
  * use the generator model to generate fake images, then concatenate it with real images, forming a new dataset. Next, **save outputs of each layers** while training GAN.  

* `computeMI_gan.py` load rawdata and compute Mutual Information of each layers

### Step 1
Implement this step to obtain generator model, skip this step if `generator_tanh.h5` exist.   

`python save_activity_gan.py --activation tanh --isTrain 1`

### Step 2
Save outputs of each layer while training GAN.  

`python save_activity_gan.py --activation tanh --isTrain 0`

### Step 3
Compute Mutual Information and plot figures.  

`python computeMI_gan.py --activation tanh`

## Results
