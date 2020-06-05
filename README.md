<!-- PROJECT LOGO -->
<p align="center">
  <h3 align="center">GOTURN-PyTorch</h3>

  <p align="center">
    PyTorch implementation of "Learning to Track at 100 FPS with Deep Regression Networks"
    <br />
    <br />
    <a href="https://github.com/nrupatunga/goturn-pytorch/issues">Report Bug</a>
    ·
    <a href="https://github.com/nrupatunga/goturn-pytorch/issues">Request Feature</a>
  </p>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
	- [Code Setup](#code-setup)
	- [Data Download](#data-download)
	- [Training](#training)
	- [Testing](#testing)
* [Tracking-Demo](#tracking-demo)

<!-- ABOUT THE PROJECT -->
## About The Project

This repository contains the reimplementation of GOTURN in PyTorch.
If you are interested in the following, you should consider using this
repository

* Understand different moving parts of the GOTURN algorithm,
independently through code. 

* Plug and play with different parts of the pipeline such as data, network, optimizers, cost functions.

* Built with following frameworks:
	- [PyTorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
	- [Visdom](https://github.com/facebookresearch/visdom)
	
* Similar to original [implementation](https://github.com/davheld/GOTURN) by the author, you will be able to train the model in a day

## Tracking Demo

|Face           |  Surfer | Face |
|------------------------|-------------------------|-------------------------|
|![](https://github.com/nrupatunga/goturn-pytorch/blob/master/doc/output/face.gif)  | ![](https://github.com/nrupatunga/goturn-pytorch/blob/master/doc/output/surfer.gif) | ![](https://github.com/nrupatunga/goturn-pytorch/blob/master/doc/output/face2.gif)

|Bike           |  Bear |
|------------------------|-------------------------|
|![](https://github.com/nrupatunga/goturn-pytorch/blob/master/doc/output/bike.gif)  | ![](https://github.com/nrupatunga/goturn-pytorch/blob/master/doc/output/bear.gif) |

<!-- GETTING STARTED -->
## Getting Started

#### Code Setup
```
# Clone the repository
$ git clone https://github.com/nrupatunga/goturn-pytorch

# install all the required repositories
$ cd goturn-pytorch/src
$ pip install -r requirements.txt

# Add current directory to environment
$ source settings.sh
```

#### Data Download
```
cd goturn-pytorch/src/scripts
$ ./download_data.sh /path/to/data/directory
```

#### Training on custom dataset

- If you want to use your own custom dataset to train, you might want to
understand the how current dataloader works. For that you might need a
smaller dataset to debug, which you can find it
[here](https://drive.google.com/file/d/1XZa4kpgIwUqmrl9vonQZMIaemSUm5P-4/view?usp=sharing).
This contains few samples of ImageNet and Alov dataset.

- Once you understand the dataloaders, please
refer [here](https://github.com/nrupatunga/goturn-pytorch/issues/2#issuecomment-609405851)
for more information on where to modify training script.

#### Training
```
$ cd goturn-pytorch/src/scripts

# Modify the following variables in the script
# Path to imagenet
IMAGENET_PATH='/media/nthere/datasets/ISLVRC2014_Det/'
# Path to alov dataset
ALOV_PATH='/media/nthere/datasets/ALOV/'
# save path for models
SAVE_PATH='./caffenet/'

# open another terminal and run
$ visdom
# You can visualize the train/val images, loss curves,
# simply open https://localhost:8097 in your browser to visualize

# training
$ bash train.sh
```
|Loss           |
|------------------------|
|![](https://github.com/nrupatunga/goturn-pytorch/blob/master/doc/images/loss.png)  |

#### Testing

In order to test the model, you can use the model in this
[link](https://drive.google.com/drive/folders/1utL6Eh7CnxPM8_o8p5T72duZkhhG0tru?usp=sharing)
or you can use your trained model

```
$ mkdir goturn-pytorch/models

# Copy the extracted caffenet folder into models folder, if you are
# using the trained model

$ cd goturn-pytorch/src/scripts
$ bash demo_folder.sh

# once you run, select the bounding box on the frame using mouse, 
# once you select the bounding box press 's' to start tracking, 
# if the model lose tracking, please press 'p' to pause and mark the
# bounding box again as before and press 's' again to continue the
# tracking


# To test on a new video, you need to extract the frames from the video
# using ffmpeg or any other tool and modify folder path in
# demo_folder.sh accordingly
```
---

## Known issues

- Model trained is Caffe is better in getting the objectness, this way
it handles some of the cases quite well than the PyTorch model I have
trained. I tried to debug this issue. You can refer to the discussion
[here](https://discuss.pytorch.org/t/implementation-of-caffe-code-in-pytorch-suboptimal-solution/73267).
If someone solves this issue, please open a PR
