<!-- PROJECT LOGO -->
<p align="center">
  <h3 align="center">PyTorch-GOTURN</h3>

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
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)

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

<!-- GETTING STARTED -->
## Getting Started

**Code Setup**
```
# Clone the repository
git clone https://github.com/nrupatunga/goturn-pytorch

# install all the required repositories
cd goturn-pytorch/src
pip install -r requirements.txt

# Add current directory to environment
source settings.sh
```

**Data Download**
```
cd goturn-pytorch/src/scripts
./download_data.sh /path/to/data/directory
```
