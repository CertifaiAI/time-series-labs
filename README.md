# Time Series Labs
This repository contains Jupyter notebooks used for training during time series course delivery.
## Contents

1. AR Modelling

2. ARIMA

3. Multilayer perceptron in time series forecasting

4. Time series forecasting using deep learning

5. Time series classification and anomaly detection using deep learning

## Getting Started

### Install Anaconda Individual Edition

Download and install [Anaconda](https://www.anaconda.com/products/individual).

### Environment Setup

You can install the dependencies through any of the following ways:

1. Setup the virtual environment using `conda` by

```
conda env create -f environment.yml
```

2. Setup the virtual environment using `virtualenv` with Python version 3.8 by

```
pip install -r requirement.txt
```

The environment setup will take some time to download required modules.

### GPU Setup (__*optional*__)
Follow the instructions below if you plan to use GPU setup.
1. Install CUDA and cuDNN
    Requirements:
   -  [CUDA 10.2](https://developer.nvidia.com/cuda-10.2-download-archive)
   -  [cuDNN 7.6](https://developer.nvidia.com/rdp/cudnn-archive)
   
> Step by step installation guides can be found [here](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn_765/cudnn-install/index.html#install-windows).

2. If you like to use different version of CUDA, please install appropriate **cudatoolkit** module by enter `conda install cudatoolkit=CUDA_VERSION`

```
conda install cudatoolkit=10.2
```

## Usage
All examples are separated into [training] and [solution] folders.

All notebooks in **training** folder have few lines commented out so that they can be taught and demonstrated in the class. The **solution** folder contains the un-commented version for every line of codes.

## Known Issues
- 
-
-