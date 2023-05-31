
# Spine detection
Detect dendritic spines from a two dimensional image

Developed by Chancellor Gary

Minor modification by Tetsuya Watabe and Ryohei Yasuda


# Installation
The following example can be used for installation. 


## via `pip`
Tested in python 3.11.3. 

```bash
git clone https://github.com/ryoheiyasuda/SpineDetector.git
cd SpineDetector
python(or python3) -m venv venv
./venv/Scripts/activate
pip install -r requirements.txt
```

## via `conda`

Create and activate environment

```bash
conda create -n SpineDetector
conda activate SpineDetector
```

Install pytorch with CUDA support. This is similar to the instructions posted on [pytorch website](https://pytorch.org/get-started/locally/), but also installs CUDA from the [nvidia channel](https://anaconda.org/nvidia/cuda-toolkit) into the same environment.

```bash
conda install pytorch torchvision pytorch-cuda=11.8 cudatoolkit=11.8 -c pytorch -c nvidia
```

Test if `pytorch` has been installed correctly with GPU support
```
python
>>> import torch
>>> torch.cuda.is_available()
True
```

Install additional dependencies from for `yolov5`
```
git clone https://github.com/ultralytics/yolov5.git yolov5
pip install -r yolov5/requirements.txt
```


# Usage
The following example demonstrates the use of the code for detecting spines from images in the "sample" folder.

```bash
python SpineDetection.py
```

# Creating network
The following example will create a network model from a training set from the "New_Dataset3" folder.

```bash
python Train.py
```
