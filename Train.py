# -*- coding: utf-8 -*-
"""
Created on Jan 30 

@author: Chancellor Gary
modified by Ryohei Yasuda
"""

import yaml
import torch
import os
import sys
from git import Repo
from pathlib import Path

file_path = str(Path.cwd().resolve())


#%%
#This will download yolo5 program.
yolo_path = os.path.join(file_path, 'yolov5')
yolo_git = 'https://github.com/ultralytics/yolov5.git'
if not os.path.exists(yolo_path):
    Repo.clone_from(yolo_git, yolo_path)
    import pip
    req_file = os.path.join(yolo_path, 'requirements.txt')
    pip.main(['install', '-r', req_file])    
    

#%%
#Data used for training and validation.
data ={
'names':['spine'],
'nc': 1,
#folders of training and validation datasets
'train': os.path.join(file_path, "New_Dataset3/train"),
'val': os.path.join(file_path, "New_Dataset3/test"),
 }
with open(os.path.join(yolo_path, 'data.yaml'), 'w') as outfile:
    yaml.dump(data, outfile)
    
    
torch.cuda.empty_cache()


#%% Run train and vaolidation scripts

# import subprocess
# os.chdir(yolo_path)

if '../' not in sys.path:
    sys.path.insert(0, '../')

from train import run as run_train


run_train(weights='yolov5l.pt', data='data.yaml', hyp='data\hyps\hyp.scratch-low.yaml', epochs=300, batch_size=32, imgsz=416, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, cache='ram', image_weights=False, multi_scale=False, single_cls=False, optimizer='SGD', sync_bn=False, workers=8, project='models/YoloDendritic', name='SpineDetection', exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0,], save_period=-1, seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1,artifact_alias='latest')

#%%
from val import run as run_val
run_val(weight='weights/best.pt', data='data.yaml', imgsz=416, project='YoloDendritic', name='Valid')
