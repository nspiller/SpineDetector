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

file = __file__
file_path = Path(file).parent.resolve()

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

if yolo_path not in sys.path:
    sys.path.insert(0, yolo_path)

import train

yolo_path1 = yolo_path + "\\"
train.run(weights=yolo_path1+'yolov5l.pt', data=yolo_path1+'data.yaml', hyp=yolo_path1+'data\hyps\hyp.scratch-low.yaml', epochs=300, batch_size=32, imgsz=416, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, cache='ram', image_weights=False, multi_scale=False, single_cls=False, optimizer='SGD', sync_bn=False, workers=8, project=yolo_path1+'models/YoloDendritic', name='SpineDetection', exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0,], save_period=-1, seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1,artifact_alias='latest')

'''
Alternatively, you can just change directory to yolo and run in the system command:
python train.py  --imgsz 416 --batch 32 --epochs 300 --data data.yaml --weights yolov5l.pt --project "models/YoloDendritic" --name SpineDetection --cache
'''


#%%
import val
val.run(weight=yolo_path1+'weights/best.pt', data=yolo_path1+'data.yaml', imgsz=416, project='YoloDendritic', name='Valid')

'''
You can also run in the system command:
python val.py --weights "weights/best.pt" --data data.yaml --imgsz 416 --project "YoloDendritic" --name Valid
'''