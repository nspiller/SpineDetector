# -*- coding: utf-8 -*-
"""
Created on Jan 30 

@author: Chancellor Gary
modified by Ryohei Yasuda
modified by Nico Spiller
"""
#%%
import yaml
import torch
import sys
from git import Repo
from pathlib import Path

# define folder of `Train.py`
file = __file__
file_path = Path(file).parent.resolve()


#%%
# check if yolo5 code is present, download from repo if not
yolo_path = file_path / 'yolov5'
yolo_git = 'https://github.com/ultralytics/yolov5.git'

if not yolo_path.exists():
    Repo.clone_from(yolo_git, yolo_path)
    import pip
    req_file = yolo_path / 'requirements.txt'
    pip.main(['install', '-r', str(req_file)])
    

#%%
# define training and validation set
data ={
    'names':    ['spine'],
    'nc':       1,
    'train':    str(file_path / 'New_Dataset3/train'), # training
    'val':      str(file_path / 'New_Dataset3/test'), # test
 }

# data.yaml file used by yolo5 
with open(yolo_path / 'data.yaml', 'w') as outfile:
    yaml.dump(data, outfile)
    
torch.cuda.empty_cache()


#%% Run train and validation scripts
# cell-based excecution of this file may not work because of ipykernel and argparse

# add yolo path so `train` and `val` become available
if yolo_path not in sys.path:
    sys.path.insert(0, str(yolo_path))

def main():

    import train

    train.run(
        project=str(yolo_path / 'models/YoloDendritic'),
        name='SpineDetection', 
        exist_ok=False, 
        weights=str(yolo_path / 'yolov5l.pt'),
        data=str(yolo_path / 'data.yaml'),
        hyp=str(yolo_path / 'data/hyps/hyp.scratch-low.yaml'),
        epochs=300,
        batch_size=32,
        imgsz=416, 
        rect=False, 
        resume=False, 
        nosave=False, 
        noval=False, 
        noautoanchor=False, 
        noplots=False, 
        evolve=None, 
        cache='ram', 
        image_weights=False, 
        multi_scale=False, 
        single_cls=False, 
        optimizer='SGD', 
        sync_bn=False, 
        workers=8, 
        quad=False, 
        cos_lr=False, 
        label_smoothing=0.0, 
        patience=100, 
        freeze=[0, ],
        save_period=-1, 
        seed=0, 
        local_rank=-1, 
        entity=None, 
        upload_dataset=False, 
        bbox_interval=-1,
        artifact_alias='latest',
        )

    '''
    Alternatively, you can just change directory to yolo and run in the system command:
    python train.py  --imgsz 416 --batch 32 --epochs 300 --data data.yaml --weights yolov5l.pt --project "models/YoloDendritic" --name SpineDetection --cache
    '''


    #%%
    import val
    val.run(
        weight=yolo_path / 'weights/best.pt',
        data=yolo_path / 'data.yaml',
        imgsz=416,
        project='YoloDendritic',
        name='Valid'
        )

    '''
    You can also run in the system command:
    python val.py --weights "weights/best.pt" --data data.yaml --imgsz 416 --project "YoloDendritic" --name Valid
    '''

if __name__ == '__main__':
    # necessary for multiprocessing on windows
    # https://pytorch.org/docs/stable/notes/windows.html#multiprocessing-error-without-if-clause-protection
    main()