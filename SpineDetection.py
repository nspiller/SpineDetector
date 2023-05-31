# -*- coding: utf-8 -*-
"""
Created on Jan 30 

@author: Chancellor Gary
modified by Tetsuya Watabe
modified by Nico Spiller
"""
from pathlib import Path
import numpy as np

import torch
import torch.utils.data

import PIL

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class SpineDetection():

    def __init__(self):

        # load model 
        self.model = torch.hub.load(
            repo_or_dir='ultralytics/yolov5', # get model from github repo
            model='custom',
            path='best.pt', # use weights strored on disk
            force_reload=False, # use cached files if available
            )
        
        # check if GPU can be used
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        print(f'INFO using device {device}')

    def image_conversion(self, tif_path :str):
        'load tiff file and return RGB PIL image'

        im_tiff = PIL.Image.open(tif_path)

        arr = np.array(im_tiff)
        arr_norm = (255 * (arr / arr.max())).astype(np.uint8)

        img = PIL.Image.fromarray(arr_norm)
        img_rgb = img.convert('RGB')

        return img_rgb
    
    def prediction(self, rgb :PIL.Image.Image):
        'predict model output for given PIL image'

        self.model.eval()

        with torch.no_grad():
            pred = self.model([rgb])

            return pred

    def save_detection(self, rgb, prediction, savepath, show=False):
        'add prediction to PIL image and save to disk'
        
        matplotlib.rcParams['interactive'] = True

        fig, ax = plt.subplots()
        ax.imshow(rgb)

        for pred in prediction.xyxy[0]:
            ax.add_patch(Rectangle((pred[0], pred[1]),
                                   pred[2]-pred[0], pred[3]-pred[1],
                                   fill = False,ec="r"))
            ax.text(pred[0],pred[1],f"{pred[4]:.2f}",c="r")
            fig.savefig(savepath,dpi=150,bbox_inches="tight")

        if show:
            img = PIL.Image.open(savepath)
            img.show()

        fig.show() #Does not work, why???
        

if __name__ == "__main__":

    # initialize SpineDetect object
    sd = SpineDetection()
    
    tifs = Path('./sample/').glob('*.tif')

    save_dir = Path('./result/')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # cycle through tif files
    for tif in tifs:

        print(tif)

        # convert to normalized RGB image
        rgb = sd.image_conversion(tif)

        # predict based on sd.model
        prediction = sd.prediction(rgb)

        # save prediction
        png_path = save_dir / tif.with_suffix('.png')
        sd.save_detection(rgb, prediction, savepath=png_path)

