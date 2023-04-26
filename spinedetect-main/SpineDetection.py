# -*- coding: utf-8 -*-
"""
Created on Jan 30 

@author: Chancellor Gary
modified by Tetsuya Watabe
"""
import os
from pathlib import Path
import glob
import numpy as np
import torch
import torch.utils.data
import PIL
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class SpineDetection():
    def __init__(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                    path='best.pt', force_reload=False)
        self.model.to(device)

    def image_conversion(self, tifffile_path :str):
        im_tiff = PIL.Image.open(tifffile_path)
        nptiff = np.array(im_tiff)
        normtiff = (255 * (nptiff / nptiff.max())).astype(np.uint8)
        norm_im = PIL.Image.fromarray(normtiff)
        rgb = norm_im.convert('RGB')
        return rgb
    
    def prediction(self, rgb :PIL.Image.Image):
        self.model.eval()
        with torch.no_grad():
            prediction = self.model([rgb])
            return prediction

    def save_detection(self, rgb, prediction, savepath, show = False):
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
    SpineDetect = SpineDetection()
    
    tifffiles = glob.glob('./sample/*.tif')
    savefolder = './result'
    os.makedirs(savefolder, exist_ok = True)
    
    for tifffile_path in tifffiles:
        print(tifffile_path)
        rgb = SpineDetect.image_conversion(tifffile_path)
        prediction = SpineDetect.prediction(rgb)
        savepath = os.path.join(savefolder, Path(tifffile_path).stem + '.png')
        SpineDetect.save_detection(rgb, prediction,
                                   savepath = savepath)

