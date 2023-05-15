# CAM reference: https://colab.research.google.com/drive/1mlap4OnlQ-NzubieisQbPX7MQ5FrGSJa#scrollTo=1BXoCm3TPrVo 

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import torch.nn as nn

from domainbed import datasets
from domainbed.lib.fast_data_loader import FastDataLoader
from domainbed import algorithms

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from skimage.io import imread
from skimage.transform import resize
from torchvision import models
from torch.utils.data import DataLoader
from torch.nn.functional import softmax, interpolate
from torchvision.io.image import read_image
from torchvision.models import resnet18
from torchvision.transforms.functional import normalize, resize, to_pil_image
from tqdm import tqdm
# CAM imports
from torchcam.methods import SmoothGradCAMpp, LayerCAM, GradCAM 
from torchcam.utils import overlay_mask
# Seeds for reproducibility 
torch.manual_seed(123)
np.random.seed(123)

## Important Variables
ENV = 3  # 0, 1, 2, 3 for TI, 2 for CMNIST

## Dataset path
# dataset = vars(datasets)["ColoredMNIST_IRM"]("./data/ColoredMNIST/", [ENV], {"data_augmentation": False}) 
dataset = vars(datasets)["TerraIncognita"]("./data/TerraIncognita/", [ENV], {"data_augmentation": False}) 

## MODEL PATH
model_path = f"/Users/sahilkhose/Downloads/GT/Projects/DomainBed_old/models/TerraIncognita/env_{ENV}/model.pkl" 
# model_path = f"/Users/sahilkhose/Downloads/GT/Projects/DomainBed/models/CMNIST/model.pkl"

model_params = torch.load(model_path, map_location=torch.device('cpu'))

## add the following to do some keeping
model_params["model_hparams"]["rsc_sched"] = None
model_params["model_hparams"]["vrex_sched"] = None

## normal code 
model = algorithms.VRex_RSC(dataset.input_shape, dataset.num_classes,
        len(dataset) - 1, model_params['model_hparams'])
model.load_state_dict(model_params['model_dict'])
model.eval()

print("__"*40)
cam_extractor = GradCAM(model.network) # CAM model
for env_i, env in enumerate(dataset):
    if env_i == ENV:
        loader = DataLoader(dataset=env, batch_size=1)
        all_y = []
        all_y_pred = []
        for num_data, (x, y) in enumerate(tqdm(loader)):
            y_pred = model.predict(x)
            print("__"*40)
            print("LABEL, PRED")
            print(y.item(), y_pred.squeeze(0).argmax().item())
            ## Metrics
            all_y.extend(y.detach().numpy().tolist())
            all_y_pred.extend(y_pred.argmax(1).detach().numpy().tolist())
            ## Image
            plt.imshow(x.squeeze(0).permute(1, 2, 0)); plt.show()
            cams = cam_extractor(y_pred.squeeze(0).argmax().item(), y_pred)
            ## Overlayed on the image
            for name, cam in zip(cam_extractor.target_names, cams):
                result = overlay_mask(to_pil_image(x.squeeze(0)), to_pil_image(cam.squeeze(0), mode='F'), alpha=0.1)
                plt.imshow(result); plt.axis('off'); plt.title(name); plt.show()
            ## Stop the loop
            if num_data==1: break # remove this for metrics
        
        ## Metrics
        # print(confusion_matrix(y_true=all_y, y_pred=all_y_pred))
        # print(accuracy_score(y_true=all_y, y_pred=all_y_pred))
        # print(classification_report(y_true=all_y, y_pred=all_y_pred))