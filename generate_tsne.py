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
ENV = 2  # 0, 1, 2, 3 for TI, 2 for CMNIST

## Dataset path
dataset = vars(datasets)["ColoredMNIST_IRM"]("./data/ColoredMNIST/", [ENV], {"data_augmentation": False}) 
# dataset = vars(datasets)["TerraIncognita"]("./data/TerraIncognita/", [ENV], {"data_augmentation": False}) 

## MODEL PATH
# model_path = f"/Users/sahilkhose/Downloads/GT/Projects/DomainBed/models/TerraIncognita/env_{ENV}/model.pkl" 
model_path = f"/Users/sahilkhose/Downloads/GT/Projects/DomainBed/models/CMNIST/model.pkl"

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
print(model.network[0]) # printing what is being used to fetch the features
for env_i, env in enumerate(dataset):
    if env_i == ENV:
        loader = DataLoader(dataset=env, batch_size=1)
        all_y = []
        all_y_pred = []
        images = []
        embeddings = []
        for num_data, (x, y) in enumerate(tqdm(loader)):
            y_pred = model.predict(x)
            embeddings.append(model.network[0](x).squeeze(0).detach().numpy()) # fetching the feature maps, ERROR could be here? 
            ## Metrics
            all_y.extend(y.detach().numpy().tolist())
            all_y_pred.extend(y_pred.argmax(1).detach().numpy().tolist())
            ## Image
            img = x.squeeze(0).permute(1, 2, 0) 
            new_img_1 = torch.cat((img, torch.zeros((img.shape[0], img.shape[1], 1))), dim=-1)
            images.append(new_img_1.detach().numpy())
    
        # Example Images    
        num_row = 2
        num_col = 5
        num = num_col * num_row
        images = np.array(images)
        embeddings = np.array(embeddings)
        print(embeddings.shape)
        print(images.shape)
        # plot images
        fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
        for i in range(num):
            ax = axes[i//num_col, i%num_col]
            ax.imshow(images[i])
            ax.set_title(f'Label: {all_y[i]} Pred: {all_y_pred[i]}')
        plt.tight_layout()
        plt.show()

        # Create a two dimensional t-SNE projection of the embeddings
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        from matplotlib import cm
        tsne = TSNE(2, verbose=1)
        tsne_proj = tsne.fit_transform(embeddings)
        # Plot those points as a scatter plot and label them based on the pred labels
        cmap = cm.get_cmap('tab20')
        fig, ax = plt.subplots(figsize=(8,8))
        num_categories = 10
        for lab in range(num_categories):
            indices = all_y_pred==lab
            ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
        ax.legend(fontsize='large', markerscale=2)
        plt.show()
        # print(confusion_matrix(y_true=all_y, y_pred=all_y_pred))
        print(accuracy_score(y_true=all_y, y_pred=all_y_pred))
        # print(classification_report(y_true=all_y, y_pred=all_y_pred))

