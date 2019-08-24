# !pip3 install --upgrade torch --user
# !pip3 install torchvision --user

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import pickle
import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn


from torchvision import datasets
from torchvision import transforms


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def normalize(im):
    if im.max() == 0:
        return im
    return (im - im.min())/(im.max()-im.min())


def compute_psnr(image, noised):
    """
    Alert: only from images with max value = 1
    """
    mse = nn.MSELoss()(image, noised).item()
    if mse == 0:
        return 0
    return 10 * np.log10(1/mse)


def pairwise_dist(arr, k):
    """
    arr: torch.Tensor with shape batch x h*w x features
    """
    r_arr = torch.sum(arr * arr, dim=2, keepdim=True) # (B,N,1)
    mul = torch.matmul(arr, arr.permute(0,2,1))         # (B,M,N)
    dist = - (r_arr - 2 * mul + r_arr.permute(0,2,1))       # (B,M,N)
    return dist.topk(k=k, dim=-1)[1]


def batched_index_select(t, dim, inds):
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy) # b x e x f
    return out


def get_closest_diff(arr, k):
    """
    arr: torch.Tensor with shape batch x h * w x features
    """
    b, hw, f = arr.shape
    dists = pairwise_dist(arr, k)
    selected = batched_index_select(arr, 1, dists.view(dists.shape[0], -1)).view(b, hw, k, f)
    diff = arr.unsqueeze(2) - selected
    return diff


def plot_diff(real, noised, denoised):
    plt.figure(figsize=(15, 15))
    
    for ind, im, noise, denoise in zip(range(1, 13, 3), real[:4], noised[:4], denoised[:4]):
        plt.subplot(4, 3, ind)
        plt.imshow(im)
        plt.yticks([])
        plt.xticks([])
        plt.title('Clear image')
        
        plt.subplot(4, 3, ind+1)
        plt.imshow(noise)
        plt.yticks([])
        plt.xticks([])
        plt.title('Noised image')
        
        plt.subplot(4, 3, ind+2)
        plt.imshow(denoise)
        plt.yticks([])
        plt.xticks([])
        plt.title('Denoised image')


    plt.show()