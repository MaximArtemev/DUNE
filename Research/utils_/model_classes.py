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

from utils.model_utils import PatchLoader, NonLocalAggregation
from utils.martemev_utils import compute_psnr, get_freer_gpu

from time import time


def get_CNN():

    class GraphConv(nn.Module):
        def __init__(self, k, input_channels, out_channels, search_area=None):
            super().__init__()
            self.conv1 = nn.Conv2d(input_channels, out_channels, 1)
            self.conv2 = nn.Conv2d(input_channels, out_channels, 3, padding=1)
            self.conv3 = nn.Conv2d(input_channels, out_channels, 5, padding=2)

        def forward(self, x):
            return torch.mean(torch.stack([self.conv1(x),
                                           self.conv2(x),
                                           self.conv3(x)]), dim=0) # here is conv instead of gc


    class PreProcessBlock(nn.Module):
        def __init__(self, input_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv2d(input_channels, out_channels, kernel_size,
                                  padding=(kernel_size//2, kernel_size//2))
            self.activ = nn.LeakyReLU(0.05)
            self.bn = nn.BatchNorm2d(out_channels)

            self.GC = GraphConv(k, out_channels, out_channels) # out_channels -> out_channels

        def forward(self, x):
            x = self.activ(self.conv(x))
            x = self.GC(x)
            x = self.activ(self.bn(x))
            return x

    class Residual(nn.Module):
        def __init__(self, k, input_channels, out_channels):
            super().__init__()
            self.pipeline = nn.Sequential(
                GraphConv(k, input_channels, input_channels),
                nn.BatchNorm2d(input_channels),
                nn.LeakyReLU(0.05),

                GraphConv(k, input_channels, out_channels),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.05),

                GraphConv(k, out_channels, out_channels),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.05),
            )

        def forward(self, x):
            return self.pipeline(x)

    class PreProcessBlock(nn.Module):
        def __init__(self, k, kernel_size, input_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv2d(input_channels, out_channels, kernel_size,
                                  padding=(kernel_size//2, kernel_size//2))
            self.activ = nn.LeakyReLU(0.05)
            self.bn = nn.BatchNorm2d(out_channels)

            self.GC = GraphConv(k, out_channels, out_channels) # out_channels -> out_channels

        def forward(self, x):
            x = self.activ(self.conv(x))
            x = self.GC(x)
            x = self.activ(self.bn(x))
            return x

    loss_mse = nn.MSELoss()

    class CNN(nn.Module):
        def __init__(self, k, input_channels, hidden_channels, patch_size=(64, 64)):
            super().__init__()
            self.patch_size = patch_size
            self.preprocessing_blocks = nn.ModuleList([
                PreProcessBlock(k, 3, input_channels, hidden_channels),
                PreProcessBlock(k, 5, input_channels, hidden_channels),
                PreProcessBlock(k, 7, input_channels, hidden_channels),
            ])
            self.residual_1 = Residual(k, hidden_channels*3, hidden_channels)
            self.residual_2 = Residual(k, hidden_channels, hidden_channels)

            self.GC = GraphConv(k, hidden_channels, input_channels)
            self.downsample = nn.Sequential(
                nn.Conv2d(hidden_channels*3, hidden_channels, 1),
                nn.BatchNorm2d(hidden_channels),
                nn.LeakyReLU(0.05),
                nn.Conv2d(hidden_channels, hidden_channels, 1),
                nn.BatchNorm2d(hidden_channels),
                nn.LeakyReLU(0.05),
            )

        def fit_image(self, image):
            processed_image = torch.cat([block(image) for block in self.preprocessing_blocks], dim=1)
            residual_1 = self.residual_1(processed_image)
            result_1 = residual_1 + self.downsample(processed_image)
            residual_2 = self.residual_2(result_1)
            result = residual_2 + result_1
            return [processed_image, residual_1, result, self.GC(result)]

        def forward(self, clear_image, noised_image):
            processed_image, residual_1, residual_2, answer = self.fit_image(clear_image)
            n_processed_image, n_residual_1, n_residual_2, n_answer = self.fit_image(noised_image)
            perceptual_loss = loss_mse(processed_image, n_processed_image) + \
                loss_mse(residual_1, n_residual_1) + \
                loss_mse(residual_2, n_residual_2)
            return n_answer, perceptual_loss

        def forward_eval(self, noised_image):
            _, _, _, answer = self.fit_image(noised_image)
            return answer


        def forward_image(self, noised_image, device, chunks=16):
            p_x, p_y = self.patch_size
            splits = torch.split(torch.stack(torch.split(noised_image, p_x)), p_y, dim=2)
            crops = torch.stack(splits, dim=2)        
            crops = crops.view(-1, 1, p_x, p_y)
            crops_ = torch.split(crops, crops.shape[0]//chunks, dim=0)
            answer = torch.cat([crop + self.forward_eval(crop.cuda(device)).cpu().data for crop in crops_], dim=0)
            a_x, a_y = noised_image.shape
            return torch.clamp(answer, 0, 1).view(a_x, a_y)
        
    return CNN


def get_GCNN():

    class GraphConv(nn.Module):
        def __init__(self, k, input_channels, out_channels, search_area=None):
            super().__init__()
            self.conv1 = nn.Conv2d(input_channels, out_channels, 1)
            self.conv2 = nn.Conv2d(input_channels, out_channels, 3, padding=1)
            self.NLA = NonLocalAggregation(k, input_channels, out_channels)

        def forward(self, x):
            return torch.mean(torch.stack([self.conv1(x),
                                           self.conv2(x),
                                           self.NLA(x)]), dim=0) # here is conv instead of gc


    class PreProcessBlock(nn.Module):
        def __init__(self, input_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv2d(input_channels, out_channels, kernel_size,
                                  padding=(kernel_size//2, kernel_size//2))
            self.activ = nn.LeakyReLU(0.05)
            self.bn = nn.BatchNorm2d(out_channels)

            self.GC = GraphConv(k, out_channels, out_channels) # out_channels -> out_channels

        def forward(self, x):
            x = self.activ(self.conv(x))
            x = self.GC(x)
            x = self.activ(self.bn(x))
            return x

    class Residual(nn.Module):
        def __init__(self, k, input_channels, out_channels):
            super().__init__()
            self.pipeline = nn.Sequential(
                GraphConv(k, input_channels, input_channels),
                nn.BatchNorm2d(input_channels),
                nn.LeakyReLU(0.05),

                GraphConv(k, input_channels, out_channels),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.05),

                GraphConv(k, out_channels, out_channels),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.05),
            )

        def forward(self, x):
            return self.pipeline(x)

    class PreProcessBlock(nn.Module):
        def __init__(self, k, kernel_size, input_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv2d(input_channels, out_channels, kernel_size,
                                  padding=(kernel_size//2, kernel_size//2))
            self.activ = nn.LeakyReLU(0.05)
            self.bn = nn.BatchNorm2d(out_channels)

            self.GC = GraphConv(k, out_channels, out_channels) # out_channels -> out_channels

        def forward(self, x):
            x = self.activ(self.conv(x))
            x = self.GC(x)
            x = self.activ(self.bn(x))
            return x

    loss_mse = nn.MSELoss()

    class GCNN(nn.Module):
        def __init__(self, k, input_channels, hidden_channels, patch_size=(64, 64)):
            super().__init__()
            self.patch_size = patch_size
            self.preprocessing_blocks = nn.ModuleList([
                PreProcessBlock(k, 3, input_channels, hidden_channels),
                PreProcessBlock(k, 5, input_channels, hidden_channels),
                PreProcessBlock(k, 7, input_channels, hidden_channels),
            ])
            self.residual_1 = Residual(k, hidden_channels*3, hidden_channels)
            self.residual_2 = Residual(k, hidden_channels, hidden_channels)

            self.GC = GraphConv(k, hidden_channels, input_channels)
            self.downsample = nn.Sequential(
                nn.Conv2d(hidden_channels*3, hidden_channels, 1),
                nn.BatchNorm2d(hidden_channels),
                nn.LeakyReLU(0.05),
                nn.Conv2d(hidden_channels, hidden_channels, 1),
                nn.BatchNorm2d(hidden_channels),
                nn.LeakyReLU(0.05),
            )

        def fit_image(self, image):
            processed_image = torch.cat([block(image) for block in self.preprocessing_blocks], dim=1)
            residual_1 = self.residual_1(processed_image)
            result_1 = residual_1 + self.downsample(processed_image)
            residual_2 = self.residual_2(result_1)
            result = residual_2 + result_1
            return [processed_image, residual_1, result, self.GC(result)]

        def forward(self, clear_image, noised_image):
            processed_image, residual_1, residual_2, answer = self.fit_image(clear_image)
            n_processed_image, n_residual_1, n_residual_2, n_answer = self.fit_image(noised_image)
            perceptual_loss = loss_mse(processed_image, n_processed_image) + \
                loss_mse(residual_1, n_residual_1) + \
                loss_mse(residual_2, n_residual_2)
            return n_answer, perceptual_loss

        def forward_eval(self, noised_image):
            _, _, _, answer = self.fit_image(noised_image)
            return answer


        def forward_image(self, noised_image, device, chunks=16):
            p_x, p_y = self.patch_size
            splits = torch.split(torch.stack(torch.split(noised_image, p_x)), p_y, dim=2)
            crops = torch.stack(splits, dim=2)        
            crops = crops.view(-1, 1, p_x, p_y)
            crops_ = torch.split(crops, crops.shape[0]//chunks, dim=0)
            answer = torch.cat([crop + self.forward_eval(crop.cuda(device)).cpu().data for crop in crops_], dim=0)
            a_x, a_y = noised_image.shape
            return torch.clamp(answer, 0, 1).view(a_x, a_y)
    return GCNN


def get_GCNN_baseline():

    class GraphConv(nn.Module):
        def __init__(self, k, input_channels, out_channels, search_area=None):
            super().__init__()
            self.conv1 = nn.Conv2d(input_channels, out_channels, 1)
            self.conv2 = nn.Conv2d(input_channels, out_channels, 3, padding=1)
            self.NLA = NonLocalAggregation(k, input_channels, out_channels)

        def forward(self, x):
            return torch.mean(torch.stack([self.conv1(x),
                                           self.conv2(x),
                                           self.NLA(x)]), dim=0) # here is gc


    class PreProcessBlock(nn.Module):
        def __init__(self, input_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv2d(input_channels, out_channels, kernel_size,
                                  padding=(kernel_size//2, kernel_size//2))
            self.activ = nn.LeakyReLU(0.05)
            self.bn = nn.BatchNorm2d(out_channels)

            self.GC = GraphConv(k, out_channels, out_channels) # out_channels -> out_channels

        def forward(self, x):
            x = self.activ(self.conv(x))
            x = self.GC(x)
            x = self.activ(self.bn(x))
            return x

    class Residual(nn.Module):
        def __init__(self, k, input_channels, out_channels):
            super().__init__()
            self.pipeline = nn.Sequential(
                GraphConv(k, input_channels, out_channels),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.05),
            )

        def forward(self, x):
            return self.pipeline(x)

    class PreProcessBlock(nn.Module):
        def __init__(self, k, kernel_size, input_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv2d(input_channels, out_channels, kernel_size,
                                  padding=(kernel_size//2, kernel_size//2))
            self.activ = nn.LeakyReLU(0.05)
            self.bn = nn.BatchNorm2d(out_channels)

            self.GC = GraphConv(k, out_channels, out_channels) # out_channels -> out_channels

        def forward(self, x):
            x = self.activ(self.conv(x))
            x = self.GC(x)
            x = self.activ(self.bn(x))
            return x
    loss_mse = nn.MSELoss()

    class GCNN(nn.Module):
        def __init__(self, k, input_channels, hidden_channels, patch_size=(64, 64)):
            super().__init__()
            self.patch_size = patch_size
            self.preprocessing_blocks = nn.ModuleList([
                PreProcessBlock(k, 3, input_channels, hidden_channels),
                PreProcessBlock(k, 5, input_channels, hidden_channels),
                PreProcessBlock(k, 7, input_channels, hidden_channels),
            ])
            self.residual_1 = Residual(k, hidden_channels*3, hidden_channels)
            self.residual_2 = Residual(k, hidden_channels, hidden_channels)

            self.GC = GraphConv(k, hidden_channels, input_channels)
            self.downsample = nn.Sequential(
                nn.Conv2d(hidden_channels*3, hidden_channels, 1),
                nn.BatchNorm2d(hidden_channels),
                nn.LeakyReLU(0.05),
            )

        def fit_image(self, image):
            processed_image = torch.cat([block(image) for block in self.preprocessing_blocks], dim=1)
            residual_1 = self.residual_1(processed_image)
            result_1 = residual_1 + self.downsample(processed_image)
            residual_2 = self.residual_2(result_1)
            result = residual_2 + result_1
            return [processed_image, residual_1, result, self.GC(result)]

        def forward(self, clear_image, noised_image):
            processed_image, residual_1, residual_2, answer = self.fit_image(clear_image)
            n_processed_image, n_residual_1, n_residual_2, n_answer = self.fit_image(noised_image)
            perceptual_loss = loss_mse(processed_image, n_processed_image) + \
                loss_mse(residual_1, n_residual_1) + \
                loss_mse(residual_2, n_residual_2)
            return n_answer, perceptual_loss

        def forward_eval(self, noised_image):
            _, _, _, answer = self.fit_image(noised_image)
            return answer


        def forward_image(self, noised_image, device, chunks=16):
            p_x, p_y = self.patch_size
            splits = torch.split(torch.stack(torch.split(noised_image, p_x)), p_y, dim=2)
            crops = torch.stack(splits, dim=2)        
            crops = crops.view(-1, 1, p_x, p_y)
            crops_ = torch.split(crops, crops.shape[0]//chunks, dim=0)
            answer = torch.cat([crop + self.forward_eval(crop.cuda(device)).cpu().data for crop in crops_], dim=0)
            a_x, a_y = noised_image.shape
            return torch.clamp(answer, 0, 1).view(a_x, a_y)

    return GCNN



def get_GCNN_fast_baseline():

    class GraphConv(nn.Module):
        def __init__(self, k, input_channels, out_channels, search_area=None):
            super().__init__()
            self.conv2 = nn.Conv2d(input_channels, out_channels, 3, padding=1)
            self.NLA = NonLocalAggregation(k, input_channels, out_channels)

        def forward(self, x):
            return torch.mean(torch.stack([self.conv2(x),
                                           self.NLA(x)]), dim=0) # here is gc


    class PreProcessBlock(nn.Module):
        def __init__(self, input_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv2d(input_channels, out_channels, kernel_size,
                                  padding=(kernel_size//2, kernel_size//2))
            self.activ = nn.LeakyReLU(0.05)
            self.bn = nn.BatchNorm2d(out_channels)

            self.GC = GraphConv(k, out_channels, out_channels) # out_channels -> out_channels

        def forward(self, x):
            x = self.activ(self.conv(x))
            x = self.GC(x)
            x = self.activ(self.bn(x))
            return x

    class Residual(nn.Module):
        def __init__(self, k, input_channels, out_channels):
            super().__init__()
            self.pipeline = nn.Sequential(
                GraphConv(k, input_channels, out_channels),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.05),
            )

        def forward(self, x):
            return self.pipeline(x)

    class PreProcessBlock(nn.Module):
        def __init__(self, k, kernel_size, input_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv2d(input_channels, out_channels, kernel_size,
                                  padding=(kernel_size//2, kernel_size//2))
            self.activ = nn.LeakyReLU(0.05)
            self.bn = nn.BatchNorm2d(out_channels)

            self.GC = GraphConv(k, out_channels, out_channels) # out_channels -> out_channels

        def forward(self, x):
            x = self.activ(self.conv(x))
            x = self.GC(x)
            x = self.activ(self.bn(x))
            return x
    loss_mse = nn.MSELoss()

    class GCNN(nn.Module):
        def __init__(self, k, input_channels, hidden_channels, patch_size=(64, 64)):
            super().__init__()
            self.patch_size = patch_size
            self.preprocessing_blocks = nn.ModuleList([
                PreProcessBlock(k, 3, input_channels, hidden_channels),
                PreProcessBlock(k, 5, input_channels, hidden_channels),
                PreProcessBlock(k, 7, input_channels, hidden_channels),
            ])

            self.GC = GraphConv(k, hidden_channels, input_channels)
            self.downsample = nn.Conv2d(hidden_channels, input_channels, 1)

        def fit_image(self, image):
            processed_image = torch.mean(torch.stack([block(image) for block in self.preprocessing_blocks], dim=0), dim=0)
            result = self.GC(processed_image) + self.downsample(processed_image)
            return [processed_image, result]

        def forward(self, clear_image, noised_image):
            processed_image, answer = self.fit_image(clear_image)
            n_processed_image, n_answer = self.fit_image(noised_image)
            perceptual_loss = loss_mse(processed_image, n_processed_image)
            return n_answer, perceptual_loss

        def forward_eval(self, noised_image):
            _, answer = self.fit_image(noised_image)
            return answer


        def forward_image(self, noised_image, device, chunks=16):
            p_x, p_y = self.patch_size
            splits = torch.split(torch.stack(torch.split(noised_image, p_x)), p_y, dim=2)
            crops = torch.stack(splits, dim=2)        
            crops = crops.view(-1, 1, p_x, p_y)
            crops_ = torch.split(crops, crops.shape[0]//chunks, dim=0)
            answer = torch.cat([crop + self.forward_eval(crop.cuda(device)).cpu().data for crop in crops_], dim=0)
            a_x, a_y = noised_image.shape
            return torch.clamp(answer, 0, 1).view(a_x, a_y)

    return GCNN

