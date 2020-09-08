#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 10:57:32 2020

@author: liuziwei
"""

from __future__ import print_function, division
import os
import torch
import numpy as np
from  PIL import Image, ImageDraw
import pandas as pd
from skimage import io, transform
import torchvision
from torchvision import transforms, utils
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import PIL
import warnings 
warnings.filterwarnings("ignore")


def shift_and_normalize(data):  #Preprocessing step from Avelino's script
    data_m = data.view(np.ma.MaskedArray)
    data_m.mask = data==0
    if data.ndim == 3:
        sub_d = np.percentile(data, 95, axis=(1,2)) #let's use the 95th as the value of the background
        data_m -= sub_d[:, None, None]
    else:
        sub_d = np.percentile(data, 95)
        data_m -= sub_d
        
    data /= 255
    return data

class new_dataset(Dataset):
    def __init__(self, root_dir, csv_file, transform= None):
        self.root_dir = root_dir
        self.label_info = pd.read_csv(csv_file)
        self.transform = transform
        self.image_arr = np.asarray(self.label_info.iloc[:, 0])  #First column, image paths

    def __len__(self):
        return len(self.label_info)
    
    def __getitem__(self,index):
        if torch.is_tensor(index):
            index = index.tolist()
       
        single_img_number = self.image_arr[index]  #Get image name from df
        single_img_name = os.path.join(self.root_dir, single_img_number)
        #image = Image.open(single_img_name)
        single_image = np.array(Image.open(single_img_name), np.float) #convert from dtype npunit8 to float64 
        single_image_data= shift_and_normalize(single_image)
        rescaled_image_data = single_image_data - single_image_data.min()
        rescaled_image_data /= rescaled_image_data.max()
        rescaled_image_data *= 255
        image = PIL.Image.fromarray(np.uint8(rescaled_image_data))
            
        labels = self.label_info.iloc[index, 2]
        labels = np.array([labels])  
        labels = labels.astype('float32').reshape(-1,1)   
        labels = torch.from_numpy(labels)
       
        if self.transform: #if any transforms were given to initialiser 
            image = self.transform(image)  
        
        return image, labels
    
training_transform = transforms.Compose([
        transforms.RandomVerticalFlip(p=0.4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(contrast=0.2, hue=0.2),
        transforms.ToTensor()])

validation_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(contrast=0.2, hue=0.2),
        transforms.ToTensor()])



classes = ('good labels', 'bad labels')


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(), # activation layer
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),# conv layer taking the output of the previous layer
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128,256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)) # activation layer
        
        self.drop_out = nn.Dropout2d(0.5)
        self.fc_layers =nn.Sequential(nn.Linear(512*5*5, 2))# define fully connected layer
    
    def forward(self, x):
        x = self.conv_layers(x) # pass input through conv layers
        x = self.drop_out(x)  
        x = x.view(x.shape[0], -1) # flatten output for fully connected layer, batchize,-1 do whatever it needs to be 
        x = self.fc_layers(x)# pass  through fully connected layer
        x = F.softmax(x, dim=1) #softmax activation function on outputs, get probability disatribution on output, all ouputs add to 1
        return x 
    
use_cuda = torch.cuda.is_available() #check if gpu is available
device = torch.device("cuda" if use_cuda else "cpu")

learning_rate = 0.0001
epochs = 200  # gradient descent that controls no of complete passes through the training dataset

cnn = ConvNet().to(device) # to instantiate model
criterion = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(cnn.parameters(), lr= learning_rate) # Use Adam optimiser, passing it to the prameters of your model and learning rate





