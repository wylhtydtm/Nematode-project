#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:02:41 2020

@author: liuziwei classes ={0: single, 1: multiple, 2: negative}
"""


from __future__ import print_function, division
import os
import torch
import numpy as np
from  PIL import Image, ImageDraw
import pandas as pd
from skimage import io, transform
from torchvision import transforms, utils
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

plt.ion()  #interactive mode


class full_dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform= None):
        self.label_info = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.image_arr = np.asarray(self.label_info.iloc[:, 0])  #First column, image paths

    def __len__(self):
        return len(self.label_info)
    
    def __getitem__(self,index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        single_img_number = self.image_arr[index]  #Get image name from df
        
        single_img_name = os.path.join(self.root_dir, '%s.png' % single_img_number)

        image = Image.open(single_img_name)
        
        labels = self.label_info.iloc[index, 1]
        labels = np.array([labels])
        labels = labels.astype('float32').reshape(-1,1)#Multiple= 1, single =0
        labels = torch.from_numpy(labels)
       
        if self.transform: #if any transforms were given to initialiser
            image = self.transform(image) 
        
        return image, labels
    
training_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(contrast=0.2, hue=0.2),
        transforms.ToTensor()])
    
validation_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor()])
    
    
train_data = full_dataset(root_dir ='/Users/liuziwei/Desktop/Training/train/img_combined',
                             csv_file ='/Users/liuziwei/Desktop/Training/train/df_combined.csv', transform=training_transform)

val_data = full_dataset(root_dir ='/Users/liuziwei/Desktop//Training/validation/img_combined',
                      csv_file ='/Users/liuziwei/Desktop/Training/validation/df1_combined.csv', transform=validation_transform) 

batch_size = 128

#Make training dataloader to random samples of our data, so our model wont have to deal wih the 

train_loader = torch.utils.data.DataLoader(train_data, shuffle= True, batch_size=batch_size, num_workers=4)
#Make validation dataloader
val_loader= torch.utils.data.DataLoader(val_data, shuffle= True, batch_size=batch_size, num_workers=4)

classes = ('Non_worm', 'Single_worm', 'Multiple_worm')

class ConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = torch.nn.Sequential(
                torch.nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=1),
                torch.nn.BatchNorm2d(16),# conv layer
                torch.nn.ReLU(), 
                torch.nn.MaxPool2d(kernel_size=2, stride=2),# activation layer
                torch.nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1),
                torch.nn.BatchNorm2d(32),# conv layer taking the output of the previous layer
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Conv2d(128,256, kernel_size=2, stride=1, padding=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 512, kernel_size =2, stride=1, padding=1),
                torch.nn.ReLU()) # activation layer
        
        self.drop_out = nn.Dropout2d(0.5)
        self.fc_layers = torch.nn.Sequential(torch.nn.Linear(512*7*7, 3))# define fully connected layer
     
    def forward(self, x):
        x = self.conv_layers(x) # pass input through conv layers
        x = self.drop_out(x)  
        x = x.view(x.shape[0], -1) # flatten output for fully connected layer, batchize,-1 do whatever it needs to be 
        x = self.fc_layers(x)# pass  through fully connected layer
        x = F.softmax(x, dim=1) #softmax activation function on outputs, get probability disatribution on output, all ouputs add to 1
        return x 

learning_rate = 0.0001
epochs = 100  # gradient descent that controls no of complete passes through the training dataset

cnn = ConvNet().to('cpu') # to instantiate model
criterion = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(cnn.parameters(), lr= learning_rate) # Use Adam optimiser, passing it to the prameters of your model and learning rate

