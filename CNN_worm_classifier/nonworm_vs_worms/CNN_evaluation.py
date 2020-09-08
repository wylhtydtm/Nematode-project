#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 19:36:59 2020

@author: liuziwei
"""

#data preprocessing

import torch
import tables
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
#%%
   
 def measure_performance(predictions, labels):
    """
    I think there's scikit learn functions for this
    but found out after writing the function
    """
    # go logical for ease
    predictions = predictions.astype(bool)
    labels = labels.astype(bool)
    # true positives
    tp = np.logical_and(predictions, labels).sum()
    # true negatives
    tn = np.logical_and(~predictions, ~labels).sum()
    # false positives
    fp = np.logical_and(predictions, ~labels).sum()
    # false negatives
    fn = np.logical_and(~predictions, labels).sum()
    # accuracy
    accuracy = (tp + tn) / len(predictions)
    print(f"accuracy = {accuracy}")
    # precision
    precision = tp / (tp + fp)
    print(f"precision = {precision}")
    # recall
    recall = tp / (tp + fn)
    print(f"recall = {recall}")
    # F1
    f1 = 2*tp / (2*tp + fp + fn)
    print(f"F1 score = {f1}")
    return


#%%
def img_rescale(img, for_pillow= True):
    """
    Rescale the image between 0 and 1, make it 3D if it was just 2D.
    Unlike prep_for_pytorch, no need to make it 4d (batches) because
    the images will be loaded through the dataloader, and that will already
    create the 4d batches.
    In tierpsy, I manually make a N_images x channels x width x height batches,
    and the input to prep_for_pytorch is n_images x w x h (because grayscale)
    While here the ndim==3 refers to channels...
    If you don't use the dataloader, you'll still need to add one dimension
    in the appropriate place"""
    assert img.ndim==2, 'img_rescale only works with 2d array for now'
    img = img - img.min()
    img = img / img.max()
    if for_pillow:
        img *= 255
        img = img.astype(np.uint8)
    else:
        img = img.astype(np.float32)[None, :, :] # c,w,h
    return img

#from tierpsy.analysis.ske_init.filterTrajectModel import shift_and_normalize: 
def shift_and_normalize(data):  #Preprocessing step 
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
#%%

class new_dataset(Dataset):

    def __init__(self, hdf5_filename, which_set='train', transform=None):

        self.fname = hdf5_filename
        self.set_name = which_set
        # get labels info
        with tables.File(self.fname, 'r') as fid:
            tmp = pd.DataFrame.from_records(
                fid.get_node('/'+self.set_name)['sample_data'].read())
        self.label_info = tmp[['img_row_id', 'is_worm', 'is_avelinos']]
        # size in hdf5 file is 160x160 (in theory), but we train on 80x80
        self.roi_size = 80  # size we want to train on
        with tables.File(self.fname, 'r') as fid:
            dataset_size = fid.get_node('/train/mask').shape[1]
        pad = (dataset_size - self.roi_size)/2
        self.dd = [pad, dataset_size-pad]
        # any transform?
        self.transform = transform

    def __len__(self):
        return len(self.label_info)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # I could just use index because img_row_id is the same as the index of
        # label_info, but just in case we ever want to shuffle...
        label_info = self.label_info.iloc[index]
        img_row_id = label_info['img_row_id']
        # read images from disk
        with tables.File(self.fname, 'r') as fid:
            roi_data = fid.get_node(
                '/' + self.set_name + '/mask')[img_row_id,
                                               self.dd[0]:self.dd[1],
                                               self.dd[0]:self.dd[1]].copy()

        # shift_and_normalize wants a float, and pytorch a single, use single
        img = roi_data.astype(np.float32)
        img = shift_and_normalize(img)

        # as of now, the model works even without PIL
        # but transform only works with pil, so:
        if self.transform:  # if any transforms were given to initialiser
            img = img_rescale(img, for_pillow=True)
            img = Image.fromarray(img)
            #img = img.convert(mode='RGB')
            img = self.transform(img)
        else:
            img = img_rescale(img, for_pillow=False)

        # read labels too
        labels = label_info['is_worm']
        labels = np.array(labels, dtype=np.float32).reshape(-1, 1)
        labels = torch.from_numpy(labels)
        #read whether it comes from the old Phenix dataset
        #is_hydra = ~label_info['is_avelinos']
        #is_hydra = np.array(is_hydra, dtype=np.float32).reshape(-1, 1)
        #is_hydra = torch.from_numpy(is_hydra)

        return img, labels #, is_hydra


if __name__ == "__main__":

    # where are things?
    hd = Path('/Users/liuziwei/Desktop/Hydra_Phenix_cnn')
    # hd = hd / 'Analysis/Ziweis_NN/Hydra_Phenix_combined'
    #hd = Path.cwd()
    fname = hd / 'Hydra_Phenix_dataset.hdf5'
    # path to the saved model_state (to test feed forward)
   #model_path = ('/Users/lferiani/OneDrive - Imperial College London/'
                 # + 'Analysis/Ziweis_NN/Model/'
                 # + 'notaworm_vs_worm_difficult_aggregate_model_state.pth')

    # parameters
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    batch_size = 5

    # define transforms
    # do we need vertical/hor flip?
    training_transform = transforms.Compose([
        transforms.RandomVerticalFlip(p=0.4),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.ColorJitter(contrast=0.2, hue=0.2),
        transforms.ToTensor()])

    validation_transform = transforms.Compose([
        transforms.RandomVerticalFlip(p=0.4),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.ColorJitter(contrast=0.2, hue=0.2),
        transforms.ToTensor()])

    # create datasets
    train_data = new_dataset(fname, which_set='train',
                             transform=training_transform)
    val_data = new_dataset(fname, which_set='val',
                           transform=validation_transform)
    test_data = new_dataset(fname, which_set='test')


    # create dataloaders
    # num_workers=4 crashes in my spyder but works on pure python
    train_loader = DataLoader(
        train_data, shuffle=True, batch_size=batch_size, num_workers=4)
    val_loader = DataLoader(
        val_data, shuffle=True, batch_size=batch_size, num_workers=4)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, num_workers=1)
    

    class ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_layers = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(), 
                    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
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
                    nn.MaxPool2d(kernel_size=2, stride=2)) 
      
            self.drop_out = nn.Dropout2d(0.5)
            self.fc_layers =nn.Sequential(nn.Linear(12800, 2))# define fully connected layer
        
        def forward(self, x):
            x = self.conv_layers(x) # pass input through conv layers
            x = self.drop_out(x)  
            x = x.view(x.shape[0], -1) 
            x = self.fc_layers(x)# pass  through fully connected layer
            return x 
    
    learning_rate = 0.0001
    epochs = 100
    
    cnn = ConvNet().to(device) # to instantiate model
    criterion = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(cnn.parameters(), lr= learning_rate)
    
    

model = ConvNet().to(device)
device = torch.device('cpu')
model.load_state_dict(torch.load('/Users/liuziwei/Downloads/model_state_isworm_20200615.pth', map_location=device))
model.eval()
    


labels = []
predictions = []
#is_hydras = []
with torch.no_grad():
    for images, labs in test_loader:
        images = images.to(device)
        preds = model(images)
        preds = torch.argmax(preds, axis=1)
        predictions.append(preds)
        labels.append(labs)
        
    # concatenate accumulators into np arrays for ease of use
predictions = np.concatenate(predictions, axis=0)
labels = np.concatenate(labels, axis=0).squeeze()
#is_hydras = np.concatenate(is_hydras, axis=0).squeeze().astype(bool)

    # measure performance
print("\nPerformance on all data")
measure_performance(predictions, labels)

    
    
labels = ['non_worm', 'worm']
cm = confusion_matrix(labels, predictions)
print(cm)


fig, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels)
plt.ylabel('True')
plt.xlabel('Predicted')
plt.title('Confusion matrix for worm and non-worm classification')
plt.show(block=False)

 #%%

#For phenix and hydra datasets separately


label_1 =['non_worm_hydra','worm_hydra','non_worm_phenix','worm_phenix']



cm = confusion_matrix(labels_1, predictions)
print(cm)
labels_1 = []
predictions_1 = []
is_hydras_1 = []
with torch.no_grad():
    for images, labs, is_h in test_loader:
        images = images.to(device)
        preds = model(images)
        preds = torch.argmax(preds, axis=1)
        predictions_1.append(preds)
        labels_1.append(labs)
        is_hydras_1.append(is_h)
    # concatenate accumulators into np arrays for ease of use
predictions_1 = np.concatenate(predictions_1, axis=0)
labels_1 = np.concatenate(labels_1, axis=0).squeeze()
is_hydras_1 = np.concatenate(is_hydras_1, axis=0).squeeze().astype(bool)

    # measure performance
print("\nPerformance on Hydra data")
measure_performance(predictions_1[is_hydras_1], labels_1[is_hydras_1])
print("\nPerformance on old data")
measure_performance(predictions_1[~is_hydras_1], labels_1[~is_hydras_1])

    

    











