#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:03:33 2020

@author: liuziwei  For a small subset of three-classes datasets; try evalutaion


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
import matplotlib.pyplot as plt

import warnings  #ignore warnings
warnings.filterwarnings("ignore")

plt.ion()  #interactive mode


class first_dataset(Dataset):
    def __init__(self, csv_path ='/Users/liuziwei/Desktop/5000samples/is_multiple.csv', root_dir= '/Users/liuziwei/Desktop/5000samples/img/', transform= None):
        self.label_info = pd.read_csv('/Users/liuziwei/Desktop/5000samples/is_multiple.csv')
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
        
        labels = self.label_info.iloc[index, 2]
        labels = np.array([labels])*1 
        labels = labels.astype('float32').reshape(-1,1)#Multiple= 1, single =0
        labels = torch.from_numpy(labels)
       
        if self.transform: #if any transforms were given to initialiser
            image = self.transform(image) 
        
        return image, labels


transformations = transforms. Compose([
    transforms.Resize (64),
    transforms.CenterCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])
    

worm_dataset = first_dataset(csv_path = '/Users/liuziwei/Desktop/5000samples/is_multiple.csv',
                             root_dir = '/Users/liuziwei/Desktop/5000samples/img/', transform= transformations)



x= worm_dataset[np.random.randint(0, 5000)][0]

print(x)  #show an example

plt.imshow(x[0].numpy(), cmap = 'gray')
plt.title('Sample #{}'.format(x[0][1]))
plt.show()


fig = plt.figure()

for i in range(len(worm_dataset)):
    ax = plt.subplot(1, 4, i+ 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
      
    if i == 3:
        plt.show()   
    break
    
#%%    
       
train_data, val_data, test_data = torch.utils.data.random_split(worm_dataset,[3900,600,500])   
batch_size = 50 # no of samples working through before updating the internal model parameters

#Make training dataloader to random samples of our data, so our model wont have to deal wih the 
train_loader = torch.utils.data.DataLoader(train_data, shuffle= True, batch_size = batch_size)  # shuffle to crease random examples in positions

#Make validation dataloader
val_loader= torch.utils.data.DataLoader(val_data, shuffle= True, batch_size = batch_size )

#Make test dataloader
test_loader= torch.utils.data.DataLoader(test_data, shuffle = True, batch_size = batch_size)

classes = ('Single worm', 'Multiple worm')

def imshow(img):
    img = image 
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(train_loader)
image, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(1)))


#%%
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
        self.fc_layers = torch.nn.Sequential(torch.nn.Linear(512*7*7, 2))# define fully connected layer
        
    def forward(self, x):
        x = self.conv_layers(x) # pass input through conv layers
        x = self.drop_out(x)  
        x = x.view(x.shape[0], -1) # flatten output for fully connected layer, batchize,-1 do whatever it needs to be 
        x = self.fc_layers(x)# pass  through fully connected layer #softmax activation function on outputs, get probability disatribution on output, all ouputs add to 1
        return x 
    
use_cuda = torch.cuda.is_available() #check if gpu is available
device = torch.device("cuda" if use_cuda else "cpu")

learning_rate = 0.0001
epochs = 50  # gradient descent that controls no of complete passes through the training dataset

cnn = ConvNet().to(device) # to instantiate model
criterion = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(cnn.parameters(), lr= learning_rate) # Use Adam optimiser, passing it to the prameters of your model and learning rate

#Set up training visualisation on a graph
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# to show our model performance on a graph

#%%

#Train
def train(model, epochs, verbose= True, tag ='Loss/Train'):
    for epoch in range(epochs):
        for index ,(inputs, labels) in enumerate(train_loader):
            inputs,labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1)
            labels = labels.long()
            prediction = model(inputs) 
            prediction = prediction.view(labels.size(0), -1)
            
         #pass the data forward through the model
            loss = criterion(prediction, labels) #compute the cost
            if verbose: print('Epoch:', epoch, '\tBatch:', index, '\tLoss', loss.item())
            optimiser.zero_grad()   #rest the gradients of all of the params to zero
            loss.backward()    #backward pass to compute and store all of the params gradients
            optimiser.step()   #update the model's params
            
            writer.add_scalar(tag, loss, epoch*len(train_loader) + index) # write loss to a graph
            
    print ('Training Complete. Final loss =', loss.item())
    
    
train(cnn, epochs)
            
PATH = '/Users/liuziwei/Desktop/5000samples/model.pth'
torch.save(cnn.state_dict(), PATH)
 
# Save the trained model

#%% 
#Reload the weights in the saved model,Sample data
def load_checkpoint(filepath):
    cnn = ConvNet()
    checkpoint = torch.load('/Users/liuziwei/Desktop/5000samples/model.pth')
    model = checkpoint['cnn']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model


model = load_checkpoint('/Users/liuziwei/Desktop/5000samples/model.pth')


cnn.load_state_dict(torch.load('/Users/liuziwei/Desktop/5000samples/model.pth'))
cnn.eval()

#%%
def calc_accuracy(model, dataloader):
    num_correct = 0
    num_examples = len(dataloader.dataset)
    for inputs, labels in dataloader:
        inputs,labels = inputs.to(device),labels.to(device)
        predictions = model(inputs)
        predictions = torch.argmax(predictions, axis=1)
        labels = labels.squeeze()
        num_correct += int(sum(predictions == labels))
        percent_correct = num_correct / num_examples * 100
    return percent_correct


print('Train Accuracy:', calc_accuracy (cnn, train_loader))
print('Validation Accuracy:', calc_accuracy(cnn, val_loader))
print('Test Accuracy:', calc_accuracy(cnn, test_loader))

#%%
#Visualizing Deep Learning Metrics using TensorBoard

network = cnn(*args, **kwargs) # Create an instance of our network
images, labels = next(iter(train_loader)) # unpack two variables: images and labels in train_loader
grid = torchvision.utils.make_grid(images) # create a grid of images that will ultimately see inside tensorbard
writer.add_image('images', grid)
writer.add_graph(cnn, images) #visualize our network inside tensorboard   
writer.close()








 










            

            

            




        
        


        

   
        



