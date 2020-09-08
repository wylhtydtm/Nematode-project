#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:19:53 2020
Create new datasets for the model

@author: liuziwei
"""
import pandas as pd


training_set = pd.read_csv('/Users/liuziwei/Desktop/Training/train/is_multiple.csv')

training_set.info()  #total 22014 images

training_set['is_multiple'].values.sum()  # 6213 Multiple worms
(~training_set['is_multiple']).values.sum()  #15801 Single worms


##

validation_set = pd.read_csv('/Users/liuziwei/Desktop/Training/validation/is_multiple.csv')

validation_set.info()  #total 496 images

validation_set['is_multiple'].values.sum()  # True, is multiple 125 images
(~validation_set['is_multiple']).values.sum()  # Flase, is single 371 images

##

test_set = pd.read_csv('/Users/liuziwei/Desktop/Training/test/is_multiple.csv')

test_set.info()  #total 503 worms

test_set['is_multiple'].values.sum()  # 151 Multiple worms
(~test_set['is_multiple']).values.sum()  #352 Single worms



#%%
#Make a new training dataset, 6213 non-worm,15801 single worms, 6213 multiple worms

newtraining_set =training_set.drop(columns='n_skels')

print(newtraining_set.head())

new_training_set = newtraining_set.iloc[:, 1]*1

#Replace is _multiple to annotations, 0 is non-worms, 1 is single worm, 2 is multiple worms
new_training_set.rename(columns = {'is_multiple': 'annotations'}, inplace = True)


#%%
# negative data
import os, os.path,shutil

image_dir = '/Users/liuziwei/Desktop/Training/negative/img'
negative_images = os.listdir(image_dir)

categories = [['training']*6213, ['validation']*125, ['test']*151, ['extra']*3183]
categories = [item for sublist in categories for item in sublist]

#split data in negative data into 6213, 125, 151, 3183)

output_path = ('/Users/liuziwei/Desktop/Training/negative/img_divided')

for i, image in enumerate(negative_images):
    
    category = categories[i]
    category_path = os.path.join(output_path, category)
    
    if not os.path.exists(category_path):
        os.makedirs(category_path)
    
    old_path = os.path.join(image_dir, image)
    new_path = os.path.join(category_path, image)
    shutil.copy(old_path, new_path)
    
#Rename every images in each folder by its index.

#%%
training_dir = os.chdir('/Users/liuziwei/Desktop/Training/negative/img_divided/training')
i=22014
for file in os.listdir(training_dir):
    src = file
    dst =  str(i) +'.png'
    os.rename(src, dst)
    i +=1
    
new_path_1 = os.path.join()

validation_dir = '/Users/liuziwei/Desktop/Training/negative/img_divided/validation'
validationimages = os.listdir(validation_dir)

m=610
for image in validationin:
    src = image
    dst =  str(m) +'.png'
    os.rename(src, dst)
    m +=1
    
print(validationimages)
   
#%%
import os, os.path,shutil
import pandas as pd

validimage_dir = '/Users/liuziwei/Desktop/Training/validation/img'
validset= os.listdir(validimage_dir)

val_categories= [['single worm'],['multiple worms']]
output_path = ('/Users/liuziwei/Desktop/Training/negative/img_divided')

df = pd.read_csv('/Users/liuziwei/Desktop/Training/validation/is_multiple.csv')
multiple_df = df.loc[df['is_multiple'] == True]
single_df = df.loc[df['is_multiple'] == False]

single_dir = os.path.join(validimage_dir, 'single' )
multiple_dir = os.path.join(validimage_dir , 'multiple')

for i, row in multiple_df.iterrows():
    
    if not os.path.exists(multiple_dir):
        os.makedirs(multiple_dir)
    
    image = os.path.join(validimage_dir, '%s.png' % row.worm_number)
    
    shutil.copy(image, multiple_dir)


#%%
def show_dataset(dataset, n=6):
    img = dataset[np.random.randint(0, 5000)][0]
    plt.imshow(img)
    plt.axis('off')
    
show_dataset(train_data)

#%%
#TO create training dataet with worms and negative dataset in a single folder

df =pd.read_csv('/Users/liuziwei/Desktop/Training/train/labels.csv')

cols2keep = [col for col in df.columns if col != "n_skels"]
df_new = df[cols2keep] *1

#path2worms = '/Users/liuziwei/Desktop/Training/train/img'
from pathlib import Path
path2negatives = Path('/Users/liuziwei/Desktop/Training/train/negative')
files2add = list(path2negatives.rglob('*.png'))
filenames = [f.stem for f in files2add]
filenames = sorted(filenames)
assert (df_new.worm_number == df_new.index).all()
df_negative = pd.DataFrame({'worm_number':filenames, 'is_multiple':2})
df_negative['worm_number'] = df_negative['worm_number'].astype(int)
df_combined = pd.concat([df_new, df_negative], axis=0, ignore_index=True)
df_combined.to_csv('/Users/liuziwei/Desktop/Training/train/df_combined.csv', index=False)

#%%
#To create validation dataset with worms and negative dataset in a single folder.

validation_dir = '/Users/liuziwei/Desktop/Training/validation/negative'
validationimages = list(Path(validation_dir).rglob('*.png'))

offset = 496
for i, file in enumerate(validationimages):
    #print(file)
    directory = str(file.parent)
    old_name = file.stem
    ext = file.suffix
    new_name = old_name.replace(old_name, str(i + offset))
    new_path = Path(os.path.join(directory, new_name + ext))
    new_path = f"{directory}/{new_name}{file.suffix}"
    file.rename(Path(new_path))
    
offset = 496  
for i, image in enumerate(os.listdir(validation_dir)):
    if '.png' in image:
        src = os.path.join(validation_dir, image)
        dst = os.path.join(validation_dir, str(i + offset) + '.png')
        os.rename(src, dst)


file.replace("", "")

validationimages = [f.name for f in validationimages]
#validationimages = os.listdir(validation_dir)

import numpy as np

start_num = 496
new_names = np.arange(start_num, start_num + len(validationimages))
for i, name in enumerate(new_names):
    old_name = validationimages[i] 
    new_name = str(name) + '.png'
    print(i, old_name, new_name)

#To update csv file in validation folder
    
df1=pd.read_csv('/Users/liuziwei/Desktop/Training/validation/labels.csv')

cols2keep = [col for col in df1.columns if col != "n_skels"]
df1_new = df1[cols2keep] *1

#path2worms = '/Users/liuziwei/Desktop/Training/validation/img'

from pathlib import Path
path2negatives = Path('/Users/liuziwei/Desktop/Training/validation/negative')
files2add = list(path2negatives.rglob('*.png'))
filenames = [f.stem for f in files2add]
filenames = sorted(filenames)
assert (df1_new.worm_number == df1_new.index).all()

df1_negative = pd.DataFrame({'worm_number':filenames, 'is_multiple':2})
df1_negative['worm_number'] = df1_negative['worm_number'].astype(int)
df1_combined = pd.concat([df1_new, df1_negative], axis=0, ignore_index=True)
df1_combined.to_csv('/Users/liuziwei/Desktop/Training/validation/df1_combined.csv', index=False)

#%%To create test dataset
test_dir = '/Users/liuziwei/Desktop/Training/test/negative'
testimages = list(Path(test_dir).rglob('*.png'))

offset = 503
for i, image in enumerate(os.listdir(test_dir)):
    if '.png' in image:
        src = os.path.join(test_dir, image)
        dst = os.path.join(test_dir, str(i + offset) + '.png')
        os.rename(src, dst)


#To update csv file in test folder
    
df2=pd.read_csv('/Users/liuziwei/Desktop/Training/test/labels.csv')

cols2keep = [col for col in df2.columns if col != "n_skels"]
df2_new = df2[cols2keep] *1

#path2worms = '/Users/liuziwei/Desktop/Training/test/img'

from pathlib import Path
path2testnegatives = Path('/Users/liuziwei/Desktop/Training/test/negative')
files2add = list(path2testnegatives.rglob('*.png'))
filenames = [f.stem for f in files2add]
filenames = sorted(filenames)
assert (df2_new.worm_number == df2_new.index).all()

df2_negative = pd.DataFrame({'worm_number':filenames, 'is_multiple':2})
df2_negative['worm_number'] = df2_negative['worm_number'].astype(int)
df2_combined = pd.concat([df2_new, df2_negative], axis=0, ignore_index=True)
df2_combined.to_csv('/Users/liuziwei/Desktop/Training/test/df2_combined.csv', index=False)




