#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 23:02:33 2020

@author: liuziwei
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.decomposition import PCA
from pathlib import Path
from sklearn.decomposition import PCA as RandomizedPCA
import os

#%%

#PCA is a linear diemsnional reduction technique to extract information from a high-dimensional space by projecting it intoa lower-dimensional sub-space


sig_features= pd.read_csv('/Volumes/Ashur Pro2/NoComp_rerun/feature_summaries_shifted10sTW/t_test_paired_tw01_shifted_10s_aftercorrection.csv', index_col=0)

sigfeats=list(sig_features.index)
sigfeats.append('window_id')
sigfeats.append('imgstore_name_bluelight')
sigfeats.append('well_name')

feat_summary= pd.read_csv('/Volumes/Ashur Pro2/NoComp_rerun/feature_summaries_shifted10sTW/updated_NoComp_shifted.csv', index_col=0)

sigfeat_df = feat_summary.filter(sigfeats)

sigfeat_df_grouped=[x for window_id, x in sigfeat_df.groupby(by='window_id')]

sigfeat_tw0 = sigfeat_df_grouped[0].reset_index(drop=True)
sigfeat_tw1 = sigfeat_df_grouped[1].reset_index(drop=True)
sigfeat_tw2 = sigfeat_df_grouped[2].reset_index(drop=True)


pre_sti_mean= sigfeat_tw0.mean(axis=0)
blue_light_mean = sigfeat_tw1.mean(axis=0)

feature_mean= pd.DataFrame(columns=[ 'pre_sti','blue_light'])
feature_mean['pre_sti'] = pre_sti_mean
feature_mean['blue_light'] = blue_light_mean

#Scale features
feat_merge= pd.merge(sigfeat_tw0, sigfeat_tw1, how ='outer')
feat_merge_all = pd.merge(feat_merge, sigfeat_tw2, how='outer')

feat_merge_1= feat_merge.drop(columns=['window_id','imgstore_name_bluelight','well_name'])
feat_merge_all_1= feat_merge_all.drop(columns=['window_id','imgstore_name_bluelight','well_name'])


feat_merge_scaled= StandardScaler().fit_transform(feat_merge_1)
#Check the normalized data  #shape= (38, 479)
np.mean(feat_merge_scaled),np.std(feat_merge_scaled)  # (-1.1290756719145039e-17, 1.0)   to (1.4638694973631335e-17, 1.0)

pca =PCA().fit(feat_merge_scaled)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of PCs')
plt.ylabel('cumulative explained variance')

plt.title('Pre-stimulus and first blue light stimulus') 'd need aboout 15 components to retain 90% of the variance

pca_2comp= PCA(n_components=2)
pca_transform= pca_2comp.fit_transform(feat_merge_scaled)
pca_2comp.explained_variance_ratio_ # array([0.54912074, 0.06706042, 0.05229013, 0.04254121, 0.03191877])

plt.plot(pca_transform[0:19,0],pca_transform[0:19,1], 'o', markersize=7, color='tab:brown', alpha=0.8, label='pre-sti')
plt.plot(pca_transform[19:38,0], pca_transform[19:38,1], '^', markersize=7, color='tab:blue', alpha=0.8, label='blue-light')
plt.xlabel('PC1 (54.9%)')
plt.ylabel('PC2 (6.7%)')
plt.legend(loc='upper left')
plt.title('PCA(significant features)')

#array([0.53962999, 0.07110877]) so 40% information will be lost by projecting the data to a 2-dimensional data

PCA = RandomizedPCA(n_components=15, svd_solver='randomized', whiten=True).fit(feat_merge_scaled)
components= PCA.transform(feat_merge_scaled)
projected_pca= PCA.inverse_transform(components)
PCA.explained_variance_ratio_
 


#%% for pre-stimulus, bluelight and post-stimulus
feat_merge_scaled_all= StandardScaler().fit_transform(feat_merge_all_1)
#Check the normalized data  #shape= (38, 479)

pca_all=PCA().fit(feat_merge_scaled_all)
plt.plot(np.cumsum(pca_all.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.title('Pre-stimulus, first blue light stimulus and post-stimulus') 

pca_all= PCA(n_components=2)
pca_transform_all= pca_all.fit_transform(feat_merge_scaled_all)
pca_all.explained_variance_ratio_ #array[0.4978965 , 0.07889594, 0.04785069, 0.03982744, 0.03638842])

plt.plot(pca_transform_all[0:19,0],pca_transform_all[0:19,1], 'o', markersize=7, color='tab:brown', alpha=0.8, label='pre-sti')
plt.plot(pca_transform_all[19:38,0], pca_transform_all[19:38,1], '^', markersize=7, color='tab:blue', alpha=0.8, label='bluelight')
plt.plot(pca_transform_all[38:58,0], pca_transform_all[38:58,1], '*', markersize=7, color='tab:orange', alpha=0.8, label='post-light')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.title('PCA (significant features only)')

#%%

loadings = pca_2comp.components_.T * np.sqrt(pca_2comp.explained_variance_)
loading_matrix = pd.DataFrame(loadings, columns=['PC1', 'PC2','PC3'], index= list(feat_merge.columns))
print(loading_matrix)

loadings = pca_5comp.components_.T * np.sqrt(pca_5comp.explained_variance_)
loading_matrix = pd.DataFrame(loadings, columns=['PC1', 'PC2','PC3','PC4','PC5'], index= list(feat_merge.columns))
print(loading_matrix)

loading_matrix["Rank"]= loading_matrix["PC1"].abs().rank(ascending=False)
loading_matrix.sort_values("Rank", inplace = True) 

loading_matrix["Rank_PC2"]= loading_matrix["PC2"].abs().rank(ascending=False)
loading_matrix.sort_values("Rank_PC2", inplace = True) 


loading_matrix["Rank_PC3"]= loading_matrix["PC3"].abs().rank(ascending=False)
loading_matrix.sort_values("Rank_PC3", inplace = True) 
loading_matrix["Rank_PC4"]= loading_matrix["PC4"].abs().rank(ascending=False)
loading_matrix.sort_values("Rank_PC4", inplace = True) 
loading_matrix["Rank_PC5"]= loading_matrix["PC5"].abs().rank(ascending=False)
loading_matrix.sort_values("Rank_PC5", inplace = True) 

hd = Path('/Volumes/Ashur Pro2/NoComp_rerun/feature_summaries_shifted10sTW')
loading_matrix_outpath = os.path.join(hd,  'loading_matrix_forpreandbluelight_PC3.csv')  
loading_matrix.to_csv(loading_matrix_outpath)




loading_matrix = pd.read_csv('/Users/liuziwei/Desktop/SyngentaScreen/loading_matrix_forpreandbluelight.csv')
loading_matrix["Rank_PC2"]= loading_matrix["PC2"].abs().rank(ascending=False)
loading_matrix.sort_values("Rank_PC2", inplace = True) 
loading_matrix_outpath = os.path.join(hd,  'loading_matrix_for_preandbluelight_PC1PC2.csv')  
loading_matrix.to_csv(loading_matrix_outpath)



def loading_plot(coeff, labels):
    #n = coeff.shape[0]
    for i in range(1):
        plt.arrow(0,0, coeff[i, 0], coeff[i,1], head_width=0.05, head_length =0.05, color = '#21918C',alpha = 0.5)
        plt.text(coeff[i,0]*1.15, coeff[i,1]*1.15, labels[i], color= '#21918C', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)  
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid()

fig, ax = plt.subplots(figsize=(7,7))
loading_plot(pca_2comp.components_.T, list(feat_merge.columns))

#%%
#Linear Discriminant Analysis to find a new feature space to project the data in order to maximize classes separately
# It takes class label into consideration. Determine a new dimension to maximize the distance between centroid of each class and minimize the variation within each category 

#rom sklearn.preprocessing import LabelEncode # Use it to enocode categorical variable sto a number
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

X= feat_merge.values
y = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=0)

sc = StandardScaler()
x_train= sc.fit_transform(x_train)
x_test= sc.fit_transform(x_test)

lda = LinearDiscriminantAnalysis(n_components=1)
x_train = lda.fit_transform(x_train, y_train)
x_test= lda.transform(x_test)

classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(x_train, y_train)
y_pred =classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy' +str(accuracy_score(y_test, y_pred))) #Accuracy1.0, with 8 test samples(too small sample size)


lda_var_ratios = lda.explained_variance_ratio_

lda_1 = LinearDiscriminantAnalysis(n_components=None)
X_lda =lda_1.fit(x_train, y_train)
lda_var_ratios = lda_1.explained_variance_ratio_

#Functin calculatingf number of componnets required to pass threshold
def select_n_components(var_ratio, goal_var: float) -> int:
    total_variance = 0.0
    n_components =0
    for explained_variance in var_ratio:
        total_variance += explained_variance
        n_components += 1
        if total_variance >= goal_var:
            break
        
    return n_components

select_n_components(lda_var_ratios, 0.99) #How many  components are required to get above threshold 0.95 or 0.99 of bariance explained

#%%
#Using full set of features and triplet of pre-sti, blue and post

df_groupbywins=[x for window_id, x in feat_summary.groupby(by='window_id')]
feat_tw0 = df_groupbywins[0].reset_index(drop=True)
feat_tw1 = df_groupbywins[1].reset_index(drop=True)
feat_tw2 = df_groupbywins[2].reset_index(drop=True)

feat_tw0= feat_tw0.drop(columns=['imgstore_name_bluelight','well_name','file_id','is_good_well'])
feat_tw1= feat_tw1.drop(columns=['imgstore_name_bluelight','well_name','file_id','is_good_well'])
feat_tw2= feat_tw2.drop(columns=['imgstore_name_bluelight','well_name','file_id','is_good_well'])

#Join three df along rows
feat_joined = pd.merge(feat_tw0, feat_tw1, how ='outer')
feat_join = pd.merge(feat_joined, feat_tw2, how ='outer' )

X=feat_join.drop('window_id', axis=1)

y=feat_join['window_id']

X= X.values
y= y.values

X= sc.fit_transform(X)

lda = LinearDiscriminantAnalysis(n_components=None)
X =lda.fit(X, y)
lda_var_ratios = lda.explained_variance_ratio_ #array([0.62447541, 0.37552459])
select_n_components(lda_var_ratios, 0.99) #Output is 2

sklearn_lda =  LinearDiscriminantAnalysis(n_components=2)
X_lda_sklearn = sklearn_lda.fit(X, y).transform(X)

plt.figure()
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.plot(X_lda_sklearn[0:19,0],X_lda_sklearn[0:19,1], 'o', markersize=7, color='tab:brown', alpha=0.8, label='pre-sti')
plt.plot(X_lda_sklearn[19:38,0],X_lda_sklearn[19:38,1], '^', markersize=7, color='tab:blue', alpha=0.8, label='blue-light')
plt.plot(X_lda_sklearn[38:58,0],X_lda_sklearn[38:58,1], 's', markersize=7, color='tab:orange', alpha=0.8, label='post-sti')

'''
plt.scatter(
    X_lda_sklearn[:,0],
    X_lda_sklearn[:,1],
    c=y,
    cmap='rainbow',
    alpha=0.7,
    edgecolors='b')
''' 
plt.legend()
plt.title('Two LDA components(all features)')

#Plot significant features
sc = StandardScaler()
X_sig=feat_merge_all_1.values
y_sig=feat_merge_all['window_id']
y_sig=y_sig.values

X_sig= sc.fit_transform(X_sig)
X_sig =lda.fit(X_sig, y_sig)
lda_var_ratios = lda.explained_variance_ratio_ #array([0.77676489, 0.22323511])
select_n_components(lda_var_ratios, 0.99) #Output is 2

sklearn_lda_1 =  LinearDiscriminantAnalysis(n_components=2)
X_lda_sklearn_1 = sklearn_lda_1.fit(X_sig,y_sig).transform(X_sig)

plt.figure()


plt.figure()
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.plot(X_lda_sklearn_1[0:19,0],X_lda_sklearn_1[0:19,1], 'o', markersize=7, color='tab:brown', alpha=0.8, label='pre-sti')
plt.plot(X_lda_sklearn_1[19:38,0],X_lda_sklearn_1[19:38,1], '^', markersize=7, color='tab:blue', alpha=0.8, label='blue-light')
plt.plot(X_lda_sklearn_1[38:58,0],X_lda_sklearn_1[38:58,1], 's', markersize=7, color='tab:orange', alpha=0.8, label='post-sti')
plt.legend()
plt.title('Two LDA components(significant features)')




#%%
#Using full features) (pre-sti, blue, and post-sti)
pca_full= PCA(n_components=2)
pca_transform= pca_full.fit_transform(X)
plt.plot(pca_transform[0:19,0],pca_transform[0:19,1], 'o', markersize=7, color='tab:brown', alpha=0.8, label='pre-sti')
plt.plot(pca_transform[19:38,0], pca_transform[19:38,1], '^', markersize=7, color='tab:blue', alpha=0.8, label='bluelight')
plt.plot(pca_transform[38:57,0],pca_transform[38:57,1], 's', markersize=7, color='tab:orange', alpha=0.8, label='post-sti')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.title('Two PCA components(all features)')

'''
pca_full.explained_variance_ratio_
Out[296]: array([0.17170074, 0.09366168])
'''

#  If just fitting data from pre-sti and blue (all features) into a PCA plot
X1=feat_joined.drop('window_id', axis=1)
X1= sc.fit_transform(X1)
pca_2=PCA(n_components=2)
pca_2_transform=pca_2.fit_transform(X1)
pca_2.explained_variance_ratio_ #array([0.19674495, 0.11552345])
plt.plot(pca_2_transform[0:19,0],pca_2_transform[0:19,1], 'o', markersize=7, color='tab:brown', alpha=0.8, label='pre-sti')
plt.plot(pca_2_transform[19:38,0], pca_2_transform[19:38,1], '^', markersize=7, color='tab:blue', alpha=0.8, label='bluelight')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.title('Two PCA components(all features)')


#%% Eleni's script
from significant_feats_Eleni import top_feat_in_LDA, k_significant_feat, k_significant_from_classifier

saveto = hd/'LDA'
saveto.mkdir(parents=True, exist_ok=True)

feat_scaled=sc.fit_transform(feat_merge_all_1.values)
lda = LinearDiscriminantAnalysis(n_components=2, solver='eigen', shrinkage='auto')
y_class =feat_merge_all['window_id'].values


lda_scalings_top_feat0, lda_scalings_scores0 = top_feat_in_LDA(feat_scaled, y_class,ldc=[0], scale=False,
                                                               k=20,feat_names= feat_merge_all_1.columns)

lda_scalings_top_feat1, lda_scalings_scores1 = top_feat_in_LDA(feat_scaled,y_class, ldc=[1], scale=False,
                                                                k=20, feat_names= feat_merge_all_1.columns)

lda_coef_top_feat, lda_coef_scores, support = k_significant_from_classifier(
    feat_scaled, y_class, lda, k=20, scale=None,
    feat_names= feat_merge_all_1.columns, figsize=None, title=None, saveto=None,
    close_after_plotting=False, plot= True)

#%% Find the highly correlated features using Pearson's correlation matrix

import seaborn as sns

PC1_loadings=pd.read_csv('/Volumes/Ashur Pro2/NoComp_rerun/feature_summaries_shifted10sTW/loading_matrix_forpreandbluelight_PC1.csv',index_col=0)
PC2_loadings=pd.read_csv('/Volumes/Ashur Pro2/NoComp_rerun/feature_summaries_shifted10sTW/loading_matrix_forpreandbluelight_PC2.csv',index_col=0)
PC3_loadings=pd.read_csv('/Volumes/Ashur Pro2/NoComp_rerun/feature_summaries_shifted10sTW/loading_matrix_forpreandbluelight_PC3.csv',index_col=0)

PC1_loads=list(PC1_loadings.index)
PC1_40loads=PC1_loads[:40]

PC2_loads=list(PC2_loadings.index)
PC2_20loads=PC2_loads[:20]

PC3_loads=list(PC3_loadings.index)
PC3_20loads=PC3_loads[:20]

df= sigfeat_tw1.drop(columns=['window_id','imgstore_name_bluelight','well_name'])

df=df[PC1_40loads]
# Create correlation matrix

corr_matrix=df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
df.drop(df[to_drop], axis=1, inplace=True)

plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()






































