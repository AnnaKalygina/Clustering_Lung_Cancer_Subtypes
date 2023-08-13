#!/usr/bin/env python
# coding: utf-8

# ## K means 
# 

# In[586]:


import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import ward, fcluster


# In[587]:


declust_res = pd.read_csv('EPIC_results.csv', index_col = 0)


# In[598]:


samples = list(declust_res.index.values)
#def hierarchical_clustering(declust_res, samples):
Z = linkage(declust_res, method='complete', metric='euclidean')

    # Plot the dendrogram
plt.figure(figsize=(10, 15))
dendrogram(Z, labels=samples, leaf_rotation=90)
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.title("Deconvolution Clustering")
plt.tight_layout()
plt.show()

#hierarchical_clustering(declust_res, samples)
Z.shape


# In[585]:


#delete the two outliers data 
declust_res = declust_res.drop({'LCNEC_S00733','LCNEC_S01580'})


# In[428]:


# see which samples correspond to which clusters 
from scipy.cluster.hierarchy import ward, fcluster

label = fcluster(Z, 4, criterion='maxclust')

df_clst = pd.DataFrame()
df_clst['index']  = declust_res.index.values
df_clst['label']  = label
clustered_samples = pd.Series()

for i in range(4):
    elements = df_clst[df_clst['label']==i+1]['index'].tolist()
    print(elements)


# In[425]:


clustered_samples


# In[577]:


# Reset sample names into two batches
sample_names = list(declust_res.index.values)
for i in range(len(sample_names)):
    if "LCNEC" in sample_names[i]:
        sample_names[i] = "LCNEC"
    else:
        sample_names[i] = "SCLC"
print(sample_names)
declust_res = declust_res.set_axis(sample_names, axis='index')


# In[578]:


samples = list(declust_res.index.values)
#def hierarchical_clustering(declust_res, samples):
Z = linkage(declust_res, method='complete', metric='euclidean')

    # Plot the dendrogram
    

plt.figure(figsize=(10, 5))
dendro = dendrogram(Z, labels=samples)
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.title("Deconvolution Clustering")
plt.tight_layout()
plt.show()


#hierarchical_clustering(declust_res, samples)
Z.shape


# In[579]:


seaborn.clustermap(declust_res.T,method='complete',metric='euclidean')


# In[436]:


from scipy.stats import mannwhitneyu
pouf_data = pd.read_csv('data_final_labelled.csv', index_col =0 )


# In[440]:


pouf_data = pouf_data[[ 'label']]
pouf_data


# In[486]:


pou2f3_samples = pouf_data[pouf_data['label'] == 1].index.tolist()
non_pou2f3_samples = pouf_data[pouf_data['label'] == 0].index.tolist()


# In[529]:


epic_res = pd.read_csv('EPIC_results.csv', index_col = 0)


# In[532]:


epic_res 


# In[533]:


labels = pouf_data['label']


# In[536]:


results = {}
for cell_types in epic_res.columns:
    
    result = mannwhitneyu(epic_res.loc[labels[labels== 1].index, cell_types],
                          epic_res.loc[labels[labels!= 1].index, cell_types],
                         )
    
    results[cell_types] = result[1]


# In[540]:


results


# In[551]:


from statsmodels.sandbox.stats.multicomp import multipletests
df_results = pd.DataFrame.from_dict(results, orient = 'index', columns=['p-value'])


# In[555]:


p_adjusted = df_results*8


# In[558]:


p_adjusted.rename(columns = {'p-value':'p-value_adjusted'}, inplace = True)
p_adjusted


# In[554]:


significant_corr = np.sum(p_adjusted < 0.05)
significant_corr

