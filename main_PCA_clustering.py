#!/usr/bin/env python
# coding: utf-8

# # Data description

# merged.csv => SCLC and LCNEC in FPKM
# 

# # Plan

# data pre-processing

# In[ ]:


import pandas as pd
from google.colab import files
import numpy as np


# In[ ]:


get_ipython().system('pip install scanorama')


# In[ ]:


import scanorama


# In[ ]:


data1 = pd.read_csv('SCLC.csv', header=2)


# In[ ]:


data_drop = data1.drop(columns='refseq')
data_drop


# In[ ]:


data_new = data_drop.groupby(by=['gene']).mean()
data_new1 = data_new.iloc[26:, :]
data_new1


# In[ ]:


data = pd.read_excel('LCNEC.xlsx', index_col=0, header=2)
data


# In[ ]:


data_all = data.join(data_new1, how='inner')
data_all


# In[ ]:


data_sum = data_all.sum(axis=1)
data_sum


# In[ ]:


data_tpm = data_all.divide(data_sum, axis='index') * 1000000
data_tpm


# In[ ]:


data_tpm.sum(axis=1)


# In[ ]:


'''data_log = data_tpm.apply(lambda x: np.log2(x))
data_log = data_log.where(data_log != -np.inf, 0)
data_log'''


# In[ ]:


data_log.sum(axis=1)


# In[ ]:


data_new1 = data_new1.T
data = data.T
data_tpm = data_tpm.T


# In[ ]:


data1st = data.loc[:, data_tpm.columns]
data1st


# In[ ]:


'''indexes = []
for i in data1.columns:
  if i in data_tpm.columns:
    indexes.append(i)'''


# In[ ]:


'''indexes = np.array(indexes)
indexes = np.unique(indexes)
len(indexes)'''


# In[ ]:


data2nd = data_new1.loc[:, data_tpm.columns]
data2nd


# In[ ]:


data_sum1 = data1st.sum(axis=1)
data_sum2 = data2nd.sum(axis=1)
data_tpm1 = data1st.divide(data_sum1, axis='index') * 1000000
data_tpm2 = data2nd.divide(data_sum2, axis='index') * 1000000
data_tpm1


# In[ ]:


data_tpm2


# In[177]:


data_deconv = data_tpm1.T.join(data_tpm2.T, how='inner')
data_deconv.T.to_csv('data_for_deconvolution.csv')
data_deconv.T


# In[ ]:


data_log1 = data_tpm1.apply(lambda x: np.log2(x))
data_log1 = data_log1.where(data_log1 != -np.inf, 0)
data_log1


# In[ ]:


data_log2 = data_tpm2.apply(lambda x: np.log2(x))
data_log2 = data_log2.where(data_log2 != -np.inf, 0)
data_log2


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


std1 = data_log1.std(axis=0)
plt.hist(std1,
         100,
         density = 1,
         color ='green',
         alpha = 0.7)


# In[ ]:


import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


# In[ ]:


selector = VarianceThreshold(threshold=4)
data1_selected = selector.fit_transform(data_log1)
cols_idx1 = selector.get_support(indices=True)

data2_selected = selector.fit_transform(data_log2)
cols_idx2 = selector.get_support(indices=True)


# In[ ]:


cols1 = data_log1.columns[cols_idx1]
cols2 = data_log2.columns[cols_idx2]
len(cols1), len(cols2)


# In[ ]:


cols = []
for column in cols1:
  if column in cols2:
    cols.append(column)
cols = np.array(cols)
cols


# In[ ]:


len(cols)


# In[ ]:


x1_all = data_log1.loc[:, cols]
x2_all = data_log2.loc[:, cols]
x1 = x1_all.iloc[:, 100:130]
x2 = x2_all.iloc[:, 100:130]


# In[ ]:


sns.histplot(data=x1, legend=False, binwidth=0.1)


# In[ ]:


sns.histplot(data=x2, legend=False, binwidth=0.1)


# In[ ]:


scaler = StandardScaler()
scaled1 = scaler.fit_transform(x1_all)
scaled2 = scaler.fit_transform(x2_all)

sns.histplot(data=scaled1[:, 100:130], legend=False, binwidth=0.1)


# In[ ]:


sns.histplot(data=scaled2[:, 100:130], legend=False, binwidth=0.1)


# In[ ]:


data_log1.to_csv('data_tpm1.csv')


# In[ ]:


data_log2.to_csv('data_tpm2.csv')


# In[ ]:


scaled1.shape[0] + scaled2.shape[0]


# In[ ]:


scaled_all = np.concatenate((scaled1, scaled2), axis=0)
scaled_all.shape


# In[ ]:


index_all = np.concatenate((data_log1.index, data_log2.index), axis=0)
index_all.shape


# In[138]:


X_selected = pd.DataFrame(scaled_all, index=index_all, columns=cols)
X_selected


# In[139]:


X_selected.to_csv('data_scaled.csv')


# In[140]:


sns.histplot(data=X_selected.iloc[:, 200:230], legend=False, binwidth=0.1)


# In[143]:


scaler = StandardScaler()
scaled1_all = scaler.fit_transform(data_log1)
scaled2_all = scaler.fit_transform(data_log2)

X_all = np.concatenate((scaled1_all, scaled2_all), axis=0)
X_all.shape


# In[145]:


data_all


# In[146]:


X_scaled_all = pd.DataFrame(X_all, index=index_all, columns=data_all.index)
X_scaled_all


# In[147]:


genes_selected = ['ASCL1', 'DDC', 'DLL3', 'FOXA2', 'SCN3A', 'NR0B2', 'GRP', 'CNKSR3', 'FBLN7',
        'RGS17', 'PTPRN2', 'SEC11C', 'NKX2-1', 'DNALI1', 'SLC36A4', 'ICA1',
        'SCG2', 'SMPD3', 'NOL4', 'SYT4', 'STK32A', 'SCGN', 'KCNH6', 'GCH1',
        'GNAO1', 'RGS7', 'GFI1B', 'ZMAT4', 'TOX', 'ETS2', 'PCSK2', 'CA8',
        'RGL3', 'RIMKLA', 'TOX3', 'PCSK1', 'CACNA1A', 'CPE', 'PRUNE2', 'KCNK3',
        'PAH', 'TFF3', 'SCN2A', 'FAM3B', 'NCALD', 'VSNL1', 'SOSTDC1', 'KCNMB2',
        'INSM1', 'SCNN1A', 'SERGEF', 'JAM3', 'WNT11', 'MGAT4C', 'HABP2',
        'NPC1L1', 'NEUROD1', 'CERKL', 'NEUROD4', 'NHLH1', 'SSTR2', 'CHRNB4', 'NEUROD2', 'CHRNA3',
        'THSD7B', 'NHLH2', 'CNTN2', 'SLC17A6', 'CLVS1', 'PROKR1', 'FNDC5',
        'GKAP1', 'FRMD3', 'SCN1B', 'SHF', 'GAS2', 'NTNG2', 'DACH1', 'MMD2',
        'TSHR', 'SEMA6A', 'CADPS', 'LMO1', 'PDE1C', 'HPCA', 'PGF', 'CAMKV',
        'PCDH8', 'ATP2B2', 'PRDM8', 'CDC42EP2', 'ADCYAP1', 'RCOR2', 'EBF3',
        'IGFBPL1', 'INSM1', 'TCP10L', 'ACSL6', 'RGS20', 'KCNQ2', 'GNG8',
        'ZDHHC22', 'EYA2', 'KIAA1614', 'BTBD17', 'LRFN5', 'PDZRN4', 'MRAP2',
        'DISP2', 'MMP24', 'LHX1', 'POU2F3', 'GFI1B', 'C11orf53', 'FOXI1', 'LRMP', 'TRPM5', 'VSNL1', 'BMX', 'PTPN18',
        'SH2D6', 'RGS13', 'ANXA1', 'ANO7', 'HTR3E', 'ART3', 'SOSTDC1', 'GNAO1',
        'LANCL3', 'ASCL2', 'ALDH3B2', 'CPE', 'DDC', 'MOCOS', 'PCSK2', 'BARX2',
        'OXGR1', 'MYB', 'ACADSB', 'IMP4', 'RGS7', 'FAM117A', 'TRIM9', 'TFAP2B',
        'KCNH6', 'CCDC115', 'SMPD3', 'OMG', 'AZGP1', 'PCSK1', 'ACSS1', 'SYP',
        'GALNT14', 'APOBEC1', 'KCNK3', 'CPLX2', 'SOX9', 'CHRM1', 'SRRM3',
        'HES2', 'KLHDC7A', 'CA8', 'TFAP2C', 'TOX', 'PHYHIPL', 'ANXA4', 'EFNA4']


# In[ ]:


pou2f3 = ['POU2F3', 'GFI1B', 'C11orf53', 'FOXI1', 'LRMP', 'TRPM5', 'VSNL1', 'BMX', 'PTPN18',
        'SH2D6', 'RGS13', 'ANXA1', 'ANO7', 'HTR3E', 'ART3', 'SOSTDC1', 'GNAO1',
        'LANCL3', 'ASCL2', 'ALDH3B2', 'CPE', 'DDC', 'MOCOS', 'PCSK2', 'BARX2',
        'OXGR1', 'MYB', 'ACADSB', 'IMP4', 'RGS7', 'FAM117A', 'TRIM9', 'TFAP2B',
        'KCNH6', 'CCDC115', 'SMPD3', 'OMG', 'AZGP1', 'PCSK1', 'ACSS1', 'SYP',
        'GALNT14', 'APOBEC1', 'KCNK3', 'CPLX2', 'SOX9', 'CHRM1', 'SRRM3',
        'HES2', 'KLHDC7A', 'CA8', 'TFAP2C', 'TOX', 'PHYHIPL', 'ANXA4', 'EFNA4']


# In[179]:


color = X_scaled_all['GFI1B']
color


# In[148]:


X_final = X_scaled_all[genes_selected]
X_final


# In[151]:


sns.histplot(data=X_final, legend=False, binwidth=0.1)


# In[157]:


from sklearn.cluster import DBSCAN
from collections import Counter


# In[164]:


dbs = DBSCAN(eps=0.5, min_samples=2)
dbs.fit(X_final)
labels = dbs.labels_

Counter(labels)


# In[ ]:


from sklearn.cluster import KMeans


# In[166]:


kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(X_final)
y = kmeans.labels_
y


# In[165]:


kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(X_selected)
y = kmeans.labels_
y


# In[168]:


from sklearn.decomposition import PCA


# In[180]:


pca = PCA(n_components=3)

pca_features = pca.fit_transform(X_final)

print('Shape before PCA: ', X_final.shape)
print('Shape after PCA: ', pca_features.shape)

pca_df = pd.DataFrame(
    data=pca_features,
    columns=['PC1', 'PC2', 'PC3'])

from mpl_toolkits import mplot3d
plt.style.use('default')

# Prepare 3D graph
fig = plt.figure()
ax = plt.axes(projection='3d')

# Plot scaled features
xdata = pca_features[:,0]
ydata = pca_features[:,1]
zdata = pca_features[:,2]

# Plot 3D plot
ax.scatter3D(xdata, ydata, zdata, c=color, cmap='viridis',)


# In[167]:


sns.heatmap(X_final, annot=True)


# In[ ]:


data_sum2


# In[ ]:


data_tpm2


# In[ ]:


x = np.array([data_tpm1.to_numpy(), data_tpm2.to_numpy()])
x


# In[ ]:


genes_list = [data_tpm1.columns.tolist(), data_tpm1.columns.tolist()]


# In[172]:


len(genes_list)


# In[ ]:


genes_list


# In[ ]:


integrated, corrected, genes = scanorama.correct(datasets, genes_list, return_dimred=True)


# In[ ]:


get_ipython().system('pip install --quiet scvi-colab')
from scvi_colab import install

install()

import scanpy as sc
import scvi

adata = scvi.data.heart_cell_atlas_subsampled()


# In[ ]:


adata


# In[ ]:


SCLC = files.upload()


# In[ ]:





# In[ ]:


data_merged =
data_merged = data_merged.append(sum.transpose())
data = np.fromfunction(
    lambda i, j: data_merged[i][j]/(sum(data_merged[i]*(10**6))), dtype=float
)
TPM_merged = pd.DataFrame(data,
                          columns = list(data_merged.columns.values),
                          index = list(data_merged.index.values))



# harmonize datasets

# In[ ]:


get_ipython().system('pip install --quiet scvi-colab')
from scvi_colab import install

install()


# In[ ]:


import scanpy as sc
import scvi

sc.set_figure_params(figsize=(4, 4))


# In[ ]:


ann_merged= scvi.data.merged()


# In[ ]:


sc.pp.filter_genes(ann_merged, min_counts=3) #preprocessing required?


# In[ ]:


ann_merged.layers["counts"] = ann_merged.X.copy()  # preserve counts
sc.pp.normalize_total(ann_merged, target_sum=1e4)
sc.pp.log1p(ann_merged)
ann_merged.raw = ann_merged  # freeze the state in `.raw`


# In[ ]:


sc.pp.highly_variable_genes(
    ann_merged,
    n_top_genes=1200,
    subset=True,
    layer="counts",
    flavor="seurat_v3",
    batch_key="cell_source",
)


# In[ ]:


pip install combat


# In[ ]:


from combat.pycombat import pycombat
import matplotlib.pyplot as plt


# In[ ]:


#we generate the list of batches

datasets = [dataset_1,dataset_2,dataset_3]
for j in range(len(datasets)):
    batch.extend([j for _ in range(len(datasets[j].columns))])

# run pyComBat
df_corrected = pycombat(df_expression,batch)

# visualise results
plt.boxplot(df_corrected)
plt.show()


# In[ ]:


batch = []
data = pycombat(df_expression,batch)

plt.boxplot(df_corrected)
plt.show()

