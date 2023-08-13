#!/usr/bin/env python
# coding: utf-8

# # Preprocessing

# In[191]:


Preprocessingimport pandas as pd
import numpy as np


# In[192]:


#data1 = pd.read_csv('SCLC.csv', header=2)
data1 = pd.read_excel('SCLC.xlsx', index_col=0, header=2)


# In[193]:


data_drop = data1.drop(columns='refseq')
data_drop


# In[194]:


data_new = data_drop.groupby(by=['gene']).mean()
data_new1 = data_new.iloc[26:, :]
data_new1


# In[195]:


data = pd.read_excel('LCNEC.xlsx', index_col=0, header=2)
data


# In[196]:


data_all = data.join(data_new1, how='inner')
data_all


# In[197]:


data_sum = data_all.sum(axis=1)
data_sum


# In[198]:


data_tpm = (data_all).divide(data_sum, axis='index') * 1000000
data_tpm


# In[199]:


data_tpm.sum(axis=1)


# In[200]:


'''data_log = data_tpm.apply(lambda x: np.log2(x))
data_log = data_log.where(data_log != -np.inf, 0)
data_log'''


# In[201]:


data_new1 = data_new1.T
data = data.T
data_tpm = data_tpm.T


# In[202]:


data1st = data.loc[:, data_tpm.columns]
data1st


# In[203]:


'''indexes = []
for i in data1.columns:
  if i in data_tpm.columns:
    indexes.append(i)'''


# In[204]:


'''indexes = np.array(indexes)
indexes = np.unique(indexes)
len(indexes)'''


# In[205]:


data2nd = data_new1.loc[:, data_tpm.columns]
data2nd


# In[206]:


data_sum1 = data1st.sum(axis=1)
data_sum2 = data2nd.sum(axis=1)
data_tpm1 = data1st.divide(data_sum1, axis='index') * 1000000
data_tpm2 = data2nd.divide(data_sum2, axis='index') * 1000000
data_tpm1


# In[207]:


data_tpm2


# In[208]:


data_deconv = data_tpm1.T.join(data_tpm2.T, how='inner')
data_deconv.T.to_csv('data_for_deconvolution.csv')
data_deconv.T


# In[209]:


data_log1 = data_tpm1.apply(lambda x: np.log2(x + 1))
#data_log1 = data_log1.where(data_log1 != -np.inf, 0)
data_log1


# In[215]:


import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


# In[210]:


data_log2 = data_tpm2.apply(lambda x: np.log2(x + 1))
#data_log2 = data_log2.where(data_log2 != -np.inf, 0)
data_log2


# In[284]:


index_all = np.concatenate((data_log1.index, data_log2.index), axis=0)
scaler = StandardScaler()
scaled1_all = scaler.fit_transform(data_log1)
scaled2_all = scaler.fit_transform(data_log2)

scaled_all16k = np.concatenate((scaled1_all, scaled2_all), axis=0)
X_all_pd = pd.DataFrame(scaled_all16k, index=index_all, columns=data_log1.columns)
X_all_pd


# In[285]:


X_all_pd.to_csv('combined_log_scaled_all.csv')


# In[282]:


scaled2_all_df


# In[ ]:


data_all_df = scaled1_all_df.join(scaled2_all_df, how='inner')
data_all_df


# In[211]:


import matplotlib.pyplot as plt


# In[365]:


std1 = data_log1.std(axis=0)
plt.hist(std1,
         100,
         density = 1,
         color ='lightblue',
         alpha = 0.7)
plt.xlabel('std value')
plt.ylabel('density')
plt.title('Filtering highly variable genes')
plt.axvline(x = 1.2, color = 'b', label = 'axvline - full height')


# In[216]:


selector = VarianceThreshold(threshold=1.4)
data1_selected = selector.fit_transform(data_log1)
cols_idx1 = selector.get_support(indices=True)

data2_selected = selector.fit_transform(data_log2)
cols_idx2 = selector.get_support(indices=True)


# In[217]:


cols1 = data_log1.columns[cols_idx1]
cols2 = data_log2.columns[cols_idx2]
len(cols1), len(cols2)


# In[218]:


cols = []
for column in cols1:
    if column in cols2:
        cols.append(column)
cols = np.array(cols)
cols


# In[219]:


len(cols)


# In[ ]:


x1_all = data_log1.loc[:, cols]
x2_all = data_log2.loc[:, cols]
x1 = x1_all.iloc[, 100:130]
x2 = x2_all.iloc[, 100:130]


# In[246]:


sns.histplot(data=x1, binwidth=0.1)
plt.xlabel('expression, tpm')
plt.ylabel('Count')
plt.legend(labels=x1.columns.to_list()[:7], title='Genes:')


# In[254]:


sns.histplot(data=x2, legend=False, binwidth=0.1)
plt.xlabel('expression, tpm')
plt.ylabel('Count')
plt.xticks(ticks=[i for i in range(15)])
plt.legend(labels=x2.columns.to_list()[:8], alignment='right', title='Genes:')
plt.tight_layout()


# In[248]:


scaler = StandardScaler()
scaled1 = scaler.fit_transform(x1_all)
scaled2 = scaler.fit_transform(x2_all)

sns.histplot(data=scaled1[:, 100:130], legend=False, binwidth=0.1)
plt.xlabel('expression, tpm')
plt.ylabel('Count')
plt.legend(labels=x2.columns.to_list()[:7], alignment='right', title='Genes:')


# In[249]:


sns.histplot(data=scaled2[:, 100:130], legend=False, binwidth=0.1)
plt.xlabel('expression, tpm')
plt.ylabel('Count')
plt.legend(labels=x2.columns.to_list()[:8], alignment='right', title='Genes:')


# In[ ]:


data_log1.to_csv('data_tpm1.csv')


# In[ ]:


data_log2.to_csv('data_tpm2.csv')


# In[75]:


scaled_all = np.concatenate((scaled1, scaled2), axis=0)
scaled_all.shape


# In[76]:


index_all = np.concatenate((data_log1.index, data_log2.index), axis=0)
index_all.shape


# In[77]:


X_selected = pd.DataFrame(scaled_all, index=index_all, columns=cols)
X_selected


# In[139]:


X_selected.to_csv('data_scaled.csv')


# In[255]:


sns.histplot(data=X_selected.iloc[:, 100:130], legend=False, binwidth=0.1)
plt.xlabel('expression, tpm')
plt.ylabel('Count')
plt.legend(labels=x2.columns.to_list()[:8], alignment='right', title='Genes:')


# In[79]:


scaler = StandardScaler()
scaled1_all = scaler.fit_transform(data_log1)
scaled2_all = scaler.fit_transform(data_log2)

X_all = np.concatenate((scaled1_all, scaled2_all), axis=0)
X_all.shape


# In[80]:


data_all


# In[81]:


X_scaled_all = pd.DataFrame(X_all, index=index_all, columns=data_all.index)
X_scaled_all


# In[177]:


genes_selected = np.array(['ASCL1', 'DDC', 'DLL3', 'FOXA2', 'SCN3A', 'NR0B2', 'GRP', 'CNKSR3', 'FBLN7',
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
        'HES2', 'KLHDC7A', 'CA8', 'TFAP2C', 'TOX', 'PHYHIPL', 'ANXA4', 'EFNA4'])


# In[189]:


POU2F3 = ['POU2F3', 'GFI1B', 'C11orf53', 'FOXI1', 'LRMP', 'TRPM5', 'VSNL1', 'BMX', 'PTPN18',
        'SH2D6', 'RGS13', 'ANXA1', 'ANO7', 'HTR3E', 'ART3', 'SOSTDC1', 'GNAO1',
        'LANCL3', 'ASCL2', 'ALDH3B2', 'CPE', 'DDC', 'MOCOS', 'PCSK2', 'BARX2',
        'OXGR1', 'MYB', 'ACADSB', 'IMP4', 'RGS7', 'FAM117A', 'TRIM9', 'TFAP2B',
        'KCNH6', 'CCDC115', 'SMPD3', 'OMG', 'AZGP1', 'PCSK1', 'ACSS1', 'SYP',
        'GALNT14', 'APOBEC1', 'KCNK3', 'CPLX2', 'SOX9', 'CHRM1', 'SRRM3',
        'HES2', 'KLHDC7A', 'CA8', 'TFAP2C', 'TOX', 'PHYHIPL', 'ANXA4', 'EFNA4']


# In[180]:


genes_selected = np.unique(genes_selected))


# In[84]:


color = X_scaled_all['GFI1B']
color


# In[292]:


X_final = X_scaled_all[genes_selected]
X_final


# In[174]:


u, i = np.unique(X_final.columns, return_counts=True)


# In[175]:


i


# In[163]:


len(genes_selected)


# In[256]:


sns.histplot(data=X_final, legend=False, binwidth=0.1)
plt.xlabel('expression, tpm')
plt.ylabel('Count')
plt.legend(labels=x2.columns.to_list()[:8], alignment='right', title='Genes:')


# # 

# In[87]:


from sklearn.cluster import DBSCAN
from collections import Counter


# In[89]:


from sklearn.cluster import KMeans


# In[90]:


kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(X_final)
y = kmeans.labels_
y


# In[91]:


kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(X_selected)
y = kmeans.labels_
y


# In[92]:


from sklearn.decomposition import PCA


# In[ ]:


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
ax.scatter3D(xdata, ydata, zdata, c=color, cmap='viridis')
#ax.legend(title="Legend")


# In[95]:


from sklearn.metrics import silhouette_score


# In[97]:


KMean= KMeans(n_clusters=2)
KMean.fit(X_final)
label=KMean.predict(X_final)
print(f'Silhouette Score(n=2): {silhouette_score(X_final, label)}')


# In[100]:


KMean= KMeans(n_clusters=3)
KMean.fit(X_final)
label=KMean.predict(X_final)
print(f'Silhouette Score(n=2): {silhouette_score(X_final, label)}')


# In[105]:


KMean= KMeans(n_clusters=2)
KMean.fit(X_scaled_all)
label=KMean.predict(X_scaled_all)
print(f'Silhouette Score(n=2): {silhouette_score(X_scaled_all, label)}')


# In[101]:


from sklearn.cluster import AgglomerativeClustering


# In[127]:


clustering = AgglomerativeClustering(n_clusters=2)
label_agg = clustering.fit_predict(X_final)
print(f'Silhouette Score(n=2): {silhouette_score(X_final, label_agg)}')


# In[128]:


label_agg


# In[108]:


clustering = AgglomerativeClustering(n_clusters=3)
label = clustering.fit_predict(X_final)
print(f'Silhouette Score(n=2): {silhouette_score(X_final, label)}')


# In[109]:


clustering = AgglomerativeClustering(n_clusters=2)
label = clustering.fit_predict(X_scaled_all)
print(f'Silhouette Score(n=2): {silhouette_score(X_scaled_all, label)}')


# In[146]:


dbs = DBSCAN(eps=13, min_samples=2)
label = dbs.fit_predict(X_final)

print(f'Silhouette Score(n=2): {silhouette_score(X_final, label)}')


# In[147]:


label


# In[149]:


from sklearn.cluster import SpectralClustering


# In[153]:


label = SpectralClustering(n_clusters=2).fit_predict(X_final)

print(f'Silhouette Score(n=2): {silhouette_score(X_final, label)}')


# In[291]:


data_final = X_final.copy()
data_final['label'] = label_agg
X_final


# In[293]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_final, label_agg, test_size=0.33, random_state=42)
X_train.shape, X_val.shape, y_train.shape, y_val.shape


# In[294]:


from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score


# In[295]:


clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
clf.fit(X_train, y_train)
SGDClassifier(max_iter=5)
y_pred = clf.predict(X_val)
f1_score(y_val, y_pred)


# In[296]:


from sklearn.neighbors import KNeighborsClassifier


# In[297]:


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
y_pred1 = neigh.predict(X_val.to_numpy())
f1_score(y_val, y_pred1)


# In[298]:


X_val


# In[183]:


data_0 = data_final[data_final['label'] == 0]
data_1 = data_final[data_final['label'] == 1]


# In[185]:


data_0.iloc[:, :-1]


# In[262]:


sns.heatmap(data_0.iloc[:, :-1])


# In[187]:


sns.heatmap(data_1.iloc[:, :-1])


# In[264]:


data_test = pd.read_table('GSE151904_expressions_tpm.tsv', index_col=0)
data_test


# In[265]:


data_test_log = data_test.apply(lambda x: np.log2(x + 1))
data_test_log


# In[271]:


sns.histplot(data=data_test_log.iloc[:, 100:130], binwidth=0.1)
plt.xlabel('expression, tpm')
plt.ylabel('Count')
plt.legend(labels=data_test_log.columns.to_list()[:7], alignment='right', title='Genes:')


# In[269]:


scaler = StandardScaler()
test_scaled = scaler.fit_transform(data_test_log)

sns.histplot(data=test_scaled[:, 100:130], binwidth=0.1)
plt.xlabel('expression, tpm')
plt.ylabel('Count')
plt.legend(labels=data_test_log.columns.to_list()[:7], alignment='right', title='Genes:')


# In[277]:


test_scaled_pd = pd.DataFrame(data=test_scaled, index=data_test_log.index, columns=data_test_log.columns)
test_scaled_pd


# In[273]:


genes_selected


# In[287]:


X_test_final = test_scaled_pd[genes_selected]
X_test_final


# In[370]:


test_final


# In[378]:


y_pred_final = neigh.predict(X_test_final.to_numpy())
test_final = X_test_final.copy()
test_final['labels'] = y_pred_final

test_final = test_final.sort_values(by='labels', ascending=False)

sample_colors = {int(0): 'yellow', int(1): 'green'}
col_colors1 = test_final['labels'].map(sample_colors)
clustermap = sns.clustermap(test_final.clip(-2, 2).T, col_colors=col_colors1, cmap = "coolwarm", dendrogram_ratio=0.1)

clustermap.ax_col_dendrogram.set_visible(False)
clustermap.ax_row_dendrogram.set_visible(False)
clustermap.fig.suptitle("Gene expression depending on type predicted")


# In[356]:


test_final


# In[306]:



fig, axes = plt.subplots(1, 2, figsize=(18, 10))
sns.heatmap(ax=axes[0], data=test_final[test_final['labels'] == 0].clip(-2, 2).iloc[:, :-1])
sns.heatmap(ax=axes[1], data=test_final[test_final['labels'] == 1].clip(-2, 2).iloc[:, :-1])


# In[310]:


test_final


# In[ ]:




