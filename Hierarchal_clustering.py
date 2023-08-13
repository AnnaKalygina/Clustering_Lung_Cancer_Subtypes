#!/usr/bin/env python
# coding: utf-8

# In[214]:


import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
import seaborn
import numpy as np


# In[161]:


genes_tf = {'ASCL1': ['DDC', 'DLL3', 'FOXA2', 'SCN3A', 'NR0B2', 'GRP', 'CNKSR3', 'FBLN7',
        'RGS17', 'PTPRN2', 'SEC11C', 'NKX2-1', 'DNALI1', 'SLC36A4', 'ICA1',
        'SCG2', 'SMPD3', 'NOL4', 'SYT4', 'STK32A', 'SCGN', 'KCNH6', 'GCH1',
        'GNAO1', 'RGS7', 'GFI1B', 'ZMAT4', 'TOX', 'ETS2', 'PCSK2', 'CA8',
        'RGL3', 'RIMKLA', 'TOX3', 'PCSK1', 'CACNA1A', 'CPE', 'PRUNE2', 'KCNK3',
        'PAH', 'TFF3', 'SCN2A', 'FAM3B', 'NCALD', 'VSNL1', 'SOSTDC1', 'KCNMB2',
        'INSM1', 'SCNN1A', 'SERGEF', 'JAM3', 'WNT11', 'MGAT4C', 'HABP2',
        'NPC1L1'],
             'NEUROD1': ['CERKL', 'NEUROD4', 'NHLH1', 'SSTR2', 'CHRNB4', 'NEUROD2', 'CHRNA3',
        'THSD7B', 'NHLH2', 'CNTN2', 'SLC17A6', 'CLVS1', 'PROKR1', 'FNDC5',
        'GKAP1', 'FRMD3', 'SCN1B', 'SHF', 'GAS2', 'NTNG2', 'DACH1', 'MMD2',
        'TSHR', 'SEMA6A', 'CADPS', 'LMO1', 'PDE1C', 'HPCA', 'PGF', 'CAMKV',
        'PCDH8', 'ATP2B2', 'PRDM8', 'CDC42EP2', 'ADCYAP1', 'RCOR2', 'EBF3',
        'IGFBPL1', 'INSM1', 'TCP10L', 'ACSL6', 'RGS20', 'KCNQ2', 'GNG8',
        'ZDHHC22', 'EYA2', 'KIAA1614', 'BTBD17', 'LRFN5', 'PDZRN4', 'MRAP2',
        'DISP2', 'MMP24', 'LHX1'],
             'POU2F3': ['GFI1B', 'C11orf53', 'FOXI1', 'LRMP', 'TRPM5', 'VSNL1', 'BMX', 'PTPN18',
        'SH2D6', 'RGS13', 'ANXA1', 'ANO7', 'HTR3E', 'ART3', 'SOSTDC1', 'GNAO1',
        'LANCL3', 'ASCL2', 'ALDH3B2', 'CPE', 'DDC', 'MOCOS', 'PCSK2', 'BARX2',
        'OXGR1', 'MYB', 'ACADSB', 'IMP4', 'RGS7', 'FAM117A', 'TRIM9', 'TFAP2B',
        'KCNH6', 'CCDC115', 'SMPD3', 'OMG', 'AZGP1', 'PCSK1', 'ACSS1', 'SYP',
        'GALNT14', 'APOBEC1', 'KCNK3', 'CPLX2', 'SOX9', 'CHRM1', 'SRRM3',
        'HES2', 'KLHDC7A', 'CA8', 'TFAP2C', 'TOX', 'PHYHIPL', 'ANXA4', 'EFNA4']}


# In[162]:


genes_subset = ['ASCL1','NEUROD1','POU2F3','DDC', 'DLL3', 'FOXA2', 'SCN3A', 
                'NR0B2', 'GRP', 'CNKSR3', 'FBLN7',
        'RGS17', 'PTPRN2', 'SEC11C', 'NKX2-1', 'DNALI1', 'SLC36A4', 'ICA1',
        'SCG2', 'SMPD3', 'NOL4', 'SYT4', 'STK32A', 'SCGN', 'KCNH6', 'GCH1',
        'GNAO1', 'RGS7', 'GFI1B', 'ZMAT4', 'TOX', 'ETS2', 'PCSK2', 'CA8',
        'RGL3', 'RIMKLA', 'TOX3', 'PCSK1', 'CACNA1A', 'CPE', 'PRUNE2', 'KCNK3',
        'PAH', 'TFF3', 'SCN2A', 'FAM3B', 'NCALD', 'VSNL1', 'SOSTDC1', 'KCNMB2',
        'INSM1', 'SCNN1A', 'SERGEF', 'JAM3', 'WNT11', 'MGAT4C', 'HABP2',
        'NPC1L1', 'CERKL', 'NEUROD4', 'NHLH1', 'SSTR2', 'CHRNB4', 'NEUROD2', 
                'CHRNA3','THSD7B', 'NHLH2', 'CNTN2', 'SLC17A6', 'CLVS1', 'PROKR1', 
                'FNDC5',
        'GKAP1', 'FRMD3', 'SCN1B', 'SHF', 'GAS2', 'NTNG2', 'DACH1', 'MMD2',
        'TSHR', 'SEMA6A', 'CADPS', 'LMO1', 'PDE1C', 'HPCA', 'PGF', 'CAMKV',
        'PCDH8', 'ATP2B2', 'PRDM8', 'CDC42EP2', 'ADCYAP1', 'RCOR2', 'EBF3',
        'IGFBPL1', 'INSM1', 'TCP10L', 'ACSL6', 'RGS20', 'KCNQ2', 'GNG8',
        'ZDHHC22', 'EYA2', 'KIAA1614', 'BTBD17', 'LRFN5', 'PDZRN4', 'MRAP2',
        'DISP2', 'MMP24', 'LHX1', 'GFI1B', 'C11orf53', 'FOXI1', 'LRMP', 'TRPM5', 
                'VSNL1', 'BMX', 'PTPN18',
        'SH2D6', 'RGS13', 'ANXA1', 'ANO7', 'HTR3E', 'ART3', 'SOSTDC1', 'GNAO1',
        'LANCL3', 'ASCL2', 'ALDH3B2', 'CPE', 'DDC', 'MOCOS', 'PCSK2', 'BARX2',
        'OXGR1', 'MYB', 'ACADSB', 'IMP4', 'RGS7', 'FAM117A', 'TRIM9', 'TFAP2B',
        'KCNH6', 'CCDC115', 'SMPD3', 'OMG', 'AZGP1', 'PCSK1', 'ACSS1', 'SYP',
        'GALNT14', 'APOBEC1', 'KCNK3', 'CPLX2', 'SOX9', 'CHRM1', 'SRRM3',
        'HES2', 'KLHDC7A', 'CA8', 'TFAP2C', 'TOX', 'PHYHIPL', 'ANXA4', 'EFNA4']


# In[166]:


# Accepts the log-transformed data
# Dataframe must have gene names as indecies, if not, please uncomment and set indecies

expression_data = pd.read_csv('data_log_scaled_1700.csv', index_col=[0],)
#expression_data = np.log2(expression_data)

# Genes must be rows and samples must be columns
gene_names = list(expression_data.columns.values)
numerical_data = expression_data.values

#scipy’s clustering algorithm clusters the rows, not the columns. 
#If we want to cluster the cell lines, we’ll need to transpose the data

#if len(sample_names) != len(numerical_data):
    #numerical_data = np.transpose(numerical_data)
    
expression_data


# In[188]:


expression_subset = expression_data.reindex(columns=genes_subset).dropna(axis=1)
expression_subset_genes = list(expression_subset.columns.values)
len(expression_subset_genes)
expression_subset


# In[225]:


# Reset sample names into two batches
sample_names = list(expression_subset.index.values)
for i in range(len(sample_names)):
    if "LCNEC" in sample_names[i]:
        sample_names[i] = "LCNEC"
    else:
        sample_names[i] = "SCLC"
print(sample_names)
expression_subset = expression_subset.set_axis(sample_names, axis='index')


# In[226]:


expression_subset


# In[180]:


# Agglomerative clustering 
hierarchical = AgglomerativeClustering()
cluster_assignments = hierarchical.fit_predict(expression_subset)

# Perform hierarchical clustering using SciPy's linkage function
linkage_matrix = linkage(expression_subset, method='ward')

# Plot the dendrogram
plt.figure(figsize=(10, 10))
dendrogram(linkage_matrix)
plt.xlabel("Genes")
plt.ylabel("Distance")
plt.title("Hierarchical Clustering Dendrogram")
plt.show()




# In[254]:



# Create a dictionary to map sample names to color codes
sample_color_mapping = {
    'LCNEC': 'purple',    # Define your own color codes
    'SCLC': 'yellow',
    # ... and so on for other samples
}
data = expression_subset.T

# Convert sample names to colors for row and column labels
row_colors = expression_subset.index.map(sample_color_mapping)

# Create the clustermap with color-coded samples
clustermap = seaborn.clustermap(
    data,
    method='ward',
    metric='euclidean',
    z_score=0,
    figsize=(10, 10),
    cbar_kws=None,
    row_cluster=True,
    col_cluster=True,
    col_colors=row_colors,    
    row_linkage=None,
    col_linkage=None,
    mask=None,
    dendrogram_ratio=0.2,
    colors_ratio=0.02,
    cbar_pos=(0.02, 0.8, 0.05, 0.18),
    tree_kws=None,
    cmap="coolwarm",
    vmin = -2,
    vmax = 2
)

# Hide the row dendrogram
clustermap.ax_col_dendrogram.set_visible(False)

# Show the plot
plt.show()


# In[250]:


seaborn.clustermap(expression_subset.T,method='ward',
    metric='euclidean')


# In[ ]:


#.clip(-2,2)
#change pallete 


# In[242]:


def hierarchical_clustering(expression_subset, expression_subset_genes):
    numerical_data = expression_subset.values
    Z = linkage(np.transpose(numerical_data), method='complete', metric='euclidean')

    # Plot the dendrogram
    plt.figure(figsize=(10, 5))
    dendrogram(Z, labels=expression_subset_genes, orientation='top', leaf_rotation=90)
    plt.xlabel("Genes")
    plt.ylabel("Distance")
    
    plt.title("Hierarchical Clustering Dendrogram")
    plt.tight_layout()
    plt.show()

# Call the function with your expression data and sample names
# expression_data is your DataFrame of gene expression data
# sample_names is a list of sample names corresponding to rows in expression_data
hierarchical_clustering(expression_subset, expression_subset_genes)

