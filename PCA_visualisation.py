#!/usr/bin/env python
# coding: utf-8

# In[71]:


import pandas as pd
import scanpy as sc
import sklearn.datasets
from sklearn.decomposition import PCA
import plotly.express as px
import anndata
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# In[100]:


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


# In[101]:


pou2f3 = ['POU2F3', 'GFI1B', 'C11orf53', 'FOXI1', 'LRMP', 'TRPM5', 'VSNL1', 'BMX', 'PTPN18',
        'SH2D6', 'RGS13', 'ANXA1', 'ANO7', 'HTR3E', 'ART3', 'SOSTDC1', 'GNAO1',
        'LANCL3', 'ASCL2', 'ALDH3B2', 'CPE', 'DDC', 'MOCOS', 'PCSK2', 'BARX2',
        'OXGR1', 'MYB', 'ACADSB', 'IMP4', 'RGS7', 'FAM117A', 'TRIM9', 'TFAP2B',
        'KCNH6', 'CCDC115', 'SMPD3', 'OMG', 'AZGP1', 'PCSK1', 'ACSS1', 'SYP',
        'GALNT14', 'APOBEC1', 'KCNK3', 'CPLX2', 'SOX9', 'CHRM1', 'SRRM3',
        'HES2', 'KLHDC7A', 'CA8', 'TFAP2C', 'TOX', 'PHYHIPL', 'ANXA4', 'EFNA4']


# In[ ]:


# pca without batch correction


# In[96]:


adata_nocorr = anndata.read_csv("combined_log_all.csv")
adata_nocorr.obs["batch"]=adata_nocorr.obs.index.to_series().str.startswith("LCNEC").map({True:"LCNEC", False:"SCLC"})


# In[97]:


sc.pp.scale(adata_nocorr, max_value=10)
sc.tl.pca(adata_nocorr, n_comps=50)
adata_nocorr


# In[98]:


sc.pl.pca(adata_nocorr, color = 'batch')


# In[ ]:


# pca after batch correction


# In[121]:


adata = anndata.read_csv("combined_log_scaled_all.csv")
adata.obs["batch"]=adata.obs.index.to_series().str.startswith("LCNEC").map({True:"LCNEC", False:"SCLC"})


# In[122]:


sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, n_comps=50)
adata


# In[104]:


#scanpy figures
sc.pl.pca(adata, color = 'batch')


# In[132]:


sc.pl.pca(adata, color=genes_selected, frameon=False)


# In[93]:


sc.pl.pca_loadings(adata)


# In[10]:


pca_loadings = pd.DataFrame(adata.varm['PCs'], columns=[f'PC{i+1}' for i in range((adata.obsm['X_pca'].shape[1]))])
pca_loadings


# In[130]:


PC1_high_ranking_genes = ["PFDN5", "FAU", "RPL27A", "RPL35", "RPS18", "RPL27", "RPL19", "RPL39", "RPS8", "RPS10", "RPL12", "USMG5", "ATP5J2", "EEF1B2", "RPL32"]
PC2_high_ranking_genes = ["IL1R1", "CD33", "NCF4", "GLIPR1", "NOD2", "HLA-DMA", "TBXAS1", "CSF2RB", "SQRDL", "SP100", "CYTH4", "C1S", "CTSH", "C10orf54", "CASP1"]
PC3_high_ranking_genes = ["FEXL15", "FBXW5", "SPATA2L", "ABHD14A", "BAIAP3", "BCAT2", "RNASEK", "TMUB1", "PHPT1", "TMEM8A", "RABAC1", "MRPL41", "NTHL1", "FASTK", "TMEM53"]

pca1_genes_intersection = list(filter(lambda value: value in PC1_high_ranking_genes, genes_selected))
pca2_genes_intersection = list(filter(lambda value: value in PC2_high_ranking_genes, genes_selected))
pca3_genes_intersection = list(filter(lambda value: value in PC3_high_ranking_genes, genes_selected))

pca_genes_intersection = pca1_genes_intersection + pca2_genes_intersection + pca3_genes_intersection

pou2f3_pca_genes = list(filter(lambda value: value in pca_genes_intersection, pou2f3))    

print(pca_genes_intersection)
print(pou2f3_pca_genes)


# In[114]:


pca_loadings = pd.DataFrame(adata.varm['PCs'], columns=[f'PC{i+1}' for i in range((adata.obsm['X_pca'].shape[1]))])
# Get genes with highest absolute loading values for each PC
num_top_genes = 10  # Number of top genes to consider

top_genes_per_pc = {}
for pc in pca_loadings.columns:
    top_genes = pca_loadings[pc].abs().nlargest(num_top_genes).index
    top_genes_per_pc[pc] = top_genes

# Print the top genes for each PC
for pc, top_genes in top_genes_per_pc.items():
    print(f"Top genes contributing to {pc}:")
    print(top_genes)
    print()


# In[115]:


names = adata.var_names.to_numpy()
names[[3244, 3953, 369, 4308, 4224, 829, 4546, 2728, 4182, 4202]]


# In[125]:


names = adata.var_names.to_numpy()
names[[1811, 600, 4407, 817, 3388, 433, 4226, 2141, 176, 3998]]


# In[124]:


#genes with indexes from pca loading that contribute to pca1
sc.pl.pca(adata, color=['PDPK1', 'SETD5', 'ATP8B2', 'SYNRG', 'SSH1', 'CCDC93', 'TNRC6A', 'MLL5', 'SPEN', 'SPTAN1'], frameon=False)


# In[126]:


#genes with indexes from pca loading that contribute to pca1
sc.pl.pca(adata, color=['C17orf51', 'ANGEL1', 'DLD', 'AQP6', 'CLIC4', 'AIMP1', 'DDX3Y','C4orf26', 'ACSL1', 'CYBA'], frameon=False)


# In[117]:


sc.pl.pca(adata, color=["PFDN5"], frameon=False, components = ['1,2', '2,3', '1,3'])


# In[94]:


sc.pl.pca_variance_ratio(adata, log=True)


# In[111]:


# compute variable genes
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
print("Highly variable genes: %d"%sum(adata.var.highly_variable))

#plot variable genes
sc.pl.highly_variable_genes(adata)

# subset for variable genes in the dataset
adata = adata[:, adata.var['highly_variable']]

