#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import matplotlib.pyplot as plt
import scanorama
import scanpy as sc
import anndata
import leidenalg


# In[43]:


adata = anndata.read_csv("data_log_scaled_1700.csv")


# In[44]:


adata.obs["batch"]=adata_nocorr.obs.index.to_series().str.startswith("LCNEC").map({True:"LCNEC", False:"SCLC"})


# In[45]:


adata_nocorr = adata.copy()


# In[47]:


sc.pp.scale(adata_nocorr)
sc.tl.pca(adata_nocorr)


# In[48]:


sc.pp.neighbors(
    adata_nocorr,
    n_pcs=30,
    n_neighbors=20,
    knn=True
)


# In[49]:


sc.tl.leiden(adata_nocorr)
sc.tl.umap(adata_nocorr)


# In[50]:


sc.pl.umap(adata_nocorr, color=["leiden", "batch"],
           title=["Cluster", "Batch"], wspace=0.4, frameon=False)


# In[ ]:





# In[ ]:


#correction with scanorama


# In[75]:


adata_scanorama = adata.copy()


# In[76]:


sc.pp.scale(adata_scanorama)
sc.tl.pca(adata_scanorama)


# In[77]:


def split_batches(adata, batch_key):
    """
    This splits one AnnData object into few ones (per batch).
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    batch_key : str
        Name of column in `adata.obs` that corresponds to the name of samples / batches.

    Returns
    ----------
    List of AnnData objects.
    """
    adatas = []
    for batch in set(adata.obs[batch_key]):
        adatas.append(adata[adata.obs[batch_key] == batch].copy())
    return adatas


# In[78]:


adatas_scanorama = split_batches(adata=adata_scanorama, batch_key="batch")
adatas_scanorama = [sc.pp.scale(ad, copy=True) for ad in adatas_scanorama]


# In[79]:


def concatenate_batches(adatas, batch_key="batch"):
    """
    This function concatenates different AnnData objects into one.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    batch_key : str, optional
        Name of column in `adata.obs` that corresponds to the name of samples / batches.

    Returns
    ----------
    AnnData object.
    """
    adata = adatas[0].concatenate(adatas[1:], batch_key=batch_key, index_unique=None)
    return adata


# In[80]:


scanorama.integrate_scanpy(adatas_scanorama, dimred=30)
adata_scanorama.obsm["X_scanorama"] = (
    concatenate_batches(adatas_scanorama)[adata_scanorama.obs.index].obsm["X_scanorama"]
)


# In[81]:


sc.pp.neighbors(
    adata_scanorama,
    use_rep="X_scanorama",
    n_pcs=30
)

sc.tl.leiden(adata_scanorama)
sc.tl.umap(adata_scanorama)


# In[30]:


sc.pl.umap(adata_scanorama, color=["leiden", "batch"],
           title=["Cluster", "Batch corrected"], wspace=0.4, frameon=False)


# In[ ]:





# In[ ]:


#PCA after scanorama batch effect correction


# In[ ]:


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


# In[82]:


sc.pp.scale(adata_scanorama, max_value=10)
sc.tl.pca(adata_scanorama, n_comps=50)
adata_scanorama


# In[84]:


sc.pl.pca(adata_scanorama, color = ['batch'])

