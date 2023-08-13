#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd 
import numpy as np


# In[52]:


data_tpm_LCNEC = pd.read_csv('data_tpm_log_LCNEC.csv', index_col = 0)
data_tpm_SCLC = pd.read_csv('data_tpm_log_SCLC.csv', index_col = 0)


# In[53]:


combined_log_all = pd.concat([data_LCNEC,data_SCLC])
combined_log_all.to_csv('combined_log_all.csv', index = True)


# In[41]:


combined_for_Kassandra = combined_nomalised_all.T


# In[43]:


combined_for_Kassandra.to_csv('combined_for_Kassandra.csv', index = True)


# In[57]:


data_for_deconvolution = pd.read_csv('data_for_deconvolution.csv', index_col=0)


# In[62]:


data_for_deconvolution = data_for_deconvolution.T
# all genes in tpm, no duplets, no log, no normalization
data_for_deconvolution


# In[64]:


data_for_deconvolution.to_csv('data_for_deconvolution_1.csv', index = True)

