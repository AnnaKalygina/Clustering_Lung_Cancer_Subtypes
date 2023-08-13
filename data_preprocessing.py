#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
import scanorama
import seaborn as sns
import openpyxl


# In[18]:


data1 = pd.read_excel("SCLC.xlsx", index_col=0, header=2)
data_drop = data1.drop(columns='refseq')
data_new = data_drop.groupby(by=['gene']).mean()
data_new1 = data_new.iloc[26:, :]
data_new1


# In[32]:


data = pd.read_excel("LCNEC.xlsx", index_col=0, header=2)
data


# In[31]:


data_new1 = data_new1.T
data = data.T
data_tpm = data_tpm.T
data1st = data.loc[:, data_tpm.columns]
data1st


# In[30]:


data_all = data.join(data1, how='inner')
data_all


# In[20]:


data2nd = data_new1.loc[:, data_tpm.columns]
data2nd


# In[ ]:


data_sum1 = data1st.sum(axis=1)
data_sum2 = data2nd.sum(axis=1)
data_tpm1 = data1st.divide(data_sum1, axis='index') * 1000000
data_tpm2 = data2nd.divide(data_sum2, axis='index') * 1000000
data_tpm1


# In[ ]:


data_log1 = data_tpm1.apply(lambda x: np.log2(x))
data_log1 = data_log1.where(data_log1 != -np.inf, 0)
data_log1


# In[ ]:


data_log2 = data_tpm2.apply(lambda x: np.log2(x))
data_log2 = data_log2.where(data_log2 != -np.inf, 0)
data_log2


# In[ ]:


#highly variable


# In[ ]:




