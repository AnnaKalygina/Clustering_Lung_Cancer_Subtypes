#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import scanpy as sc
import numpy as np


# In[4]:


data_1700 = pd.read_csv("data_log_scaled_1700.csv", index_col=0)
data_1700


# In[6]:


data_labeled = pd.read_csv("data_final_labelled.csv", index_col=0)
data_labeled


# In[11]:


labels = data_labeled['label']


# In[13]:


X = data_labeled.drop(columns=['label'])


# In[14]:


X


# In[15]:


from scipy.stats import mannwhitneyu


# In[50]:


results = {}

for gene in data_1700.columns:
    
    result = mannwhitneyu(data_1700.loc[labels[labels== 1].index, gene],
                          data_1700.loc[labels[labels!= 1].index, gene],
                         )
    
    results[gene] = (result[0], result[1])


# In[51]:


results


# In[54]:


#bonferroni
# Create a list of the adjusted p-values
df_results = pd.DataFrame.from_dict(results, orient = 'index', columns=['statistic ', 'p_value'])
df_results


# In[82]:


(df_results['p_value']*1700 < 0.05).sum()


# In[37]:


significant = np.sum(df_results < 0.05)
significant


# In[55]:


df_results['p_value_cor']=df_results['p_value']*1698


# In[66]:


df_results['significant'] = (df_results['p_value_cor'] < 0.05)
df_results


# In[57]:


significant = np.sum(df_results['p_value_cor'] < 0.05)
significant


# In[85]:


df_results.sort_values(by='p_value_cor').head(20)


# In[86]:


df_results.sort_values(by='p_value_cor').head(20).index


# In[62]:


import seaborn as sns


# In[70]:


sns.histplot(df_results['statistic '])

