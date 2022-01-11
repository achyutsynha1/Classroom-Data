#!/usr/bin/env python
# coding: utf-8

# In[118]:


import numpy as np 
import os 
import pandas as pd 
import seaborn as sns
#using the style 
plt.style.use('ggplot') 


# In[119]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 


# In[120]:


OCD= pd.read_csv("C:\\Users\\Particle\\OneDrive\\Desktop\\devp\\17-online_classroom_data.csv")


# # Basic Operations - Describe, Head , Info

# In[121]:


OCD.describe


# In[131]:


OCD.info


# In[123]:


OCD.head()


# # Graphical Representations 

# In[135]:


plt.figure(figsize=(12,12))
sns.countplot(x='helpful_post', data=OCD)


# In[126]:


def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number])
    df = df.dropna('columns') #removing columns with Null/None/NA values from DataFrame  
    df = df[[col for col in df if df[col].nunique() > 1]]
    columnNames = list(df) #Making the column names with the names given in the CSV list
    if len(columnNames) > 10: # for reducing the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.8, figsize=[plotSize, plotSize], diagonal='kde') # KDE is a
    #non-parametric way to estimate the probability density function of any variable we wish to view
    corrs = df.corr().values #Pairwise correlation of columns
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', 
                          ha='center', va='center', size=textSize) 
        plt.suptitle('Scatter & Density Plot')
    plt.show()


# In[127]:


plotScatterMatrix(OCD, 20, 10)


# In[105]:


data=OCD.select_dtypes(include=['float64','int64'])


# In[130]:


plt.figure(figsize=(10,10))
pd.plotting.parallel_coordinates(data,class_column='timeonline',color=('black','red'))


# In[128]:


ss=StandardScaler()


# In[129]:


plt.figure(figsize=(6,6)) 
_=sns.violinplot(x="Approved", y="timeonline", data=OCD)


# In[ ]:




