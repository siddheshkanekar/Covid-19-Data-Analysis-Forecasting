#!/usr/bin/env python
# coding: utf-8

# In[44]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[45]:


FCG1=pd.read_excel(r"C:\Users\dbda.STUDENTSDC\Desktop\Project\FCG1.xlsx")
correlation = FCG1.corr()
axis_corr = sns.heatmap(
correlation,
vmin=-1, vmax=1, center=0,annot=True,
cmap=sns.diverging_palette(50, 500, n=500),
square=True
)

plt.show()


# In[46]:


correlation


# ###  
# ![image.png](attachment:image.png)

# In[47]:


FCG1


# In[48]:


df['Date']=pd.to_datetime(df['Date'])


# In[49]:


FCG1=FCG1.groupby('Date')['Confirmed','Deaths','Recovered','Active','New cases','New deaths','New recovered'].sum().reset_index()


# In[50]:


FCG1


# In[60]:


FCG1['RecoveryRate']=FCG1['Recovered']/FCG1['Confirmed']


# In[61]:


FCG1


# In[62]:


Avrage_recovery_rate=FCG1['RecoveryRate'].mean()


# In[63]:


Avrage_recovery_rate


# ### Avrage_recovery_rate=0.34426219002904174

# In[64]:


FCG1['DeathRate']=FCG1['Deaths']/FCG1['Confirmed']


# In[65]:


FCG1


# In[66]:


Avrage_Death_rate=FCG1['DeathRate'].mean()


# In[67]:


Avrage_Death_rate


# ### Avrage_Death_rate=0.04862476305829724

# In[ ]:




