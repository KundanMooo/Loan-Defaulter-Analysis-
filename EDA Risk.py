#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('D:/DS/resume projects/EDA risk analysis/application_data.csv')


# In[3]:


#overviewing data
df


# In[4]:


#checking unique ID
df[df.duplicated('SK_ID_CURR')==True]


# In[5]:


# where data cleaning needed
df.describe()


# In[6]:


df.info()


# In[7]:


# finding % of null values
dfna=df.isnull().mean()*100


# In[8]:


dfna


# In[9]:


# identifying columns with more than 45% null values
nn=df.isna().sum().sort_values(ascending=False)
nn=nn[nn.values>0.45*len(df)].reset_index()
#nn=nn.rename(columns={'index':'null'})
nn=nn.rename(columns={'index': 'columns', 0: 'count'})
nn


# In[10]:


#ploting columns with more than 45% nulls
plt.figure(figsize=(15,5))
sns.barplot(x='columns', y='count', data=nn)
plt.xticks(rotation=90)
plt.show()


# In[11]:


#dropping columns with more than 45% null 
l=list(nn['columns'])
dfd=df.drop(l,axis=1)
dfdd=dfd.isna().sum().sort_values(ascending=False)
dfdd=dfdd.reset_index()
dfdd.iloc[:,1]
dfdd['pp']=(100*dfdd.iloc[:,1])/len(df)


# In[12]:


dfdd


# In[13]:


#cleaning  Query with time columns 
df.AMT_REQ_CREDIT_BUREAU_YEAR.value_counts()


# In[14]:


df.AMT_REQ_CREDIT_BUREAU_YEAR.value_counts().plot(kind='bar')


# In[15]:


l=['AMT_REQ_CREDIT_BUREAU_YEAR','AMT_REQ_CREDIT_BUREAU_QRT','.AMT_REQ_CREDIT_BUREAU_MON']


# In[16]:


l


# In[17]:


dfd1=dfd


# In[18]:


dfd1.AMT_REQ_CREDIT_BUREAU_HOUR.mode()


# In[19]:


# zero is most repeating number in this columns hence replacing null by zero
dfd1[['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']]=dfd[['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']].fillna(0)


# In[20]:


# cleaning AMT_ANNUITY column
plt.figure(figsize=(10,1))
sns.displot(dfd1.AMT_ANNUITY)
plt.xticks(range(0,20000,10000))
plt.show()


# In[21]:


dfd1.AMT_ANNUITY.isna().sum()


# In[22]:


dfd1['AMT_ANNUITY']=dfd['AMT_ANNUITY'].fillna(dfd.AMT_ANNUITY.median())


# In[23]:


dfd1.AMT_ANNUITY.isna().sum()


# In[24]:


# cleaning Gender column
dfd1.CODE_GENDER.value_counts()


# In[25]:


# remove unwanter gender
dfd1.loc[dfd1.CODE_GENDER=='XNA','CODE_GENDER']=np.NaN


# In[26]:


dfd1.CODE_GENDER.value_counts()


# In[27]:


dfd1


# In[28]:


#removing unwanter organization type
dfd1.ORGANIZATION_TYPE


# In[29]:


dfd1.loc[dfd1.ORGANIZATION_TYPE=='XNA','ORGANIZATION_TYPE']=np.NaN


# In[30]:


#making 5 category in income
dfd1['AMT_INCOME_RANGE']=pd.qcut(dfd1.AMT_INCOME_TOTAL,q=[0,0.2,0.5,0.85,0.95,1],labels=['VERY_LOW','LOW','MEDIUM','HIGH','VERY HIGH'])


# In[31]:


dfd1['AMT_INCOME_RANGE']


# In[32]:


sns.displot(dfd1['AMT_INCOME_RANGE'])


# In[33]:


#cleaning days column which have negative values
err=[i for i in dfd1 if i.startswith('DAYS')]


# In[34]:


err


# In[35]:


dfd1[err]=abs(dfd1[err])


# In[36]:


# making category based or age
dfd1.DAYS_BIRTH=(dfd1.DAYS_BIRTH/365).astype(int)
dfd1['DAYS_BIRTH_BINS']=pd.cut(dfd1.DAYS_BIRTH,bins=[16,25,35,40,60,100],labels=['VERY_YOUNG','YOUNG','MEDIUM AGE','SENIOR_CITIZEN','OLD'])


# In[37]:


dfd1['DAYS_BIRTH_BINS'].value_counts()


# In[38]:


#getting data of defaulters and non_defaulters
dfd1.TARGET.value_counts()


# In[39]:


defolter=dfd1[dfd1.TARGET==1]
non_defolter=dfd1[dfd1.TARGET==0]


# In[40]:


######## visialisation
#loan application by occupation type
c=dfd1.OCCUPATION_TYPE.value_counts()
sns.barplot(x=c.index,y=c.values)
plt.xticks(rotation=90)
plt.show()


# In[60]:


# percentage of non_defaulter by gender

fi, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].set_title('non_defaulter by gender')
ax[0].pie(non_defolter.CODE_GENDER.value_counts(),autopct='%1.2f%%')

ax[1].set_title('defaulter by gender')
ax[1].pie(defolter.CODE_GENDER.value_counts(),autopct='%1.2f%%')

plt.tight_layout()
plt.show()


# In[62]:


# percentage of defaulter by income type
# plt.pie(defolter.NAME_INCOME_TYPE.value_counts(),autopct='%1.2f%%')
# plt.show()



fi, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].set_title('defaulter by Income type ')
ax[0].pie(defolter.NAME_INCOME_TYPE.value_counts(),autopct='%1.2f%%')

ax[1].set_title('non_defaulter by Income type ')
ax[1].pie(non_defolter.NAME_INCOME_TYPE.value_counts(),autopct='%1.2f%%')

plt.tight_layout()
plt.show()


# In[44]:


# percentage of non_defaulter by income type
plt.pie(non_defolter.NAME_INCOME_TYPE.value_counts(),autopct='%1.2f%%')
plt.show()


# In[45]:


# Credit distribution of different income categoury and there family type  
def ss(col1, col2, col3):
    f,ax=plt.subplots(1,2,figsize=(15,5))
    ax[0].set_title('Boxplot for Defolter')
    sns.boxplot(data=defolter,x=col1, y=col2, hue=col3, ax=ax[0])
    
    ax[1].set_title('Boxplot for Non-Defolter')
    sns.boxplot(data=non_defolter,x=col1, y=col2, hue=col3, ax=ax[1])

    plt.show()

ss('AMT_INCOME_RANGE', 'AMT_CREDIT', 'NAME_FAMILY_STATUS')


# In[52]:


f0=dfd1.groupby('AMT_INCOME_RANGE')['TARGET'].count()
f=defolter.groupby('AMT_INCOME_RANGE')['TARGET'].count()
pct=round(f0/f,1)


# In[53]:


pct


# In[55]:



fi, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].set_title('Barplot for Total')
sns.barplot(x=f0.index, y=f0.values, ax=ax[0])

ax[1].set_title('Percentage Defolter')
sns.barplot(x=pct.index, y=pct.values, ax=ax[1])

plt.tight_layout()
plt.show()


# In[57]:


#gLANCING CORRELATION BETWEEN VARIOUS NUMERIC COLUMNS
sns.heatmap(dfd1[['AMT_REQ_CREDIT_BUREAU_YEAR','AMT_GOODS_PRICE','AMT_INCOME_TOTAL','AMT_ANNUITY','DAYS_BIRTH']].corr())


# In[ ]:




