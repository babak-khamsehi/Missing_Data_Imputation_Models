
# coding: utf-8

# <h1 align='center'> Missing Data Handling and their Imputation Models  </h1>
# <h1 align='center'> Courtesy of Numeris Inc, Toronto, Canada  </h1>
# <h1 align='center'> Developed by Babak Khamsehi, June 24th- 28th, 2018 </h1>

# <img src='Numeris.png'>

# ### Import some libraries and reading in the data file

# In[1]:

import pandas as pd
df= pd.read_csv('C:/Google Drive/Presentation Task/data/RANDRANDOMSAMPLEFA15RDD.csv')
df.columns


# In[2]:

education_classes= df['EDUCATION COMPLETED'].unique()
education_classes


# In[3]:

df1=df.copy() # A copy of df because we are dummy coding for categorical features
import numpy as np
df1.loc[:,:].replace('NO REPLY' , np.nan , inplace=True)


# In[3]:


df1['EDUCATION COMPLETED'].unique()


# ### Exploring the Missingness Potential Patterns in the Features 

# In[4]:


df1 = df1.replace("nan", np.nan)
edu= df1["EDUCATION COMPLETED"]
pd.isnull(edu).sum()


# In[144]:


from quilt.data.ResidentMario import missingno_data
from quilt.data.ResidentMario import missingno_data 
import missingno as msno
# Necessary for Jupyter, not for Spyder
#get_ipython().magic('matplotlib inline')
msno.matrix(df1.sample(10000))




# In[145]:


msno.bar(df1.sample(10000))


# In[146]:


msno.heatmap(df1)


# In[147]:


msno.dendrogram(df1)


# ### Exploring the features

# In[148]:


df1.columns


# ### Exploring numerical features

# In[149]:


# Slicing the dataframe into (numerical) versus (categorical & Ordinal) dataframes 
df1_num = df1[['RESPID', 'HHID', 'RESP', 'CELL', 'Weight', 'Reach ALL', 'Reach CAAA',
               'Reach CBBB', 'Imp ALL', 'Imp CAAA', 'Imp CBBB', 
               'A18+ AT HOME', 'T12-17 AT HOME', 'C7-11 AT HOME', 'C0-6 AT HOME', ]]
# ^ 15 numerical features
df1_num.head()


# ### Primary id/key variable dataframe

# ### Scatterplot matrices : linear correlation between primary id/key features?

# In[150]:


import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
df2= df1_num.copy()
df2_pi = df2[['RESPID', 'HHID', 'RESP', 'CELL']]      # primary id variable dataframe
get_ipython().magic('matplotlib inline')

from matplotlib import cm
cmap = cm.get_cmap('gnuplot')

scatter = pd.plotting.scatter_matrix(df2_pi, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)


# In[151]:


df2_pi.head(10)


# ### Tunning dataframe

# ### Scatterplot matrices : linear correlation between tunning features?

# In[152]:


df2_tu = df2[['Weight', 'Reach ALL', 'Reach CAAA',
             'Reach CBBB', 'Imp ALL', 'Imp CAAA', 'Imp CBBB']]      # Tunning dataframe

scatter = pd.plotting.scatter_matrix(df2_tu, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)


# In[153]:


df2_tu.head(10)


# ### Numbers of Human dataframe

# ### Scatterplot matrices : linear correlation between numbers of human features?

# In[154]:


df2_nh = df2[['A18+ AT HOME', 'T12-17 AT HOME', 
              'C7-11 AT HOME', 'C0-6 AT HOME']]      # numbers of human dataframe

scatter = pd.plotting.scatter_matrix(df2_nh, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)


# In[155]:


df2_nh.head(10)


# ### Exploring cateogorical, ordinal and binary features

# In[156]:


df1_cat=df1.drop(df1_num , axis=1)
df1_cat.head()

pd.isnull(df1_cat).sum()
# In[157]:


df1_cat= df1_cat.rename(columns={'CHILDREN < 12 ?': 'c_u_12'})
df1_cat['c_u_12'].head(10)


# In[164]:


df1_cat.columns
df5= df1_cat # before factors 
df5.head() 


# ### Frequency tables for categorical features


# ### Number of ordinal verus categorical features 

# In[163]:


df5.shape[1] -2 



# ### Differentiting  ordinal  versys categorical (including binary) features

# ### Conceptual: 
# #### 4 ordinal features = 'AGE',   'HOURS WORKED', 'HOUSEHOLD SIZE', 'QUINTILES'
# 

# ###    We treat our target variables,  'HHLD INCOME', 'EDUCATION COMPLETED' as ordinal 
#      ' 

#  ### Dummy coding for 12 categorical variables
#       The first cateogorical feature is the reference group
#       Original  categorical varibales will be deleted 

# In[114]:


df1_cat.head()
df1_ord = df1_cat[[ 'AGE', 'HOURS WORKED', 'HOUSEHOLD SIZE', 'QUINTILES', # ordinal feature variables
                   'HHLD INCOME', 'EDUCATION COMPLETED' ]]                # ordinal targets
df1_ord.head()


# In[105]:


df1_cat2=df1_cat.drop(df1_ord , axis=1) # Double categorical 
#df1_cat3=df1_cat2.drop(['HHLD INCOME', 'EDUCATION COMPLETED']  , axis=1) # Triple categorical, dropped numerical, ordinal features, ordinal targets 
#df1_cat3.head() # Double categorical 
#df1_cat3.head()
#df4=df1_cat3


# In[106]:


df1_cat2.columns
pd.isnull(df1_cat2).sum()

df4= df1_cat2.copy()
# In[ ]:


##  Dummy coding for 12 categorical features variables


# In[108]:


# Dummy coding for 12 categorical variables
df5= pd.get_dummies(df4, columns=['SEX', 'HOME LANGUAGE', 'OFFICIAL LANGUAGE', 'MOTHER TONGUE',
       'NOT WORKING', 'HOUSEHOLD STATUS', 'c_u_12', 'FRANCO POP',
       'MARITAL STATUS', 'FEM HEAD W/CHILD', 'INDUSTRY', 'OCCUPATION'], drop_first=True) #
df5.head(10)


# ### Ordinal feature and target variables

# In[115]:


df1_ord.head()
df6= df1_ord.copy()

# ### Label encoding for ordinal features

# In[116]:


# Label Encoding for ordinal target variable, Education Level
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()

edu_mapping = {
       'NO CERT. OR DIPLOMA': 1,  
       'SECONDARY SCHOOL DIPLOMA OR EQUVIALENCY CERT.': 2, 
       'REG. APPRENTICESHIP, TRADES CERT. OR OTHER TRADES DIPLOMA': 3,
       'COLLEGE, CEGEP OR OTHER NON UNIV. CERT. OR DIPLOMA': 4,
       'UNIV. UNDERGRAD DEGREE, CERT. OR DIPLOMA': 5, 
       'UNIV. POSTGRAD DEGREE' : 6}

df6['EDUCATION COMPLETED'] = df6['EDUCATION COMPLETED'].map(edu_mapping)

df6['EDUCATION COMPLETED'].head(10)

# In[ ]:

# Label Encoding for ordinal target variable, HHDL income
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()

hhld_inc_mapping = {
       'UNDER $20,000':   1,  
       '$20,000-$29,999': 2, 
       '$30,000-$39,999': 3,
       '$40,000-$49,999': 4,
       '$50,000-$59,999': 5,
       '$60,000-$74,999': 6, 
       '$75,000-$99,999': 7,
       '$100,000-$124,999': 8,
       '$125,000 - $149,999': 9,
       '$150,000 OR MORE': 10}

df6['HHLD INCOME'] = df6['HHLD INCOME'].map(hhld_inc_mapping)

df6['HHLD INCOME'].head(10)



# In[]:

# Label Encoding for ordinal target variable, HHDL income
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()

q_mapping = {
       'Q1-LIGHT':   1,  
       'Q2-LIGHT/MED': 2, 
       'Q3-MEDIUM': 3,
       'Q4-MED/HEAVY': 4,
       'Q5-HEAVY': 5}

df6['QUINTILES'] = df6['QUINTILES'].map(q_mapping)

df6['QUINTILES'].head(10)



# In[]:

# Label Encoding for ordinal target variable, HHDL income
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()

s_mapping = {
       '1 PERSON':  1,  
       '2 PERSONS': 2, 
       '3 PERSONS': 3,
       '4 PERSONS': 4,
       '5 OR MORE PERSONS': 5, # 5 t0o 9 persons
       '10 PERSON': 10, 
       '11 PERSON': 11,
       '13 PERSON': 13,
       '15 PERSON': 15}

df6['HOUSEHOLD SIZE'] = df6['HOUSEHOLD SIZE'].map(s_mapping)

df6['HOUSEHOLD SIZE'].head(10)


# In[]:

# Label Encoding for ordinal target variable, HHDL income
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()

h_mapping = {
       'NONE':  1,  
       '1 to 19': 2, 
       '20 to 29': 3,
       '30 PLUS': 4}

df6['HOURS WORKED'] = df6['HOURS WORKED'].map(h_mapping)

df6['HOURS WORKED'].head(10)


# In[]:

# Label Encoding for ordinal target variable, HHDL income
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()

a_mapping = {
       '2 to 17':  1,  
       '18-24': 2, 
       '25-34': 3,
       '35-44': 4,
       '45-49': 5,
       '50-54': 6,
       '55-59': 7,
       '60-64': 8,
       '65 PLUS': 9}

df6['AGE'] = df6['AGE'].map(a_mapping)

df6['AGE'].head(10)



# In[]:

df7 = df6.copy()
# Dummy coding for 4 ordinal features,and 2 ordincal target variables 
df7= pd.get_dummies(df7, columns=['AGE', 'HOURS WORKED', 'HOUSEHOLD SIZE', 'QUINTILES', 'HHLD INCOME',
       'EDUCATION COMPLETED'], drop_first=True, dummy_na= True) #
df7.head(10)




# In[]:

# Combining the dummy coded datasets 

df_all = pd.concat([df1_num, df5, df7], axis=1, join_axes=[df1_num.index])
# ## Feature importance using Decision Tree Classifier

pd.isnull(df1_num).sum()
pd.isnull(df5).sum()
pd.isnull(df7).sum()

df_all.columns

# Exporting the dataframe into csv file whcih is fully dummy coded

# df_all.to_csv('C:/Google Drive/Presentation Task/final/df_all.csv')




