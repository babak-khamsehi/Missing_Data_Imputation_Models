# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 06:53:18 2018

@author: HP
"""
# In[]:
import pandas as pd
df= pd.read_csv('C:/Google Drive/numeris_final/data/RANDRANDOMSAMPLEFA15RDD.csv')
df.columns

df1=df.copy() # A copy of df because we are dummy coding for categorical features
import numpy as np
df1.loc[:,:].replace('NO REPLY' , np.nan , inplace=True)


df1 = df1.replace("nan", np.nan)
edu= df1["EDUCATION COMPLETED"]
pd.isnull(edu).sum()

# Slicing the dataframe into (numerical) versus (categorical & Ordinal) dataframes 
df1_num = df1[['RESPID', 'HHID', 'RESP', 'CELL', 'Weight', 'Reach ALL', 'Reach CAAA',
               'Reach CBBB', 'Imp ALL', 'Imp CAAA', 'Imp CBBB', 
               'A18+ AT HOME', 'T12-17 AT HOME', 'C7-11 AT HOME', 'C0-6 AT HOME', ]]
# ^ 15 numerical features
df1_num.head()


df1_cat=df1.drop(df1_num , axis=1)
df1_cat.head()

pd.isnull(df1_cat).sum()

df1_cat= df1_cat.rename(columns={'CHILDREN < 12 ?': 'c_u_12'})
df1_cat['c_u_12'].head(10)


# In[159]:


# SEX and SEX cat
df1_cat["SEX_cat"] = pd.factorize(df1_cat["SEX"])[0]
# cross tables 
freq_SEX = pd.crosstab(index=[df1_cat["SEX_cat"], df1_cat["SEX"]],  # Make a crosstab
                        columns="count", margins=True)
print(freq_SEX) 
print()

# AGE and AGE cat
df1_cat["AGE_cat"] = pd.factorize(df1_cat["AGE"])[0]
# cross tables 
freq_AGE = pd.crosstab(index=[df1_cat["AGE_cat"], df1_cat["AGE"]],  # Make a crosstab
                        columns="count", margins=True)
print(freq_AGE) 
print()

# HOME LANGUAGE and HOME LANGUAGE cat
df1_cat["HOME LANGUAGE_cat"] = pd.factorize(df1_cat["HOME LANGUAGE"])[0]
# cross tables 
freq_HOME_LANGUAGE = pd.crosstab(index=[df1_cat["HOME LANGUAGE_cat"], df1_cat["HOME LANGUAGE"]],  # Make a crosstab
                        columns="count", margins=True)
print(freq_HOME_LANGUAGE) 
print()

# OFFICIAL LANGUAGE and OFFICIAL LANGUAGE cat
df1_cat["OFFICIAL LANGUAGE_cat"] = pd.factorize(df1_cat["OFFICIAL LANGUAGE"])[0]
# cross tables 
freq_OFFICIAL_LANGUAGE = pd.crosstab(index=[df1_cat["OFFICIAL LANGUAGE_cat"], df1_cat["OFFICIAL LANGUAGE"]],  # Make a crosstab
                        columns="count", margins=True)
print(freq_OFFICIAL_LANGUAGE) 
print()

# OFFICIAL LANGUAGE and OFFICIAL LANGUAGE cat
df1_cat["MOTHER TONGUE_cat"] = pd.factorize(df1_cat["MOTHER TONGUE"])[0]
# cross tables 
freq_MOTHER_TONGUE = pd.crosstab(index=[df1_cat["MOTHER TONGUE_cat"], df1_cat["MOTHER TONGUE"]],  # Make a crosstab
                        columns="count", margins=True)
print(freq_MOTHER_TONGUE) 
print()

# HOURS WORKED and HOURS WORKED cat
df1_cat["HOURS WORKED_cat"] = pd.factorize(df1_cat["HOURS WORKED"])[0]
# cross tables 
freq_HOURS_WORKED = pd.crosstab(index=[df1_cat["HOURS WORKED_cat"], df1_cat["HOURS WORKED"]],  # Make a crosstab
                        columns="count", margins=True)
print(freq_HOURS_WORKED) 
print()

# NOT WORKING  and NOT WORKING  cat
df1_cat["NOT WORKING_cat"] = pd.factorize(df1_cat["NOT WORKING"])[0]
# cross tables 
freq_NOT_WORKING= pd.crosstab(index=[df1_cat["NOT WORKING_cat"], df1_cat["NOT WORKING"]],  # Make a crosstab
                        columns="count", margins=True)
print(freq_NOT_WORKING) 
print()

# HOUSEHOLD STATUS  and HOUSEHOLD STATUS  cat
df1_cat["HOUSEHOLD STATUS_cat"] = pd.factorize(df1_cat["HOUSEHOLD STATUS"])[0]
# cross tables 
freq_HOUSEHOLD_STATUS= pd.crosstab(index=[df1_cat["HOUSEHOLD STATUS_cat"], df1_cat["HOUSEHOLD STATUS"]],  # Make a crosstab
                        columns="count", margins=True)
print(freq_HOUSEHOLD_STATUS) 
print()

# HOUSEHOLD SIZE  and HOUSEHOLD SIZE  cat
df1_cat["HOUSEHOLD SIZE_cat"] = pd.factorize(df1_cat["HOUSEHOLD SIZE"])[0]
# cross tables 
freq_HOUSEHOLD_SIZE= pd.crosstab(index=[df1_cat["HOUSEHOLD SIZE_cat"], df1_cat["HOUSEHOLD SIZE"]],  # Make a crosstab
                        columns="count", margins=True)
print(freq_HOUSEHOLD_SIZE) 
print()

# FRANCO POP  and FRANCO POP SIZE  cat
df1_cat["FRANCO POP_cat"] = pd.factorize(df1_cat["FRANCO POP"])[0]
# cross tables 
freq_FRANCO_POP= pd.crosstab(index=[df1_cat["FRANCO POP_cat"], df1_cat["FRANCO POP"]],  # Make a crosstab
                        columns="count", margins=True)
print(freq_FRANCO_POP) 
print()

# c_u_12  and c_u_12 SIZE  cat
df1_cat["c_u_12_cat"] = pd.factorize(df1_cat["c_u_12"])[0]
# cross tables 
freq_c_u_12= pd.crosstab(index=[df1_cat["c_u_12_cat"], df1_cat["c_u_12"]],  # Make a crosstab
                        columns="count", margins=True)
print(freq_c_u_12) 
print()

# MARITAL STATUS  and MARITAL STATUS  cat
df1_cat["MARITAL STATUS_cat"] = pd.factorize(df1_cat["MARITAL STATUS"])[0]
# cross tables 
freq_MARITAL_STATUS= pd.crosstab(index=[df1_cat["MARITAL STATUS"], df1_cat["MARITAL STATUS_cat"]],  # Make a crosstab
                        columns="count", margins=True)
print(freq_MARITAL_STATUS) 
print()

# FEM HEAD W/CHILD and FEM HEAD W/CHILD  cat
df1_cat["QUINTILES_cat"] = pd.factorize(df1_cat["QUINTILES"])[0]
# cross tables 
freq_QUINTILES= pd.crosstab(index=[df1_cat["QUINTILES"], df1_cat["QUINTILES_cat"]],  # Make a crosstab
                        columns="count", margins=True)
print(freq_QUINTILES) 
print()

# FEM HEAD W/CHILD and FEM HEAD W/CHILD cat
df1_cat["FEM HEAD W/CHILD_cat"] = pd.factorize(df1_cat["FEM HEAD W/CHILD"])[0]
# cross tables 
freq_F_C= pd.crosstab(index=[df1_cat["FEM HEAD W/CHILD"], df1_cat["FEM HEAD W/CHILD_cat"]],  # Make a crosstab
                        columns="count", margins=True)
print(freq_F_C) 
print()

# INDUSTRY and INDUSTRY cat
df1_cat["INDUSTRY_cat"] = pd.factorize(df1_cat["INDUSTRY"])[0]
# cross tables 
freq_INDUSTRY= pd.crosstab(index=[df1_cat["INDUSTRY"], df1_cat["INDUSTRY_cat"]],  # Make a crosstab
                        columns="count", margins=True)
print(freq_INDUSTRY) 
print()

# INDUSTRY and INDUSTRY cat
df1_cat["OCCUPATION_cat"] = pd.factorize(df1_cat["OCCUPATION"])[0]
# cross tables 
freq_OCCUPATION= pd.crosstab(index=[df1_cat["OCCUPATION"], df1_cat["OCCUPATION_cat"]],  # Make a crosstab
                        columns="count", margins=True)
print(freq_OCCUPATION) 
print()


