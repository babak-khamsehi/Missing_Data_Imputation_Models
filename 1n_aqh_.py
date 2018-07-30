#******************************************************************************************************
# Developing Imputation models for Numeris survey data 
# *****************************************************************************************************
# Created by Babak khamsehi (BK), June 25th, 2018
# Modified by BK, June ..., 2018

#*********************************************************************************************************************

# ********************************************************************************************************************
# Reading in the Numeris Survey data for 2015

# In[1]:
import pandas as pd
df= pd.read_csv('C:/Google Drive/numeris_final/data/RANDRANDOMSAMPLEFA15RDD.csv')
df.columns

# In[2]:
# "Exploring SEX feature
df["sex_cat"] = pd.factorize(df["SEX"])[0]
# cross tables 
freq_sex_cat = pd.crosstab(index=[df["sex_cat"], df["SEX"]],  # Make a crosstab
                        columns="count", margins= True)
print(freq_sex_cat) 

#*********************************************************************************************************************
# ^^^ roughly 54% Female and 46% Male respondents  ^^^
# ^^^ Only two classes of SEX, zero missing value  ^^^
#*********************************************************************************************************************
# In[3]:
# Computing AQH(000) for Male and Female respondents
# slicing the df into dfm(for male) and dff(df for female)
dfm=df[df['SEX']=='MALE'] ## Slicing the df into only MALE 
# Computing AQH(000) for Male
AQH000_m_CAAA = 1/(4480*1000)*(dfm['Imp CAAA'].sum())       # for dfm and  Imp CAAA
print(AQH000_m_CAAA)
## ^^ AQH000_f_CAAA = 66.45603868772312

# Computing AQH(000) for Female
dff=df[df['SEX']=='FEMALE'] ## Slicing the df into only MALE 

AQH000_f_CAAA = 1/(4480*1000)*(dff['Imp CAAA'].sum())      # for dff and  Imp CAAA
print(AQH000_f_CAAA)
## ^^ AQH000_f_CAAA = 67.24764970111619


# In[4]:

# Computing Imp_CAAA by EDUCATION COMPLETED before the imputation models for missing models
# convert non-numeric to factors

# Frequency tables for "EDUCATION COMPLETED" features
# edu_cat and EDUCATION COMPLETED
df["edu_cat"] = pd.factorize(df["EDUCATION COMPLETED"])[0]

# cross tables 
freq_edu_cat = pd.crosstab(index=[df["edu_cat"], df["EDUCATION COMPLETED"]],  # Make a crosstab
                        columns="count", margins=True)
print(freq_edu_cat) 

#*********************************************************************************************************************
# ^^^ 549 (1.61%)  "NO REPLY"'s  out of 34174 respondents ^^^
# ^^^ 7 classes/categories of EDUCATION COMPLETED levels  ^^^
# ********************************************************************************************************************

df['Imp CAAA_new'] = df['Imp CAAA']                         # backuping the variable
df= df.rename(columns={'Imp CAAA_new': 'Imp_CAAA'})         # rename to Imp_CAAA


df['Imp CBBB_new'] = df['Imp CBBB']                         # backuping the variable
df= df.rename(columns={'Imp CBBB_new': 'Imp_CBBB'})     # rename to Imp_CAAA

# groupping the IMP_CAA by EDUCATION COMPLETED & Computing the sum for each group naming it to total_edu_cat & resetting the index
aqh_edu_cat_A = df.groupby('EDUCATION COMPLETED')["Imp_CAAA"].sum().rename("total_edu_cat_A").reset_index()
# Multiplying by 1/(4480*1000) to add the aqh_edu for each edu_cat
aqh_edu_cat_A["AQH000_CAAA "] = 1/(4480*1000)* (aqh_edu_cat_A["total_edu_cat_A"])

# groupping the IMP_CBBB by EDUCATION COMPLETED & Computing the sum for each group naming it to total_edu_cat & resetting the index
aqh_edu_cat_B = df.groupby('EDUCATION COMPLETED')["Imp_CBBB"].sum().rename("total_edu_cat_B").reset_index()
# Multiplying by 1/(4480*1000) to add the aqh_edu for each edu_cat
aqh_edu_cat_B["AQH000_CBBB "] = 1/(4480*1000)* (aqh_edu_cat_B["total_edu_cat_B"])

# In[5]:

# Computing Imp_CAAA by HHLD INCOME before the imputation models for missing models
# convert non-numeric to factors

# "hhld_income_cat and HHLD INCOME
df["hhld_income_cat"] = pd.factorize(df["HHLD INCOME"])[0]
# cross tables 
freq_hhld_income = pd.crosstab(index=[df["hhld_income_cat"], df["HHLD INCOME"]],  # Make a crosstab
                        columns="count", margins= True)
print(freq_hhld_income) 

#*********************************************************************************************************************
# ^^^ 149 (0.44%) "NO REPLY"'s  out of 34174 respondents  ^^^
# ^^^ 11 classes/categories of HHLD INCOME levels         ^^^
#*********************************************************************************************************************

# groupping the IMP_CAA by EDUCATION COMPLETED & Computing the sum for each group naming it to total_edu_cat & resetting the index
aqh__hhld_income_A = df.groupby('HHLD INCOME')["Imp_CAAA"].sum().rename("total_hhld_income_A").reset_index()
# Multiplying by 1/(4480*1000) to add the aqh_edu for each edu_cat
aqh__hhld_income_A["AQH000_CAAA"] = 1/(4480*1000)* (aqh__hhld_income_A["total_hhld_income_A"])

# groupping the IMP_CBBB by EDUCATION COMPLETED & Computing the sum for each group naming it to total_edu_cat & resetting the index
aqh_hhld_income_B = df.groupby('HHLD INCOME')["Imp_CBBB"].sum().rename("total_hhld_income_B").reset_index()
# Multiplying by 1/(4480*1000) to add the aqh_edu for each edu_cat
aqh_hhld_income_B["AQH000_CBBB "] = 1/(4480*1000)* (aqh_hhld_income_B["total_hhld_income_B"])

