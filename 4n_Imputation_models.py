
# In[]:
import pandas as pd
df_all= pd.read_csv('C:/Google Drive/Presentation Task/final/df_all.csv')
import numpy as np
df_all.loc[:,:].replace('NO REPLY' , np.nan , inplace=True)
df_all = df_all.replace("nan", np.nan)


pd.isnull(df_all).sum()


df= pd.read_csv('C:/Google Drive/Presentation Task/data/RANDRANDOMSAMPLEFA15RDD.csv')

import numpy as np
df.loc[:,:].replace('NO REPLY' , np.nan , inplace=True)
df = df.replace("nan", np.nan)
pd.isnull(df).sum()

#
#pd.isnull(df_all['Imp CAAA']).sum()
#pd.isnull(df_all['Imp CBBB']).sum()

pd.isnull(df).sum()

df_all.columns
# import numpy as np
c1= pd.DataFrame(df_all['Imp CAAA']) 
c2= pd.DataFrame(df_all['Imp CBBB']) 
c3= pd.DataFrame(df['HHLD INCOME']) 
c4= pd.DataFrame(df['EDUCATION COMPLETED']) 

df_s = pd.concat([c1, c2, c3, c4], axis=1, join_axes=[c1.index])
pd.isnull(df_s).sum()


# In[]:
from quilt.data.ResidentMario import missingno_data
from quilt.data.ResidentMario import missingno_data 
import missingno as msno

msno.matrix(df_s.sample(10000))
msno.bar(df_s.sample(10000))


# In[]:

# Before imputation, X_missing, y_missing

X_num_missing = df_all.copy()
y_missing = y_full.copy()
pd.isnull(X_num_missing).sum()
 
X_num_missing.loc[:,:].replace('np.nan' , 0 , inplace=True)




# In[146]:


