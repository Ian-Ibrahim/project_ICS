import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler

df=pd.read_csv("project_files/analytical_table.csv")
#df.select_dtypes(exclude=['object']).isnull().sum()
print(df.shape)
df.loc[(df['category']=='For Rent')]=df[df.price < 300000]
print(df.shape)
