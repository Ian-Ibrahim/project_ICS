import pandas as pd
df=pd.read_csv("project_files/cleaned_listings.csv")
df_house=df[df.state=='Kajiado']
print(df_house.locality.value_counts())