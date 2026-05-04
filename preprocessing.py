import pandas as pd
df=pd.read_excel(r"C:\Users\ANANTHKUMAR YASARLA\Downloads\Dataset (15)\New folder\IEX_Weather_final.xlsx")
print(df.columns)
df.shape
df.head()
df.tail()
df.info()
df.isnull().sum()
df.duplicated().sum()
df.describe()
df.dtypes
df.columns
df = df.dropna()
#mean
num_cols=df.select_dtypes(include=['int64','float64']).columns
for col in num_cols:
    print(f"{col} mean:",df[col].mean())
#median
for col in num_cols:
    print(f"{col}median:",df[col].median())
#mode  
for col in num_cols :
    print(f"{col}mode:",df[col].mode().iloc[0])

#variance
for col in num_cols:
    print(f"{col}variance:",df[col].var())
#standardeviation
for col in num_cols:
    print(f"{col}standardeviation:",df[col].std())
#range
for col in num_cols:
    print(f"{col}range:",df[col].max()- df[col].min())

#skewness
for col in num_cols:
    print(f"{col}skewness:",df[col].skew())

#kurtosis
for col in num_cols:
    print(f"{col}kurtosis:",df[col].kurt())
