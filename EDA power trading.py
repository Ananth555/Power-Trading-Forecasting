import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor



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

#Histogram
for i in range(0, len(num_cols), 9):   # process 9 columns at a time
    cols = num_cols[i:i+9]
    
    plt.figure(figsize=(12,8))
    
    for j, col in enumerate(cols, 1):
        plt.subplot(3, 3, j)
        sns.histplot(df[col], kde=True)
        plt.title(col)
    
    plt.tight_layout()
    plt.show()
    
#Boxplot
for i in range(0, len(num_cols), 9):
    cols = num_cols[i:i+9]
    
    plt.figure(figsize=(12,8))
    
    for j, col in enumerate(cols, 1):
        plt.subplot(3, 3, j)
        sns.boxplot(x=df[col])
        plt.title(col)
    
    plt.tight_layout()
    plt.show()
    
#Denisty plot
for i in range(0,len(num_cols),9):
    cols=num_cols[i:i+9]
    plt.figure(figsize=(12,8))
    for j,col in enumerate(cols,1):
        plt.subplot(3,3,j)
        sns.kdeplot(x=df[col])
        plt.title(col)
    plt.tight_layout()
    plt.show()

#Scatter Plot
sns.scatterplot(x=df[num_cols[0]], y=df[num_cols[1]])
plt.title(f"{num_cols[0]} vs {num_cols[1]}")
plt.show()

#Heat Map
plt.figure(figsize=(15,10))
sns.heatmap(df[num_cols].corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#pair plot
sns.pairplot(df[num_cols[:5]])
plt.title(col)
plt.show()

#Detect Outliers
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers")
#Replace Outliers
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = df[col].clip(lower, upper)

df = df.drop(columns=['target'], errors='ignore')

#lag features
target = 'MCP (Rs/MWh) *'

df['lag_1'] = df[target].shift(1)     # previous 15 min
df['lag_4'] = df[target].shift(4)     # previous 1 hour
df['lag_96'] = df[target].shift(96)   # previous day (24 hrs)
df = df.dropna()

selected_features = [
    'Hour',
    'Purchase Bid (MW)',
    'Sell Bid (MW)',
    'MCV (MW)',
    'Final Scheduled Volume (MW)',
    'Mundra_temperature_2m (°C)',
    'Mundra_relative_humidity_2m (%)',
    'Mundra_cloud_cover (%)',
    'Mundra_wind_speed_10m (km/h)',
    'lag_1',
    'lag_4',
    'lag_96'
]

X = df[selected_features]
y = df[target]

#modeling
target = 'MCP (Rs/MWh) *'
X = df.drop(columns=[target])
y = df[target]
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#RandomForest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
import pickle
with open(r"C:\Power Trading Forecasting\Streamlit\model.pkl", "wb") as f:
    pickle.dump(model, f)

y_pred = model.predict(X_test)

print("MAPE:", mean_absolute_percentage_error(y_test, y_pred) * 100)

#LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred_lr = lr.predict(X_test)

mape_lr = mean_absolute_percentage_error(y_test, y_pred_lr)*100
print("Linear Regressioin MAPE:",mape_lr)

# XGBoost
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

mape_xgb = mean_absolute_percentage_error(y_test,y_pred_xgb)*100
print("XGBoost MAPE:",mape_xgb)

#Feature Importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(importance.head(10))

#Scatter Plot
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.show()

#Histogram
residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()
