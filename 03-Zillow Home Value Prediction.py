#Zillow Home Value Prediction
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sb 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn import metrics 
from sklearn.svm import SVC 
from xgboost import XGBRegressor 
from sklearn.linear_model import LinearRegression, Lasso, Ridge 
from sklearn.ensemble import RandomForestRegressor 

import warnings 
warnings.filterwarnings('ignore') 

df = pd.read_csv('Zillow.csv') 
df.head()
df.shape

to_remove = [] 
for col in df.columns: 
	if df[col].nunique() == 1: 
		to_remove.append(col) 
	elif (df[col].isnull()).mean() > 0.60: 
		to_remove.append(col) 

print(len(to_remove)) 

df.drop(to_remove,
		axis=1,
		inplace=True)

df.info()

df.isnull().sum().plot.bar()
plt.show()

for col in df.columns:
	if df[col].dtype == 'object':
		df[col] = df[col].fillna(df[col].mode()[0])
	elif df[col].dtype == np.number: 
        df[col] = df[col].fillna(df[col].mean()) 
  
df.isnull().sum().sum() 	
	
ints, objects, floats = [], [], [] 

for col in df.columns: 
	if df[col].dtype == float: 
		floats.append(col) 
	elif df[col].dtype == int: 
		ints.append(col) 
	else: 
		objects.append(col) 

len(ints), len(floats), len(objects) 
for col in objects: 
	print(col, ' -> ', df[col].nunique()) 
	print(df[col].unique()) 
	print() 
plt.figure(figsize=(8, 5)) 
sb.distplot(df['target']) 
plt.show() 
plt.figure(figsize=(8, 5)) 
sb.boxplot(df['target']) 
plt.show() 
print('Shape of the dataframe before removal of outliers', df.shape) 
df = df[(df['target'] > -1) & (df['target'] < 1)] 
print('Shape of the dataframe after removal of outliers ', df.shape) 
for col in objects: 
	le = LabelEncoder() 
	df[col] = le.fit_transform(df[col]) 
plt.figure(figsize=(15, 15)) 
sb.heatmap(df.corr() > 0.8, 
		annot=True, 
		cbar=False) 
plt.show() 
to_remove = ['calculatedbathnbr', 'fullbathcnt', 'fips', 
			'rawcensustractandblock', 'taxvaluedollarcnt', 
			'finishedsquarefeet12', 'landtaxvaluedollarcnt'] 

df.drop(to_remove, axis=1, inplace=True) 
features = df.drop(['parcelid'], axis=1) 
target = df['target'].values 

X_train, X_val,\ 
	Y_train, Y_val = train_test_split(features, target, 
									test_size=0.1, 
									random_state=22) 
X_train.shape, X_val.shape 
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
X_val = scaler.transform(X_val) 
from sklearn.metrics import mean_absolute_error as mae 
models = [LinearRegression(), XGBRegressor(), 
		Lasso(), RandomForestRegressor(), Ridge()] 

for i in range(5): 
	models[i].fit(X_train, Y_train) 

	print(f'{models[i]} : ') 

	train_preds = models[i].predict(X_train) 
	print('Training Error : ', mae(Y_train, train_preds)) 

	val_preds = models[i].predict(X_val) 
	print('Validation Error : ', mae(Y_val, val_preds)) 
	print() 
