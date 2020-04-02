import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("Sales.csv",encoding='latin1')
x=data.iloc[:,:-1].values
y=data.iloc[:,24].values
print("\t\t\t\t\t\tSales Predictions")
print("Dataset head :: ",data.head())
print("dataset shape :: ",data.shape)

#encodeing
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()
x[:,5]=labelencoder_x.fit_transform(x[:,5])
x[:,6]=labelencoder_x.fit_transform(x[:,6])
x[:,10]=labelencoder_x.fit_transform(x[:,10])
x[:,12]=labelencoder_x.fit_transform(x[:,12])
x[:,13]=labelencoder_x.fit_transform(x[:,13])
x[:,14]=labelencoder_x.fit_transform(x[:,14])
x[:,15]=labelencoder_x.fit_transform(x[:,15])
x[:,16]=labelencoder_x.fit_transform(x[:,16].astype(str))
x[:,17]=labelencoder_x.fit_transform(x[:,17])
x[:,18]=labelencoder_x.fit_transform(x[:,18].astype(str))
x[:,19]=labelencoder_x.fit_transform(x[:,19].astype(str))
x[:,20]=labelencoder_x.fit_transform(x[:,20])
x[:,21]=labelencoder_x.fit_transform(x[:,21].astype(str))
x[:,22]=labelencoder_x.fit_transform(x[:,22])
x[:,23]=labelencoder_x.fit_transform(x[:,23])
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)


#after backword elimination
x=x[:,[1,2,4]]

#Imputation
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy='mean',axis=1)
x=imputer.fit_transform(x)

#split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#standardlization
from sklearn.preprocessing import StandardScaler
st_x=StandardScaler()
x_train=st_x.fit_transform(x_train)
x_test=st_x.transform(x_test)

#fitting
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#score and error
print("Train score :: ",regressor.score(x_train,y_train))
print("Test score :: ",regressor.score(x_test,y_test))
y_pred=regressor.predict(x_test)
from sklearn.metrics import mean_squared_error
print("Mean Squared error :: ",mean_squared_error(y_test,y_pred))


#backword elemination
import statsmodels.api as sm
x = np.append(arr = np.ones((2823, 1)).astype(int),  values = x, axis = 1)
#x_opt=x[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]]
#x_opt=x[:,[2,3,5]]
x_opt=x[:,[0]]
ols=sm.OLS(endog=y,exog=x_opt).fit()
print(ols.summary())


