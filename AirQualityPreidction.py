# %%
# import libraries

import os 
import pandas as pd
from datetime import timedelta
from matplotlib import pyplot as plt  

os.chdir('C:/Users/Tam/Documents/Tam/Delhi-Air-Quality-Prediction-master')

# %%
# read weather data

fileA = pd.read_excel('city_weather.xls')
fileA['Date'] = pd.to_datetime(fileA['Date'],format='%d.%m.%Y %H:%M')
fileA['Datetime'] = fileA['Date'] - timedelta(minutes=30)
fileA.drop(['Date'],axis=1, inplace=True)
fileA.head()

# %%
temp_fileA = fileA[['T','Po','P','Pa','U','VV','Td','Datetime']]
temp_fileA.head()

# %%
# read air quality data

fileB = pd.read_csv('city_hour.csv')
fileB = pd.DataFrame(fileB)
fileC = fileB.set_index('City')
fileD = fileC.loc['Delhi']
temp_fileD = fileD.drop(['AQI','AQI_Bucket'],axis=1)
temp_fileD['Datetime'] = pd.to_datetime(temp_fileD['Datetime'])
temp_fileD.head()

# %%
combine = pd.merge(temp_fileA, temp_fileD, on='Datetime')
final = combine[['T','Po','P','Pa','U','VV','Td','PM2.5']]
final.isnull().sum()

# %%
final_filled = final.fillna(method='ffill')
final_filled.info()

# %%
final_filled.loc[final_filled['VV'] == "less than 0.1", 'VV'] = 0.1
final_filled.loc[final_filled['VV'] == "less than 0.05", 'VV'] = 0.05

# %%
# Applied machine learning algorithms

y = final_filled['PM2.5']
X = final_filled.drop(['PM2.5'], axis=1)

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1234)

# %%
X_train.shape, y_train.shape

# %%
X_test.shape, y_test.shape

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create linear regression object
regr = LinearRegression()

# %%
# Train the model using the training sets
regr.fit(X_train, y_train)

# %%
# Make predictions using the testing set
lin_pred = regr.predict(X_test)

# %%
linear_regression_score = regr.score(X_test, y_test)
linear_regression_score

# %%
from math import sqrt
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, lin_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, lin_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, lin_pred))

# %%
plt.scatter(y_test, lin_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Linear Regression Predicted vs Actual')
plt.show()

# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Create Random Forrest Regressor object
regr_rf = RandomForestRegressor(n_estimators=200, random_state=1234)

# %%
regr_rf.fit(X_train, y_train)

# %%
# Score the model
decision_forest_score = regr_rf.score(X_test, y_test)
decision_forest_score

# %%
# Make predictions using the testing set
regr_rf_pred = regr_rf.predict(X_test)

# %%
# The mean squared error
print("Root mean squared error: %.2f"
      % sqrt(mean_squared_error(y_test, regr_rf_pred)))
# The absolute squared error
print("Mean absolute error: %.2f"
      % mean_absolute_error(y_test, regr_rf_pred))
# Explained variance score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(y_test, regr_rf_pred))

# %%
plt.scatter(y_test, regr_rf_pred)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Decision Forest Predicted vs Actual')
plt.show()


