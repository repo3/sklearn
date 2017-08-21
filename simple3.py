import pandas as pd

# http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv
data=pd.read_csv('Advertising.csv',index_col=0)
#print data.head()
#print data.tail()
print data.shape

import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(data,x_vars=['TV','Radio','Newspaper'],y_vars='Sales',size=7,aspect=.7,kind='reg')
#plt.show()

feature_cols=['TV','Radio','Newspaper']
X=data[feature_cols]
X=data[['TV','Radio','Newspaper']]
X.head()

print X.head()
y=data['Sales']
print y.head()
print X.shape
print y.shape

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)
print X_train.shape
print y_train.shape
print X_test.shape
print y_test.shape

from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
linreg.fit(X_train,y_train)
print linreg.intercept_
print linreg.coef_
for fea,coef in zip(feature_cols,linreg.coef_):
  print '%10s %f' % (fea,coef)
y_pred=linreg.predict(X_test)

# model evaluation for linear regression (RMSE: Root Mean Square Error)
from sklearn import metrics
import numpy as np
print np.sqrt(metrics.mean_squared_error(y_test,y_pred))

# use RMSE for feature selection (Does remove features improve RMSE?)
feature_cols=['TV','Radio']
X=data[feature_cols]
y=data.Sales
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)
linreg.fit(X_train,y_train)
y_pred=linreg.predict(X_test)
print np.sqrt(metrics.mean_squared_error(y_test,y_pred))
