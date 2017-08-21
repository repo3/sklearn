# cross-validaiton example: parameter tuning
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
#from sklearn import metrics

iris=load_iris()
X=iris.data
y=iris.target
#X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=4)

knn=KNeighborsClassifier(n_neighbors=5)
#knn.fit(X_train,y_train)
#y_pred=knn.predict(X_test)
#print metrics.accuracy_score(y_test,y_pred)
scores=cross_val_score(knn,X,y,cv=10,scoring='accuracy')
print scores
print scores.mean()

#cross-validation example: finding best parameter
k_range=range(1,31)
k_scores=[]
for k in k_range:
  knn=KNeighborsClassifier(n_neighbors=k)
  scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy')
  k_scores.append(scores.mean())
print k_scores

plt.plot(k_range,k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
#plt.show()

# cross-validaiton example: model selection
knn=KNeighborsClassifier(n_neighbors=k)
print cross_val_score(knn,X,y,cv=10,scoring='accuracy').mean()

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
print cross_val_score(logreg,X,y,cv=10,scoring='accuracy').mean()

# cross-validation example: feature selection

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data=pd.read_csv('Advertising.csv')
feature_cols=['TV','Radio','Newspaper']
X=data[feature_cols]
y=data.Sales
lm=LinearRegression()
scores=cross_val_score(lm,X,y,cv=10,scoring='mean_squared_error')
print scores
# fix the sign of MSE scores
mse_scores = -scores
print mse_scores
# convert MSE to RMSE
rmse_scores=np.sqrt(mse_scores)
print rmse_scores
# calculate the average of RMSE
print rmse_scores.mean()
#10-fold cross-validation with two features (excluding Newspaper)
feature_cols=['TV','Radio']
X=data[feature_cols]
print np.sqrt(-cross_val_score(lm,X,y,cv=10,scoring='mean_squared_error')).mean()

