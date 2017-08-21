from sklearn.datasets import load_iris

iris=load_iris()
X=iris.data
y=iris.target

from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()
logreg.fit(X,y)
y_pred=logreg.predict(X)
#print y_pred

from sklearn import metrics

print metrics.accuracy_score(y,y_pred)
# 0.96

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
y_pred=knn.predict(X)
print metrics.accuracy_score(y,y_pred)
# 0.9666666

#from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X,y)
y_pred=knn.predict(X)
print metrics.accuracy_score(y,y_pred)
# 1.0

#train test split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=4)
print X_train.shape
print X_test.shape
print y_train.shape
print y_test.shape
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
print metrics.accuracy_score(y_test,y_pred)
# 0.95

#from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print metrics.accuracy_score(y_test,y_pred)

#from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print metrics.accuracy_score(y_test,y_pred)


#search for best k of k-nearest nieghbors
k_range = range(1,26)
scores=[]
for k in k_range:
  knn=KNeighborsClassifier(n_neighbors=k)
  knn.fit(X_train,y_train)
  y_pred=knn.predict(X_test)
  scores.append( metrics.accuracy_score(y_test,y_pred) )
#print scores

import matplotlib.pyplot as plt

plt.plot(k_range,scores,'bo-')
plt.xlabel('Value of k for KNN')
plt.ylabel('Level of Accuracy %')
plt.show()

knn=KNeighborsClassifier(n_neighbors=11)
knn.fit(X,y)
print knn.predict([[3,5,4,2]])

