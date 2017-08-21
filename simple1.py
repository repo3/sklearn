# https://github.com/justmarkham/scikit-learn-videos
from sklearn.datasets import load_iris

iris = load_iris()
#print type(iris)
#print iris.data
#print iris.feature_names
#print iris.target
#print iris.target_names
print type(iris.data)
print type(iris.target)
print iris.data.shape
print iris.target.shape
X=iris.data
y=iris.target

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1)
#print knn
knn.fit(X,y)
print knn.predict([[3,5,4,2]])
#array([2])
X_new = [[3,5,4,2],[5,4,3,2]]
print knn.predict(X_new)
#array([2,1])

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X,y)
print logreg.predict(X_new)

#exit(0)
