from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

data = load_breast_cancer()

X = data.data
Y = data.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

#SVM
clf = SVC(kernel='linear', C=3) #linear/rbf/polynomial/...
clf.fit(x_train, y_train)
#Evaluation di SVM, decisamente maggiore di KNeighbors
print(clf.score(x_test, y_test))

#K Nearest Neighbors
clf2 = KNeighborsClassifier(n_neighbors=3)
clf2.fit(x_train, y_train)
#Evaluation di KNeighbors
print(clf2.score(x_test, y_test))