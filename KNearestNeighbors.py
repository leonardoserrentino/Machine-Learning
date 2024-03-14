"""
Questo perogramma permette di classificare in modo 
supervised i tumori al seno come benigni o maligni
"""

from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import numpy as np

#Caricamento e visualizzazione del dataset
"""
data = load_breast_cancer()
print(data.feature_names)
"""

#Split
x_train, x_test, y_train, y_test = train_test_split(np.array(data.data), np.array(data.target), test_size=0.2)

#Training
clf = KNeighborsClassifier(n_neighbors=3) #numero di neighbors
clf.fit(x_train, y_train)

#Evaluation of testing
print(clf.score(x_test, y_test))

#Predict the value for not seen x's
#clf.predict('array di x')