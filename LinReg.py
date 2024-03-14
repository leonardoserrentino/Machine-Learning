"""
Questo programma fa una regressione lineare prendendo
in input x e y, con gli eventuali reshape eccetera

Nella seconda parte come fare uno split del dataset in 
train e test set per la verifica dell'accuratezza
"""

#Per entrambi
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#Per lo split
from sklearn.model_selection import train_test_split

#Dati in input su cui fare la regressione
time = np.array([20, 50, 32, 65, 23, 43, 10, 5, 22, 35, 29, 5, 56]).reshape(-1,1) #come sklearn li vuole
score = np.array([56, 83, 47, 93, 47, 82, 45, 23, 55, 67, 57, 4, 89]).reshape(-1,1)

#Regressione Lineare pura senza split
model = LinearRegression()
model.fit(time, score)

plt.scatter(time, score)
plt.plot(np.linspace(0,70,100).reshape(-1,1), model.predict(np.linspace(0,70,100).reshape(-1,1)), 'r')
plt.ylim(0, 100)
plt.show()


#Split
time_train, time_test, score_train, score_test = train_test_split(time, score, test_size=0.2)

#Model Training
splitModel = LinearRegression()
splitModel.fit(time_train, score_train)

#Model Testing
print(splitModel.score(time_test, score_test))

