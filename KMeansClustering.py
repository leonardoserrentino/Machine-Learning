from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits

digits = load_digits()
data = scale(digits.data)

model = KMeans(n_clusters=10, init='random', n_init=10) #numero di classi e' 10 in questo caso perche' sono 10 numeri 0->9
model.fit(data)

#model.predict([...]) nelle quadre metti lo scan dei pixel di un numero da leggere
