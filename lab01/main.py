import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names


print("Kształt macierzy X:", X.shape)
print("Liczba obserwacji:", X.shape[0])
print("Liczba cech:", X.shape[1])


print("\nKlasy:")
for name in iris.target_names:
    print(name)

print("\nNazwy cech:")
for feature in iris.feature_names:
    print(feature)

df = pd.DataFrame(X, columns=iris.feature_names)
print(df.describe())

mean_v = np.mean(X, axis=0)
print(f"Średnia każdej cechy:", mean_v)

x_min = X.min(axis = 0)
x_max = X.max(axis = 0)
x_centered = X - mean_v
#print(x_centered)

x_norm = (X - x_min) / (x_max-x_min)
