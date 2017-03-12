import numpy as np
import scipy as sp
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import time

np.set_printoptions(precision=10,
                    threshold = 10000,
                    suppress = True)

# Загружаем данные и удаляем наблюдения с пропущенными значениями
data = np.genfromtxt("cs-data.csv", delimiter = ',', 
                     skip_header = 1, usecols=list(range(1, 11)))
data = data[~np.isnan(data).any(axis = 1)]

# Выполняем метод главных компонент
data = scale(data)
pca = PCA(svd_solver='full')
pca.fit(data)

print("Размерность данных \n", data.shape, "\n")
# Вклад каждого фактора в объяснение вариации
print("Вклад каждого фактора в объяснение вариации \n", pca.explained_variance_ratio_, "\n") 
# Рост доли объясненной вариации с увеличением числа главных факторов
var = np.round(np.cumsum(pca.explained_variance_ratio_), decimals=4)
print("Рост доли объясненной вариации с увеличением числа главных факторов \n", var, "\n")
plt.plot(np.arange(1,11), var)
plt.ylabel('Variation')
plt.xlabel('Number of PC')

# Время выполнения
T = np.array([])
for i in range(0, 1000): 
    t = time.process_time()
    pca.fit(data)
    t = time.process_time() - t
    T = np.append(T, t)
print("Processor time for Scikit-learn:")
print("Min time: ", np.min(T))
print("Mean time: ", np.mean(T))
print("Median time: ", np.median(T))
print("Max time: ", np.max(T), "\n")

