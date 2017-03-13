import io
import numpy as np
from sklearn import svm
import scipy as sp
import pandas as pd
import time

train = np.genfromtxt("", delimiter=',') #There should be names of files, which contain train and test parts of dataset
test = np.genfromtxt("", delimiter=',')
n = 102 #number of features (class labels not included)

classifier = svm.SVC(kernel='rbf', max_iter=200000, cache_size=8000)
t1 = time.time()
t2 = time.perf_counter()
t3 = time.process_time()

classifier.fit(train[:,0:n],train[:,n])

prediction = classifier.predict(test[:,0:n])

print("Total time = ", time.time() - t1)
print("Performance time = ", time.perf_counter() - t2)
print("Processor time = ", time.process_time() - t3)

print(np.sum(prediction))

tab = pd.crosstab(index=prediction, columns=test[:,n]) #confusion matrix

print(tab)
