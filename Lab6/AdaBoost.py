import numpy as np
from sklearn import ensemble
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

import pandas as pd
import time

#There should be names of files, which contain train and test parts of the dataset
train = np.genfromtxt("", delimiter=',')
test = np.genfromtxt("", delimiter=',')

n = train.shape[1] - 1

classifier = ensemble.AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=100)

t3 = time.process_time()
classifier.fit(train[:,0:n],train[:,n])
print("Processor time = ", time.process_time() - t3)

prediction = classifier.predict(test[:,0:n])
tab = pd.crosstab(index = prediction, columns= test[:,n])

print(tab)

