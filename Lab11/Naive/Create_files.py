from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os
import math

newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test')

vectorizer = CountVectorizer()
sparse_train = vectorizer.fit_transform(newsgroups_train.data)
sparse_test = vectorizer.transform(newsgroups_test.data)

numberOfFiles = 100
print("Number of files: %d." % (numberOfFiles))
numberOfObjects = math.ceil(sparse_train.shape[0]/numberOfFiles)
print("Number of objects: %d." % (numberOfObjects))
print(sparse_train.shape)
print(sparse_test.shape)
"""
dense_train = np.append(sparse_train.toarray(), newsgroups_train.target[np.newaxis].T, axis=1)
dense_test = sparse_test.toarray()

for i in range(numberOfFiles):
    filename =  "news_train_dense_dist_data_" + str(i+1) + ".csv"
    filenameTrain = os.path.join(os.getcwd(), filename)
    print("Write to %s." % (filenameTrain))
    firstIndex = i*numberOfObjects
    lastIndex = min(dense_train.shape[0],(i+1)*numberOfObjects)
    np.savetxt(filenameTrain,
               dense_train[firstIndex:lastIndex, ],
               delimiter=",", fmt='%.1d')

    filename = "news_train_dense_label_" + str(i + 1) + ".csv"
#    filenameTrainLabel = os.path.join(os.getcwd(), filename)
#    print("Write to %s." % (filenameTrainLabel))
#    np.savetxt(filenameTrainLabel,
#               dense_train_label[firstIndex:lastIndex, ],
#               delimiter=",", fmt='%.1d')
# Test data
np.savetxt("news_test_dense_dist_data.csv",
           dense_test,
           delimiter=",", fmt='%.1d')
np.savetxt("news_test_dense_dist_label.csv",
           newsgroups_test.target[np.newaxis].T,
           delimiter=",", fmt='%.1d')
"""