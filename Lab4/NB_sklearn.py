from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import time

newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test')

print("Число наблюдений в обучающей выборке\n", 
      newsgroups_train.filenames.shape)
print("Число наблюдений в тестовой выборке\n", 
      newsgroups_test.filenames.shape)

print("Список новостных рубрик\n")
pprint(list(newsgroups_train.target_names))
pprint(list(newsgroups_test.target_names))

vectorizer = CountVectorizer()
sparse_train = vectorizer.fit_transform(newsgroups_train.data)
sparse_test = vectorizer.transform(newsgroups_test.data)
print("Размерность обучающей выборки sparse\n", sparse_train.shape)
print("Размерность тестовой выборки sparse\n", sparse_test.shape)

dense_train = sparse_train.toarray()
dense_test = sparse_test.toarray()
print("Размерность обучающей выборки dense\n", dense_train.shape)
print("Размерность тестовой выборки dense\n", dense_test.shape)


clf = MultinomialNB(alpha= 1)
T_sparse_train = np.array([])
T_sparse_test = np.array([])
for i in range(0, 100): 
    t = time.process_time()
    clf.fit(sparse_train, newsgroups_train.target)
    t = time.process_time() - t
    T_sparse_train = np.append(T_sparse_train, t)
    del t
    t = time.process_time()
    pred = clf.predict(sparse_test)
    t = time.process_time() - t
    T_sparse_test = np.append(T_sparse_test, t)
    del t
print("Processor time for Sparse Training:")
print("Min time: ", np.min(T_sparse_train))
print("Mean time: ", np.mean(T_sparse_train))
print("Median time: ", np.median(T_sparse_train))
print("Max time: ", np.max(T_sparse_train), "\n")
print("\n")
print("Processor time for Sparse Predict:")
print("Min time: ", np.min(T_sparse_test))
print("Mean time: ", np.mean(T_sparse_test))
print("Median time: ", np.median(T_sparse_test))
print("Max time: ", np.max(T_sparse_test), "\n")

print("Точность классификации\n")
metrics.accuracy_score(newsgroups_test.target, pred, normalize=True)
a = metrics.confusion_matrix(newsgroups_test.target, pred)

clf = MultinomialNB(alpha=1)
T_dense_train = np.array([])
T_dense_test = np.array([])
for i in range(0, 100): 
    t = time.process_time()
    clf.fit(dense_train, newsgroups_train.target)
    t = time.process_time() - t
    T_dense_train = np.append(T_dense_train, t)
    del t
    t = time.process_time()
    pred = clf.predict(dense_test)
    t = time.process_time() - t
    T_dense_test = np.append(T_dense_test, t)
    del t
print("Processor time for Dense Training:")
print("Min time: ", np.min(T_dense_train))
print("Mean time: ", np.mean(T_dense_train))
print("Median time: ", np.median(T_dense_train))
print("Max time: ", np.max(T_dense_train), "\n")
print("\n")
print("Processor time for Dense Predict:")
print("Min time: ", np.min(T_dense_test))
print("Mean time: ", np.mean(T_dense_test))
print("Median time: ", np.median(T_dense_test))
print("Max time: ", np.max(T_dense_test), "\n")

print("Точность классификации\n")
metrics.accuracy_score(newsgroups_test.target, pred, normalize=True)

