from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import numpy as np
import time

from daal.algorithms.multinomial_naive_bayes import prediction, training
from daal.algorithms import classifier
from daal.data_management import ( CSRNumericTable, 
                                  HomogenNumericTable, 
                                  BlockDescriptor_Float64, 
                                  readOnly
                                  )

newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test')

vectorizer = CountVectorizer()
sparse_train = vectorizer.fit_transform(newsgroups_train.data)
sparse_test = vectorizer.transform(newsgroups_test.data)

sparse_train_nt = CSRNumericTable(sparse_train.data.astype(np.float64), 
                            sparse_train.indices.astype(np.uint64)+1, 
                            sparse_train.indptr.astype(np.uint64)+1, 
                            101631, 
                            11314)

sparse_test_nt =  CSRNumericTable(sparse_test.data.astype(np.float64), 
                            sparse_test.indices.astype(np.uint64)+1, 
                            sparse_test.indptr.astype(np.uint64)+1, 
                            101631, 
                            7532)

train_labels = newsgroups_train.target.astype(np.float64)
train_labels.shape = [11314,1]
train_labels_nt = HomogenNumericTable(train_labels)

nTrainObservations = 11314
nTestObservations = 7532
nClasses = 20

# Create an algorithm object to train the Naive Bayes model
algorithm_train = training.Batch_Float64FastCSR(nClasses)
# Pass a training data set and dependent values to the algorithm
algorithm_train.input.set(classifier.training.data,   sparse_train_nt)
algorithm_train.input.set(classifier.training.labels, train_labels_nt)
# Build the Naive Bayes model and retrieve the algorithm results
T_sparse_train = np.array([])
for i in range(0, 100): 
    t = time.process_time()
    trainingResult = algorithm_train.compute()
    t = time.process_time() - t
    T_sparse_train = np.append(T_sparse_train, t)
    del t
print("Median Training Time: ", np.median(T_sparse_train), " seconds\n")

# Create an algorithm object to predict Naive Bayes values
algorithm_test = prediction.Batch_Float64FastCSR(nClasses)
# Pass a testing data set and the trained model to the algorithm
algorithm_test.input.setTable(classifier.prediction.data,  sparse_test_nt)
algorithm_test.input.setModel(classifier.prediction.model, trainingResult.get(classifier.training.model))
# Predict Naive Bayes values and retrieve the algorithm results (Result class from classifier.prediction)
T_sparse_test = np.array([])
for i in range(0, 100): 
    t = time.process_time()
    predictionResult = algorithm_test.compute()
    t = time.process_time() - t
    T_sparse_test = np.append(T_sparse_test, t)
    del t
print("Median Prediction Time: ", np.median(T_sparse_test), " seconds\n")

def getArrayFromNT(table, nrows=0):
    bd = BlockDescriptor_Float64()
    if nrows == 0:
        nrows = table.getNumberOfRows()
    table.getBlockOfRows(0, nrows, readOnly, bd)
    npa = bd.getArray()
    table.releaseBlockOfRows(bd)
    return npa

test_pred = getArrayFromNT(predictionResult.get(classifier.prediction.prediction))
test_truth = newsgroups_test.target
print("Точность классификации – доля верно классифицированных объектов из тестовой выборки \n")
metrics.accuracy_score(test_pred, test_truth, normalize=True)




