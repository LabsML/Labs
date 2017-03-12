from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import numpy as np
import time

from daal.algorithms.multinomial_naive_bayes import prediction, training
from daal.algorithms import classifier
from daal.data_management import (
    FileDataSource, HomogenNumericTable,
    MergedNumericTable, DataSourceIface, NumericTableIface,
    BlockDescriptor_Float64, readOnly
)

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

np.savetxt("news_train_dense.csv", 
           np.append(dense_train, newsgroups_train.target[np.newaxis].T, axis=1), 
           delimiter=",", fmt='%.1d')
np.savetxt("news_test_dense.csv", 
           np.append(dense_test, newsgroups_test.target[np.newaxis].T, axis=1), 
           delimiter=",", fmt='%.1d')


# Input data set parameters
trainDatasetFileName = "news_train_dense.csv"
testDatasetFileName = "news_test_dense.csv"

nFeatures = 101631
nClasses = 20


## Training
# Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
trainDataSource = FileDataSource(
    trainDatasetFileName, DataSourceIface.notAllocateNumericTable,
    DataSourceIface.doDictionaryFromContext
)

# Create Numeric Tables for training data and labels
trainData = HomogenNumericTable(nFeatures, 0, NumericTableIface.notAllocate)
trainGroundTruth = HomogenNumericTable(1, 0, NumericTableIface.notAllocate)
mergedData = MergedNumericTable(trainData, trainGroundTruth)

# Retrieve the data from the input file
t = time.process_time()
trainDataSource.loadDataBlock(mergedData)
t = time.process_time() - t 
print("Load time: ", t, "\n") 

# Create an algorithm object to train the Naive Bayes model
algorithm = training.Batch(nClasses)

# Pass a training data set and dependent values to the algorithm
algorithm.input.set(classifier.training.data,   trainData)
algorithm.input.set(classifier.training.labels, trainGroundTruth)

# Build the Naive Bayes model and retrieve the algorithm results
T_dense_train = np.array([])
for i in range(0, 100): 
    t = time.process_time()
    trainingResult = algorithm.compute()
    t = time.process_time() - t
    T_dense_train = np.append(T_dense_train, t)
    del t
print("Median Training Time: ", np.median(T_dense_train), " seconds\n")


## Testing
# Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file
testDataSource = FileDataSource(
    testDatasetFileName, DataSourceIface.notAllocateNumericTable,
    DataSourceIface.doDictionaryFromContext
)

# Create Numeric Tables for testing data and labels
testData = HomogenNumericTable(nFeatures, 0, NumericTableIface.notAllocate)
testGroundTruth = HomogenNumericTable(1, 0, NumericTableIface.notAllocate)
mergedData = MergedNumericTable(testData, testGroundTruth)

# Retrieve the data from input file
testDataSource.loadDataBlock(mergedData)

# Create an algorithm object to predict Naive Bayes values
algorithm = prediction.Batch(nClasses)

# Pass a testing data set and the trained model to the algorithm
algorithm.input.setTable(classifier.prediction.data,  testData)
algorithm.input.setModel(classifier.prediction.model, trainingResult.get(classifier.training.model))

# Predict Naive Bayes values (Result class from classifier.prediction)
T_dense_test = np.array([])
for i in range(0, 100): 
    t = time.process_time()
    predictionResult = algorithm.compute()  # Retrieve the algorithm results
    t = time.process_time() - t
    T_dense_test = np.append(T_dense_test, t)
    del t   
print("Median Prediction Time: ", np.median(T_denrse_test), " seconds\n")

def getArrayFromNT(table, nrows=0):
    bd = BlockDescriptor_Float64()
    if nrows == 0:
        nrows = table.getNumberOfRows()
    table.getBlockOfRows(0, nrows, readOnly, bd)
    npa = bd.getArray()
    table.releaseBlockOfRows(bd)
    return npa

test_pred = getArrayFromNT(predictionResult.get(classifier.prediction.prediction))
test_truth = getArrayFromNT(testGroundTruth)
print("Точность классификации – доля верно классифицированных объектов из тестовой выборки \n")
metrics.accuracy_score(test_pred, test_truth, normalize=True)
