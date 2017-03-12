from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import numpy as np
import time

from daal.algorithms.multinomial_naive_bayes import prediction, training
from daal.algorithms import classifier
from daal.data_management import (
    FileDataSource, DataSourceIface,
    HomogenNumericTable, MergedNumericTable, NumericTableIface,
    BlockDescriptor_Float64, readOnly
)


# Create dataset
newsgroups_train = fetch_20newsgroups(subset='train',
                                      remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test')

vectorizer = CountVectorizer()
sparse_train = vectorizer.fit_transform(newsgroups_train.data)
sparse_test = vectorizer.transform(newsgroups_test.data)
dense_train = sparse_train.toarray()
dense_test = sparse_test.toarray()

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

#############################################################################

def getArrayFromNT(table, nrows=0):
    bd = BlockDescriptor_Float64()
    if nrows == 0:
        nrows = table.getNumberOfRows()
    table.getBlockOfRows(0, nrows, readOnly, bd)
    npa = bd.getArray()
    table.releaseBlockOfRows(bd)
    return npa

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
algorithm_test = prediction.Batch(nClasses)

# Pass a testing data set and the trained model to the algorithm
algorithm_test.input.setTable(classifier.prediction.data,  testData)

n_experiments = 1

t_train_100 = np.zeros(n_experiments)
t_train_500 = np.zeros(n_experiments)
t_train_1000 = np.zeros(n_experiments)
t_train_2000 = np.zeros(n_experiments)
t_train_5000 =np.zeros(n_experiments)

test_acc_100 = np.array([])
test_acc_500 = np.array([])
test_acc_1000 = np.array([])
test_acc_2000 = np.array([])
test_acc_5000 = np.array([])

#############################################################################
for i in range(0, n_experiments):
    
    nTrainVectorsInBlock = 100
    
    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    trainDataSource = FileDataSource(
        trainDatasetFileName, DataSourceIface.notAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )
    
    # Create Numeric Tables for training data and labels
    trainData = HomogenNumericTable(nFeatures, 0, NumericTableIface.notAllocate)
    trainGroundTruth = HomogenNumericTable(1, 0, NumericTableIface.notAllocate)
    mergedData = MergedNumericTable(trainData, trainGroundTruth)
    
    # Create an algorithm object to train the Naive Bayes model
    algorithm_100 = training.Online(nClasses)
    
    while(trainDataSource.loadDataBlock(nTrainVectorsInBlock, mergedData) > 0):
        # Pass a training data set and dependent values to the algorithm
        algorithm_100.input.set(classifier.training.data,   trainData)
        algorithm_100.input.set(classifier.training.labels, trainGroundTruth)
    
        # Build the Naive Bayes model
        t = time.process_time()
        algorithm_100.compute()
        t = time.process_time() - t
        t_train_100[i] = t_train_100[i] + t
       
    # Finalize the Naive Bayes model
    trainingResult_100 = algorithm_100.finalizeCompute()  # Retrieve the algorithm results
    
# Prediction accuracy
algorithm_test.input.setModel(classifier.prediction.model, trainingResult_100.get(classifier.training.model))
predictionResult = algorithm_test.compute()  # Retrieve the algorithm results
test_acc_100 = np.append(test_acc_100, 
                         metrics.accuracy_score(getArrayFromNT(predictionResult.get(classifier.prediction.prediction)), 
                                                               getArrayFromNT(testGroundTruth), 
                                                                normalize=True))
    
#############################################################################
for i in range(0, n_experiments):
    
    nTrainVectorsInBlock = 500
    
    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    trainDataSource = FileDataSource(
        trainDatasetFileName, DataSourceIface.notAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )
    
    # Create Numeric Tables for training data and labels
    trainData = HomogenNumericTable(nFeatures, 0, NumericTableIface.notAllocate)
    trainGroundTruth = HomogenNumericTable(1, 0, NumericTableIface.notAllocate)
    mergedData = MergedNumericTable(trainData, trainGroundTruth)
    
    # Create an algorithm object to train the Naive Bayes model
    algorithm_500 = training.Online(nClasses)
    
    while(trainDataSource.loadDataBlock(nTrainVectorsInBlock, mergedData) > 0):
        # Pass a training data set and dependent values to the algorithm
        algorithm_500.input.set(classifier.training.data,   trainData)
        algorithm_500.input.set(classifier.training.labels, trainGroundTruth)
    
        # Build the Naive Bayes model
        t = time.process_time()
        algorithm_500.compute()
        t = time.process_time() - t
        t_train_500[i] = t_train_500[i] + t
    
    
    # Finalize the Naive Bayes model
    trainingResult_500 = algorithm_500.finalizeCompute()  # Retrieve the algorithm results
    
    ## Prediction accuracy
algorithm_test.input.setModel(classifier.prediction.model, trainingResult_500.get(classifier.training.model))
predictionResult = algorithm_test.compute()  # Retrieve the algorithm results
test_acc_500 = np.append(test_acc_500, 
                         metrics.accuracy_score(getArrayFromNT(predictionResult.get(classifier.prediction.prediction)), 
                                                               getArrayFromNT(testGroundTruth), 
                                                                normalize=True))
    
#############################################################################
for i in range(0, n_experiments):
    
    nTrainVectorsInBlock = 1000
    
    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    trainDataSource = FileDataSource(
        trainDatasetFileName, DataSourceIface.notAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )
    
    # Create Numeric Tables for training data and labels
    trainData = HomogenNumericTable(nFeatures, 0, NumericTableIface.notAllocate)
    trainGroundTruth = HomogenNumericTable(1, 0, NumericTableIface.notAllocate)
    mergedData = MergedNumericTable(trainData, trainGroundTruth)
    
    # Create an algorithm object to train the Naive Bayes model
    algorithm_1000 = training.Online(nClasses)
    
    while(trainDataSource.loadDataBlock(nTrainVectorsInBlock, mergedData) > 0):
        # Pass a training data set and dependent values to the algorithm
        algorithm_1000.input.set(classifier.training.data,   trainData)
        algorithm_1000.input.set(classifier.training.labels, trainGroundTruth)
    
        # Build the Naive Bayes model
        t = time.process_time()
        algorithm_1000.compute()
        t = time.process_time() - t
        t_train_1000[i] = t_train_1000[i] + t
    
    
    # Finalize the Naive Bayes model
    trainingResult_1000 = algorithm_1000.finalizeCompute()  # Retrieve the algorithm results
    
    ## Prediction accuracy
algorithm_test.input.setModel(classifier.prediction.model, trainingResult_1000.get(classifier.training.model))
predictionResult = algorithm_test.compute()  # Retrieve the algorithm results
test_acc_1000 = np.append(test_acc_1000, 
                         metrics.accuracy_score(getArrayFromNT(predictionResult.get(classifier.prediction.prediction)), 
                                                               getArrayFromNT(testGroundTruth), 
                                                                normalize=True))
    
#############################################################################
for i in range(0, n_experiments):
    
    nTrainVectorsInBlock = 2000
    
    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    trainDataSource = FileDataSource(
        trainDatasetFileName, DataSourceIface.notAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )
    
    # Create Numeric Tables for training data and labels
    trainData = HomogenNumericTable(nFeatures, 0, NumericTableIface.notAllocate)
    trainGroundTruth = HomogenNumericTable(1, 0, NumericTableIface.notAllocate)
    mergedData = MergedNumericTable(trainData, trainGroundTruth)
    
    # Create an algorithm object to train the Naive Bayes model
    algorithm_2000 = training.Online(nClasses)
    
    while(trainDataSource.loadDataBlock(nTrainVectorsInBlock, mergedData) > 0):
        # Pass a training data set and dependent values to the algorithm
        algorithm_2000.input.set(classifier.training.data,   trainData)
        algorithm_2000.input.set(classifier.training.labels, trainGroundTruth)
    
        # Build the Naive Bayes model
        t = time.process_time()
        algorithm_2000.compute()
        t = time.process_time() - t
        t_train_2000[i] = t_train_2000[i] + t
    
    
    # Finalize the Naive Bayes model
    trainingResult_2000 = algorithm_2000.finalizeCompute()  # Retrieve the algorithm results
    
    ## Prediction accuracy
algorithm_test.input.setModel(classifier.prediction.model, trainingResult_2000.get(classifier.training.model))
predictionResult = algorithm_test.compute()  # Retrieve the algorithm results
test_acc_2000 = np.append(test_acc_2000, 
                         metrics.accuracy_score(getArrayFromNT(predictionResult.get(classifier.prediction.prediction)), 
                                                               getArrayFromNT(testGroundTruth), 
                                                                normalize=True))
    
#############################################################################
for i in range(0, n_experiments):
    
    nTrainVectorsInBlock = 5000
    
    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    trainDataSource = FileDataSource(
        trainDatasetFileName, DataSourceIface.notAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )
    
    # Create Numeric Tables for training data and labels
    trainData = HomogenNumericTable(nFeatures, 0, NumericTableIface.notAllocate)
    trainGroundTruth = HomogenNumericTable(1, 0, NumericTableIface.notAllocate)
    mergedData = MergedNumericTable(trainData, trainGroundTruth)
    
    # Create an algorithm object to train the Naive Bayes model
    algorithm_5000 = training.Online(nClasses)
    
    while(trainDataSource.loadDataBlock(nTrainVectorsInBlock, mergedData) > 0):
        # Pass a training data set and dependent values to the algorithm
        algorithm_5000.input.set(classifier.training.data,   trainData)
        algorithm_5000.input.set(classifier.training.labels, trainGroundTruth)
    
        # Build the Naive Bayes model
        t = time.process_time()
        algorithm_5000.compute()
        t = time.process_time() - t
        t_train_5000[i] = t_train_5000[i] + t
    
    
    # Finalize the Naive Bayes model
    trainingResult_5000 = algorithm_5000.finalizeCompute()  # Retrieve the algorithm results
    
    ## Prediction accuracy
algorithm_test.input.setModel(classifier.prediction.model, trainingResult_5000.get(classifier.training.model))
predictionResult = algorithm_test.compute()  # Retrieve the algorithm results
test_acc_5000 = np.append(test_acc_5000, 
                         metrics.accuracy_score(getArrayFromNT(predictionResult.get(classifier.prediction.prediction)), 
                                                               getArrayFromNT(testGroundTruth), 
                                                                normalize=True))
    

##############################################################################
##############################################################################






