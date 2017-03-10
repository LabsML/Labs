from daal.data_management import BlockDescriptor, readOnly
from daal.algorithms.ridge_regression import training, prediction
from daal.data_management import (
    DataSourceIface, FileDataSource, HomogenNumericTable,
    MergedNumericTable, NumericTableIface
)
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score,confusion_matrix
import time
from __init__ import printNumericTable

trainDatasetFileName = '1988_14col_train_data.csv'
testDatasetFileName = '1988_14col_test_data.csv'

nTrainVectorsInBlock = 100
nFeatures = 493
nDependentVariables = 1
batchIndex = 0
onlineIndex = 1

trainingResult = None
predictionResult = None

def getClassVector(variables, threshold):
    classVector = np.zeros((len(variables)), dtype=np.int)
    for i in range(len(variables)):
        if (variables[i] > threshold):
            classVector[i] = 1
    return classVector
def R2(dependentVariables, predictedVariables):
    return r2_score(dependentVariables, predictedVariables)
def RMSE(dependentVariables, predictedVariables):
    return np.sqrt(mean_squared_error(dependentVariables, predictedVariables))
def classification_accuracy(dependentVariables, predictedVariables):
    return accuracy_score(dependentVariables, predictedVariables)
def trainModelOnline():
    global trainingResult

    trainDataSource = FileDataSource(
        trainDatasetFileName, DataSourceIface.notAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    trainData = HomogenNumericTable(nFeatures, 0, NumericTableIface.notAllocate)
    trainDependentVariables = HomogenNumericTable(
        nDependentVariables, 0, NumericTableIface.notAllocate
    )
    mergedData = MergedNumericTable(trainData, trainDependentVariables)

    algorithm = training.Online()

    readBlockNumber = trainDataSource.loadDataBlock(nTrainVectorsInBlock, mergedData)
    while(readBlockNumber > 0):
        algorithm.input.set(training.data, trainData)
        algorithm.input.set(training.dependentVariables, trainDependentVariables)
        algorithm.compute()
        readBlockNumber = trainDataSource.loadDataBlock(nTrainVectorsInBlock, mergedData)

    trainingResult = algorithm.finalizeCompute()

def testModelOnline():
    global trainingResult, predictionResult

    testDataSource = FileDataSource(
        testDatasetFileName, DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    testData = HomogenNumericTable(nFeatures, 0, NumericTableIface.notAllocate)
    testGroundTruth = HomogenNumericTable(nDependentVariables, 0, NumericTableIface.notAllocate)
    mergedData = MergedNumericTable(testData, testGroundTruth)

    nTrainVectorsInBlock = 50000

    y_true = np.array([])
    predictionRegression = np.array([])
    readBlockNumber = testDataSource.loadDataBlock(nTrainVectorsInBlock, mergedData)
    while (readBlockNumber > 0):
        algorithm = prediction.Batch()
        algorithm.input.setModelInput(prediction.model, trainingResult.get(training.model))
        algorithm.input.setNumericTableInput(prediction.data, testData)
        predictionResult = algorithm.compute()
        block1 = BlockDescriptor()
        block2 = BlockDescriptor()
        testGroundTruth.getBlockOfRows(0, testGroundTruth.getNumberOfRows(), readOnly, block1)
        predictionResult.get(prediction.prediction).getBlockOfRows(0, testGroundTruth.getNumberOfRows(), readOnly,
                                                                   block2)
        y_true = np.append(y_true, getClassVector(block1.getArray(), 0.000000000000))
        predictionRegression = np.append(predictionRegression, block2.getArray())
        readBlockNumber = testDataSource.loadDataBlock(nTrainVectorsInBlock, mergedData)

    best_threshold = None
    best_precision = -1
    for threshold in np.linspace(-25.0, 25.0, num=101):
        y_pred = getClassVector(predictionRegression, threshold)
        cur_precision = accuracy_score(y_true, y_pred)
        if (cur_precision > best_precision):
            best_threshold = threshold
            best_precision = cur_precision

    y_pred = getClassVector(predictionRegression, best_threshold)
    print('Test set. Number of objects of 0 class:{:.4f}.Number of objects of 1 class:{:.4f}. Frequency of 1 class:{:.4f}'.format(
            len(y_true) - np.count_nonzero(y_true), np.count_nonzero(y_true), np.count_nonzero(y_true) / len(y_true)))
    print('Best threshold:{:.4f}.'.format(best_threshold),end='')
    print('Classification precision:{:.4f}'.format(classification_accuracy(y_true, y_pred)))
    print(confusion_matrix(y_true, y_pred))

    best_threshold = 0.0000
    print('Threshold:{:.4f}.'.format(best_threshold), end='')
    y_pred = getClassVector(predictionRegression, best_threshold)
    print('Classification precision:{:.4f}'.format(
        classification_accuracy(y_true, y_pred)))
    print(confusion_matrix(y_true, y_pred))

def trainModelBatch():
    global trainingResult

    trainDataSource = FileDataSource(
        trainDatasetFileName, DataSourceIface.notAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )
    trainData = HomogenNumericTable(nFeatures, 0, NumericTableIface.notAllocate)
    trainDependentVariables = HomogenNumericTable(nDependentVariables, 0, NumericTableIface.notAllocate)
    mergedData = MergedNumericTable(trainData, trainDependentVariables)
    trainDataSource.loadDataBlock(mergedData)

    algorithm = training.Batch_Float64NormEqDense()
    algorithm.input.set(training.data, trainData)
    algorithm.input.set(training.dependentVariables, trainDependentVariables)

    # Build the multiple linear regression model and retrieve the algorithm results
    trainingResult = algorithm.compute()

def testModelBatch():
    global trainingResult, predictionResult

    testDataSource = FileDataSource(
        testDatasetFileName, DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )
    testData = HomogenNumericTable(nFeatures, 0, NumericTableIface.notAllocate)
    testGroundTruth = HomogenNumericTable(nDependentVariables, 0, NumericTableIface.notAllocate)
    mergedData = MergedNumericTable(testData, testGroundTruth)
    testDataSource.loadDataBlock(mergedData)

    algorithm = prediction.Batch()
    algorithm.input.setTable(prediction.data, testData)
    algorithm.input.setModel(prediction.model, trainingResult.get(training.model))

    predictionResult = algorithm.compute()

def predictResults(metodIndex):
    if metodIndex == batchIndex:
        return testModelBatch()
    else:
        return testModelOnline()

def trainModel(metodIndex):
    if metodIndex == batchIndex:
        return trainModelBatch()
    else:
        return trainModelOnline()

def execute(linearRegressionModelIndex):
    if linearRegressionModelIndex == batchIndex:
        print('\nExecution of Batch_Float64NormEqDense() function.')
    else:
        print('\nExecution of Online_Float64NormEqDense() function. nTrainVectorsInBlock=%s.' % (nTrainVectorsInBlock))

    start = time.time()
    for num in range(1):
        trainingResult = trainModel(linearRegressionModelIndex)
    end = time.time()
    print('Performance comparison. Time training: %s seconds' % (end - start))
    predictResults(linearRegressionModelIndex)

for vectorsInBlock in range (100000, 100100, 200000):
    trainingResult = None
    predictionResult = None
    nTrainVectorsInBlock = vectorsInBlock
    execute(onlineIndex)
execute(batchIndex)