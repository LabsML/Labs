import numpy as np
from daal.data_management import HomogenNumericTable, BlockDescriptor, readOnly
from daal.algorithms.linear_regression import training, prediction
import pandas as pd
from sklearn.metrics import mean_squared_error
import time

def RMSE(dependentVariables, predictedVariables):
    block1 = BlockDescriptor()
    block2 = BlockDescriptor()
    dependentVariables.getBlockOfRows(0, dependentVariables.getNumberOfRows(), readOnly, block1)
    predictedVariables.getBlockOfRows(0, dependentVariables.getNumberOfRows(), readOnly, block2)
    return np.sqrt(mean_squared_error(block1.getArray(), block2.getArray()))
def predictResults(data, model):
    algorithm = prediction.Batch()
    algorithm.input.setTable(prediction.data, data)
    algorithm.input.setModel(prediction.model, model)
    return algorithm.compute()
def trainModel(trainData, trainDependentVariables,metodIndex):
    if (metodIndex == 0):
        algorithm = training.Batch_Float64NormEqDense()
    else:
        algorithm = training.Batch_Float64QrDense()
    algorithm.input.set(training.data, trainData)
    algorithm.input.set(training.dependentVariables, trainDependentVariables)
    return algorithm.compute()
train_data=np.genfromtxt('kc_house_train_data.csv', delimiter=',')
test_data=np.genfromtxt("kc_house_test_data.csv", delimiter=',')
nFeatures = train_data.shape[1] - 1
trainX = train_data[:,1:(nFeatures+1)]
testX = test_data[:,1:(nFeatures+1)]

trainY = train_data[:,0:1].copy()
testY = test_data[:,0:1].copy()


def execute(linearRegressionModelIndex):
    if (linearRegressionModelIndex == 0):
        print('\nExecution of Batch_Float64NormEqDense() function:')
    else:
        print('\nExecution of Batch_Float64QrDense() function:')
    remainingIndexes = [i for i in range(0, nFeatures)]
    start = time.time()
    trainDependentVariables = HomogenNumericTable(trainY)
    testDependentVariables = HomogenNumericTable(testY)

    trainDataNumTable = HomogenNumericTable(trainX.copy())
    testDataNumTable = HomogenNumericTable(testX.copy())

    start = time.time()
    for num in range(1000):
        trainingResult = trainModel(trainDataNumTable, trainDependentVariables, linearRegressionModelIndex)
        model = trainingResult.get(training.model)
    end = time.time()

    print('Performance comparison. Time: %s seconds' % (end - start))

    predictionResult = predictResults(testDataNumTable, model)
    predicted = predictionResult.get(prediction.prediction)
    print('Linear regression. Test error: {:.2f}'.format(
        RMSE(testDependentVariables, predicted)))

for i in range(0, 2):
    execute(i)