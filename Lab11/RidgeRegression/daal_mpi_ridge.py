from os.path import join as jp, realpath, abspath

from mpi4py import MPI

import numpy as np
from sklearn.metrics import accuracy_score
import daal.algorithms.ridge_regression.prediction as prediction
import daal.algorithms.ridge_regression.training as training
from daal.data_management import (
    DataSourceIface, FileDataSource, OutputDataArchive, InputDataArchive,
    HomogenNumericTable, MergedNumericTable, NumericTableIface, BlockDescriptor, readOnly
)
import os
import fnmatch
from __init__ import printNumericTable

nFeatures = 541
nDependentVariables = 1

trainingResult = None
predictionResult = None

MPI_ROOT = 0

datasetFolder = os.path.join(os.getcwd())
trainDatasetFileNames = []
testDatasetFileNames = []
def getDatasetFileNames(filematching):
    matches = []
    for root, dirnames, filenames in os.walk(datasetFolder):
        for filename in fnmatch.filter(filenames, filematching):
            matches.append(os.path.join(root, filename))
    return matches
def getClassVector(variables, threshold):
    classVector = np.zeros((len(variables)), dtype=np.int)
    for i in range(len(variables)):
        if (variables[i] > threshold):
            classVector[i] = 1
    return classVector

def trainModel():
    global trainingResult
    masterAlgorithm = training.Distributed_Step2MasterFloat64NormEqDense()

    for filenameIndex in range(rankId, len(trainDatasetFileNames), comm_size):
        trainDataSource = FileDataSource(trainDatasetFileNames[filenameIndex],
                                         DataSourceIface.notAllocateNumericTable,
                                         DataSourceIface.doDictionaryFromContext)
        trainData = HomogenNumericTable(nFeatures, 0, NumericTableIface.notAllocate)
        trainDependentVariables = HomogenNumericTable(nDependentVariables, 0, NumericTableIface.notAllocate)
        mergedData = MergedNumericTable(trainData, trainDependentVariables)
        trainDataSource.loadDataBlock(mergedData)

        localAlgorithm = training.Distributed_Step1LocalFloat64NormEqDense()
        localAlgorithm.input.set(training.data, trainData)
        localAlgorithm.input.set(training.dependentVariables, trainDependentVariables)
        pres = localAlgorithm.compute()
        masterAlgorithm.input.add(training.partialModels, pres)

        mergedData.freeDataMemory()
        trainData.freeDataMemory()
        trainDependentVariables.freeDataMemory()

    pres = masterAlgorithm.compute()
    dataArch = InputDataArchive()
    pres.serialize(dataArch)
    nodeResults = dataArch.getArchiveAsArray()
    serializedData = comm.gather(nodeResults)

    if rankId == MPI_ROOT:
        print("Number of processes is %d." % (len(serializedData)))
        masterAlgorithm = training.Distributed_Step2MasterFloat64NormEqDense()

        for i in range(comm_size):
            dataArch = OutputDataArchive(serializedData[i])
            dataForStep2FromStep1 = training.PartialResult()
            dataForStep2FromStep1.deserialize(dataArch)
            masterAlgorithm.input.add(training.partialModels, dataForStep2FromStep1)
        masterAlgorithm.compute()
        trainingResult = masterAlgorithm.finalizeCompute()

def testModel():
    thresholdValues = np.linspace(-25.0, 25.0, num=101)
    numberOfCorrectlyClassifiedObjects = np.zeros(len(thresholdValues))
    numberOfObjectsInTestFiles = 0
    numberOfNonzeroObjectsInTestFiles = 0
    for filenameIndex in range(0, len(testDatasetFileNames)):
        testDataSource = FileDataSource(testDatasetFileNames[filenameIndex],
                                    DataSourceIface.doAllocateNumericTable,
                                    DataSourceIface.doDictionaryFromContext)
        testData = HomogenNumericTable(nFeatures, 0, NumericTableIface.notAllocate)
        testGroundTruth = HomogenNumericTable(nDependentVariables, 0, NumericTableIface.notAllocate)
        mergedData = MergedNumericTable(testData, testGroundTruth)
        testDataSource.loadDataBlock(mergedData)

        algorithm = prediction.Batch_Float64DefaultDense()
        algorithm.input.setNumericTableInput(prediction.data, testData)
        algorithm.input.setModelInput(prediction.model, trainingResult.get(training.model))
        predictionResult = algorithm.compute()

        block1 = BlockDescriptor()
        block2 = BlockDescriptor()
        testGroundTruth.getBlockOfRows(0, testGroundTruth.getNumberOfRows(), readOnly, block1)
        predictionResult.get(prediction.prediction).getBlockOfRows(0, testGroundTruth.getNumberOfRows(), readOnly, block2)
        y_true = getClassVector(block1.getArray(), 0.000000000000)
        predictionRegression = block2.getArray()
        for thresholdIndex in range(0,len(thresholdValues)):
            y_pred = getClassVector(predictionRegression, thresholdValues[thresholdIndex])
            numberOfCorrectlyClassifiedObjects[thresholdIndex] += accuracy_score(y_true, y_pred, normalize=False)
        numberOfObjectsInTestFiles += len(y_true)
        numberOfNonzeroObjectsInTestFiles += np.count_nonzero(y_true)
        mergedData.freeDataMemory()
        testData.freeDataMemory()
        testGroundTruth.freeDataMemory()

    classificationAccuracyResult = np.zeros(len(thresholdValues))
    best_threshold = None
    best_accuracy = -1
    for thresholdIndex in range(0, len(thresholdValues)):
        classificationAccuracyResult[thresholdIndex] = numberOfCorrectlyClassifiedObjects[thresholdIndex] / numberOfObjectsInTestFiles
        if (classificationAccuracyResult[thresholdIndex] > best_accuracy):
            best_threshold = thresholdValues[thresholdIndex]
            best_accuracy = classificationAccuracyResult[thresholdIndex]
    print('Best threshold:{:.4f}. Best accuracy:{:.4f}'.format(best_threshold, best_accuracy))
    print('Test set. Number of objects of 0 class:{:.4f}.Number of objects of 1 class:{:.4f}. '
              'Frequency of 1 class:{:.4f}'.format(numberOfObjectsInTestFiles - numberOfNonzeroObjectsInTestFiles,
                                                   numberOfNonzeroObjectsInTestFiles,
                                                   numberOfNonzeroObjectsInTestFiles / numberOfObjectsInTestFiles))
    indexOfZeroThreshold = np.where(thresholdValues==0.0)[0][0]
    print('Threshold=0. Classification accuracy:{:.4f}'.format(classificationAccuracyResult[indexOfZeroThreshold]))

if __name__ == "__main__":
    trainDatasetFileNames = getDatasetFileNames('train_features_*.csv')
    testDatasetFileNames = getDatasetFileNames('test_features_*.csv')
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    rankId = comm.Get_rank()
    print("I am a worker with rank %d on %s." % (rankId, MPI.Get_processor_name()))

    start = MPI.Wtime()
    trainModel()
    if rankId == MPI_ROOT:
        end = MPI.Wtime()
        printNumericTable(trainingResult.get(training.model).getBeta(), "Ridge Regression coefficients:",
                          num_printed_cols=20)
#       testModel()
        print ('Computational time: {:.2f}'.format(end - start))
