import os
from os.path import join as jp

from mpi4py import MPI

from sklearn import metrics
import daal.algorithms.classifier as classifier
import daal.algorithms.multinomial_naive_bayes.prediction as prediction
import daal.algorithms.multinomial_naive_bayes.training as training
import fnmatch
from daal.data_management import (
    HomogenNumericTable, MergedNumericTable,DataSourceIface, FileDataSource, OutputDataArchive, InputDataArchive,
    BlockDescriptor_Float64, readOnly, NumericTableIface
)
datasetFolder = os.getcwd()
trainDatasetFileNames = []
testDatasetFileName = jp(datasetFolder, 'news_test_dense_dist_data.csv')
testGroundTruthFileName = jp(datasetFolder, 'news_test_dense_dist_label.csv')
from utils import printNumericTables

def getDatasetFileNames(filematching):
    matches = []
    for root, dirnames, filenames in os.walk(datasetFolder):
        for filename in fnmatch.filter(filenames, filematching):
            matches.append(os.path.join(root, filename))
    return matches

nClasses = 20
nFeatures = 101631

MPI_ROOT = 0

trainingResult = None
predictionResult = None


def trainModel():
    global trainingResult
    nodeResults = []
    # Create an algorithm object to build the final Naive Bayes model on the master node
    masterAlgorithm = training.Distributed_Step2MasterFloat64DefaultDense(nClasses)
    for filenameIndex in range(rankId, len(trainDatasetFileNames), comm_size):
        # Initialize FileDataSource to retrieve the input data from a .csv file
        #print("The worker with rank %d will read %s." % (rankId, trainDatasetFileNames[filenameIndex]))
        trainDataSource = FileDataSource(trainDatasetFileNames[filenameIndex],
                                         DataSourceIface.notAllocateNumericTable,
                                         DataSourceIface.doDictionaryFromContext)

        # Create Numeric Tables for training data and labels
        trainData = HomogenNumericTable(nFeatures, 0, NumericTableIface.notAllocate)
        trainDependentVariables = HomogenNumericTable(1, 0, NumericTableIface.notAllocate)
        mergedData = MergedNumericTable(trainData, trainDependentVariables)

        # Retrieve the data from the input file
        trainDataSource.loadDataBlock(mergedData)

        # Create an algorithm object to train the Naive Bayes model based on the local-node data
        localAlgorithm = training.Distributed_Step1LocalFloat64DefaultDense(nClasses)

        # Pass a training data set and dependent values to the algorithm
        localAlgorithm.input.set(classifier.training.data, trainData)
        localAlgorithm.input.set(classifier.training.labels, trainDependentVariables)

        # Train the Naive Bayes model on local nodes
        pres = localAlgorithm.compute()
        # Serialize partial results required by step 2
        dataArch = InputDataArchive()
        pres.serialize(dataArch)

        masterAlgorithm.input.add(classifier.training.partialModels, pres)
        """
        nodeResults.append(dataArch.getArchiveAsArray().copy())
        localAlgorithm.clean()
        """
        mergedData.freeDataMemory()
        trainData.freeDataMemory()
        trainDependentVariables.freeDataMemory()
    # Transfer partial results to step 2 on the root node
    pres = masterAlgorithm.compute()
    dataArch = InputDataArchive()
    pres.serialize(dataArch)
    nodeResults.append(dataArch.getArchiveAsArray().copy())
    serializedData = comm.gather(nodeResults)

    if rankId == MPI_ROOT:
        # Create an algorithm object to build the final Naive Bayes model on the master node
        masterAlgorithm = training.Distributed_Step2MasterFloat64DefaultDense(nClasses)

        for currentRank in range(len(serializedData)):
            for currentBlock in range(0, len(serializedData[currentRank])):
                # Deserialize partial results from step 1
                dataArch = OutputDataArchive(serializedData[currentRank][currentBlock])

                dataForStep2FromStep1 = classifier.training.PartialResult()
                dataForStep2FromStep1.deserialize(dataArch)

                # Set the local Naive Bayes model as input for the master-node algorithm
                masterAlgorithm.input.add(classifier.training.partialModels, dataForStep2FromStep1)

        # Merge and finalizeCompute the Naive Bayes model on the master node
        masterAlgorithm.compute()
        trainingResult = masterAlgorithm.finalizeCompute()

def testModel():
    global predictionResult

    # Initialize FileDataSource to retrieve the input data from a .csv file
    testDataSource = FileDataSource(testDatasetFileName,
                                    DataSourceIface.doAllocateNumericTable,
                                    DataSourceIface.doDictionaryFromContext)

    # Retrieve the data from an input file
    testDataSource.loadDataBlock()

    # Create an algorithm object to predict values of the Naive Bayes model
    algorithm = prediction.Batch(nClasses)

    # Pass a testing data set and the trained model to the algorithm
    algorithm.input.setTable(classifier.prediction.data,  testDataSource.getNumericTable())
    algorithm.input.setModel(classifier.prediction.model, trainingResult.get(classifier.training.model))

    # Predict values of the Naive Bayes model
    # Result class from classifier.prediction
    predictionResult = algorithm.compute()


def printResults():

    testGroundTruth = FileDataSource(testGroundTruthFileName,
                                     DataSourceIface.doAllocateNumericTable,
                                     DataSourceIface.doDictionaryFromContext)
    testGroundTruth.loadDataBlock()

    printNumericTables(testGroundTruth.getNumericTable(),
                       predictionResult.get(classifier.prediction.prediction),
                       "Ground truth",
                       "Classification results",
                       "NaiveBayes classification results (first 20 observations):",
                       20,
                       interval=15,
                       flt64=False)

def getArrayFromNT(table, nrows=0):
    bd = BlockDescriptor_Float64()
    if nrows == 0:
        nrows = table.getNumberOfRows()
    table.getBlockOfRows(0, nrows, readOnly, bd)
    npa = bd.getArray()
    table.releaseBlockOfRows(bd)
    return npa

if __name__ == "__main__":
    trainDatasetFileNames = getDatasetFileNames('news_train_dense_dist_data_*.csv')
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    rankId = comm.Get_rank()
    print("I am a worker with rank %d on %s." % (rankId, MPI.Get_processor_name()))
    start = MPI.Wtime()
    trainModel()
    if rankId == MPI_ROOT:
        end = MPI.Wtime()
        testModel()
        testGroundTruth = FileDataSource(testGroundTruthFileName,
                                         DataSourceIface.doAllocateNumericTable,
                                         DataSourceIface.doDictionaryFromContext)
        testGroundTruth.loadDataBlock()
        a = getArrayFromNT(predictionResult.get(classifier.prediction.prediction))
        b = getArrayFromNT(testGroundTruth.getNumericTable())
        acc = metrics.accuracy_score(a, b, normalize=True)
        print('Accuracy: {:.4f}'.format(acc))
        print('Computational time: {:.2f}'.format(end - start))

