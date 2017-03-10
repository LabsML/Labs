import numpy as np
from daal.algorithms import optimization_solver
from daal.algorithms.neural_networks import training, prediction, initializers, layers
from daal.algorithms.neural_networks.layers import fullyconnected
from daal.algorithms.neural_networks.initializers import xavier
from daal.algorithms.neural_networks.layers import logistic
from daal.algorithms.neural_networks.layers.loss import softmax_cross
from daal.data_management import (NumericTable, readOnly, HomogenNumericTable, SubtensorDescriptor)
import time
from __init__ import printTensors, readTensorFromCSV

trainDatasetFile = 'MNIST_train.csv'
trainGroundTruthFile = 'MNIST_train_ground_truth.csv'
testDatasetFile = 'MNIST_test.csv'
testGroundTruthFile = 'MNIST_test_ground_truth.csv'

def configureNet(hidden_layers_sizes):
    topology = training.Topology()
    for layerNumber, layerSize in enumerate(hidden_layers_sizes):
        fullyConnectedLayer = layers.fullyconnected.Batch(layerSize)
        fullyConnectedLayer.parameter.weightsInitializer = initializers.xavier.Batch(55)
        fullyConnectedLayer.parameter.biasesInitializer = initializers.xavier.Batch(55)
        topology.push_back(fullyConnectedLayer)

        if layerNumber != len(hidden_layers_sizes)-1:
            logisticLayer = layers.logistic.Batch()
            topology.push_back(logisticLayer)
        else:
            softmaxCrossEntropyLayer = layers.loss.softmax_cross.Batch()
            topology.push_back(softmaxCrossEntropyLayer)
    for index in range(topology.size()-1):
        topology.get(index).addNext(index + 1)
    return topology

def trainModel(hidden_layers_sizes):
    trainingData = readTensorFromCSV(trainDatasetFile)
    trainingGroundTruth = readTensorFromCSV(trainGroundTruthFile)

    net = training.Batch()
    net.parameter.batchSize = 1
    topology = configureNet(hidden_layers_sizes)
    net.initialize(trainingData.getDimensions(), topology)

    net.input.setInput(training.data, trainingData)
    net.input.setInput(training.groundTruth, trainingGroundTruth)

    sgdAlgorithm = optimization_solver.sgd.Batch(fptype=np.float32)
    learningRate = 0.001
    sgdAlgorithm.parameter.learningRateSequence = HomogenNumericTable(1, 1, NumericTable.doAllocate, learningRate)
    sgdAlgorithm.parameter.batchSize = 200
    sgdAlgorithm.parameter.nIterations = 300
    sgdAlgorithm.parameter.accuracyThreshold = 1e-4
    net.parameter.optimizationSolver = sgdAlgorithm

    trainingModel = net.compute().get(training.model)
    return trainingModel.getPredictionModel_Float32()

def testModel(predictionModel, predictionDataFile):
    predictionData = readTensorFromCSV(predictionDataFile)
    net = prediction.Batch()
    net.parameter.batchSize = predictionData.getDimensionSize(0)
    net.input.setModelInput(prediction.model, predictionModel)
    net.input.setTensorInput(prediction.data, predictionData)
    return net.compute()

def printResults(predictionResult, predictionGroundTruthFile):
    predictionGroundTruth = readTensorFromCSV(predictionGroundTruthFile)
    printTensors(predictionGroundTruth, predictionResult.getResult(prediction.prediction),
                 "Ground truth", "Neural network predictions: each class probability",
                 "Neural network classification results (first 20 observations):", 20)

def findClasses(dataTable):
    dims1 = dataTable.getDimensions()
    nRows1 = int(dims1[0])
    block1 = SubtensorDescriptor()
    dataTable.getSubtensor([], 0, nRows1, readOnly, block1)
    nCols1 = int(block1.getSize() / nRows1)
    dataType = block1.getArray().flatten()
    dataType = np.reshape(dataType, (nRows1, nCols1))
    classes = np.argmax(dataType, axis=1)
    dataTable.releaseSubtensor(block1)
    return classes

def testModelQuality(predictionGroundTruthFile, predictedClasses):
    predictionGroundTruth = np.genfromtxt(predictionGroundTruthFile, delimiter=',',dtype=int)
    if len(predictionGroundTruth) != len(predictedClasses):
        return -1;
    truepositive = 0
    for i in range (len(predictionGroundTruth)):
        if (predictionGroundTruth[i] == predictedClasses[i]):
            truepositive = truepositive + 1
    return truepositive/len(predictionGroundTruth)

numberOfClasses = 10
topologyList = [[100,numberOfClasses],[100,100,numberOfClasses],[100,100,100,numberOfClasses]] #
for current_hidden_layers_sizes in topologyList:
    start = time.time()
    predictionModel = trainModel(current_hidden_layers_sizes)
    end = time.time()
    print("Time: %f" % (end - start))
    predictionResult = testModel(predictionModel, trainDatasetFile)
    #printResults(predictionResult, trainGroundTruthFile)
    predictedClasses = findClasses(predictionResult.getResult(prediction.prediction))
    print("Training set score: %f" %testModelQuality(trainGroundTruthFile, predictedClasses))

    predictionModel = trainModel(current_hidden_layers_sizes)
    predictionResult = testModel(predictionModel,testDatasetFile)
    #printResults(predictionResult, testGroundTruthFile)
    predictedClasses = findClasses(predictionResult.getResult(prediction.prediction))
    print("Test set score: %f" %testModelQuality(testGroundTruthFile, predictedClasses))

