import numpy as np
from sklearn.neural_network import MLPClassifier
import time

trainX = np.genfromtxt('MNIST_train.csv', delimiter=',')
trainY = np.genfromtxt('MNIST_train_ground_truth.csv', delimiter=',')
testX = np.genfromtxt('MNIST_test.csv', delimiter=',')
testY = np.genfromtxt('MNIST_test_ground_truth.csv', delimiter=',')

hidden_layer_sizes = [(100,),(100,100),(100,100,100)]
for hidden_layer_size in hidden_layer_sizes:
    print("Hidden layers sizes: ", end="")
    print(hidden_layer_size)
    start = time.time()
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_size, batch_size=200, max_iter=50, alpha=0.0, # alpha-regularization parameter
                        solver='sgd', activation='logistic', verbose=True, tol=1e-4, random_state=1, #identity’ np.random.uniform(-1,0)
                        learning_rate_init=.1)
    mlp.fit(trainX, trainY)
    end = time.time()
    print("Time: %f" % (end - start))
    print("Training set score: %f" % mlp.score(trainX, trainY))
    print("Test set score: %f" % mlp.score(testX, testY))
    print([coef.shape for coef in mlp.coefs_])
    print([intercept.shape for intercept in mlp.intercepts_])


