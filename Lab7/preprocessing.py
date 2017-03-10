import numpy as np
import idx2numpy

X_train = idx2numpy.convert_from_file('train-images.idx3-ubyte')
y_train = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
X_test = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
y_test = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')

X_train = np.reshape(X_train, (60000, 784))
X_test = np.reshape(X_test, (10000, 784))
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
unique, counts = np.unique(y_train, return_counts=True)
print(dict(zip(unique, counts)))

np.savetxt("MNIST_train.csv",X_train / 255.0,delimiter=',',fmt='%.6f')
np.savetxt("MNIST_test.csv",X_test / 255.0,delimiter=',',fmt='%.6f')
np.savetxt("MNIST_train_ground_truth.csv",y_train,delimiter=',',fmt='%i')
np.savetxt("MNIST_test_ground_truth.csv",y_test,delimiter=',',fmt='%i')