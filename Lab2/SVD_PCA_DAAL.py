import daal.algorithms.pca as pca
from daal.data_management import HomogenNumericTable, BlockDescriptor_Float64, readOnly

import numpy as np
from sklearn.preprocessing import scale

import matplotlib.pyplot as plt

import time

def getArrayFromNT(table, nrows=0):
    bd = BlockDescriptor_Float64()
    if nrows == 0:
        nrows = table.getNumberOfRows()
    table.getBlockOfRows(0, nrows, readOnly, bd)
    npa = bd.getArray()
    table.releaseBlockOfRows(bd)
    return npa

def printNT(table, nrows = 0, message=''):
    npa = getArrayFromNT(table, nrows)
    print(message, '\n', npa)


class PCA:

    def __init__(self, method = 'svd'):
        """Initialize class parameters
        Args:
           method: The default method is based on correation matrix. It
           can also be the SVD method ('svd')
        """

        if method != 'correlation' and method != 'svd':
            warnings.warn(method + 
            ' method is not supported. Default method is used', 
            UserWarning)

        self.method_ = method
        self.eigenvalues_ = None
        self.eigenvectors_ = None


    def compute(self, data):
        """Compute PCA the input data
        Args:
           data: Input data 
        """

        # Create an algorithm object for PCA
        if self.method_ == 'svd':
            pca_alg = pca.Batch_Float64SvdDense()
        else:
            pca_alg = pca.Batch_Float64CorrelationDense()

        # Set input
        pca_alg.input.setDataset(pca.data, data)
        # compute
        result = pca_alg.compute()
        self.eigenvalues_ = result.get(pca.eigenvalues)
        self.eigenvectors_ = result.get(pca.eigenvectors)
        
data = np.genfromtxt("cs-data.csv", delimiter = ',', dtype=np.double, 
                     skip_header = 1, usecols=list(range(1, 11)))
data = data[~np.isnan(data).any(axis = 1)]
data = scale(data)

data_nt = HomogenNumericTable(data)
print(data_nt.getNumberOfRows(), data_nt.getNumberOfColumns())

# PCA via SVD
pr = PCA(method='svd')
pr.compute(data_nt)
loadings = getArrayFromNT(pr.eigenvectors_)
ev = getArrayFromNT(pr.eigenvalues_)
print(ev/np.sum(ev))

# PCA via covariances
cov_data = np.cov(data.transpose())
cov_nt = HomogenNumericTable(cov_data)
pr = PCA(method='correlation')
pr.compute(cov_nt)
loadings = getArrayFromNT(pr.eigenvectors_)
ev = getArrayFromNT(pr.eigenvalues_)
print(ev/np.sum(ev))
var = np.round(np.cumsum(ev/np.sum(ev)), decimals=4)
plt.plot(np.arange(1,11), var)

# Execution time
pr = PCA(method='svd')
T = np.array([])
for i in range(0, 1000): 
    t = time.process_time()
    pr.compute(data_nt)
    t = time.process_time() - t
    T = np.append(T, t)
print("Processor time for SVD:")
print("Min time: ", np.min(T))
print("Mean time: ", np.mean(T))
print("Median time: ", np.median(T))
print("Max time: ", np.max(T), "\n")

pr = PCA(method='correlation')
T = np.array([])
for i in range(0, 1000): 
    t = time.process_time()
    pr.compute(cov_nt)
    t = time.process_time() - t
    T = np.append(T, t)
print("Processor time for Cov:")
print("Min time: ", np.min(T))
print("Mean time: ", np.mean(T))
print("Median time: ", np.median(T))
print("Max time: ", np.max(T), "\n")



