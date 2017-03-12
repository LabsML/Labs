from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import time

newsgroups_train = fetch_20newsgroups_vectorized(subset='train', 
                                                 remove = ('headers', 'footers', 'quotes'))


svd = TruncatedSVD(n_components = 3000, algorithm = "randomized")
t = time.process_time()
svd.fit(newsgroups_train.data)
t = time.process_time() - t
print("Sklearn time: ", t, "\n")

print(svd.explained_variance_ratio_) 
print(svd.explained_variance_ratio_.sum()) 

var_nwsd = np.round(np.cumsum(svd.explained_variance_ratio_), decimals=4)
plt.plot(np.arange(1,3001), var_nwsd)
plt.ylabel('Variation'); plt.xlabel('Number of PC')

from daal.algorithms.svd import Batch, data, singularValues, rightSingularMatrix, leftSingularMatrix
from daal.data_management import HomogenNumericTable, BlockDescriptor_Float64, readOnly
import numpy as np
from sklearn.preprocessing import scale
import scipy as sp
from scipy.sparse import csr_matrix

def getArrayFromNT(table, nrows=0):
    bd = BlockDescriptor_Float64()
    if nrows == 0:
        nrows = table.getNumberOfRows()
    table.getBlockOfRows(0, nrows, readOnly, bd)
    npa = bd.getArray()
    table.releaseBlockOfRows(bd)
    return npa

nwsd = newsgroups_train.data
nwsd = nwsd.transpose()
nwsd_dense = nwsd.toarray()
print("Размерность данных \n", nwsd_dense.shape, "\n") 
nwsd_dense_nt = HomogenNumericTable(nwsd_dense)

algorithm = Batch()
algorithm.input.set(data, nwsd_dense_nt)
t = time.process_time()
result = algorithm.compute()
t = time.process_time() – t
print("DAAL time: ", t, "\n")

ev = np.square(getArrayFromNT(result.get(singularValues)))
var_all = ev.sum()
var = ev/var_all
var_explained = np.round(np.cumsum(var), decimals=4)
print(" Доля объясненной вариации \n", var_explained[2999].sum(), "\n") 
plt.plot(np.arange(1,3000), var_explained[0:2999])
plt.ylabel('Variation'); plt.xlabel('Number of PC')



