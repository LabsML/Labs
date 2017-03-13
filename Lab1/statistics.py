import numpy as np
import scipy as sp
import time
import scipy.stats as spstats

data = sp.genfromtxt("", delimiter=',') #There should be the path to the file

t1 = time.time()
t2 = time.perf_counter()
t3 = time.process_time()

for i in range(0,data.shape[1]): #data.shape[1] is the size of the second dimension of the matrix (number of columns)
    print("Feature ",i-2)
    x = data[:,i]
    size = np.size(x)
    min = np.min(x)
    max = np.max(x)
    print("Minimum: ",min)
    print("Maximum: ",max)

    sum = np.sum(x)
    print("Sum: ",sum)

    sum2 = np.dot(x,x)
    print("Sum squared: ", sum2)

    mean = sum/size
    mean2 = sum2/size
    print("Mean: ",mean)
    print("Second order moment: ", sum2/size)

    var = np.var(x)
    SDM = var * size
    print("Sum of squared differences from the means", SDM)
    print("Variance: ",var)

    std = np.sqrt(var)
    varcoef = std/mean
    print("Standard deviation: ",std)
    print("Variation coefficient",varcoef)
    print(" ")

print("Total time = ", time.time() - t1)
print("Performance time = ", time.perf_counter() - t2)
print("Processor time = ", time.process_time() - t3)
corr = np.corrcoef(data,rowvar = 0)
print(np.max(corr))
print(corr)

x = data[:,0]

print("Quantiles: ",np.percentile(x,[25,50,75]))

