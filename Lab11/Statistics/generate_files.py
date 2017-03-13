from numpy import random
import numpy as np
import sys

N = 100000000

NUMBER_OF_FILES = 10

NUMBER_OF_SAMPLES_IN_FILE = int(N/NUMBER_OF_FILES)

PATH = "" #Path for files to generate

mean = [0, 0, 0, 0, 0]
cov = [[1,0,0,0,0],
       [0,1,0,0,0],
       [0,0,1,0,0],
       [0,0,0,1,0],
       [0,0,0,0,1]]


for i in range(NUMBER_OF_FILES):

    X[j, :] = random.multivariate_normal(mean=mean, cov=cov, size=NUMBER_OF_SAMPLES_IN_FILE)
    filename = PATH + "low" + str(i+1) + ".csv"
    np.savetxt(filename, X, delimiter=",")


