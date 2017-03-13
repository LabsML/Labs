import numpy as np
from sklearn import cluster
from sklearn import mixture
import scipy as sp
import pandas as pd
import time

dataset = np.genfromtxt("", delimiter=',') #There should be the path to the file

n = 50 #number of components

model = cluster.KMeans(n_clusters=n,init='random', algorithm='full', max_iter=10000,n_init=1) #Kmeans
clusterobj = model.fit(dataset)

model = mixture.GaussianMixture(n_components=50, max_iter=10000,n_init=1, init_params="random") #EM
obj = model.fit(dataset)
