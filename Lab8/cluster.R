data = read.table("") #There should be path to the file and parameters, that depend on the structure of the file
n = 20 #number of clusters
data.kmeans.20 = kmeans(data, n, iter.max = 100000, nstart = 1, algorithm = "Lloyd")
data.kmeans.20$centers #printing centers of the clusters


#EM algorithm by package mclust
z = array(0,dim = c(2178,n))
for(i in 1:2178)
{
  y = sample(1:n, 1)
  z[i,y] = 1
  #z[i,(i %% n) + 1] = 1
}

Mmodel = mstep("EEE",data,z)
emres = em(Mmodel$modelName,data,Mmodel$parameters,control =  emControl(itmax = c(10000,1)))
emres$parameters$mean #centers of EM clusters



