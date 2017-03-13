X = read.table("") #There should be name of the file and reading parameters, depending on the structure of the file

ptm <- proc.time()

for(i in 1:dim(M)[2])
{
  X = M[,i]
  n = length(X)
  min(X)
  max(X)
  S = sum(X)
  X2 = X%*%X
  MX = S/n
  X2/n
  VX = var(X)
  VX*n
  SVX = sqrt(VX)
  SVX/MX
}

p = proc.time() - ptm

quantile(X,probs = c(0.25,0.5,0.75))
cor(iris[1:50,1:2])[1,2]