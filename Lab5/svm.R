ptm = proc.time()
data.train = read.csv("train.csv",header = FALSE)   #reading train and test parts of dataset
data.test = read.csv("test.csv",header = FALSE)
print(proc.time() - ptm)

ptm <- proc.time()
svm.model = svm(V5~., data = data.train, kernel = "radial", gamma = 1, cachesize = 200) 
#V5 is a column with class labels. If class labels are in n-th column, then Vn is a column in R
prediction = predict(svm.model,data.test)

print(proc.time() - ptm)

#predictions can be expressed as double values - but at the should be divided in two classes
for(i in 1:length(prediction))
{
  if(prediction[i] > 0)
    prediction[i] = 1
  else
    prediction[i] = -1
}

table(prediction,data.test$V5)

binarization <- function(df) #function which binarizes any categorical column
{
  N = dim(df)[1]
  M = matrix(0,nrow = N,ncol = 0)
  for(i in 1:length(df))
  {
    if(is.factor(df[,i]))
    {
      n = length(levels(df[,i]))
      if(n == 2)
      {
        V = matrix(0,nrow = N,ncol = 1)
        for(j in 1:N)
        {
          f = levels(df[,i])[1]
          if(df[j,i] == f)
            V[j,1] = 1
          else
            V[j,1] = 0
        }
        M = cbind(M,V)
      }
      else
      {
        for(k in 1:n)
        {
          V = matrix(0,nrow = N,ncol = 1)
          for(j in 1:N)
          {
            f = levels(df[,i])[k]
            if(df[j,i] == f)
              V[j,1] = 1
            else
              V[j,1] = 0
          }
          M = cbind(M,V)
        }
      }
    }
    else
      M = cbind(M,df[,i])
  }
  return(M)
}
