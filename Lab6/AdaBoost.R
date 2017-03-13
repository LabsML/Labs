data.train = read.csv("train.csv", header = FALSE)
data.test = read.csv("test.csv", header = FALSE) #reading files with train and test parts of the dataset
data.train[,11] = as.factor(data.train[,11])
data.test[,11] = as.factor(data.test[,11]) #Here column 11 is class labels column

ptm = proc.time()
model = boosting(V11~.,data.train,mfinal = 10)
print(proc.time() - ptm)
prediction = predict(model,data.test)
prediction$confusion #confusion matrix for prediction