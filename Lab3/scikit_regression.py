import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import time
from sklearn.metrics import r2_score

def linear_regression_model(dataX, dataY):
    regr = linear_model.LinearRegression() # Create a linear regression object normalize=True
    regr.fit(dataX, dataY) # Train the model
    return regr
def ridge_regression_model(dataX, dataY, alphaParam):
    ridge = linear_model.Ridge(alpha = alphaParam) # initialize the model
    ridge.fit(dataX, dataY) # fit the train data
    return ridge
def lasso_model(dataX, dataY, alphaParam):
    lasso = linear_model.Lasso(alpha=alphaParam, max_iter=1000) # initialize the model
    lasso.fit(dataX, dataY)  # fit the train data
    return lasso
def RMSE(data, actualY, regressionModel):
    return np.sqrt(mean_squared_error(actualY, regressionModel.predict(data)))
def R2(data, actualY, regressionModel):
    return r2_score(actualY, regressionModel.predict(data))

train_data=np.genfromtxt('kc_house_train_data.csv', delimiter=',')
test_data=np.genfromtxt("kc_house_test_data.csv", delimiter=',')
nFeatures = train_data.shape[1] - 1
trainX = train_data[:,1:(nFeatures+1)]
testX = test_data[:,1:(nFeatures+1)]

trainY = train_data[:,0]
testY = test_data[:,0]

remainingIndexes = [i for i in range(0, nFeatures)]
# list of features included in the regression model and the calculated train and validation errors (RMSE)

start = time.time()
for num in range(100):
    linRegr = linear_regression_model(trainX, trainY)
end = time.time()
print('Performance comparison. Time: %s seconds' % (end - start))
coefs = []
coefs.append(linRegr.coef_)
print('Linear regression with all features. Train set RMSE={:.2f} and R2={:.4f}; Test set RMSE={:.2f} and R2={:.4f}'.format(
    RMSE(trainX, trainY, linRegr), R2(trainX, trainY, linRegr),
    RMSE(testX, testY, linRegr),R2(testX, testY, linRegr)))

best_test_RMSE = float("inf")
best_alpha = -1
for alpha in np.linspace(0.0,100.0,num = 101):
    ridge = ridge_regression_model(trainX, trainY, alpha)
    testRMSE = RMSE(testX, testY, ridge)
    if (testRMSE < best_test_RMSE):
        best_alpha = alpha
        best_test_RMSE = testRMSE
ridge = ridge_regression_model(trainX, trainY, best_alpha)
coefs.append(ridge.coef_)
print('Ridge regression (with alpha={:.4f}). Train set RMSE={:.2f} and R2={:.4f}; Test set RMSE={:.2f} and R2={:.4f}'.format
       (best_alpha,RMSE(trainX, trainY, ridge),R2(trainX, trainY, ridge),RMSE(testX, testY, ridge),R2(testX, testY, ridge)))
best_test_RMSE = float("inf")
best_alpha = -1
for alpha in np.linspace(0.0, 1000.0, num=101):
    lasso = lasso_model(trainX, trainY, alpha)
    validationRMSE = RMSE(testX, testY, lasso)
    if (validationRMSE < best_test_RMSE):
        best_alpha = alpha
        best_test_RMSE = validationRMSE
lasso = lasso_model(trainX, trainY, best_alpha)
coefs.append(lasso.coef_)
print('Lasso (with alpha={:.3f}). Train set RMSE={:.2f} and R2={:.4f}; Test set RMSE={:.2f} and R2={:.4f}'.format(
    best_alpha, RMSE(trainX, trainY, lasso), R2(trainX, trainY, lasso), RMSE(testX, testY, lasso), R2(testX, testY, lasso)))
print (lasso.sparse_coef_.getnnz)
np.savetxt('coefficients.csv', coefs, delimiter=';', fmt='%.3f',)
