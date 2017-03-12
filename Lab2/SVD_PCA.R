setwd("path/to/wd")

options(digits = 10, "scipen" = 10)

require(magrittr)
c("data.table", "dplyr", "ggplot2") %>% 
  sapply(require, character.only = TRUE)

X <- fread("cs-data.csv", drop = c(1), showProgress = FALSE) %>% 
             na.omit

Y <- scale(X, 
           center = T, 
           scale = T)

## ==================================================================================
## PCA via covariance matrix

Y_cov <- cov.wt(Y)
Y_pca <- princomp(x = Y, covmat = Y_cov, scores = TRUE)
str(Y_pca)

e_var <- c(0, 
           sapply(1:(length(Y_pca$sdev^2)), 
                  function(x, y) {sum(y[1:x])/sum(y)}, 
                  y = Y_pca$sdev^2)) 

ggplot(mapping = aes(x = 0:(length(e_var)-1), y = e_var)) + 
  geom_point() + geom_line() + 
  xlab("Число главных факторов") + 
  ylab("Доля объясненной вариации") + 
  scale_x_continuous(breaks = 0:(length(e_var)-1)) + 
  scale_y_continuous(breaks = seq(0, 1, 0.1))

# Loadings
L <- Y_pca$loadings[ ,1:k] %*% t(diag(Y_pca$sdev[1:k])) %>%
  as.data.frame

rownames(L) <- colnames(X) 
colnames(L) <- paste("u", 1:k, sep = "")

## ==================================================================================
## PSA via SVD
Y_svd <- svd(Y)

Y_svd$d^2 %>% `/`(., sum(.))

err <- c(1, 
         sapply(1:(length(Y_svd$d^2)-1), 
                function(x, y) {1 - sum(y[1:x])/sum(y)}, 
                y = Y_svd$d^2),
         0)
qplot(x = 0:(length(err)-1), 
      y = err, 
      geom = c("point", "line"),
      xlab = "Число главных компонент",
      ylab = "Ошибка в объясненной вариации")


## ==================================================================================
## Execution time

require(microbenchmark)
X <- fread("cs-data.csv", drop = c(1), showProgress = FALSE) %>%
  na.omit

cat("Processor time for R:\n")
print(
  microbenchmark(
    svd(Y),
    princomp(x = Y, covmat = Y_cov, scores = TRUE),
    times = 1000,
    unit = 's'))