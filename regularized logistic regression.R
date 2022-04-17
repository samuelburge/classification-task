# Load the necessary packages
require(MASS)
require(class)
require(glmnet)
require(tidyverse)

# Set working directory and import the data file
setwd("C:\\Users\\SamBu\\Desktop\\STAT 639")
load("class_data.RData")
dataset <- data.frame(y, x) 

# Standardize the features in the data matrix
scaled_x <- x
for (i in 1:length(x)) {
  scaled_x[,i] <- as.vector(scale(x[,i],center = TRUE, scale = TRUE))
}

# Coerce data vector to matrix, calculate training sample size, and create folds
xtrain <- as.matrix(scaled_x)
n <- nrow(xtrain)
k <- 10
folds <- sample(1:k, n, replace=TRUE)

# ====================================================================================
# Fit regularized logistic regression models using L1, elastic net, and L2
# using the tuning parameters (alpha and lambda) that minimize k-fold CV error rates
# ====================================================================================

# ====================================================================================
#           Estimate the test error for each fitted model using 10-fold CV
# ====================================================================================

# Initialize vectors to store the training errors for each fold
lasso.train.errors <- rep(0,k)
net.train.errors <- rep(0,k)
ridge.train.errors <- rep(0,k)

# Initialize vectors to store the CV errors for each fold
lasso.fold.errors <- rep(0,k)
net.fold.errors <- rep(0,k)
ridge.fold.errors <- rep(0,k)

# ===================================================================================
#                           K-fold cross validation loop
# ===================================================================================

for (j in 1:k) {
  
  # ====================================
  #      Lasso logistic regression
  # ====================================
  lasso <- cv.glmnet(xtrain[folds != j,], y[folds != j],
                  family = "binomial", alpha = 1, nfolds = 10,
                  type.measure="class")
  
  
  # Error on the training set   
  lasso.tr.pred <- predict(lasso, newx = xtrain[folds != j,],
                        s = lasso$lambda.min, type = 'class')
  
  lasso.train.errors[j] <- mean(lasso.tr.pred != y[folds != j])
  
  # Error on the validation set 
  lasso.pred <- predict(lasso, newx = xtrain[folds == j,],
                      s = lasso$lambda.min, type = 'class')
  
  lasso.fold.errors[j] <- mean(lasso.pred != y[folds == j])


  # ====================================
  #   Elastic net logistic regression
  # ====================================
  
  # Initialize vector to store misclassification rates and lambdas for varying alpha levels 
  
  alphas <- seq(from = 0.05, to = 0.95, by = 0.05)  # weighting parameter for L1 and L2 penalties
  lambdas <- rep(0, length(alphas))                 # shrinkage parameter
  net_error_rates <- length(alphas)                 # Vector to store results
  
  # Iterate over the alpha values, finding the lambda that minimizes error rate
  for (i in 1:length(alphas)) {
    cv.net <- cv.glmnet(xtrain, y,
                        alpha = alphas[i],
                        family = "binomial",
                        nfolds = 10,
                        type.measure = 'class')
    
    lambdas[i] <- cv.net$lambda.min
    net_error_rates[i] <- mean((predict(cv.net, xtrain, type = 'class',
                                        s = cv.net$lambda.min) != y))
  }
  
  best.alpha <- alphas[which.min(net_error_rates)]
  best.lambda <- lambdas[which.min(net_error_rates)]
  
  net <- glmnet(xtrain[folds != j,], y[folds != j],
                family = "binomial", alpha = best.alpha, lambda = best.lambda,
                type.measure="class")
  
  # Error on the training set   
  net.tr.pred <- predict(net, newx = xtrain[folds != j,],
                        s = best.lambda, type = 'class')
  
  net.train.errors[j] <- mean(lasso.tr.pred != y[folds != j])
  
  # Error on the validation set
  net.pred <- predict(net, newx = xtrain[folds == j,],
                      s = best.lambda, type = 'class')
  
  net.fold.errors[j] <- mean(net.pred != y[folds==j])

  # ====================================
  #      Ridge logistic regression
  # ====================================
  
  ridge <- cv.glmnet(xtrain[folds != j,], y[folds != j],
                  family = "binomial", alpha = 0, nfolds = 10,
                  type.measure="class")
  
  # Error on the training set   
  ridge.tr.pred <- predict(ridge, newx = xtrain[folds != j,],
                      s = ridge$lambda.min, type = 'class')
  
  ridge.train.errors[j] <- mean(ridge.tr.pred != y[folds != j]) 
  
  # Error on the validation set
  ridge.pred <- predict(ridge, newx = xtrain[folds == j,],
                        s = ridge$lambda.min, type = 'class')
  
  ridge.fold.errors[j] <- mean(ridge.pred != y[folds==j])
  
}

# Compute the average training error
lasso.train.error <- sum(lasso.train.errors)/k
net.train.error <- sum(net.train.errors)/k
ridge.train.error <- sum(ridge.train.errors)/k

# Compute the average validation error
lasso.cv.error <- sum(lasso.fold.errors)/k 
net.cv.error <- sum(net.fold.errors)/k
ridge.cv.error <- sum(ridge.fold.errors)/k

# Combine the estimated train and test errors into vectors
train.errors <- c(lasso.train.error, net.train.error, ridge.train.error)
names(cv.errors) <- c('Lasso','Net','Ridge')

cv.errors <- c(lasso.cv.error, net.cv.error, ridge.cv.error)
names(cv.errors) <- c('Lasso','Net','Ridge')

# Combine the training and test errors together for comparison
errors_matrix <- cbind(train.errors, cv.errors)
colnames(errors_matrix) <- c("Training Error","CV Error")
errors_matrix
     