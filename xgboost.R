library(xgboost)
library(tidyverse)

# Set working directory and import the data file
setwd("C:\\Users\\SamBu\\Desktop\\STAT 639")
load("class_data.RData")
set.seed(639)

# Coerce data vector to matrix, calculate training sample size, and create folds
k <- 10
folds <- sample(1:k, length(y), replace = TRUE)
train_err <- rep(0, k)
test_err <- rep(0, k)

for (i in 1:k) {
  
  # Convert the training and test sets of each fold into a data matrix (DMatrix)
  train <- xgb.DMatrix(data = as.matrix(x[folds != i, ]), label = y[folds != i])
  test <- xgb.DMatrix(data = as.matrix(x[folds == i, ]), label = y[folds == i])
  
  fit <- xgboost(data = train,
               objective = "binary:logistic",  # "binary:logistic" for logistic regression
               metrics = c('error', 'auc'),
               
               params = list(booster = 'gbtree',
                             subsample = 0.5,   # The number of obs. randomly selected for training iter.
                             eta = 0.05,        # Learning rate
                             max_depth = 6),    # Max. depth of each tree iteration
               
               nround = 10000,                  # max number of boosting iterations
               early_stopping_rounds = 100)     # stop boosting after k number of iteration with no improvement
  
  y_fit <- predict(fit, data.matrix(x[folds != i, ]))
  train_err[i] <- sum((y_fit > 0.5) != y[folds != i]) / length(y[folds != i])
  
  y_pred <- predict(fit, data.matrix(x[folds == i, ]))
  test_err[i] <- sum((y_pred > 0.5) != y[folds == i]) / length(y[folds == i])
  
}

# Print the training and test errors for each fold and calculate the CV score
train_err
test_err
cv_score <- mean(test_err)

# Alternatively, the built-in CV function be used as well to get error estimates.
data <- xgb.DMatrix(data = as.matrix(x), label = y)
 
fit  <- xgb.cv(data = data,
               objective = "binary:logistic",  # "binary:logistic" for logistic regression
               metrics = c('error', 'auc'),
       
              params = list(booster = 'gbtree',
                       subsample = 0.5,   # The number of obs. randomly selected for training iter.
                       eta = 0.05,        # Learning rate
                       max_depth = 6),    # Max. depth of each tree iteration
       
              nround = 10000,                  # max number of boosting iterations
              nfold = 5,                       # number of cross-validation folds
              early_stopping_rounds = 1000)    # stop boosting after k number of iteration with no improvement

# The lowest score for the 'test_error_mean' is the CV score
fit$evaluation_log
  