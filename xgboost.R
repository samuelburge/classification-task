library(xgboost)
library(tidyverse)

# Set working directory and import the data file
setwd("C:\\Users\\SamBu\\Desktop\\STAT 639")
load("class_data.RData")
set.seed(639)

# Coerce data vector to matrix, calculate training sample size, and create folds
k <- 10
folds <- sample(1:k, length(y), replace = TRUE)
boost.train.errors <- rep(0, k)
boost.test.errors <- rep(0, k)

# Create grid 
boost.tuning.params <- cbind(fold = seq(1:k),
                             subsample = rep(0, 10),
                             max_depth = rep(0,10),
                             eta = rep(0, 10),
                             nrounds = rep(0, 10))

grid_search <- expand.grid(subsample = seq(from = 0.5, to = 1, by = 0.1),
                           max_depth = c(1, 2, 3, 4, 5),
                           eta = seq(0.001, 0.01, 0.005),
                           best_iteration = 0,
                           test_error_mean = 0)

for (j in 1:k) {
  
  # Convert the training and test sets of each fold into a data matrix (DMatrix)
  train <- xgb.DMatrix(data = as.matrix(x[folds != j, ]), label = y[folds != j])
  test <- xgb.DMatrix(data = as.matrix(x[folds == j, ]), label = y[folds == j])
  best_iteration <- rep(0, nrow(grid_search))
  
  for (i in 1:nrow(grid_search)) {
  inner_fit <- xgb.cv(data = train,
                      objective = "binary:logistic",  # "binary:logistic" for logistic regression
                      metrics = c('error'),
               
                      params = list(booster = 'gbtree',
                                    subsample = grid_search[i, 'subsample'],
                                    eta = grid_search[i, 'eta'],
                                    max_depth = grid_search[i, 'max_depth']),
               
                 nrounds = 10000,              # max number of boosting iterations
                 early_stopping_rounds = 150,  # stop boosting after x iterations with no improvement
                 nfold = 5)
  
  # Store the results for each combination of tuning parameters and the associated performance
  grid_search[i, 'best_iteration'] = inner_fit$best_iteration
  grid_search[i, 'test_error_mean'] = inner_fit$evaluation_log[inner_fit$best_iteration, 'test_error_mean']
  }
  
  boost.tuning.params[j, 'subsample'] <- grid_search[which.min(grid_search$test_error_mean), 'subsample']
  boost.tuning.params[j, 'eta'] <- grid_search[which.min(grid_search$test_error_mean), 'eta']
  boost.tuning.params[j, 'max_depth'] <- grid_search[which.min(grid_search$test_error_mean), 'max_depth']
  boost.tuning.params[j, 'nrounds'] <- grid_search[which.min(grid_search$test_error_mean), 'best_iteration']
  
  fit <- xgboost(data = train,
         objective = "binary:logistic",  # "binary:logistic" for logistic regression
         metrics = c('error'),
         
         params = list(booster = 'gbtree',
                       subsample = boost.tuning.params[j, 'subsample'],
                       eta = boost.tuning.params[j, 'eta'],
                       max_depth = boost.tuning.params[j, 'max_depth']),
         
         nrounds = boost.tuning.params[j, 'nrounds']) 

  y_fit <- predict(fit, data.matrix(x[folds != j, ]))
  boost.train.errors[j] <- sum((y_fit > 0.5) != y[folds != j]) / length(y[folds != j])
  
  y_pred <- predict(fit, data.matrix(x[folds == j, ]))
  boost.test.errors[j] <- sum((y_pred > 0.5) != y[folds == j]) / length(y[folds == j])
  
}

# Print the training and test errors for each fold and calculate the CV score
boost.tuning.params
boost.train.errors
boost.test.errors

boost.train.error <- mean(boost.train.errors)
boost.cv.error <- mean(boost.test.errors)

save.image('xgboost.RData')
