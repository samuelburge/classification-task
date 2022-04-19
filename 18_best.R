# ================================================================
#  STAT 639 FINAL PROJECT (GROUP 18)
#  BEST CLASSIFICATION MODEL
#  AUTHORS: Samuel Burge, Chad Austgen, Skylar Liu
#  DATE: April 19, 2022
# ================================================================

# Set seed for reproducibility
set.seed(639)

# Load the necessary packages for the classification task
require(tidyverse)
require(MASS)
require(class)
require(glmnet)
require(ROCR)
require(e1071)
require(naivebayes)
require(tree)
require(randomForest)
require(xgboost)
require(gbm)

# ====================================================================================
#                                CLASSIFICATION TASK
# ====================================================================================

# Set working directory and import the data file
setwd('C:\\Users\\SamBu\\Documents\\GitHub\\data-mining-project')
load('class_data.RData')

# ====================================================================================
# Set-up the k-fold cross validation
# ====================================================================================

# Needs to be a matrix for glmnet function to work
x <- as.matrix(x)

# Standardize the features in the data matrix
scaled_x <- scale(x, center = TRUE, scale = TRUE)

# Combine the label and features into a single data set for some algorithms (SVMs)
dataset <- data.frame(y = as.factor(y), scaled_x)

# Coerce data vector to matrix, calculate training sample size, and create folds
n <- nrow(x)
k <- 10
folds <- sample(1:k, n, replace=TRUE)


# Create search grid of SVM tuning parameters for cross-validation testing
cost_grid <- seq(0.01, 100, length.out = 15)  # cost for all kernels
gamma_grid <- seq(0, 1, length.out = 15)      # gamma for all kernels
degree_grid <- seq(1, 5)                      # degrees for polynomial kernel

# Grid search for the best parameters for the boosted trees
grid_search <- expand.grid(subsample = seq(from = 0.5, to = 1, by = 0.1),
                           max_depth = c(1, 2, 3, 4, 5),
                           eta = seq(0.001, 0.01, 0.005),
                           best_iteration = 0,
                           test_error_mean = 0)

# Storage for the best parameters of each boosted tree in each fold
boost.tuning.params <- cbind(fold = seq(1:k),
                             subsample = rep(0, 10),
                             max_depth = rep(0,10),
                             eta = rep(0, 10),
                             nrounds = rep(0, 10))

# ====================================================================================
#           Estimate the test error for each fitted model using 10-fold CV
# ====================================================================================

# Initialize vectors to store the training and estimated test error
boost.fold.errors <- rep(0,k)

# ===================================================================================
#                           K-fold cross validation loop
# ===================================================================================

for (j in 1:k) {
  
  # ====================================
  #      Boosted Trees (XGBoost)
  # ====================================
  
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
                        nfold = k)
    
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
  boost.fold.errors[j] <- sum((y_pred > 0.5) != y[folds == j]) / length(y[folds == j])
  
} # END OUTER CV LOOP

# ===================================================================================
#  Compute test error for the best model, refit the model, and compute predictions
# ===================================================================================

# Calculate the estimated test error
test_error <- mean(boost.fold.errors)

# Re-tune the model using all the training data
final_grid_search <- expand.grid(subsample = seq(from = 0.5, to = 1, by = 0.1),
                                 max_depth = c(1, 2, 3, 4, 5),
                                 eta = seq(0.001, 0.01, 0.005),
                                 best_iteration = 0,
                                 test_error_mean = 0)

data <- xgb.DMatrix(data = as.matrix(x), label = y)

for (i in 1:nrow(grid_search)) {
  inner_fit <- xgb.cv(data = data,
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

# Identify the best tuning parameters
boost.tuning.params[j, 'subsample'] <- grid_search[which.min(grid_search$test_error_mean), 'subsample']
boost.tuning.params[j, 'eta'] <- grid_search[which.min(grid_search$test_error_mean), 'eta']
boost.tuning.params[j, 'max_depth'] <- grid_search[which.min(grid_search$test_error_mean), 'max_depth']
boost.tuning.params[j, 'nrounds'] <- grid_search[which.min(grid_search$test_error_mean), 'best_iteration']

# Fit the data with the best tuning parameters 
best_fit <- xgboost(data = data,
               objective = "binary:logistic",  # "binary:logistic" for logistic regression
               metrics = c('error'),
               
               params = list(booster = 'gbtree',
                             subsample = boost.tuning.params[j, 'subsample'],
                             eta = boost.tuning.params[j, 'eta'],
                             max_depth = boost.tuning.params[j, 'max_depth']),
               
               nrounds = boost.tuning.params[j, 'nrounds'])

# Generate predictions on the test set
ynew <- predict(best_fit, data.matrix(xnew))

# Converts the prediction to 0, 1 labels like the training y labels
ynew <- as.numeric(ynew >= 0.5)

# Save the test predictions and the estimated test error as specified
save(ynew, test_error, file = "18.RData")
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  