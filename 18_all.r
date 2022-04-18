# ================================================================
#  STAT 639 FINAL PROJECT (GROUP 18)
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

# Load the necessary packages for the clustering task
require(factoextra)
require(cluster)
require(NbClust)
require(dbscan)

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

# Create search grid of SVM tuning parameters for cross-validation testing
cost_grid <- seq(0.01, 100, length.out = 15)  # cost for all kernels
gamma_grid <- seq(0, 1, length.out = 15)      # gamma for all kernels
degree_grid <- seq(1, 5)                      # degrees for polynomial kernel

# Coerce data vector to matrix, calculate training sample size, and create folds
n <- nrow(x)
k <- 10
folds <- sample(1:k, n, replace=TRUE)

# ====================================================================================
#           Estimate the test error for each fitted model using 10-fold CV
# ====================================================================================

# Initialize vectors to store the training error for each fold
lasso.train.errors <- rep(0,k)
net.train.errors <- rep(0,k)
ridge.train.errors <- rep(0,k)
naive.train.errors <- rep(0,k)
radialSVM.train.errors <- rep(0,k)
polySVM.train.errors <- rep(0,k)
sigmoidSVM.train.errors <- rep(0,k)
boost.train.errors <- rep(0,k)


# Initialize vectors to store the CV error for each fold
lasso.fold.errors <- rep(0,k)
net.fold.errors <- rep(0,k)
ridge.fold.errors <- rep(0,k)
naive.fold.errors <- rep(0,k)
radialSVM.fold.errors <- rep(0,k)
polySVM.fold.errors <- rep(0,k)
sigmoidSVM.fold.errors <- rep(0,k)
boost.fold.errors <- rep(0,k)

# ===================================================================================
#                           K-fold cross validation loop
# ===================================================================================

for (j in 1:1) {

  # =================================================================================
  # Fit regularized logistic regression models using L1, elastic net, and L2
  # using the tuning parameters (alpha and lambda) that min. k-fold CV error rates
  # =================================================================================
    
  # ====================================
  #      Lasso logistic regression
  # ====================================
  lasso <- cv.glmnet(x[folds != j, ], y[folds != j],
                     family = "binomial", alpha = 1, nfolds = 10,
                     type.measure="class")
  
  # Error on the training set   
  lasso.tr.pred <- predict(lasso, newx = x[folds != j, ],
                           s = lasso$lambda.min, type = 'class')
  
  lasso.train.errors[j] <- mean(lasso.tr.pred != y[folds != j])
  
  # Error on the validation set 
  lasso.pred <- predict(lasso, newx = x[folds == j, ],
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
    cv.net <- cv.glmnet(x, y,
                        alpha = alphas[i],
                        family = "binomial",
                        nfolds = 10,
                        type.measure = 'class')
    
    lambdas[i] <- cv.net$lambda.min
    net_error_rates[i] <- mean((predict(cv.net, x, type = 'class',
                                        s = cv.net$lambda.min) != y))
  }
  
  best.alpha <- alphas[which.min(net_error_rates)]
  best.lambda <- lambdas[which.min(net_error_rates)]
  
  net <- glmnet(x[folds != j, ], y[folds != j],
                family = "binomial", alpha = best.alpha, lambda = best.lambda,
                type.measure="class")
  
  # Error on the training set   
  net.tr.pred <- predict(net, newx = x[folds != j, ],
                         s = best.lambda, type = 'class')
  
  net.train.errors[j] <- mean(lasso.tr.pred != y[folds != j])
  
  # Error on the validation set
  net.pred <- predict(net, newx = x[folds == j, ],
                      s = best.lambda, type = 'class')
  
  net.fold.errors[j] <- mean(net.pred != y[folds==j])
  
  # ====================================
  #      Ridge logistic regression
  # ====================================
  
  ridge <- cv.glmnet(x[folds != j, ], y[folds != j],
                     family = "binomial", alpha = 0, nfolds = 10,
                     type.measure="class")
  
  # Error on the training set   
  ridge.tr.pred <- predict(ridge, newx = x[folds != j, ],
                           s = ridge$lambda.min, type = 'class')
  
  ridge.train.errors[j] <- mean(ridge.tr.pred != y[folds != j]) 
  
  # Error on the validation set
  ridge.pred <- predict(ridge, newx = x[folds == j, ],
                        s = ridge$lambda.min, type = 'class')
  
  ridge.fold.errors[j] <- mean(ridge.pred != y[folds==j])
  
  # ====================================
  #            Naive Bayes
  # ====================================
  
  fit <- naive_bayes(y ~ .,
                     data = data.frame(x[folds != j, ], y = as.factor(y[folds != j])),
                     usekernel = TRUE)
  
  train.pred <- predict(fit, x[folds != j, ])
  naive.train.errors[j] <- mean(train.pred == y[folds != j])
  
  cv.pred <- predict(fit, x[folds == j , ])
  naive.fold.errors[j] <- mean(cv.pred == y[folds == j])
  
  # ====================================
  #      SVM with radial kernel
  # ====================================
  
  # This will be used for the remaining SVM algorithms as well
  cv_train_set <- dataset[(folds != j), ]
  cv_test_set <- dataset[(folds == j), ]
  
  # Inner cross-validation for tuning parameters
  # (The tune function will do inner cross-validation)
  tune.out <- tune( svm, y ~ ., data = cv_train_set,
                    
                    kernel = "radial",
                    
                    ranges = list(cost = cost_grid,
                                  gamma = gamma_grid),
                    
                    tunecontrol = tune.control(sampling = "cross",
                                               cross = k,
                                               best.model = TRUE) )
  
  # Output the best model from the hyper-parameter tuning inner CV
  params <- tune.out$best.parameters
  
  radialSVM.train.errors[j] <- tune.out$best.performance
  
  # Retrain the SVM on the full fold using the best cost value
  fit <- svm(y ~ ., data = cv_train_set,
             
             kernel = 'radial',
             
             cost = params$cost,
             
             gamma = params$gamma)
  
  # predict on the test set
  pred <- predict(fit, newdata = cv_test_set)
  
  # Calculate the test error
  radialSVM.fold.errors[j] <- sum(pred != cv_test_set[ ,1]) / nrow(cv_test_set)
  
  # ====================================
  #      SVM with poly. kernel
  # ====================================
  
  # Inner cross-validation for tuning parameters
  # (I believe the tune function will do inner cross-validation)
  tune.out <- tune( svm, y ~ ., data = cv_train_set,
                    
                    kernel = "polynomial",
                    
                    ranges = list(cost = cost_grid,
                                  gamma = gamma_grid,
                                  degree = degree_grid),
                    
                    tunecontrol = tune.control(sampling = "cross",
                                               cross = k,
                                               best.model = TRUE) )
  
  # Output the best model from the hyper-parameter tuning inner CV
  params <- tune.out$best.parameters
  
  polySVM.train.errors[j] <- tune.out$best.performance
  
  # Retrain the SVM on the full fold using the best cost value
  fit <- svm(y ~ ., data = cv_train_set,
             
             kernel = 'polynomial',
             
             cost = params$cost,
             
             gamma = params$gamma,
             
             degree = params$degree )
  
  # predict on the test set
  pred <- predict(fit, newdata = cv_test_set)
  
  # Calculate the test error
  polySVM.fold.errors[j] <- sum(pred != cv_test_set[ ,1]) / nrow(cv_test_set)
  
  # ====================================
  #      SVM with Sigmoid kernel
  # ====================================
  
  tune.out <- tune( svm, y ~ ., data = cv_train_set,
                    
                    kernel = "sigmoid",
                    
                    ranges = list(cost = cost_grid,
                                  gamma = gamma_grid),
                    
                    tunecontrol = tune.control(sampling = "cross",
                                               cross = k,
                                               best.model = TRUE) )
  
  # Output the best model from the hyper-parameter tuning inner CV
  params <- tune.out$best.parameters
  
  sigmoidSVM.train.errors[j] <- tune.out$best.performance
  
  # Retrain the SVM on the full fold using the best cost value
  fit <- svm(y ~ ., data = cv_train_set,
             
             kernel = 'sigmoid',
             
             cost = params$cost,
             
             gamma = params$gamma)
  
  # predict on the test set
  pred <- predict(fit, newdata = cv_test_set)
  
  # Calculate the test error
  sigmoidSVM.train.errors[j] <- sum(pred != cv_test_set[ ,1]) / nrow(cv_test_set)
  
  # ====================================
  #          Boosted Trees
  # ====================================
  
  
}

# ===================================================================================
#      Random forest doesn't need to be cross-validated (will use OOB error)
# ===================================================================================
     
# Fit a random forest to the data
fit <- randomForest(y = dataset[, 1],
                    x = dataset[,-1],
                    ntree = 10000)

# Calculate the training error rate
oob_error_rate <- fit$err.rate[10000]

# ===================================================================================
#        Compute the CV error for each algorithm and compare the results
# ===================================================================================

# Compute the average training error
     lasso.train.error <- sum(lasso.train.errors) / k
       net.train.error <- sum(net.train.errors) / k
     ridge.train.error <- sum(ridge.train.errors) / k
     naive.train.error <- sum(naive.train.errors) / k
 radialSVM.train.error <- sum(radialSVM.train.errors) / k
   polySVM.train.error <- sum(polySVM.train.errors) / k
sigmoidSVM.train.error <- sum(sigmoidSVM.train.errors) / k
randforest.train.error <- sum(randforest.train.errors) / k
     boost.train.error <- sum(boost.train.errors) / k

# Compute the average validation error
     lasso.cv.error <- sum(lasso.fold.errors) / k 
       net.cv.error <- sum(net.fold.errors) / k
     ridge.cv.error <- sum(ridge.fold.errors) / k
     naive.cv.error <- sum(naive.fold.errors) / k
 radialSVM.cv.error <- sum(radialSVM.fold.errors) / k
   polySVM.cv.error <- sum(polySVM.fold.errors) / k
sigmoidSVM.cv.error <- sum(sigmoidSVM.fold.errors) / k
randforest.cv.error <- sum(randforest.fold.errors) / k
     boost.cv.error <- sum(boost.fold.errors) / k

# Combine the estimated train and test errors into vectors
    train.errors <- c(lasso.train.error, net.train.error, ridge.train.error, naive.train.error,
                      radialSVM.train.error, polySVM.train.error, sigmoidSVM.train.error,
                      randforest.train.error, boost.train.error)
    
names(train.errors) <- c('Lasso','Net','Ridge', 'Naive Bayes', 'Radial SVM',
                         'Poly. SVM', 'Sigmoid SVM', 'Random Forest', 'Boosted Trees')

       cv.errors <- c(lasso.cv.error, net.cv.error, ridge.cv.error, naive.cv.error,
                      radialSVM.cv.error, polySVM.cv.error, sigmoidSVM.cv.error,
                      randforest.cv.error, boost.cv.error)
       
names(cv.errors) <- c('Lasso','Net','Ridge', 'Naive Bayes', 'Radial SVM',
                      'Poly. SVM', 'Sigmoid SVM', 'Random Forest', 'Boosted Trees')

# Combine the training and test errors together for comparison
          errors_matrix <- cbind(train.errors, cv.errors)
colnames(errors_matrix) <- c("Avg. Training Error","Est. Test Error")

# Print the results
errors_matrix




# ===================================================================================
#                                  CLUSTERING TASK
# ===================================================================================

# Set working directory and import the data file
setwd("C:\\Users\\SamBu\\Desktop\\STAT 639")
load("cluster_data.RData")










