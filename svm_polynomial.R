# Load the necessary packages
library(ROCR)
library(e1071)

# Set working directory and import the data file
setwd("C:\\Users\\SamBu\\Desktop\\STAT 639")
load("class_data.RData")
dataset <- data.frame(y = as.factor(y), scale(x)) # required for SVM!
n <- nrow(dataset)

# Coerce data vector to matrix, calculate training sample size, and create folds
k <- 5
folds <- sample(1:k, n, replace=TRUE)
train_err <- rep(0, k)
test_err <- rep(0, k)

# Create search grid of tuning parameters for cross-validation testing
cost_grid <- seq(0.01, 100, length.out = 15)   # cost for all kernels
gamma_grid <- seq(0, 1, length.out = 15)       # gamma for all kernels
degree_grid <- seq(1, 5)                       # order for polynomial kernel


# Outer cross-validation for CV error
for (i in (1:k)) {
  
  cv_train_set <- dataset[(folds != i), ]
  cv_test_set <- dataset[(folds == i), ]
  
  # Inner cross-validation for tuning parameters
  # (I believe the tune function will do inner cross-validation)
  tune.out <- tune( svm, y ~ ., data = cv_train_set,
                    
                    kernel = "polynomial",
                   
                    ranges = list(cost = cost_grid,
                                  gamma = gamma_grid,
                                  degree = degree_grid
                                  ),
                   
                    tunecontrol = tune.control(sampling = "cross",
                                              cross = k,
                                              best.model = TRUE) )
  
  # Output the best model from the hyper-parameter tuning inner CV
  params <- tune.out$best.parameters
  
  train_err[i] <- tune.out$best.performance
  
  # Retrain the SVM on the full fold using the best cost value
  fit <- svm(y ~ ., data = cv_train_set,
             
             kernel = 'polynomial',
             
             cost = params$cost,
             
             gamma = params$gamma,
             
             degree = params$degree
             )
  
  # predict on the test set
  pred <- predict(fit, newdata = cv_test_set)
  
  # Calculate the test error
  test_err[i] <- sum(pred != cv_test_set[ ,1]) / nrow(cv_test_set)
  
}

# Calculate the CV score
print(cbind(1:5, train_err, test_err))
cv_score <- mean(test_err)
paste("The CV score is:", round(cv_score, 4))

save.image("svm_polynomial.RData")
