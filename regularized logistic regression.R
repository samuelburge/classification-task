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

# Initialize vector to store misclassification rates and lambdas for varying alpha levels 
alphas <- seq(from = 0, to = 1, by = 0.01)  # weighting parameter for L1 and L2 penalties
lambdas <- rep(0, length(alphas))           # shrinkage parameter
net_error_rates <- rep(0, length(alphas))

# Iterate over the alpha values, finding the lambda that minimizes error rate
for (i in 1:length(alphas)) {
  cv.net <- cv.glmnet(xtrain, y,
                      alpha = alphas[i],
                      family = "binomial",
                      foldid = folds,
                      type.measure = 'class')
  
  lambdas[i] <- cv.net$lambda.min
  net_error_rates[i] <- mean((predict(cv.net, xtrain, type = 'class', s = cv.net$lambda.min) != y))
}

# Collect the results into a matrix
net_cv_results <- cbind(alphas, lambdas, net_error_rates)
colnames(net_cv_results) <- c("Alpha","Lambda","Error Rate")
net_cv_results

# Find the best elastic net result (0 < alpha < 1)
trim_results <- net_cv_results[2:100,]
filter <- trim_results[order(trim_results[,'Error Rate']),][1,1]

cv.lasso <- cv.glmnet(xtrain, y, alpha = 1, family = "binomial", foldid = folds,  type.measure = 'class')
cv.net <- cv.glmnet(xtrain, y, alpha = filter, family = "binomial", foldid = folds,  type.measure = 'class')
cv.ridge <- cv.glmnet(xtrain, y, alpha = 0, family = "binomial", foldid = folds,  type.measure = 'class')

# Calculate the estimated probabilities for the training data (class = 'response' gives probability)
lasso_probs <- predict(cv.lasso, xtrain, type = 'response', s = cv.lasso$lambda.min)
net_probs <- predict(cv.net, xtrain, type = 'response', s = cv.net$lambda.min)
ridge_probs <- predict(cv.ridge, xtrain, type = 'response', s = cv.ridge$lambda.min)

# Puts estimated probabilities into a matrix for easy comparisons
probs <- cbind(lasso_probs,
               net_probs,
               ridge_probs)

colnames(probs) <- c('Lasso','Net','Ridge')
round(probs, 5)

coefs <- cbind(coef(cv.lasso),
               coef(cv.net),
               coef(cv.ridge))

colnames(coefs) <- c('L1','Net','L2')
round(coefs, 5)

# Calculate the training misclassification rates (type = 'class' gives the prediction)
training.errors <- c(
mean((predict(cv.lasso, xtrain, type = 'class', s = cv.lasso$lambda.min) != y)),
mean((predict(cv.net, xtrain, type = 'class', s = cv.net$lambda.min) != y)),
mean((predict(cv.ridge, xtrain, type = 'class', s = cv.ridge$lambda.min) != y))
)

names(training.errors) <- c('Lasso','Net','Ridge')

# Print the lambda values for the fitted models above
tuning_params <- c(cv.lasso$lambda.min, cv.net$lambda.min, cv.ridge$lambda.min)
names(tuning_params) <- c('Lasso','Net','Ridge')

# Plot showing the varying training errors given values of lambda
plot(cv.lasso)
plot(cv.net)
plot(cv.ridge)

# Compile fitted models into a list
models <- list(cv.lasso,
               cv.net,
               cv.ridge)

# ====================================================================================
# Estimate the test error for each fitted model using 10-fold CV
# ====================================================================================

# Initialize vectors to store the CV errors for each fold
lasso.fold.errors <- rep(0,k)
net.fold.errors <- rep(0,k)
ridge.fold.errors <- rep(0,k)

# K-fold cross validation loop
for (j in 1:k) {
  
  # Lasso logistic regression
  lasso <- glmnet(xtrain[folds != j,], y[folds != j],
                  family = "binomial", alpha = 1, lambda = cv.lasso$lambda.min)
      
  lasso.pred <- predict(lasso, newx = xtrain[folds == j,],
                      s = cv.lasso$lambda.min, type = 'class')
      
  lasso.fold.errors[j] <- sum(lasso.pred != y[folds==j])
  lasso.cv.error <- sum(lasso.fold.errors)/n 

  # Elastic net logistic regression
  net <- glmnet(xtrain[folds != j,], y[folds != j],
                family = "binomial", alpha = filter, lambda = cv.net$lambda.min)
  
  net.pred <- predict(net, newx = xtrain[folds == j,],
                      s = cv.net$lambda.min, type = 'class')
  
  net.fold.errors[j] <- sum(net.pred != y[folds==j])
  net.cv.error <- sum(net.fold.errors)/n

  # Ridge logistic regression
  ridge <- glmnet(xtrain[folds != j,], y[folds != j],
                  family = "binomial", alpha = 0, lambda = cv.ridge$lambda.min)
  
  ridge.pred <- predict(ridge, newx = xtrain[folds == j,],
                        s = cv.ridge$lambda.min, type = 'class')
  
  ridge.fold.errors[j] <- sum(ridge.pred != y[folds==j])
  ridge.cv.error <- sum(ridge.fold.errors)/n
  
}

# Combine the estimated test errors into a vector
cv.errors <- c(lasso.cv.error, net.cv.error, ridge.cv.error)
names(cv.errors) <- c('Lasso','Net','Ridge')

# Combine the training and test errors together for comparison
errors_matrix <- cbind(training.errors, cv.errors)
colnames(errors_matrix) <- c("Training Error","CV Error")
errors_matrix
     