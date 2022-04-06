# Load the necessary packages
require(MASS)
require(class)
require(tree)
require(tidyverse)
require(naivebayes)

# Set working directory and import the data file
setwd("C:\\Users\\SamBu\\Desktop\\STAT 639")
load("class_data.RData")
dataset <- data.frame(y = as.factor(y), x)
tail(dataset)[ ,1:10] # take a peek to make sure it is right
str(dataset)

# Partition the data set into training and test sets
split <- sample(c(TRUE,FALSE), nrow(dataset), replace = TRUE, prob = c(0.8, 0.2))
train_set <- dataset[split, ]
test_set <- dataset[!split, ]

# Define the folds for k-fold cross validation
train_size <- nrow(train_set)
k <- 10
folds <- sample(1:k, train_size, replace = TRUE)
cv_errors <- rep(0, 10)

# Perform k-fold cross validation
for (i in 1:k) {
  
  cv_train <- train_set[folds != i, ]  
  cv_validate <- train_set[folds == i, ]
  
  fit <- naive_bayes(y ~ ., data = cv_train, usekernel = TRUE)
  train_pred <- predict(fit, cv_train[, -1])
  cv_train_error <- sum(train_pred == cv_train[, 1]) / nrow(cv_train)
  
  cv_pred <- predict(fit, cv_validate[ ,-1])
  cv_errors[i] <- sum(cv_pred == cv_validate[ , 1]) / length(cv_validate[ ,1])
  
}

# Print results of each fold and calculate total CV error
cv_errors
cv_test_error <- mean(cv_errors)

# Calculate test error using the test set
fit <- naive_bayes(y ~ ., data = train_set, usekernel = TRUE)
test_pred <- predict(fit, test_set[, -1])

# Outputs the probability of each class used to make prediction and combines in a table
test_pred_prob <- predict(fit, test_set[, -1], type = 'prob')
cbind(test_pred, round(test_pred_prob, 4))

# Calculate the test error
test_error <- sum(test_pred == test_set[, 1]) / nrow(test_set)
test_error # yeah this is trash


