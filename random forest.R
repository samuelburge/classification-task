# Load the necessary packages
require(MASS)
require(class)
require(tree)
require(tidyverse)
require(randomForest)
require(gbm)

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

# Fit a random forest to the data
fit <- randomForest(y ~ . ,data = train_set, ntree = 10000)
fit$err.rate[10000]

# Test the random forest performance on a test set (might not be necessary)
pred <- predict(fit, test_set[, 2:501])
# CHECK THIS!!! misclass_rate <- sum(test_set[, 1] == pred) / length(test_set)

oob.err <- rep(0, 500)
test.err <- rep(0, 500)
  
for(mtry in 1:500){
  fit <- randomForest(y ~ . ,data = train_set, mtry = mtry, ntree = 400)
  oob.err[mtry] <- fit$mse[400]
  pred <- predict(fit, Boston[-train, ])
  test.err[mtry] <- mean((Boston[-train, ]$medv-pred)^2)
}


