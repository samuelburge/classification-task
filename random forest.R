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

# Fit a random forest to the data
fit <- randomForest(y = dataset[, 1],
                    x = dataset[,-1],
                    ntree = 10000)

# Calculate the training error rate
oob_error_rate <- fit$err.rate[10000]

# Plot the misclassification rates for all (black), 0 (blue), and 1 (red)
plot(1:nrow(fit$err.rate), fit$err.rate[ ,3], type = "l", col = 'red',
     xlab = 'Trees', ylab = 'Misclassification Rate')
lines(1:nrow(fit$err.rate), fit$err.rate[ ,2], col = 'blue')
lines(1:nrow(fit$err.rate), fit$err.rate[ ,1], col = 'black')
