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

# Partition the data set into training and test sets
split <- sample(c(TRUE,FALSE), nrow(dataset), replace = TRUE, prob = c(0.8, 0.2))
train_set <- dataset[split, ]
test_set <- dataset[!split, ]

# Fit gradient-boosted trees to the data
boost.fit <- gbm(y ~ ., data = train_set,
                 distribution = "bernoulli",
                 n.trees = 10,
                 shrinkage=0.01,
                 interaction.depth = 4,
                 cv.fold = 10)

boost.fit <- gbm(y ~ ., data = train_set,
                 distribution = "bernoulli", n.trees = 10,
                 shrinkage=0.001,
                 interaction.depth = 4,
                 cv.fold = 10)

n.trees.cv <- gbm.perf(boost.fit, method = "cv", oobag.curve = TRUE, overlay = TRUE)

gbmpred <- predict(boost.fit, newdata = test_est,
                   n.trees = n.trees.cv)

# Calculate the test error
