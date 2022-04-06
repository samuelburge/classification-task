# Load the necessary packages
library(MASS)
library(class)
library(tree)
library(randomForest)
library(gbm)

# Set working directory and import the data file
setwd("C:\\Users\\SamBu\\Desktop\\STAT 639")
load("class_data.RData")
dataset <- data.frame(y, x) # Do not convert y to factor; crashed the gbm() function

# Partition the data set into training and test sets
split <- sample(c(TRUE,FALSE), nrow(dataset), replace = TRUE, prob = c(0.8, 0.2))
train_set <- dataset[split, ]
test_set <- dataset[!split, ]

# Fit boosted trees to the data with shrinkage = 0.01
boost_fit <- gbm(y ~ ., data = train_set,
                 distribution = "bernoulli",
                 n.trees = 10000,
                 shrinkage = 0.01,
                 interaction.depth = 2,
                 cv.folds = 10,
                 verbose = TRUE)

# Cross-validation to select the best number of boosts(trees)
n_trees_cv <- gbm.perf(boost_fit, method = 'cv', plot.it = T)

gbmpred <- predict(boost_fit, newdata = test_set[,-1],
                   n.trees = n_trees_cv, type = 'response')

# Calculate test misclassification rate assuming threshold 0.5
sum((gbmpred >= 0.5) != test_set[ ,1]) / nrow(test_set)


# Fit boosted trees to the data with shrinkage = 0.001
boost_fit_001 <- gbm(y ~ ., data = train_set,
                 distribution = "bernoulli",
                 n.trees = 10000,
                 shrinkage=0.001,
                 interaction.depth = 2,
                 cv.fold = 10)

n_trees_cv_001 <- gbm.perf(boost_fit_001, method = 'cv')

gbmpred <- predict(boost_fit_001, newdata = test_est,
                   n.trees = n_trees_cv_001)

# Calculate the test misclassification rate
gbmpred_001 <- predict(boost_fit_001, newdata = test_set[,-1],
                   n.trees = n_trees_cv_001, type = 'response')

# Calculate test misclassification rate assuming threshold 0.5
sum((gbmpred_001 >= 0.5) != test_set[ ,1]) / nrow(test_set)
