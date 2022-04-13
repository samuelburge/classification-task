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
n <- length(dataset)
set.seed(639)

# Initialize vector to store the results of the tuning
tuning_grid_results <- data.frame(rep(0,25), rep(0,25), rep(0, 25),rep(0, 25),rep(0, 25),
                             rep(0, 25),rep(0, 25),rep(0, 25), rep(0, 25))

colnames(tuning_grid_results) <- c("outer_fold","inner_fold","shrinkage",
                                   "interaction.depth", "n.minobsinnode",
                                   "bag.fraction","optimal_trees","min_loss","misclass_rate")

# Create outer cross-validation folds
outer_K <- 5
outer_folds <- sample(1:outer_K, nrow(dataset), replace = TRUE)
outer_tr_err <- rep(0, outer_K) 
outer_val_err <- rep(0, outer_K) 

# ============================================================================
# Outer cross-validation (model assessment)
# ============================================================================

for (k in 1:3) {

  # ==========================================================================
  # Inner cross-validation (model selection)
  # ==========================================================================
  
  # Separate the data out for the inner fold
  inner_data <- dataset[outer_folds != k, ]
  inner_K <- 5
  inner_folds <- sample(1:inner_K, nrow(inner_data), replace = TRUE)
  
  # create hyper-parameter grid
  grid <- expand.grid(
    shrinkage = c(0.001, 0.01, 0.1),      # learning rate
    interaction.depth = c(2, 3, 4, 5),    # depth of the trees
    n.minobsinnode = c(5, 10, 15),        # min. obs. in a (terminal) node
    bag.fraction = c(0.65, 0.8, 1),       # number of obs. to use in training
    optimal_trees_perf = 0,
    optimal_trees = 0,                    # a place to store results
    min_misclass_rate = 0)                # a place to store results
  
  for (i in 1:3) {
    
    # Clarify the training and validation set for the inner CV fold
    tr_data <- inner_data[inner_folds != i, ]
    val_data <- inner_data[inner_folds == i, ]
    
  
      # ========================================================================
      # Grid search to select optimal tuning parameter values
      # ========================================================================
  
      for (j in 1:3) {
            
            # Randomize the data set since the train.fraction() argument takes
            # in the data in order starting from the first obs.
            tr_data <- tr_data[sample(1:nrow(tr_data), replace = FALSE), ]
            
            # Fit the model using set of tuning parameters from search grid
            boost <- gbm(y ~ .,
                             data = tr_data,
                             distribution = "bernoulli",
                             n.trees = 7500,
                             shrinkage = grid$shrinkage[j],
                             interaction.depth = grid$interaction.depth[j],
                             n.minobsinnode = grid$n.minobsinnode[j],
                             bag.fraction = grid$bag.fraction[j],
                             cv.folds = 5,
                             train.fraction = 0.75,
                             verbose = FALSE)
            
            # Compute predicted values on the training data
            boost_tr <- predict(boost,
                                newdata = tr_data[ ,-1],
                                n.trees = gbm.perf(boost),
                                type = "response")
            
            # Compute the training misclassification rate
            boost_tr_error <- sum((boost_tr >= 0.5) != tr_data[ ,1]) / nrow(tr_data)
            
            # Save the outputs for selecting the best model
            grid$optimal_trees_perf[j] <- gbm.perf(boost)
            grid$optimal_trees[j] <- which.min(boost$valid.error)
            grid$min_loss[j] <- min(boost$valid.error)
            grid$misclass_rate[j] <- boost_tr_error
            
      } # END GRID SEARCH
    
  # Select the tuning parameters that performed best
  best_perf <- grid[which.min(grid$misclass_rate), 1:5]
      
  # Re-train the model using the best tuning parameters
  best_boost <- gbm(y ~ .,
                    data = tr_data,
                    distribution = "bernoulli",
                    shrinkage = best_perf[1],
                    interaction.depth = best_perf[2],
                    n.minobsinnode = best_perf[3],
                    bag.fraction = best_perf[4],
                    n.trees = as.numeric(best_perf[5]),
                    cv.folds = 5,
                    train.fraction = 0.75,
                    verbose = FALSE)
      
  # Compute predicted values on the training data
  best_pred <- predict(best_boost,
                      newdata = val_data[ ,-1],
                      n.trees = gbm.perf(best_boost),
                      type = "response")
      
  # Compute the training misclassification rate
  best_boost_err <- sum((best_pred >= 0.5) != val_data[ ,1]) / nrow(val_data)
  
  # Save the results to a master table to keep track of everything
  tuning_grid_results[(k - 1)*5 + i, 1] <- k
  tuning_grid_results[(k - 1)*5 + i, 2] <- i
  tuning_grid_results[(k - 1)*5 + i, 3] <- best_boost$shrinkage  
  tuning_grid_results[(k - 1)*5 + i, 4] <- best_boost$interaction.depth 
  tuning_grid_results[(k - 1)*5 + i, 5] <- best_boost$n.minobsinnode
  tuning_grid_results[(k - 1)*5 + i, 6] <- best_boost$bag.fraction
  tuning_grid_results[(k - 1)*5 + i, 7] <- gbm.perf(best_boost)
  tuning_grid_results[(k - 1)*5 + i, 8] <- min(boost$valid.error)
  tuning_grid_results[(k - 1)*5 + i, 9] <- best_boost_err
  } # END INNER CV
  
  
# Fit the final model for the outer fold 
outer_boost <- gbm(y ~ .,
                  data = dataset[outer_folds != k, ],
                  distribution = "bernoulli",
                  shrinkage = 0.1,
                  interaction.depth = 4,
                  n.minobsinnode = 10,
                  bag.fraction = 1,
                  n.trees = 7000,
                  cv.folds = 5,
                  train.fraction = 0.75,
                  verbose = FALSE)
  
# Compute predicted values on the training data
outer_tr_pred <- predict(outer_boost,
                       newdata = dataset[outer_folds != k, -1],
                       n.trees = gbm.perf(outer_boost),
                       type = "response")
  
# Compute the training misclassification rate
outer_tr_err[k] <- sum((outer_tr_pred >= 0.5) != dataset[outer_folds != k, 1]) /
                        length(dataset[outer_folds != k, 1])

# Compute predicted values on the validation data
outer_val_pred <- predict(outer_boost,
                         newdata = dataset[(outer_folds == k), -1],
                         n.trees = gbm.perf(outer_boost),
                         type = "response")

# Compute the validation misclassification rate
outer_val_err[k] <- sum((outer_val_pred >= 0.5) != dataset[outer_folds == k, 1]) /
                        length(dataset[outer_folds == k, 1])

} # END OUTER CV

# Calculate the cross-validation error to assess the performance of the model
cv_score <- sum(outer_val_err)/n

# Save the workspace so I don't have to run everything again
#save.image('boosted_nested_cv.RData')





