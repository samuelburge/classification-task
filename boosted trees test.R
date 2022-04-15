# Load the necessary packages
library(MASS)
library(class)
library(tree)
library(randomForest)
library(gbm)
library(caret)
library(dplyr)

# Set working directory and import the data file
#setwd("C:\\Users\\SamBu\\Desktop\\STAT 639")
setwd("C:\\Users\\CRAUST~1\\DOCUME~1\\DATAMI~1\\DATA-M~1")
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

#Create Two Cross Validation Sets for Inner and Outer Processes
sample_size = floor(0.5*nrow(dataset))
picked = sample(seq_len(nrow(dataset)),size = sample_size)
outerSplit<-dataset[picked,]
innerSplit<-dataset[-picked,]

# ============================================================================
# Parameter Tuning CV (Model Selection)
# ============================================================================

#Establish Possible Parameters

#interactionDepthVector = c(1, 2, 3)       # depth of the trees
#shrinkageVector = c(0.001, 0.01, 0.005)         # learning rate
#bagFractionVector = c(0.3, 0.5, 0.75)          # number of obs. to use in training
   
          boost.inner=gbm(y~.,data=innerSplit,
                          distribution="bernoulli",
                          n.trees=5000,
                          shrinkage=0.001,
                          interaction.depth=1,
                          cv.fold=10)
          boost.inner.cvtrees<-gbm.perf(boost.inner, method = "cv")
          boost.outer.pred=predict(boost.inner,newdata=outerSplit,n.trees=boost.inner.cvtrees,type="response" )

sum((boost.outer.pred>=0.5) != outerSplit$y)/nrow(outerSplit)

  
  
  
  
  
  
  






