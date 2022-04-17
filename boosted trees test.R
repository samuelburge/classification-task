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

#Establish Possible Parameters

interactionDepthVector = c(1, 2, 3, 4, 5)       # depth of the trees
shrinkageVector = seq(0.001, 0.01, 0.005)         # learning rate
bagFractionVector = seq(from = 0.2, to = 0.7, by = 0.1)          # number of obs. to use in training
bootStraps <- 6

#Initialize Results DataFrames

names<- c('bootStrapNum','interactionDepth','shrinkage','bagFraction','totalTrees','validationAccuracy','bootstrapID')
tempResultsDataFrame <- data.frame(matrix(ncol = length(names), nrow = 0)) 
colnames(tempResultsDataFrame) <- names
resultsDataFrame <- tempResultsDataFrame

bootstrapID <- 0

# ============================================================================
# Parameter Tuning CV (Model Selection)
# ============================================================================

    for (i in 1:length(interactionDepthVector)){
        for (j in 1:length(shrinkageVector)){
            for (k in 1:length(bagFractionVector)){
    
bootstrapID <- bootstrapID + 1

for (b in 1:bootStraps){
  
  picked = sample(seq_len(nrow(dataset)),size = sample_size)
  outerSplit<-dataset[picked,]
  innerSplit<-dataset[-picked,]
  
          boost.inner=gbm(y~.,data=innerSplit,
                          distribution="bernoulli",
                          n.trees=10000,
                          shrinkage=shrinkageVector[j],
                          interaction.depth=interactionDepthVector[i],
                          bag.fraction=bagFractionVector[k],
                          cv.fold=20)
          boost.inner.cvtrees<-gbm.perf(boost.inner, method = "cv")
          boost.outer.pred=predict(boost.inner,
                                   newdata=outerSplit,
                                   n.trees=boost.inner.cvtrees,
                                   type='response')
          boost.inner.pred=predict(boost.inner,
                                   newdata=outerSplit,
                                   n.trees=boost.inner.cvtrees,
                                   type='response')

          responseVector<-boost.inner.pred>=0.5
          accuracy<-(nrow(outerSplit)-sum(abs(responseVector-outerSplit$y)))/nrow(outerSplit)
          strapResult<-c(b,interactionDepthVector[i],shrinkageVector[j],
                         bagFractionVector[k],boost.inner.cvtrees,accuracy,bootstrapID)
          resultsDataFrame<-rbind(resultsDataFrame,strapResult)
          print(paste(i,j,k,b,'complete'))
}
            }}}


colnames(resultsDataFrame) <- names
resultsDataFrame

summary = resultsDataFrame %>%
  group_by(bootstrapID) %>%
  summarize(meanAccuracy = mean(validationAccuracy)) %>%
  arrange(meanAccuracy, desc=TRUE)
  print(n=60)

