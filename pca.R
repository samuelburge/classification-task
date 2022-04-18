# Set working directory and import the data file
setwd("C:\\Users\\SamBu\\Desktop\\STAT 639")
load("class_data.RData")
dataset <- data.frame(y, x) # Do not convert y to factor; crashed the gbm() function


# Check for any class imbalances in the labels
table(y)

ggplot(data.frame(y), mapping = aes(x = y)) +
  geom_bar() +
  theme_classic()

# Perform PCA as part of exploratory data analysis
x_pca <- prcomp(x, scale = TRUE, center = TRUE)
names(x)

# Score vector for each observation in original data set
x_pca$x

# Principal component loading vectors
x_pca$rotation

# Calculate the proportion of variance explained
x_pca$sdev
pr.var <- (x_pca$sdev)^2
pr.var

pve <- pr.var / sum(pr.var)
pve

plot(pve, ylim = c(0,1), type = 'b',
     xlab = "Principal Component",
     ylab = "Proportion of Variance Explained")

plot(cumsum(pve), ylim = c(0,1), type = 'b',
     xlab="Principal Component",
     ylab="Cumulative Proportion of Variance Explained")

# cumsum is a function to calculate cumulative sum of each element in a vector
# see example below
a=c(1,2,8,-3)
cumsum(a)