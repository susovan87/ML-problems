##### Preparing Your R Workspace #####

# Clean workspace
rm(list=ls())

# Set current working directory
setwd('/Users/susovan/Documents/github/ML-problems/kaggle-titanic')

# Load training data
training.data.file <- "../data/train.csv"
base.train <- read.csv(training.data.file)

# Load test data
test.data.file <- "../data/test.csv"
base.test <- read.csv(test.data.file)

