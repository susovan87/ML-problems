##### Import training data set #####
training_data_url <- "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train <- read.csv(url(training_data_url))

test_data_url <- "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test <- read.csv(url(test_data_url))


##### Re-engineering dataset #####
reEngineeringDS <- function(dataSet){
  dataSet$family_size <- dataSet$SibSp + dataSet$Parch + 1
  dataSet$Name <- as.character(dataSet$Name)
  dataSet$Title <- sapply(dataSet$Name, FUN=function(x){trimws(strsplit(x, "[.,]")[[1]][2])})
  #dataSet$Title <- factor(dataSet$Title)
  
  return(dataSet)
}



##### Fill missing data #####
library(rpart)
fillMissingData <- function(dataSet){
  dataSet$Embarked[dataSet$Embarked == ""] <- "S"
  dataSet$Embarked <- factor(dataSet$Embarked)
  dataSet$Fare[dataSet$Fare==0] <- median(dataSet$Fare[dataSet$Fare!=0])
  
  # Prediction of a passengers Age using the other variables and a decision tree model. 
  predicted_age <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + family_size,
                         data = dataSet[!is.na(dataSet$Age),], method = "anova")
  dataSet$Age[is.na(dataSet$Age)] <- predict(predicted_age, dataSet[is.na(dataSet$Age),])

  return(dataSet)
}


##### split training data into train batch and test batch #####
library("caret")
set.seed(123)
training_rows <- createDataPartition(train$Survived, p = 0.8, list = FALSE)
train_batch <- train[training_rows, ]
test_batch <- train[-training_rows, ]


##### Build the decision tree #####
decision_tree_1 <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + family_size + Title,
                         data = train_batch,
                         method = "class",
                         control = rpart.control(minsplit = 2, cp = 0))

test_batch$prediction1 <- predict(decision_tree_1, test_batch, type = "class")
accuracy1 <- sum(test_batch$Survived == test_batch$prediction1, na.rm=TRUE)/nrow(test_batch)


##### Plot fancy tree #####
library(rattle)
library(rpart.plot)
library(RColorBrewer)

fancyRpartPlot(decision_tree_1)


##### Using Random Forest Analysis #####
library(randomForest)
my_forest <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title, 
                          train_batch, 
                          importance=TRUE, 
                          ntree=1000)
# Make your prediction using the test set
test_batch$prediction2 <- predict(my_forest, test_batch)
accuracy2 <- sum(test_batch$Survived == test_batch$prediction2, na.rm=TRUE)/nrow(test_batch)


##### Important variables #####
varImpPlot(my_forest)


##### Save solution #####
generateSolution <- function(passengerIds, prediction){
  # Finish the data.frame() call
  solution <- data.frame(PassengerId = passengerIds, Survived = prediction)
  
  # Use nrow() on solution
  nrow(solution)
  
  # Finish the write.csv() call
  write.csv(solution, file = "solution.csv", row.names = FALSE)
}


##### Final solution #####
test <- reEngineeringDS(test)
test <- fillMissingData(test)
test_prediction <- predict(my_forest, test)
generateSolution(test$PassengerId, test_prediction)

dirname(parent.frame(2)$ofile)
