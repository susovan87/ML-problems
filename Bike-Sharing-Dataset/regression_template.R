# Import dataset
dataset = read.csv('hour.csv')
summary(dataset)
dataset = dataset[3:17]
dataset = dataset[,-c(13,14)]


# Encoding categorical data
dataset$weathersit = factor(dataset$weathersit,
                            labels  = c('Clear', 'Mist', 'Light Rain', 'Heavy Rain'),
                            levels= c(1, 2, 3, 4))
 

# Split dataset into Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(12345)
split = sample.split(dataset$cnt, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)


# Fit Regression Model (Create regressor)
# dataset$Cat2 = dataset$Cat^2
# dataset$Cat3 = dataset$Cat^3
# dataset$Cat4 = dataset$Cat^4
regressor = lm(formula = cnt ~ season + yr + mnth + hr + holiday + weekday + workingday + weathersit + temp + atemp + hum + windspeed, data = training_set)
summary(regressor)

# remove `heavy rain` factor
# factor(training_set$weathersit, exclude=c(4))
regressor = lm(formula = cnt ~ season + yr + mnth + hr + holiday + weekday + workingday + factor(weathersit, exclude=c(3,4)) + temp + atemp + hum + windspeed, data = training_set)
summary(regressor)

# eliminate `mnth`
regressor = lm(formula = cnt ~ season + yr + hr + holiday + weekday + workingday + factor(weathersit, exclude=c(4)) + temp + atemp + hum + windspeed, data = training_set)
summary(regressor)

# eliminate `workingday`
regressor = lm(formula = cnt ~ season + yr + hr + holiday + weekday + factor(weathersit, exclude=c(4)) + temp + atemp + hum + windspeed, data = training_set)
summary(regressor)


# # Predict result
# y_pred = predict(regressor, data.frame(Level = 420))


# # Visualize result
# # install.packages('ggplot2')
# library(ggplot2)
# ggplot() +
#   geom_point(aes(x = dataset$Cat, y = dataset$Salary),
#              colour = 'red') +
#   geom_line(aes(x = dataset$Cat, y = predict(regressor, newdata = dataset)),
#             colour = 'blue') +
#   ggtitle('ML Study (Regression Model)') +
#   xlab('X Lable') +
#   ylab('Y Lable')


# # Visualize result in higher resolution with smoother curve
# # install.packages('ggplot2')
# library(ggplot2)
# x_grid = seq(min(dataset$cnt), max(dataset$cnt), 0.1)
# ggplot() +
#   geom_point(aes(x = dataset$Cat, y = dataset$Salary),
#              colour = 'red') +
#   geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Cat = x_grid))),
#             colour = 'blue') +
#   ggtitle('ML Study (Regression Model)') +
#   xlab('X Lable') +
#   ylab('Y Lable')
