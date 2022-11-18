# Determine correlation between the different variables inside the dataset
# Create data Partition and classify flowers as either Setosa|Verginica|Versicolor using the SVMPoly model
# Determine what factors affect the prediction performance the most?

# Clearing the Global Env
rm(list=ls())

# Installing the required libraries 
install.packages("RCurl")
install.packages("mosaic")
install.packages("skimr")
install.packages("hrbrthemes")
install.packages("caret")
install.packages("DataExplorer")

# Loading the required libraries
library(datasets)
library(RCurl)
library(mosaic)
library(skimr)
library(ggplot2)
library(dplyr)
library(hrbrthemes)
library(caret)
library(DataExplorer)
library(kernlab)

# Loading the dataset
data <- datasets::iris; data

# Viewing the dataset in a spreadsheet format
View(data)

# Generating summary statistics for the dataset
summary(data)
head(data)
names(data)
str(data)
table(data$Species)
nrow(data)
ncol(data)
n_distinct(data)

# Checking for missing values
sum(is.na(data)) # No missing values in the dataset

# Generating a comprehensive summary for the dataset
skim(data)
data %>% 
  group_by(Species) %>% 
  skim()

# Checking for Correlations
corr <- round(cor(data[, 1:4]), 2); corr

# Data viz
dev.off()
plot(data)

# Correlation heat map
dev.off()
DataExplorer::plot_correlation(data)

# Box distributions of Sepal and Petal length and width
dev.off()
par(mar=c(7, 7, 7, 7))
boxplot(data,las=2)

# Boxplot distributions of Sepal length and Sepal Width as a function of Species 
dev.off()
arr <- c('Sepal.Length', 'Sepal.Width')
plot <- function (arr) {
  ggplot(mapping= aes_string(x="Species", y=arr), data= data )+
    geom_boxplot() 
}

lapply(arr, FUN = plot)

# Boxplot distributions of Petal length and Petal Width as a function of Species 
dev.off()
arr2 <- c("Petal.Length", "Petal.Width")

plot2 <- function(arr2){
  ggplot(data= data, mapping = aes_string(x= "Species", y= arr2)) +
    geom_boxplot()
}

lapply(arr2, FUN= plot2)

# Overlay Feature Plot
dev.off()
ftplt <- caret::featurePlot(x= data[, 1:4], 
                            y= data$Species, 
                            plot= "box", 
                            scales= list(x= list(relation= "free"), 
                                         y= list(relation= "free"))); ftplt

# Histogram distributions of Petal length and Sepal Width as a function of Species 
# Side-by-side distribution visualization of Sepal attr
dev.off()
par(mfrow= c(1, 2))
hist(x= data$Sepal.Length, col="orange",
     xlab= "Sepal Length (cm)", main= "Distribution of Sepal Length")
hist(x= data$Sepal.Width, col="orange",
     xlab= "Sepal Width (cm)", main= "Distribution of Sepal Width")
# Side-by-side distribution visualization of Petal attr
dev.off()
par(mfrow= c(1, 2))
hist(x= data$Petal.Length, col="orange",
     xlab= "Sepal Length (cm)", main= "Distribution of Petal Length", ylim=c(0, 40))
hist(x= data$Petal.Width, col="orange",
     xlab= "Sepal Width (cm)", main= "Distribution of Petal Width")
# Overlay faceted plot
dev.off()
par(mfrow= c(2, 2))
hist(x= data$Sepal.Length, col="orange",
     xlab= "Sepal Length (cm)", main= "Distribution of Sepal Length")
hist(x= data$Sepal.Width, col="orange",
     xlab= "Sepal Width (cm)", main= "Distribution of Sepal Width")
hist(x= data$Petal.Length, col="orange",
     xlab= "Sepal Length (cm)", main= "Distribution of Petal Length", ylim=c(0, 40))
hist(x= data$Petal.Width, col="orange",
     xlab= "Sepal Width (cm)", main= "Distribution of Petal Width")

# Setting seed for reproducibility
set.seed(100)

# Generating a stratified random split of the dataset
trainIndex <- createDataPartition(data$Species, p = 0.8, list= F)
# Training Set
train <- data[trainIndex, ]
# Testing Set
test <- data[-trainIndex, ]

# SVMPoly Model
# Building Training Model
model.train <- train(Species~., data= train,
                     method= "svmPoly", 
                     preProcess= c("scale", "center"), 
                     trControl= trainControl(method= "none"), 
                     tuneGrid= data.frame(degree= 1, scale= 1, C= 1)); model.train

model.cv <- train(Species ~., data= train, 
                  preProcess= c("scale", "center"), 
                  method= "svmPoly", 
                  trControl= trainControl(method= "cv", number = 10), 
                  tuneGrid= data.frame(degree=1, scale=1, C=1)); model.cv

# Applying the models for prediction
# Applying model.train to predict the training set
predict.training <- predict(model.train, train)
# Applying model.train to predict the testing set
predict.testing <- predict(model.train, test)
# Cross Validation
predict.cv <- predict(model.cv, train)

# Guaging model performance
(predict.training_confusionMatrix <- confusionMatrix(predict.training, train$Species))
(predict.testing_confusionMatrix <- confusionMatrix(predict.testing, test$Species))
(predict.cv_confusionMatrix <- confusionMatrix(predict.cv, train$Species))

# Checking for feature importance
ft.importance <- varImp(model.train); ft.importance
graphics::plot(ft.importance, col= "black")

