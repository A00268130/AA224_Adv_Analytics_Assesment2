# Title: "Advanced Analytics Group Project Assignment 2"
# Authors: Gavin Byrne (A00267975), Luke Feeney (A00268130),Brendan Lynch (A00267986)
# Date: Nov 2019

# Install Libraries used if they are not already installed
if("tidyr" %in% rownames(installed.packages()) == FALSE) {
  install.packages("tidyr")
}
if("tibble" %in% rownames(installed.packages()) == FALSE) {
  install.packages("tibble")
}
if("dplyr" %in% rownames(installed.packages()) == FALSE) {
  install.packages("dplyr")
}
if("stats" %in% rownames(installed.packages()) == FALSE) {
  install.packages("stats")
}
if("kableExtra" %in% rownames(installed.packages()) == FALSE) {
  install.packages("kableExtra")
}
if("knitr" %in% rownames(installed.packages()) == FALSE) {
  install.packages("knitr")
}
if("data.table" %in% rownames(installed.packages()) == FALSE) {
  install.packages("data.table")
}
if("tibble" %in% rownames(installed.packages()) == FALSE) {
  install.packages("tibble")
}
if("fastDummies" %in% rownames(installed.packages()) == FALSE) {
  install.packages("fastDummies", dependencies = T)
}
if("gmodels" %in% rownames(installed.packages()) == FALSE) {
  install.packages("gmodels", dependencies = T)
}
if("psych" %in% rownames(installed.packages()) == FALSE) {
  install.packages("psych", dependencies = T)
}
# Load libraries used in the project
library(dplyr)
library(tidyr)
library(kableExtra)
library(knitr)
library(data.table)
library(tibble)
library(fastDummies)
library(class)
library(gmodels)
library(caret)
library(psych)

# Section 1 - Data Loading, Cleaning & Exploration ###############################
## Reading and preprocessing of kNN data
data_knn = read.csv("data/breast-cancer-data.csv")
colnames(data_knn) <-  c("class","age","menopause","tumor_size","inv_nodes","node_caps","deg_malig","breast","breast_quad","irradiat")
str(data_knn)
head(data_knn)
summary(data_knn)
# factorise single variable
data_knn$deg_malig = as.factor(data_knn$deg_malig)

#examining class breakdown percentages
round(prop.table(table(data_knn$class)) * 100, digits = 1)

# Regression: Import data for regression question - concrete compressive strength
df_concrete <- read.csv("data/Concrete_Data.csv")

# Regression: Rename headers (shorten the names)
colnames(df_concrete) <- c("Cement",
                           "Blast Furnace Slag",
                           "Fly Ash",
                           "Water",
                           "Superplasticizer",
                           "Coarse Aggregate",
                           "Fine Aggregate",
                           "Age",
                           "Concrete Comp. Strength")

# Regression: Summary of data
summary(df_concrete)
str(df_concrete)
head(df_concrete)

# Section 2 - Decision Trees ###############################


# Section 3 - kNN ###############################
# split into training and test data
set.seed(12)
train <-sample(nrow(data_knn), 0.7*nrow(data_knn), replace = FALSE)

# model 1
# create dummy variables for categorical/nominal attributes
data_knn_m1 = fastDummies::dummy_cols(data_knn,remove_most_frequent_dummy = T,ignore_na = T )
data_knn_m1 = data_knn_m1[,-c(2:11)]

#training and test sets
train_knn_m1 <-data_knn_m1[train,-1]
test_knn_m1 <-data_knn_m1[-train,-1]
#training and test class labels
train_knn_class_m1 <- data_knn_m1[train,1]
test_knn_class_m1 <- data_knn_m1[-train,1]

# run for multiple values of k to find most accurate and lowest p value
m1_results = data.frame(k=c(0,0),Accuracy=c(0,0), Acc_pvalue = c(0,0))
m1_models <- list()
for (k in c(1:50)) {
  knn_pred_m1 <- knn(train_knn_m1,test_knn_m1, cl=train_knn_class_m1, k=k)
  cm_m1 <- confusionMatrix(knn_pred_m1, test_knn_class_m1)
  m1_results[k,] = c(k,cm_m1$overall[1],cm_m1$overall[6])
  m1_models[[k]] = knn_pred_m1
}

# Plot of accuracy and p-value vs k chosen
par(mar = c(5,5,2,5))
plot(m1_results$k,m1_results$Accuracy, type="l", xlab="k", ylab="accuracy", col="blue",main ="kNN - model 1 - All Nominal")

par(new = T)
plot(m1_results$k,m1_results$Acc_pvalue, type="l", axes=F, xlab=NA, ylab=NA, col="red")
axis(side = 4)
mtext(side = 4, line = 3, 'p-value')
legend("topright", legend = c("accuracy", "p-value"), col=c("blue","red"), cex=0.6, lty = 1)

# select k=3 as maximum accuracy and minimum p-value
cm_m1 <- confusionMatrix(m1_models[[3]], test_knn_class_m1)
cm_m1$overall[1]

error_check = function(actual, predicted) {
  mean(actual != predicted)
}

error_check(actual= test_knn_class_m1,predicted = m1_models[[4]])

# model 2
data_knn_nom_vars <- c("menopause","node_caps","breast","breast_quad","irradiat")
data_knn_m2 = fastDummies::dummy_cols(data_knn,remove_most_frequent_dummy = T,ignore_na = T, select_columns = data_knn_nom_vars)
data_knn_ord_vars <- colnames(data_knn)[!(colnames(data_knn) %in% c(data_knn_nom_vars,"class"))]
data_knn_m2$age_n <- as.numeric(data_knn_m2$age)-1
data_knn_m2$tumor_size <- factor(data_knn_m2$tumor_size, levels = c('0-4','5-9','10-14','15-19','20-24','25-29','30-34','35-39','40-44','45-49','50-54'))
data_knn_m2$tumor_size_n <- as.numeric(data_knn_m2$tumor_size)-1
data_knn_m2$inv_nodes <- factor(data_knn_m2$inv_nodes, levels = c("0-2","3-5","6-8","9-11","12-14", "15-17","24-26"))
data_knn_m2$inv_nodes_n <- as.numeric(data_knn_m2$inv_nodes)-1
data_knn_m2$deg_malig_n <- as.numeric(data_knn_m2$deg_malig) - 1

# need to normalise these new numeric features to be between 0 and 1
normfn <- function(inlist) {
  inlist = inlist/max(inlist)
  return(inlist)
}
data_knn_m2[,names(data_knn_m2) %in% paste0(data_knn_ord_vars,"_n")] <- lapply(data_knn_m2[,names(data_knn_m2) %in% paste0(data_knn_ord_vars,"_n")], normfn)
data_knn_m2 = data_knn_m2[,-c(2:10)]

#training and test sets
train_knn_m2 <-data_knn_m2[train,-1]
test_knn_m2 <-data_knn_m2[-train,-1]
#training and test class labels
train_knn_class_m2 <- data_knn_m2[train,1]
test_knn_class_m2 <- data_knn_m2[-train,1]

# run for multiple values of k to find most accurate and lowest p value
m2_results = data.frame(k=c(0,0),Accuracy=c(0,0), Acc_pvalue = c(0,0))
m2_models <- list()
for (k in c(1:50)) {
  knn_pred_m2 <- knn(train_knn_m2,test_knn_m2, cl=train_knn_class_m2, k=k)
  cm_m2 <- confusionMatrix(knn_pred_m2, test_knn_class_m2)
  m2_results[k,] = c(k,cm_m2$overall[1],cm_m2$overall[6])
  m2_models[[k]] = knn_pred_m2
}
summary(knn_pred_m2)
# Plot of accuracy and p-value vs k chosen
par(mar = c(5,5,2,5))
plot(m2_results$k,m2_results$Accuracy, type="l", ylim=c(0.63,0.82), xlab="k", ylab="accuracy", col="blue", main ="kNN - model 2 - Ordinal")

par(new = T)
plot(m2_results$k,m2_results$Acc_pvalue, type="l", axes=F, xlab=NA, ylab=NA, col="red")
axis(side = 4)
mtext(side = 4, line = 3, 'p-value')
legend("topright", legend = c("accuracy", "p-value"), col=c("blue","red"), cex=0.6, lty = 1)

# select k=6 as maximum accuracy and minimum p-value
cm_m2 <- confusionMatrix(m2_models[[6]], test_knn_class_m2)
cm_m2$overall[1]

error_check(actual= test_knn_class_m2,predicted = m2_models[[6]])

# Section 4 - Regression ###############################
# Investigating the dependant variable
hist(df_concrete$"Concrete Comp. Strength", main='Histogram of Concrete Compressive Strength', xlab='Concrete Compressive Strength')

# Creating a scatterplot matrix to investigate the relationships between the features
pairs.panels(df_concrete[c("Cement",
                           "Blast Furnace Slag",
                           "Fly Ash",
                           "Water",
                           "Superplasticizer",
                           "Coarse Aggregate",
                           "Fine Aggregate",
                           "Age",
                           "Concrete Comp. Strength")])


# create a model on the concrete data using all features
concrete_model <- lm(df_concrete$"Concrete Comp. Strength" ~ Cement + `Blast Furnace Slag` + 
                       `Fly Ash` + Water + Superplasticizer + `Coarse Aggregate` + 
                       `Fine Aggregate` + Age, data = df_concrete)

# Use stepwise regression to check if any variables can be possibly removed from the model
step(concrete_model)

# Regression coefficents
concrete_model

# Evaluate model perfomance using all the features
summary(concrete_model)

# Plots of model generated
par(mfrow=c(2,2))
plot(concrete_model)

# Confidence intervals
confint(concrete_model)

# Adding interaction term to generate new model (testing for interaction between cement and age)
concrete_model_CA <- lm(df_concrete$"Concrete Comp. Strength" ~ Cement + `Blast Furnace Slag` + 
                          `Fly Ash` + Water + Superplasticizer + `Coarse Aggregate` + 
                          `Fine Aggregate` + Age + Cement*Age, data = df_concrete)

# Evaluate model perfomance with new feature added
summary(concrete_model_CA)

# Plots of model generated
par(mfrow=c(2,2))
plot(concrete_model_CA)

# Confidence intervals
confint(concrete_model_CA)