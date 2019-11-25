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
if("psych" %in% rownames(installed.packages()) == FALSE) {
  install.packages("psych")
}


# Load libraries used in the project
library(dplyr)
library(tidyr)
library(kableExtra)
library(knitr)
library(psych)



# Section 1 - Data Loading, Cleaning & Exploration ###############################

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

# Regression coefficents
concrete_model

# Evaluate model perfomance using all the features
summary(concrete_model)

# Plots of model generated
par(mfrow=c(2,2))
plot(concrete_model)

# Confidence intervals
confint(concrete_model)

# Use stepwise regression to check if model can be improved
step(concrete_model)

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
