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


# Load libraries used in the project
library(dplyr)
library(tidyr)
library(kableExtra)
library(knitr)


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
hist(df_concrete$"Concrete Comp. Strength")

# Creating a scatterplot matrix to investigate the relationships between the features

