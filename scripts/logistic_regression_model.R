# packages
library(tidyverse)
library(tidymodels)
library(modelr)
library(Matrix)
library(sparsesvd)
library(glmnet)

# path to activity files on repo
url <- 'https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/activities/data/'
load("data/claims-raw.RData")
source(paste(url, 'projection-functions.R', sep = ''))
source('scripts/preprocessing.R')

# read in data
claims <- claims_raw %>%
  parse_data()
# preview
claims

# partition data
set.seed(102722)
partitions <- claims %>% 
  initial_split(prop = 0.8)


#----------------------------------------------------
# separate DTM from labels
test_dtm <- testing(partitions) %>%
  select(-.id, -bclass, -mclass)
test_labels <- testing(partitions) %>%
  select(.id, bclass, mclass)

# training set
train_dtm <- training(partitions) %>%
  select(-.id, -bclass, -mclass) 

train_labels <- training(partitions) %>%
  select(.id, bclass, mclass)

proj_out <- projection_fn(.dtm = train_dtm, .prop = 0.7)
train_dtm_projected <- proj_out$data

train <- train_labels %>%
  transmute(bclass = factor(bclass)) %>%  # Convert bclass to a factor
  bind_cols(train_dtm_projected)

# Fit the logistic regression model
fit <- glm(
  bclass ~ .,  
  data = train, 
  family = "binomial"  
)
summary(fit)

test_dtm_projected <- reproject_fn(.dtm = test_dtm, proj_out)

test <- test_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(test_dtm_projected)


test_predictions <- predict(fit, newdata = test, type = "response")

predicted_classes <- ifelse(test_predictions > 0.5, "relevant", "irrelevant")

confusion_matrix <- table(Predicted = predicted_classes, Actual = test$bclass)
print(confusion_matrix)

accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Test Accuracy:", accuracy, "\n")
