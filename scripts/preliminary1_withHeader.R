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

save(claims, file = 'data/claims-clean-example.RData')

#----------------------------------------------------
parse_fn <- function(.html){
  read_html(.html) %>%
    html_elements('p,h1,h2,h3,h4,h5,h6') %>%
    html_text2() %>%
    str_c(collapse = ' ') %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all('\'') %>%
    str_replace_all(paste(c('\n', 
                            '[[:punct:]]', 
                            'nbsp', 
                            '[[:digit:]]', 
                            '[[:symbol:]]'),
                          collapse = '|'), ' ') %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
}
#----------------------------------------------------


# function to apply to claims data
parse_data_header <- function(.df){
  out <- .df %>%
    filter(str_detect(text_tmp, '<!')) %>%
    rowwise() %>%
    mutate(text_clean = parse_fn(text_tmp)) %>%
    unnest(text_clean) 
  return(out)
}

nlp_fn <- function(parse_data_header.out){
  out <- parse_data_header.out %>% 
    unnest_tokens(output = token, 
                  input = text_clean, 
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, bclass, token.lem, name = 'n') %>%
    bind_tf_idf(term = token.lem, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass'),
                names_from = 'token.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}
#----------------------------------------------------
claims_h <- claims_raw %>%
  parse_data_header()

cleaned_claims_h <- nlp_fn(claims_h)

# partition data
set.seed(102722)
partitions_h <- cleaned_claims_h %>% 
  initial_split(prop = 0.8)

#colnames(testing(partitions))
# separate DTM from labels
test_dtm <- testing(partitions_h) %>%
  select(-.id, -bclass)
test_labels <- testing(partitions_h) %>%
  select(.id, bclass)

# training set
train_dtm <- training(partitions_h) %>%
  select(-.id, -bclass) 

train_labels <- training(partitions_h) %>%
  select(.id, bclass)

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
#summary(fit)

test_dtm_projected <- reproject_fn(.dtm = test_dtm, proj_out)

test <- test_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(test_dtm_projected)


test_predictions <- predict(fit, newdata = test, type = "response")

predicted_classes <- ifelse(test_predictions > 0.5, "relevant", "irrelevant")

confusion_matrix <- table(Predicted = predicted_classes, Actual = test$bclass)
print(confusion_matrix)

accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Test Accuracy with Header Informations:", accuracy, "\n")
