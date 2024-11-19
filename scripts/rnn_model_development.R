
library(keras)
library(tensorflow)
library(tidyverse)

# Load the preprocessed data
load("data/claims-clean-example.RData")

# Split data
set.seed(110122)
partitions <- claims_clean %>% initial_split(prop = 0.8)
train_data <- training(partitions)
test_data <- testing(partitions)

# Prepare text and labels for training
train_text <- train_data %>% pull(text_clean)
train_labels <- train_data %>% pull(mclass)  

test_text <- test_data %>% pull(text_clean)
test_labels <- test_data %>% pull(mclass) 

# 
num_classes <- length(unique(train_labels))  
train_labels <- to_categorical(as.numeric(train_labels) - 1, num_classes = num_classes)
test_labels <- to_categorical(as.numeric(test_labels) - 1, num_classes = num_classes)

# Tokenize 
tokenizer <- text_tokenizer(num_words = 10000)  
tokenizer %>% fit_text_tokenizer(train_text)

train_sequences <- texts_to_sequences(tokenizer, train_text)
test_sequences <- texts_to_sequences(tokenizer, test_text)

maxlen <- 512  
train_padded <- pad_sequences(train_sequences, maxlen = maxlen)
test_padded <- pad_sequences(test_sequences, maxlen = maxlen)

# RNN model
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 64, input_length = maxlen) %>%
  layer_simple_rnn(units = 64, return_sequences = FALSE) %>% 
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>%  
  layer_dense(units = num_classes, activation = "softmax")  


model %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",  
  metrics = c("accuracy")
)

summary(model)

# Train the model
history <- model %>% fit(
  x = train_padded,
  y = train_labels,
  validation_split = 0.2,   
  epochs = 5,
  batch_size = 32,      
  verbose = 1
)

results <- model %>% evaluate(test_padded, test_labels, verbose = 0)
cat("Test Loss:", results["loss"], "\n")
cat("Test Accuracy:", results["accuracy"], "\n")


save_model_tf(model, "results/rnn_multiclass_model")
