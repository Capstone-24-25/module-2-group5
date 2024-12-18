[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/9zKiFNZz)
# module2-claims

This is a template repository for the module 2 group assignment for PSTAT197A. The assignment objective is to develop a predictive model for webpage classification based on scraping text content.

## Repository content

-   `data` contains several `.RData` files:

    -   `claims-raw.RData` contains raw HTML and class labels for 2,165 webpages

    -   `claims-test.RData` contains raw HTML for 929 unlabeled webpages

    -   `claims-clean.RData` contains cleaned data for use in training; file generated by `scripts/nlp-model-development.R` and should be replaced as preprocessing pipeline is updated

-   `scripts` contains some starter R scripts

    -   `preprocessing.R` contains *functions* for quickly ingesting raw HTML and outputting analysis-ready data; this file is intended to be sourced

    -   `nlp-model-development.R` shows example codes for preprocessing data and training a neural network and exporting the model

    -   `prediction.R` imports a trained model and generates predictions

-   `results` contains example technical outputs: a trained model and a set of predictions

-   `writeups` contains templates for written summaries

## Assignment instructions

Analytical tasks for this assignment fall under two categories:

-   The ***preliminary tasks*** extend the in-class analysis in various ways; these are more exploratory in nature.

-   The ***primary task*** is to develop a predictive model. This is open-ended but should be informed to some degree by the in-class analysis and the preliminary tasks.

We have provided a collection of unlabeled pages for you to classify. Once projects are complete we'll evaluate the accuracy of your model using the actual labels.

### Preliminary tasks

1.  Augment the HTML scraping strategy so that header information is captured in addition to paragraph content. Are binary class predictions improved using logistic principal component regression?

2.  Perform a secondary tokenization of the data to obtain bigrams. Fit a logistic principal component regression model to the word-tokenized data, and then input the predicted log-odds-ratios together with some number of principal components of the bigram-tokenized data to a *second* logistic regression model. Based on the results, does it seem like the bigrams capture additional information about the claims status of a page?

3.  OPTIONAL: Build and train a neural network model. Experiment with architectures and training configurations until you find a model that performs at least as well as principal component logistic regression from task 1. Quantify the predictive accuracy.

### Primary task

1.  Develop the best predictive model you can for (a) binary classification and also for (b) the multi-class setting. You can use any preprocessing and any modeling technique you like, including ones not discussed in class (consider exploring RNN or SVM). Export a deployable copy of each model and a set of predictions from each model on the test data `claims-test.RData`. Predictions should be formatted as a single data frame named `pred_df` with columns

    -    `.id` containing the URL ID

    -   `bclass.pred` containing the predicted label for the binary class setting

    -   `mclass.pred` containing the predicted label for the multiclass setting

### Deliverables

If this were a technical project, you might be expected to provide a *brief* executive summary of your work and a deployable version of your predictive model that could be easily used by another team or individual. In addition, you might use your model to classify new pages. Deliverables are designed to emulate these project outputs.

-   ***Deliverable 1:*** write a 1-page summary of findings for tasks 1-3; store the rendered document in the `writeups` directory.

-   ***Deliverable 2:*** write a 1-page summary of methods and findings for task 4. This should describe text preprocessing, your predictive models, and estimated prediction accuracy for each model. Store the rendered document in the `writeups` directory.

-   ***Deliverable 3:*** a set of multiclass and binary predictions on the test data, stored as `results/preds-group[N].RData` .

-   ***Deliverable 4:*** deployable copies of your fitted predictive models, stored in the `results` directory, with a short script illustrating their use to generate predictions on the test data, stored in the `scripts` directory.

*Your written summaries should be **very** high-level with a minimum of technical detail. The audience for your writeups is a project mentor or supervisor: assume the reader is familiar with the project and has a working knowledge of the methods you've used. The aim of each summary is to convey the task(s) and result(s) as simply as possible in a linear fashion: task 1, result 1; task 2, result 2; and so on. Note that your results for each task are simply a predictive accuracy (one or a few numbers) and a comment on the accuracy (generally, comparison with another result).*
