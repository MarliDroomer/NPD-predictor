# Predicting the Next Purchase Date for an Individual Customer using Machine Learning.
The Next Purchase Date (NPD) predictor is designed to predict the next purchase date for an individual customer and is based in the instacart dataset. This page contains the resources for the NPD predictor.

## Abstract
We live in a world that is rapidly changing when it comes to technology. Gathering a customer’s information becomes easier as companies have loyalty programs that
track the customer’s purchasing behaviour. We live in an era where search engines suggest your next word, online shopping is no longer scary, and people order a
ride by means of an application. The fact is that technology is evolving, and gathering information from customers is becoming easier. Given this change,
the questions, however, are: How do companies use this information to gain a competitive advantage? Do they use this information to benefit the customer?
How can a company use customer information to give each individual a unique experience?

A research study was conducted to determine if an individual customer’s next purchase date for specific products can be predicted by means of machine learning.
The focus was on fast-moving consumer goods in retail. This next purchase date can then be used to individualise marketing to customers, which benefits the company
and the customer. In this study, the customer’s purchase history is used to train machine learning models. These models are then used to predict the next purchase
date for a customer-product pair. The different machine learning models that are used are recurrent neural networks, linear regression, extreme gradient boosting
and an artificial neural network. Combination approaches are also investigated, and the models are compared by the absolute error, in days, that the model predicts
from the target variable.

The artificial neural network model performed the best, predicting 31.8% of the dataset with an absolute error of less than one day, and 55% of the dataset with
an absolute error of less than three days. The application of the artificial neural network as the Next Purchase Date Predictor is also demonstrated and shows how
individualised marketing can be done using the Next Purchase Date Predictor. The encouraging results of the Next Purchase Date Predictor showed that machine
learning could be used to predict the next purchase date for an individual customer.

## Required Software
The software is written Python 3.7 in depends few libraries. The operations are quite memory intensive and we recommend having atleast 16Gb of memory available. We developed the software with the following versions, and recommended installing the dependencies through Anaconda Python:
- numpy 1.18.1
- tensorflow 2.1.0
- matplotlib 3.2.1
- matplotlib-venn 0.11.5
- pandas 0.25.1
- pytorch 1.4.0
- xgboost 0.9
- jupyter 1.0.0
- scikit-learn 0.21.3

## Dataset
The NPD predictor is designed for the Instacart Dataset, which was released for a Kaggle competition. The dataset along with information regarding the data is available at https://www.kaggle.com/c/instacart-market-basket-analysis. For the NPD predictor to work, the unzipped folder containing the dataset should be downloaded to the root folder of the repository.

## Structure
This section describes the structure of the repository
- notebooks - Contains the notebooks to create the features and train the models
- scr - contains accompanying source files for the notebooks
- features - contains the test and training datasets as pkl files
- instacart - should contain the dataset (download from kaggle)
- results - Contains the results of the training dataset

