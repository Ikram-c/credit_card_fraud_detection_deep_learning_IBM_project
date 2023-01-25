# Credit Card Fraud Detection Deep Learning (part of IBM ML certification)


## Overview of project
The aim of the project was to use Deep Learning, specifically autoencoding, to help classify which credit card transactions are fraudulent. The data used for this project is the "Credit Card Fraud Detection" dataset (https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) which contains transactions made using credit cards in September 2013 by European cardholders over a period of two days. 

The dataset was considerably larger than most portfolio projects as it contained 284,807 rows and 31 columns, furthemore, the data was imbalanced with only 492 fraud transactions occuring out of 284,807 total transactions. It was for this reason that traditional EDA methods would not be viable and the dataset highly benefitted from Deep Learning as it can discover complex patterns within the data to help distintinguish between minority and majority classes in the dataset.


For the EDA process the data was checked for nulls and its distribution was checked through the ***Anderson-Darling*** test, the ***Shapiro-Wilk*** test and the ***D'Agostino's K^2*** test as histograms were not viable. The values were then scaled using the ***MinMaxScaler***, the ***StandardScaler*** and using ***log_10*** with each normalization method being tested using the Z score, with the most optimal model being used. A ***t-SNE*** plot was used to vizualise data using a sample of 2000 non fraudulent cases.

The hyperparameters of the autoencoder were modified to try to improve the accuracy of the model resulting in effectively 8 different deep learning models which were used in conjunction with 3 supervised classification models. 
The hyperparmeters for the autoencoders that were used were:

-***MSE loss function***

-***Binary crossentropy loss function***

-***SGD optimizer***

-***Adam optimizer***

-***sigmoid activation function***

-***tanh activation function***

These hyperparemeters were used in various combinations with each other and were then used with the following supervised clustering algorithms:

-***Logistic Regression***

-***KNeighbors Classifier***

-***Decision Tree Classifier***

The performance of the models were then assessed using their F1 score with the top 3 models having an accuracy of:

0.99 for the Sigmoid SGD MSE autoencoder
0.96 for the Sigmoid SGD binarycrossentropy autoencoder
0.96 for the tanh Adam MSE autoencoder

(values have been rounded to 2 d.p)


See Notebook for detailed breakdown.


