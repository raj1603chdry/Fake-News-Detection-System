# Fake News Detection System

The proposed system was divided into stages to completely segregate the work based on the field of data mining operations like data collection, data preprocessing, feature extraction, feature selection and implementation of machine learning models for making the prediction for classifying the news into True or False and also predict the probability of the news belonging to the predicted label.

A number of machine learning models were implemented and the performance of the machine learning models were compared on the basis of metrics such as accuracy, f1 score, precision and recall. The main deciding metric for evaluating the performance of the models was chosen as f1 score that considers the tradeoff between precision and recall.

After the following machine learning models – SVM, Logistic Regression, Naïve Bayes and Random forest were trained and tuned, a Voting Classifier was implemented that combined all above mentioned models and formed an ensemble classifier that used all these classifiers to predict the label and class probability and used the soft voting method for making the final prediction.

__Proposed system steps:__

![Proposed system steps](./images/system_steps.jpg)
