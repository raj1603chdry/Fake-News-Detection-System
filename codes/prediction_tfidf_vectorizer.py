"""
 # @author rajchoudhary
 # @email raj.choudhary1603@gmail.com
 # @create date 2019-03-18 23:30:10
 # @modify date 2019-04-09 01:43:00
 # @desc [File for predicting the label of the news entered by the user.]
"""

# Importing the libraries
import os
import pickle


user_input = input('Enter the news to verify:\t')
print('The news entered:\t{}'.format(user_input))


# Function for making the prediction
def make_prediction(user_input):
    """Function to load the desired model and make the prediction using
    the model and displaying the label to the user.

    Parameters:
    -----------
    user_input: string
        The news the user wants to confirm.
    """
    # Loading the desired model
    final_model = pickle.load(
        open(os.path.join('../models', 
            'voting_classifier_tfidf_vectorizer.pkl'), 'rb'), 
    )

    # Making prediction on the user_input and displaying the result
    prediction = final_model.predict([user_input])
    prediction_probability = final_model.predict_proba([user_input])
    print('Predicted label:\t{}'.format(prediction))
    print('Truth probability:\t{}'.format(prediction_probability[0][1]))


if __name__ == '__main__':
    make_prediction(user_input)