"""
This module contains some functions some helper functions for our deep recommender system implementation.
"""

import pandas as pd
import numpy as np
from keras import backend
from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input, Dense, Flatten, Dropout
from keras.layers.merge import Dot, multiply, concatenate
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams
from collections import defaultdict
from sklearn.cross_validation import train_test_split

 
def rmse(y_true, y_pred):
    """
    This function calculates the RMSE between the true values and the predicted values.
    Args:
        y_true (tensor): the true ratings
        y_pred (tensor): the predicted ratings
    Returns:
        (tensor): the RMSE between y_true and y_pred
    """
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


def get_data():
    """
    This function reads in the training and the test data from the csv files
    Returns:
        (tuple): a tuple containing the training data set, the test data set as well as the maximum user and movie ids
    """
    # train set
    train = pd.read_csv("train.csv")

    # test set
    test = pd.read_csv("test.csv")

    max_user_id = max(train["userId"].tolist() )
    max_movie_id = max(train["movieId"].tolist() )


    return train, test, max_user_id, max_movie_id


def build_model(max_movie_id, max_user_id):
    """
    This function builds up the model (neural network) which learn low dimensional embeddings of movies and users, and calculated
    Args:
        max_movie_id (int): The maximum movie id
        max_user_id (int): The maximum user id
    Returns:
        (Model): A neural network model, built up from the specified layers
    """
    # the number of dimensions our low dimensional embeddings should have
    dims_of_embedding = 35
    bias = 1
    
    # inputs
    # define an input layer by the shape of the tensors it takes
    w_inputs = Input(shape=(1,), dtype='int32')
    w = Embedding(max_work+1, dim_embedddings, name="work")(w_inputs)
    w_bis = Embedding(max_work + 1, bias, name="workbias")(w_inputs)

    # context
    u_inputs = Input(shape=(1,), dtype='int32')
    u = Embedding(max_user+1, dim_embedddings, name="user")(u_inputs)
    u_bis = Embedding(max_user + 1, bias, name="userbias")(u_inputs)
    o = multiply([w, u])
    o = Dropout(0.5)(o)
    o = concatenate([o, u_bis, w_bis])
    o = Flatten()(o)
    o = Dense(10, activation="relu")(o)
    o = Dense(1)(o)

    rec_model = Model(inputs=[w_inputs, u_inputs], outputs=o)
    rec_model.compile(loss='mae', optimizer='adam', metrics=[rmse])

    return rec_model

def get_array(series):
    return np.array([[element] for element in series])