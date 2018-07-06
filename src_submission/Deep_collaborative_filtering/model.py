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


def get_train_data():
    """
    This function reads in the training data from the csv file
    Returns:
        (DataFrame): the training data set
    """
    # train set
    train = pd.read_csv("train.csv")

    return train

def get_test_data():
    """
    This function reads in the test data from the csv file
    Returns:
        (DataFrame): the test data set
    """

    # test set
    test = pd.read_csv("test.csv")

    return test


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
    # define an input layer: set the datatype and the the shape of the tensors it takes, the true shape would be (number_of_movies, 1) but number_of_movies is the batch size, and keras ignores the first dimensions (batch_size), so we give a shape (1,)
    movie_inputs = Input(shape=(1,), dtype='int32')
    
    # define an embedding layer with input dimension (max_movie_id+1) and output dimension (dims_of_embedding). This layer takes a positive integer index (movie id) and outputs a tensor of shape (None, 1,dims_of_embedding), where None is the batch dimension. So this actually where the low-dimensional embedding of movies happens
    movie_embedding = Embedding(max_movie_id+1, dims_of_embedding, name="movie")(movie_inputs)
    movie_bis = Embedding(max_work + 1, bias, name="moviebias")(movie_inputs)

    # define analogous layers for the users
    user_inputs = Input(shape=(1,), dtype='int32')
    user_embedding = Embedding(max_user+1, dims_of_embedding, name="user")(user_inputs)
    user_bis = Embedding(max_user_id + 1, bias, name="userbias")(user_inputs)
    
    # the following layers together basically calculate the dot product of user and movie embedding vectors to predict the corresponding rating
    # the multiply layer multiplies (element-wise) a list of input tensors (all of same shape) and returns a single tensor of the same shape
    output = multiply([movie_embedding, user_embedding])
    # a Dropout layer applies Dropout to the input, which means setting a fraction rate (=0.5) of input units to 0 at each update during training. This helps preventing overfitting.
    output = Dropout(0.5)(output)
    # a Concatente layer concatenates a list of input tensors into a single tensor
    output = concatenate([output, user_bis, movie_bis])
    # a Flatten layer flattens the input
    output = Flatten()(output)
    # just regular densely, connected NN layers, where can define the number of units in the layer (e.g. 10), and the activation function of the units (e.g. 'relu')
    output = Dense(10, activation="relu")(output)
    output = Dense(1)(output)

    # define a model with its input(s) and output(s)
    model = Model(inputs=[movie_inputs, user_inputs], outputs=output)
    # prepare the model for training, choose loss, optimizers and potentially metrics
    model.compile(loss='mae', optimizer='adam', metrics=[rmse])

    return model

def get_array(series):
    return np.array([[element] for element in series])