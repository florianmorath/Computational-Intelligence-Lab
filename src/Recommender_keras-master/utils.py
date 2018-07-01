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
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

def get_mapping(series):
    occurances = defaultdict(int)
    for element in series:
        occurances[element] += 1
    mapping = {}
    i = 0
    for element in occurances:
        i += 1
        mapping[element] = i

    return mapping




def get_data():
    # train
    train = pd.read_csv("foo_train.csv")

    mapping_work = get_mapping(train["movieId"])

    train["movieId"] = train["movieId"].map(mapping_work)

    mapping_users = get_mapping(train["movieId"])

    train["movieId"] = train["movieId"].map(mapping_users)


    # test
    test = pd.read_csv("foo_test.csv")

    mapping_work = get_mapping(test["movieId"])

    test["movieId"] = test["movieId"].map(mapping_work)

    mapping_users = get_mapping(test["movieId"])

    test["movieId"] = test["movieId"].map(mapping_users)


    max_user = max(train["userId"].tolist() )
    max_work = max(train["movieId"].tolist() )


    return train, test, max_user, max_work, mapping_work


def get_model_3(max_work, max_user):
    dim_embedddings = 35
    bias = 1
    # inputs
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
    #rec_model.summary()
    rec_model.compile(loss='mae', optimizer='adam', metrics=[rmse])

    return rec_model

def get_array(series):
    return np.array([[element] for element in series])