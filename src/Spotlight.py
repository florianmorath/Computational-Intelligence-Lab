
# coding: utf-8

# In[1]:


import helpers
import numpy as np
import torch
from spotlight.evaluation import rmse_score
from spotlight.interactions import Interactions
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.cross_validation import random_train_test_split

from matrix_helpers import write_submission_file
from helpers import get_test_file_path, load_ratings_from_file_path
# ## Data loading

# ## Search over params

# In[29]:


# get the ratings as list of triplets (user, movie, rating)
triplets = helpers.load_ratings_from_file_path(helpers.get_train_file_path())

user_ids = []
movie_ids = []
ratings = []

for (user, movie, rating) in triplets:
    user_ids.append(user)
    movie_ids.append(movie)
    ratings.append(rating)

explicit_interactions = Interactions(np.asarray(user_ids, dtype=np.int32), np.asarray(movie_ids, dtype=np.int32), np.asarray(ratings, dtype=np.float32))
model = ExplicitFactorizationModel(loss='regression', embedding_dim=5, n_iter=100, batch_size=256, learning_rate=0.005, l2=0.0, sparse=False)


# train, test = random_train_test_split(explicit_interactions, random_state=np.random.RandomState(42))
# print('Split into \n {} and \n {}.'.format(train, test))

# model.fit(train, verbose=True)
# train_rmse = rmse_score(model, train)
# test_rmse = rmse_score(model, test)
# print('Train RMSE {:.3f}, test RMSE {:.3f}'.format(train_rmse, test_rmse))

model.fit(explicit_interactions, verbose=True)
X_pred = np.zeros([10000,1000])
ratings = load_ratings_from_file_path(get_test_file_path())
for (row, col, _) in ratings:
        X_pred[row, col] = model.predict(np.array(row))[col]

write_submission_file(X_pred, 'spotlight_explicit_factor_0.csv')