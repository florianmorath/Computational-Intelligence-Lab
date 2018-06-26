
# coding: utf-8

# # k-NN using scikit-surprise

# In[1]:


import helpers
from surprise_helpers import CustomReader, get_ratings_from_predictions
from surprise import Reader, Dataset
import time

# ## Data loading
# We load the data using our custom reader.
# See: http://surprise.readthedocs.io/en/stable/getting_started.html#use-a-custom-dataset

# In[2]:


reader = CustomReader()
filepath = helpers.get_train_file_path()
data = Dataset.load_from_file(filepath, reader=reader)


# ## Parameter search
# We search for good values of parameters of the chosen algorithm.
# 
# First we need to define the search space.

# In[25]:


from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline
from surprise.model_selection import RandomizedSearchCV
from scipy import stats
import pandas as pd

algos = [
    (KNNBasic, 'KNNBasic'),
    (KNNWithMeans, 'KNNWithMeans'),
    (KNNWithZScore, 'KNNWithZScore'),
    (KNNBaseline, 'KNNBaseline')
]   
param_grid = {
    'k': stats.randint(10,80),
    'min_k': stats.randint(1, 9),
    'sim_options': {
        'name': ['cosine', 'msd', 'pearson', 'pearson_baseline'],
        'user_based': [False, True],
        'min_support': [1, 10],
        'shrinkage': [0, 100]
    },
    'bsl_options': {
        'method': ['als'],
        'n_epochs': range(10, 15),
        'reg_i': range(8, 12),
        'reg_u': range(12, 18)
    }
}


# We loop through all algo types:

# In[26]:


def fit_and_store(algos):
    best_algos = []
    for algo, algo_name in algos:
        rs = RandomizedSearchCV(algo,
                                param_grid,
                                n_iter=25,
                                measures=['rmse'], cv=10, n_jobs=-1,
                                refit=True # so we can use test() directly
                                )

        rs.fit(data)
        timestamp = time.time()
        print('Best score {} of {} with parameters:'.format(rs.best_score['rmse'], timestamp))
        best_params_df = pd.DataFrame.from_dict(rs.best_params['rmse'])
        best_params_df.to_pickle('{}_best_params_{}.pkl'.format(algo_name, timestamp))
        results_df = pd.DataFrame.from_dict(rs.cv_results)
        results_df.to_pickle('{}_results_{}.pkl'.format(algo_name, timestamp))
        yield (rs, algo_name, timestamp)


# In[ ]:


best_algos = fit_and_store(algos)


# ## Predicting
# We load the test data to predict.

# In[20]:


test_file_path = helpers.get_test_file_path()
test_data = Dataset.load_from_file(test_file_path, reader=reader)
testset = test_data.construct_testset(test_data.raw_ratings)


# We write the predictions for all algorithms.

# In[27]:


def predict_and_write(best_algos):
    for rs, algo_name, timestamp in best_algos:
        predictions = rs.test(testset)
        # We need to convert the predictions into the right format.
        ratings = get_ratings_from_predictions(predictions)
        file_name = 'submission_{}_{}.csv'.format(algo_name, timestamp)
        # Now we can write the file.
        output = helpers.write_submission(ratings, file_name)
        print('Wrote predictions to "{}"'.format(file_name))


# In[ ]:


predict_and_write(best_algos)

