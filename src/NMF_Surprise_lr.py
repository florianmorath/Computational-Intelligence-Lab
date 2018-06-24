
# coding: utf-8

# # NMF baseline algorithm using surprise package:
# 
# using matrix_factorization.NMF algorithm from http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.NMF
# 

# In[ ]:


import pandas as pd
import numpy as np
from scipy import stats

import helpers
from surprise_helpers import CustomReader, get_ratings_from_predictions
from surprise import Reader, Dataset
from surprise.model_selection.search import RandomizedSearchCV
from surprise.prediction_algorithms.matrix_factorization import NMF


# ## Data Loading

# In[ ]:


reader = CustomReader()
filepath = helpers.get_train_file_path()
data = Dataset.load_from_file(filepath, reader=reader)


# ## Search over Parameters

# In[ ]:


param_grid = {#'n_epochs': stats.randint(1,300), 
              'lr_all': stats.uniform(0,10),
              #'reg_all': stats.uniform(0.0,1),
              #'n_factors': stats.randint(1,200),
             }      
        

gs = RandomizedSearchCV(algo_class=NMF, param_distributions=param_grid, measures=['rmse'], 
                        cv=10, joblib_verbose=100, n_jobs=-1, n_iter=20)
gs.fit(data)

# best RMSE score
print(gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])

