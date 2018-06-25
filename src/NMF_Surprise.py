
# coding: utf-8

# # NMF baseline algorithm using surprise package:
# 
# using matrix_factorization.NMF algorithm from http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.NMF
# 

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats

import helpers
from surprise_helpers import CustomReader, get_ratings_from_predictions
from surprise import Reader, Dataset
from surprise.model_selection.search import RandomizedSearchCV
from surprise.prediction_algorithms.matrix_factorization import NMF


# ## Data Loading

# In[2]:


reader = CustomReader()
filepath = helpers.get_train_file_path()
data = Dataset.load_from_file(filepath, reader=reader)


# ## Search over Parameters

# In[ ]:


param_grid = {'n_epochs': stats.randint(230,290), 
              'n_factors': stats.randint(1,30),
                'reg_pu': stats.uniform(0.1,0.2),
                'reg_qi': stats.uniform(0.1,0.2),
                'biased': [True, False],
             }      
        

gs = RandomizedSearchCV(algo_class=NMF, param_distributions=param_grid, measures=['rmse'], 
                        cv=10, joblib_verbose=100, n_jobs=-1, n_iter=5)
gs.fit(data)

# best RMSE score
print(gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])


# In[ ]:


pd.DataFrame.from_dict(gs.cv_results)


# ## Results: params 
# 
# note: run on Leonhard cluster (20 cores and 22GB mem) <br/>
# cv=10
# 
# 0.994469288179
# {'n_epochs': 261, 'n_factors': 23, 'reg_bi': 1.6667791514258101, 'reg_bu': 0.64371831932001311, 'reg_pu': 0.10279700072169747, 'reg_qi': 0.12452450006647738}
# 
# 0.995282415101
# {'n_epochs': 256}
# 
# 1.00796302363
# {'reg_pu': 0.16886198480906289}
# 
# 1.00722502711
# {'reg_qi': 0.13702113849142605}
# 
# 1.0092083515
# {'reg_bu': 0.34284468705230597}
# 
# 1.00944021138
# {'reg_bi': 0.93748069281861202}
# 
# 1.00976479276
# {'n_factors': 15}
# 
# 1.00594746757
# {'n_epochs': 262, 'n_factors': 21, 'reg_bi': 1.4693650385908883, 'reg_bu': 0.29370679004185357, 'reg_pu': 0.16575459605631229, 'reg_qi': 0.13074898536926785}

# ## Train

# In[4]:


# choose optimal params from above
algo = NMF(n_epochs=256, reg_pu=0.1686198480906289, reg_qi=0.13702113849142605, n_factors=15, reg_bu=0.34284468705230597, reg_bi=0.93748069281861202)

# train 
algo.fit(data.build_full_trainset())


# ## Predicting
# We load the test data to predict.

# In[5]:


test_file_path = helpers.get_test_file_path()
test_data = Dataset.load_from_file(test_file_path, reader=reader)
testset = test_data.construct_testset(test_data.raw_ratings)
predictions = algo.test(testset)
predictions[0]


# We need to convert the predictions into the right format.

# In[6]:


ratings = get_ratings_from_predictions(predictions)


# Now we can write the file.

# In[7]:


output = helpers.write_submission(ratings, 'submission_surprise_NMF_0.csv')
print(output[0:100])

