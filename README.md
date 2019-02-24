# Computational Intelligence Lab project

The goal of Collaborative Filtering (CF) is to predict a user’s unknown rating for a certain item, based on its own available ratings for other items, and the preferences of other users. In this project we focused on two approaches to CF: latent factor models and neighborhood models. The methods we implemented include: Average-over-items, Average-over-users, SVD with dimension reduction, ALS, kNN and NMF. We got the best results with an improved version of SVD, referred to as SVD+. Compared to the basic SVD with dimension reduction, this gained a 7.66% improvement on the Kaggle public dataset.

## Overview

```
report.pdf          -- paper describing our approaches
/data               -- given training data      
/papers             -- literature about CF and recommender systems
/predictions_csv    -- obtained predictions (kaggle submissions)
/src                -- implemented ML models
```

The files in the src-folder can be divided into three categories, 
namely baseline algortihms, novel solutions and data handler files.

basline algorithms:
- average_over_items.ipynb
- average_over_users.ipynb
- SVD_basic.ipynb
- ALS.ipynb
- KNN_basic.ipynb
- NMF.ipynb

novel solutions:
- SVD_improved.ipynb
- Deep_collaborative_filtering directoy

data handler files:
- data_handler.py
- surprise_extensions.py

## Installation
To run the code, it is recommended to first run the following instructions:

Create a virtual environment:
```
virtualenv --python=python3 <venv-name>  
source \<venv-name\>/bin/activate
```

Install required packages inside it:
```
cd \<path-where-requirements-file-is-located\>  
pip install -r requirements.txt
```
