# CIL_project

This file explains the structure of the src-folder. The files can be divided into three categories, 
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


To run the code it is recommended to first run the following instructions:

# create a virtual environment
virtualenv --python=python3 <venv-name>
source <venv-name>/bin/activate

# install required packages inside it
cd <path-where-requirements-file-is-located>
pip install -r requirements.txt

