from helpers import load_ratings_from_file_path, get_train_file_path, get_test_file_path, write_submission
import numpy as np

# Matrix helper functions
            
def load_data():
    # X has dim (USER_COUNT x ITEM_COUNT)
    USER_COUNT = 10000
    ITEM_COUNT = 1000

    ratings = load_ratings_from_file_path(get_train_file_path())

    X = np.zeros([USER_COUNT, ITEM_COUNT], dtype=np.float32)
    for (row, col, rating) in ratings:
        X[row, col] = rating
    return X

def load_pred_data():
    # X has dim (USER_COUNT x ITEM_COUNT)
    USER_COUNT = 10000
    ITEM_COUNT = 1000

    ratings = load_ratings_from_file_path(get_test_file_path())

    X = np.zeros([USER_COUNT, ITEM_COUNT])
    for (row, col, rating) in ratings:
        X[row, col] = rating
    return X


def get_prediction_ratings_from_matrix(X_pred):
    ratings = load_ratings_from_file_path(get_test_file_path())
    for (row, col, _) in ratings:
        yield row, col, X_pred[row, col]
        
    
def write_submission_file(X_pred, file_name):
    ratings = get_prediction_ratings_from_matrix(X_pred)
    write_submission(ratings, file_name)