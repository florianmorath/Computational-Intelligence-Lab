import numpy as np
import os


def get_train_file_path():
    return '../data/data_train.csv'


def get_test_file_path():
    return '../data/sampleSubmission.csv'


def parse_line(line):
    # Format is user, item, rating
    key, value = line.split(",")
    rating = int(value)
    row_string, col_string = key.split("_")
    row = int(row_string[1:])
    col = int(col_string[1:])
    return row-1, col-1, rating


def load_ratings_from_file_path(file_path):
    ratings = []
    with open(file_path) as file:
        file.readline() # remove header
        for line in file:
            ratings.append(parse_line(line))
    return ratings


def write_submission(ratings, file_name):
    # Build output string
    output = "Id,Prediction\n"
    for (row, col, rat) in ratings:
        output += "r%d_c%d,%f\n" % (row + 1, col + 1, rat)
    
    # Write file 
    with open(os.path.join('../predictions_csv', file_name), 'w') as output_file:
        output_file.write(output)
        
    return output

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