"""This module contains functions to read in the training and test data as well as functions to write our predictions for the missing ratings back to a submission file
"""
import numpy as np
import os


def get_train_file_path():
    """
    Returns:
        string: The path to the file containing the training data
    """
    return '../data/data_train.csv'


def get_test_file_path():
    """
    Returns:
        string: The path to the file containing the test data
    """
    return '../data/sampleSubmission.csv'


def parse_line(line):
    """This function parses a line of a csv file. The line has the format 'rX_cY, R', where X, Y and R are integers. X and Y correspond to row(user) and column(movie) indices in ratings matrix, while R is the value(rating) of the matrix's entry indexed by X and Y.    
    Args:
        line (str): A line of a csv file
    Returns:
        tuple: A tuple of the form (user id, movie id, rating)
    """
    # line is of the form r44_c1,4
    key, value = line.split(",")
    # key is a string containing the user id and movie id, value contains the rating
    rating = int(value)
    row_string, col_string = key.split("_")
    # extract the ids from the row/col strings
    row = int(row_string[1:])
    col = int(col_string[1:])
    # row and col ids in the train data set use 1-based indexes, while numpy arrays use 0-base indexes, so we have to decrease the read row and col indices by one
    return row-1, col-1, rating


def load_ratings_from_file_path(file_path):
    """This function loads ratings from a given csv file
    Args:
        file_path (str): A path to a csv file
    Returns:
        list: A list of tuples. Each tuple has the form (userId, movieId, rating)
    """
    ratings = []
    with open(file_path) as file:
        file.readline() # remove header
        for line in file:
            ratings.append(parse_line(line))
    return ratings


def write_submission(ratings, file_name):
    """This function writes the predicted ratings back to a csv file
    Args:
        ratings (list): A list of tuples. The tuples are of the form (userId, movieId, rating)
        file_name (str): The name of the file in which the predicted ratings are written
    """
    # Build output string to write into the file
    output = "Id,Prediction\n"
    for (row, col, rat) in ratings:
        # every line is of the format 'rX_cY,R' where X and Y correspond to row(user) and column(movie) indices and R is the rating
        # we have do increase row and col by one because numpy arrays use 0-base indexing while movie/user indices start at 1
        output += "r%d_c%d,%f\n" % (row + 1, col + 1, rat)
    
    # Write file 
    with open(os.path.join('../predictions_csv', file_name), 'w') as output_file:
        output_file.write(output)
        
    return output
     
    
def load_train_data():
    """This function build up a rating matrix from the ratings found in the training data file
    Returns:
        numpy.ndarray: The rating matrix
    """
    # X has dim (USER_COUNT x ITEM_COUNT)
    USER_COUNT = 10000
    ITEM_COUNT = 1000

    ratings = load_ratings_from_file_path(get_train_file_path())

    X = np.zeros([USER_COUNT, ITEM_COUNT], dtype=np.float32)
    for (row, col, rating) in ratings:
        X[row, col] = rating
    return X


def load_test_data():
    """This function build up a rating matrix from the ratings found in the test data file
    Returns:
        numpy.ndarray: The rating matrix
    """
    # X has dim (USER_COUNT x ITEM_COUNT)
    USER_COUNT = 10000
    ITEM_COUNT = 1000

    ratings = load_ratings_from_file_path(get_test_file_path())

    X = np.zeros([USER_COUNT, ITEM_COUNT])
    for (row, col, rating) in ratings:
        X[row, col] = rating
    return X


def get_prediction_ratings_from_matrix(X_pred):
    """This function builds a list of tuples containg userId, movieId and the corresponding predicted rating
    Args:
        X_pred (numpy.ndarray): The rating matrix filled with our predicted ratings
    Returns:
        list: A list of tuples of the form (userId, movieId, rating)
    """
    ratings = load_ratings_from_file_path(get_test_file_path())
    for (row, col, _) in ratings:
        yield row, col, X_pred[row, col]
        