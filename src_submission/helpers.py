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
