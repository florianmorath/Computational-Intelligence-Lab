import numpy as np

def load_data():
    # X has dim (USER_COUNT x ITEM_COUNT)
    USER_COUNT = 10000
    ITEM_COUNT = 1000

    ratings = []
    with open('../data/data_train.csv') as file:
        file.readline() # remove header
        for line in file:
            key, value = line.split(",")
            rating = int(value)
            row_string, col_string = key.split("_")
            row = int(row_string[1:])
            col = int(col_string[1:])
            ratings.append((row-1, col-1, rating))

        X = np.zeros([USER_COUNT, ITEM_COUNT])
        for (row, col, rating) in ratings:
            X[row, col] = rating
    return X

    