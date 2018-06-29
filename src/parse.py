import numpy as np
import pandas as pd 


def parse_line(line):
    key, value = line.split(",")
    rating = int(value)
    row_string, col_string = key.split("_")
    row = int(row_string[1:])
    col = int(col_string[1:])
    return row, col, rating


ratings = []
with open('../data/data_train.csv') as file:
    file.readline()
    for line in file:
        ratings.append(parse_line(line))


# np.savetxt("foo.csv", ratings, delimiter=",")
df = pd.DataFrame(ratings)
df.to_csv("foo.csv")