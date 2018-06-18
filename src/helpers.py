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

    
def write_submission_file(X_pred, file_name){
    # create submission string
    ratings = []
    with open('../data/sampleSubmission.csv') as file:
        file.readline() # remove header
        for line in file:
            key, value = line.split(",")
            rating = int(value)
            row_string, col_string = key.split("_")
            row = int(row_string[1:])
            col = int(col_string[1:])
            ratings.append((row-1, col-1, rating))

    output = "Id,Prediction\n"
    for (row, col, _) in ratings:
        output += "r%d_c%d,%f\n" % (row + 1, col + 1, X_pred[row, col])
        
    # write file    
    with open(file_name, 'w') as output_file:
        output_file.write(output)
    return output
}