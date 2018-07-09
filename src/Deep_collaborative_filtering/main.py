"""This module trains the deep CF model from model.py, does visualization of the loss and makes predictions.

"""

from model import *
from sklearn.metrics import mean_absolute_error
import pickle

## SET BACKEND
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def visualize_loss(training_loss, test_loss):
    """ Visualize loss history.
    
    Args:
        training_loss (History): Keras History object with stored training losses.
        test_loss (History): Keras History object with stored test losses.
        
    Note: stores plot in current folder as 'keras_plot.png'.
 
    """   
    epoch_count = range(1, len(training_loss) + 1)
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('keras_plot.png')

    
def parse_line(line):
    """ Parse a line of the form 'rX_cY,R'.
    
    """
    key, value = line.split(",")
    rating = int(value)
    row_string, col_string = key.split("_")
    row = int(row_string[1:])
    col = int(col_string[1:])
    return row, col, rating


def make_submission(predictions, file_name):
    """ Create csv submission file based on predicitons.
    
    """
    ratings_a = []
    for pred in predictions:
        ratings_a.append(pred[0])

    ratings = []
    with open('sampleSubmission.csv') as file:
        file.readline()
        for line in file:
            row, col, rat = parse_line(line)
            ratings.append((row,col,rat))

    # Build output string
    output = "Id,Prediction\n"
    for (row, col, rat) in ratings:
        output += "r%d_c%d,%f\n" % (row , col, ratings_a.pop(0))

    # Write file 
    with open(file_name, 'w') as output_file:
        output_file.write(output)

        
        
# get the data
train = get_train_data()
test = get_test_data()
max_user = max(train["userId"].tolist())
max_work = max(train["movieId"].tolist())

# get the model
model = build_model(max_work, max_user)

# train the model
# note: only use validation_split for visulaization and parameter search.
history = model.fit([get_array(train["movieId"]), get_array(train["userId"])], get_array(train["rating"]), nb_epoch=20, validation_split=0.2, verbose=2)

# Get training and test loss histories for visualization
training_loss = history.history['loss']
test_loss = history.history['val_loss']
visualize_loss(training_loss, test_loss)

# make predictions and submission file
predictions = model.predict([get_array(test["movieId"]), get_array(test["userId"])])
make_submission(predictions, 'DCF_0.csv')
