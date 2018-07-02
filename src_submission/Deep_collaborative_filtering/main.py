from model import *
from sklearn.metrics import mean_absolute_error
import pickle

## SET BACKEND
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

def parse_line(line):
    key, value = line.split(",")
    rating = int(value)
    row_string, col_string = key.split("_")
    row = int(row_string[1:])
    col = int(col_string[1:])
    return row, col, rating



# get the data
train, test, max_user, max_work = get_data()

# get the model
model = get_model(max_work, max_user)

# train the model
history = model.fit([get_array(train["movieId"]), get_array(train["userId"])], get_array(train["rating"]), nb_epoch=20,
                    validation_split=0.2, verbose=2)

# Get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('keras_plot.png')
plt.show();


# make predictions
predictions = model.predict([get_array(test["movieId"]), get_array(test["userId"])])

# create csv submission file of predictions
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
with open('sub_5.csv', 'w') as output_file:
    output_file.write(output)