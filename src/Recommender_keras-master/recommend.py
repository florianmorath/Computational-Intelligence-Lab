from utils import *
from sklearn.metrics import mean_absolute_error
import pickle


train, test, max_user, max_work, mapping_work = get_data()

def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))


model = get_model_3(max_work, max_user)

history = model.fit([get_array(train["movieId"]), get_array(train["userId"])], get_array(train["rating"]), nb_epoch=1,
                    validation_split=0, verbose=2)

predictions = model.predict([get_array(test["movieId"]), get_array(test["userId"])])


ratings_a = []
for pred in predictions:
    ratings_a.append(pred[0])

ratings = []
with open('sampleSubmission.csv') as file:
    file.readline()
    for line in file:
        row, col = parse_line(line)
        rating = model.predict([get_array(col), get_array(row)])
        ratings.append((row,col,rating))

# Build output string
output = "Id,Prediction\n"
for (row, col, rat) in ratings:
    output += "r%d_c%d,%f\n" % (row , col, ratings_a.pop(0))

# Write file 
with open('sub_0.csv', 'w') as output_file:
    output_file.write(output)