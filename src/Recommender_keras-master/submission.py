from utils import *
from sklearn.metrics import mean_absolute_error
import pickle


train, test, max_user, max_work, mapping_work = get_data()

def rmse(y_true, y_pred):
	return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

#pickle.dump(mapping_work, open('mapping_work.pkl', 'wb'))
data = pd.read_csv("foo4.csv")

mapping_work = get_mapping(data["movieId"])

data["movieId"] = data["movieId"].map(mapping_work)

mapping_users = get_mapping(data["movieId"])

data["movieId"] = data["movieId"].map(mapping_users)

cols = ["userId", "movieId", "rating"]


max_user = max(data["userId"].tolist() )
max_work = max(data["movieId"].tolist() )

    
model = get_model_3(max_work, max_user)

history = model.fit([get_array(data["movieId"]), get_array(data["userId"])], get_array(data["rating"]), nb_epoch=10,
                    validation_split=0, verbose=2)



def parse_line(line):
    key, value = line.split(",")
    row_string, col_string = key.split("_")
    row = int(row_string[1:])
    col = int(col_string[1:])
    return row, col


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
    output += "r%d_c%d,%f\n" % (row , col, rat)

# Write file 
with open('sub_0.csv', 'w') as output_file:
    output_file.write(output)