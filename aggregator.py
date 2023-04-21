import json
from helper import read_from_file, decode_weights, ReadandProcessData, write_to_file, encode_weights
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import model_from_json

weights_arr = read_from_file("received_updated_weights.txt")
weights_arr = json.loads(weights_arr)

average_weights = {}
for i in range(len(weights_arr)):
    weights_arr[i] = json.loads(weights_arr[i])
    keys_arr = weights_arr[i].keys()
    for layer_name in keys_arr:
        if layer_name == "number_of_train":
            continue
        weights_arr[i][layer_name] = decode_weights(weights_arr[i][layer_name])

# load model weights
model_weights = decode_weights(read_from_file("model_weights.txt"))
# load model from json file
model = model_from_json(read_from_file("model_config.json"))
# compile the model
model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# set weights to the model
model.set_weights(model_weights)

# Counting the number of data points
total_data_points = 0
for i in range(len(weights_arr)):
    total_data_points += weights_arr[i]["number_of_train"]

# updating weights in each layer
index = 0
for layer in model.layers:
    if layer.trainable_weights:
        # weights_arr[0][layer.name] = np.array(weights_arr[0][layer.name])
        weights_arr[0][layer.name][0] *= (weights_arr[0]["number_of_train"]/total_data_points)
        weights_arr[0][layer.name][1] *= (weights_arr[0]["number_of_train"]/total_data_points)
        # temp_list = np.array(weights_arr[0][layer.name],dtype=float)
        # print(temp_list)
        # temp_list /= weights_arr[0]["number_of_train"]/total_data_points
        # weights_arr[0][layer.name] /= (weights_arr[0]["number_of_train"]/total_data_points)
        for i in range(1,len(weights_arr)):
            weights_arr[0][layer.name][0] += ((weights_arr[i]["number_of_train"]/total_data_points) * weights_arr[i][layer.name][0])
            weights_arr[0][layer.name][1] += ((weights_arr[i]["number_of_train"]/total_data_points) * weights_arr[i][layer.name][1])
        model.layers[index].set_weights(weights_arr[0][layer.name])
    index += 1

X_test,y_test = ReadandProcessData("pest/test")
y_pred = model.predict(X_test,verbose=0)
y_pred = [np.argmax(ele) for ele in y_pred]
print(accuracy_score(y_test,y_pred)*100)

write_to_file('model_weights.txt',encode_weights(model.get_weights()))