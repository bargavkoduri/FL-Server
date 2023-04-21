from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D
from helper import encode_weights,write_to_file


# Model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), strides=1,activation='relu', input_shape=(125,125,1)))
model.add(AveragePooling2D())
model.add(Conv2D(64, kernel_size=(3,3), strides=1,  activation='relu'))
model.add(AveragePooling2D())
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))


# configuration of the model
model_config = model.to_json()
# Write the JSON string to a file
write_to_file('model_config.json', model_config)

encoded_weight = encode_weights(model.get_weights())  # encoding weights of the model to base64 string and writing to a file
write_to_file('model_weights.txt',encoded_weight)