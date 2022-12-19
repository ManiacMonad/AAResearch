# import tensorflow and keras
import tensorflow as tf
from tensorflow import keras

# import Sequential model from tensorflow
from tensorflow.keras.models import Sequential

# import tensorflow legacy optimizers
from tensorflow.keras.optimizers import legacy

# create a Sequential model with an input layer of 2 nodes, 3 layers of 64 nodes each and an output layer of 1 node. the output number is between 0~100
model = Sequential(
    [
        keras.layers.Dense(64, input_shape=(2,), activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1),
    ]
)
# compile the model with the Adam optimizer, the mean squared error loss function and the accuracy metric
model.compile(optimizer=legacy.Adam(), loss="mse", metrics=["accuracy"])
x = []
y = []
for i in range(0, 100):
    for j in range(0, 100):
        # create a training data with 0~i,j step=10 as input and i+j as output
        x.append([i, j])
        y.append([i + j])
# train the model
model.fit(x, y, epochs=5, batch_size=128)

# interact with the model via console for given input a and b output c based on the model's prediction
while True:
    a = int(input("a: "))
    b = int(input("b: "))
    print("c: " + str(model.predict([[a, b]])[0][0]))
# save the model to a file
model.save("model.h5")
