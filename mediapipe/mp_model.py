import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import legacy


class mp_model:
    def __init__(self, loadfromfile=False):
        if loadfromfile:
            self.model = keras.models.load_model("mp_gold_fish.h5")
        else:
            self.model = Sequential(
                [
                    keras.layers.Dense(64, input_shape=(33 * 2 + 33 * 2 + 2,), activation="relu"),
                    keras.layers.Dense(80, activation="relu"),
                    keras.layers.Dense(80, activation="relu"),
                    keras.layers.Dense(1, activation="sigmoid"),
                ]
            )
        self.model.compile(optimizer=legacy.Adam(), loss="mse", metrics=["accuracy"])

    def train(self, x, y):
        self.model.fit(x, y, epochs=150, batch_size=128)

    def predict(self, velocityPerSample):
        return self.model.predict(velocityPerSample)

    def save(self):
        self.model.save("mp_gold_fish.h5")
