import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import legacy
from utils import _BASE_VIDEO_STREAM
import cv2
from pathlib import Path


class mp_model:
    def __init__(self, loadfromfile=False):
        if loadfromfile:
            self.model = keras.models.load_model("mp_gold_fish.h5")
        else:
            self.model = Sequential(
                [
                    keras.layers.Dense(32, input_shape=((33 - 5) * 2 + (33 - 5) * 2 + 2,), activation="relu"),
                    keras.layers.Dense(48, activation="relu"),
                    keras.layers.Dense(40, activation="sigmoid"),
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


class STREAM_READ_UR_FALL(_BASE_VIDEO_STREAM):
    def __init__(self, foldername, fps) -> None:
        self.folderName = foldername
        self.fps = fps
        self.currentFrame = 0

    def readNextImage(self):
        self.currentFrame += 1
        imageName = str(
            Path.home() / f"Downloads/ur_fall/{self.folderName}/{self.folderName}-{self.currentFrame:03d}.png"
        )
        image = cv2.imread(imageName)
        if image is None:
            print(f"Failed to read image {imageName}")
        return image

    def dispose(self):
        cv2.destroyAllWindows()

    def getFrameRate(self):
        return self.fps


class VIDEO_STREAM(_BASE_VIDEO_STREAM):
    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(0)

    def readNextImage(self):
        ret, image = self.cap.read()
        if not ret:
            print("Failed to read image")
        return image

    def dispose(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def getFrameRate(self):
        return self.cap.get(cv2.CAP_PROP_FPS)
