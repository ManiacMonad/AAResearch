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
                    keras.layers.Dense(32, input_shape=((33 - 5) * 2 + (33 - 5) * 2,), activation="relu"),
                    keras.layers.Dense(128, activation="relu"),
                    keras.layers.Dense(128, activation="relu"),
                    keras.layers.Dense(11, activation="sigmoid"),
                ]
            )
        self.model.compile(optimizer=legacy.Adam(), loss="mse", metrics=["accuracy"])

    def train(self, x, y):
        self.model.fit(x, y, epochs=2, batch_size=128)

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


# The dataset collected at the University of Florence during 2012, has been captured using a Kinect camera.
# It includes 9 activities: wave, drink from a bottle, answer phone,clap, tight lace, sit down, stand up, read watch, bow.
# During acquisition, 10 subjects were asked to perform the above actions for 2/3 times. This resulted in a total of 215 activity samples.
# We suggest a leave-one-actor-out protocol: train your classifier using all the sequences from 9 out of 10 actors
# and test on the remaining one. Repeat this procedure for all actors and average the 10 classification accuracy values.
# Videos depicting the actions are named GestureRecording_Id<ID_GESTURE>actor<ID_ACTOR>idAction<ID_ACTION>category<ID_CATEGORY>.avi


class STREAM_READ_FLORENCE_FALL(_BASE_VIDEO_STREAM):
    def __init__(self, fps, ID_GESTURE, ID_ACTOR, ID_ACTION, ID_CATEGORY) -> None:
        self.fps = fps
        self.currentFrame = 0
        self.videoName = str(
            Path.home()
            / f"Downloads/Florence_3d_actions/GestureRecording_Id{ID_GESTURE}actor{ID_ACTOR}idAction{ID_ACTION}category{ID_CATEGORY}.avi"
        )
        self.capture = cv2.VideoCapture(self.videoName)

    def readNextImage(self):
        self.currentFrame += 1
        ret, image = self.capture.read()
        if not ret:
            print(f"Failed to read image from video ")
        return image

    def dispose(self):
        self.capture.release()
        cv2.destroyAllWindows()

    def getFrameRate(self):
        return self.fps
