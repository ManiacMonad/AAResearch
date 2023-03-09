import cv2 as cv
from cv2 import resize
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import parse
import os
from configs import BASE_CONFIGURATIONS
from mp_model import STREAM_READ_UR_FALL, STREAM_READ_FLORENCE_FALL
from pathlib import Path
from utils import processLandmarks, getLandmarkVelocity, putText

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Input, GlobalMaxPooling2D, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.optimizers import legacy
from tensorflow.keras import Sequential

# import mode
from scipy.stats import mode


TP_0 = 0
TN_0 = 0
FP_0 = 0
FN_0 = 0

TP_1 = 0
TN_1 = 0
FP_1 = 0
FN_1 = 0

Mis1_0 = 0  # predict fall
Mis2_0 = 0  # predict not fall but wrong gesture
Correct_0 = 0  # predict not fall and correct gesture

Mis1_1 = 0  # predict fall
Mis2_1 = 0  # predict not fall but wrong gesture
Correct_1 = 0  # predict not fall and correct gesture


class CONFIGURATIONS:
    def __init__(
        self,
        pyr_scale=0.4,  # pyr_scale = 0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.
        levels=3,  # levels = 1 means that no extra layers are created and only the original images are used.
        winsize=4,  # winsize = 15 means that 15x15 windows are used to compute the optical flow.
        iterations=8,  # iterations = 3 means that three iterations are done at each pyramid level.
        poly_n=6,  # poly_n = 5 means that each pixel neighborhood has a size of 5x5 pixels.
        poly_sigma=1.1,  # poly_sigma = 1.1 is the standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion.
        flags=cv.OPTFLOW_LK_GET_MIN_EIGENVALS,  # flags = 0 means that the algorithm calculates the minimum eigenvalue of the 2x2 normal matrix of optical flow equations (this is the fastest method).
        threshold=10.0,  # threshold = 10.0 means that the minimum eigenvalue of the 2x2 normal matrix of optical flow equations is greater than 10.0.
        plot=False,  # plot = True means that the accumulators are plotted.
        rgb=True,  # rgb = True means that the RGB mask is shown.
        size=10,  # size = 10 means that the size of the accumulator for directions map is 10x10.
    ):
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flags = flags
        self.threshold = threshold
        self.plot = plot
        self.rgb = rgb
        self.size = size


def opticalflow_determine(frame, mag, ang, configs, fallornot):
    global TP_0, TN_0, FP_0, FN_0, Mis1_0, Mis2_0, Correct_0
    directions_map = np.zeros([configs.size, 5])
    result = 0
    ang_180 = ang / 2
    move_sense = ang[mag > configs.threshold]
    move_mode = mode(move_sense)[0]
    if 10 < move_mode <= 100:
        directions_map[-1, 0] = 1
        directions_map[-1, 1:] = 0
        directions_map = np.roll(directions_map, -1, axis=0)
    elif 100 < move_mode <= 190:
        directions_map[-1, 1] = 1
        directions_map[-1, :1] = 0
        directions_map[-1, 2:] = 0
        directions_map = np.roll(directions_map, -1, axis=0)
    elif 190 < move_mode <= 280:
        directions_map[-1, 2] = 1
        directions_map[-1, :2] = 0
        directions_map[-1, 3:] = 0
        directions_map = np.roll(directions_map, -1, axis=0)
    elif 280 < move_mode or move_mode < 10:
        directions_map[-1, 3] = 1
        directions_map[-1, :3] = 0
        directions_map[-1, 4:] = 0
        directions_map = np.roll(directions_map, -1, axis=0)
    else:
        directions_map[-1, -1] = 1
        directions_map[-1, :-1] = 0
        directions_map = np.roll(directions_map, 1, axis=0)
    loc = directions_map.mean(axis=0).argmax()
    if loc == 0:
        result = 1
    if result != 1:
        if result == fallornot:
            Correct_0 += 1
        else:
            Mis2_0 += 1
    else:
        Mis1_0 += 1
    return result


class cnn_model:
    def __init__(self, loadfromfile=False):
        if loadfromfile:
            self.model = keras.models.load_model("cnn_whale.h5")
        else:
            n_classes = 11
            # create resnet50 model
            base_model = MobileNetV2(
                weights="imagenet", include_top=False, input_shape=(None, None, 3)
            )  # ! todo: fixed input size, dimension
            x = base_model.output
            x = GlobalMaxPooling2D()(x)
            x = Dense(512, activation="relu")(x)
            predictions = Dense(n_classes, activation="softmax")(x)
            self.model = Model(inputs=base_model.input, outputs=predictions)
        self.model.compile(optimizer=legacy.Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

    def train(self, x, y):
        self.model.fit(x, y, epochs=3, batch_size=16)

    def predict(self, velocityPerSample):
        return self.model.predict(velocityPerSample)

    def save(self):
        self.model.save("cnn_whale.h5")


class dnn_model:
    def __init__(self, loadfromfile=False):
        if loadfromfile:
            self.model = keras.models.load_model("dnn_whale.h5")
        else:
            n_classes = 11
            # input: vector 2
            # output: vector 11
            self.model = Sequential()
            self.model.add(Dense(512, input_dim=2, activation="relu"))
            self.model.add(Dense(n_classes, activation="softmax"))
        self.model.compile(optimizer=legacy.Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

    def train(self, x, y):
        self.model.fit(x, y, epochs=1, batch_size=16)

    def predict(self, velocityPerSample):
        return self.model.predict(velocityPerSample)

    def save(self):
        self.model.save("dnn_whale.h5")


MARK = True
TRAIN = False
batchInputs = []
batchInputsDNN = []
batchOutputs = []
workingModel = cnn_model(loadfromfile=True)
workingModelDNN = dnn_model(loadfromfile=True)


def masscenter_determine(optFrame, ang, mag, action):
    global batchInputs, batchOutputs, workingModel, batchInputsDNN, workingModelDNN, MARK, TRAIN, TP_1, TN_1, FP_1, FN_1, Mis1_1, Mis2_1, Correct_1, Mis1_0, Mis2_0, Correct_0
    if TRAIN:
        arr = [0.0] * 11
        arr[action] = 1.0
        batchInputs.append(optFrame)
        batchOutputs.append(arr)
        batchInputsDNN.append([ang, mag])
        return action
    else:
        result = np.argmax(workingModelDNN.predict([[ang, mag]])[0])
        # cnn_result = np.argmax(workingModel.predict([optFrame])[0])
        # if cnn_result != 1:
        # if cnn_result == action:
        # Correct_0 += 1
        # else:
        # Mis2_0 += 1
        # else:
        # Mis1_0 += 1

        if result != 1:
            if result == action:
                Correct_1 += 1
            else:
                Mis2_1 += 1
        else:
            Mis1_1 += 1
        return result


def process_video_stream(stream_source, configs, action, ismark):
    global batchInputs, batchOutputs, workingModel, batchInputsDNN, workingModelDNN, MARK, TRAIN, TP, TN, FP, FN
    param = {
        "pyr_scale": configs.pyr_scale,
        "levels": configs.levels,
        "winsize": configs.winsize,
        "iterations": configs.iterations,
        "poly_n": configs.poly_n,
        "poly_sigma": configs.poly_sigma,
        "flags": configs.flags,
    }
    pose = mp.solutions.pose.Pose(min_detection_confidence=0.55, min_tracking_confidence=0.55)

    frame_previous = stream_source.readNextImage()
    gray_previous = cv.cvtColor(frame_previous, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame_previous)
    hsv[:, :, 1] = 255
    lastMassCenter = [0, 0, 0]
    registered_const = False

    # read falling or not data from file according to the mark function
    numbers = []
    if ismark:
        with open(f"markers/{stream_source.folderName}.txt", "r") as f:
            numbers = [int(num) for line in f for num in line.strip().split()]

    i = 0
    while True:
        frame = stream_source.readNextImage()
        if frame is None:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame.flags.writeable = False
        results = pose.process(frame)
        if results.pose_landmarks is None:
            continue

        flow = cv.calcOpticalFlowFarneback(gray_previous, gray, None, **param)  # TODO: use cnn
        gray_previous = gray
        mag, ang = cv.cartToPolar(
            flow[:, :, 0], flow[:, :, 1], angleInDegrees=True
        )  # TODO: use dnn, comparison with mediapipe
        ang_180 = ang / 2
        # convert flow to vector3 array
        flow = np.dstack((flow, np.zeros_like(flow[:, :, 0])))
        flow = np.expand_dims(flow, axis=0).tolist()[0]
        # optresult = opticalflow_determine(frame, mag, ang, configs, action)

        ang = float(np.mean(ang.flatten()))
        massResult = masscenter_determine(flow, ang, float(np.mean(mag.flatten())), numbers[i] if ismark else action)

        print(massResult)
        text = f"{TRAIN  == True and 'mc' or 'dnn'}: {massResult and 'FALL' or 'OK'}"

        hsv[:, :, 0] = ang_180
        hsv[:, :, 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        rgb = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        frame.flags.writeable = True

        frame = cv.flip(frame, 1)
        cv.putText(frame, text, (30, 90), cv.FONT_HERSHEY_COMPLEX, frame.shape[1] / 500, (0, 255, 255), 2)

        k = cv.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if configs.rgb:
            cv.imshow("Mask", cv.flip(rgb, 1))
        cv.imshow("Frame", frame)
        i += 1

    stream_source.dispose()
    if configs.plot:
        plt.ioff()
    cv.destroyAllWindows()


# read a video stream from class STREAM_READ_UR_FALL and process it, let user choose whether the current frame is a fall or not and save the result to a file
def mark_video_stream(stream_source):

    i = 0
    while True:
        frame = stream_source.readNextImage()
        if frame is None:
            break
        frame.flags.writeable = False
        # display the image
        cv.imshow("Frame", frame)
        k = cv.waitKey() & 0xFF
        if k == ord("q"):
            break
        if k == ord("y"):
            with open(f"markers/{stream_source.folderName}.txt", "a") as f:
                f.write(f"1 ")
        else:
            with open(f"markers/{stream_source.folderName}.txt", "a") as f:
                f.write(f"0 ")
        i += 1


def trainModels():
    global batchInputs, batchOutputs, workingModel, batchInputsDNN, workingModelDNN
    workingModel.train(batchInputs, batchOutputs)
    workingModel.save()
    # workingModelDNN.train(batchInputsDNN, batchOutputs)
    # workingModelDNN.save()
    batchInputs = []
    batchOutputs = []
    batchInputsDNN = []


if __name__ == "__main__":
    if MARK:
        for i in range(15, 16):
            mark_video_stream(STREAM_READ_UR_FALL(f"fall-{i:02d}-cam0-rgb", 30))
        exit(0)
    configs = CONFIGURATIONS()

    for i in range(1, 2):
        cap = STREAM_READ_UR_FALL(f"fall-{i:02d}-cam0-rgb", 30)
        if configs.plot:
            plt.ion()
        process_video_stream(cap, configs, 0, True)
        if TRAIN:
            trainModels()

    i = 0
    for filename in os.listdir(str(Path.home() / "Downloads/Florence_3d_actions/")):
        if i > 1 and TRAIN:
            break
        if not filename.endswith(".avi"):
            continue
        idGesture, idActor, idAction, idCategory = (
            int(i) for i in parse.parse("GestureRecording_Id{}actor{}idAction{}category{}.avi", filename)
        )
        if idCategory != 2:
            continue
        print(
            f"Processing {filename} with idGesture: {idGesture}, idActor: {idActor}, idAction: {idAction}, idCategory: {idCategory}"
        )
        cap = STREAM_READ_FLORENCE_FALL(30, idGesture, idActor, idAction, idCategory)
        process_video_stream(cap, configs, idCategory, False)
        if TRAIN:
            trainModels()
        i += 1
    if not TRAIN:
        print(f"CNN method. Mis1: {Mis1_0}, Mis2: {Mis2_0}, Correct: {Correct_0}")
        print(f"DNN method. Mis1: {Mis1_1}, Mis2: {Mis2_1}, Correct: {Correct_1}")
