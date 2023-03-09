import cv2
import numpy as np
import math
import pickle
from xgboost import XGBClassifier
from utils import MODEL_TYPES, get_center_of_mass
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import legacy
from sklearn import tree
import mediapipe as mp
from configs import GLOBAL_CONFIGS
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose


class DecisionTree:
    def __init__(self, configs, loadfromfile=False) -> None:
        self.configs = configs
        self.type = MODEL_TYPES.Mediapipe_CLF
        self.filename = f"models/{GLOBAL_CONFIGS.input_str}_decision_tree_cons_{configs.consecutive_frame_count:02d}.h5"
        if loadfromfile:
            self.model = pickle.load(open(self.filename, "rb"))
        else:
            self.model = tree.DecisionTreeClassifier()

    def train(self, x, y):
        self.model = self.model.fit(x, [self.conv_train_to_format(sample) for sample in y])

    def predict(self, input):
        val = self.model.predict([self.conv_to_format_input(x) for x in input])
        batch = []
        for samp in val:
            arr = [0.0] * 11
            arr[samp] = 1.0
            batch.append(arr)
        return batch

    def visualize(self):
        tree.plot_tree(self.model)
        plt.show()

    def to_text(self):
        return tree.export_text(self.model)

    def conv_to_format_input(self, flatten):
        return flatten

    def conv_train_to_format(self, train_y_array):
        return 1 if train_y_array[1] == 1 else 0

    def save(self):
        pickle.dump(self.model, open(self.filename, "wb"))


class XGBoostModel:
    def __init__(self, configs, loadfromfile=False) -> None:
        self.configs = configs
        self.type = MODEL_TYPES.Mediapipe_XGBoost
        self.filename = f"models/{GLOBAL_CONFIGS.input_str}_xgboost_cons_{configs.consecutive_frame_count:02d}.h5"
        if loadfromfile:
            self.model = pickle.load(open(self.filename, "rb"))
        else:
            self.model = XGBClassifier(n_estimators=100, learning_rate=0.3)

    def train(self, x, y):
        self.model = self.model.fit(x, [self.conv_train_to_format(sample) for sample in y])

    def predict(self, input):
        val = self.model.predict([self.conv_to_format_input(x) for x in input])
        batch = []
        for samp in val:
            arr = [0.0] * 11
            arr[samp] = 1.0
            batch.append(arr)
        return batch

    def conv_to_format_input(self, flatten):
        return flatten

    def conv_train_to_format(self, train_y_array):
        return 1 if train_y_array[1] == 1 else 0

    def save(self):
        pickle.dump(self.model, open(self.filename, "wb"))


class ManualDecisionTree:
    def __init__(self, configs, com_mag_threshold, torso_mag_threshold, loadfromfile=False) -> None:
        self.type = MODEL_TYPES.Manual_CLF
        self.configs = configs
        self.com_mag_threshold = com_mag_threshold
        self.torso_mag_threshold = torso_mag_threshold

    def train(self, x, y):
        pass

    def predict(self, input):
        return self.sub_predict([self.conv_to_format_input(x) for x in input])

    def sub_predict(self, val):
        batch = []
        for x in val:
            arr = [0.0] * 11
            com_mag, torso_mag = x
            arr[1] = 1.0
            if com_mag < self.com_mag_threshold or torso_mag < self.torso_mag_threshold:
                arr[0] = 1.0
                arr[1] = 0.0

            batch.append(arr)

        return batch

    def conv_to_format_input(self, flatten):
        return flatten

    def conv_train_to_format(self, train_y_array):
        return 1 if train_y_array[1] == 1 else 0

    def save(self):
        pass


class DNNModel:
    def __init__(self, configs, loadfromfile=False):
        self.type = MODEL_TYPES.Mediapipe_DNN
        self.configs = configs
        self.filename = f"models/{GLOBAL_CONFIGS.input_str}_Mediapipe_DNN_cons_{configs.consecutive_frame_count:02d}.h5"
        if loadfromfile:
            self.model = keras.models.load_model(self.filename)

        else:
            self.model = Sequential(
                [
                    Dense(
                        512,
                        input_shape=(
                            (
                                2 * (configs.consecutive_frame_count - 1)
                                if GLOBAL_CONFIGS.input_str == "proc"
                                else 68 * (configs.consecutive_frame_count)
                            ),
                        ),
                        activation="relu",
                    ),
                    Dense(512, activation="relu"),
                    Dense(512, activation="relu"),
                    Dense(11, activation="softmax"),
                ]
            )
        self.model.compile(optimizer=legacy.Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

    def train(self, x, y):
        self.model.fit(x, y, epochs=500, batch_size=128)

    def predict(self, input):
        return self.model.predict([self.conv_to_format_input(x) for x in input])

    def conv_to_format_input(self, flatten):
        return flatten

    def save(self):
        print("Overriding DNNModel file")
        self.model.save(self.filename)


def yolo_detect(image, net):
    blob = cv2.dnn.blobFromImage(format_yolov5(image), 1 / 255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    return preds


def format_yolov5(frame):

    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result


def yolo_wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / 640
    y_factor = image_height / 640

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if classes_scores[class_id] > 0.25:

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes


yolo_classes = None
with open("coco.names", "r") as f:
    yolo_classes = [line.strip() for line in f.readlines()]

yolo_colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]


def yolo_draw(img, class_ids, confidences, boxes):
    global yolo_classes, yolo_colors
    for (classid, confidence, box) in zip(class_ids, confidences, boxes):

        color = yolo_colors[int(classid) % len(yolo_colors)]
        cv2.rectangle(img, box, color, 2)
        cv2.rectangle(img, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
