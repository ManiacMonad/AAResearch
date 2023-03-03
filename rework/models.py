import cv2
import numpy as np
import math
import pickle
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import legacy
from sklearn import tree


class DecisionTree:
    filename = "decision_tree_sklearn.pickle"

    def __init__(self, loadfromfile=False) -> None:
        if loadfromfile:
            self.model = pickle.load(open(DecisionTree.filename, "rb"))
        else:
            self.model = tree.DecisionTreeClassifier()

    def train(self, x, y):
        self.model = self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def save(self):
        pickle.dump(self.model, open(DecisionTree.filename, "wb"))


class DNNModel:
    def __init__(self, loadfromfile=False):
        if loadfromfile:
            self.model = keras.models.load_model("DNN_consecutive.h5")

        else:
            self.model = Sequential(
                [
                    Dense(512, input_shape=(330,), activation="relu"),
                    Dense(512, activation="relu"),
                    Dense(256, activation="relu"),
                    Dense(11, activation="softmax"),
                ]
            )
        self.model.compile(optimizer=legacy.Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

    def train(self, x, y):
        self.model.fit(x, y, epochs=150, batch_size=128)

    def predict(self, velocityPerSample):
        return self.model.predict(velocityPerSample)

    def save(self):
        print("Overriding DNNModel file")
        self.model.save("DNN_consecutive.h5")


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
