from utils import (
    DOWNLOAD_DIRECTORY,
    enum_ur_fall,
    enum_florence_3d,
    parse_florence_3d_name,
    get_center_of_mass,
    Configs,
    get_landmark,
    putText,
    flatten_landmark,
    Mediapipe_Person,
)
from stream import VideoHandler, VideoStream, FolderHandler
from models import DNNModel, DecisionTree, yolo_detect, yolo_wrap_detection, yolo_draw
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import cv2
import mediapipe as mp
import numpy as np
import math


def mediapipe_dnn_stream(configs):
    pose = mp.solutions.pose.Pose(min_detection_confidence=0.55, min_tracking_confidence=0.55)
    model = DNNModel(loadfromfile=True)
    clf = DecisionTree(loadfromfile=True)

    y_vals = []
    test_vals = []
    test_clf_vals = []
    train_x = []
    train_y = []

    for (folder_name, full_name) in enum_ur_fall(0, 1):
        stream = VideoStream(FolderHandler(full_name, configs, suffix=".png"), configs)
        aggregate_landmarks = []
        numbers = []
        with open(f"markers/{folder_name}.txt", "r") as f:
            numbers = [int(num) for line in f for num in line.strip().split()]

        frame = -1
        while True:
            frame += 1
            img = stream.get_image()
            if img is None:
                break
            results = pose.process(img)
            if results.pose_landmarks is None:
                continue
            aggregate_landmarks.append(flatten_landmark(get_landmark(results.pose_landmarks.landmark, configs)))

            if len(aggregate_landmarks) > configs.consecutive_frame_count:
                aggregate_landmarks.pop(0)
            elif len(aggregate_landmarks) < configs.consecutive_frame_count:
                continue

            flatten = [val for landmarks in aggregate_landmarks for val in landmarks]

            y_vals.append(numbers[frame])

            if configs.train:
                train_x.append(flatten)
                y_sample = [0] * 11
                if numbers[frame] == 0:
                    y_sample[0] = 1
                elif numbers[frame] == 1:
                    y_sample[1] = 1
                train_y.append(y_sample)
                putText(img, f"training... y_sam={numbers[frame]}")
            else:
                print(model.predict([flatten])[0])
                test_vals.append(0 if np.argmax(model.predict([flatten])[0]) == 0 else 1)
                test_clf_vals.append(clf.predict([flatten])[0])
                putText(img, f"act = {test_vals[len(test_vals)-1]}")
            cv2.imshow("mediapipe stream", img)
            cv2.waitKey(10)

        stream.dispose()
    if configs.train:
        model.train(train_x, train_y)
        model.save()
        clf.train(train_x, y_vals)
        clf.save()
    else:
        cf = confusion_matrix(y_vals, test_vals)
        print("dnn (tn, fp, fn, tp) = ", cf.ravel())
        cf = confusion_matrix(y_vals, test_clf_vals)
        print("clf (tn, fp, fn, tp) = ", cf.ravel())
        print(precision_recall_fscore_support(y_vals, test_vals))


def multiple_stream(configs):

    pose = mp.solutions.pose.Pose(min_detection_confidence=0.55, min_tracking_confidence=0.55)

    net = cv2.dnn.readNet("yolov5s.onnx")
    stream = VideoStream(VideoHandler(str(DOWNLOAD_DIRECTORY / "video.mp4"), configs), configs)
    frame = -1

    test_people = []
    while True:
        frame += 1
        img = stream.get_image()
        if img is None:
            break
        img = cv2.resize(img, (640, 640))
        img = cv2.flip(img, 1)
        maxheight, maxwidth, _ = img.shape

        predictions = yolo_detect(img, net)

        class_ids, confidences, boxes = yolo_wrap_detection(img, predictions[0])
        yolo_draw(img, class_ids, confidences, boxes)

        people_count = 0

        for (classid, confidence, box) in zip(class_ids, confidences, boxes):
            if classid == 0:
                left, top, width, height = box
                results = pose.process(
                    img[
                        max(0, top - 30) : min(maxheight, top + height + 30),
                        max(0, left - 30) : min(maxwidth, left + width + 30),
                    ]
                )
                if results.pose_landmarks is None:
                    continue
                people_count += 1
                translated = get_landmark(results.pose_landmarks.landmark, configs)
                mass_center = get_center_of_mass(translated)
                rect_center = (top + height / 2, left + width / 2)
                person_id = -1
                for id in range(0, len(test_people)):
                    dist = np.linalg.norm(np.subtract(test_people[id].rect_center, rect_center))
                    if dist < 60:
                        person_id = id
                        break

                if person_id == -1:
                    test_people.append(Mediapipe_Person(rect_center))
                else:
                    test_people[person_id].rect_center = rect_center
                cv2.putText(
                    img,
                    "P" + str(person_id),
                    (round(box[0]) - 10, round(box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2,
                )
                for tup in translated:
                    x, y = math.floor(left + tup[0] * width), math.floor(top + tup[1] * height)
                    if x < 0 or y < 0:
                        continue
                    cv2.circle(
                        img,
                        (x, y),
                        2,
                        (180, 255, 255),
                        thickness=6,
                    )
        if people_count == 0:
            results = pose.process(img)
            if results.pose_landmarks is not None:
                for tup in translated:
                    x, y = math.floor(left + tup[0] * width), math.floor(top + tup[1] * height)
                    if x < 0 or y < 0:
                        continue
                    cv2.circle(
                        img,
                        (x, y),
                        2,
                        (180, 255, 255),
                        thickness=6,
                    )

        cv2.imshow("live stream", img)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            # breaking the loop if the user types q
            # note that the video window must be highlighted!
            break

    stream.dispose()


def main():
    cv2.startWindowThread()
    configs = Configs(render=True, train=False, consecutive_frame_count=5)
    mediapipe_dnn_stream(configs)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
