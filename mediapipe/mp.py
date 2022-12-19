import math
import time
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from utils import putText, processLandmarks
from mp_model import mp_model
from pathlib import Path


def processStream(machineLearningModel, folderName, batchInputs, batchOutputs, TRAIN):

    # Configurations
    FPS = 30

    capture = None
    if folderName is None:
        capture = cv2.VideoCapture(0)

    currentFrame = 0
    pose = mp.solutions.pose.Pose(min_detection_confidence=0.55, min_tracking_confidence=0.55)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    lastMassCenter = [0, 0, 0]
    lastLandmarkPositions = []
    falling = False
    while capture.isOpened() if folderName is None else True:
        currentFrame += 1
        image = None
        if folderName is None:
            ret, image = capture.read()
            if ret is False:
                print("Failed to read from camera")
                continue
        else:
            imageName = str(Path.home() / f"Downloads/ur_fall/{folderName}/{folderName}-{currentFrame:03d}.png")
            image = cv2.imread(imageName)
            if image is None:
                print(f"Failed to read image {imageName}")
                break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if results.pose_landmarks == None:
            break

        massCenter, pixelCenter, relativePixels = processLandmarks(image, results)
        image.flags.writeable = True
        if currentFrame == 1:
            lastMassCenter = massCenter
            for i in range(0, 32 + 1):
                landmark = results.pose_landmarks.landmark[i]
                lastLandmarkPositions.append([landmark.x, landmark.y])
            continue

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )
        image = cv2.flip(image, 1)

        delatMassCenter = np.subtract(massCenter, lastMassCenter)
        lastMassCenter = massCenter
        deltaLandmarkVelocity = []
        for i in range(0, 32 + 1):
            currentLandmark = results.pose_landmarks.landmark[i]
            delta = np.subtract(
                [currentLandmark.x, currentLandmark.y],
                [lastLandmarkPositions[i][0], lastLandmarkPositions[i][1]],
            )
            deltaLandmarkVelocity.append(delta * FPS)
            lastLandmarkPositions[i] = [currentLandmark.x, currentLandmark.y]
        deltaLandmarkVelocity = list(np.array(deltaLandmarkVelocity).flatten())
        deltaLandmarkVelocity.append(delatMassCenter[0] * FPS)
        deltaLandmarkVelocity.append(delatMassCenter[1] * FPS)

        deltaLandmarkVelocity = list(np.array(deltaLandmarkVelocity).flatten()) + list(
            np.array(relativePixels).flatten()
        )
        massCenterVelocity = np.linalg.norm(delatMassCenter * FPS)
        if TRAIN:
            if massCenterVelocity >= 1.2:
                batchInputs.append(deltaLandmarkVelocity)
                falling = True
                batchOutputs.append([1.0])
            elif massCenterVelocity < 0.6:
                batchInputs.append(deltaLandmarkVelocity)
                batchOutputs.append([0.0])
        else:
            predictValue = machineLearningModel.predict([deltaLandmarkVelocity])[0][0]
            if predictValue >= 0.5:
                falling = True
            else:
                falling = False
        putText(image, f"falling: {falling}", pos=(10, 30), text_color=(255, 0, 0) if falling else (0, 255, 0))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("MediaPipe Pose", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    if capture is not None:
        capture.release()
    cv2.destroyAllWindows()


def main():
    TRAIN = False
    LOAD_FROM_VIDEO = True
    workingModel = mp_model(loadfromfile=not TRAIN)
    batchInputs = []
    batchOutputs = []
    if LOAD_FROM_VIDEO:
        for folderIndex in range(0, 20 + 1):
            folderName = f"fall-{folderIndex:02d}-cam0-rgb"
            processStream(workingModel, folderName, batchInputs, batchOutputs, TRAIN)
    else:
        processStream(workingModel, None, batchInputs, batchOutputs, TRAIN)
    if TRAIN:
        workingModel.train(batchInputs, batchOutputs)
        workingModel.save()


main()
