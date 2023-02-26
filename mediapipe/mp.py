import cv2
import mediapipe as mp
import numpy as np
import parse
import os
from pathlib import Path
from typing import Callable, Tuple, List
from utils import putText, processLandmarks, getLandmarkVelocity
from mp_model import mp_model, STREAM_READ_UR_FALL, VIDEO_STREAM, STREAM_READ_FLORENCE_FALL
from configs import BASE_CONFIGURATIONS


def processStream(
    determineAndTrain: Callable[
        [BASE_CONFIGURATIONS, Tuple[int, int], List[Tuple[int, int]]], bool
    ],  # Lambdacall = (CONFIGURATIONS, massCenterVelocity, deltaLandmarkVelocity) ->
    streamObject,
    CONFIGURATIONS,
    action,
    STOP_FRAME=0,
):

    # Configurations
    FPS = 30

    currentFrame = 0
    registered_const = False
    pose = mp.solutions.pose.Pose(min_detection_confidence=0.55, min_tracking_confidence=0.55)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    lastMassCenter = [0, 0, 0]
    lastLandmarkPositions = []
    falling = False

    # numbers = []
    # with open(f"markers/{streamObject.folderName}.txt", "r") as f:
    # numbers = [int(num) for line in f for num in line.strip().split()]

    while True:
        if STOP_FRAME != 0 and currentFrame > STOP_FRAME:
            break
        image = streamObject.readNextImage()
        if image is None:
            print("Failed to read image")
            break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if results.pose_landmarks == None:
            continue

        massCenter, pixelCenter, relativePixels = processLandmarks(image, results, CONFIGURATIONS)
        image.flags.writeable = True
        if not registered_const:
            registered_const = True
            lastMassCenter = massCenter
            for i in range(0, 32 + 1):
                landmark = results.pose_landmarks.landmark[i]
                lastLandmarkPositions.append([landmark.x, landmark.y])
            continue

        if CONFIGURATIONS.RENDER_IMAGE:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )
            image = cv2.flip(image, 1)

        delatMassCenter = np.subtract(massCenter, lastMassCenter)
        lastMassCenter = massCenter
        deltaLandmarkVelocity = getLandmarkVelocity(
            results.pose_landmarks.landmark, lastLandmarkPositions, FPS, CONFIGURATIONS
        )
        for i in range(0, 32 + 1):
            landmark = results.pose_landmarks.landmark[i]
            lastLandmarkPositions.append([landmark.x, landmark.y])

        deltaLandmarkVelocity = list(np.array(deltaLandmarkVelocity).flatten()) + list(
            np.array(relativePixels).flatten()
        )
        massCenterVelocity = np.linalg.norm(delatMassCenter * FPS)
        falling = determineAndTrain(CONFIGURATIONS, massCenterVelocity, deltaLandmarkVelocity, action)
        if CONFIGURATIONS.RENDER_IMAGE:
            putText(image, f"falling: {falling}", pos=(10, 30), text_color=(255, 0, 0))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("MediaPipe Pose", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                continue
        currentFrame += 1
    streamObject.dispose()


def main():
    CONFIGURATIONS = BASE_CONFIGURATIONS(RENDER_IMAGE=True, TRAIN=False)
    workingModel = mp_model(loadfromfile=True)
    batchInputs = []
    batchOutputs = []

    if CONFIGURATIONS.LOAD_FROM_FOLDER:
        # UR_FALL
        TN = 0
        TP = 0
        FN = 0
        FP = 0

        Mis1 = 0
        Mis2 = 0
        Correct = 0

        def determine(CONFIGURATIONS, massCenterVelocity, deltaLandmarkVelocity, fallOrNot) -> float:
            nonlocal TN, TP, FN, FP, Mis1, Mis2, Correct
            if CONFIGURATIONS.TRAIN:

                arr = [0.0] * 11
                print(f"fallOrNot: {fallOrNot}")
                arr[fallOrNot] = 1.0
                batchInputs.append(deltaLandmarkVelocity)
                batchOutputs.append(arr)
            else:
                # predictValue = np.argmax(workingModel.predict([deltaLandmarkVelocity])[0])
                predictValue = massCenterVelocity > 1.2 and 1 or 0
                print(f"predictValue: {predictValue}, fallOrNot: {fallOrNot}")
                if predictValue != 1:
                    if predictValue == fallOrNot:
                        Correct += 1
                    else:
                        Mis2 += 1
                else:
                    Mis1 += 1
                return predictValue

        # totalIndex = 14
        # for folderIndex in range(1, totalIndex + 1):
        # processStream(determine, STREAM_READ_UR_FALL(f"fall-{folderIndex:02d}-cam0-rgb", 30), CONFIGURATIONS, 0)
        # print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
        # for filename in os.listdir(str(Path.home() / "Downloads/Florence_3d_actions/")):
        #     if not filename.endswith(".avi"):
        #         continue
        #     idGesture, idActor, idAction, idCategory = (
        #         int(i) for i in parse.parse("GestureRecording_Id{}actor{}idAction{}category{}.avi", filename)
        #     )
        #     if idCategory != 2:
        #         continue
        #     print(
        #         f"Processing {filename} with idGesture: {idGesture}, idActor: {idActor}, idAction: {idAction}, idCategory: {idCategory}"
        #     )
        #     processStream(
        #         determine,
        #         STREAM_READ_FLORENCE_FALL(30, idGesture, idActor, idAction, idCategory),
        #         CONFIGURATIONS,
        #         idCategory,
        #     )
        processStream(determine, VIDEO_STREAM(), CONFIGURATIONS, 1)
        print(f"mp dnn: Correct: {Correct}, Mis1: {Mis1}, Mis2: {Mis2}")
    else:
        processStream(determine, VIDEO_STREAM(), CONFIGURATIONS)

    # print(f"Average percentage: { (niceCount / fileCount) * 100 }% ({niceCount}/{fileCount})")
    if CONFIGURATIONS.TRAIN:
        print("Training")
        workingModel.train(
            batchInputs,
            batchOutputs,
        )
        workingModel.save()


main()
