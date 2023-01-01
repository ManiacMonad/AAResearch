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
    while True:
        currentFrame += 1
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
        deltaLandmarkVelocity = list(np.array(deltaLandmarkVelocity).flatten())
        # deltaLandmarkVelocity.append(delatMassCenter[0] * FPS)
        # deltaLandmarkVelocity.append(delatMassCenter[1] * FPS)

        deltaLandmarkVelocity = list(np.array(deltaLandmarkVelocity).flatten()) + list(
            np.array(relativePixels).flatten()
        )
        massCenterVelocity = np.linalg.norm(delatMassCenter * FPS)
        falling = determineAndTrain(CONFIGURATIONS, massCenterVelocity, deltaLandmarkVelocity)
        if CONFIGURATIONS.RENDER_IMAGE:
            putText(image, f"falling: {falling}", pos=(10, 30), text_color=(255, 0, 0))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("MediaPipe Pose", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                continue
    streamObject.dispose()


pass_Lambda = False
X_TRAIN_PERCENTAGE = 0.1


def main():
    global pass_Lambda
    CONFIGURATIONS = BASE_CONFIGURATIONS()
    workingModel = mp_model(loadfromfile=False)
    batchInputs = []
    batchOutputs = []

    if CONFIGURATIONS.LOAD_FROM_FOLDER:
        # UR_FALL
        fileCount = 0
        passCount = 0

        def determine(CONFIGURATIONS, massCenterVelocity, deltaLandmarkVelocity) -> float:
            global pass_Lambda
            if CONFIGURATIONS.TRAIN:

                arr = [0.0] * 11
                if CONFIGURATIONS.NO_FALL:
                    arr[0] = 1.0
                    batchInputs.append(deltaLandmarkVelocity)
                    batchOutputs.append(arr)
                    return 0
                elif massCenterVelocity >= 1.55:
                    arr[1] = 1.0
                    batchInputs.append(deltaLandmarkVelocity)
                    batchOutputs.append(arr)
                    return 1
                else:
                    if massCenterVelocity < 0.8:
                        arr[0] = 1.0
                        batchInputs.append(deltaLandmarkVelocity)
                        batchOutputs.append(arr)
                    return 0
            else:
                if pass_Lambda:
                    return 1
                predictValue = np.argmax(workingModel.predict([deltaLandmarkVelocity])[0])
                if predictValue == 1:
                    pass_Lambda = True
                return predictValue

        # Training
        print(f"Beginnning the training process on {X_TRAIN_PERCENTAGE * 100}% of UR_FALL dataset")
        CONFIGURATIONS.TRAIN = True
        totalIndex = 30
        endingIndex = int(totalIndex * X_TRAIN_PERCENTAGE)
        for folderIndex in range(0, endingIndex + 1):
            processStream(
                determine,
                STREAM_READ_UR_FALL(f"fall-{folderIndex:02d}-cam0-rgb", 30),
                CONFIGURATIONS,
            )
        print(f"Preparing to fit the model with {len(batchInputs)} inputs and {len(batchOutputs)} outputs")
        workingModel.train(batchInputs, batchOutputs)
        print(f"Training on {X_TRAIN_PERCENTAGE * 100}% of UR_FALL dataset complete")

        # Testing
        print("Training complete, beginning testing on Florence dataset")
        CONFIGURATIONS.TRAIN = False
        for folderIndex in range(endingIndex + 1, totalIndex + 1):
            pass_Lambda = False
            fileCount += 1
            processStream(
                determine,
                STREAM_READ_UR_FALL(f"fall-{folderIndex:02d}-cam0-rgb", 30),
                CONFIGURATIONS,
            )
            if pass_Lambda:
                passCount += 1
        for filename in os.listdir(str(Path.home() / "Downloads/Florence_3d_actions/")):
            if not filename.endswith(".avi"):
                continue
            fileCount += 1
            idGesture, idActor, idAction, idCategory = (
                int(i) for i in parse.parse("GestureRecording_Id{}actor{}idAction{}category{}.avi", filename)
            )
            print(
                f"Processing {filename} with idGesture: {idGesture}, idActor: {idActor}, idAction: {idAction}, idCategory: {idCategory}"
            )
            pass_Lambda = True

            def determine_florence(CONFIGURATIONS, massCenterVelocity, deltaLandmarkVelocity) -> float:
                global pass_Lambda
                if CONFIGURATIONS.TRAIN:
                    arr = [0.0] * 11
                    arr[idCategory + 1] = 1.0
                    batchInputs.append(deltaLandmarkVelocity)
                    batchOutputs.append(arr)
                    return idCategory
                else:
                    if pass_Lambda == False:
                        return 0
                    predictValue = np.argmax(workingModel.predict([deltaLandmarkVelocity])[0])
                    if predictValue == 1:  # detect falling
                        pass_Lambda = False
                    return predictValue

            processStream(
                determine_florence,
                STREAM_READ_FLORENCE_FALL(30, idGesture, idActor, idAction, idCategory),
                CONFIGURATIONS,
            )
            if pass_Lambda:
                passCount += 1
        print(f"Pass percentage: {passCount / fileCount * 100}% ({passCount}/{fileCount})")
    else:
        processStream(determine, VIDEO_STREAM(), CONFIGURATIONS)

    # print(f"Average percentage: { (niceCount / fileCount) * 100 }% ({niceCount}/{fileCount})")

    if CONFIGURATIONS.TRAIN:
        workingModel.train(batchInputs, batchOutputs)
        workingModel.save()


main()
