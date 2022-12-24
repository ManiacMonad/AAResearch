import cv2
import mediapipe as mp
import numpy as np
from typing import Callable, Tuple, List
from utils import putText, processLandmarks, getLandmarkVelocity
from mp_model import mp_model, STREAM_READ_UR_FALL, VIDEO_STREAM
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
        deltaLandmarkVelocity.append(delatMassCenter[0] * FPS)
        deltaLandmarkVelocity.append(delatMassCenter[1] * FPS)

        deltaLandmarkVelocity = list(np.array(deltaLandmarkVelocity).flatten()) + list(
            np.array(relativePixels).flatten()
        )
        massCenterVelocity = np.linalg.norm(delatMassCenter * FPS)
        falling = determineAndTrain(CONFIGURATIONS, massCenterVelocity, deltaLandmarkVelocity)
        if CONFIGURATIONS.RENDER_IMAGE:
            putText(image, f"falling: {falling}", pos=(10, 30), text_color=(255, 0, 0) if falling else (0, 255, 0))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow("MediaPipe Pose", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    streamObject.dispose()


def main():
    CONFIGURATIONS = BASE_CONFIGURATIONS()
    workingModel = mp_model(loadfromfile=True)
    batchInputs = []
    batchOutputs = []

    def determine(CONFIGURATIONS, massCenterVelocity, deltaLandmarkVelocity) -> bool:
        if CONFIGURATIONS.TRAIN:
            if CONFIGURATIONS.NO_FALL:
                batchInputs.append(deltaLandmarkVelocity)
                batchOutputs.append([0.0])
                return False
            elif massCenterVelocity >= 1.55:
                batchInputs.append(deltaLandmarkVelocity)
                batchOutputs.append([1.0])
                return True
            else:
                if massCenterVelocity < 0.8:
                    batchInputs.append(deltaLandmarkVelocity)
                    batchOutputs.append([0.0])
                return False
        else:
            predictValue = workingModel.predict([deltaLandmarkVelocity])[0][0]
            return True if predictValue >= 0.5 else False

    if CONFIGURATIONS.LOAD_FROM_FOLDER:
        for folderIndex in range(10, 30 + 1):
            processStream(
                determine,
                STREAM_READ_UR_FALL(f"fall-{folderIndex:02d}-cam0-rgb", 30),
                CONFIGURATIONS,
            )
    else:
        processStream(determine, VIDEO_STREAM(), CONFIGURATIONS)

    if CONFIGURATIONS.TRAIN:
        workingModel.train(batchInputs, batchOutputs)
        workingModel.save()


main()
