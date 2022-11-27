import time
import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


currentDelta = 0


def getFPS():
    global currentDelta
    ctime = time.time()
    currentDelta = ctime - getFPS.ptime
    fps = int(1/(currentDelta))
    getFPS.ptime = ctime
    return fps


getFPS.ptime = time.time()


def processLandmarks(image, results):
    centerOfMass = [0, 0]
    centerOfPixel = [0, 0]
    length = 1
    for landmark, world_landmark in zip(results.pose_landmarks.landmark, results.pose_world_landmarks.landmark):
        centerOfMass = np.add(
            centerOfMass, [world_landmark.x, world_landmark.y])
        centerOfPixel = np.add(centerOfPixel, [landmark.x, landmark.y])
        length += 1
    centerOfMass = np.divide(centerOfMass, length)
    centerOfPixel = np.divide(centerOfPixel, length)
    h, w, c = image.shape
    print(centerOfMass)
    centerOfPixel = (int(centerOfPixel[0]*w), int(centerOfPixel[1]*h))
    cv2.circle(image, centerOfPixel, 4, (255, 0, 0),
               thickness=6, lineType=8, shift=0)
    return centerOfMass


cap = cv2.VideoCapture(0)
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks == None:
            continue

        processLandmarks(image, results)
        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        image = cv2.flip(image, 1)
        cv2.putText(image, "FPS = " + str(getFPS()), (25, 60),
                    cv2.FONT_HERSHEY_PLAIN, 2, (25, 50, 25), 2)
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
