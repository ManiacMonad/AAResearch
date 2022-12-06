import time
import math
import cv2
import glob
import mediapipe as mp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def putText(image, text, pos, font=cv2.FONT_HERSHEY_PLAIN, scale=1, color=(25, 50, 25), thick=2):
    cv2.putText(image, text, pos, font, scale, color, thick)


def processLandmarks(image, results):
    centerOfMass = [0, 0, 0]
    centerOfPixel = [0, 0, 0]
    length = 0
    useful = [mp_pose.PoseLandmark.NOSE,
              mp_pose.PoseLandmark.RIGHT_SHOULDER,
              mp_pose.PoseLandmark.LEFT_HIP,
              mp_pose.PoseLandmark.RIGHT_HIP]
    for i in range(0, 32+1):
        landmark = results.pose_landmarks.landmark[i]
        world_landmark = results.pose_world_landmarks.landmark[i]
        centerOfMass = np.add(
            centerOfMass, [world_landmark.x, world_landmark.y, world_landmark.z])
        centerOfPixel = np.add(
            centerOfPixel, [landmark.x, landmark.y, landmark.z])
        length += 1
    centerOfMass = np.divide(centerOfMass, length)
    centerOfPixel = np.divide(centerOfPixel, length)
    h, w, c = image.shape
    print(centerOfPixel)
    cv2.circle(image, (int(centerOfPixel[0]*w), int(centerOfPixel[1]*h)), 4, (255, 120, 120),
               thickness=16, lineType=8, shift=0)
    return centerOfMass, centerOfPixel


with mp_pose.Pose(
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55) as pose:
    for folderIndex in range(1, 15 + 1):
        plt.clf()
        ax = plt.subplot()
        ax.grid()
        folderName = f"fall-{folderIndex:02d}-cam0-rgb"
        currentFrame = 0
        lastMass = [0, 0, 0]
        lastPixelCenter = [0, 0, 0]
        prevFrame = 0
        plotXAxis = []
        velocityData = []
        pixelVelData = []
        while (True):
            imageName = str(
                Path.home() / (f"Downloads/ur_fall/{folderName}/{folderName}-{(currentFrame+1):03d}.png"))
            image = cv2.imread(imageName)
            if image is None:
                break
            # while cap.isOpened():
            #     success, image = cap.read()
            #     if not success:
            #         print("Ignoring empty camera frame.")
            #         continue
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            if results.pose_landmarks == None:
                break
            # 找出重心與影像中的重心
            massCenter, pixelCenter = processLandmarks(image, results)
            # 畫節點與骨頭
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # 影像鏡像處理
            image = cv2.flip(image, 1)
            # 計算現在重心與上個重心的delta
            deltaMassCenter = np.subtract(massCenter, lastMass)  # m
            deltaPixelCenter = np.subtract(
                pixelCenter, lastPixelCenter)  # ratio
            # 找出速率 m/s
            velocity = np.linalg.norm(deltaMassCenter) * 30
            # 找出圖像速率 ratio/s
            pixelVelocity = np.linalg.norm(deltaPixelCenter) * 30
            # 畫上幀率(更新率)
            putText(
                image, f"{int(currentFrame):03d} {math.floor(velocity*100):03d} cm/s {int(pixelVelocity * 100):03d} ratio/s", (50, 30), scale=0.9)
            cv2.imshow('MediaPipe Pose', image)
            # 畫圖表x=time y=vel
            if currentFrame - prevFrame >= 1:
                prevFrame = currentFrame
                plotXAxis.append(currentFrame)
                pixelVelData.append(pixelVelocity)
                velocityData.append(velocity)
                ax.plot(plotXAxis, velocityData, "b-")
                ax.plot(plotXAxis, pixelVelData, 'g-')
                plt.pause(0.001)

            if cv2.waitKey(5) & 0xFF == 27:
                break
            lastMass = massCenter
            lastPixelCenter = pixelCenter
            currentFrame += 1
# cap.release()
print("Done")
