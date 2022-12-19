import cv2
import numpy as np


def putText(
    img,
    text,
    pos=(0, 0),
    font=cv2.FONT_HERSHEY_PLAIN,
    font_scale=3,
    font_thickness=2,
    text_color=(0, 255, 0),
    text_color_bg=(0, 0, 0),
):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(
        img,
        text,
        (x, y + text_h + font_scale - 1),
        font,
        font_scale,
        text_color,
        font_thickness,
    )

    return text_size


def processLandmarks(image, results):
    centerOfMass = [0, 0, 0]
    centerOfPixel = [0, 0, 0]
    length = 0
    minvec = [0, 0]
    maxvec = [0, 0]
    relativePixels = []
    for i in range(0, 32 + 1):
        landmark = results.pose_landmarks.landmark[i]
        world_landmark = results.pose_world_landmarks.landmark[i]
        centerOfMass = np.add(centerOfMass, [world_landmark.x, world_landmark.y, world_landmark.z])
        centerOfPixel = np.add(centerOfPixel, [landmark.x, landmark.y, landmark.z])
        length += 1
        if landmark.x > maxvec[0]:
            maxvec[0] = landmark.x
        elif landmark.x < minvec[0]:
            minvec[0] = landmark.x

        if landmark.y > maxvec[1]:
            maxvec[1] = landmark.y
        elif landmark.y < minvec[1]:
            minvec[1] = landmark.y
    centerOfMass = np.divide(centerOfMass, length)
    centerOfPixel = np.divide(centerOfPixel, length)
    for i in range(0, 32 + 1):
        landmark = results.pose_landmarks.landmark[i]
        relativePixels.append(
            [
                (landmark.x - centerOfPixel[0]) / (maxvec[0] - minvec[0]),
                (landmark.y - centerOfPixel[1]) / (maxvec[1] - minvec[1]),
            ]
        )
    h, w, c = image.shape

    # 畫出重心
    cv2.circle(
        image,
        (int(centerOfPixel[0] * w), int(centerOfPixel[1] * h)),
        4,
        (255, 120, 120),
        thickness=16,
        lineType=8,
        shift=0,
    )
    cv2.rectangle(
        image,
        (int(minvec[0] * w) - 20, int(minvec[1] * h) - 20),
        (int(minvec[1] * w) + 20, int(minvec[1] * h) + 20),
        (0, 255, 0),
        2,
    )
    return centerOfMass, centerOfPixel, relativePixels
