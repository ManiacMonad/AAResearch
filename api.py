import mediapipe as mp
import cv2

# TEST TEST TEsT TEst tEst


class Optional:
    """
    回傳特殊值
    設計目標:製作簡單程式碼
    """

    def __init__(self, value):
        self.available = value is not None
        self.value = value


class MediapipeAPI:
    """
    Mediapipe 互動介面
    設計目標:簡單化Mediapipe的開發
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    def __init__(self, video_port=0):
        # 設定上下文(Context)
        self.videoCapture = cv2.VideoCapture(video_port)
        self.pose = MediapipeAPI.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def isPortOpened(self):
        return self.videoCapture.isOpened()

    def print(self, x):
        global print
        print("[Mediapipe API]: " + x)

    def readImage(self):
        """載入照片

        Returns:
            Optional<image>:可省略照片
        """
        success, image = self.videoCapture.read()
        if not success:
            self.print("Unable to retrieve image!")
            return Optional()
        return Optional(image)

    def processImage(self, image):
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        image.flags.writeable = True
        return results, cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def overlayLandmarks(self, image, results):
        MediapipeAPI.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            MediapipeAPI.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=MediapipeAPI.mp_drawing_styles.get_default_pose_landmarks_style())

    def showImage(self, image):
        cv2.imshow('MediaPipe API Render Port', cv2.flip(image, 1))


apiHandle = MediapipeAPI()

while apiHandle.isPortOpened():
    capturedImage = apiHandle.readImage()
    if not capturedImage.available:
        print("Ignoring empty camera frame.")
        continue

    results, newImage = apiHandle.processImage(capturedImage.value)
    apiHandle.overlayLandmarks(newImage, results)
    apiHandle.showImage(newImage)

    # Flip the image horizontally for a selfie-view display.
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
