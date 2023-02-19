import mediapipe as mp

mp_pose = mp.solutions.pose


class BASE_CONFIGURATIONS:
    def __init__(
        self,
        TRAIN=True,
        RENDER_IMAGE=True,
        LOAD_FROM_FOLDER=True,
        NO_FALL=False,
        EXCLUDE_VERTICES=[
            mp_pose.PoseLandmark.LEFT_EYE_INNER,
            mp_pose.PoseLandmark.LEFT_EYE,
            mp_pose.PoseLandmark.LEFT_EYE_OUTER,
            mp_pose.PoseLandmark.RIGHT_EYE_INNER,
            mp_pose.PoseLandmark.RIGHT_EYE,
            mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
            mp_pose.PoseLandmark.LEFT_EAR,
            mp_pose.PoseLandmark.RIGHT_EAR,
            mp_pose.PoseLandmark.MOUTH_LEFT,
            mp_pose.PoseLandmark.MOUTH_RIGHT,
        ],
    ):
        self.TRAIN = TRAIN
        self.RENDER_IMAGE = RENDER_IMAGE
        self.LOAD_FROM_FOLDER = LOAD_FROM_FOLDER
        self.NO_FALL = NO_FALL
        self.EXCLUDE_VERTICES = EXCLUDE_VERTICES
