from pathlib import Path
from typing import Tuple, Generator
import mediapipe as mp
import os
import parse
import cv2

DOWNLOAD_DIRECTORY = Path.home() / "Downloads"
DATASETS = ["ur_fall", "Florence_3d_actions", "Dataset CAUCAFall"]

CAUCAFALL_ENUMS = {
    "Fall backwards": 0,
    "Fall forward": 1,
    "Fall left": 2,
    "Fall right": 3,
    "Fall sitting": 4,
    "Hop": 5,
    "Kneel": 6,
    "Pick up object": 7,
    "Sit down": 8,
    "Walk": 9,
}

CAUCAFALL_ACTION_TYPES = [
    "Fall backwards",
    "Fall forward",
    "Fall left",
    "Fall right",
    "Fall sitting",
    "Hop",
    "Kneel",
    "Pick up object",
    "Sit down",
    "Walk",
]


class Configs:
    def __init__(
        self,
        render=False,
        train=False,
        consecutive_frame_count=5,
        exclude_verticies=[
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
    ) -> None:
        self.render = render
        self.train = train
        self.consecutive_frame_count = consecutive_frame_count
        self.exclude_verticies = exclude_verticies


def parse_florence_3d_name(mini_name) -> Tuple[int]:
    if not mini_name.endswith(".avi"):
        return (0, 0, 0, 0)

    idGesture, idActor, idAction, idCategory = (
        int(i) for i in parse.parse("GestureRecording_Id{}actor{}idAction{}category{}.avi", mini_name)
    )
    return (idGesture, idActor, idAction, idCategory)


def enum_ur_fall(start_index=None, end_index=None) -> Generator[Tuple[str, str], None, None]:
    start_index = start_index or 1
    end_index = end_index or 30
    for i in range(start_index, end_index + 1):
        folder_name = f"fall-{i:02d}-cam0-rgb"
        full_folder_name = str(DOWNLOAD_DIRECTORY / ("ur_fall/" + folder_name))
        yield (folder_name, full_folder_name)


def enum_florence_3d() -> Generator[Tuple[str, str], None, None]:
    filenames = os.listdir(str(DOWNLOAD_DIRECTORY / "Florence_3d_actions"))
    filenames.sort(key=lambda a: parse_florence_3d_name(a))
    for fname in filenames:
        if not fname.endswith(".avi"):
            continue
        full_fname = str(DOWNLOAD_DIRECTORY / fname)
        yield (fname, full_fname)


def enum_cauca_fall() -> Generator[Tuple[str, str], None, None]:
    for subject_index in range(1, 10 + 1):
        for action in CAUCAFALL_ACTION_TYPES:
            target_dir = str(DOWNLOAD_DIRECTORY / f"Dataset CAUCAFall/CAUCAFall/Subject.{subject_index}/{action}")
            yield (target_dir, CAUCAFALL_ENUMS[action])


def enum_landmark_id(configs) -> Generator[int, None, None]:
    for i in range(0, 32 + 1):
        if i in configs.exclude_verticies:
            continue
        yield i


def get_landmark(raw_landmarks, configs):
    landmark_pos = []
    i = -1
    for raw_landmark in raw_landmarks:
        landmark_pos.append((0.0, 0.0))
        i += 1
        if i in configs.exclude_verticies:
            continue
        landmark_pos[i] = (raw_landmark.x, raw_landmark.y)
    return landmark_pos


def get_center_of_mass(landmarks):
    center = (0.0, 0.0)
    i = 0
    for ld in landmarks:
        if ld[0] == 0.0 and ld[1] == 0.0:
            continue
        center = (center[0] + ld[0], center[1] + ld[1])
        i += 1
    return (center[0] / i, center[1] / i)


def get_landmark_velocity(landmarks, lastlandmarks, fps, configs):
    velocity = []
    for i in enum_landmark_id(configs):
        velocity.append(
            (
                (landmarks[i].x - lastlandmarks[i][0]) * fps,
                (landmarks[i].y - lastlandmarks[i][1]) * fps,
            )
        )
    return velocity


def flatten_landmark(landmarks):
    return [pos_value for coord_tuple in landmarks for pos_value in coord_tuple]


def putText(
    img,
    text: str,
    pos: tuple = (0, 0),
    font=cv2.FONT_HERSHEY_PLAIN,
    font_scale: int = 3,
    font_thickness: int = 2,
    text_color: Tuple[int, int, int] = (0, 255, 0),
    text_color_bg: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple:

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


class Mediapipe_Person:
    def __init__(self, rect_center) -> None:
        self.rect_center = rect_center


class Mediapipe_Pose:
    def __init__(self) -> None:
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.55, min_tracking_confidence=0.55)

    def process(self, img):
        return self.pose.process(img)
