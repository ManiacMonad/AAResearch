from pathlib import Path
import os
import parse

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
    ) -> None:
        self.render = render


def parse_florence_3d_name(mini_name):
    if not mini_name.endswith(".avi"):
        return (0, 0, 0, 0)

    idGesture, idActor, idAction, idCategory = (
        int(i) for i in parse.parse("GestureRecording_Id{}actor{}idAction{}category{}.avi", mini_name)
    )
    print(idGesture)
    return idGesture, idActor, idAction, idCategory


def enum_ur_fall(start_index=None, end_index=None):
    start_index = start_index or 1
    end_index = end_index or 30
    for i in range(start_index, end_index + 1):
        folder_name = f"fall-{i:02d}-cam0-rgb"
        full_folder_name = str(DOWNLOAD_DIRECTORY / ("ur_fall/" + folder_name))
        yield (folder_name, full_folder_name)


def enum_florence_3d():
    filenames = os.listdir(str(DOWNLOAD_DIRECTORY / "Florence_3d_actions"))
    filenames.sort(key=lambda a: parse_florence_3d_name(a))
    for fname in filenames:
        if not fname.endswith(".avi"):
            continue
        full_fname = str(DOWNLOAD_DIRECTORY / fname)
        yield (fname, full_fname)


def enum_cauca_fall():
    for subject_index in range(1, 10 + 1):
        for action in CAUCAFALL_ACTION_TYPES:
            target_dir = str(DOWNLOAD_DIRECTORY / f"Dataset CAUCAFall/CAUCAFall/Subject.{subject_index}/{action}")
            yield (target_dir, CAUCAFALL_ENUMS[action])
