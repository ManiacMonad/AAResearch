from utils import DOWNLOAD_DIRECTORY, enum_ur_fall, enum_florence_3d, parse_florence_3d_name, Configs
from stream import VideoHandler, VideoStream, FolderHandler
import cv2


def main():
    configs = Configs(render=True)
    for (folder_name, full_name) in enum_ur_fall():
        stream = VideoStream(5, FolderHandler(full_name, configs, suffix=".png"), configs)
        while stream.get_image() is not None:
            data = stream.get_data()
            if data is None:
                continue
            cv2.imshow("img", data[len(data) - 1])
            cv2.waitKey(1)
            if data is None:
                continue

        stream.dispose()


if __name__ == "__main__":
    main()
