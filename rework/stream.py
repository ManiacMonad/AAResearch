import cv2
import os

# This is a video stream file


class VideoHandler:
    def __init__(self, video_name, configs) -> None:
        self.video_name = video_name
        self.capture = None
        self.configs = configs

    def setup_capture(self):
        self.capture = cv2.VideoCapture(self.video_name)

    def read_image(self):
        success, image = self.capture.read()
        if success:
            return image
        return None

    def dispose(self):
        self.capture.release()


class FolderHandler:
    def __init__(self, folder_name, configs, suffix=".jpg") -> None:
        self.folder_name = folder_name
        self.suffix = suffix
        self.capture = None
        self.configs = configs
        self.index = 0

    def setup_capture(self):
        self.capture = os.listdir(self.folder_name)
        self.capture.sort()

    def read_image(self):
        if len(self.capture) <= self.index:
            return None
        file_name = self.capture[self.index]
        self.index += 1
        if not file_name.endswith(self.suffix):
            return self.read_image()

        print("FolderHandler Debug: " + file_name)
        return cv2.imread(self.folder_name + "/" + file_name)

    def dispose(self):
        pass


class VideoStream:
    def __init__(self, image_handler, configs) -> None:
        self.data = []
        self.image_handler = image_handler
        self.image_handler.setup_capture()
        self.configs = configs

    def get_image(self):
        img = self.image_handler.read_image()
        return img

    def dispose(self):
        self.image_handler.dispose()
