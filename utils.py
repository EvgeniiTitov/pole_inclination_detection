import cv2
import os
import sys
import numpy as np


class ResultsHandler:

    def __init__(
            self,
            save_path,
            line_thickness=2,
            window_name='window'
    ):
        self._window_name = window_name
        self._is_window_created = False
        self._sleeping_time = 1

        self._line_thickness = line_thickness
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = 1
        self._font_colour = (255, 0, 255)
        self._line_type = 3

        self.save_path = save_path

    @property
    def is_window_created(self):
        return self._is_window_created

    def create_window(self):
        cv2.namedWindow(self._window_name)
        self._is_window_created = True

    def show(self, image):
        cv2.imshow(self._window_name, image)

    def destroy_windows(self):
        cv2.destroyWindow(self._window_name)
        self._is_window_created = False

    def draw_lines_write_text(self,
                              lines,
                              image,
                              angle):

        for line in lines:
            cv2.line(image,
                     (line[0][0], line[0][1]),
                     (line[1][0], line[1][1]),
                     (0, 0, 255),
                     self._line_thickness)

        cv2.putText(image,
                    str(angle),
                    (int(image.shape[1]*0.35), int(image.shape[0]*0.95)),
                    self._font,
                    self._font_scale,
                    self._font_colour,
                    self._line_type)

        return image

    def save_image(self,
                   lines,
                   image,
                   image_name,
                   angle):


        cv2.imwrite(os.path.join(self.save_path, image_name),
                    self.draw_lines_write_text(lines, image, angle))

    def save_image_2(self,
                     image_name,
                     image,
                     path):

        cv2.imwrite(
            os.path.join(path, image_name),
            image
                    )

    def show_image(self,
                   image,
                   lines=None,
                   angle=None):

        cv2.imshow('Extracted Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
