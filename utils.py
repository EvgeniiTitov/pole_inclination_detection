import cv2
import os
import numpy as np


def calculate_angle(the_lines: list) -> float:
    """
    Calculates angle of the line(s) provided
    :param the_lines: list of lists, lines found and filtered
    :return: angle
    """
    if len(the_lines) == 2:
        x1_1 = the_lines[0][0][0]
        y1_1 = the_lines[0][0][1]
        x2_1 = the_lines[0][1][0]
        y2_1 = the_lines[0][1][1]

        # Original approach
        # angle_1 = round(90 - np.rad2deg(np.arctan2(abs(y2_1 - y1_1), abs(x2_1 - x1_1))), 2)

        angle_1 = round(np.rad2deg(np.arctan(abs(x2_1 - x1_1) / abs(y2_1 - y1_1))), 2)

        x1_2 = the_lines[1][0][0]
        y1_2 = the_lines[1][0][1]
        x2_2 = the_lines[1][1][0]
        y2_2 = the_lines[1][1][1]

        # Original approach
        # angle_2 = round(90 - np.rad2deg(np.arctan2(abs(y2_2 - y1_2), abs(x2_2 - x1_2))), 2)

        angle_2 = round(np.rad2deg(np.arctan(abs(x2_2 - x1_2) / abs(y2_2 - y1_2))), 2)

        return round((angle_1 + angle_2) / 2, 2)

    else:
        x1 = the_lines[0][0][0]
        y1 = the_lines[0][0][1]
        x2 = the_lines[0][1][0]
        y2 = the_lines[0][1][1]

        # Original approach
        # return round(90 - np.rad2deg(np.arctan2(abs(y2 - y1), abs(x2 - x1))), 2)
        return round(np.rad2deg(np.arctan(abs(x2 - x1) / abs(y2 - y1))), 2)


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
                     image):

        cv2.imwrite(
            os.path.join(self.save_path, image_name),
            image
        )

    def show_image(self,
                   image,
                   lines=None,
                   angle=None):

        cv2.imshow('Extracted Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
