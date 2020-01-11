from tilt_detector import TiltDetector, LineMerger
from utils import ResultsHandler
import os
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image', type=str, help='Path to an image.')
    parser.add_argument('--folder', type=str, help="Path to a folder with images to process")
    parser.add_argument('--save_path', type=str, default=None, help="If None, don't save results, show them")

    arguments = parser.parse_args()

    return arguments


def main():

    arguments = parse_args()

    assert arguments.image or arguments.folder, "No input data provided"

    images_to_process = list()

    if arguments.image:

        if os.path.isfile(arguments.image):
            images_to_process.append(arguments.image)
        else:
            print("ERROR: Provided image is not an image")
            sys.exit()

    else:
        if not os.path.isdir(arguments.folder):
            print("ERROR: Provided folder is not a folder")
            sys.exit()

        for image_name in os.listdir(arguments.folder):
            if not any(image_name.endswith(ext) for ext in [".jpg", ".png", ".jpeg", ".JPG", ".JPEG", ".PNG"]):
                continue

            images_to_process.append(os.path.join(arguments.folder, image_name))

    # If save path has been provided, all images will be processed and saved there
    # Otherwise, after each image gets processed it will be shown to a user until
    # he clicks a button to proceed to the next image if any.
    # In order to just calculate and receive angle = flag is 0

    if arguments.save_path is not None:
        if not os.path.exists(arguments.save_path):
            os.mkdir(arguments.save_path)

        results_handling = 1, arguments.save_path
        handler = ResultsHandler(save_path=arguments.save_path)
    else:
        results_handling = 2, ''
        handler = None

    merger = LineMerger()

    detector = TiltDetector(results_handling_way=results_handling,
                            line_merger=merger,
                            results_processor=handler)

    total_error = 0
    images_with_calculated_angles = 0

    for path_to_image in images_to_process:

        image_name = os.path.split(path_to_image)[-1]
        print(image_name)

        truth_angle = float(image_name[3:7])
        predicted_tilt_angle = detector.process_image(path_to_image)

        if predicted_tilt_angle:
            difference = abs(truth_angle - predicted_tilt_angle)
            error = round(difference / truth_angle, 3)
            print("Error:", error)

            total_error += error
            images_with_calculated_angles += 1

    mean_error = round(total_error / images_with_calculated_angles, 3)
    print("\nMEAN ERROR:", mean_error * 100, "%")


if __name__ == "__main__":
    main()
