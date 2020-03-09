#from concrete_polygon_extractor import LineExtender, PolygonRetriever
#from tilt_detector import TiltDetector, LineMerger

from line_modifier import LineModifier
from concrete_extractor import ConcreteExtractor
from utils import ResultsHandler, calculate_angle
import os
import argparse
import cv2
import time
import sys


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image', type=str, help='Path to an image.')
    parser.add_argument('--folder', type=str, help="Path to a folder with images to process")
    parser.add_argument('--save_path', type=str, default=None, help="If None, don't save results, show them")

    parser.add_argument('--retrieve', type=int, default=0, help="Retrieve image section defined by the two pole"
                                                                "edge lines detected")
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
            raise IOError("Provided image is not an image")

    else:
        if not os.path.isdir(arguments.folder):
            raise IOError("Provided folder is not a folder")

        for image_name in os.listdir(arguments.folder):
            if not any(image_name.endswith(ext) for ext in [".jpg", ".png", ".jpeg",
                                                            ".JPG", ".JPEG", ".PNG"]):
                continue

            images_to_process.append(os.path.join(arguments.folder, image_name))

    # If save path has been provided, all images will be processed and saved there
    # Otherwise, after each image gets processed it will be shown to a user until
    # he clicks a button to proceed to the next image if any.
    # In order to just calculate and receive angle = flag is 0
    if arguments.save_path is not None:
        if not os.path.exists(arguments.save_path):
            os.mkdir(arguments.save_path)

    handler = ResultsHandler(save_path=arguments.save_path)
    concrete_detector = ConcreteExtractor(line_modifier=LineModifier)

    total_error = 0.0
    images_with_calculated_angles = 0
    images_without_angle_calculated = list()
    performance_tracker = list()

    # Process all images
    for path_to_image in images_to_process:

        image_name = os.path.split(path_to_image)[-1]
        print('\n', image_name)

        try:
            image = cv2.imread(path_to_image)
        except:
            print("ERROR: Failed to open an image", image_name)
            continue

        img_res = (image.shape[0], image.shape[1])

        # Find edges
        start_time = time.time()
        the_edges = concrete_detector.find_pole_edges(image=image)
        inference_time = time.time() - start_time

        #assert 1 <= len(the_edges) <= 2, "ERROR: Wrong number of edges!"

        if the_edges:

            # Calculate angle
            the_angle = calculate_angle(the_lines=the_edges)

            performance_tracker.append(
                (image_name, img_res, inference_time)
            )

            try:
                # Extract angle from the image name (NAMING CONVENTION)
                truth_angle = float(image_name.split("_")[-1][:-4])
            except:
                print(f"\nERROR: Naming convention error: {image_name}")
                images_with_calculated_angles += 1
                continue

            difference = abs(truth_angle - the_angle)
            error = round(difference / truth_angle, 3)
            print(f"Predicted: {the_angle}, Truth: {truth_angle}, Error: {error}")

            total_error += error
            images_with_calculated_angles += 1

            # Postprocess the results
            # image = handler.draw_lines_write_text(lines=the_edges,
            #                                       image=image,
            #                                       angle=the_angle)
            # handler.save_image_2(image_name=image_name,
            #                    image=image)

        else:
            images_without_angle_calculated.append(image_name)
            print("Failed to detect edges for:", image_name)

        # Retrieve area defined by the lines for future cracks detection
        if arguments.retrieve:

            concrete_polygon = concrete_detector.retrieve_polygon(the_lines=the_edges,
                                                                  image=image)

            # DELETE ME I AM FOR TESTING
            handler.save_image_2(image_name,
                                 concrete_polygon)
            #handler.show_image(concrete_polygon)

    if images_with_calculated_angles > 0:
        mean_error = round(total_error / images_with_calculated_angles, 3)
        print("\nMEAN ERROR:", mean_error * 100, "%")

        total_time = sum(t[-1] for t in performance_tracker)
        print("TOTAL TIME TAKEN:", total_time)
        print("AVERAGE TIME PER IMAGE:", round((total_time / images_with_calculated_angles), 3))

        print()
        for name, res, t in performance_tracker:
            print(name, "Res:", res, "Time:", round(t, 2), " seconds")

    else:
        print("\nCannot calculate MEAN ERROR. Failed to calculate angle for any images")

    if images_without_angle_calculated:
        print("\nFAILED TO CALCULATE ANGLE FOR:",
                                ' '.join(images_without_angle_calculated))


if __name__ == "__main__":
    main()
