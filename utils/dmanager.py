import json
import math
import os
from typing import TYPE_CHECKING, TypeAlias

import cv2
import yaml
from tqdm import tqdm

if TYPE_CHECKING:
    from utils.pipeline import Pipeline


class DataManager:
    """DataManager class manages accessing the dataset in YOLO-format"""

    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        # Open and parse the yaml data file
        with open(yaml_path, encoding="utf-8") as data_yaml:
            data = yaml.load(data_yaml, Loader=yaml.FullLoader)
            # Read the data from the yaml file
            train = data["train"].replace("../", "")
            val = data["val"].replace("../", "")
            # Get the dataset directory name
            dataset_dir = os.path.dirname(yaml_path)
            # Construct the image and annotation paths
            self.train_image_path = os.path.join(dataset_dir, train)
            self.train_annotation_path = os.path.join(
                dataset_dir, train.replace("images", "labels")
            )
            self.val_image_path = os.path.join(dataset_dir, val)
            self.val_annotation_path = os.path.join(
                dataset_dir, val.replace("images", "labels")
            )
            # Save the number of classes and class names
            self.names: dict[int, str] = data["names"]
            # Define the structure of the internal data store
            # to avoid constantly recalculating and opening the
            # image to measure it's sizes.
            self.store: dict[str, dict[str, int]] = {}

    def resolve_image_id_by_name(self, file_name: str) -> int:
        """Return the image id corresponding to the file name in the store
        :param file_name: The name of the file to resolve
        :return: The image id corresponding to the file name
        """
        return self.store[file_name]["id"]

    def calculate_annotation_area(self, bbox: tuple[int, int, int, int]):
        """Calculate the area of an annotation given it's bounding box
        :param bbox: The bounding box of the annotation
        :return: The area of the annotation
        """
        _, _, width, height = bbox
        return height * width

    def get_coco_val_gt_path(self, pipeline: "Pipeline"):
        """Get the path to the COCO ground truth file for the validation set"""
        images = []
        annotations = []
        categories = [{"id": i, "name": name} for i, name in enumerate(self.names)]

        prev_annotation_id = 0

        for annotation_file in tqdm(
            os.listdir(self.val_annotation_path), desc="Processing images"
        ):
            # Remove image extension to get the image name
            img_name = annotation_file.replace(".txt", "")

            for image_id, image_file in enumerate(os.listdir(self.val_image_path)):
                if img_name in image_file:
                    # Open the annotation file
                    with open(
                        f"{self.val_annotation_path}/{annotation_file}",
                        encoding="utf-8",
                    ) as annotation:
                        # Open the image file
                        im = cv2.imread(f"{self.val_image_path}/{image_file}")
                        # Get image width and height
                        image_height, image_width, _ = im.shape
                        # Append the image to the images list
                        images.append(
                            {
                                "id": image_id,
                                "file_name": image_file,
                                "width": image_width,
                                "height": image_height,
                            }
                        )

                        image_name = image_file.split(".")[0]

                        # Save in
                        self.store.update(
                            {
                                f"{image_name}": {
                                    "id": image_id,
                                    "width": image_width,
                                    "height": image_height,
                                }
                            }
                        )
                        # For each line in the annotation file
                        for line in annotation.readlines():
                            # Increment the annotation id by 1
                            annotation_id = prev_annotation_id + 1
                            # Parse the annotation
                            (
                                s_category_id,
                                s_x_center,
                                s_y_center,
                                s_annotation_width,
                                s_annotation_height,
                            ) = line.split(" ")
                            # Format cat id as int
                            category_id: int = int(s_category_id)
                            # Format values as floats
                            x_center = float(s_x_center)
                            y_center = float(s_y_center)
                            annotation_width = float(s_annotation_width)
                            annotation_height = float(s_annotation_height)
                            # Calculate COCO format bounding box
                            bbox = self.yolo_to_coco(
                                x_center,
                                y_center,
                                annotation_width,
                                annotation_height,
                                image_width,
                                image_height,
                            )
                            # Calculate the area of the annotation
                            area = self.calculate_annotation_area(bbox)
                            # Append the annotation to the annotations list
                            annotations.append(
                                {
                                    "id": annotation_id,
                                    "image_id": image_id,
                                    "category_id": category_id,
                                    "iscrowd": 0,
                                    "bbox": bbox,
                                    "area": area,
                                }
                            )
                            # Update the previous annotation id before the
                            # next iteration
                            prev_annotation_id = annotation_id

        data = {"images": images, "annotations": annotations, "categories": categories}
        # Convert the data to json encoded strings and save it to a file
        data_str = json.dumps(data)
        pipeline.persistence.save_file("gen__coco-gt.json", data_str)

    def get_annotation_areas(self, target_size: int = 640) -> list[float]:
        areas: list[float] = []

        for annotation_file in tqdm(
            os.listdir(self.train_annotation_path),
            desc=f"Processing images ({self.train_annotation_path})",
        ):
            img_name = annotation_file.replace(".txt", "")

            for image_file in os.listdir(self.train_image_path):
                if img_name in image_file:
                    with open(
                        f"{self.train_annotation_path}/{annotation_file}",
                        encoding="utf-8",
                    ) as annotation:
                        im = cv2.imread(f"{self.train_image_path}/{image_file}")

                        image_height, image_width, _ = im.shape

                        # Ultralytics-style aspect-ratio preserving resize
                        scale = min(
                            target_size / image_width,
                            target_size / image_height,
                        )

                        for line in annotation.readlines():
                            (
                                _,
                                _,
                                _,
                                s_annotation_width,
                                s_annotation_height,
                            ) = line.split()

                            # YOLO bbox dimensions are normalized
                            annotation_width = float(s_annotation_width)
                            annotation_height = float(s_annotation_height)

                            # Convert normalized bbox to original pixel dimensions
                            bbox_width_px = annotation_width * image_width
                            bbox_height_px = annotation_height * image_height

                            # Convert to dimensions as seen by the model input
                            resized_width = bbox_width_px * scale
                            resized_height = bbox_height_px * scale

                            # Padding does not change bbox width/height,
                            # so it does not affect area
                            area = resized_width * resized_height

                            areas.append(area)

                    break

        return areas

    def yolo_to_coco(self, x_center, y_center, w, h, image_w, image_h):
        """Convert bounding boxes from YOLO format to COCO format"""
        w = w * image_w
        h = h * image_h
        x1 = ((2 * x_center * image_w) - w) / 2
        y1 = ((2 * y_center * image_h) - h) / 2
        return x1, y1, w, h

    FitnessScores: TypeAlias = tuple[float, float, float, float, float]

    def calculate_fitness_scores(
        self, area: float, perturb: tuple[float, float] = (1, 1)
    ) -> FitnessScores:
        """Calculate the scores of an area for each head
        :param area: The area to calculate the scores of.
        :param perturb: The amount to perturb the area by - note this was
                        used for the sensitivity analysis and is not changed
                        during training.
        :return: The scores of the area for each head"""

        def calculate_score(
            area: float,
            ref_area: float,
            bandwidth: float,
            perturb: tuple[float, float] = (1, 1),
        ) -> float:
            """Calculate the score of an area given a mean + standard deviation
            :param area:      The area to calculate the score of.
            :param ref_area:  The reference area for the scoring function.
            :param bandwidth: The bandwidth parameter of the scoring function.
            :param perturb:   The amount to perturb the area by - note this was
                              used for the sensitivity analysis and is not changed
                              during training.
            :return: The score of the area."""
            return math.exp(
                -(((area - ref_area * perturb[0]) / (bandwidth * perturb[1])) ** 2)
            )

        return (
            calculate_score(area, 16**2, 16**2, perturb),  # P1
            calculate_score(area, 32**2, 32**2, perturb),  # P2
            calculate_score(area, 64**2, 64**2, perturb),  # P3
            calculate_score(area, 96**2, 96**2, perturb),  # P4
            calculate_score(area, 128**2, 128**2, perturb),  # P5
        )

    HeadFitnessScores: TypeAlias = dict[str, float]

    def judge_heads(self) -> HeadFitnessScores:
        """Judge the fitness of each head by calculating the score of each area
        :return: The fitness scores of each head."""
        # Count scores for each area
        p5score = p4score = p3score = p2score = p1score = 0.0
        # Open the ground truth files
        for area in self.get_annotation_areas():
            scores = self.calculate_fitness_scores(area)

            p1score += scores[0]
            p2score += scores[1]
            p3score += scores[2]
            p4score += scores[3]
            p5score += scores[4]

        # Return the scores for each head
        return {
            "P1": p1score,
            "P2": p2score,
            "P3": p3score,
            "P4": p4score,
            "P5": p5score,
        }
