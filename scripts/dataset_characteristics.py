import os
from collections import Counter

import cv2
import numpy as np

# Change this import to wherever DataManager is defined
from dataset_adaptive_yolo11.dmanager import DataManager

dmanagers: dict[str, DataManager] = {
    "VisDrone-DET": DataManager("/data2/jd1/datasets/VisDrone/data.yaml"),
    "SeaDronesSee": DataManager("/data2/jd1/datasets/sds-ds/sds/data.yaml"),
    "KITTI": DataManager("/data2/jd1/datasets/kitti/data.yaml"),
    "humancar_50m": DataManager("/data2/jd1/datasets/humancar_50m/data.yaml"),
    "License Plate Rec.": DataManager("/data2/jd1/datasets/lpr/data.yaml"),
    "Landing Pad Det.": DataManager(
        "/data2/jd1/datasets/landing-pad-detection-2-v4/data.yaml"
    ),
    "Medical Pills": DataManager("/data2/jd1/datasets/medical-pills/data.yaml"),
}


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def get_image_files(path: str) -> list[str]:
    """Return all image files in a directory."""
    return [
        f
        for f in os.listdir(path)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]


def count_objects(annotation_path: str) -> int:
    """Count the total number of YOLO annotations in a split."""
    count = 0

    for file_name in os.listdir(annotation_path):
        if not file_name.endswith(".txt"):
            continue

        path = os.path.join(annotation_path, file_name)

        with open(path, encoding="utf-8") as f:
            count += sum(1 for line in f if line.strip())

    return count


def get_native_resolutions(image_path: str):
    """
    Read native image resolutions.

    Returns:
        widths:  array of image widths
        heights: array of image heights
        counts:  frequency of each (width, height) resolution
    """
    widths = []
    heights = []
    resolutions = Counter()

    for file_name in get_image_files(image_path):
        path = os.path.join(image_path, file_name)

        image = cv2.imread(path)

        if image is None:
            print(f"WARNING: Could not read {path}")
            continue

        height, width = image.shape[:2]

        widths.append(width)
        heights.append(height)
        resolutions[(width, height)] += 1

    return np.array(widths), np.array(heights), resolutions


def get_annotation_areas_normalized(
    image_path: str,
    annotation_path: str,
) -> list[float]:
    """
    Return bounding-box areas as a percentage of the native image area.

    YOLO width and height values are already normalized relative to the
    image dimensions, so:

        relative area = bbox_width * bbox_height

    Multiplying by 100 gives percentage of image area.
    """
    areas = []

    for annotation_file in os.listdir(annotation_path):
        if not annotation_file.endswith(".txt"):
            continue

        path = os.path.join(annotation_path, annotation_file)

        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                values = line.split()

                if len(values) < 5:
                    continue

                bbox_width = float(values[3])
                bbox_height = float(values[4])

                area_percent = bbox_width * bbox_height * 100
                areas.append(area_percent)

    return areas


def format_resolution_distribution(
    widths: np.ndarray,
    heights: np.ndarray,
) -> str:
    """
    Produce a compact description of the native resolution distribution.
    """
    if len(widths) == 0:
        return "N/A"

    min_w = int(np.min(widths))
    max_w = int(np.max(widths))
    min_h = int(np.min(heights))
    max_h = int(np.max(heights))

    median_w = int(np.median(widths))
    median_h = int(np.median(heights))

    if min_w == max_w and min_h == max_h:
        return f"{min_w}x{min_h}"

    return f"{median_w}x{median_h} [{min_w}-{max_w} x {min_h}-{max_h}]"


def collect_statistics(name: str, dm: DataManager) -> dict:
    """Collect dataset statistics."""

    train_images = get_image_files(dm.train_image_path)
    val_images = get_image_files(dm.val_image_path)

    train_objects = count_objects(dm.train_annotation_path)
    val_objects = count_objects(dm.val_annotation_path)

    # Native resolutions across both train and validation sets
    train_widths, train_heights, _ = get_native_resolutions(dm.train_image_path)
    val_widths, val_heights, _ = get_native_resolutions(dm.val_image_path)

    widths = np.concatenate([train_widths, val_widths])
    heights = np.concatenate([train_heights, val_heights])

    # Calculate object-area distribution from the TRAINING split.
    #
    # This matches the idea of characterising the data presented to the
    # model during training.
    areas = get_annotation_areas_normalized(
        dm.train_image_path,
        dm.train_annotation_path,
    )

    median_area = float(np.median(areas)) if areas else float("nan")

    return {
        "dataset": name,
        "train_images": len(train_images),
        "val_images": len(val_images),
        "train_objects": train_objects,
        "val_objects": val_objects,
        "classes": len(dm.names),
        "resolution": format_resolution_distribution(widths, heights),
        "median_area": median_area,
    }


def print_markdown_table(results: list[dict]):
    """Print a readable table for checking the results."""

    print(
        "| Dataset | Images [Train/Val] | Objects [Train/Val] | "
        "Classes | Native Resolution | Median Obj. Area |"
    )
    print("|---|---:|---:|---:|---|---:|")
    for r in results:
        print(
            f"| {r['dataset']} "
            f"| {r['train_images']:,}/{r['val_images']:,} "
            f"| {r['train_objects']:,}/{r['val_objects']:,} "
            f"| {r['classes']} "
            f"| {r['resolution']} "
            f"| {r['median_area']:.3f}% |"
        )


if __name__ == "__main__":
    results = []

    for name, dm in dmanagers.items():
        print(f"Processing {name}...")
        results.append(collect_statistics(name, dm))

    print()
    print_markdown_table(results)
