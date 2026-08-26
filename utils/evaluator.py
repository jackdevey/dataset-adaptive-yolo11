import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

if TYPE_CHECKING:
    from utils.pipeline import Pipeline


@dataclass
class COCOEvalResults:
    ap_50_95_all: float
    ap_50_all: float
    ap_75_all: float
    ap_50_95_small: float
    ap_50_95_medium: float
    ap_50_95_large: float
    ar_50_95_all_1: float
    ar_50_95_all_10: float
    ar_50_95_all_100: float
    ar_50_95_small: float
    ar_50_95_medium: float
    ar_50_95_large: float


class COCOEvaluator:
    pipeline: "Pipeline"
    predictions_file: str | None
    custom_gt_path: str | None

    def __init__(self, pipeline: "Pipeline", custom_gt_path: str | None = None):
        self.pipeline = pipeline
        self.predictions_file = None
        self.custom_gt_path = custom_gt_path

    def evaluate(self, tag: str | None = None) -> COCOEvalResults:
        """Execute the COCO evaluation process"""
        # Evaluate the model
        self.evaluate_model(tag=tag)

        # Only create ground truth if no custom path is provided
        if not self.custom_gt_path:
            self.create_ground_truth(tag=tag)
            gt_path = self.pipeline.persistence.resolve_asset_path(
                "evaluation/gt.json", tag=tag
            )
        else:
            gt_path = self.custom_gt_path

        # Load COCO ground truth data
        coco_gt = COCO(gt_path)
        # Load the predictions
        predictions_path = self.pipeline.persistence.resolve_asset_path(
            "evaluation/predictions.json", tag=tag
        )
        # Validate that predictions are compatible with the ground truth
        self._validate_predictions(
            coco_gt,
            predictions_path,
        )
        coco_dt = coco_gt.loadRes(predictions_path)
        # Create the evaluation object
        self.pipeline.logger.info("Evaluating using the COCO evaluator")
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        # Run the evaluation in a func to
        # record the evaluation output by
        # redirecting the stdout to a file

        def process_func(_):
            coco_eval.params.imgIds = list(coco_gt.getImgIds())
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        # Record the evaluation output
        self.pipeline.persistence.capture_stdout_log(
            name="coco_evaluation", func=process_func, tag=tag
        )
        # Store the results as a COCOEvalResults object
        results = COCOEvalResults(
            ap_50_95_all=coco_eval.stats[0],
            ap_50_all=coco_eval.stats[1],
            ap_75_all=coco_eval.stats[2],
            ap_50_95_small=coco_eval.stats[3],
            ap_50_95_medium=coco_eval.stats[4],
            ap_50_95_large=coco_eval.stats[5],
            ar_50_95_all_1=coco_eval.stats[6],
            ar_50_95_all_10=coco_eval.stats[7],
            ar_50_95_all_100=coco_eval.stats[8],
            ar_50_95_small=coco_eval.stats[9],
            ar_50_95_medium=coco_eval.stats[10],
            ar_50_95_large=coco_eval.stats[11],
        )

        print("✅ COCO evaluation successfully completed")
        print("COCO evaluation output saved to file in output directory")
        # Return results
        return results

    def evaluate_model(self, tag: str | None = None):
        self.pipeline.logger.info("Using ultralytics API to evaluate model")
        # Validate the model with save_json set to true
        model = self.pipeline.get_model()
        assert model is not None
        metrics = model.val(save_json=True, data=self.pipeline.dmanager.yaml_path)
        # Copy the output into the output folder
        self.pipeline.persistence.move_from_folder(
            str(metrics.save_dir), "evaluation", tag=tag
        )
        self.pipeline.logger.success("Completed ultralytics API evaluation")

    def create_ground_truth(self, tag: str | None = None):
        images = []
        annotations = []

        val_image_path = Path(self.pipeline.dmanager.val_image_path)
        val_annotation_path = Path(self.pipeline.dmanager.val_annotation_path)

        categories = [
            {
                "id": i + 1,  # ONLY keep +1 if predictions.json uses +1
                "name": name,
            }
            for i, name in enumerate(self.pipeline.dmanager.names)
        ]

        annotation_id = 1

        image_files = [
            path
            for path in val_image_path.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        ]

        for image_path in tqdm(
            image_files,
            desc=f"Creating COCO ground truth from {val_image_path}",
        ):
            image = cv2.imread(str(image_path))

            if image is None:
                self.pipeline.logger.warning(f"Could not read image: {image_path}")
                continue

            image_height, image_width = image.shape[:2]

            # IMPORTANT:
            # This must match Ultralytics predictions.json image_id.
            stem = image_path.stem
            image_id = int(stem) if stem.isdigit() else stem

            images.append(
                {
                    "id": image_id,
                    "file_name": image_path.name,
                    "width": image_width,
                    "height": image_height,
                }
            )

            label_path = val_annotation_path / f"{stem}.txt"

            # Images with no annotations still belong in images[].
            if not label_path.exists():
                continue

            with label_path.open(encoding="utf-8") as annotation_file:
                for line in annotation_file:
                    line = line.strip()

                    if not line:
                        continue

                    (
                        s_category_id,
                        s_x_center,
                        s_y_center,
                        s_width,
                        s_height,
                    ) = line.split()

                    category_id = int(s_category_id) + 1

                    x_center = float(s_x_center)
                    y_center = float(s_y_center)
                    width = float(s_width)
                    height = float(s_height)

                    bbox = self.pipeline.dmanager.yolo_to_coco(
                        x_center,
                        y_center,
                        width,
                        height,
                        image_width,
                        image_height,
                    )

                    annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "iscrowd": 0,
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                        }
                    )

                    annotation_id += 1

        data = {
            "info": {},
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

        self.pipeline.persistence.save_file(
            name="evaluation/gt.json",
            data=json.dumps(data, indent=2),
            tag=tag,
        )

    def _validate_predictions(self, coco_gt: COCO, predictions_path: str) -> None:
        """Validate that the prediction JSON is compatible with the COCO ground truth."""

        with open(predictions_path, encoding="utf-8") as f:
            predictions = json.load(f)

        gt_image_ids = set(coco_gt.getImgIds())
        pred_image_ids = {pred["image_id"] for pred in predictions}

        missing_image_ids = pred_image_ids - gt_image_ids

        if missing_image_ids:
            raise ValueError(
                "Prediction image IDs are not present in the COCO ground truth.\n"
                f"Missing image IDs ({len(missing_image_ids)}): "
                f"{sorted(missing_image_ids)[:20]}"
            )

        gt_category_ids = set(coco_gt.getCatIds())
        pred_category_ids = {pred["category_id"] for pred in predictions}

        missing_category_ids = pred_category_ids - gt_category_ids

        if missing_category_ids:
            raise ValueError(
                "Prediction category IDs are not present in the COCO ground truth.\n"
                f"Missing category IDs: {sorted(missing_category_ids)}"
            )

        self.pipeline.logger.success(
            "Prediction JSON successfully validated against COCO ground truth."
        )
