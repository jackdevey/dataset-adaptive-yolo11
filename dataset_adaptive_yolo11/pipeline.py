import json
import uuid

import loguru
from loguru import logger
from pydantic import BaseModel, Field
from ultralytics import YOLO

from dataset_adaptive_yolo11.adapter import Adapter
from dataset_adaptive_yolo11.dmanager import DataManager
from dataset_adaptive_yolo11.evaluator import COCOEvaluator
from dataset_adaptive_yolo11.notifications import TelegramAPI, add_training_callbacks
from dataset_adaptive_yolo11.persistence import Persistence


class Pipeline:
    class Config(BaseModel):
        name: str | None = Field(default=None, description="Name of the pipeline")
        telegram: TelegramAPI | None = Field(
            default=None, description="Telegram API for notifications"
        )
        device: str | int | None = Field(
            default=None,
            description="Device to use for training (e.g. 'cpu', 0, '0,1,2,3')",
        )
        seed: int = Field(default=0, description="Random seed for training")

    heads: dict[str, float] | None = None

    def __init__(
        self,
        dataset_yaml_path: str,
        project: str,
        config: dict | None = None,
    ) -> None:
        # Parse the configuration dictionary once as a pipeline config object
        self.config = self.Config(**config or {})

        self.project = project

        # Initialise persitence
        self.persistence = Persistence(self)

        # Setup logger & make it save to log file with retention of 10 days
        self.logger: loguru.Logger = logger
        self.logger.add(
            self.persistence.resolve_asset_path("output.log"), retention="10 Days"
        )

        # Log the output directory
        self.persistence.log_path()

        self.adapter = Adapter(self)
        # Initialise data manager
        self.dmanager = DataManager(dataset_yaml_path)
        # Create a variable to hold the model
        self.model: YOLO | None = None
        # If using the telegram API, log it
        if self.config.telegram:
            self.logger.success("Using Telegram API for notifications")
        else:
            self.logger.info("Not sending notifications")

    def set_model(self, model: YOLO):
        """Set the model for the pipeline."""
        self.model = model
        # Add training callbacks if telegram is configured
        if self.config.telegram:
            add_training_callbacks(self.model, self.config.telegram)
        # Save model info to output folder
        model_info = {
            "model_type": str(type(model).__name__),
            "model_metrics": {
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "layers": len(list(getattr(model.model, "modules", lambda: [])())),
                "input_size": getattr(model.model, "img_size", None),
                "num_classes": getattr(model.model, "nc", None),
            },
        }
        self.persistence.save_file("model_info.json", json.dumps(model_info, indent=2))

    def create_model(
        self, skip_and_use: str | None = None, abl_branch: str | None = None
    ):
        # Find a score for each head according to the dataset
        self.heads = self.dmanager.judge_heads()
        # Save the head scores to a file
        self.persistence.save_file("heads.json", str(self.heads))
        # Generate a new model using the adapter
        model = self.adapter.execute(self.heads, skip_and_use, abl_branch)
        # Set the model for the pipeline
        self.set_model(model)

    def load_model(self, path: str):
        # Load the model from the path
        model = YOLO(path)
        # Set the model for the pipeline
        self.set_model(model)

    def train(self, epochs: int, batch: int = -1, seed: int | None = None):
        # Throw an error if the model is not initialised
        assert self.model is not None, "Model is not initialised"
        # Generate an id for collecting the training artefacts
        id = str(uuid.uuid4())
        self.logger.info(
            "Using ultralytics API to train " + f"model for {epochs} epochs"
        )
        # Use the ultralytics API to train the model
        results = self.model.train(
            data=self.dmanager.yaml_path,
            epochs=epochs,
            batch=batch,
            verbose=True,
            project=self.project,
            name=self.config.name if self.config.name else id,
            device=str(self.config.device) if self.config.device else None,
            seed=seed if seed else 0,
            save_period=10,
        )
        self.logger.success(
            "Completed ultralytics API training " + f"for {epochs} epochs"
        )

    def evaluate(self, gt_path: str | None = None) -> str:
        """Evaluate the model using the COCOEvaluator."""
        assert self.model is not None, "Model is not initialised"
        self.eval_output = COCOEvaluator(self, gt_path).evaluate()
        self.logger.info(f"From COCO evaluator:\n{self.eval_output}")
        return f"{self.eval_output}"

    def get_model(self) -> YOLO | None:
        return self.model

    def __str__(self) -> str:
        """Return a string represntation of the pipeline."""
        return self.config.name or "pipeline"
