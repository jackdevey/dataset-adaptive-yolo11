from dataset_adaptive_yolo11.pipeline import Pipeline

p = Pipeline(
    dataset_yaml_path="DATASET_PATH",
    project="PROJECT_NAME",
    config={
        "device": "DEVICE",
        "name": "RUN_NAME",
    },
)

# For default behaviour - select a branch
p.create_model()

# For ablation - provide a branch to select (A, B or C)
# p.create_model(abl_branch="BRANCH_CHAR")

# To select default YOLO11 (or other model config / weights)
# p.create_model(skip_and_use="yolo11m.yaml")

p.train(epochs=100, batch=16, seed=1)

p.evaluate()
