from dataset_adaptive_yolo11.pipeline import Pipeline

p = Pipeline(dataset_yaml_path="", project="", config={"device": "", "name": ""})

p.create_model()

p.train(epochs=100, batch=16, seed=1)

p.evaluate()
