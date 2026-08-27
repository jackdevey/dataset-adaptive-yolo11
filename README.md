# Source code and experimental artefacts for '[A Dataset-Adaptive YOLO11 Architecture Modification Framework for Enhancing Small-Object Detection]()'

**Jack Devey**<sup>1</sup> <a href="https://orcid.org/0009-0002-9513-2817"><img src="https://upload.wikimedia.org/wikipedia/commons/0/06/ORCID_iD.svg" width="16"></a>, **Moad Idrissi**<sup>2</sup> <a href="https://orcid.org/0000-0002-9995-3180"><img src="https://upload.wikimedia.org/wikipedia/commons/0/06/ORCID_iD.svg" width="16"></a>, **Haitham Hassan Mahmoud**<sup>1</sup>, **Mohamed Gaber**<sup>1</sup>, **Rehan Bhana**<sup>1</sup>

1. Department of Computer Science, Birmingham City University, Birmingham B4 7BD, UK </br> `jack.devey@mail.bcu.ac.uk, {haitham.mahmoud,mohamed.gaber,rehan.bhana}@bcu.ac.uk`
2. School of Computing and Data Science, Oryx Universal College | Liverpool John Moores University (OUC-LJMU), Doha, Qatar </br> `moad.i@oryx.edu.qa`

## Installation

The code was developed and used with Python 3.13.2. A CUDA-capable GPU is recommended for model training.

Clone the repository and enter the project directory:

```sh
git clone https://github.com/jackdevey/dataset-adaptive-yolo11
cd dataset-adaptive-yolo11
```

It is recommended to create a virtual environment before installing the dependencies:

```sh
python 3.13 -m venv .venv
source .venv/bin/activate
```

Install PyTorch for the target environment. The experiments reported in the paper used PyTorch 2.6.0 with CUDA 11.8. For reproduction of this environment:

```sh
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

Users with different CUDA requirements or without a CUDA-capable GPU should install the appropriate PyTorch build for their environment.

Then install the remaining dependencies:

```sh
pip install -r requirements.txt
```

See `example.py` for an example of using the dataset-adaptive framework.

## Contents

### Dataset-Adaptive YOLO11 - `dataset_adaptive_yolo11/`

Implementation of the proposed dataset-adaptive framework as a python module. See `example.py` for an example of using the dataset-adaptive framework.

### Experiments - `experiments/`

Experimental artefacts are organised by dataset, model (branch) type, and run, with each run corresponding to a different random seed.

For example, the files in `experiments/sds/dataset-adapted/2` correspond to the second run of the dataset-adapted approach on the SeaDronesSee dataset.

For each experiment, the following files are provided:

- `coco_eval.txt` — COCO evaluation output produced by the Python COCO API.
- `config.json` — Training configuration reported by the Ultralytics API.
- `output.log` — Full training log produced by the Ultralytics API.
- `results.csv` — Per-epoch training and validation metrics produced by the Ultralytics API.

### Experiments - `scripts/`

Miscellaneous scripts used to analyse datasets, prepare figures, and conduct the parameter sensitivity analysis.

The scripts may import from the `dataset_adaptive_yolo11` package and should therefore be executed from the repository root, for example:

```sh
python -m scripts.dataset-boxplots [...]
```

The scripts are provided in the form used during the experiments and may contain hard-coded file paths. These paths should be adjusted as required for the local environment.

### Example - 'example.py'

`example.py` provides a minimal example of the complete training and evaluation pipeline. It demonstrates how to configure a dataset and experiment, create a model using the proposed dataset-adaptive approach, train the resulting model, and evaluate its performance.

By default, `create_model()` automatically selects a branch based on the dataset. The example also shows how to explicitly select a branch (A, B, or C) for ablation experiments, or bypass the dataset-adaptive approach and provide a standard YOLO model configuration or weights.

Before running the example, replace DATASET_PATH, PROJECT_NAME, DEVICE, and RUN_NAME with values appropriate for the target environment.

The example should then be executed from the repository root:

```sh
python -m example
```
