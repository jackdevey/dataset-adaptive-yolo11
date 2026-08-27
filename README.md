# Source code and experimental artefacts for '[A Dataset-Adaptive YOLO11 Architecture Modification Framework for Enhancing Small-Object Detection]()'

**Jack Devey**<sup>1</sup> <a href="https://orcid.org/0009-0002-9513-2817"><img src="https://upload.wikimedia.org/wikipedia/commons/0/06/ORCID_iD.svg" width="16"></a>, **Moad Idrissi**<sup>2</sup> <a href="https://orcid.org/0000-0002-9995-3180"><img src="https://upload.wikimedia.org/wikipedia/commons/0/06/ORCID_iD.svg" width="16"></a>, **Haitham Hassan Mahmoud**<sup>1</sup>, **Mohamed Gaber**<sup>1</sup>, **Rehan Bhana**<sup>1</sup>

1. Department of Computer Science, Birmingham City University, Birmingham B4 7BD, UK </br> `jack.devey@mail.bcu.ac.uk, {haitham.mahmoud,mohamed.gaber,rehan.bhana}@bcu.ac.uk`
2. School of Computing and Data Science, Oryx Universal College | Liverpool John Moores University (OUC-LJMU), Doha, Qatar </br> `moad.i@oryx.edu.qa`

## Dataset Adaptive YOLO11

## Experiments

Experimental artefacts are organised by dataset, model (branch) type, and run, with each run corresponding to a different random seed.

For example, the files in `experiments/sds/dataset-adapted/2` correspond to the second run of the dataset-adapted approach on the SeaDronesSee dataset.

For each experiment, the following files are provided:

- `coco_eval.txt` — COCO evaluation output produced by the Python COCO API.
- `config.json` — Training configuration reported by the Ultralytics API.
- `output.log` — Full training log produced by the Ultralytics API.
- `results.csv` — Per-epoch training and validation metrics produced by the Ultralytics API.

## Scripts

Miscellaneous scripts used to analyse datasets, prepare figures, and conduct the parameter sensitivity analysis.

The scripts may import from the `dataset_adaptive_yolo11` package and should therefore be executed from the repository root, for example:

```sh
python -m scripts.dataset-boxplots [...]
```

The scripts are provided in the form used during the experiments and may contain hard-coded file paths. These paths should be adjusted as required for the local environment.
