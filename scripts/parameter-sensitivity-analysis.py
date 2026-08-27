import json

from dataset_adaptive_yolo11.dmanager import DataManager

PYRAMID_LEVELS = ("P1", "P2", "P3", "P4", "P5")

PERTURBATIONS = {
    "baseline": (1.0, 1.0),
    "reference_-10%": (0.9, 1.0),
    "reference_+10%": (1.1, 1.0),
    "bandwidth_-10%": (1.0, 0.9),
    "bandwidth_+10%": (1.0, 1.1),
}


def calculate_branch_scores(
    feature_scores: dict[str, float],
) -> dict[str, float]:
    return {
        "A": (feature_scores["P1"] + feature_scores["P2"] + feature_scores["P3"]),
        "B": (feature_scores["P2"] + feature_scores["P3"] + feature_scores["P4"]),
        "C": (feature_scores["P3"] + feature_scores["P4"] + feature_scores["P5"]),
    }


def calculate_baseline_margin(branch_scores: dict[str, float]) -> float:
    scores = sorted(branch_scores.values(), reverse=True)

    best = scores[0]
    second_best = scores[1]

    return ((best - second_best) / best) * 100


def calculate_branch_outcome(
    branch_scores: dict[str, float],
) -> str:
    return max(branch_scores, key=branch_scores.get)


def calculate_dataset_scores(
    dmanager: DataManager,
    areas: list[float],
    perturbation: tuple[float, float],
) -> dict[str, float]:

    feature_scores = {level: 0.0 for level in PYRAMID_LEVELS}

    for area in areas:
        scores = dmanager.calculate_fitness_scores(
            area,
            preturb=perturbation,
        )

        for level, score in zip(PYRAMID_LEVELS, scores):
            feature_scores[level] += score

    return feature_scores


def main() -> None:
    dmanagers: dict[str, DataManager] = {
        "VD": DataManager("/data2/jd1/datasets/VisDrone/data.yaml"),
        "SDS": DataManager("/data2/jd1/datasets/sds-ds/sds/data.yaml"),
        "KITTI": DataManager("/data2/jd1/datasets/kitti/data.yaml"),
        "humancar_50m": DataManager("/data2/jd1/datasets/humancar_50m/data.yaml"),
        "LPR": DataManager("/data2/jd1/datasets/lpr/data.yaml"),
        "LPD": DataManager("/data2/jd1/datasets/landing-pad-detection-2-v4/data.yaml"),
        "MP": DataManager("/data2/jd1/datasets/medical-pills/data.yaml"),
    }

    output: dict[str, dict] = {}

    for dataset, dmanager in dmanagers.items():
        print(f"Processing {dataset}...")

        # Calculate areas ONCE

        areas = dmanager.get_annotation_areas()

        dataset_output = {}

        for name, perturbation in PERTURBATIONS.items():
            feature_scores = calculate_dataset_scores(
                dmanager,
                areas,
                perturbation,
            )

            branch_scores = calculate_branch_scores(feature_scores)

            dataset_output[name] = {
                "feature_scores": feature_scores,
                "branch_scores": branch_scores,
                "outcome": calculate_branch_outcome(branch_scores),
                "margin": calculate_baseline_margin(branch_scores),
            }

        output[dataset] = dataset_output

    print("Writing to psa-output.json")

    with open("psa-output.json", "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)


if __name__ == "__main__":
    main()
