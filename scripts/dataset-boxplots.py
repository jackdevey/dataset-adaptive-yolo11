import argparse
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def truncate_label(label, max_chars=10):
    label = str(label)

    if len(label) > max_chars:
        return label[:max_chars] + "..."

    return label


def find_dataset_yaml(dataset_path: Path) -> Path:
    """Resolve either a YOLO dataset YAML or a directory containing one."""
    if dataset_path.is_file():
        if dataset_path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(f"Expected a YAML file, got: {dataset_path}")
        return dataset_path

    yaml_files = list(dataset_path.glob("*.yaml")) + list(dataset_path.glob("*.yml"))

    if not yaml_files:
        raise FileNotFoundError(
            f"No YAML dataset configuration found in {dataset_path}"
        )

    if len(yaml_files) > 1:
        print(f"Warning: found multiple YAML files. Using: {yaml_files[0].name}")

    return yaml_files[0]


def load_class_names(yaml_path: Path):
    """Load YOLO class names from the dataset YAML."""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    if "names" not in config:
        raise KeyError(f"'names' field not found in {yaml_path}")

    names = config["names"]

    # YOLO supports either:
    # names: [cat, dog]
    # or
    # names:
    #   0: cat
    #   1: dog
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]

    return names, config


def find_label_directories(dataset_root: Path):
    """
    Find YOLO label directories.

    Supports common layouts such as:
        dataset/
            images/train/
            images/val/
            labels/train/
            labels/val/
    """
    labels_root = dataset_root / "labels"

    if labels_root.exists():
        dirs = [p for p in labels_root.iterdir() if p.is_dir()]

        if dirs:
            return dirs

        # labels may exist directly inside labels/
        if list(labels_root.glob("*.txt")):
            return [labels_root]

    # Fallback: recursively search for directories named "labels"
    label_dirs = []

    for path in dataset_root.rglob("labels"):
        if path.is_dir():
            subdirs = [p for p in path.iterdir() if p.is_dir()]
            label_dirs.extend(subdirs if subdirs else [path])

    return list(dict.fromkeys(label_dirs))


def analyze_labels(label_dirs, class_names):
    class_counts = Counter()
    class_areas = defaultdict(list)

    total_files = 0
    files_with_annotations = 0
    malformed_lines = 0

    for label_dir in label_dirs:
        for label_file in label_dir.rglob("*.txt"):
            total_files += 1
            has_annotation = False

            with open(label_file, "r") as f:
                for line_number, line in enumerate(f, start=1):
                    line = line.strip()

                    if not line:
                        continue

                    parts = line.split()

                    if len(parts) < 5:
                        print(
                            f"Warning: malformed annotation in "
                            f"{label_file}:{line_number}"
                        )
                        malformed_lines += 1
                        continue

                    try:
                        class_id = int(float(parts[0]))
                        width = float(parts[3])
                        height = float(parts[4])
                    except ValueError:
                        print(f"Warning: invalid values in {label_file}:{line_number}")
                        malformed_lines += 1
                        continue

                    if class_id < 0 or class_id >= len(class_names):
                        print(
                            f"Warning: unknown class ID {class_id} in "
                            f"{label_file}:{line_number}"
                        )
                        continue

                    area = width * height

                    class_counts[class_id] += 1
                    class_areas[class_id].append(area)

                    has_annotation = True

            if has_annotation:
                files_with_annotations += 1

    return (
        class_counts,
        class_areas,
        total_files,
        files_with_annotations,
        malformed_lines,
    )


def build_summary(class_names, class_counts, class_areas):
    total_annotations = sum(class_counts.values())

    rows = []

    for class_id, class_name in enumerate(class_names):
        count = class_counts[class_id]
        areas = class_areas[class_id]

        prevalence = 100 * count / total_annotations if total_annotations > 0 else 0

        if areas:
            series = pd.Series(areas)

            rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "annotations": count,
                    "prevalence_percent": prevalence,
                    "mean_area": series.mean(),
                    "median_area": series.median(),
                    "std_area": series.std(),
                    "min_area": series.min(),
                    "q1_area": series.quantile(0.25),
                    "q3_area": series.quantile(0.75),
                    "max_area": series.max(),
                }
            )
        else:
            rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "annotations": 0,
                    "prevalence_percent": 0,
                    "mean_area": None,
                    "median_area": None,
                    "std_area": None,
                    "min_area": None,
                    "q1_area": None,
                    "q3_area": None,
                    "max_area": None,
                }
            )

    return pd.DataFrame(rows)


def plot_area_boxplot(class_names, class_areas, dataset_name):
    labels = []
    data = []
    counts = []

    for class_id, class_name in enumerate(class_names):
        areas = class_areas[class_id]

        if areas:
            labels.append(truncate_label(class_name, max_chars=10))

            # Convert normalized area (0–1) to percentage of image area
            data.append([area * 100 for area in areas])

            # Number of annotations in this class
            counts.append(len(areas))

    if not data:
        print("No bounding boxes found; skipping area boxplot.")
        return

    # --------------------------------
    # Compact publication-sized figure
    # --------------------------------
    fig, ax1 = plt.subplots(figsize=(3.2, 2.25))

    x_positions = list(range(1, len(labels) + 1))

    # --------------------------------
    # Secondary Y-axis:
    # class distribution
    # --------------------------------
    ax2 = ax1.twinx()

    ax2.bar(
        x_positions,
        counts,
        width=0.72,
        alpha=0.10,
        zorder=1,
    )

    # Keep the axis description but hide exact values
    ax2.set_ylabel(
        "Number of annotations",
        fontsize=8,
        labelpad=2,
    )

    ax2.set_yticks([])
    ax2.grid(False)

    if max(counts) > 0:
        ax2.set_ylim(0, max(counts) * 1.10)

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # --------------------------------
    # Compact boxplot styling
    # --------------------------------
    medianprops = dict(
        linewidth=1.3,
    )

    boxprops = dict(
        linewidth=0.8,
    )

    whiskerprops = dict(
        linewidth=0.8,
    )

    capprops = dict(
        linewidth=0.8,
    )

    # --------------------------------
    # Bounding-box area boxplots
    # --------------------------------
    bp = ax1.boxplot(
        data,
        positions=x_positions,
        widths=0.42,
        labels=labels,
        showfliers=False,
        patch_artist=True,
        medianprops=medianprops,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        zorder=3,
    )

    # Make boxes mostly opaque so faded bars
    # do not visually interfere with them.
    for box in bp["boxes"]:
        box.set_facecolor("white")
        box.set_alpha(0.95)

    # --------------------------------
    # Primary Y-axis
    # --------------------------------
    ax1.set_yscale("log")

    ax1.set_ylabel(
        "Bounding-box area (%)",
        fontsize=8,
        labelpad=2,
    )

    ax1.tick_params(
        axis="y",
        labelsize=7,
        direction="out",
        length=2.5,
        width=0.6,
        pad=2,
    )

    ax1.tick_params(
        axis="x",
        labelsize=7,
        direction="out",
        length=2.5,
        width=0.6,
        pad=2,
        rotation=45,
    )

    # Light horizontal grid matching the mAP figures
    ax1.grid(
        axis="y",
        alpha=0.15,
        linewidth=0.5,
    )

    # --------------------------------
    # Spine styling
    # --------------------------------
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax1.spines["left"].set_linewidth(0.6)
    ax1.spines["bottom"].set_linewidth(0.6)

    # --------------------------------
    # Layer ordering
    # --------------------------------
    # Put boxplots visually above the faded bars.
    ax1.set_zorder(ax2.get_zorder() + 1)

    # Transparent primary-axis background so bars remain visible.
    ax1.patch.set_visible(False)

    # --------------------------------
    # X-axis spacing
    # --------------------------------
    ax1.set_xlim(0.5, len(labels) + 0.5)
    ax1.margins(x=0)

    # --------------------------------
    # Explicit figure margins
    # --------------------------------
    fig.subplots_adjust(
        left=0.18,
        right=0.995,
        bottom=0.30,
        top=0.995,
    )

    # --------------------------------
    # Save
    # --------------------------------
    fig.savefig(
        f"{dataset_name}.pdf",
        bbox_inches="tight",
        pad_inches=0.01,
    )

    fig.savefig(
        f"{dataset_name}.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.close(fig)

    print(f"Saved {dataset_name}.pdf")
    print(f"Saved {dataset_name}.png")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyse class distribution and bounding-box areas in a YOLO dataset."
        )
    )

    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to YOLO dataset directory or data.yaml",
    )

    parser.add_argument(
        "--name",
        type=str,
        help="Dataset name",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("yolo_dataset_analysis"),
        help="Output directory",
    )

    args = parser.parse_args()

    dataset_yaml = find_dataset_yaml(args.dataset)

    dataset_name = args.name

    # Fall back to the YAML stem if --name is omitted
    if dataset_name is None:
        dataset_name = dataset_yaml.stem

    class_names, config = load_class_names(dataset_yaml)

    dataset_root = dataset_yaml.parent

    print(f"Dataset YAML: {dataset_yaml}")
    print(f"Dataset name: {dataset_name}")
    print(f"Dataset root: {dataset_root}")
    print(f"Classes ({len(class_names)}): {class_names}")

    label_dirs = find_label_directories(dataset_root)

    if not label_dirs:
        raise FileNotFoundError(f"No label directories found under {dataset_root}")

    print("\nLabel directories:")
    for directory in label_dirs:
        print(f"  - {directory}")

    (
        class_counts,
        class_areas,
        total_files,
        files_with_annotations,
        malformed_lines,
    ) = analyze_labels(
        label_dirs,
        class_names,
    )

    summary = build_summary(
        class_names,
        class_counts,
        class_areas,
    )

    total_annotations = int(summary["annotations"].sum())

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(f"Label files:             {total_files:,}")
    print(f"Files with annotations:  {files_with_annotations:,}")
    print(f"Total annotations:       {total_annotations:,}")
    print(f"Malformed lines skipped: {malformed_lines:,}")

    print("\nCLASS DISTRIBUTION")

    print(
        summary[
            [
                "class_id",
                "class_name",
                "annotations",
                "prevalence_percent",
                "median_area",
            ]
        ].to_string(
            index=False,
            formatters={
                "prevalence_percent": lambda x: f"{x:.2f}%",
                "median_area": lambda x: f"{x:.6f}" if pd.notna(x) else "-",
            },
        )
    )

    plot_area_boxplot(
        class_names,
        class_areas,
        dataset_name,
    )


if __name__ == "__main__":
    main()