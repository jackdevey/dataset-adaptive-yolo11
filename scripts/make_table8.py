import re
import statistics
from pathlib import Path

EXPERIMENTS = Path("experiments")

DATASETS = {
    "vd": "VisDrone-DET",
    "sds": "SeaDronesSee",
    "kitti": "KITTI",
    "hc5": "humancar 50m",
    "lpr": "License Plate Rec.",
    "lpd2": "Landing Pad Det.",
    "mp": "Medical Pills",
}

METRICS = [
    ("log", "map5095"),
    ("log", "map50"),
    ("coco", "ap_50_95_small"),
    ("coco", "ap_50_95_medium"),
    ("coco", "ap_50_95_large"),
    ("coco", "ar_50_95_small"),
    ("coco", "ar_50_95_medium"),
    ("coco", "ar_50_95_large"),
]

HEADER = r"""\begin{tabular}{ll|ll|lll|lll}
        \toprule
            \textbf{Dataset}
            & \textbf{Branch}
            & \textbf{mAP}$^{\mathbf{50}}_{\mathbf{\text{--}95}}$
            & \textbf{mAP}$_{\mathbf{50}}$
            & \textbf{AP}$_{\textbf{S}}$
            & \textbf{AP}$_{\textbf{M}}$
            & \textbf{AP}$_{\textbf{L}}$
            & \textbf{AR}$_{\textbf{S}}$
            & \textbf{AR}$_{\textbf{M}}$
            & \textbf{AR}$_{\textbf{L}}$\\
        \midrule"""

FLOAT = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def parse_coco(path):
    pairs = re.findall(rf"(\w+)=(?:np\.float64\()?({FLOAT})\)?", path.read_text())
    return {k: float(v) for k, v in pairs}


def parse_log(path):
    text = path.read_text(errors="ignore")
    text = text[text.rfind("Validating") :]
    m = re.search(
        rf"^\s+all\s+\d+\s+\d+\s+{FLOAT}\s+{FLOAT}\s+({FLOAT})\s+({FLOAT})", text, re.M
    )
    return {"map50": float(m.group(1)), "map5095": float(m.group(2))}


def branch_map(dataset_dir):
    variants = {p.name for p in dataset_dir.iterdir() if p.is_dir()}
    mapping = {}
    for v in variants:
        if v == "default-yolo11":
            mapping[v] = ("C", "dataset-adapted" not in variants)
        elif v.startswith("ablation-branch-"):
            mapping[v] = (v[-1].upper(), False)
    if "dataset-adapted" in variants:
        remaining = {"A", "B", "C"} - {b for b, _ in mapping.values()}
        mapping["dataset-adapted"] = (remaining.pop(), True)
    return mapping


def summarise(vals):
    if any(v < 0 for v in vals):
        return None
    pct = [v * 100 for v in vals]
    return statistics.mean(pct), statistics.stdev(pct)


def cell(mean_sd, bold):
    mean = f"{mean_sd[0]:05.2f}"
    mean = rf"{{\bfseries {mean}}}" if bold else mean + " "
    return rf"{mean}\footnotesize{{$\pm${mean_sd[1]:.1f}}}"


def main():
    lines = [HEADER]

    for i, (ds, name) in enumerate(DATASETS.items()):
        stats, selected = {}, {}
        for variant, (branch, sel) in branch_map(EXPERIMENTS / ds).items():
            runs = sorted(
                p for p in (EXPERIMENTS / ds / variant).iterdir() if p.is_dir()
            )
            values = {key: [] for _, key in METRICS}
            for run in runs:
                log = parse_log(run / "output.log")
                coco = parse_coco(run / "coco_eval.txt")
                for source, key in METRICS:
                    values[key].append(log[key] if source == "log" else coco[key])
            stats[branch] = {key: summarise(v) for key, v in values.items()}
            selected[branch] = sel

        best = {}
        for _, key in METRICS:
            means = [round(s[key][0], 2) for s in stats.values() if s[key]]
            best[key] = max(means) if means else None

        for b in ("A", "B", "C"):
            row = [name, rf"$\lambda_{b}$" + (r"$^{*}$" if selected[b] else "")]
            for _, key in METRICS:
                s = stats[b][key]
                row.append("-" if s is None else cell(s, round(s[0], 2) == best[key]))
            lines.append(" & ".join(row) + r" \\")

        if i != len(DATASETS) - 1:
            lines.append(r"\midrule")

    lines.append(r"\botrule")
    lines.append(r"\end{tabular}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
