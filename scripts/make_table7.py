import re
import statistics
from pathlib import Path

EXPERIMENTS = Path("experiments")

DATASETS = {
    "vd": "VisDrone-DET",
    "sds": "SeaDronesSee",
    "kitti": "KITTI",
    "hc5": r"humancar\_50m",
    "lpr": "License Plate Rec.",
    "lpd2": "Landing Pad Det.",
    "mp": "Medical Pills",
}

HEADER = r"""\begin{tabular}{ll|ll|ll}
        \toprule
            \textbf{YOLO11 Model} 
            & \textbf{Dataset}
            & \textbf{mAP}$^{\mathbf{50}}_{\mathbf{\text{--}95}}$
            & \textbf{mAP}$_{\mathbf{50}}$
            & $\mathbf{\Delta}$\textbf{mAP}$^{\mathbf{50}}_{\mathbf{\text{--}95}}$
            & $\mathbf{\Delta}$\textbf{mAP}$_{\mathbf{50}}$\\
        \midrule"""

FLOAT = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def parse_log(path):
    text = path.read_text(errors="ignore")
    text = text[text.rfind("Validating") :]
    m = re.search(
        rf"^\s+all\s+\d+\s+\d+\s+{FLOAT}\s+{FLOAT}\s+({FLOAT})\s+({FLOAT})", text, re.M
    )
    return {"map50": float(m.group(1)), "map5095": float(m.group(2))}


def summarise(variant_dir):
    runs = sorted(p for p in variant_dir.iterdir() if p.is_dir())
    logs = [parse_log(run / "output.log") for run in runs]
    out = {}
    for key in ("map5095", "map50"):
        pct = [l[key] * 100 for l in logs]
        out[key] = (statistics.mean(pct), statistics.stdev(pct))
    return out


def branch_of(dataset_dir):
    variants = {p.name for p in dataset_dir.iterdir() if p.is_dir()}
    if "dataset-adapted" not in variants:
        return "C"
    used = {v[-1].upper() for v in variants if v.startswith("ablation-branch-")}
    return ({"A", "B"} - used).pop()


def cell(mean_sd, bold):
    mean = f"{mean_sd[0]:05.2f}"
    mean = rf"{{\bfseries {mean}}}" if bold else mean + " "
    return rf"{mean}\footnotesize{{$\pm${mean_sd[1]:.1f}}}"


def delta(base, adapted):
    b, a = round(base[0], 2), round(adapted[0], 2)
    change = (a - b) / b * 100
    sign = "+" if change >= 0 else "$-$"
    return rf"{sign}{abs(change):05.2f}\%"


def main():
    lines = [HEADER]

    for i, (ds, name) in enumerate(DATASETS.items()):
        branch = branch_of(EXPERIMENTS / ds)
        default = branch == "C"
        base = summarise(EXPERIMENTS / ds / "default-yolo11")
        adapted = base if default else summarise(EXPERIMENTS / ds / "dataset-adapted")

        rows = []
        for label, own, other, is_base in (
            ("Baseline", base, adapted, True),
            (rf"Dataset-Adapted [$\lambda_{branch}$]", adapted, base, False),
        ):
            row = [f"            {label} & {name}"]
            for key, comment in (("map5095", "map5095"), ("map50", "map50")):
                mine, theirs = round(own[key][0], 2), round(other[key][0], 2)
                bold = not default and (
                    mine > theirs or (mine == theirs and not is_base)
                )
                row.append(f"            % {comment}")
                row.append(f"            & {cell(own[key], bold)}")
            for key in ("map5095", "map50"):
                d = "-" if default or is_base else delta(base[key], adapted[key])
                row.append(f"            & {d}")
            row[-1] += r" \\"
            rows.extend(row)
        lines.extend(rows)

        if i != len(DATASETS) - 1:
            lines.append(r"        \midrule")

    lines.append(r"        \botrule")
    lines.append(r"        \end{tabular}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
