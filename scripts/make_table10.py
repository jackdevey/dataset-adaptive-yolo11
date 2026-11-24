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

HEADER = r"""\begin{tabular}{ll|ll|lll}
        \toprule
            \textbf{Dataset}
            & \textbf{Branch} 
            & \textbf{mAP}$^{\mathbf{50}}_{\mathbf{\text{--}95}}$
            & \textbf{mAP}$_{\mathbf{50}}$
            & \textbf{Parameters}
            & \textbf{GFLOPs}
            & \textbf{Infer. Latency}\\
        \midrule"""

FLOAT = r"[-+]?\d*\.?\d+"


def parse_log(path):
    text = path.read_text(errors="ignore")
    text = text[text.rfind("Validating"):]
    m = re.search(rf"^\s+all\s+\d+\s+\d+\s+{FLOAT}\s+{FLOAT}\s+({FLOAT})\s+({FLOAT})", text, re.M)
    summary = re.search(r"summary \(fused\): \d+ layers, ([\d,]+) parameters, \d+ gradients, ([\d.]+) GFLOPs", text)
    speed = re.search(rf"Speed: {FLOAT}ms preprocess, ({FLOAT})ms inference", text)
    return {
        "map50": float(m.group(1)) * 100,
        "map5095": float(m.group(2)) * 100,
        "params": int(summary.group(1).replace(",", "")),
        "gflops": float(summary.group(2)),
        "latency": float(speed.group(1)),
    }


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


def mean_sd(vals):
    return statistics.mean(vals), statistics.stdev(vals)


def cell(s, bold=False, space=True):
    mean = f"{s[0]:05.2f}"
    if bold:
        mean = rf"{{\bfseries {mean}}}"
    elif space:
        mean += " "
    return rf"{mean}\footnotesize{{$\pm${s[1]:.1f}}}"


def main():
    lines = [HEADER]

    for i, (ds, name) in enumerate(DATASETS.items()):
        stats, selected = {}, {}
        for variant, (branch, sel) in branch_map(EXPERIMENTS / ds).items():
            runs = sorted(p for p in (EXPERIMENTS / ds / variant).iterdir() if p.is_dir())
            logs = [parse_log(run / "output.log") for run in runs]
            assert len({l["params"] for l in logs}) == 1, f"{ds}/{variant}: params differ across seeds"
            stats[branch] = {
                "map5095": mean_sd([l["map5095"] for l in logs]),
                "map50": mean_sd([l["map50"] for l in logs]),
                "latency": mean_sd([l["latency"] for l in logs]),
                "params": logs[0]["params"],
                "gflops": logs[0]["gflops"],
            }
            selected[branch] = sel

        best = {key: max(round(s[key][0], 2) for s in stats.values()) for key in ("map5095", "map50")}

        for b in ("A", "B", "C"):
            s = stats[b]
            lines.append(rf"            {name} & $\lambda_{b}$" + (r" $\ast$" if selected[b] else ""))
            for key in ("map5095", "map50"):
                lines.append(f"            % {key}")
                lines.append(f"            & {cell(s[key], round(s[key][0], 2) == best[key])}")
            lines.append("            % params")
            lines.append(f"            & {s['params']:,}")
            lines.append("            % gflops")
            lines.append(f"            & {s['gflops']:.1f}")
            lines.append("            % inference latency")
            lines.append(rf"            & {cell(s['latency'], space=False)}\\")

        if i != len(DATASETS) - 1:
            lines.append(r"        \midrule")

    lines.append(r"        \botrule")
    lines.append(r"        \end{tabular}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
