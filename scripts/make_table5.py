import re
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

HEADER = r"""\begin{tabular}{l|llll}
        \toprule
            \textbf{Dataset}
            & \textbf{Optimizer}
            & \textbf{Initial LR}
            & \textbf{Momentum}
            & \textbf{Weight Decay}\\
        \midrule"""

FLOAT = r"[-+]?\d*\.?\d+"


def parse_log(path):
    text = path.read_text(errors="ignore")
    nominal = re.search(rf"'optimizer=auto' found, ignoring 'lr0=({FLOAT})' and 'momentum=({FLOAT})'", text)
    resolved = re.search(rf"optimizer: (\w+)\(lr=({FLOAT}), momentum=({FLOAT})\) with parameter groups \d+ weight\(decay={FLOAT}\), \d+ weight\(decay=({FLOAT})\), \d+ bias", text)
    return {
        "nominal": (nominal.group(1), nominal.group(2)),
        "resolved": (resolved.group(1), resolved.group(2), resolved.group(3), resolved.group(4)),
    }


def main():
    print(HEADER)
    for folder, name in DATASETS.items():
        logs = sorted(EXPERIMENTS.glob(f"{folder}/*/*/output.log"))
        runs = {str(p.parent.relative_to(EXPERIMENTS)): parse_log(p) for p in logs}

        nominal = {r["nominal"] for r in runs.values()}
        resolved = {r["resolved"] for r in runs.values()}
        assert len(nominal) == 1, f"{folder}: nominal args differ {nominal}"
        assert len(resolved) == 1, f"{folder}: resolved optimizer differs {resolved}"

        opt, lr, mom, wd = resolved.pop()
        lr0, mom0 = nominal.pop()
        print(f"            % {len(runs)} runs checked: {', '.join(runs)}")
        print(f"            % nominal lr0={lr0} momentum={mom0} -> ignored by optimizer=auto")
        print(f"            {name} & {opt} & {lr} & {mom} & {wd}\\\\")
    print(r"""        \botrule
        \end{tabular}""")


if __name__ == "__main__":
    main()
