import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FIGSIZE = (3.2, 2.25)
FONT_LABEL = 10
FONT_TICK = 10

plt.rcParams.update(
    {
        "font.size": FONT_TICK,
        "axes.labelsize": FONT_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_TICK,
    }
)

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"

# (experiment folder, legend label, line style)
MODELS = [
    ("dataset-adapted", "Dataset-Adapted", dict(color="C0", linestyle="-")),
    ("default-yolo11", "Baseline", dict(color="C1", linestyle="--")),
]

METRICS = {
    "mAP50-95": ("metrics/mAP50-95(B)", r"mAP$_{50\text{–}95}$"),
    "mAP50": ("metrics/mAP50(B)", r"mAP$_{50}$"),
}


def load_runs(model_dir: Path, column: str) -> pd.DataFrame:
    """Return a DataFrame indexed by epoch with one column per seed."""
    csvs = sorted(model_dir.glob("*/results.csv"))
    if not csvs:
        available = sorted(p.name for p in model_dir.parent.glob("*") if p.is_dir())
        raise FileNotFoundError(
            f"No */results.csv in {model_dir}. "
            f"Available for this dataset: {', '.join(available) or 'none'}"
        )

    runs = {}
    for csv in csvs:
        df = pd.read_csv(csv)
        df.columns = df.columns.str.strip()  # Ultralytics pads some headers
        runs[csv.parent.name] = df.set_index("epoch")[column]
    return pd.DataFrame(runs)


def save(fig, out_stem: Path):
    # No bbox_inches="tight": it would change the figure size
    pdf = out_stem.parent / f"{out_stem.name}.pdf"
    png = out_stem.parent / f"{out_stem.name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    plt.close(fig)
    print(f"Saved {pdf}\nSaved {png}")


def plot(dataset: str, metric: str, out_stem: Path):
    column, ylabel = METRICS[metric]

    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")

    for folder, label, style in MODELS:
        model_dir = EXPERIMENTS_DIR / dataset / folder
        if folder == "dataset-adapted" and not model_dir.is_dir():
            print("No dataset-adapted runs; plotting baseline only")
            continue
        runs = load_runs(model_dir, column)
        mean, std = runs.mean(axis=1), runs.std(axis=1)

        ax.fill_between(
            runs.index,
            mean - std,
            mean + std,
            color=style["color"],
            alpha=0.2,
            linewidth=0,
        )
        ax.plot(runs.index, mean, label=label, linewidth=1.3, **style)

        print(
            f"{label}: {runs.shape[1]} runs, final {metric} "
            f"{mean.iloc[-1]:.3f} ± {std.iloc[-1]:.3f}"
        )

    ax.set_xlabel("Epoch", labelpad=2)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.set_xlim(0, runs.index.max())
    ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
    ax.tick_params(length=2.5, width=0.6, pad=2)
    ax.grid(axis="y", alpha=0.15, linewidth=0.5)
    # Curves rise from the bottom-left, so the bottom-right is always empty
    ax.legend(loc="lower right", frameon=False, handlelength=1.8, borderaxespad=0.2)

    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.6)

    save(fig, out_stem)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", help="Folder name under experiments/, e.g. hc5")
    p.add_argument("outdir", type=Path, help="Output directory (created if missing)")
    p.add_argument("--metric", choices=METRICS, default="mAP50-95")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    plot(args.dataset, args.metric, args.outdir / args.dataset)


if __name__ == "__main__":
    main()
