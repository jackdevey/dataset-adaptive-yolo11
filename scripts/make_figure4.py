import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


FIGSIZE = (3.2, 2.25)
FONT_LABEL = 10
FONT_TICK = 10

plt.rcParams.update(
    {
        "font.size": FONT_TICK,
        "axes.labelsize": FONT_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
    }
)

# Short names for long class labels
LABEL_MAP = {
    "pedestrian": "ped.",
    "awning-tricycle": "awn.-tri.",
    "Person_sitting": "Pers. sit.",
    "life_saving_appliances": "life-saving",
}


def load_class_names(dataset: Path):
    yaml_path = (
        dataset
        if dataset.is_file()
        else next(p for p in dataset.iterdir() if p.suffix in {".yaml", ".yml"})
    )
    names = yaml.safe_load(yaml_path.read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    return names, yaml_path.parent


def collect_areas(root: Path, n_classes: int):
    """Return {class_id: [area_percent, ...]} from every YOLO label file."""
    areas = defaultdict(list)
    for f in root.rglob("labels/**/*.txt"):
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            if 0 <= cid < n_classes:
                areas[cid].append(float(parts[3]) * float(parts[4]) * 100)
    return areas


def labels_overlap(fig, ax, gap_px=2):
    """True if any horizontal x tick labels touch their neighbours."""
    fig.canvas.draw()
    boxes = [t.get_window_extent() for t in ax.get_xticklabels()]
    return any(a.x1 + gap_px > b.x0 for a, b in zip(boxes, boxes[1:]))


def plot(class_names, areas, out_stem: Path):
    ids = [i for i in range(len(class_names)) if areas[i]]
    labels = [LABEL_MAP.get(class_names[i], class_names[i]) for i in ids]
    data = [areas[i] for i in ids]
    counts = [len(a) for a in data]
    x = range(1, len(ids) + 1)

    # constrained layout fits the labels inside FIGSIZE
    fig, ax1 = plt.subplots(figsize=FIGSIZE, layout="constrained")
    ax2 = ax1.twinx()

    # Background bars: annotation counts (values hidden)
    ax2.bar(x, counts, width=0.72, alpha=0.10, zorder=1)
    ax2.set_ylim(0, max(counts) * 1.10)
    ax2.set_yticks([])
    ax2.set_ylabel("Number of annotations", labelpad=2)

    # Foreground boxplots: box area (% of image)
    line = dict(linewidth=0.8)
    bp = ax1.boxplot(
        data,
        positions=x,
        widths=0.42,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(linewidth=1.3),
        boxprops=line,
        whiskerprops=line,
        capprops=line,
        zorder=3,
    )
    for box in bp["boxes"]:
        box.set_facecolor("white")

    ax1.set_yscale("log")
    ax1.set_ylabel("Bounding-box area (%)", labelpad=2)
    ax1.set_xticks(list(x), labels)
    if labels_overlap(fig, ax1):  # only rotate when horizontal labels collide
        ax1.set_xticks(list(x), labels, rotation=45, ha="right", rotation_mode="anchor")
    ax1.tick_params(length=2.5, width=0.6, pad=2)
    ax1.set_xlim(0.5, len(ids) + 0.5)
    ax1.grid(axis="y", alpha=0.15, linewidth=0.5)

    # Boxplots drawn above bars
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax1.spines[s].set_linewidth(0.6)

    pdf = out_stem.parent / f"{out_stem.name}.pdf"
    png = out_stem.parent / f"{out_stem.name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    plt.close(fig)
    print(f"Saved {pdf}\nSaved {png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", type=Path, help="YOLO dataset dir or data.yaml")
    p.add_argument("outdir", type=Path, help="Output directory (created if missing)")
    p.add_argument("--name", required=True, help="Output file stem, e.g. visdrone")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    names, root = load_class_names(args.dataset)
    plot(names, collect_areas(root, len(names)), args.outdir / args.name)


if __name__ == "__main__":
    main()
