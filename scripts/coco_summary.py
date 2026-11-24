import re
import statistics as st
from pathlib import Path

FIELDS = {"AP": "ap_50_95_all", "AP50": "ap_50_all"}

for ds in ["vd", "sds"]:
    for cfg in ["default-yolo11", "dataset-adapted"]:
        vals = {k: [] for k in FIELDS}
        for seed in ["1", "2", "3"]:
            path = Path(f"experiments/{ds}/{cfg}/{seed}/coco_eval.txt")
            if not path.exists():
                print(f"missing: {path}")
                continue
            text = path.read_text()
            for k, field in FIELDS.items():
                m = re.search(rf"\b{field}=(?:np\.float64\()?([\d.eE+-]+)", text)
                if m is None:
                    raise ValueError(f"{field} not found in {path}")
                vals[k].append(float(m.group(1)) * 100)
        if all(len(v) == 3 for v in vals.values()):
            out = ", ".join(
                f"{k} {st.mean(v):.2f} ± {st.stdev(v):.1f} (seeds: {', '.join(f'{x:.2f}' for x in v)})"
                for k, v in vals.items()
            )
            print(f"{ds:4s} {cfg:16s} {out}")
