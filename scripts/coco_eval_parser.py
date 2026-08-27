import argparse
import re


def extract_coco_metrics(text: str) -> dict[str, float]:
    metrics = {}

    for metric in ["ap", "ar"]:
        for size in ["small", "medium", "large"]:
            pattern = rf"{metric}_50_95_{size}=np\.float64\(([-+0-9.eE]+)\)"
            match = re.search(pattern, text)

            if match:
                metrics[f"{metric.upper()}_{size[0].upper()}"] = float(match.group(1))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Extract COCO metrics from text")
    parser.add_argument(
        "path",
        type=str,
        help="The path to a text file containing the COCO eval results in python dataclass format",
    )
    args = parser.parse_args()

    with open(args.path, "r") as f:
        text = f.read()

    m = extract_coco_metrics(text)
    print(f"AP S/M/L: {m['AP_S']} / {m['AP_M']} / {m['AP_L']}")
    print(f"AR S/M/L: {m['AR_S']} / {m['AR_M']} / {m['AR_L']}")


if __name__ == "__main__":
    main()
