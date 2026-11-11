"""Write the label CSVs that the S2D data loader reads.

S2D reads a CSV with a header, the clip id in the first column (= name of the folder
of face crops) and the label(s) in the following column(s).

MAFW (single label, 11 classes, 5 folds) -> data/processed/MAFW/anno/set_<k>/{train,test}.csv
    columns: video_name,labels
    Class indices follow the official S2D order (alphabetical), so the released
    S2D MAFW checkpoints can be used unchanged:
    0 anger, 1 anxiety, 2 contempt, 3 disappointment, 4 disgust, 5 fear,
    6 happiness, 7 helplessness, 8 neutral, 9 sadness, 10 surprise

ElderReact (multi-label, 6 yes/no emotions) -> data/processed/ElderReact/anno/{train,dev,test}.csv
    columns: video_name,anger,disgust,fear,happiness,sadness,surprise

Run from the repo root:
    python s2d/make_annotations.py
"""
import argparse
import csv
from pathlib import Path

MAFW_CLASSES = ["anger", "anxiety", "contempt", "disappointment", "disgust", "fear",
                "happiness", "helplessness", "neutral", "sadness", "surprise"]
ELDER_CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
# Index of each ElderReact emotion in the MAFW class list (used later to initialise
# the 6-output head from the MAFW model and to compare zero-shot predictions).
ELDER_TO_MAFW = {c: MAFW_CLASSES.index(c) for c in ELDER_CLASSES}


def convert_mafw(mafw_root: Path, out_root: Path):
    split_root = mafw_root / "Train & Test Set" / "single" / "no_caption"
    for k in range(1, 6):
        for part in ("train", "test"):
            src = split_root / f"set_{k}" / f"{part}.txt"
            dst = out_root / f"set_{k}" / f"{part}.csv"
            dst.parent.mkdir(parents=True, exist_ok=True)
            n = 0
            with open(src) as f, open(dst, "w", newline="") as g:
                w = csv.writer(g, lineterminator="\n")
                w.writerow(["video_name", "labels"])
                for line in f:
                    if not line.strip():
                        continue
                    name, label = line.split()
                    w.writerow([Path(name).stem, MAFW_CLASSES.index(label.lower())])
                    n += 1
            print(f"MAFW set_{k} {part}: {n} clips -> {dst}")


def convert_elderreact(er_root: Path, out_root: Path):
    out_root.mkdir(parents=True, exist_ok=True)
    for part in ("train", "dev", "test"):
        src = er_root / f"{part}_labels.txt"
        dst = out_root / f"{part}.csv"
        n = 0
        with open(src) as f, open(dst, "w", newline="") as g:
            w = csv.writer(g, lineterminator="\n")
            w.writerow(["video_name"] + ELDER_CLASSES)
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                # file, 6 binary emotions, gender, valence
                w.writerow([Path(parts[0]).stem] + [int(float(x)) for x in parts[1:7]])
                n += 1
        print(f"ElderReact {part}: {n} clips -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mafw", default="data/MFAW_Data")
    ap.add_argument("--elderreact", default="data/ElderReact_Data")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()
    out = Path(args.out)
    convert_mafw(Path(args.mafw), out / "MAFW" / "anno")
    convert_elderreact(Path(args.elderreact), out / "ElderReact" / "anno")


if __name__ == "__main__":
    main()
