"""Run a stage from the terminal - the same functions the notebooks call.

    python tools/run.py preprocess --model s2d [--diagnostic]
    python tools/run.py train      --model maedfer [--resume]
    python tools/run.py evaluate   --model clustering [--runs ae_kmeans]
    python tools/run.py patients   --model s2d [--preset maedfer] [--run baseline]
    python tools/run.py compare    [--runs s2d/finetuned maedfer/finetuned] [--reference s2d/finetuned]

Any config value can be overridden with --set section.key=value (YAML syntax), e.g.
    --set s2d.trainable=head --set s2d.lr=1e-3 --set faces.detector=yolo
Add --smoke for SMOKE_TEST mode, --redo to recompute finished outputs.
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["preprocess", "train", "evaluate", "patients", "compare"])
    ap.add_argument("--model", choices=["s2d", "maedfer", "clustering"])
    ap.add_argument("--runs", nargs="+")
    ap.add_argument("--run")
    ap.add_argument("--reference")
    ap.add_argument("--preset", choices=["s2d", "maedfer"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--diagnostic", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    if args.stage != "compare" and not args.model:
        ap.error("--model is required for this stage")

    from common import stages
    from common.env import init
    overrides = {"smoke_test": args.smoke}
    for item in args.set:
        key, val = item.split("=", 1)
        overrides[key.replace(".", "__")] = yaml.safe_load(val)
    if args.preset:
        overrides["patient__preset"] = args.preset
    cfg = init(args.model, **overrides)

    if args.stage == "preprocess":
        df = stages.prepare_split(cfg)
        print(stages.build_face_cache(cfg, df, redo=args.redo))
        if args.diagnostic:
            print(stages.face_diagnostic(cfg, df))
        if cfg.model == "clustering":
            stages.extract_blendshapes(cfg, df, redo=args.redo)
    elif args.stage == "train":
        stages.train(cfg, resume=args.resume, redo=args.redo)
    elif args.stage == "evaluate":
        print(stages.evaluate(cfg, args.runs, redo=args.redo).to_string())
    elif args.stage == "patients":
        print(stages.patient_videos(cfg, run=args.run, redo=args.redo).to_string())
    else:
        table, diffs = stages.compare(cfg, args.runs, args.reference, args.preset)
        print(table.to_string(), "\n", diffs.to_string(index=False))


if __name__ == "__main__":
    main()
