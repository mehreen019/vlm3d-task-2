#!/usr/bin/env python3
"""
Run 5-fold CV over pre-made fold CSVs and call your existing run_task2.py.
Works on Windows/macOS/Linux (no .bat/.sh needed).

Prereq: you already generated the folds, e.g. in ct_rate_2d/cv/valid_k5/
        train_slices_fold0.csv ... train_slices_fold4.csv
        valid_slices_fold0.csv ... valid_slices_fold4.csv
"""
import argparse, os, sys, shutil, subprocess, re, statistics as stats
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-dir", default="ct_rate_2d/cv/valid_k5", help="Where the per-fold CSVs live")
    ap.add_argument("--slice-dir", default="ct_rate_2d", help="Your slice root (expects <slice-dir>/splits/...)")
    ap.add_argument("--results-dir", default="results", help="Where to write per-fold logs")
    ap.add_argument("--folds", type=int, default=5)

    # training args to pass through to run_task2.py
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--loss-type", default="focal")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--gpu-device", type=int, default=None)
    ap.add_argument("--use-attention", choices=["none", "se", "cbam"], default="none")
    ap.add_argument("--use-multiscale", action="store_true")
    ap.add_argument("--cutmix-prob", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--early-stopping-patience", type=int, default=10)
    ap.add_argument("--progressive-unfreeze", action="store_true")
    args = ap.parse_args()

    cv_dir     = Path(args.cv_dir)
    slice_dir  = Path(args.slice_dir)
    splits_dir = slice_dir / "splits"
    results    = Path(args.results_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    # optional: back up originals if present
    orig_tr = splits_dir / "train_slices.csv"
    orig_va = splits_dir / "valid_slices.csv"
    backup_tr = splits_dir / "train_slices.csv.orig"
    backup_va = splits_dir / "valid_slices.csv.orig"
    if orig_tr.exists(): shutil.copyfile(orig_tr, backup_tr)
    if orig_va.exists(): shutil.copyfile(orig_va, backup_va)

    logs = []
    try:
        for f in range(args.folds):
            print(f"======== FOLD {f} ========")
            src_tr = cv_dir / f"train_slices_fold{f}.csv"
            src_va = cv_dir / f"valid_slices_fold{f}.csv"
            assert src_tr.exists() and src_va.exists(), f"Missing fold files for fold {f}"

            # Copy into filenames the runner expects
            shutil.copyfile(src_tr, splits_dir / "train_slices.csv")
            shutil.copyfile(src_va, splits_dir / "valid_slices.csv")

            # Build command to call your existing runner
            cmd = [sys.executable, "run_task2.py",
                   "--mode", "both",
                   "--model", args.model,
                   "--loss-type", args.loss_type,
                   "--epochs", str(args.epochs),
                   "--batch-size", str(args.batch_size),
                   "--early-stopping-patience", str(args.early_stopping_patience),
                   "--use-attention", args.use_attention,
                   "--cutmix-prob", str(args.cutmix_prob)]
            if args.freeze_backbone:
                cmd.append("--freeze-backbone")
            if args.use_multiscale:
                cmd.append("--use-multiscale")
            if args.progressive_unfreeze:
                cmd.append("--progressive-unfreeze")
            if args.gpu_device is not None:
                cmd += ["--gpu-device", str(args.gpu_device)]

            # Per-fold log file
            log_path = results / f"cv_valid_fold{f}.log"
            env = os.environ.copy()
            env["FOLD_INDEX"] = str(f)
            env["CHECKPOINT_DIR"] = str(Path("checkpoints") / f"fold{f}")
            env["LOG_DIR"]        = str(Path("logs")        / f"fold{f}")  # handy if you want per-fold dirs in run_task2.py
            with open(log_path, "w", encoding="utf-8") as lf:
                subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, check=True)
            logs.append(log_path)

        # Summarize metrics from logs
        rx = {
            "auroc_macro": re.compile(r"AUROC \(macro\):\s*([0-9.]+)"),
            "f1_macro":    re.compile(r"F1 \(macro\):\s*([0-9.]+)"),
            "accuracy":    re.compile(r"Accuracy:\s*([0-9.]+)")
        }
        summary = {}
        for name, pat in rx.items():
            vals = []
            for p in logs:
                txt = Path(p).read_text(errors="ignore")
                found = pat.findall(txt)
                if found:
                    vals.append(float(found[-1]))  # last printed value
            if vals:
                summary[name] = (round(stats.mean(vals), 4), round(stats.pstdev(vals), 4), len(vals))
        print("\nCV summary (mean, std, n):")
        for k, (m, s, n) in summary.items():
            print(f"  {k}: mean={m:.4f}, std={s:.4f}, n={n}")

    finally:
        # Restore original split files if we backed them up
        if backup_tr.exists(): shutil.copyfile(backup_tr, orig_tr); backup_tr.unlink()
        if backup_va.exists(): shutil.copyfile(backup_va, orig_va); backup_va.unlink()

if __name__ == "__main__":
    main()
