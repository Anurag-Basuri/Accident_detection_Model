import os
import shutil
import argparse

TARGET_NON = "Non_Accident"


def normalize_split(root: str):
    for split in ("train", "val", "test"):
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        # Handle Non Accident variations
        for cand in ("Non Accident", "Non-Accident", "non-accident", "Non_Accident"):
            cand_dir = os.path.join(split_dir, cand)
            if os.path.isdir(cand_dir):
                target_dir = os.path.join(split_dir, TARGET_NON)
                os.makedirs(target_dir, exist_ok=True)
                if cand != TARGET_NON:
                    for name in os.listdir(cand_dir):
                        src = os.path.join(cand_dir, name)
                        if os.path.isfile(src):
                            shutil.move(src, os.path.join(target_dir, name))
                    shutil.rmtree(cand_dir, ignore_errors=True)
        # Remove stray .lnk files
        for cls in ("Accident", TARGET_NON):
            cls_dir = os.path.join(split_dir, cls)
            if os.path.isdir(cls_dir):
                for name in os.listdir(cls_dir):
                    if name.lower().endswith(".lnk"):
                        try:
                            os.remove(os.path.join(cls_dir, name))
                        except Exception:
                            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root containing train/val/test")
    args = ap.parse_args()
    normalize_split(args.root)
    print(f"Normalized dataset at {args.root}")


if __name__ == "__main__":
    main()
