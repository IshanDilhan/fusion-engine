"""
Download the released trained gesture model from GitHub Releases.

Weights are published as GitHub Release assets (not stored in git) so the repo
stays lean and each model version is a separate, downloadable release. By default
this fetches both assets from the *latest* release, so you always get the newest
version. Needs no extra packages and no GitHub auth (public release).

Run:
    python scripts/download_model.py                 # latest version
    python scripts/download_model.py --tag gesture-v2.1   # a specific version
"""
import argparse
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHECKPOINT_DIR

REPO = "KesharaGunathilaka/adaptive-multimodal-hri"
# best_TCN.pth (trained 2026-07-15, the checkpoint that fixed the previously
# dead point/both_hands_up classes) is the deployed checkpoint (84.1% acc /
# 82.9% macro-F1 on held-out test subjects). model_config.json must ship
# alongside it — it pins the exact architecture/labels/window so inference
# can't silently drift from training.
ASSETS = ["best_TCN.pth", "model_config.json"]


def _progress(block_num, block_size, total_size):
    done = block_num * block_size
    if total_size > 0:
        pct = min(100, done * 100 / total_size)
        sys.stdout.write(f"\r  {pct:5.1f}%  ({done/1e6:.1f}/{total_size/1e6:.1f} MB)")
    else:
        sys.stdout.write(f"\r  {done/1e6:.1f} MB")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="Release tag, e.g. gesture-v2.1 (default: latest).")
    ap.add_argument("--out-dir", default=CHECKPOINT_DIR, help="Destination directory.")
    ap.add_argument("--force", action="store_true", help="Re-download even if it exists.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for asset in ASSETS:
        out = os.path.join(args.out_dir, asset)
        if args.tag:
            url = f"https://github.com/{REPO}/releases/download/{args.tag}/{asset}"
        else:
            url = f"https://github.com/{REPO}/releases/latest/download/{asset}"

        if os.path.exists(out) and not args.force:
            print(f"Already present: {out}\n(use --force to re-download)")
            continue

        print(f"Downloading {url}")
        try:
            urllib.request.urlretrieve(url, out, _progress)
        except urllib.error.HTTPError as e:
            print(f"\nDownload failed ({e.code}). Make sure a release exists with asset "
                  f"'{asset}'. Browse: https://github.com/{REPO}/releases")
            sys.exit(1)
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
