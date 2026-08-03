"""
Download the released trained emotion model from GitHub Releases.

Weights are published as GitHub Release assets (not stored in git) so the repo
stays lean and each model version is a separate, downloadable release. By default
this fetches the asset from the *latest* release, so you always get the newest
version. Needs no extra packages and no GitHub auth (public release).

Run:
    python scripts/download_model.py                 # latest version
    python scripts/download_model.py --tag emotion-v3.0   # a specific version
    python scripts/download_model.py --url <asset-url>    # explicit override
"""
import argparse
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHECKPOINT_DIR

REPO = "KesharaGunathilaka/adaptive-multimodal-hri"
# finetuned_MobileNetV2.pth (RAF-DB + real-world fine-tune) is the deployed
# checkpoint (92.5% acc / 90.1% macro-F1 on held-out test subjects). Earlier
# releases (emotion-v1.0/v2.0) shipped best_MobileNetV2.pth, the RAF-DB-only
# baseline (58.8%/38.9% on the same test subjects) — do not revert this.
ASSET = "finetuned_MobileNetV2.pth"
# GitHub serves the latest release's asset at a stable URL:
#   https://github.com/<owner>/<repo>/releases/latest/download/<asset>
# and a specific version at:
#   https://github.com/<owner>/<repo>/releases/download/<tag>/<asset>
LATEST_URL = f"https://github.com/{REPO}/releases/latest/download/{ASSET}"


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
    ap.add_argument("--tag", default=None, help="Release tag, e.g. emotion-v1.0 (default: latest).")
    ap.add_argument("--url", default=None, help="Explicit asset URL (overrides --tag).")
    ap.add_argument("--out", default=os.path.join(CHECKPOINT_DIR, ASSET), help="Destination path.")
    ap.add_argument("--force", action="store_true", help="Re-download even if it exists.")
    args = ap.parse_args()

    if args.url:
        url = args.url
    elif args.tag:
        url = f"https://github.com/{REPO}/releases/download/{args.tag}/{ASSET}"
    else:
        url = LATEST_URL

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out) and not args.force:
        print(f"Already present: {args.out}\n(use --force to re-download)")
        return

    print(f"Downloading {url}")
    try:
        urllib.request.urlretrieve(url, args.out, _progress)
    except urllib.error.HTTPError as e:
        print(f"\nDownload failed ({e.code}). Make sure a release exists with asset "
              f"'{ASSET}'. Browse: https://github.com/{REPO}/releases")
        sys.exit(1)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
