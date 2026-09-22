"""Download the things the repo deliberately does not carry.

Model weights are ~45 MB and freely downloadable, so they stay out of git. Run
this once after cloning -- and run it *before the fair*, because the venue wifi
will fail you.

    python tools/fetch_assets.py
"""
from __future__ import annotations

import argparse
import os
import shutil
import ssl
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ULTRALYTICS = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
NAP = ("https://raw.githubusercontent.com/Bimo99B9/"
       "NaturalisticAdversarialPatches/main/adversarialYolo")

MODELS = {
    "yolov8n.pt": f"{ULTRALYTICS}/yolov8n.pt",     # the demo detector
    "yolov10n.pt": f"{ULTRALYTICS}/yolov10n.pt",   # expert mode (E)
}
EXTRA_MODELS = {                                   # only for tools/validate_patch.py
    "yolov8s.pt": f"{ULTRALYTICS}/yolov8s.pt",
    "yolov5nu.pt": f"{ULTRALYTICS}/yolov5nu.pt",
    "yolov9t.pt": f"{ULTRALYTICS}/yolov9t.pt",
}
TEST_IMAGES = {
    "data_person.jpg": f"{NAP}/data/person.jpg",
    "sample_person.jpg": f"{NAP}/sample/person.jpg",
    "test_img_crop001024.png": f"{NAP}/test/img/crop001024.png",
}


def _ssl_context() -> ssl.SSLContext:
    """python.org builds on macOS ship without CA certificates installed, so a
    plain urlopen dies with CERTIFICATE_VERIFY_FAILED. certifi comes along with
    ultralytics, so use its bundle when it is there."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def fetch(url: str, dest: str) -> bool:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  have  {os.path.relpath(dest, ROOT)}")
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    try:
        with urllib.request.urlopen(url, timeout=120,
                                    context=_ssl_context()) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 16):
                f.write(chunk)
        os.replace(tmp, dest)
    except Exception as exc:
        if os.path.exists(tmp):
            os.remove(tmp)
        # Last resort: curl has the system trust store and is always on macOS.
        if shutil.which("curl") and subprocess.run(
                ["curl", "-sSL", "--fail", "--max-time", "180", "-o", dest, url],
                capture_output=True).returncode == 0:
            print(f"  got   {os.path.relpath(dest, ROOT)} (via curl)")
            return True
        if os.path.exists(dest):
            os.remove(dest)
        print(f"  FAIL  {os.path.relpath(dest, ROOT)}: {exc}")
        return False
    print(f"  got   {os.path.relpath(dest, ROOT)} "
          f"({os.path.getsize(dest) // 1024} KB)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-models", action="store_true",
                    help="also fetch the extra detectors validate_patch.py compares")
    args = ap.parse_args()

    ok = True
    print("models:")
    wanted = dict(MODELS)
    if args.all_models:
        wanted.update(EXTRA_MODELS)
    for name, url in wanted.items():
        ok &= fetch(url, os.path.join(ROOT, "assets/models", name))

    print("test images:")
    for name, url in TEST_IMAGES.items():
        ok &= fetch(url, os.path.join(ROOT, "assets/testimg", name))

    print("\nDone." if ok else "\nSome downloads failed -- rerun when online.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
