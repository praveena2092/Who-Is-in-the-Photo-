"""One-time download of the two InsightFace buffalo_l models (needs internet once).

After this the app runs fully offline: models are read from ./models/buffalo_l.
"""

import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip"
NEEDED = {"det_10g.onnx", "w600k_r50.onnx"}
DEST = Path(__file__).resolve().parents[1] / "models" / "buffalo_l"


def main() -> None:
    if all((DEST / n).exists() for n in NEEDED):
        print(f"Models already present in {DEST}")
        return
    DEST.mkdir(parents=True, exist_ok=True)
    zip_path = DEST.parent / "buffalo_l.zip"
    print(f"Downloading {URL} (~280 MB, one time only)…")

    def progress(blocks, block_size, total):
        if total > 0:
            done = min(100, blocks * block_size * 100 // total)
            sys.stdout.write(f"\r  {done}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(URL, zip_path, progress)
    print()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            name = Path(member).name
            if name in NEEDED:
                with zf.open(member) as src, open(DEST / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    zip_path.unlink()
    for n in sorted(NEEDED):
        h = hashlib.sha256((DEST / n).read_bytes()).hexdigest()[:16]
        print(f"  {n}  sha256:{h}…")
    print(f"Done. Models saved to {DEST}")


if __name__ == "__main__":
    main()
