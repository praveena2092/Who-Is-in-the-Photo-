"""Measure same-person vs different-person similarity on labelled pairs.

Usage:
    python scripts/calibrate.py PAIRS.csv IMAGE_DIR

PAIRS.csv has columns file_x,file_y,Decision (Yes = same person), like the
DeepFace test set (https://github.com/serengil/deepface/tree/master/tests/unit/dataset).
Prints score distributions and false-accept / false-reject rates per threshold,
which is how DEFAULT_THRESHOLD in backend/app/matching.py was chosen.
"""

import csv
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
warnings.filterwarnings("ignore")

from app import imageio  # noqa: E402
from app.engine import FaceEngine  # noqa: E402


def main(pairs_csv: str, image_dir: str) -> None:
    engine, cache = FaceEngine(), {}

    def emb(name):
        if name not in cache:
            img = imageio.decode((Path(image_dir) / name).read_bytes(), 1600)
            faces = engine.faces(imageio.to_bgr(img))
            cache[name] = max(faces, key=lambda f: f.area).embedding if faces else None
        return cache[name]

    same, diff = [], []
    for row in csv.DictReader(open(pairs_csv)):
        a, b = emb(row["file_x"]), emb(row["file_y"])
        if a is None or b is None:
            continue
        (same if row["Decision"].strip().lower() == "yes" else diff).append(float(a @ b))
    same, diff = np.array(same), np.array(diff)
    print(f"same person      n={len(same):4d}  min={same.min():.3f}  p5={np.percentile(same, 5):.3f}  median={np.median(same):.3f}")
    print(f"different people n={len(diff):4d}  max={diff.max():.3f}  p95={np.percentile(diff, 95):.3f}  median={np.median(diff):.3f}")
    print("\nthreshold  false-reject  false-accept")
    for t in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        print(f"  {t:.2f}       {(same < t).mean():6.1%}       {(diff >= t).mean():6.1%}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
