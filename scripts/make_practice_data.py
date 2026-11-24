"""Build a small practice set that mimics the demo: a named folder + group photos
that contain both people from the folder and strangers.

Sources (public, already used as test data by the face-recognition community):
  * DeepFace test images   https://github.com/serengil/deepface/tree/master/tests/unit/dataset
  * InsightFace sample     https://github.com/deepinsight/insightface/tree/master/python-package/insightface/data/images

Usage:
    git clone --depth 1 https://github.com/serengil/deepface /tmp/deepface
    git clone --depth 1 https://github.com/deepinsight/insightface /tmp/insightface
    python scripts/make_practice_data.py /tmp/deepface /tmp/insightface practice_data

Output:
    practice_data/people/<Name>.jpg          the folder you open in the app
    practice_data/group_collage.jpg          6 people from the folder + 4 strangers
    practice_data/group_real.jpg             a real group photo; 4 of its 6 people are in the folder
    practice_data/expected.json              who should be found in each group photo

Names are invented placeholders; the photos are of public figures from the
test sets above. Use it only for testing.
"""

import csv
import json
import shutil
import sys
import warnings
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
warnings.filterwarnings("ignore")
from app import imageio  # noqa: E402
from app.engine import FaceEngine  # noqa: E402

NAMES = ["Priya Sharma", "Arjun Mehta", "Ananya Iyer", "Rohan Kapoor", "Kavya Nair",
         "Vikram Rao", "Sneha Reddy", "Aditya Joshi", "Meera Pillai",
         "Rahul Verma", "Divya Menon", "Karan Malhotra", "Isha Gupta"]


def identities(deepface_dir: Path) -> list[list[str]]:
    ds = deepface_dir / "tests/unit/dataset"
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            x = parent[x]
        return x

    files = set()
    for r in csv.DictReader(open(ds / "master.csv")):
        files |= {r["file_x"], r["file_y"]}
        if r["Decision"] == "Yes":
            parent[find(r["file_x"])] = find(r["file_y"])
    groups = defaultdict(list)
    for f in files:
        groups[find(f)].append(f)
    return sorted((sorted(g, key=lambda n: int(n[3:-4])) for g in groups.values() if len(g) >= 2), key=len, reverse=True)


def face_crop(img: Image.Image, bbox, margin=0.9) -> Image.Image:
    x1, y1, x2, y2 = bbox
    cx, cy, half = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * margin
    return img.crop((int(max(0, cx - half)), int(max(0, cy - half * 1.1)),
                     int(min(img.width, cx + half)), int(min(img.height, cy + half * 1.3))))


def main(deepface_dir, insightface_dir, out_dir):
    deepface_dir, insightface_dir, out = Path(deepface_dir), Path(insightface_dir), Path(out_dir)
    ds = deepface_dir / "tests/unit/dataset"
    shutil.rmtree(out, ignore_errors=True)
    (out / "people").mkdir(parents=True)
    engine = FaceEngine()
    ids = identities(deepface_dir)
    expected = {"group_collage.jpg": [], "group_real.jpg": []}

    # 1) folder: first photo of each identity, under an invented name
    for name, files in zip(NAMES, ids):
        shutil.copy(ds / files[0], out / "people" / f"{name}.jpg")

    # 2) real group photo. Its people may overlap the DeepFace identities (it does:
    # one actress is in both sets), so check before deciding who is "unknown".
    ref = {}
    for name, files in zip(NAMES, ids):
        im = Image.open(ds / files[0]).convert("RGB")
        ref[name] = max(engine.faces(imageio.to_bgr(im)), key=lambda f: f.area).embedding
    real = ImageOps.exif_transpose(Image.open(insightface_dir / "python-package/insightface/data/images/t1.jpg")).convert("RGB")
    faces = engine.faces(imageio.to_bgr(real), group=True)
    added = 0
    for f in faces:
        best = max(ref, key=lambda n: float(ref[n] @ f.embedding))
        if float(ref[best] @ f.embedding) >= 0.4:
            expected["group_real.jpg"].append(best)      # already in the folder
        elif added < 4:
            name = NAMES[len(ids) + added]
            face_crop(real, f.bbox).save(out / "people" / f"{name}.jpg", quality=92)
            expected["group_real.jpg"].append(name)
            added += 1
        else:
            expected["group_real.jpg"].append("Unknown")  # deliberately left out
    real.save(out / "group_real.jpg", quality=92)

    # 3) collage: a *different* photo of 6 folder people + 4 strangers
    tiles = []
    for name, files in list(zip(NAMES, ids))[:6]:
        tiles.append((name, Image.open(ds / files[1]).convert("RGB")))
    # Strangers: faces from a selfie whose people are not in the folder. Keep only
    # crops that contain exactly one face (wide crops can catch a neighbour).
    strangers = []
    img = Image.open(ds / "selfie-many-people.jpg").convert("RGB")
    for f in engine.faces(imageio.to_bgr(img), group=True):
        crop = face_crop(img, f.bbox, 0.75)
        if len(engine.faces(imageio.to_bgr(crop.resize((crop.width * 3, crop.height * 3))))) == 1:
            strangers.append(crop)
    tiles += [("Unknown", s) for s in strangers[:4]]
    tiles = [t for t in tiles if len(engine.faces(imageio.to_bgr(t[1]))) == 1]  # one face per tile
    order = [0, 6, 1, 2, 7, 3, 8, 4, 5, 9]  # mix strangers in
    tiles = [tiles[i] for i in order if i < len(tiles)]
    W = 360
    canvas = Image.new("RGB", (W * 5, int(W * 1.25) * 2), (235, 232, 226))
    for i, (name, im) in enumerate(tiles):
        im = ImageOps.fit(im, (W - 16, int(W * 1.25) - 16), centering=(0.5, 0.35))
        canvas.paste(im, ((i % 5) * W + 8, (i // 5) * int(W * 1.25) + 8))
        expected["group_collage.jpg"].append(name)
    canvas.save(out / "group_collage.jpg", quality=92)

    (out / "expected.json").write_text(json.dumps(expected, indent=2))
    print(f"Wrote {len(list((out / 'people').iterdir()))} people and 2 group photos to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
