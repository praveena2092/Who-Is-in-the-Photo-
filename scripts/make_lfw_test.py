"""Build a demo-style test set from LFW (Labeled Faces in the Wild).

Mimics the real demo: a folder of N named people (one photo each) and group
photos of ~10 faces where most people are in the folder and some are strangers.
Each person's photo in a group is a DIFFERENT photo from the one in the folder,
so the test is honest.

1. Download LFW from Kaggle and unzip it anywhere:
       https://www.kaggle.com/datasets/jessicali9530/lfw-dataset
2. Run:
       python scripts/make_lfw_test.py PATH_TO_UNZIPPED_LFW lfw_test
   Options:
       --people 50       people in the folder (default 50)
       --groups 5        number of group photos (default 5)
       --per-group 10    faces per group photo (default 10)
       --strangers 3     faces per group NOT in the folder (default 3)
       --seed 7          change to get a different random set
3. Test:
       python scripts/check_practice.py lfw_test     (with the app running)
   or in the browser: choose folder  lfw_test/people , upload  lfw_test/group_1.jpg ...

Output:
    lfw_test/people/<First Last>.jpg     the folder to open in the app
    lfw_test/group_<n>.jpg               group photos (collages of real LFW photos)
    lfw_test/expected.json               who is in each group photo ("Unknown" = stranger)

Note: LFW has no real multi-person photos, so the group photos are collages.
They test recognition and "unknown" handling well, but not small or angled faces;
also test with at least one real group photo before the demo.
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageOps

IMG_EXT = {".jpg", ".jpeg", ".png"}


def find_people_root(root: Path) -> Path:
    """Kaggle zips nest folders (lfw-deepfunneled/lfw-deepfunneled/<Name>/...).
    Find the directory whose subfolders hold the person images."""
    best, best_count = None, 0
    for d in [root, *[p for p in root.rglob("*") if p.is_dir()]]:
        subs = [s for s in d.iterdir() if s.is_dir()]
        count = sum(1 for s in subs[:200] if any(f.suffix.lower() in IMG_EXT for f in s.iterdir()))
        if count > best_count:
            best, best_count = d, count
        if best_count > 1000:
            break
    if not best or best_count < 10:
        sys.exit(f"Couldn't find LFW person folders under {root}. Point this at the unzipped Kaggle download.")
    return best


def pretty(folder_name: str) -> str:
    return folder_name.replace("_", " ").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lfw_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--people", type=int, default=50)
    ap.add_argument("--groups", type=int, default=5)
    ap.add_argument("--per-group", type=int, default=10)
    ap.add_argument("--strangers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rnd = random.Random(a.seed)
    root = find_people_root(Path(a.lfw_dir))
    persons = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = sorted(f for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
        if imgs:
            persons[d.name] = imgs

    # Folder people need >= 2 photos: one for the folder, another for group photos.
    multi = [n for n, imgs in persons.items() if len(imgs) >= 2]
    if len(multi) < a.people:
        sys.exit(f"Only {len(multi)} people have 2+ photos; lower --people.")
    folder_people = rnd.sample(multi, a.people)
    strangers_pool = [n for n in persons if n not in set(folder_people)]
    if len(strangers_pool) < a.strangers:
        sys.exit("Not enough people outside the folder to use as strangers; lower --people or --strangers.")
    known_per_group = a.per_group - a.strangers
    if known_per_group < 0 or known_per_group > a.people:
        sys.exit("--per-group / --strangers don't fit the number of people.")

    out = Path(a.out_dir)
    shutil.rmtree(out, ignore_errors=True)
    (out / "people").mkdir(parents=True)

    for n in folder_people:
        src = persons[n][0]
        shutil.copy(src, out / "people" / f"{pretty(n)}{src.suffix.lower()}")

    expected = {}
    used_strangers = set()
    cols = 5
    tile = 250  # LFW images are 250x250
    for g in range(1, a.groups + 1):
        known = rnd.sample(folder_people, known_per_group)
        pool = [s for s in strangers_pool if s not in used_strangers]
        if len(pool) < a.strangers:  # ran out of fresh strangers: allow repeats across photos
            pool = strangers_pool
        strangers = rnd.sample(pool, a.strangers)
        used_strangers.update(strangers)

        faces = [(pretty(n), rnd.choice(persons[n][1:])) for n in known]   # never the folder photo
        faces += [("Unknown", rnd.choice(persons[s])) for s in strangers]
        rnd.shuffle(faces)

        rows = (len(faces) + cols - 1) // cols
        gap = 12
        canvas = Image.new("RGB", (cols * (tile + gap) + gap, rows * (tile + gap) + gap), (236, 233, 227))
        for i, (_, path) in enumerate(faces):
            im = ImageOps.fit(Image.open(path).convert("RGB"), (tile, tile))
            canvas.paste(im, (gap + (i % cols) * (tile + gap), gap + (i // cols) * (tile + gap)))
        name = f"group_{g}.jpg"
        canvas.save(out / name, quality=92)
        expected[name] = [label for label, _ in faces]

    (out / "expected.json").write_text(json.dumps(expected, indent=2))
    print(f"LFW people folders found in: {root}")
    print(f"Wrote {a.people} people to {out / 'people'}")
    print(f"Wrote {a.groups} group photos ({known_per_group} known + {a.strangers} strangers each) and expected.json")
    print(f"\nNext: start the app, then  python scripts/check_practice.py {out}")


if __name__ == "__main__":
    main()
