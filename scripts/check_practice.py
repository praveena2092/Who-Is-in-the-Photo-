"""End-to-end check against the running app, using practice data.

    python scripts/check_practice.py practice_data [http://127.0.0.1:8000]

Uploads practice_data/people/ as the folder, then each group photo listed in
expected.json, and compares the names (left to right) with what's expected.
"""

import json
import sys
import time
from pathlib import Path

import requests


def main(data_dir: str, base: str = "http://127.0.0.1:8000") -> int:
    data = Path(data_dir)
    requests.delete(f"{base}/api/gallery").raise_for_status()
    people = sorted((data / "people").iterdir())
    t = time.time()
    for i in range(0, len(people), 6):
        files = [("files", (p.name, p.read_bytes())) for p in people[i:i + 6]]
        r = requests.post(f"{base}/api/gallery/photos", files=files)
        r.raise_for_status()
        for rep in r.json()["reports"]:
            if rep["status"] != "ok":
                print(f"  folder: {rep['file']}: {rep['status']} - {rep['message']}")
    print(f"folder: {len(people)} files in {time.time() - t:.1f}s")

    expected = json.loads((data / "expected.json").read_text())
    failures = 0
    tot = {"faces": 0, "right_name": 0, "wrong_name": 0, "missed": 0, "stranger_named": 0, "stranger_ok": 0}
    for photo, want in expected.items():
        t = time.time()
        r = requests.post(f"{base}/api/photos", files={"file": (photo, (data / photo).read_bytes())})
        r.raise_for_status()
        res = r.json()
        got = [f["name"] for f in res["faces"]]
        ok = sorted(got) == sorted(want)  # same people found (order can differ in multi-row photos)
        failures += not ok
        # Per-face tally (order-free): compare as multisets.
        remaining = list(want)
        for name in got:
            tot["faces"] += 1
            if name in remaining:
                remaining.remove(name)
                tot["stranger_ok" if name == "Unknown" else "right_name"] += 1
            elif name == "Unknown":
                tot["missed"] += 1
            elif "Unknown" in remaining:
                remaining.remove("Unknown")
                tot["stranger_named"] += 1
            else:
                tot["wrong_name"] += 1
        print(f"\n{photo}: {'PASS' if ok else 'FAIL'}  ({time.time() - t:.1f}s, {res['summary']})")
        for f in res["faces"]:
            mark = "  " if f["name"] in want else "✗ "
            cand = ", ".join(f"{c['name']} {c['score']:.2f}" for c in f["candidates"][:2])
            print(f"  {mark}{f['number']:>2}. {f['name']:<16} score={f['score']:.2f}  top: {cand}")
    print(f"\nTOTAL  {len(expected) - failures}/{len(expected)} photos fully correct")
    print(f"  known people named correctly : {tot['right_name']}")
    print(f"  strangers marked Unknown     : {tot['stranger_ok']}")
    print(f"  known person missed (Unknown): {tot['missed']}   -> move strictness towards Relaxed")
    print(f"  stranger given a name        : {tot['stranger_named']}   -> move strictness towards Strict")
    print(f"  wrong person's name          : {tot['wrong_name']}")
    return failures


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
