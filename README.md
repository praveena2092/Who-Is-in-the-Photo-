# Who's in the photo? — Task A

Open a folder of named photos (`Priya Sharma.jpg`, …), upload a group photo, and see a name on every face — or **Unknown** for anyone who isn't in the folder. Results show on the photo and as a list.

Runs **entirely on your laptop**. No internet, cloud or online API at runtime.

<!-- TODO: add a screenshot of the results screen with your own practice photos: docs/screenshot-results.png -->

## Quick start

You need **Python 3.10–3.13** and **Node.js 18+**. Setup needs internet once (to install packages and download the face models). After that the app works fully offline.

**Windows:** double-click `install.bat` once, then `start.bat`.

**macOS / Linux:**
```bash
./install.sh     # one time: venv, Python packages, face models (~280 MB), builds the UI
./start.sh       # opens http://127.0.0.1:8000
```

<details><summary>Manual setup (any OS)</summary>

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt
python scripts/download_models.py                      # saves models to ./models/buffalo_l
cd frontend && npm install && npm run build && cd ..
cd backend && python -m uvicorn app.main:app --port 8000
```
Then open http://127.0.0.1:8000.

For UI development with hot reload, run the backend as above and `npm run dev` in `frontend/` (opens :5173, proxies `/api` to :8000).
</details>

## Using it

1. **Choose folder.** Pick the folder of people (or drag it onto the page). Each photo's file name is that person's name. The app reads every face and shows who it found; photos with no usable face are flagged so you know who can't be recognised.
2. **Upload group photo.** Each face gets a numbered box: **green** = recognised, **amber dashed** = unknown. The list on the right has the same numbers; hover either one to highlight the other.
3. **Optional:** the *Match strictness* slider trades "fewer wrong names" (strict) against "fewer missed people" (relaxed). Results update instantly without re-scanning. *Labelled photo* downloads the annotated image; *Copy* / *CSV* export the list.

File names are cleaned up: `priya_sharma.png`, `Priya Sharma (2).jpg` and `Priya Sharma - Copy.jpg` all become **Priya Sharma**, and several photos of the same person are combined. JPG, PNG, WEBP, BMP, TIFF and iPhone HEIC are supported; phone photos are rotated correctly.

## How it works (short)

| Step | What | Why |
|---|---|---|
| Detect | SCRFD-10GF face detector (InsightFace `buffalo_l`) at 1280 px for group photos | Finds small faces at the back of a group |
| Align + embed | 5-point alignment → ArcFace ResNet-50 → 512-number face "fingerprint" | Best accuracy among free offline models we compared |
| Compare | Cosine similarity of each face against every folder photo | Fast: 10 × 50 comparisons is instant |
| Decide | Hungarian assignment + threshold 0.35 | Each name used at most once; below the threshold → **Unknown** |

Details and trade-offs: [ARCHITECTURE.md](docs/architecture.html). Options considered: [RESEARCH.md](docs/RESEARCH.md). How decisions were made with AI: [AI_COLLAB.md](docs/AI_COLLAB.md).

## Testing

```bash
pytest tests/                                            # logic tests (no models needed)

# Build practice data from public test images, then check the running app end to end
git clone --depth 1 https://github.com/serengil/deepface /tmp/deepface
git clone --depth 1 https://github.com/deepinsight/insightface /tmp/insightface
python scripts/make_practice_data.py /tmp/deepface /tmp/insightface practice_data
python scripts/check_practice.py practice_data           # with the app running

# Prove it works with no network at all (Linux)
cd backend && unshare -rn env PYTHONPATH=. python ../scripts/offline_check.py ../practice_data

# How the 0.35 threshold was chosen
python scripts/calibrate.py /tmp/deepface/tests/unit/dataset/master.csv /tmp/deepface/tests/unit/dataset
```

Current results: both practice group photos (one real, one collage; mix of known and unknown people) are labelled 100 % correctly; on 300 labelled pairs, same-person scores were ≥ 0.54 and different-person scores ≤ 0.23.

## Troubleshooting

- **"Model files not found"** — run `python scripts/download_models.py` once with internet.
- **Folder picker uploads nothing** — use Chrome, Edge or Firefox; or drag the folder onto the page.
- **A person isn't recognised** — check the folder summary for "no usable face"; use a clear, front-facing photo. Try moving *Match strictness* towards *Relaxed*.
- **Port 8000 busy** — change `--port` in `start.sh` / `start.bat`.
- **GPU** — optional: `pip install onnxruntime-gpu` and set `FACE_USE_GPU=1`. CPU is fast enough (≈ 2–4 s per group photo).

## Licence note

The code here is MIT. The InsightFace `buffalo_l` model weights are licensed for **non-commercial research use** only; a commercial deployment would need a licence from InsightFace or a swap to a commercially licensed model (see RESEARCH.md).
