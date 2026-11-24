# Research

What I looked into before (and while) building: the problem, existing tools and models, papers, data sources, what I compared, what I picked and why, and what I rejected.

## 1. Understanding the problem

This is **open-set face identification**:

- **One photo per person.** The folder has exactly one example of each of the 50 people, so there is nothing to "train". The model must recognise someone from a single photo (one-shot).
- **Group photo is harder than the portraits.** Faces are smaller, angled, and in different light.
- **Some faces belong to nobody in the folder.** A model always finds *a* closest person, so the system needs a rule for saying **Unknown**. Wrongly naming a stranger is the worst mistake, because it's confidently wrong.
- **A person can appear only once in a photo**, a constraint worth using.
- **Offline, ordinary laptop, live demo on unseen data.** Easy install and reliability matter as much as accuracy.

The standard pipeline, used by every serious system I looked at:

**detect** faces and 5 landmarks → **align** each face to a standard pose → **embed** it into a vector where the same person's faces lie close together → **compare** vectors with cosine similarity → **decide** (name or Unknown).

## 2. Recognition models compared

| Option | What it is | Accuracy | Offline install | Licence | Verdict |
|---|---|---|---|---|---|
| **InsightFace `buffalo_l`** | SCRFD-10GF detector + ArcFace ResNet-50 trained on WebFace600K (ONNX) | LFW 99.83, IJB-C 97.25, MR-ALL 91.25 | `pip install insightface` (pure-Python wheel since 2.x) + onnxruntime; we use 2 of its 5 files (~180 MB) | Code MIT; **model weights non-commercial research only** | ✅ **Picked** |
| InsightFace `buffalo_s` | SCRFD-500MF + MobileFaceNet | LFW 99.70, IJB-C 95.02, **MR-ALL 71.87** | Same, smaller (159 MB pack) | Same | ❌ Similar on easy LFW, much weaker on hard benchmarks (MR-ALL 72 vs 91) |
| InsightFace `antelopev2` | SCRFD-10GF + ResNet-100 (Glint360K) | Higher still | 407 MB pack; **not auto-downloadable** (manual Google Drive link) | Same | ❌ Bigger and slower on CPU, harder to install; `buffalo_l` is already more than accurate enough on our tests |
| InsightFace 2.x `raccoon_*` / `cheetah_*` | New packs for privacy blurring (Raccoon) and body + face person tracking (Cheetah) | Cheetah's face models are the same files as `buffalo_s` / `buffalo_l` | Larger, more parts | Same | ❌ Built for other tasks (blurring, video/body re-ID); nothing extra for still photos |
| OpenCV Zoo **YuNet + SFace** | Tiny detector + MobileFaceNet | SFace 99.40 on LFW; YuNet WIDER Face hard AP 0.75 | Only `opencv-python` + two small files | Apache-2.0 | 🔁 Fallback: commercially usable and light, but weaker on small faces in group shots |
| **face_recognition** (dlib) | dlib detector + ResNet embedding | 99.38 on LFW; weak on small / side faces | dlib often needs a C++ compiler on Windows | MIT | ❌ Older, install risk on the demo laptop |
| **DeepFace** | Wrapper around many models (ArcFace, FaceNet, VGG-Face…) | Depends on backend | Pulls in TensorFlow | MIT (models vary) | ❌ Too heavy for the app; I used its labelled test images instead |
| **FaceNet-PyTorch** | MTCNN + InceptionResnetV1 (VGGFace2) | Good, below ArcFace | PyTorch (~800 MB) | MIT | ❌ Big install for lower accuracy |
| **AuraFace** | ArcFace-style model, commercial licence | Below `buffalo_l` in a third-party test (recall@1 0.927 vs 0.976) | Same tooling as InsightFace | Apache-2.0 | 📝 The swap to make if this were a commercial product |
| Cloud APIs (AWS Rekognition, Azure Face, Gemini) | Hosted services | High | ❌ Need internet | — | ❌ Not allowed by the rules |

**Why `buffalo_l`:** the demo uses unseen real photos, so accuracy on *hard* faces (small, angled, different lighting) is what matters. On easy LFW photos `buffalo_s` and `buffalo_l` look almost the same (99.70 vs 99.83), but on the harder MR-ALL benchmark the gap is large (71.87 vs 91.25). It still runs fast on a laptop CPU (about 3 s per group photo in my tests). It's also a proven choice: Immich, a popular open-source photo app, ships the same models for face recognition.

**Caveat I noted:** `buffalo_l` is weakest on the East Asian subset (74.96, vs 93.16 South Asian and 94.70 Caucasian). Worth knowing if the demo photos are mostly East Asian faces.

**Only two of the five files are loaded:** the detector and the recogniser. The age/gender and landmark models aren't needed. That means faster startup and less memory, and I can use a bigger detector input for group photos than for portraits.

## 3. Deciding "Unknown"

| Option | Problem / benefit | Verdict |
|---|---|---|
| Nearest match for every face | Every stranger gets someone's name; two faces can get the same name | ❌ |
| Nearest match + similarity threshold | Fixes strangers; still allows "two Priyas" | Partial |
| **Threshold + Hungarian assignment** (`scipy.optimize.linear_sum_assignment`) | Each name goes to at most one face, the best fit overall; anything below the threshold is **Unknown** | ✅ **Picked** |
| Train a classifier (SVM / kNN) on the folder | One photo per person is too little to train; no natural "unknown" | ❌ |
| Margin check (best match must beat 2nd best by 0.05–0.08) | Suggested by AI. My tests already had 0 strangers named, and the rule could reject real matches between lookalikes in the folder | ❌ Not added |
| "Low quality" label for tiny or blurry faces | Could hide real people at the back of a group photo | ❌ Not added; only specks under 20 px are ignored |
| Grouping unknowns as Unknown_1, Unknown_2… | Only useful across several photos; the demo uploads one | ❌ Not added |

### How the threshold was chosen (measured, not guessed)

**Step 1, labelled pairs** (DeepFace test set, `scripts/calibrate.py`, 300 pairs):

| | lowest / highest | 5th / 95th percentile | median |
|---|---|---|---|
| Same person (38 pairs) | lowest **0.54** | p5 0.68 | 0.77 |
| Different people (262 pairs) | highest **0.23** | p95 0.12 | 0.01 |

**Step 2, a real group photo:** one actress matched between a decades-old TV still and a recent photo at only **0.40**, so real group photos score lower than clean portraits.

**Step 3, a demo-sized test** (`scripts/make_lfw_test.py` → 50-person folder from LFW, 5 group photos of 7 known people + 3 strangers each; `scripts/check_practice.py`):

| Result | Count |
|---|---|
| People from the folder named correctly | **35 / 35** |
| Strangers marked Unknown | **15 / 15** |
| Wrong name on a face | **0** |
| Known people's scores | 0.50 – 0.83 |
| Highest stranger score | 0.21 |

So the default is **0.35**: below every genuine match seen (including the hard 0.40 case) and well above every stranger. The UI's *Match strictness* slider lets a user adjust it for unusual photos.

**Lesson from the first LFW run:** it reported "0/5 photos correct" with 17 "missed". The details showed every known person was named and every stranger rejected. The "misses" were **extra background faces**: LFW images are cropped from news photos and often include a second person at the edge, which the app correctly marked Unknown. The test, not the app, was wrong, so I fixed the test builder (crop each tile to its centre face) and the checker (count background faces separately).

## 4. App shape

| Option | Verdict |
|---|---|
| Command-line tool | ❌ The users are office staff, and the UI is scored |
| Streamlit / Gradio | ❌ Quick, but generic-looking; hard to make interactive labels on the photo |
| Electron / Tauri desktop app | ❌ Packaging takes too long for the time available |
| **FastAPI backend + React frontend on localhost** | ✅ Full control of the UI; Python for the ML; one double-click start script; nothing leaves `127.0.0.1` |

## 5. Data sources

No data was provided.

| Source | What it gave me | Used for |
|---|---|---|
| **LFW** (Labeled Faces in the Wild) via the Kaggle mirror | 13,233 photos of 5,749 named people, one folder per person | The 50-person demo-style test (`make_lfw_test.py`) |
| **DeepFace unit-test images** | ~60 celebrity photos + `master.csv` of same/different pairs | Threshold calibration, first practice set |
| **InsightFace sample `t1.jpg`** | A real multi-person group photo | Testing on a real group shot |
| Celebrity Together (Oxford VGG) | Full images with several labelled celebrities (194k images, avg 2.8 faces) | 📝 Ideal match for the task, but large and hard to get in the time; not used |
| PIPA (People In Photo Albums) | ~60k instances of ~2,000 people from Flickr albums; only about half are frontal faces | 📝 Realistic but hard and no longer distributed by Berkeley; not used |
| Generated faces (Gemini) | Allowed by the brief for practice data | 📝 Keeping the same face across separate images is unreliable; considered as an extra test |

**Lessons:**
- The DeepFace and InsightFace samples **share a person**. The model spotted it (0.80 similarity) before I did, so the practice-data builder now checks for overlap instead of assuming the sources are separate.
- LFW has no real group photos, so its group photos are collages. They test recognition and Unknown handling well, but not small or angled faces, so I also tested with a real group photo.
- Test photos are real people's photos, so they're kept out of the repo (`.gitignore`); the scripts rebuild them.

## 6. Links

**Models and tools**
- InsightFace (code, model zoo, benchmarks): https://github.com/deepinsight/insightface
- InsightFace model guide (pack table, accuracy, licence): https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md
- InsightFace commercial licensing: https://www.insightface.ai/solutions/face-recognition-licensing
- Immich ships `buffalo_l`: https://huggingface.co/immich-app/buffalo_l
- OpenCV Zoo SFace: https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
- OpenCV Zoo YuNet: https://huggingface.co/opencv/face_detection_yunet
- face_recognition (dlib): https://github.com/ageitgey/face_recognition
- DeepFace: https://github.com/serengil/deepface
- AuraFace comparison: https://mixpeek.com/models/auraface-v1
- SciPy Hungarian algorithm: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html

**Papers**
- ArcFace: Deng et al., *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*, CVPR 2019. https://arxiv.org/abs/1801.07698
- SCRFD: Guo et al., *Sample and Computation Redistribution for Efficient Face Detection*, 2021. https://arxiv.org/abs/2105.04714
- Celebrity Together dataset: Zhong, Arandjelović, Zisserman, *Compact Deep Aggregation for Set Retrieval*. https://arxiv.org/abs/2003.11794
- PIPA dataset: Zhang et al., *Beyond Frontal Faces: Improving Person Recognition Using Multiple Cues*, CVPR 2015. https://arxiv.org/abs/1501.05703

**Data**
- LFW (official): http://vis-www.cs.umass.edu/lfw/
- LFW Kaggle mirror: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset
- DeepFace test images: https://github.com/serengil/deepface/tree/master/tests/unit/dataset
- PIPA (Max Planck Institute): https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/people-detection-pose-estimation-and-tracking/person-recognition-in-personal-photo-collections
