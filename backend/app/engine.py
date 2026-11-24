"""Face detection + embedding, fully offline.

Uses two ONNX models from InsightFace's buffalo_l pack:
  * det_10g.onnx   - SCRFD-10GF face detector (boxes + 5 landmarks)
  * w600k_r50.onnx - ArcFace ResNet-50 (WebFace600K), 512-d embeddings

We load the two models directly instead of using FaceAnalysis so we only pay
for what we use (no age/gender/3D-landmark models) and so the detector input
size can differ between portraits (small, fast) and group photos (larger, to
catch small faces at the back of the group).
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from insightface.model_zoo import get_model

MODEL_DIR = Path(os.environ.get("FACE_MODEL_DIR", Path(__file__).resolve().parents[2] / "models" / "buffalo_l"))
DET_FILE = "det_10g.onnx"
REC_FILE = "w600k_r50.onnx"

# Detector input sizes. Portraits: one big face, 640 is plenty.
# Group photos: ~10 people, faces can be small, so give the detector more pixels.
PORTRAIT_DET_SIZE = (640, 640)
GROUP_DET_SIZE = (1280, 1280)
DET_THRESHOLD = 0.5
MIN_FACE_PX = 20  # ignore specks (posters on the wall, reflections)


@dataclass
class DetectedFace:
    bbox: np.ndarray        # x1, y1, x2, y2 in pixels
    kps: np.ndarray         # 5 x 2 landmarks
    det_score: float
    embedding: np.ndarray   # L2-normalised, shape (512,)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return float(max(0, x2 - x1) * max(0, y2 - y1))


class ModelsMissingError(RuntimeError):
    pass


class FaceEngine:
    def __init__(self, model_dir: Path = MODEL_DIR):
        det_path = model_dir / DET_FILE
        rec_path = model_dir / REC_FILE
        missing = [str(p) for p in (det_path, rec_path) if not p.exists()]
        if missing:
            raise ModelsMissingError(
                "Model files not found: " + ", ".join(missing)
                + ". Run `python scripts/download_models.py` once (needs internet), then start again."
            )
        providers = _pick_providers()
        self.det = get_model(str(det_path), providers=providers)
        self.rec = get_model(str(rec_path), providers=providers)
        self.det.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1,
                         input_size=PORTRAIT_DET_SIZE, det_thresh=DET_THRESHOLD)
        self.rec.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1)
        self.providers = providers
        # onnxruntime sessions are thread-safe for run(), but SCRFD keeps some
        # per-call state on the object, so serialise detector calls.
        self._lock = threading.Lock()

    def faces(self, img_bgr: np.ndarray, group: bool = False) -> list[DetectedFace]:
        size = GROUP_DET_SIZE if group else PORTRAIT_DET_SIZE
        # Don't upscale tiny images past their own size: wastes time, adds false positives.
        h, w = img_bgr.shape[:2]
        if max(h, w) < size[0]:
            side = max(320, int(np.ceil(max(h, w) / 32) * 32))
            size = (side, side)
        with self._lock:
            bboxes, kpss = self.det.detect(img_bgr, input_size=size)
            out: list[DetectedFace] = []
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i, :4]
                if min(bbox[2] - bbox[0], bbox[3] - bbox[1]) < MIN_FACE_PX:
                    continue
                kps = kpss[i]
                emb = self._embed(img_bgr, kps)
                out.append(DetectedFace(bbox=bbox, kps=kps, det_score=float(bboxes[i, 4]), embedding=emb))
        return out

    def _embed(self, img_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        from insightface.utils import face_align

        aligned = face_align.norm_crop(img_bgr, landmark=kps, image_size=self.rec.input_size[0])
        feat = self.rec.get_feat(aligned).flatten().astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-12)


def _pick_providers() -> list[str]:
    available = ort.get_available_providers()
    if os.environ.get("FACE_USE_GPU") == "1" and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
