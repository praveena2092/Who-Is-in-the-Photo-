"""HTTP API + serves the built React app. Everything runs on localhost."""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import imageio, matching
from .engine import FaceEngine, ModelsMissingError
from .gallery import Gallery

GROUP_MAX_SIDE = 3200
MAX_UPLOAD_BYTES = 40 * 1024 * 1024
KEEP_PHOTOS = 8

app = FastAPI(title="Who Is in the Photo?")


# ---------------------------------------------------------------- state ----

class State:
    engine: FaceEngine | None = None
    gallery: Gallery | None = None
    error: str | None = None
    loading = True


state = State()


@dataclass
class Photo:
    id: str
    jpeg: bytes
    width: int
    height: int
    faces: list            # DetectedFace
    thumbs: list[bytes]


photos: "OrderedDict[str, Photo]" = OrderedDict()
photos_lock = threading.Lock()


def _load_models() -> None:
    try:
        state.engine = FaceEngine()
        state.gallery = Gallery(state.engine)
    except ModelsMissingError as exc:
        state.error = str(exc)
    except Exception as exc:  # pragma: no cover
        state.error = f"Could not load the face models: {exc}"
    finally:
        state.loading = False


@app.on_event("startup")
def _startup() -> None:
    threading.Thread(target=_load_models, daemon=True).start()


def _require_ready() -> Gallery:
    if state.loading:
        raise HTTPException(503, "The face models are still loading. Try again in a few seconds.")
    if state.error or state.gallery is None:
        raise HTTPException(500, state.error or "Face models unavailable.")
    return state.gallery


async def _read(upload: UploadFile) -> bytes:
    data = await upload.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{upload.filename} is larger than 40 MB.")
    return data


# ------------------------------------------------------------------ api ----

@app.get("/api/status")
def status():
    g = state.gallery
    return {
        "ready": not state.loading and state.error is None,
        "loading": state.loading,
        "error": state.error,
        "people": len(g.people) if g else 0,
        "providers": state.engine.providers if state.engine else [],
        "default_threshold": matching.DEFAULT_THRESHOLD,
    }


@app.delete("/api/gallery")
def clear_gallery():
    _require_ready().clear()
    return {"people": 0}


@app.post("/api/gallery/photos")
async def add_gallery_photos(files: list[UploadFile] = File(...)):
    gallery = _require_ready()
    reports = []
    for f in files:
        data = await _read(f)
        # Inference is CPU-bound; keep the event loop free.
        from starlette.concurrency import run_in_threadpool
        rep = await run_in_threadpool(gallery.add, f.filename or "unnamed", data)
        reports.append(rep.__dict__)
    return {"reports": reports, "people": len(gallery.people)}


@app.get("/api/gallery")
def list_gallery():
    gallery = _require_ready()
    return {"people": [
        {"id": p.id, "name": p.name, "files": p.files, "thumb_url": f"/api/gallery/{p.id}/thumb.jpg"}
        for p in sorted(gallery.people.values(), key=lambda p: p.name.casefold())
    ]}


@app.get("/api/gallery/{pid}/thumb.jpg")
def gallery_thumb(pid: str):
    gallery = _require_ready()
    p = gallery.people.get(pid)
    if not p or not p.thumb:
        raise HTTPException(404)
    return Response(p.thumb, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


class MatchRequest(BaseModel):
    threshold: float = Field(matching.DEFAULT_THRESHOLD, ge=0.1, le=0.9)


@app.post("/api/photos")
async def recognize(file: UploadFile = File(...), threshold: float = matching.DEFAULT_THRESHOLD):
    gallery = _require_ready()
    data = await _read(file)
    from starlette.concurrency import run_in_threadpool

    def work() -> Photo:
        img = imageio.decode(data, GROUP_MAX_SIDE)
        faces = state.engine.faces(imageio.to_bgr(img), group=True)
        faces = reading_order(faces)
        thumbs = [imageio.face_thumb(img, f.bbox) for f in faces]
        return Photo(uuid.uuid4().hex[:12], imageio.jpeg(img, 90), img.width, img.height, faces, thumbs)

    try:
        photo = await run_in_threadpool(work)
    except imageio.BadImageError as exc:
        raise HTTPException(400, str(exc))

    with photos_lock:
        photos[photo.id] = photo
        while len(photos) > KEEP_PHOTOS:
            photos.popitem(last=False)
    return _result(gallery, photo, threshold)


@app.post("/api/photos/{photo_id}/match")
def rematch(photo_id: str, req: MatchRequest):
    """Re-run matching with a different strictness, without re-detecting."""
    gallery = _require_ready()
    photo = photos.get(photo_id)
    if not photo:
        raise HTTPException(404, "That photo has expired. Please upload it again.")
    return _result(gallery, photo, req.threshold)


@app.get("/api/photos/{photo_id}/image.jpg")
def photo_image(photo_id: str):
    photo = photos.get(photo_id)
    if not photo:
        raise HTTPException(404)
    return Response(photo.jpeg, media_type="image/jpeg")


@app.get("/api/photos/{photo_id}/faces/{idx}.jpg")
def photo_face(photo_id: str, idx: int):
    photo = photos.get(photo_id)
    if not photo or not (0 <= idx < len(photo.thumbs)):
        raise HTTPException(404)
    return Response(photo.thumbs[idx], media_type="image/jpeg")


def reading_order(faces: list) -> list:
    """Number faces the way people read a group photo: row by row, left to right."""
    if not faces:
        return faces
    heights = sorted(float(f.bbox[3] - f.bbox[1]) for f in faces)
    gap = heights[len(heights) // 2] * 0.6
    by_y = sorted(faces, key=lambda f: float(f.bbox[1] + f.bbox[3]) / 2)
    rows, current, last_y = [], [], None
    for f in by_y:
        cy = float(f.bbox[1] + f.bbox[3]) / 2
        if last_y is not None and cy - last_y > gap:
            rows.append(current)
            current = []
        current.append(f)
        last_y = cy
    rows.append(current)
    return [f for row in rows for f in sorted(row, key=lambda f: float(f.bbox[0]))]


def _result(gallery: Gallery, photo: Photo, threshold: float) -> dict:
    people, refs, owner = gallery.matrix()
    embs = np.stack([f.embedding for f in photo.faces]) if photo.faces else np.zeros((0, 512), np.float32)
    scores = matching.person_scores(embs, refs, owner, len(people))
    found = matching.assign(scores, threshold)

    faces = []
    for i, (face, m) in enumerate(zip(photo.faces, found)):
        x1, y1, x2, y2 = [float(v) for v in face.bbox]
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(photo.width), x2), min(float(photo.height), y2)
        known = m.person_index is not None
        person = people[m.person_index] if known else None
        faces.append({
            "index": i,
            "number": i + 1,
            "box": {"x": x1 / photo.width, "y": y1 / photo.height,
                    "w": (x2 - x1) / photo.width, "h": (y2 - y1) / photo.height},
            "status": "known" if known else "unknown",
            "person_id": person.id if person else None,
            "name": person.name if person else "Unknown",
            "score": round(m.score, 3),
            "confidence": matching.confidence_label(m.score, threshold) if known else None,
            "candidates": [{"person_id": people[c.person_index].id, "name": people[c.person_index].name,
                            "score": round(c.score, 3)} for c in m.candidates],
            "thumb_url": f"/api/photos/{photo.id}/faces/{i}.jpg",
        })
    known = sum(f["status"] == "known" for f in faces)
    return {
        "photo_id": photo.id,
        "image_url": f"/api/photos/{photo.id}/image.jpg",
        "width": photo.width,
        "height": photo.height,
        "threshold": threshold,
        "gallery_size": len(people),
        "faces": faces,
        "summary": {"total": len(faces), "known": known, "unknown": len(faces) - known},
    }


# ------------------------------------------------------------- frontend ----

DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        target = DIST / path
        if path and target.is_file() and DIST in target.resolve().parents:
            return FileResponse(target)
        return FileResponse(DIST / "index.html")
else:
    @app.get("/", include_in_schema=False)
    def no_ui():
        return JSONResponse({"message": "API is running. Build the UI with `npm run build` in frontend/, "
                                        "or run `npm run dev` there for development."})
