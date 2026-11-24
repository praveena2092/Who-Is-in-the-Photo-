"""The reference folder: one person per file, file name = person's name."""

from __future__ import annotations

import hashlib
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import numpy as np

from . import imageio
from .engine import FaceEngine

REFERENCE_MAX_SIDE = 1600

# Files a folder picker drags along that are never people.
JUNK_NAMES = {".ds_store", "thumbs.db", "desktop.ini"}

_COPY_SUFFIX = re.compile(r"(\s*\(\d+\)|\s*-\s*copy(\s*\(\d+\))?|[ _]+\d{1,2})$", re.IGNORECASE)


def name_from_filename(filename: str) -> str:
    """'Priya Sharma.jpg' -> 'Priya Sharma'; 'priya_sharma (2).png' -> 'Priya Sharma'."""
    stem = PurePosixPath(filename.replace("\\", "/")).stem
    stem = _COPY_SUFFIX.sub("", stem).strip() or stem
    if " " not in stem and re.search(r"[_.]", stem):
        stem = re.sub(r"[_.]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    if stem.islower():
        stem = stem.title()
    return stem


@dataclass
class Person:
    id: str
    name: str
    files: list[str] = field(default_factory=list)
    embeddings: list[np.ndarray] = field(default_factory=list)
    digests: list[str] = field(default_factory=list)
    thumb: bytes | None = None


@dataclass
class FileReport:
    file: str
    status: str          # ok | multiple_faces | no_face | unreadable | skipped
    message: str
    person_id: str | None = None
    name: str | None = None


class Gallery:
    def __init__(self, engine: FaceEngine):
        self.engine = engine
        self.people: dict[str, Person] = {}
        self._by_name: dict[str, str] = {}
        self._emb_cache: dict[str, tuple[np.ndarray, bytes, int]] = {}  # sha1 -> (emb, thumb, n_faces)
        self._lock = threading.Lock()
        self.version = 0  # bumps on every change, so stale photo results can be re-matched

    def clear(self) -> None:
        with self._lock:
            self.people.clear()
            self._by_name.clear()
            self.version += 1

    def add(self, filename: str, data: bytes) -> FileReport:
        base = PurePosixPath(filename.replace("\\", "/")).name
        ext = PurePosixPath(base).suffix.lower()
        if base.lower() in JUNK_NAMES or base.startswith("._") or base.startswith("."):
            return FileReport(base, "skipped", "System file, ignored.")
        if ext not in imageio.IMAGE_EXTS:
            return FileReport(base, "skipped", "Not an image, ignored.")

        digest = hashlib.sha1(data).hexdigest()
        cached = self._emb_cache.get(digest)
        if cached is None:
            try:
                img = imageio.decode(data, REFERENCE_MAX_SIDE)
            except imageio.BadImageError as exc:
                return FileReport(base, "unreadable", str(exc))
            faces = self.engine.faces(imageio.to_bgr(img), group=False)
            if not faces:
                return FileReport(base, "no_face", "No face found in this photo. Try a clearer, front-facing photo.")
            # A reference portrait sometimes has someone in the background:
            # the person it's named after is almost always the biggest face.
            main = max(faces, key=lambda f: f.area)
            cached = (main.embedding, imageio.face_thumb(img, main.bbox), len(faces))
            self._emb_cache[digest] = cached

        emb, thumb, n_faces = cached
        name = name_from_filename(base)
        key = name.casefold()
        with self._lock:
            pid = self._by_name.get(key)
            if pid is None:
                pid = uuid.uuid4().hex[:10]
                self._by_name[key] = pid
                self.people[pid] = Person(id=pid, name=name, thumb=thumb)
            person = self.people[pid]
            if digest not in person.digests:  # same file twice (e.g. a copy) adds nothing
                person.digests.append(digest)
                person.files.append(base)
                person.embeddings.append(emb)
            self.version += 1

        if n_faces > 1:
            return FileReport(base, "multiple_faces",
                              f"{n_faces} faces found; using the largest one.", pid, name)
        return FileReport(base, "ok", "Ready.", pid, name)

    def matrix(self) -> tuple[list[Person], np.ndarray, np.ndarray]:
        """All reference embeddings stacked, plus which person each row belongs to."""
        with self._lock:
            people = list(self.people.values())
            rows, owner = [], []
            for idx, p in enumerate(people):
                for e in p.embeddings:
                    rows.append(e)
                    owner.append(idx)
        if not rows:
            return people, np.zeros((0, 512), np.float32), np.zeros((0,), int)
        return people, np.stack(rows), np.array(owner)
