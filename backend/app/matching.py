"""Who is who: turn similarity scores into names (or 'unknown').

Two rules matter more than the model:
  1. A threshold. Below it we say "unknown" rather than guess. The demo
     deliberately includes people who are not in the folder.
  2. One person can appear only once in a photo. Picking each face's best
     match independently can label two faces "Priya Sharma"; solving it as an
     assignment problem (Hungarian algorithm) gives each name to at most one
     face, the one that fits best overall.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

# Cosine similarity of ArcFace (buffalo_l) embeddings. Same person usually
# scores 0.45-0.8, different people usually below 0.2. Tuned on practice data;
# see README "Tuning". The UI exposes it as a strictness slider.
DEFAULT_THRESHOLD = 0.35
STRONG_MATCH = 0.50


@dataclass
class Candidate:
    person_index: int
    score: float


@dataclass
class FaceMatch:
    person_index: int | None   # None = unknown
    score: float               # score of the assigned person (or best score if unknown)
    candidates: list[Candidate]


def person_scores(face_embs: np.ndarray, ref_embs: np.ndarray, owner: np.ndarray, n_people: int) -> np.ndarray:
    """faces x people similarity; a person with several photos takes their best one."""
    if face_embs.size == 0 or ref_embs.size == 0:
        return np.zeros((len(face_embs), n_people), np.float32)
    sims = face_embs @ ref_embs.T  # embeddings are L2-normalised -> cosine
    out = np.full((len(face_embs), n_people), -1.0, np.float32)
    for j in range(n_people):
        cols = owner == j
        if cols.any():
            out[:, j] = sims[:, cols].max(axis=1)
    return out


def assign(scores: np.ndarray, threshold: float = DEFAULT_THRESHOLD, top_k: int = 3) -> list[FaceMatch]:
    n_faces, n_people = scores.shape
    matches: list[FaceMatch] = []
    assigned: dict[int, int] = {}
    if n_faces and n_people:
        # Only pairs above the threshold are allowed; others get a large cost so
        # the solver never prefers them.
        cost = np.where(scores >= threshold, -scores, 10.0)
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if scores[r, c] >= threshold:
                assigned[int(r)] = int(c)

    for i in range(n_faces):
        order = np.argsort(-scores[i])[:top_k] if n_people else []
        cands = [Candidate(int(j), float(scores[i, j])) for j in order]
        if i in assigned:
            j = assigned[i]
            matches.append(FaceMatch(j, float(scores[i, j]), cands))
        else:
            best = float(scores[i].max()) if n_people else 0.0
            matches.append(FaceMatch(None, best, cands))
    return matches


def confidence_label(score: float, threshold: float) -> str:
    if score >= max(STRONG_MATCH, threshold + 0.1):
        return "high"
    return "medium"
