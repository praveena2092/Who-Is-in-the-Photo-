"""Fast tests for the parts that don't need the models: run with  pytest tests/"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.gallery import name_from_filename  # noqa: E402
from app.matching import assign, person_scores  # noqa: E402


def test_names_from_files():
    assert name_from_filename("Priya Sharma.jpg") == "Priya Sharma"
    assert name_from_filename("people/Priya Sharma.JPG") == "Priya Sharma"
    assert name_from_filename("priya_sharma.png") == "Priya Sharma"
    assert name_from_filename("Priya Sharma (2).jpg") == "Priya Sharma"
    assert name_from_filename("Priya Sharma - Copy.jpg") == "Priya Sharma"
    assert name_from_filename("Priya Sharma 2.jpg") == "Priya Sharma"
    assert name_from_filename("Dr. A.P.J. Kalam.jpeg") == "Dr. A.P.J. Kalam"
    assert name_from_filename("O'Brien.webp") == "O'Brien"


def test_one_name_per_face():
    # Two faces both look most like person 0; the better one gets the name,
    # the other falls back to its own best remaining match.
    scores = np.array([[0.70, 0.10], [0.60, 0.55]])
    m = assign(scores, threshold=0.35)
    assert m[0].person_index == 0 and m[1].person_index == 1


def test_unknown_below_threshold():
    scores = np.array([[0.20, 0.10], [0.80, 0.05]])
    m = assign(scores, threshold=0.35)
    assert m[0].person_index is None
    assert m[1].person_index == 0


def test_never_reuses_a_person():
    scores = np.array([[0.9], [0.8], [0.7]])
    m = assign(scores, threshold=0.35)
    assert [x.person_index for x in m].count(0) == 1


def test_person_with_several_photos_uses_best():
    faces = np.eye(3, dtype=np.float32)
    refs = np.eye(3, dtype=np.float32)[[0, 1, 2]]
    owner = np.array([0, 0, 1])  # person 0 has two photos
    s = person_scores(faces, refs, owner, 2)
    assert s[1, 0] == 1.0 and s[2, 1] == 1.0


def test_empty_inputs():
    assert assign(np.zeros((0, 5))) == []
    assert [x.person_index for x in assign(np.zeros((3, 0)))] == [None, None, None]
