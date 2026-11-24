"""Prove the app needs no internet: run on Linux with  unshare -rn  (no network at all).

    cd backend && unshare -rn env PYTHONPATH=. python ../scripts/offline_check.py ../practice_data
"""
import json, time, socket
from pathlib import Path
try:
    socket.create_connection(("pypi.org", 443), timeout=3); print("internet: REACHABLE")
except OSError as e:
    print("internet: none ->", e.__class__.__name__)
from fastapi.testclient import TestClient
from app.main import app
import sys
D = Path(sys.argv[1] if len(sys.argv) > 1 else "practice_data")
with TestClient(app) as c:
    while c.get("/api/status").json()["loading"]: time.sleep(0.5)
    print("status:", c.get("/api/status").json())
    c.delete("/api/gallery")
    for p in sorted((D/"people").iterdir()):
        c.post("/api/gallery/photos", files=[("files", (p.name, p.read_bytes()))])
    for photo, want in json.loads((D/"expected.json").read_text()).items():
        r = c.post("/api/photos", files={"file": (photo, (D/photo).read_bytes())}).json()
        got = [f["name"] for f in r["faces"]]
        print(photo, "PASS" if sorted(got) == sorted(want) else "FAIL", r["summary"])
