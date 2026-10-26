"""Integrity manifest of all stored runs: results/MANIFEST.csv (path, bytes, SHA-256, algorithm, seed, nfe,
configuration hash, code hash). Re-run after new campaigns; verify with --check.
Usage: python -I -X utf8 results_manifest.py [--check]"""
import argparse
import csv
import glob
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "MANIFEST.csv")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build():
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "**", "*.npz"), recursive=True)):
        rel = os.path.relpath(f, RES).replace("\\", "/")
        rec = [rel, os.path.getsize(f), sha(f), "", "", "", "", ""]
        try:
            z = np.load(f, allow_pickle=False)
            if "meta" in z.files:
                m = json.loads(str(z["meta"]))
                rec[3:] = [m.get("alg", ""), m.get("seed", ""), m.get("nfe", ""), m.get("cfg_hash", ""), m.get("code_hash", "")]
        except Exception:
            pass
        rows.append(rec)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "bytes", "sha256", "alg", "seed", "nfe", "cfg_hash", "code_hash"])
        w.writerows(rows)
    return len(rows)


def check():
    bad = 0
    with open(OUT, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            p = os.path.join(RES, r["path"])
            if not os.path.exists(p) or sha(p) != r["sha256"]:
                bad += 1
                print("MISMATCH", r["path"])
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    print("mismatches:", check()) if a.check else print("files in manifest:", build())
