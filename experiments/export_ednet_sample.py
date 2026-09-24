"""Publish the deterministic EdNet-KT1 user sample for exact reproducibility.

Re-runs the load_users selection of run_ednet.py (same seed, same inclusion
rule) and writes the kept user ids to results/ednet_sample_uids.csv.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import seed_everything
import run_ednet as red

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

if __name__ == "__main__":
    seed_everything()
    flat, _, _, _ = red.load_users()
    uids = sorted(flat.uid.astype(str))
    with open(os.path.join(OUT, "ednet_sample_uids.csv"), "w") as fh:
        fh.write("uid\n")
        fh.write("\n".join(uids) + "\n")
    print(f"wrote {len(uids):,} uids")
