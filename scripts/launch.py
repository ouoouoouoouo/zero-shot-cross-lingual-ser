"""Run a list of shell commands, one at a time per GPU.

    python scripts/launch.py --gpus 0 1 2 3 4 5 < jobs.txt
Each line of stdin is one command; it runs with CUDA_VISIBLE_DEVICES set to a
free GPU. Output goes to logs/<line number>.log. Exits non-zero if any job failed.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--gpus", nargs="+", required=True)
ap.add_argument("--logdir", default="logs")
a = ap.parse_args()

jobs = [l.strip() for l in sys.stdin if l.strip() and not l.startswith("#")]
Path(a.logdir).mkdir(parents=True, exist_ok=True)
free, running, failed = list(a.gpus), {}, []
for i, cmd in enumerate(jobs):
    while not free:
        for gpu, (p, c) in list(running.items()):
            if p.poll() is not None:
                if p.returncode:
                    failed.append(c)
                del running[gpu]
                free.append(gpu)
        time.sleep(2)
    gpu = free.pop(0)
    log = open(Path(a.logdir) / f"{i:03d}.log", "w")
    print(f"[gpu {gpu}] ({i + 1}/{len(jobs)}) {cmd}", flush=True)
    running[gpu] = (subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT,
                                     env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}), cmd)
for p, c in running.values():
    if p.wait():
        failed.append(c)
print(f"done: {len(jobs) - len(failed)} ok, {len(failed)} failed")
for c in failed:
    print("FAILED:", c)
sys.exit(1 if failed else 0)
