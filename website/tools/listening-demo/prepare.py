"""Render a recording through the 2ndPressing engine for the listening demo.

Writes into OUT_DIR:
  audio/original.mp4, audio/repaired.mp4, audio/clicks.mp4  (AAC, aligned)
  demo.json   click times (output timeline), waveform overviews, settings,
              chapters, credit and the private flag for build.py

Run from the repo's tools/ directory (numpy + soundfile; macOS afconvert):
  uv run python ../website/tools/listening-demo/prepare.py IN_FILE OUT_DIR \\
      --credit "Artist, “Title” (label, year)" \\
      --chapter "0=end of the previous song" --chapter "5=crackle between the songs" \\
      [--private]

All three versions go through the same latency compensation, so they line
up sample for sample; they are encoded identically. Settings: the engine
defaults (sensitivity 50 %, percussion protection Extended).
"""
import argparse, json, pathlib, re, subprocess, tempfile
import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[3]          # 2ndPressing/
CLI = ROOT / "runtime" / "target" / "release" / "pressing"

ap = argparse.ArgumentParser()
ap.add_argument("input"); ap.add_argument("out")
ap.add_argument("--credit", default="")
ap.add_argument("--chapter", action="append", default=[], help='"SECONDS=label"')
ap.add_argument("--note", default="", help="extra listening note (HTML allowed)")
ap.add_argument("--private", action="store_true", help="show the private-prototype banner")
a = ap.parse_args()
out = pathlib.Path(a.out); (out / "audio").mkdir(parents=True, exist_ok=True)

with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d)
    rep = subprocess.run([str(CLI), "declick", a.input, str(d / "repaired.wav"), "--report"],
                         check=True, capture_output=True, text=True)
    report = rep.stdout + rep.stderr
    for mode, name in (("clicks-only", "clicks"), ("bypass", "original")):
        subprocess.run([str(CLI), "declick", a.input, str(d / f"{name}.wav"), "--mode", mode],
                       check=True, capture_output=True)
    for name in ("original", "repaired", "clicks"):
        subprocess.run(["afconvert", "-f", "mp4f", "-d", "aac", "-b", "256000", "-q", "127", "-s", "3",
                        str(d / f"{name}.wav"), str(out / "audio" / f"{name}.mp4")], check=True)
    x, sr = sf.read(d / "original.wav"); c, _ = sf.read(d / "clicks.wav")

times = [round(int(m[1]) * 60 + float(m[2]), 4)
         for m in re.finditer(r"^\s+(\d+):(\d+\.\d+)\s+\d\s+\d+\s+\d+\s*$", report, re.M)]
eng = re.search(r"threshold ([\d.]+), percussion protection (\w+), latency (\d+) samples", report)
bins = 720; n = len(x) // bins
env = [round(float(np.abs(x[i*n:(i+1)*n]).max()), 3) for i in range(bins)]
cenv = [round(float(np.abs(c[i*n:(i+1)*n]).max()), 3) for i in range(bins)]
chapters = []
for ch in a.chapter:
    sec, label = ch.split("=", 1)
    chapters.append({"t": float(sec), "label": label})
json.dump({"duration": round(len(x) / sr, 4), "sr": sr, "latency_s": round(int(eng[3]) / sr, 4),
           "threshold": float(eng[1]), "sensitivity": 50, "protection": eng[2],
           "regions": times, "env": env, "cenv": cenv, "chapters": chapters,
           "credit": a.credit, "note": a.note, "private": a.private},
          open(out / "demo.json", "w"))
print(f"{len(times)} repairs in {len(x)/sr:.2f} s -> {out}")
