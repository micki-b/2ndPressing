"""Build public/assets/repair.svg: a synthetic test signal with one injected
click, before and after the real 2ndPressing engine (protection off).

Run from the repo's tools/ directory (needs numpy + soundfile and a release
build of runtime/ for the `pressing` CLI):
    uv run python ../website/tools/make_figure.py
"""
import pathlib, subprocess, tempfile
import numpy as np, soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "website/public/assets/repair.svg"
CLI = ROOT / "runtime/target/release/pressing"

sr, n = 48000, 96000
t = np.arange(n) / sr
rng = np.random.default_rng(7)
x = np.zeros(n)
for f, a in [(220.0, .22), (277.18, .15), (329.63, .13), (440.0, .07), (659.3, .04), (880.0, .025)]:
    x += a * np.sin(2 * np.pi * f * t * (1 + .0015 * np.sin(2 * np.pi * 5.1 * t)) + rng.uniform(0, 6.28))
x = 0.8 * x + np.convolve(rng.normal(0, .004, n), np.ones(6) / 6, "same")
c = int(1.2 * sr)
k = np.arange(48)
click = 0.55 * np.exp(-k / 5.0) * np.cos(2 * np.pi * k / 9.0)
click[0:2] = [0.25, 0.7]
x[c:c + 48] += click

with tempfile.TemporaryDirectory() as d:
    src, dst = f"{d}/in.wav", f"{d}/out.wav"
    sf.write(src, np.stack([x, x * .97], 1).astype(np.float32), sr, subtype="PCM_24")
    subprocess.run([str(CLI), "declick", src, dst, "--protection", "off"], check=True, capture_output=True)
    xi, _ = sf.read(src)
    y, _ = sf.read(dst)

lo, hi = c - 150, c + 210
xs, ys = xi[lo:hi, 0], y[lo:hi, 0]
changed = np.nonzero(np.abs(ys - xs) > 1e-6)[0]
f0, f1 = int(changed.min()), int(changed.max())

W, H, PAD = 720.0, 300.0, 6.0
def px(i): return PAD + i * (W - 2 * PAD) / (len(xs) - 1)
def py(v): return H / 2 - v * (H / 2 - PAD) / 0.9
def path(v): return "M" + " L".join(f"{px(i):.1f},{py(s):.1f}" for i, s in enumerate(v))

grid = []
for i in range(0, len(xs), 24):          # 0.5 ms columns
    grid.append(f'<line x1="{px(i):.1f}" y1="0" x2="{px(i):.1f}" y2="{H:.0f}"/>')
for v in (-0.6, -0.3, 0.3, 0.6):
    grid.append(f'<line x1="0" y1="{py(v):.1f}" x2="{W:.0f}" y2="{py(v):.1f}"/>')
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" role="img" aria-label="A click spike in the input waveform, and the engine's rebuilt waveform through the same span">
<rect width="100%" height="100%" fill="#F4ECD4"/>
<g stroke="#D9CBA4" stroke-width="1">{"".join(grid)}</g>
<line x1="0" y1="{H/2:.0f}" x2="{W:.0f}" y2="{H/2:.0f}" stroke="#C9B98F" stroke-width="1.2"/>
<rect x="{px(f0) - 2:.1f}" y="0" width="{px(f1) - px(f0) + 4:.1f}" height="{H:.0f}" fill="#F2A93B" fill-opacity=".22"/>
<path d="{path(xs)}" fill="none" stroke="#E2553F" stroke-width="2" stroke-linejoin="round"/>
<path d="{path(ys)}" fill="none" stroke="#2F3B6B" stroke-width="2.4" stroke-linejoin="round"/>
</svg>
'''
OUT.write_text(svg)
print(f"wrote {OUT}: {len(changed)} samples rebuilt ({len(changed) / sr * 1000:.2f} ms)")
