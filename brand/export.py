"""Export the approved 2ndPressing logo (A1, view V2) to every place it is used.

Run from anywhere (needs Google Chrome for rasterising):
    uvx --with fonttools --with pillow python brand/export.py

Writes:
  brand/out/                      master files (SVG, 2400 px PNG, 1024 px app icon, AppIcon.icns)
  website/public/assets/          hero WebP, header icon, favicon, apple-touch-icon
  runtime/crates/pressing-plugin/assets/brand/   band icon as raw RGBA (faceplate header)
  runtime/crates/pressing-app/assets/            window icon as raw RGBA
  runtime/crates/pressing-app/macos/AppIcon.icns
"""
import pathlib, shutil, subprocess, sys, tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / "logo3d"))
from model import Cam, FIN_A1, centroid, dot, model, render, shade, sub, wordmark_path  # noqa: E402
from PIL import Image  # noqa: E402

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
OUT = HERE / "out"
WEB = ROOT / "website" / "public" / "assets"
PLUG = ROOT / "runtime" / "crates" / "pressing-plugin" / "assets" / "brand"
APP = ROOT / "runtime" / "crates" / "pressing-app"
V2 = dict(target=(1, 9, 12), yaw=32, pitch=20, dist=130)


def front_icon(fin, with_name=True):
    """Front view of the same model (near-orthographic) with a record slab."""
    cam = Cam(target=(4, 13, 10), yaw=0, pitch=4, dist=4000, fov=.62)
    S = 400.0
    faces = [f for f in model(fin) if max(p[2] for p in f.pts) < 64]
    vis = sorted(((cam.cs(centroid(f.pts))[2] - f.bias, f) for f in faces
                  if dot(f.n, sub(cam.pos, centroid(f.pts))) > 0), key=lambda t: -t[0])
    pts = [cam.proj(p, S) for _, f in vis for p in f.pts] + [cam.proj((0, 0, 0), S)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    tipx, tipy = cam.proj((0, 0, 0), S)
    side = max(x1 - x0, y1 - y0) * 1.04
    cx = (x0 + x1) / 2
    vb = (cx - side / 2, tipy + side * .10 - side, side, side)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb[0]:.2f} {vb[1]:.2f} {vb[2]:.2f} {vb[3]:.2f}">']
    for _, f in vis:
        q = [cam.proj(p, S) for p in f.pts]
        c = '#%02x%02x%02x' % shade(f, cam)
        out.append('<polygon points="' + " ".join(f"{a:.2f},{b:.2f}" for a, b in q) + f'" fill="{c}" stroke="{c}" stroke-width=".3"/>')
        if f.decal == "name" and with_name:
            o = cam.proj((0.0, 18.6, 2.6), S); X = cam.proj((1.0, 18.6, 2.6), S); Y = cam.proj((0.0, 17.6, 2.6), S)
            m = f'matrix({X[0]-o[0]:.4f} {X[1]-o[1]:.4f} {Y[0]-o[0]:.4f} {Y[1]-o[1]:.4f} {o[0]:.2f} {o[1]:.2f})'
            out.append(f'<g transform="{m}">{wordmark_path(7.2, fin["text"], 0, 9.4, nd_only=True)}</g>')
    gw, gd, th = side * .045, side * .045, side * .09
    L, R = vb[0] - 1, vb[0] + vb[2] + 1
    out.append(f'<path d="M{L:.2f} {tipy:.2f} H{tipx-gw:.2f} L{tipx:.2f} {tipy+gd:.2f} L{tipx+gw:.2f} {tipy:.2f} H{R:.2f} V{tipy+th:.2f} H{L:.2f} Z" fill="#1B1A17"/>')
    out.append(f'<path d="M{L:.2f} {tipy:.2f} H{tipx-gw:.2f} L{tipx:.2f} {tipy+gd:.2f} L{tipx+gw:.2f} {tipy:.2f} H{R:.2f}" fill="none" stroke="#4A4740" stroke-width="{side*.008:.2f}"/>')
    out.append('</svg>')
    return "\n".join(out)


def nest(svg, x, y, w, h):
    """Place a standalone SVG inside another at x, y, w, h."""
    return svg.replace('<svg xmlns="http://www.w3.org/2000/svg" ', f'<svg x="{x}" y="{y}" width="{w}" height="{h}" ', 1)


def app_icon(icon_svg, full_bleed=False):
    """macOS-style app icon on a 1024 canvas: warm panel squircle, cartridge inside."""
    if full_bleed:
        body = '<rect width="1024" height="1024" fill="url(#bg)"/>'
        inner = nest(icon_svg, 150, 140, 724, 724)
    else:
        body = ('<rect x="100" y="112" width="824" height="824" rx="185" fill="#000" fill-opacity=".28" filter="url(#sh)"/>'
                '<rect x="100" y="100" width="824" height="824" rx="185" fill="url(#bg)"/>'
                '<rect x="104" y="104" width="816" height="816" rx="181" fill="none" stroke="#ffffff" stroke-opacity=".55" stroke-width="6"/>'
                '<rect x="100" y="100" width="824" height="824" rx="185" fill="none" stroke="#8C8577" stroke-opacity=".7" stroke-width="3"/>')
        inner = nest(icon_svg, 190, 176, 644, 644)
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024">'
            '<defs><linearGradient id="bg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#DAD5C9"/><stop offset="1" stop-color="#BEB8AA"/></linearGradient>'
            '<filter id="sh" x="-10%" y="-10%" width="120%" height="130%"><feGaussianBlur stdDeviation="14"/></filter></defs>'
            f'{body}{inner}</svg>')


def raster(svg_path, png_path, w, h):
    """Rasterise an SVG with headless Chrome on a transparent background."""
    with tempfile.TemporaryDirectory() as d:
        html = pathlib.Path(d) / "r.html"
        html.write_text(f'<!doctype html><html><body style="margin:0;background:transparent">'
                        f'<img src="file://{svg_path}" style="display:block;width:{w}px;height:{h}px"></body></html>')
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1",
                        "--default-background-color=00000000", f"--window-size={w},{h}", "--virtual-time-budget=4000",
                        f"--screenshot={png_path}", f"file://{html}"], check=True, capture_output=True)
    return Image.open(png_path).convert("RGBA").crop((0, 0, w, h))


def write_rgba(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(img.convert("RGBA").tobytes())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # 1. Master files.
    hero = render(FIN_A1, Cam(**V2), uid="logo", focus=66, aspect=1.45, grooves=True)
    (OUT / "2ndpressing-cartridge.svg").write_text(hero)
    icon = front_icon(FIN_A1, True)
    plain = front_icon(FIN_A1, False)
    (OUT / "icon-front.svg").write_text(icon)
    (OUT / "icon-front-plain.svg").write_text(plain)
    (OUT / "app-icon.svg").write_text(app_icon(icon))
    (OUT / "app-icon-full-bleed.svg").write_text(app_icon(icon, full_bleed=True))

    big = raster(OUT / "2ndpressing-cartridge.svg", OUT / "2ndpressing-cartridge-2400.png", 2400, round(2400 / 1.45))
    appimg = raster(OUT / "app-icon.svg", OUT / "app-icon-1024.png", 1024, 1024)
    bleed = raster(OUT / "app-icon-full-bleed.svg", OUT / "app-icon-full-bleed-1024.png", 1024, 1024)
    plainimg = raster(OUT / "icon-front-plain.svg", OUT / "icon-front-plain-1024.png", 1024, 1024)

    # 2. macOS .icns.
    with tempfile.TemporaryDirectory() as d:
        iset = pathlib.Path(d) / "AppIcon.iconset"
        iset.mkdir()
        for px in (16, 32, 128, 256, 512):
            appimg.resize((px, px), Image.LANCZOS).save(iset / f"icon_{px}x{px}.png")
            appimg.resize((px * 2, px * 2), Image.LANCZOS).save(iset / f"icon_{px}x{px}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iset), "-o", str(OUT / "AppIcon.icns")], check=True)
    shutil.copy(OUT / "AppIcon.icns", APP / "macos" / "AppIcon.icns")

    # 3. Website.
    for w in (900, 1600):
        big.resize((w, round(w / 1.45)), Image.LANCZOS).save(WEB / f"cartridge-{w}.webp", quality=86, method=6)
    (WEB / "header-icon.svg").write_text(plain)
    (WEB / "favicon.svg").write_text(app_icon(icon))
    bleed.resize((180, 180), Image.LANCZOS).convert("RGB").save(WEB / "apple-touch-icon.png")

    # 4. Plugin faceplate band icon (3x for Retina at 34 pt) and app window icon.
    write_rgba(plainimg.resize((102, 102), Image.LANCZOS), PLUG / "cartridge-plain-102.rgba")
    write_rgba(appimg.resize((256, 256), Image.LANCZOS), APP / "assets" / "window-icon-256.rgba")
    print("exported:", ", ".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
