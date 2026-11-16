"""Round 4 review page: six camera views of one SL-1200-scale model, plus the
front-view icon rendered from the same model."""
import pathlib
from model import *

OUT = pathlib.Path(__file__).with_name("logo-options.html")
VIEWS = [
 ("V1", "Low three-quarter", dict(target=(1, 9, 12), yaw=34, pitch=6, dist=120), 66,
  "Camera 6° above the record, like your photo. Name clear, finger lift and arm rising behind.", "Website hero, installer"),
 ("V2", "Three-quarter, 20° up", dict(target=(1, 9, 12), yaw=32, pitch=20, dist=130), 66,
  "Shows the headshell top with its ½″ slots and screws, the finger lift and the locking ring, with the name still easy to read.", "All-round logo"),
 ("V3", "Almost head-on", dict(target=(0, 9, 12), yaw=14, pitch=9, dist=120), 66,
  "The name at its largest. The headshell, lift and arm frame it without competing.", "About box, social image"),
 ("V4", "High three-quarter", dict(target=(2, 9, 18), yaw=42, pitch=34, dist=150), 66,
  "Looking down on the whole headshell: slots, screws, finger lift, collar. The name is foreshortened.", "Product shot, manual cover"),
 ("V5", "Macro, low and close", dict(target=(1, 8, 10), yaw=26, pitch=4, dist=62, fov=46), 66,
  "Wide-angle close-up. Strong perspective, the stylus and grooves up front. The most dramatic.", "Splash screen, poster"),
 ("V6", "Wide, with the S-arm", dict(target=(6, 10, 34), yaw=52, pitch=24, dist=230), 80,
  "Pulled back to show the SL-1200’s S-shaped arm and knurled locking ring. Most turntable context, smallest name.", "Website section art"),
]

def icon(fin, uid, name="2nd"):
    """Front view from the same model (near-orthographic), with a record slab."""
    cam = Cam(target=(4, 13, 10), yaw=0, pitch=4, dist=4000, fov=.62)
    S = 400.0
    faces = [f for f in model(fin) if max(p[2] for p in f.pts) < 64]
    vis = sorted(((cam.cs(centroid(f.pts))[2] - f.bias, f) for f in faces if dot(f.n, sub(cam.pos, centroid(f.pts))) > 0), key=lambda t: -t[0])
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
        if f.decal == "name" and name:
            o = cam.proj((0.0, 18.6, 2.6), S); X = cam.proj((1.0, 18.6, 2.6), S); Y = cam.proj((0.0, 17.6, 2.6), S)
            m = f'matrix({X[0]-o[0]:.4f} {X[1]-o[1]:.4f} {Y[0]-o[0]:.4f} {Y[1]-o[1]:.4f} {o[0]:.2f} {o[1]:.2f})'
            r = .30 * 7.2
            out.append(f'<g transform="{m}"><text x="0" y="9.4" font-family="\'IBM Plex Sans Condensed\', \'Arial Narrow\', sans-serif" '
                       f'font-weight="600" font-size="7.2" fill="{fin["text"]}" text-anchor="middle">2<tspan font-size="{.6*7.2:.2f}" dy="-{r:.2f}">nd</tspan></text></g>')
    # Record slab with a V groove under the tip.
    gw, gd, th = side * .045, side * .045, side * .09
    L, R = vb[0] - 1, vb[0] + vb[2] + 1
    out.append(f'<path d="M{L:.2f} {tipy:.2f} H{tipx-gw:.2f} L{tipx:.2f} {tipy+gd:.2f} L{tipx+gw:.2f} {tipy:.2f} H{R:.2f} V{tipy+th:.2f} H{L:.2f} Z" fill="#1B1A17"/>')
    out.append(f'<path d="M{L:.2f} {tipy:.2f} H{tipx-gw:.2f} L{tipx:.2f} {tipy+gd:.2f} L{tipx+gw:.2f} {tipy:.2f} H{R:.2f}" fill="none" stroke="#4A4740" stroke-width="{side*.008:.2f}"/>')
    out.append('</svg>')
    return "\n".join(out)

def wm_svg(x, y, s, fill, extra=""):
    r = round(.30 * s, 2)
    return (f'<text x="{x}" y="{y}" font-family="\'IBM Plex Sans Condensed\', \'Arial Narrow\', sans-serif" font-weight="600" font-size="{s}" fill="{fill}">'
            f'2<tspan font-size="{round(.6*s,2)}" dy="-{r}">nd</tspan><tspan font-size="{s}" dy="{r}">Pressing</tspan>{extra}</text>')

V2 = dict(target=(1, 9, 12), yaw=32, pitch=20, dist=130)
hero_a1 = render(FIN_A1, Cam(**V2), uid="h1", focus=66, aspect=1.45, grooves=True, full=True)
cards = [f'''<article class="card hero-card"><div class="art">{hero_a1}</div>
<div class="meta"><span class="letter">A1</span><h2>Black cartridge, amber stylus, silver headshell and arm</h2></div></article>''']
ic_a1 = icon(FIN_A1, "i1")
ic_band = icon(FIN_A1, "ib", name=None)  # plugin banner: no "2nd" on the cartridge
def app(svg, px):
    return (f'<div class="app" style="width:{px}px;height:{px}px;padding:{round(px*.1)}px">{svg}</div>')

icons = f'''
<section class="icons">
  <div class="icons-copy"><p class="eyebrow">Icon · front view from the same model</p>
  <h2>App icon and favicon</h2>
  <p>Seen straight on, the cartridge carries “2nd”. The finger lift on the right and the locking ring behind give it a recognisable outline. Shown at 128 and 64 px as an app icon, and at 32 and 16 px as a favicon.</p></div>
  <div class="icon-row">
    {app(ic_a1, 128)}{app(ic_a1, 64)}<div class="raw r32">{ic_a1}</div><div class="raw r16">{ic_a1}</div>
  </div>
  <div class="bandwrap"><svg viewBox="0 0 520 54" role="img" aria-label="Plugin header with the icon">
    <rect width="520" height="54" fill="#3C3A36"/>
    <foreignObject x="14" y="6" width="42" height="42"><div xmlns="http://www.w3.org/1999/xhtml" style="width:42px;height:42px">{ic_band}</div></foreignObject>
    {wm_svg(62, 34, 18, "#EBE6DA", '<tspan dx="8" fill="#A9A395" font-size="18">DECLICKER</tspan>')}
    <text x="502" y="33" text-anchor="end" font-family="'IBM Plex Mono', ui-monospace, monospace" font-size="9.5" fill="#A9A395">CLICK REPAIR · STEREO</text></svg></div>
</section>'''

html = f'''<title>2ndPressing Logo Options</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>
/* Layout: contact sheet. Six camera views of one model in a 3-up grid, then
   the icon strip. Single light theme by choice: judged on the brand panel. */
:root {{
  --page: #E4DFD4; --card: #EEEAE1; --rule: #CFC8B8; --text: #2C2A26; --muted: #6A6455;
  --band: #3C3A36; --band-dim: #A9A395; --amber: #F2A93B; --tile: #F7F4EE;
  --display: "IBM Plex Sans Condensed", "Arial Narrow", sans-serif;
  --body: "IBM Plex Sans", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, monospace;
  color-scheme: light;
}}
body {{ background: var(--page); color: var(--text); font: 400 15px/1.55 var(--body); }}
.wrap {{ max-width: 1320px; margin: 0 auto; padding-block: 40px 56px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
.top {{ display: grid; gap: 10px; margin-bottom: 26px; }}
.eyebrow {{ font: 600 12px/1 var(--display); letter-spacing: .14em; text-transform: uppercase; color: var(--muted); margin: 0; }}
h1 {{ font: 600 clamp(30px, 4vw, 42px)/1.05 var(--display); margin: 0; text-wrap: balance; }}
.intro {{ max-width: 76ch; margin: 0; color: #3A3732; }}
.grid {{ display: grid; grid-template-columns: minmax(0, 1fr); gap: 18px; max-width: 980px; }}
.card {{ background: var(--card); border: 1px solid var(--rule); border-radius: 10px; padding: 12px 12px 16px; display: grid; gap: 8px; align-content: start; min-width: 0; }}
.art {{ background: var(--tile); border: 1px solid #DDD6C8; border-radius: 6px; overflow: hidden; }}
.art svg {{ width: 100%; height: auto; display: block; }}
.meta {{ display: flex; align-items: baseline; gap: 10px; padding: 4px 4px 0; }}
.letter {{ font: 500 12px/1 var(--mono); color: var(--amber); background: #1B1A17; border-radius: 3px; padding: 5px 7px; }}
h2 {{ font: 600 21px/1.15 var(--display); margin: 0; }}
.what {{ margin: 0; padding: 0 4px; font-size: 14px; color: #3A3732; }}
.use {{ margin: 0; padding: 0 4px; font-size: 13px; color: var(--muted); }}
.use span {{ font: 600 11px/1 var(--display); letter-spacing: .1em; text-transform: uppercase; margin-right: 6px; }}
.icons {{ margin-top: 22px; background: var(--card); border: 1px solid var(--rule); border-radius: 10px; padding: 18px; display: grid; gap: 16px; }}
.icons-copy {{ display: grid; gap: 6px; }}
.icons-copy p:not(.eyebrow) {{ margin: 0; max-width: 76ch; color: #3A3732; }}
.icons h2 {{ font-size: 24px; }}
.icon-row {{ display: flex; flex-wrap: wrap; align-items: flex-end; gap: 18px; background: var(--tile); border: 1px solid #DDD6C8; border-radius: 6px; padding: 18px; }}
.app {{ border-radius: 22%; overflow: hidden; background: linear-gradient(#DAD5C9, #BEB8AA); box-shadow: 0 0 0 1px rgba(140,133,119,.6) inset, 0 6px 12px rgba(40,36,30,.25); box-sizing: border-box; flex: none; }}
.app svg, .raw svg {{ width: 100%; height: 100%; display: block; }}
.raw.r32 {{ width: 32px; height: 32px; flex: none; }} .raw.r16 {{ width: 16px; height: 16px; flex: none; }}
.gap {{ width: 24px; }}
.bandwrap {{ border-radius: 6px; overflow: hidden; }}
.bandwrap svg {{ width: 100%; height: auto; display: block; }}
.reply {{ margin-top: 22px; padding: 18px 20px; background: var(--band); color: #EBE6DA; border-radius: 8px; display: grid; gap: 6px; }}
.reply p {{ margin: 0; max-width: 90ch; }}
.reply strong {{ color: var(--amber); font-weight: 600; }}
@media (max-width: 1080px) {{ .grid {{ grid-template-columns: minmax(0, 1fr); }} }}
@media (max-width: 640px) {{ .grid {{ grid-template-columns: minmax(0, 1fr); }} }}
</style>
<div class="wrap">
  <header class="top">
    <p class="eyebrow">Approved logo · A1</p>
    <h1>2ndPressing cartridge logo</h1>
    <p class="intro">A1, view V2: black cartridge with amber stylus on an SL-1200 style silver headshell and arm. The arm now runs almost straight from the collar, the finger lift is a solid bar at 90° off the headshell’s outer side, and the record fills the whole frame, with its grooves and the smoother bands between tracks. SL-1200 geometry throughout: 22° offset, 230 mm effective length.</p>
  </header>
  <section class="grid">{"".join(cards)}</section>
  {icons}
  <footer class="reply">
    <p><strong>Next:</strong> flag anything else that looks wrong. Once you’re happy I’ll export the logo set (SVG and PNG, light and dark) and put it into the website, plugin and app.</p>
  </footer>
</div>
'''
OUT.write_text(html)
print("wrote", OUT, len(html))
