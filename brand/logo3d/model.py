"""Tiny perspective SVG renderer + an SL-1200-style headshell, MM cartridge and
S-arm, modelled in millimetres. Origin: stylus tip on the record surface.
Axes: x = viewer's right when facing the cartridge front (the outer side of
the record), y = up, z = toward the back of the turntable (toward the arm)."""
import math
import os

# Lettering as vector outlines (IBM Plex Sans Condensed SemiBold) when
# fontTools is available, so exported SVGs need no installed font.
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runtime", "crates",
                         "pressing-plugin", "assets", "fonts", "IBMPlexSansCondensed-SemiBold.ttf")
_FONT = []

def _font():
    if not _FONT:
        try:
            from fontTools.ttLib import TTFont
            _FONT.append(TTFont(FONT_PATH))
        except Exception:
            _FONT.append(None)
    return _FONT[0]

def wordmark_path(size, fill, x=0.0, y=0.0, nd_only=False):
    """'2ndPressing' (or '2nd') with a moderate superscript, centred on x,
    baseline y, as a single SVG path; None if fontTools is unavailable."""
    f = _font()
    if f is None:
        return None
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen
    gs, cmap, upm, hmtx = f.getGlyphSet(), f.getBestCmap(), f["head"].unitsPerEm, f["hmtx"]
    runs = [("2", size, 0.0), ("nd", .6 * size, -.30 * size)] + ([] if nd_only else [("Pressing", size, 0.0)])
    width = sum(hmtx[cmap[ord(c)]][0] * sz / upm for t, sz, _ in runs for c in t)
    cx, d = x - width / 2, []
    for text, sz, dy in runs:
        k = sz / upm
        for ch in text:
            g = cmap[ord(ch)]
            pen = SVGPathPen(gs)
            gs[g].draw(TransformPen(pen, (k, 0, 0, -k, cx, y + dy)))
            d.append(pen.getCommands())
            cx += hmtx[g][0] * k
    return f'<path d="{" ".join(d)}" fill="{fill}"/>'

def add(a, b): return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
def sub(a, b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def mul(a, k): return (a[0]*k, a[1]*k, a[2]*k)
def dot(a, b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def norm(a):
    l = math.sqrt(dot(a, a)) or 1.0
    return (a[0]/l, a[1]/l, a[2]/l)
def centroid(pts): return mul(pts[0] if len(pts) == 1 else tuple(map(sum, zip(*pts))), 1.0/len(pts))

class Face:
    def __init__(self, pts, color, mat="satin", inside=None, decal=None, nocull=False, outline=True, bias=0.0):
        self.pts, self.color, self.mat, self.decal, self.nocull, self.outline, self.bias = pts, color, mat, decal, nocull, outline, bias
        n = norm(cross(sub(pts[1], pts[0]), sub(pts[2], pts[0])))
        if inside is not None and dot(n, sub(centroid(pts), inside)) < 0:
            n = mul(n, -1)
        self.n = n

def box(x0, x1, y0, y1, z0, z1, color, mat="satin", skip=(), decals=None):
    c = ((x0+x1)/2, (y0+y1)/2, (z0+z1)/2)
    F = {
      "front": [(x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0)],
      "back":  [(x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)],
      "left":  [(x0,y0,z0),(x0,y1,z0),(x0,y1,z1),(x0,y0,z1)],
      "right": [(x1,y0,z0),(x1,y1,z0),(x1,y1,z1),(x1,y0,z1)],
      "top":   [(x0,y1,z0),(x1,y1,z0),(x1,y1,z1),(x0,y1,z1)],
      "bottom":[(x0,y0,z0),(x1,y0,z0),(x1,y0,z1),(x0,y0,z1)],
    }
    return [Face(p, color, mat, c, (decals or {}).get(k)) for k, p in F.items() if k not in skip]

def prism(poly_xz, y0, y1, color, mat="satin", top_color=None):
    """Extrude a convex-ish xz polygon between y0 and y1."""
    cx = sum(p[0] for p in poly_xz)/len(poly_xz); cz = sum(p[1] for p in poly_xz)/len(poly_xz)
    c = (cx, (y0+y1)/2, cz)
    out = [Face([(x, y1, z) for x, z in poly_xz], top_color or color, mat, c),
           Face([(x, y0, z) for x, z in poly_xz], color, mat, c)]
    for i in range(len(poly_xz)):
        a, b = poly_xz[i], poly_xz[(i+1) % len(poly_xz)]
        out.append(Face([(a[0], y0, a[1]), (b[0], y0, b[1]), (b[0], y1, b[1]), (a[0], y1, a[1])], color, mat, c))
    return out

def sweep(path, section, color, mat="metal", caps=True, up=(0, 1, 0)):
    """Sweep a closed 2D section [(u, v)] along a 3D polyline."""
    rings = []
    for i, p in enumerate(path):
        t = norm(sub(path[min(i+1, len(path)-1)], path[max(i-1, 0)]))
        ref = up if abs(dot(t, up)) < .95 else (1, 0, 0)
        u = norm(cross(ref, t)); v = cross(t, u)
        rings.append((p, [add(p, add(mul(u, a), mul(v, b))) for a, b in section]))
    out = []
    for (p0, r0), (p1, r1) in zip(rings, rings[1:]):
        mid = mul(add(p0, p1), .5)
        for j in range(len(section)):
            k = (j+1) % len(section)
            out.append(Face([r0[j], r0[k], r1[k], r1[j]], color, mat, mid))
    if caps:
        for (p, r), other in ((rings[0], rings[1][0]), (rings[-1], rings[-2][0])):
            out.append(Face(list(r), color, mat, other))
    return out

def loft(rings, color, mat="metal"):
    """Skin consecutive rings (same vertex count) into quads."""
    out = []
    for r0, r1 in zip(rings, rings[1:]):
        mid = mul(add(centroid(r0), centroid(r1)), .5)
        for j in range(len(r0)):
            k = (j + 1) % len(r0)
            out.append(Face([r0[j], r0[k], r1[k], r1[j]], color, mat, mid))
    return out

def superellipse(a, b, n, cy, z, cnt=32, cx=0.0):
    pts = []
    for i in range(cnt):
        t = 2 * math.pi * i / cnt
        c, s_ = math.cos(t), math.sin(t)
        u = a * math.copysign(abs(c) ** (2.0 / n), c)
        v = b * math.copysign(abs(s_) ** (2.0 / n), s_)
        pts.append((cx + u, cy + v, z))
    return pts

def circle_section(r, n=18): return [(r*math.cos(2*math.pi*i/n), r*math.sin(2*math.pi*i/n)) for i in range(n)]
def rect_section(w, h): return [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]

# ---------------- The model ----------------
def model(fin):
    F = []
    # Cartridge body: 17.5 W x 14 H (main) x 24 D, chamfered front vertical edges.
    x0, x1, y0, y1, z0, z1, ch = -8.75, 8.75, 4.6, 18.6, 2.6, 26.6, 1.1
    poly = [(x0+ch, z0), (x1-ch, z0), (x1, z0+ch), (x1, z1), (x0, z1), (x0, z0+ch)]
    body = [f for f in prism(poly, y0, y1, fin["body"], "satin") if f.n[1] < .9]
    for f in body:  # the front face carries the name
        if abs(f.n[2] + 1) < 1e-6:
            f.decal = "name"
    F += body
    # Stylus assembly (removable stylus): lower block plus a grip nose.
    F += box(-6.4, 6.4, 1.9, 4.6, 3.4, 21.0, fin["sty"], "gloss", skip=("top", "bottom"))
    F += box(-2.4, 2.4, 2.3, 4.3, 0.9, 3.4, fin["sty"], "gloss")
    # Cantilever (aluminium tube) and diamond tip.
    F += sweep([(0, 2.15, 2.4), (0, 0.32, 0.05)], circle_section(.32, 8), fin["cant"], "metal")
    F += box(-.28, .28, 0.0, .45, -.28, .28, fin["tip"], "gloss")
    # Headshell (Technics style): 1.6 mm plate, 21 mm wide, tapering neck.
    hs = [(-9.3, -1.5), (9.3, -1.5), (10.5, -0.3), (10.5, 29), (5.6, 40.2), (-5.6, 40.2), (-10.5, 29), (-10.5, -0.3)]
    F += prism(hs, 18.6, 20.2, fin["hs"], "metal", fin["hs_top"])
    # Side lips under the plate edges.
    F += box(10.0, 10.5, 17.3, 18.6, -0.3, 29, fin["hs"], "metal")
    F += box(-10.5, -10.0, 17.3, 18.6, -0.3, 29, fin["hs"], "metal")
    # Mounting slots (1/2-inch spacing) and screw heads.
    for sx in (-6.35, 6.35):
        F.append(Face([(sx-1.2, 20.22, 8), (sx+1.2, 20.22, 8), (sx+1.2, 20.22, 21), (sx-1.2, 20.22, 21)],
                      (14, 13, 12), "matte", (sx, 0, 14), outline=False, bias=6.0))
        for f in sweep([(sx, 20.2, 14.5), (sx, 21.3, 14.5)], circle_section(2.0, 14), fin["screw"], "metal", up=(0, 0, 1)):
            f.bias = 9.0
            F.append(f)
    # Finger lift on the outer (right) side: a bent flat strip, out then up.
    # Finger lift: a solid bar straight out from the outer side, at 90 degrees.
    F += box(10.2, 24.3, 18.75, 20.05, 6.6, 9.8, fin["hs"], "metal")  # 14.1 mm long
    # Bayonet connector, locking ring (knurled), then the S-shaped arm.
    ay = 19.4
    rings = []
    for i in range(9):
        t = i / 8.0
        e = t * t * (3 - 2 * t)  # smoothstep
        rings.append(superellipse(5.6 + (4.0 - 5.6) * e, .8 + (4.0 - .8) * e, 10 + (2 - 10) * e, ay, 40.2 + 9.8 * t))
    F += loft(rings, fin["hs"], "metal")
    F += sweep([(0, ay, 50), (0, ay, 57)], circle_section(4.0, 32), fin["arm"], "metal", up=(0, 1, 0))
    F += sweep([(0, ay, 57), (0, ay, 61)], circle_section(5.0, 40), fin["ring"], "metal", up=(0, 1, 0))
    for i in range(30):  # fine knurl
        a = 2*math.pi*i/30
        F += sweep([(5.04*math.cos(a), ay+5.04*math.sin(a), 57.4), (5.04*math.cos(a), ay+5.04*math.sin(a), 60.6)],
                   circle_section(.13, 4), fin["ring_dark"], "metal", caps=False)
    # S-arm: leaves along the headshell axis, curves out, then heads for the
    # pivot at the 22-degree offset (effective length 230 mm).
    piv = (230*math.sin(math.radians(22)), ay, 230*math.cos(math.radians(22)))
    def herm(za, xa, ma, zb, xb, mb, n):
        L_ = zb - za
        out_ = []
        for i in range(n + 1):
            u_ = i / n
            h00 = 2*u_**3 - 3*u_**2 + 1; h10 = u_**3 - 2*u_**2 + u_; h01 = -2*u_**3 + 3*u_**2; h11 = u_**3 - u_**2
            out_.append((h00*xa + h10*L_*ma + h01*xb + h11*L_*mb, ay, za + L_*u_))
        return out_
    # First bend of the S (inward), then straight; the second bend lies far out of frame.
    # Tight first bend just behind the collar (radius ~50 mm), then straight.
    # Almost straight: a barely perceptible ~3 degree drift after the collar.
    bend = [(0.0, ay, 61.0)] + herm(64.0, 0.0, 0.0, 110.0, -1.1, -0.05, 10)
    straight = [(-1.1 - 0.05 * (z - 110.0), ay, z) for z in (140.0, 170.0, 185.0)]
    pts = bend + straight + herm(185.0, -1.1 - 0.05 * 75.0, -0.05, piv[2], piv[0], 0.6, 10)[1:]
    F += sweep(pts, circle_section(4.1, 28), fin["arm"], "metal")
    # Lead wires: cartridge back pins to the connector (R, G = right; W, B = left).
    for (px, py, col), k in zip(((-2.6, 8.6, (200, 66, 58)), (2.6, 8.6, (62, 142, 78)),
                                  (-2.6, 13.4, (233, 229, 220)), (2.6, 13.4, (61, 95, 168))), range(4)):
        path = [(px, py, 26.6), (px*.9, py+2, 31), (px*.6, py+4, 35.5), (-1.2 + k*.8, 17.4, 41)]
        F += sweep(path, circle_section(.38, 6), col, "gloss")
    return F

# ---------------- Camera, shading, SVG ----------------
class Cam:
    def __init__(self, target, yaw, pitch, dist, fov=28.0):
        cy, sy, cp, sp = math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
        self.pos = add(target, (dist*sy*cp, dist*sp, -dist*cy*cp))
        self.fwd = norm(sub(target, self.pos))
        self.right = norm(cross((0, 1, 0), self.fwd))
        self.up = cross(self.fwd, self.right)
        self.f = 1.0/math.tan(math.radians(fov/2))
    def cs(self, p):
        d = sub(p, self.pos)
        return (dot(d, self.right), dot(d, self.up), dot(d, self.fwd))
    def proj(self, p, S=400.0):
        x, y, z = self.cs(p)
        z = max(z, 1e-3)
        return (S*self.f*x/z, -S*self.f*y/z)

LIGHT = norm((-0.55, 0.85, -0.6))
FILL = norm((0.8, 0.3, -0.2))

def shade(face, cam):
    n = face.n
    v = norm(sub(cam.pos, centroid(face.pts)))
    d = max(0.0, dot(n, LIGHT)); f2 = max(0.0, dot(n, FILL))
    amb, kd, kf, ks, sh = {"matte": (.26, .78, .10, .05, 8), "satin": (.24, .78, .12, .26, 20),
                           "gloss": (.28, .72, .12, .50, 40), "metal": (.18, .74, .16, .85, 22)}[face.mat]
    h = norm(add(LIGHT, v))
    spec = ks * max(0.0, dot(n, h)) ** sh
    k = amb + kd*d + kf*f2
    r, g, b = face.color
    return tuple(int(min(255, c*k + 255*spec)) for c in (r, g, b))

def wordmark_svg(size, fill, x=0.0, y=0.0):
    r = .30*size
    return (f'<text x="{x:.3f}" y="{y:.3f}" font-family="\'IBM Plex Sans Condensed\', \'Arial Narrow\', sans-serif" '
            f'font-weight="600" font-size="{size:.3f}" fill="{fill}" text-anchor="middle">2<tspan font-size="{.6*size:.3f}" dy="-{r:.3f}">nd</tspan>'
            f'<tspan font-size="{size:.3f}" dy="{r:.3f}">Pressing</tspan></text>')

def decal_svg(face, cam, fin, S):
    """Name and model line on the cartridge front, mapped by an affine matrix."""
    # Front face local frame: origin at front centre-top of the body.
    o = (0.0, 18.6, 2.6); ex = (1.0, 0, 0); ey = (0, -1.0, 0)
    P = lambda p: cam.proj(p, S)
    O, X, Y = P(o), P(add(o, ex)), P(add(o, ey))
    a, b = X[0]-O[0], X[1]-O[1]; c, d = Y[0]-O[0], Y[1]-O[1]
    m = f'matrix({a:.4f} {b:.4f} {c:.4f} {d:.4f} {O[0]:.2f} {O[1]:.2f})'
    # Name only, centred on the 14 mm tall front face.
    return f'<g transform="{m}">{wordmark_path(2.9, fin["text"], 0, 8.0) or wordmark_svg(2.9, fin["text"], 0, 8.0)}</g>'

def full_record(cam, S, vb, uid):
    """A record large enough to fill the whole frame: concentric grooves round
    the spindle (to the left, at x = -100), track bands, a glossy streak where
    the key light reflects, and grooves thinning out with distance."""
    cxr = -100.0
    x0, y0, w, h = vb
    m = .03
    def inside(q):
        return x0 - w*m <= q[0] <= x0 + w*(1+m) and y0 - h*m <= q[1] <= y0 + h*(1+m)
    out = [f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" fill="url(#{uid}-rec)"/>']
    near = cam.cs((0, 0, 0))[2]
    GLARE_T, GLARE_W = 0.98, 0.12  # band angle (rad, from the spindle) and half-width
    def glare(t):
        d = (t - GLARE_T) / GLARE_W
        return math.exp(-d * d)
    # Soft sheen along the band: a broad blurred streak on the surface.
    q1 = cam.proj((cxr + 95 * math.cos(GLARE_T), 0, 95 * math.sin(GLARE_T)), S)
    q2 = cam.proj((cxr + 330 * math.cos(GLARE_T), 0, 330 * math.sin(GLARE_T)), S)
    out.append(f'<line x1="{q1[0]:.1f}" y1="{q1[1]:.1f}" x2="{q2[0]:.1f}" y2="{q2[1]:.1f}" stroke="#EDE6D6" '
               f'stroke-opacity=".24" stroke-width="{w*.12:.1f}" stroke-linecap="round" filter="url(#{uid}-glow)"/>')
    R, k = 20.0, 0
    while R < 560:
        k += 1
        gap = (k % 26 == 0)
        R += 3.4 if gap else .78
        n = max(24, int(2.7 * R / 2.0))
        run = []
        def flush(run):
            for i in range(0, len(run) - 1, 9):
                chunk = run[i:i + 10]
                if len(chunk) < 2:
                    continue
                spec = sum(c[1] for c in chunk) / len(chunk) + .9 * sum(c[3] for c in chunk) / len(chunk)
                fade = min(1.0, near / max(chunk[0][2], 1.0)) ** .9
                op = ((.03 if gap else .24) + .7 * spec) * fade
                if op < .025:
                    continue
                pts = " ".join(f"{q[0]:.1f},{q[1]:.1f}" for q, _, _, _ in chunk)
                out.append(f'<polyline points="{pts}" fill="none" stroke="#A39F95" stroke-opacity="{min(op, .95):.3f}" stroke-width="{.3 + .4*spec:.2f}"/>')
        for i in range(n + 1):
            t = -1.25 + 2.85 * i / n
            p = (cxr + R*math.cos(t), 0, R*math.sin(t))
            zc = cam.cs(p)[2]
            q = cam.proj(p, S) if zc > 1 else None
            if q is None or not inside(q):
                flush(run); run = []
                continue
            v = norm(sub(cam.pos, p)); hv = norm(add(LIGHT, v))
            radial = (math.cos(t), 0, math.sin(t))
            nrm = norm(add((0, 1, 0), mul(radial, .35 if k % 2 else -.35)))
            run.append((q, max(0.0, dot(nrm, hv)) ** 60, zc, glare(t)))
        flush(run)
    return out

def render(fin, cam, S=400.0, pad=0.08, focus=None, aspect=None, uid="v", grooves=False, full=False):
    faces = model(fin)
    vis = []
    for f in faces:
        c = centroid(f.pts)
        if not f.nocull and dot(f.n, sub(cam.pos, c)) <= 0:
            continue
        depth = cam.cs(c)[2] - f.bias
        vis.append((depth, f))
    vis.sort(key=lambda t: -t[0])
    # Frame: cartridge + headshell + lift + first part of the arm.
    fpts = []
    for f in faces:
        for p in f.pts:
            if p[2] < (focus or 70):
                fpts.append(cam.proj(p, S))
    fpts.append(cam.proj((0, 0, 0), S))
    xs, ys = [p[0] for p in fpts], [p[1] for p in fpts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    w, h = x1-x0, y1-y0
    x0 -= w*pad; x1 += w*pad; y0 -= h*pad; y1 += h*(pad+.10)
    if aspect:
        W, H = x1-x0, y1-y0
        if W/H < aspect:
            extra = H*aspect - W; x0 -= extra/2; x1 += extra/2
        else:
            extra = W/aspect - H; y0 -= extra/2; y1 += extra/2
    vb = (x0, y0, x1-x0, y1-y0)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb[0]:.1f} {vb[1]:.1f} {vb[2]:.1f} {vb[3]:.1f}">']
    # Record: glossy black with concentric grooves round the spindle at x = -100.
    cxr = -100.0
    tip = cam.proj((0, 0, 0), S)
    rr = max(vb[2], vb[3])*.62
    out.append(f'''<defs>
<radialGradient id="{uid}-fade" cx=".5" cy=".52" r=".48">
<stop offset=".45" stop-color="#fff"/><stop offset=".78" stop-color="#fff" stop-opacity=".5"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></radialGradient>
<mask id="{uid}-m"><rect x="{vb[0]:.1f}" y="{vb[1]:.1f}" width="{vb[2]:.1f}" height="{vb[3]:.1f}" fill="url(#{uid}-fade)"/></mask>
<filter id="{uid}-blur" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="{vb[2]*.012:.2f}"/></filter>
<linearGradient id="{uid}-gx" x1="0" y1="0" x2="1" y2="0"><stop offset=".86" stop-color="#fff"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
<linearGradient id="{uid}-gy" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset=".16" stop-color="#fff"/></linearGradient>
<filter id="{uid}-glow" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="{vb[2]*.025:.2f}"/></filter>
<filter id="{uid}-soft" x="-10%" y="-10%" width="120%" height="120%"><feGaussianBlur stdDeviation="{vb[2]*.004:.2f}"/></filter>
<linearGradient id="{uid}-rec" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#24221F"/><stop offset="1" stop-color="#121110"/></linearGradient>
<mask id="{uid}-ex"><rect x="{vb[0]:.1f}" y="{vb[1]:.1f}" width="{vb[2]:.1f}" height="{vb[3]:.1f}" fill="url(#{uid}-gx)"/></mask>
<mask id="{uid}-ey"><rect x="{vb[0]:.1f}" y="{vb[1]:.1f}" width="{vb[2]:.1f}" height="{vb[3]:.1f}" fill="url(#{uid}-gy)"/></mask>
</defs>''')
    out.append('<g><g>' if full else f'<g mask="url(#{uid}-ex)"><g mask="url(#{uid}-ey)">')
    disc = [cam.proj((cxr + 150*math.cos(t), 0, 150*math.sin(t)), S) for t in [2*math.pi*i/180 for i in range(180)]
            if cam.cs((cxr + 150*math.cos(2*math.pi*0), 0, 0))[2] > 0]
    if full:
        g = ['<g>'] + full_record(cam, S, vb, uid)
        radii_ok = False
    else:
        g = [f'<g mask="url(#{uid}-m)">']
        g.append('<polygon points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in disc) + '" fill="#171614"/>')
        radii_ok = True
    # Grooves with a glossy streak where the key light reflects toward the camera.
    # Groove rings; with grooves=True, a fine visible pitch and the slightly
    # wider, smoother bands between tracks, as on a real LP.
    radii = []
    if not radii_ok:
        pass
    elif grooves:
        R = 52.0
        k = 0
        while R < 156:
            k += 1
            gap = (k % 26 == 0)
            R += 3.4 if gap else .78
            radii.append((R, gap))
    elif radii_ok:
        radii = [(52 + k*1.45, False) for k in range(72)]
    for k, (R, gap) in enumerate(radii):
        seg = []
        nseg = 120 if grooves else 240
        for i in range(0, nseg + 1):
            t = -math.pi/2 + math.pi*i/nseg
            p = (cxr + R*math.cos(t), 0, R*math.sin(t))
            if abs(p[0]) > 70 or p[2] < -60 or p[2] > 160:
                continue
            if cam.cs(p)[2] <= 1: continue
            v = norm(sub(cam.pos, p)); hv = norm(add(LIGHT, v))
            # Groove walls tilt the reflection: use the radial direction.
            radial = norm((math.cos(t), 0, math.sin(t)))
            nrm = norm(add((0, 1, 0), mul(radial, .35 if k % 2 else -.35)))
            spec = max(0.0, dot(nrm, hv)) ** 60
            seg.append((cam.proj(p, S), spec))
        for (a, s1), (b, s2) in zip(seg, seg[1:]):
            s = (s1+s2)/2
            if grooves:
                op = (.04 if gap else .28) + .7*s
                g.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="#A39F95" stroke-opacity="{min(op, .95):.2f}" stroke-width="{.3 + .4*s:.2f}"/>')
            else:
                op = .10 + .80*s
                g.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" stroke="#8A867D" stroke-opacity="{op:.2f}" stroke-width="{.35 + .5*s:.2f}"/>') if op > .12 else None
    g.append('</g>')
    out += [x for x in g if x]
    if full:
        refl = []
        for f in faces:
            if max(p[2] for p in f.pts) > 42:  # cartridge and headshell only
                continue
            mp = [(p[0], -p[1], p[2]) for p in f.pts]
            mn = (f.n[0], -f.n[1], f.n[2])
            c = centroid(mp)
            if dot(mn, sub(cam.pos, c)) <= 0:
                continue
            col = shade(f, cam)
            refl.append((cam.cs(c)[2], mp, '#%02x%02x%02x' % col))
        refl.sort(key=lambda t: -t[0])
        out.append(f'<g opacity=".16" filter="url(#{uid}-soft)">')
        for _, mp, col in refl:
            q = [cam.proj(p, S) for p in mp]
            out.append('<polygon points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in q) + f'" fill="{col}"/>')
        out.append('</g>')
    # Contact shadow.
    sh = [cam.proj(p, S) for p in [(-10, 0, 0), (10, 0, 0), (10, 0, 28), (-10, 0, 28)]]
    out.append('<polygon points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in sh) + f'" fill="#000" fill-opacity=".55" filter="url(#{uid}-blur)"/>')
    for depth, f in vis:
        pts = [cam.proj(p, S) for p in f.pts]
        col = shade(f, cam)
        hexc = '#%02x%02x%02x' % col
        stroke = f' stroke="{hexc}" stroke-width=".25" stroke-linejoin="round"' if f.outline else ''
        out.append('<polygon points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + f'" fill="{hexc}"{stroke}/>')
        if f.decal == "name":
            out.append(decal_svg(f, cam, fin, S))
    if full:
        # Light catching the headshell's front and outer top edges.
        edge = [cam.proj(p, S) for p in [(-9.3, 20.2, -1.5), (9.3, 20.2, -1.5), (10.5, 20.2, -0.3), (10.5, 20.2, 29), (5.6, 20.2, 40.2)]]
        out.append('<polyline points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in edge)
                   + '" fill="none" stroke="#F4F1EA" stroke-opacity=".55" stroke-width="1.1" stroke-linejoin="round" stroke-linecap="round"/>')
    out.append('</g></g></svg>')  # close the edge fades
    return "\n".join(out)

FIN_A1 = dict(body=(40, 38, 35), sty=(242, 169, 59), cant=(214, 210, 200), tip=(250, 248, 240),
              hs=(176, 171, 160), hs_top=(196, 191, 180), screw=(150, 146, 138), arm=(196, 192, 184),
              ring=(170, 166, 158), ring_dark=(110, 106, 98), text="#EBE6DA", sub="#A9A395")
FIN_A2 = dict(body=(214, 209, 198), sty=(44, 42, 38), cant=(214, 210, 200), tip=(242, 169, 59),
              hs=(52, 50, 46), hs_top=(70, 67, 62), screw=(150, 146, 138), arm=(52, 50, 46),
              ring=(44, 42, 38), ring_dark=(20, 19, 17), text="#2C2A26", sub="#6A6455")
