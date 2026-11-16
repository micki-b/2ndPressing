"""Listening demo: a lifelike, working replica of the 2ndPressing faceplate
that plays a clip as repaired / original (bypass) / clicks only, in sync,
with the meter and strip chart driven by the engine's real repairs.

  python3 build.py OUT_DIR          # review page: OUT_DIR/listening-demo.html
  python3 build.py OUT_DIR --site   # secondpressing.com: OUT_DIR/index.html + listen.js
Reads OUT_DIR/demo.json (from prepare.py). The page expects the audio next
to it in audio/*.mp4. --site makes a full document that uses the site's own
fonts and an external script (the site's CSP allows no inline scripts or
third-party fonts).
"""
import html as H, json, math, pathlib, sys
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
SITE = "--site" in sys.argv
D = pathlib.Path(ARGS[0] if ARGS else ".")
data = json.loads((D / "demo.json").read_text())
ROOT = pathlib.Path(__file__).resolve().parents[2]          # website/
icon = (ROOT / "public" / "assets" / "header-icon.svg").read_text()
KHZ = f'{data["sr"] / 1000:.1f}'
LAT = f'{data["latency_s"]:.2f} s' if data["latency_s"] >= 1 else f'{data["latency_s"] * 1000:.0f} ms'
PROT = {"Off": 0, "Standard": 1, "Extended": 2}.get(data["protection"], 2)
def mmss(t):
    m, s = int(t // 60), t % 60
    return f"{m}:{int(s):02d}" if float(s).is_integer() else f"{m}:{s:04.1f}"

C = dict(PANEL_TOP="#D4CFC3", PANEL_BOT="#C6C0B2", BAND="#3C3A36", BAND_TEXT="#EBE6DA", BAND_DIM="#A9A395",
         LEGEND="#3C3A36", GLASS_TOP="#1B1A17", GLASS_BOT="#26241F", AMBER="#F2A93B", AMBER_DIM="#C98A2E",
         AMBER_FAINT="#8C7547", KEY_TOP="#ECE8DE", KEY_BOT="#CFC9BB", KEY_ON_TOP="#D9D4C7", KEY_ON_BOT="#BDB6A6",
         KEY_EDGE="#8C8577", KEY_TEXT="#2C2A26", KEY_SUB="#6A6455", LAMP_OFF="#6D6553", RED="#E2553F",
         GREEN="#7EC27A", LED_OFF="#4A463D", PAPER="#F4ECD4", PAPER_EDGE="#A89F8A", GRID="#C9B98F", INK="#2F3B6B",
         PAPER_LABEL="#8A7A52", FOOT="#BDB6A7", KNOB_EDGE="#34322F", KNOB_MID="#4B4945", KNOB_HI="#7D7A73",
         KNOB_CAP="#55534E", KNOB_LINE="#F0EBDF")
COND = "'IBM Plex Sans Condensed','Arial Narrow',sans-serif"
MONO = "'IBM Plex Mono',ui-monospace,monospace"

def t(x, y, s, txt, fam, fill, anchor="start", weight=None, extra=""):
    w = f' font-weight="{weight}"' if weight else ""
    return f'<text x="{x}" y="{y}" font-family="{fam}" font-size="{s}"{w} fill="{fill}" text-anchor="{anchor}" {extra}>{txt}</text>'

def lerp(a, b, k):
    a = [int(a[i:i+2], 16) for i in (1, 3, 5)]; b = [int(b[i:i+2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * k) for x, y in zip(a, b))

def brand(x, y, s, fill):
    r = round(.30 * s, 2)
    return (f'<text x="{x}" y="{y}" font-family="{COND}" font-weight="600" font-size="{s}" fill="{fill}">2'
            f'<tspan font-size="{.6*s:.1f}" dy="-{r}">nd</tspan><tspan font-size="{s}" dy="{r}">Pressing</tspan>'
            f'<tspan font-size="{s}" fill="{C["BAND_DIM"]}" dx="8">DECLICKER</tspan></text>')

def key(id_, x, y, w, h, label, sub, lamp_on, role):
    """A latching key; JS toggles the 'on' class (pressed look + lit lamp)."""
    cx = x + w / 2
    ly = y + h / 2 + (2 if sub else 4)
    subt = t(cx, y + h - 6, 8.5, sub, MONO, C["KEY_SUB"], "middle") if sub else ""
    return f'''<g class="key" id="{id_}" data-role="{role}" data-lamp="{lamp_on}" tabindex="0" role="button" aria-pressed="false" aria-label="{label} {sub}">
  <rect x="{x}" y="{y+3}" width="{w}" height="{h}" rx="3" fill="{C['KEY_EDGE']}"/>
  <g class="cap">
    <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="url(#keyg)" stroke="{C['KEY_EDGE']}" stroke-width="1"/>
    <rect class="lamp" x="{cx-9}" y="{y+7}" width="18" height="4" rx="1" fill="{C['LAMP_OFF']}"/>
    {t(cx, ly, 11, label, COND, C['KEY_TEXT'], "middle", 600, 'dominant-baseline="middle"')}
    {subt}
  </g>
</g>'''

def faceplate():
    s = []
    s.append(f'''<defs>
<linearGradient id="panel" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{C['PANEL_TOP']}"/><stop offset="1" stop-color="{C['PANEL_BOT']}"/></linearGradient>
<linearGradient id="glass" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{C['GLASS_TOP']}"/><stop offset="1" stop-color="{C['GLASS_BOT']}"/></linearGradient>
<linearGradient id="keyg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{C['KEY_TOP']}"/><stop offset="1" stop-color="{C['KEY_BOT']}"/></linearGradient>
<linearGradient id="keyon" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{C['KEY_ON_TOP']}"/><stop offset="1" stop-color="{C['KEY_ON_BOT']}"/></linearGradient>
</defs>''')
    s.append(f'<rect width="520" height="470" fill="url(#panel)"/>')
    # Band.
    s.append(f'<rect width="520" height="54" fill="{C["BAND"]}"/>')
    s.append(icon.replace('<svg xmlns="http://www.w3.org/2000/svg" ', '<svg x="18" y="9" width="36" height="36" ', 1))
    s.append(brand(60, 34, 18, C["BAND_TEXT"]))
    s.append(t(432, 33, 9.5, "CLICK REPAIR · STEREO", MONO, C["BAND_DIM"], "end"))
    s.append(f'''<g opacity=".38" aria-disabled="true" style="cursor:not-allowed"><title>Not available in this demo</title><rect x="444" y="16" width="52" height="26" rx="3" fill="#000" fill-opacity=".35"/>
<rect x="444" y="14" width="52" height="26" rx="3" fill="url(#keyg)" stroke="{C['KEY_EDGE']}"/>
{t(470, 27, 10, "INFO", COND, C['KEY_TEXT'], "middle", 600, 'dominant-baseline="middle"')}</g>''')
    # Sensitivity readout (fixed at the setting the demo was processed with).
    s.append(t(26, 80, 10, "SENSITIVITY", COND, C["LEGEND"], weight=600, extra='letter-spacing=".3"'))
    s.append(f'<rect x="26" y="90" width="200" height="56" fill="url(#glass)"/>')
    s.append(t(34, 103, 8.5, f'THRESHOLD {data["threshold"]:.2f}', MONO, C["AMBER_FAINT"]))
    s.append(t(200, 133, 36, str(data["sensitivity"]), MONO, C["AMBER"], "end", 500))
    s.append(t(205, 132, 13, "%", MONO, C["AMBER_DIM"]))
    # Knob and scale.
    cx, cy = 126, 224
    def d(deg, r):
        a = math.radians(deg - 90); return cx + math.cos(a) * r, cy + math.sin(a) * r
    for i in range(21):
        deg = -135 + i * 13.5; major = i % 5 == 0
        x1, y1 = d(deg, 46); x2, y2 = d(deg, 55 if major else 51)
        s.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{C["LEGEND"]}" stroke-width="{1.3 if major else .8}"/>')
        if major:
            tx, ty = d(deg, 66); s.append(t(f"{tx:.1f}", f"{ty:.1f}", 8.5, str(i * 5), MONO, C["LEGEND"], "middle", extra='dominant-baseline="middle"'))
    s.append(f'<circle cx="{cx+2}" cy="{cy+3}" r="40" fill="#000" fill-opacity=".27"/><circle cx="{cx}" cy="{cy}" r="40" fill="{C["KNOB_EDGE"]}"/>')
    for k in range(10):
        q = k / 9
        s.append(f'<circle cx="{cx-5*q:.2f}" cy="{cy-7*q:.2f}" r="{38*(1-q*.55):.2f}" fill="{lerp(C["KNOB_MID"], C["KNOB_HI"], q*.8)}"/>')
    s.append(f'<circle cx="{cx}" cy="{cy}" r="30" fill="{C["KNOB_CAP"]}" stroke="{C["KNOB_EDGE"]}"/>')
    deg = -135 + 270 * data["sensitivity"] / 100; x1, y1 = d(deg, 13); x2, y2 = d(deg, 27)
    s.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{C["KNOB_LINE"]}" stroke-width="2.6" stroke-linecap="round"/>')
    sens = data["sensitivity"]
    s.append(f'<circle id="knob" cx="{cx}" cy="{cy}" r="44" fill="transparent" class="knob-hit" tabindex="0" role="img" aria-label="Sensitivity {sens} %. This demo was processed at {sens}%"><title>This demo was processed at {sens}%</title></circle>')
    s.append(f'<text id="knobnote" x="{cx}" y="294" font-family="{MONO}" font-size="8.5" fill="{C["KEY_SUB"]}" text-anchor="middle" opacity="0">This demo was processed at {sens}%</text>')
    # Clicks-per-minute meter.
    s.append(t(262, 80, 10, "CLICKS PER MINUTE", COND, C["LEGEND"], weight=600, extra='letter-spacing=".3"'))
    s.append(f'<rect x="262" y="90" width="232" height="56" fill="url(#glass)"/>')
    s.append(t(270, 103, 8.5, "PRESS PLAY", MONO, C["AMBER_FAINT"], extra='id="zone"'))
    s.append(t(418, 133, 36, "--", MONO, C["AMBER"], "end", 500, 'id="rate"'))
    s.append(t(423, 131, 12, "/min", MONO, C["AMBER_DIM"]))
    n, x0, w, gap = 30, 262.0, 232.0, 2.0
    sw = (w - gap * (n - 1)) / n
    for i in range(n):
        s.append(f'<rect class="led" x="{x0 + i*(sw+gap):.2f}" y="156" width="{sw:.2f}" height="14" rx="1" fill="{C["LED_OFF"]}"/>')
    for v in (10, 100, 1000):
        x = x0 + w * math.log(1 + v) / math.log(1 + 3000)
        s.append(f'<line x1="{x:.1f}" y1="172" x2="{x:.1f}" y2="177" stroke="{C["LEGEND"]}"/>' + t(f"{x:.1f}", 187, 8.5, str(v), MONO, C["LEGEND"], "middle"))
    s.append(f'<rect x="262" y="196" width="232" height="70" rx="2" fill="{C["PAPER"]}"/>')
    x = 272
    while x < 494:
        s.append(f'<line x1="{x}" y1="196" x2="{x}" y2="266" stroke="{C["GRID"]}" stroke-width=".5"/>'); x += 10
    y = 204
    while y < 266:
        s.append(f'<line x1="262" y1="{y}" x2="494" y2="{y}" stroke="{C["GRID"]}" stroke-width=".5"/>'); y += 8
    s.append(f'<rect x="262.5" y="196.5" width="231" height="69" rx="2" fill="none" stroke="{C["PAPER_EDGE"]}"/>')
    s.append(t(268, 206, 8, "60 s chart", MONO, C["PAPER_LABEL"]))
    s.append(f'<polyline id="trace" points="" fill="none" stroke="{C["INK"]}" stroke-width="1.2" stroke-linejoin="round"/>')
    # Keys.
    s.append(t(26, 310, 10, "LISTEN", COND, C["LEGEND"], weight=600, extra='letter-spacing=".3"'))
    lw = (200 - 5) / 2
    s.append(key("k-rep", 26, 320, lw, 46, "REPAIRED", "audio", C["AMBER"], "monitor"))
    s.append(key("k-clk", 26 + lw + 5, 320, lw, 46, "CLICKS", "only", C["AMBER"], "monitor"))
    s.append(key("k-byp", 26, 376, 200, 40, "BYPASS", "", C["RED"], "bypass"))
    s.append(t(262, 310, 10, "PERCUSSION PROTECTION", COND, C["LEGEND"], weight=600, extra='letter-spacing=".3"'))
    for i, (lab, sub) in enumerate((("OFF", "32 ms"), ("STANDARD", "≈ 1 s"), ("EXTENDED", "≈ 2 s"))):
        s.append(key(f"k-p{i}", 262 + i * 79, 320, 74, 46, lab, sub, C["AMBER"], "protection"))
    s.append(f'<text id="protnote" x="378" y="392" font-family="{MONO}" font-size="8.5" fill="{C["KEY_SUB"]}" text-anchor="middle" opacity="0">This demo was processed with {data["protection"]}</text>')
    # Status band.
    s.append(f'<rect x="0" y="428" width="520" height="42" fill="{C["FOOT"]}"/>')
    s.append(t(26, 452, 9.5, f"LATENCY {LAT}    {KHZ} kHz    0 repaired", MONO, C["LEGEND"], extra='id="status" xml:space="preserve" style="white-space:pre"'))
    return "\n".join(s)

def timeline():
    env, cenv, dur = data["env"], data["cenv"], data["duration"]
    W, H = 720, 64
    bars = []
    for i, v in enumerate(env):
        h = max(.6, v * (H - 8) / 2)
        bars.append(f'<rect x="{i}" y="{H/2 - h:.1f}" width="1" height="{2*h:.1f}"/>')
    marks = "".join(f'<rect x="{r / dur * W:.1f}" y="2" width="1" height="5"/>' for r in data["regions"])
    return f'''<svg id="tl" viewBox="0 0 {W} {H}" preserveAspectRatio="none" role="slider" aria-label="Playback position" tabindex="0">
<rect width="{W}" height="{H}" fill="#1E1D1A"/>
<g fill="#6E695F">{"".join(bars)}</g>
<g fill="{C['AMBER']}" opacity=".85">{marks}</g>
<rect id="played" x="0" y="0" width="0" height="{H}" fill="#F2A93B" fill-opacity=".10"/>
<line id="head" x1="0" y1="0" x2="0" y2="{H}" stroke="#F4ECD4" stroke-width="1.5"/>
</svg>'''

JS = r'''
const DATA = __DATA__;
const $ = (q) => document.querySelector(q);
const ZONE = (r) => r < 100 ? ["#7EC27A", "clean"] : r < 600 ? ["#F2A93B", "moderate"] : ["#E2553F", "heavy"];
const FRAC = (r) => Math.min(1, Math.max(0, Math.log(1 + Math.max(0, r)) / Math.log(1 + 3000)));
const leds = [...document.querySelectorAll(".led")];
const LEDV = leds.map((_, i) => Math.exp((i + 0.5) / leds.length * Math.log(1 + 3000)) - 1);

let ctx = null, buffers = null, srcs = [], gains = [], startedAt = 0, offset = 0, playing = false, loading = null;
const state = { monitor: "rep", bypass: false };
const FILES = ["audio/original.mp4", "audio/repaired.mp4", "audio/clicks.mp4"];

function targetIndex() { return state.bypass ? 0 : (state.monitor === "rep" ? 1 : 2); }
function setGains(ramp = 0.010) {
  if (!ctx) return;
  const ti = targetIndex(), now = ctx.currentTime;
  gains.forEach((g, i) => { g.gain.cancelScheduledValues(now); g.gain.setValueAtTime(g.gain.value, now); g.gain.linearRampToValueAtTime(i === ti ? 1 : 0, now + ramp); });
}
async function load() {
  if (loading) return loading;
  ctx = new (window.AudioContext || window.webkitAudioContext)();
  // Decode offline: some browsers hold decodeAudioData on a context that is
  // still suspended (no click yet), and AudioBuffers work in any context.
  const Off = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  const dec = Off ? new Off(2, 1, ctx.sampleRate) : ctx;
  $("#msg").textContent = "Loading audio…";
  loading = Promise.all(FILES.map(f => fetch(f).then(r => { if (!r.ok) throw new Error(f); return r.arrayBuffer(); })
      .then(b => new Promise((ok, err) => dec.decodeAudioData(b, ok, err)))))
    .then(b => { buffers = b; gains = b.map(() => { const g = ctx.createGain(); g.gain.value = 0; g.connect(ctx.destination); return g; }); $("#msg").textContent = ""; })
    .catch(e => { $("#msg").textContent = "Could not load the audio (" + e.message + ")."; throw e; });
  return loading;
}
function now() { return playing ? (ctx.currentTime - startedAt + offset) : offset; }
function startAt(pos) {
  stopSources();
  const t0 = ctx.currentTime + 0.03;
  srcs = buffers.map((b, i) => { const s = ctx.createBufferSource(); s.buffer = b; s.loop = $("#loop").checked; s.connect(gains[i]); s.start(t0, pos); return s; });
  startedAt = t0; offset = pos; playing = true;
  gains.forEach((g, i) => g.gain.setValueAtTime(i === targetIndex() ? 1 : 0, t0));
  srcs[0].onended = () => { if (playing && !$("#loop").checked) { playing = false; offset = 0; render(); updatePlay(); } };
}
function stopSources() { srcs.forEach(s => { try { s.onended = null; s.stop(); } catch (e) {} }); srcs = []; }
async function togglePlay() {
  await load();
  if (ctx.state === "suspended") await ctx.resume();
  if (playing) { offset = pos(); stopSources(); playing = false; } else { startAt(offset >= DATA.duration - 0.05 ? 0 : offset); }
  updatePlay();
}
function pos() { let t = now(); if ($("#loop").checked) t = t % DATA.duration; return Math.min(Math.max(t, 0), DATA.duration); }
async function seek(t) {
  t = Math.min(Math.max(t, 0), DATA.duration - 0.01);
  if (playing) { await load(); startAt(t); } else { offset = t; }
  render();
}
function updatePlay() { $("#play").classList.toggle("on", playing); $("#play .lbl").textContent = playing ? "PAUSE" : "PLAY"; $("#play").setAttribute("aria-pressed", playing); }

function setKey(id, on) { const k = document.getElementById(id); k.classList.toggle("on", on); k.setAttribute("aria-pressed", on); k.querySelector(".lamp").setAttribute("fill", on ? k.dataset.lamp : "#6D6553"); }
function syncKeys() { setKey("k-rep", state.monitor === "rep"); setKey("k-clk", state.monitor === "clk"); setKey("k-byp", state.bypass); [0, 1, 2].forEach(i => setKey("k-p" + i, i === DATA.prot)); $("#hearing").textContent = state.bypass ? "Original (bypass)" : state.monitor === "rep" ? "Repaired" : "Clicks only: what was removed"; }
function press(id) {
  if (id === "k-rep") state.monitor = "rep";
  else if (id === "k-clk") state.monitor = "clk";
  else if (id === "k-byp") state.bypass = !state.bypass;
  else if (id.startsWith("k-p")) { const n = $("#protnote"); n.setAttribute("opacity", 1); clearTimeout(n._t); n._t = setTimeout(() => n.setAttribute("opacity", 0), 2200); return; }
  syncKeys(); setGains();
}
function note(id) { const n = document.getElementById(id); n.setAttribute("opacity", 1); clearTimeout(n._t); n._t = setTimeout(() => n.setAttribute("opacity", 0), 2200); }
const knob = document.getElementById("knob");
["pointerdown", "wheel"].forEach(ev => knob.addEventListener(ev, e => { e.preventDefault(); note("knobnote"); }, { passive: false }));
knob.addEventListener("keydown", e => { if (["Enter", " ", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.key)) { e.preventDefault(); note("knobnote"); } });
document.querySelectorAll(".key").forEach(k => {
  k.addEventListener("click", () => press(k.id));
  k.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); press(k.id); } });
});

function fmt(t) { const m = Math.floor(t / 60), s = t - m * 60; return m + ":" + (s < 10 ? "0" : "") + s.toFixed(1); }
function render() {
  const t = pos(), R = DATA.regions;
  let n = 0; while (n < R.length && R[n] <= t) n++;
  let rate = null;
  if (t >= 1) { const w0 = Math.max(0, t - 10); let c = 0; for (const r of R) if (r > w0 && r <= t) c++; rate = c / ((t - w0) / 60); }
  if (rate === null) { $("#rate").textContent = "--"; $("#zone").textContent = playing ? "LISTENING…" : "PRESS PLAY"; }
  else { $("#rate").textContent = Math.round(rate); $("#zone").textContent = ZONE(rate)[1].toUpperCase(); }
  const lit = rate === null ? 0 : Math.round(FRAC(rate) * leds.length);
  leds.forEach((l, i) => l.setAttribute("fill", i < lit ? ZONE(LEDV[i])[0] : "#4A463D"));
  const pts = [];
  for (let s = 1; s <= Math.floor(t); s++) { const w0 = Math.max(0, s - 10); let c = 0; for (const r of R) if (r > w0 && r <= s) c++; pts.push(c / ((s - w0) / 60)); }
  const hist = pts.slice(-60);
  $("#trace").setAttribute("points", hist.map((r, i) => (266 + 224 * (i / 59)).toFixed(1) + "," + (262 - 54 * FRAC(r)).toFixed(1)).join(" "));
  $("#status").textContent = DATA.statusPrefix + n + " repaired";
  const x = t / DATA.duration * 720;
  $("#head").setAttribute("x1", x); $("#head").setAttribute("x2", x); $("#played").setAttribute("width", x);
  $("#time").textContent = fmt(t) + " / " + fmt(DATA.duration);
}
function loop() { render(); requestAnimationFrame(loop); }

const tl = $("#tl");
function tlSeek(e) { const r = tl.getBoundingClientRect(); seek((e.clientX - r.left) / r.width * DATA.duration); }
let drag = false;
tl.addEventListener("pointerdown", e => { drag = true; tl.setPointerCapture(e.pointerId); tlSeek(e); });
tl.addEventListener("pointermove", e => { if (drag) tlSeek(e); });
tl.addEventListener("pointerup", () => { drag = false; });
tl.addEventListener("keydown", e => { if (e.key === "ArrowRight") seek(pos() + 2); if (e.key === "ArrowLeft") seek(pos() - 2); });
$("#play").addEventListener("click", togglePlay);
document.querySelectorAll("[data-jump]").forEach(b => b.addEventListener("click", async () => { await seek(+b.dataset.jump); if (!playing) togglePlay(); }));
$("#loop").addEventListener("change", () => { if (playing) startAt(pos()); });
document.addEventListener("keydown", e => {
  if (e.target.closest && e.target.closest("input,button,[role=button],[role=slider]")) return;
  if (e.code === "Space") { e.preventDefault(); togglePlay(); }
  if (e.key === "b" || e.key === "B") press("k-byp");
  if (e.key === "c" || e.key === "C") press(state.monitor === "rep" ? "k-clk" : "k-rep");
});
syncKeys(); render(); requestAnimationFrame(loop);
// Self-test (#selftest): decode all three files and report their lengths.
if (location.hash === "#selftest") load().then(() => { document.title = "SELFTEST OK " + buffers.map(b => b.duration.toFixed(3) + "s/" + b.numberOfChannels + "ch").join(" "); }).catch(e => { document.title = "SELFTEST FAIL " + e; });
'''

PRIVATE = (f'<span class="private">Private prototype: {H.escape(data["credit"]) or "this recording"}, for review only. Not for the public site.</span>'
           if data.get("private") else "")
JUMPS = "".join(f'<button class="jump" type="button" data-jump="{c["t"]}">{mmss(c["t"])} · {H.escape(c["label"])}</button>'
                for c in data.get("chapters", []))
NOTE = f'<li>{data["note"]}</li>' if data.get("note") else ""
CREDIT = f' Recording: {H.escape(data["credit"])}.' if data.get("credit") and not data.get("private") else ""
ASIDE = f'<aside class="aside">{data["aside"]}</aside>' if data.get("aside") else ""
EYEBROW = ("Listening demo" if SITE else "Listening demo · prototype") + (f" · {H.escape(data['title'])}" if data.get("title") else "")
PAGE_TITLE = (f"{H.escape(data['title'])} · " if data.get("title") else "") + "Listen · 2ndPressing"
BACK = (data.get("back") or {"href": "/", "label": "Home"}) if SITE else None
SCRIPT_DATA = json.dumps(dict(data, prot=PROT, statusPrefix=f"LATENCY {LAT}    {KHZ} kHz    "))
FONTS_REMOTE = """<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@600&family=IBM+Plex+Sans:wght@400;600&display=swap">"""
FONTS_SITE = "".join(
    f'@font-face {{ font-family: "{fam}"; src: url("/assets/fonts/{f}.woff2") format("woff2"); font-weight: {w}; font-display: swap; }}\n'
    for fam, f, w in (("IBM Plex Sans Condensed", "IBMPlexSansCondensed-SemiBold", 600),
                      ("IBM Plex Sans", "IBMPlexSans-Regular", 400), ("IBM Plex Sans", "IBMPlexSans-SemiBold", 600),
                      ("IBM Plex Mono", "IBMPlexMono-Regular", 400), ("IBM Plex Mono", "IBMPlexMono-Medium", 500)))

HEAD = (f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{PAGE_TITLE}</title>
<meta name="description" content="Hear 2ndPressing remove clicks: switch between the repaired audio, the original and the removed clicks on a working replica of the plugin.">
<link rel="icon" href="/assets/favicon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="/assets/apple-touch-icon.png">
<style>
{FONTS_SITE}''' if SITE else f'''<title>2ndPressing Listening Demo</title>
<meta name="claude:viewport-fit-sides" content="cover">
{FONTS_REMOTE}
<style>''')
SITEBAR = (f'''<div class="sitebar"><a href="/" class="sb-brand">2<span class="nd">nd</span>Pressing</a><a href="{H.escape(BACK["href"])}" class="sb-back">{H.escape(BACK["label"])}</a></div>''' if SITE else "")

html = f'''{HEAD}
/* Layout: the faceplate as the hero on a dark "listening room" ground, a
   transport strip under it in the same hardware language, notes below.
   Single dark theme by choice: it's a product-on-a-desk presentation. */
:root {{
  --room: #24221F; --room-2: #2C2A27; --text: #EBE6DA; --dim: #A9A395; --amber: #F2A93B;
  --panel: #D4CFC3; --panel-2: #C6C0B2; --key-edge: #8C8577; --glass: #1B1A17;
  --cond: "IBM Plex Sans Condensed", "Arial Narrow", sans-serif;
  --sans: "IBM Plex Sans", system-ui, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, monospace;
  color-scheme: dark;
}}
body {{ background: radial-gradient(1200px 700px at 50% 0%, #34322E, var(--room)); color: var(--text); font: 400 15px/1.55 var(--sans); min-height: 100vh; }}
.wrap {{ max-width: 760px; margin: 0 auto; padding-block: 28px 56px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }}
.top {{ display: grid; gap: 8px; margin-bottom: 22px; }}
.eyebrow {{ font: 600 12px/1 var(--cond); letter-spacing: .14em; text-transform: uppercase; color: var(--dim); margin: 0; }}
h1 {{ font: 600 clamp(28px, 4vw, 38px)/1.05 var(--cond); margin: 0; text-wrap: balance; }}
.intro {{ margin: 0; color: #CFC9BB; max-width: 66ch; }}
.private {{ display: inline-block; margin-top: 4px; font: 500 12px/1.4 var(--mono); color: #E9B5A9; border: 1px solid #6E3B32; background: #3A2622; border-radius: 4px; padding: 6px 10px; }}
.unit {{ border-radius: 10px; overflow: hidden; box-shadow: 0 30px 60px rgba(0,0,0,.45), 0 0 0 1px #14130F; background: var(--panel-2); }}
.unit svg.face {{ width: 100%; height: auto; display: block; }}
.key {{ cursor: pointer; outline: none; }}
.key .cap {{ transition: transform .05s; }}
.key.on .cap {{ transform: translateY(2px); }}
.key.on .cap > rect:first-child {{ fill: url(#keyon); }}
.key:focus-visible .cap > rect:first-child {{ stroke: var(--amber); stroke-width: 2; }}
.key:active .cap {{ transform: translateY(2px); }}
.knob-hit {{ cursor: help; outline: none; }}
.knob-hit:focus-visible {{ stroke: var(--amber); stroke-width: 2; }}
.deck {{ background: linear-gradient(var(--panel), var(--panel-2)); border-top: 1px solid var(--key-edge); padding: 14px 16px 16px; display: grid; gap: 10px; color: #2C2A26; }}
.deck-row {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
.pkey {{ font: 600 13px/1 var(--cond); letter-spacing: .1em; color: #2C2A26; background: linear-gradient(#ECE8DE, #CFC9BB); border: 1px solid var(--key-edge); border-radius: 4px; box-shadow: 0 3px 0 var(--key-edge); padding: 10px 16px 10px 12px; display: inline-flex; align-items: center; gap: 10px; cursor: pointer; }}
.pkey .lamp {{ width: 16px; height: 4px; border-radius: 1px; background: #6D6553; }}
.pkey.on {{ background: linear-gradient(#D9D4C7, #BDB6A6); box-shadow: 0 1px 0 var(--key-edge); transform: translateY(2px); }}
.pkey.on .lamp {{ background: var(--amber); box-shadow: 0 0 8px rgba(242,169,59,.7); }}
.pkey:focus-visible, .jump:focus-visible, #tl:focus-visible {{ outline: 2px solid #A8711F; outline-offset: 2px; }}
.readout {{ font: 500 15px/1 var(--mono); color: var(--amber); background: linear-gradient(#1B1A17, #26241F); border-radius: 3px; padding: 9px 12px; min-width: 132px; text-align: center; font-variant-numeric: tabular-nums; }}
.hearing {{ font: 600 12px/1.3 var(--cond); letter-spacing: .1em; text-transform: uppercase; color: #4A463E; }}
.hearing b {{ color: #2C2A26; }}
.loop {{ margin-left: auto; font: 600 12px/1 var(--cond); letter-spacing: .1em; text-transform: uppercase; color: #4A463E; display: inline-flex; align-items: center; gap: 6px; }}
#tl {{ width: 100%; height: 64px; display: block; border-radius: 3px; cursor: pointer; touch-action: none; }}
.jumps {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.jump {{ font: 500 12px/1 var(--mono); color: #2C2A26; background: #F6F3EC; border: 1px solid #CFC8B8; border-radius: 3px; padding: 7px 10px; cursor: pointer; }}
#msg {{ font: 400 12px/1.3 var(--mono); color: #6A6455; min-height: 1em; }}
.notes {{ margin-top: 26px; display: grid; gap: 12px; color: #CFC9BB; font-size: 14.5px; }}
.notes h2 {{ font: 600 13px/1 var(--cond); letter-spacing: .14em; text-transform: uppercase; color: var(--dim); margin: 8px 0 0; }}
.notes p, .notes ul {{ margin: 0; max-width: 68ch; }}
.notes ul {{ padding-left: 20px; display: grid; gap: 6px; }}
kbd {{ font: 500 12px/1 var(--mono); background: #3A3834; border: 1px solid #55524B; border-bottom-width: 2px; border-radius: 3px; padding: 2px 6px; color: var(--text); }}
@media (max-width: 520px) {{ .loop {{ margin-left: 0; }} .readout {{ min-width: 0; }} }}
@media (prefers-reduced-motion: reduce) {{ .key .cap {{ transition: none; }} }}
.aside {{ margin-top: 6px; background: #2F2C27; border: 1px solid #5A5348; border-left: 3px solid var(--amber); border-radius: 4px; padding: 10px 14px; color: #E6DFD0; font-size: 14.5px; max-width: 68ch; }}
.aside b {{ color: var(--amber); }}
.sitebar {{ display: flex; justify-content: space-between; align-items: center; padding-bottom: 18px; margin-bottom: 18px; border-bottom: 1px solid #3E3B36; }}
.sb-brand {{ font: 600 20px/1 var(--cond); color: var(--text); text-decoration: none; }}
.sb-brand .nd {{ font-size: .6em; vertical-align: .45em; line-height: 0; }}
.sb-back {{ font: 600 12px/1 var(--cond); letter-spacing: .12em; text-transform: uppercase; color: var(--dim); text-decoration: none; }}
.sb-back:hover {{ color: var(--text); }}
</style>
{"</head><body>" if SITE else ""}

<div class="wrap">
  {SITEBAR}
  <header class="top">
    <p class="eyebrow">{EYEBROW}</p>
    <h1>Hear it on the faceplate</h1>
    <p class="intro">Press play, then switch with the faceplate’s own keys: <b>REPAIRED</b> for the declicked audio, <b>BYPASS</b> for the original, <b>CLICKS</b> to hear only what was removed. All three play in sync, so a key flips the sound mid-phrase. The meter and strip chart follow the engine’s real list of repairs.</p>
    {PRIVATE}
    {ASIDE}
  </header>

  <div class="unit">
    <svg class="face" viewBox="0 0 520 470" role="group" aria-label="2ndPressing faceplate">{faceplate()}</svg>
    <div class="deck">
      <div class="deck-row">
        <button class="pkey" id="play" type="button" aria-pressed="false"><span class="lamp"></span><span class="lbl">PLAY</span></button>
        <span class="readout" id="time">0:00.0 / {mmss(data["duration"])}</span>
        <span class="hearing">Hearing: <b id="hearing">Repaired</b></span>
        <label class="loop"><input type="checkbox" id="loop" checked> Loop</label>
      </div>
      {timeline()}
      <div class="jumps">{JUMPS}</div>
      <div id="msg" aria-live="polite"></div>
    </div>
  </div>

  <section class="notes">
    <h2>How to listen</h2>
    <ul>
      <li>Headphones help. Flip <b>BYPASS</b> on and off while it plays (or press <kbd>B</kbd>); <kbd>C</kbd> toggles Repaired and Clicks only; <kbd>Space</kbd> plays and pauses.</li>
      <li>Amber ticks above the waveform mark each of the {len(data["regions"])} repairs. Click or drag the waveform to jump.</li>
      {NOTE}
    </ul>
    <h2>How it was made</h2>
    <p>The real 2ndPressing engine (ClickNet R2, sensitivity {data["sensitivity"]} %, percussion protection {data["protection"]}) processed this {data["duration"]:.1f} s clip at {KHZ} kHz. Original, repaired and clicks-only were rendered with the same latency compensation, so they line up sample for sample, then encoded identically as AAC. Switching uses a 10 ms crossfade, as the plugin does. The sensitivity knob and protection keys show the settings this clip was processed with; they don’t reprocess audio in the browser.{CREDIT}</p>
  </section>
</div>
{'<script src="listen.js"></script></body></html>' if SITE else "<script>" + JS.replace("__DATA__", SCRIPT_DATA) + "</script>"}
'''
if SITE:
    (D / "listen.js").write_text(JS.replace("__DATA__", SCRIPT_DATA))
    (D / "index.html").write_text(html)
else:
    (D / "listening-demo.html").write_text(html)
print("wrote", len(html), "bytes")
