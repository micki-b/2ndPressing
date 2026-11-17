# Listening demo generator

Builds a self-contained page with a working replica of the 2ndPressing
faceplate.

**What the page does:**
- It plays a clip in three versions, in sync: repaired, original (BYPASS) and
  clicks only. The faceplate's keys switch between them with a 10 ms
  crossfade, the same as the plugin.
- The clicks-per-minute meter, LED ladder, strip chart and "N repaired" count
  follow the engine's real list of repairs.
- The sensitivity knob and percussion-protection keys show the settings the
  clip was processed with; they don't reprocess anything. INFO is disabled.
- Below the faceplate is a transport strip: play/pause, time, a waveform
  with a tick per repair for seeking, chapter buttons and loop. Keyboard:
  Space plays and pauses, B toggles bypass, C switches between Repaired and
  Clicks.

## 1. Prepare a clip (real engine)

Run from the repo's `tools/` directory. It needs a release build of
`runtime` and macOS `afconvert`.

```sh
uv run python ../website/tools/listening-demo/prepare.py IN_FILE OUT_DIR \
  --credit "Artist, “Title” (label, year)" \
  --chapter "0=lead-in crackle" --chapter "13.6=hand claps start" \
  --note "Optional listening note (HTML allowed)." \
  [--private]
```

This writes:
- `OUT_DIR/audio/{original,repaired,clicks}.mp4`: AAC, all with the same
  latency compensation, so they line up sample for sample.
- `OUT_DIR/demo.json`: click times, waveform overview and settings.

## 2. Build the page

```sh
python3 website/tools/listening-demo/build.py OUT_DIR   # -> OUT_DIR/listening-demo.html
```

Serve the page together with its `audio/` folder; `fetch()` needs http(s),
not `file://`.

To check the audio decodes, append `#selftest` to the URL. The page title
then reports each file's length and channel count.

## Rights

Only publish recordings you have the rights to. Examples are public-domain
recordings (in the US, those published before 1926), Creative Commons or
artist-permitted releases, or your own.

`--private` adds a "private prototype, not for the public site" notice. Use
it for anything else, such as the commercial Fleetwood Mac sample used for
the prototype review. Generated audio and data stay out of git.

## 3. Publish on secondpressing.com

```sh
python3 website/tools/listening-demo/build.py OUT_DIR --site   # -> OUT_DIR/index.html + listen.js
```

`--site` makes a full document that uses the site's own fonts and loads the
script from `listen.js`, as the site's CSP requires. Optional keys in
`demo.json` that only matter here:
- `title`: the recording's short name, shown in the eyebrow and tab title.
- `back`: `{"href": "/listen/", "label": "All demos"}` for the sitebar link
  (the default is Home).
- `aside`: an HTML note shown in an amber box under the intro.

Copy `index.html`, `listen.js` and `audio/` to `website/public/listen/NAME/`
and add the recording to the chooser, `website/public/listen/index.html`.
The audio is git-ignored, like the installers.

The published demos, from the Internet Archive's Great 78 Project
(`--protection off`, sensitivity 50 %):

| Folder | Archive item | Clip |
| --- | --- | --- |
| `kreisler` | `78_pierrots-dance-song_fritz-kreisler-carl-lamson-e-w-korngold-f-kreisler_gbia0197691a` | 4.5–34.5 s; chapters 0 run-in groove, 3.5 music begins, 15 midpoint |
| `lonesome` | `78_lonesome-mamma-blues_jazz-bos-carolina-serenaders-nickel-brown_gbia0101529b` | 0–30 s; chapters 0 run-in groove, 3 band begins, 15 midpoint |

Headless Chrome's `--virtual-time-budget` can end before `#selftest`
finishes on a local server, because decoding doesn't hold virtual time; the
title then stays unchanged. That is the harness, not the page: test against
the live site, or keep a slow request open while decoding.
