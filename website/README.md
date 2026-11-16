# secondpressing.com

A static, single-page site: plain HTML and CSS, no JavaScript, no cookies,
no third-party requests (fonts are self-hosted). Everything that goes on the
server is in `public/`.

| Path | What |
|---|---|
| `public/index.html` | The page |
| `public/404.html` | "Not found" page |
| `public/.htaccess` | Apache settings: HTTPS + bare-domain redirect, caching, compression, security headers |
| `public/assets/site.css` | Styles (bump `?v=` in the HTML after editing, so browsers refetch) |
| `public/assets/fonts/` | IBM Plex subsets (WOFF2, SIL OFL 1.1) |
| `public/assets/cartridge-*.webp`, `header-icon.svg`, `favicon.svg`, `apple-touch-icon.png` | The logo: hero image, header icon, favicon and home-screen icon, from `brand/export.py` |
| `public/assets/repair.svg` | Strip chart in "How it works": real engine output; rebuild with `tools/make_figure.py` |
| `public/assets/og.png` | Share image; rebuild with `tools/render_images.sh` |

## Preview locally

```sh
cd website/public && python3 -m http.server 8000
```

Then open http://localhost:8000. `index.html` links its files relatively
(`assets/…`), so it also opens straight from Finder and in Pinegrow. Keep it
that way. `404.html` must keep site-absolute paths (`/assets/…`), because
it is served for missing pages at any depth.

## Visual editing

Pinegrow (`brew install --cask pinegrow`) edits these files in place:
*Open project* › `website/public`. Its own files (`_pgbackup/`, `_pginfo/`,
`pinegrow.json`) stay local: they are git-ignored and left out of the zip and
`deploy.sh`. Afterwards, upload the changed files,
or rebuild the zip:

```sh
cd website/public && zip -qr -X ../secondpressing-site.zip . -x '*.DS_Store' -x '_pgbackup/*' -x '_pginfo/*' -x 'pinegrow.json'
```

## Publish on Bluehost

The domain is registered at **Dynu**, and its DNS is there too. The site is
hosted at **Bluehost**.

1. **Add the domain to your Bluehost hosting.**
   - In the Bluehost portal, go to *Websites* (or *Domains*) and choose
     *Add site / Add domain*. Pick "use a domain I already own" and enter
     `secondpressing.com`.
   - Note the **document root** Bluehost creates, for example
     `public_html/secondpressing.com`. The files go there.
   - If your plan allows only one website (captainbraun.com already uses
     it), Bluehost will ask you to upgrade.
2. **Point the domain at Bluehost.** In your Dynu control panel, choose one of:
   - **Simplest:** change the domain's name servers to `ns1.bluehost.com` and
     `ns2.bluehost.com`. Bluehost then manages DNS, the SSL certificate and
     email for the domain.
   - **Keep DNS at Dynu:** add `A` records for `secondpressing.com` and
     `www` pointing to your Bluehost server's IP address. Bluehost shows it
     under *Hosting › Server information*. To use Bluehost email, also add
     its `MX` records.

   Changes take from minutes up to a day to spread.
3. **Upload the files.**
   - In Bluehost *Advanced › File Manager*, open the document root.
   - Turn on *Settings › Show hidden files*, so `.htaccess` is visible.
   - Upload everything **inside** `public/`, keeping the `assets/` folder
     structure.
   - Alternatively, enable SSH and run `./deploy.sh` (see the comments in the
     script).
4. **Turn on SSL.** In *Security › SSL*, enable the free certificate for
   `secondpressing.com`. It is issued once the DNS points at Bluehost.
   - `.htaccess` sends every visitor to `https://secondpressing.com`.
   - If the browser shows a certificate warning, SSL isn't active yet. Wait,
     or temporarily comment out the four `Rewrite` lines.
5. **Create the beta address.** The page asks people to write to
   `hello@secondpressing.com`. In Bluehost *Email*, create that mailbox, or a
   forwarder to the address you already read.
6. **Check:**
   - `http://secondpressing.com` and `https://www.secondpressing.com` both
     land on `https://secondpressing.com/`.
   - `/nothing-here` shows the 404 page.
   - Pasting the URL into Messages or Slack shows the share image.

## Editing

Text lives in `public/index.html`. Measurement figures and footnotes come
from `marketing/performance-marketing.md` and the runtime's README. Keep them
in step when results change.

The Content-Security-Policy in `.htaccess` allows only files from the site
itself, with no scripts, inline styles or external fonts. If you later add
analytics, an embedded video or a form service, the policy has to be widened
for it.
