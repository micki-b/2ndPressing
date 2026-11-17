#!/bin/sh
# Mirror website/public/ to Bluehost over SSH (rsync: uploads only changes,
# deletes files that no longer exist locally).
#
#   ./deploy.sh            # sync
#   DRYRUN=1 ./deploy.sh   # show what would change, change nothing
#
# Settings come from deploy.env next to this script (git-ignored):
#   BH_HOST=secondpressing          # an ~/.ssh/config Host alias, or user@server
#   BH_DIR=public_html/website_...  # the site's document root on the server
#
# --delete removes files on the server that are not in public/, so BH_DIR
# must be this site's own folder, never a parent shared with other sites.
set -eu
cd "$(dirname "$0")"
[ -f deploy.env ] && . ./deploy.env
: "${BH_HOST:?set BH_HOST in deploy.env}" "${BH_DIR:?set BH_DIR in deploy.env}"
case "$BH_DIR" in
  ""|"/"|public_html|public_html/|~|"~/") echo "Refusing to sync into '$BH_DIR' (must be the site's own folder)"; exit 1;;
esac
DRY=""
[ "${DRYRUN:-0}" = "1" ] && DRY="--dry-run" && echo "== Dry run: nothing will change on the server"
rsync -rlptz --delete --itemize-changes $DRY \
  --exclude '.DS_Store' --exclude '_pgbackup' --exclude '_pginfo' --exclude 'pinegrow.json' \
  --exclude 'cgi-bin' --exclude '.well-known' \
  public/ "$BH_HOST:$BH_DIR/"
