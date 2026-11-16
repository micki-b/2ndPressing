#!/bin/sh
# Upload public/ to Bluehost over SSH (optional; the File Manager works too).
# One-time: enable SSH in Bluehost (Advanced > SSH Access) and note your
# username, server host name and the domain's document root folder.
#
#   BH_USER=youruser BH_HOST=yourserver.bluehost.com BH_DIR=public_html/secondpressing.com ./deploy.sh
#
# --delete removes files on the server that are no longer in public/, so
# BH_DIR must be this site's own folder, never a parent shared with other sites.
set -eu
: "${BH_USER:?set BH_USER}" "${BH_HOST:?set BH_HOST}" "${BH_DIR:?set BH_DIR}"
cd "$(dirname "$0")"
rsync -avz --delete --exclude '.DS_Store' public/ "$BH_USER@$BH_HOST:$BH_DIR/"
