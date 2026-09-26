#!/usr/bin/env bash
# Deploy the zone editor and the signal spot-check next to the labeling tool.
# Run from the repo root on a machine with ssh access to the server (alias crm-vps).
# The token never leaves the server: the remote part reads it from ~/wiut/labeler/token.
#
# Needs on the server (see README.md): ~/wiut/scene/assets/{ref_4k.jpg,bg_<STEM>.jpg}
# from make_backgrounds.py and ~/wiut/scene/strips/ from context_strips.py.
set -euo pipefail
HOST=${HOST:-crm-vps}
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

scp -q "$HOST:wiut/scene/strips/strips.json" "$STAGE/strips.json" 2>/dev/null || echo "no context strips on the server yet"
python3 tools/scene/web/build_data.py --eda reports/eda --out "$STAGE/site" --strips "$STAGE/strips.json"
cp tools/scene/web/zones.html tools/scene/web/spotcheck.html tools/scene/web/tools.html tools/labeler/index.html "$STAGE/site/"
cp tools/scene/reference_C3897.jpg "$STAGE/site/scene/ref_1080.jpg"
mkdir -p "$STAGE/srv"
cp tools/labeler/server.py "$STAGE/srv/server.py"

COPYFILE_DISABLE=1 tar --no-xattrs -C "$STAGE" -czf - site srv | ssh "$HOST" 'set -euo pipefail
  T=$(cat ~/wiut/labeler/token); D=/var/www/salen-label/$T; S=~/wiut/scene/deploy_stage
  rm -rf "$S"; mkdir -p "$S"; tar -C "$S" -xzf -
  python3 -m py_compile "$S/srv/server.py"
  mkdir -p "$D/scene" "$D/spotcheck/strips"
  cp -r "$S/site/." "$D/"
  cp ~/wiut/scene/assets/ref_4k.jpg ~/wiut/scene/assets/bg_*.jpg "$D/scene/"
  if [ -d ~/wiut/scene/strips ]; then cp -r ~/wiut/scene/strips/C* "$D/spotcheck/strips/"; fi
  cp "$S/site/spotcheck/items.json" ~/wiut/scene/spotcheck_items.json
  cp "$S/site/index.html" ~/wiut/labeler/index.html
  cp ~/wiut/labeler/server.py ~/wiut/labeler/server.py.prev
  cp "$S/srv/server.py" ~/wiut/labeler/server.py
  sudo -n systemctl restart salen-labeler
  sleep 1
  systemctl is-active salen-labeler
  rm -rf "$S"'
echo "deployed; page: https://label-salen.<host>/<TOKEN>/zones.html and /spotcheck.html"
