#!/usr/bin/env bash
# The ABC dataset's chunk 0000 on the server, from its source (the user's rule 2026-09-29: "download the set on the
# server side new from source"). Runs ON the server, detached by the caller:
#   bash server/server_a/fetch_abc.sh [chunk, default 0000]
# Source: the link list https://deep-geometry.github.io/abc-dataset/data/step_v00.txt (New York University's archive)
# and the checksums .../data/md5.yml. Steps, each skipped when its result is there: download, md5, unpack (7z), the
# files flat in step/, the manifest of ALL files by 8 freecadcmd processes (scripts/abc_manifest.py), joined.
# Output in $ABC (default ~/data/shape_reco/parts/abc): the archive, step/, stl/, manifest.csv, fetch.log, DONE.
set -uo pipefail
ROOT=${ROOT:-/data/user/shape_reco}
ABC=${ABC:-$HOME/data/shape_reco/parts/abc}
HERE=$(cd "$(dirname "$0")/../.." && pwd)
CHUNK=${1:-0000}
NAME=abc_${CHUNK}_step_v00.7z
PARTS=${PARTS:-8}
mkdir -p "$ABC" && cd "$ABC"
rm -f DONE
log() { echo "$(date -Is) $*" | tee -a fetch.log; }
curl -sSL --max-time 60 -o step_v00.txt https://deep-geometry.github.io/abc-dataset/data/step_v00.txt
curl -sSL --max-time 60 -o md5.yml https://deep-geometry.github.io/abc-dataset/data/md5.yml
URL=$(awk -v n="$NAME" '$2 == n {print $1}' step_v00.txt)
WANT=$(grep -i "$NAME" md5.yml | grep -o -E "[0-9a-f]{32}" | head -1)
[ -n "$URL" ] || { log "no link for $NAME in step_v00.txt"; exit 1; }
if [ ! -f "$NAME" ] || [ "$(md5sum "$NAME" | cut -d' ' -f1)" != "$WANT" ]; then
  log "download $URL"
  curl -sSL --retry 5 --retry-delay 20 -C - -o "$NAME" "$URL" || { log "download failed"; exit 1; }
fi
HAVE=$(md5sum "$NAME" | cut -d' ' -f1)
log "md5 $HAVE, the dataset's list says ${WANT:-nothing}"
[ -z "$WANT" ] || [ "$HAVE" = "$WANT" ] || { log "md5 differs"; exit 1; }
if [ "$(ls step 2>/dev/null | wc -l)" -lt 100 ]; then
  log "unpack"
  rm -rf unpacked && mkdir -p unpacked step
  nice -n 10 7z x -y -ounpacked "$NAME" > unpack.log 2>&1 || { log "7z failed, see unpack.log"; exit 1; }
  find unpacked -name '*.step' -exec mv -t step {} +
  rm -rf unpacked
fi
log "$(ls step | wc -l) STEP files, $(du -sh step | cut -f1)"
if [ ! -f manifest.csv ]; then
  log "manifest by $PARTS processes"
  for i in $(seq 0 $((PARTS - 1))); do
    ABC_DIR="$ABC" ABC_PART=$i/$PARTS nice -n 10 "$ROOT/bin/freecadcmd" "$HERE/scripts/abc_manifest.py" > manifest.$i.log 2>&1 &
  done
  wait
  head -1 manifest.0.csv > manifest.csv
  for i in $(seq 0 $((PARTS - 1))); do tail -n +2 manifest.$i.csv; done | sort >> manifest.csv
fi
log "manifest: $(($(wc -l < manifest.csv) - 1)) rows, $(grep -c ',ok$' manifest.csv) ok"
echo "finished $(date -Is)" > DONE
