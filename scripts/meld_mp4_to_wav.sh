#!/usr/bin/env bash
# Extract 16 kHz mono wav from MELD.Raw into the layout xlser/corpora/meld.py expects.
#   bash scripts/meld_mp4_to_wav.sh /path/to/MELD.Raw /path/to/meld_wav
set -euo pipefail
SRC=${1:?MELD.Raw dir}; DST=${2:?output dir}
declare -A DIRS=([train]=train_splits [dev]=dev_splits_complete [test]=output_repeated_splits_test)
for split in train dev test; do
  mkdir -p "$DST/$split"
  cp "$SRC/${split}_sent_emo.csv" "$DST/" 2>/dev/null || cp "$SRC/../${split}_sent_emo.csv" "$DST/"
  for f in "$SRC/${DIRS[$split]}"/dia*_utt*.mp4; do
    ffmpeg -loglevel error -nostdin -y -i "$f" -vn -ac 1 -ar 16000 "$DST/$split/$(basename "${f%.mp4}").wav" \
      || echo "failed: $f" >&2
  done
done
