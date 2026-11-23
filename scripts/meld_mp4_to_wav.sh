#!/usr/bin/env bash
# Extract 16 kHz mono wav from MELD.Raw into the layout xlser/corpora/meld.py expects.
#   bash scripts/meld_mp4_to_wav.sh /path/to/MELD.Raw /path/to/meld_wav [jobs]
# Runs `jobs` ffmpeg processes in parallel (default: all cores), skips wavs that
# already exist (safe to re-run after an interruption), and lists failed files.
set -euo pipefail
SRC=${1:?MELD.Raw dir}; DST=${2:?output dir}; JOBS=${3:-$(nproc)}
command -v ffmpeg >/dev/null || { echo "ffmpeg not found (conda install -c conda-forge ffmpeg)" >&2; exit 1; }
declare -A DIRS=([train]=train_splits [dev]=dev_splits_complete [test]=output_repeated_splits_test)
FAILED="$DST/failed.txt"; mkdir -p "$DST"; : > "$FAILED"
for split in train dev test; do
  mkdir -p "$DST/$split"
  cp "$SRC/${split}_sent_emo.csv" "$DST/" 2>/dev/null || cp "$SRC/../${split}_sent_emo.csv" "$DST/"
  find "$SRC/${DIRS[$split]}" -maxdepth 1 -name 'dia*_utt*.mp4' -print0 |
    DST="$DST/$split" FAILED="$FAILED" xargs -0 -P "$JOBS" -I{} bash -c '
      out="$DST/$(basename "${1%.mp4}").wav"
      [ -s "$out" ] && exit 0
      ffmpeg -loglevel error -nostdin -y -i "$1" -vn -ac 1 -ar 16000 "$out.tmp.wav" 2>/dev/null \
        && mv "$out.tmp.wav" "$out" || { rm -f "$out.tmp.wav"; echo "$1" >> "$FAILED"; }' _ {}
  echo "$split: $(find "$DST/$split" -name '*.wav' | wc -l) wav"
done
echo "failed: $(wc -l < "$FAILED") (listed in $FAILED)"
