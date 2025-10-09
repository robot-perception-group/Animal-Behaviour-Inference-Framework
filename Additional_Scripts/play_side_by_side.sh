#!/usr/bin/env bash
# side_by_side_pairs_ffmpeg_pipe_mpv_skiptiny.sh  (fixed)
# Pair two folders side-by-side with labels, n/p/q controls, optional left delay,
# and "skip tiny head" alignment to handle extra split files reliably.
#
# Usage:
#   ./side_by_side_pairs_ffmpeg_pipe_mpv_skiptiny.sh /path/left /path/right [delay_ms_for_left]
#
# Keys in mpv:
#   space=Pause • ←/→=Seek • [ / ]=Speed • { }=Half/Double • n=Next • p=Previous • q=Quit
#
# Env (optional):
#   SXS_HEIGHT=720   SXS_FPS=30   SXS_FONTSIZE=26   CRF=18   PRESET=veryfast   FFLOGLEVEL=fatal   WRAP=1
#   SPLIT_MERGE_SEC=8     # "tiny" if <= this many seconds
#   SPLIT_RATIO=0.30      # ...and <= this fraction of the NEXT file
#   DEBUG=0/1             # print decisions and mpv exit codes

: "${SXS_HEIGHT:=720}"
: "${SXS_FPS:=30}"
: "${SXS_FONTSIZE:=26}"
: "${CRF:=18}"
: "${PRESET:=veryfast}"
: "${FFLOGLEVEL:=fatal}"
: "${WRAP:=1}"
: "${SPLIT_MERGE_SEC:=8}"
: "${SPLIT_RATIO:=0.30}"
: "${DEBUG:=0}"

set +o pipefail 2>/dev/null || true

# ---- args ----
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 left_folder right_folder [delay_ms_for_left]" >&2; exit 1; fi
L_DIR="$1"; R_DIR="$2"; DELAY_MS="${3:-0}"
[[ -d "$L_DIR" && -d "$R_DIR" ]] || { echo "Folders not found."; exit 1; }
[[ "$DELAY_MS" =~ ^[0-9]+$ ]] || { echo "delay_ms must be non-negative integer"; exit 1; }
command -v ffmpeg  >/dev/null || { echo "ffmpeg required"; exit 1; }
command -v ffprobe >/dev/null || { echo "ffprobe required"; exit 1; }
command -v mpv     >/dev/null || { echo "mpv required"; exit 1; }

DELAY_SEC="$(awk "BEGIN{printf \"%.6f\", $DELAY_MS/1000}")"

# ---- helpers ----
if sort -V </dev/null >/dev/null 2>&1; then SORT=(sort -V); else SORT=(sort); fi
list_vids(){ find "$1" -maxdepth 1 -type f \
  \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.mkv' -o -iname '*.avi' -o -iname '*.m4v' -o -iname '*.webm' \) \
  -exec basename {} \; | "${SORT[@]}"; }

probe_dur() { # seconds (float)
  local f="$1" d
  d="$(ffprobe -v error -select_streams v:0 -show_entries stream=duration -of default=nw=1:nk=1 "$f" 2>/dev/null)"
  [[ -z "$d" || "$d" == "N/A" ]] && d="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$f" 2>/dev/null)"
  [[ -z "$d" || "$d" == "N/A" ]] && d="0"
  awk -v x="$d" 'BEGIN{ printf("%.6f", (x+0)) }'
}

is_tiny_head() {  # $1=dir, $2=array-name, $3=index  -> returns 0 (true) if tiny, 1 otherwise
  local dir="$1"; local -n ARR="$2"; local k="$3"
  local n=${#ARR[@]}
  (( k < 0 || k >= n )) && return 1
  # duration of this item
  local d1; d1="$(probe_dur "$dir/${ARR[$k]}")"
  # if last item: consider tiny only by absolute threshold (no next to compare)
  if (( k == n-1 )); then
    awk -v a="$d1" -v maxs="$SPLIT_MERGE_SEC" 'BEGIN{ exit !(a>0 && a<=maxs) }'
    return $?
  fi
  # compare to next
  local d2; d2="$(probe_dur "$dir/${ARR[$((k+1))]}")"
  awk -v a="$d1" -v b="$d2" -v maxs="$SPLIT_MERGE_SEC" -v ratio="$SPLIT_RATIO" \
      'BEGIN{ exit !( (a>0) && (a<=maxs) && (b>0) && (a/b <= ratio) ) }'
}

# Build aligned pairs by skipping tiny heads BEFORE pairing
# Corrected signature (no stray 'shift' games):
#
